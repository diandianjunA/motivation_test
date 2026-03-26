#pragma once

#include <atomic>
#include <chrono>
#include <future>
#include <thread>

#include "common/types.hh"
#include "hnsw/hnsw.hh"
#include "io/database.hh"

namespace service {

struct InsertRequest {
  node_t id;
  vec<element_t> components;
  std::promise<bool> result;
};

struct InsertBatch {
  vec<InsertRequest*> requests;
};

struct QueryRequest {
  vec<element_t> components;
  u32 k;
  std::promise<vec<node_t>> result;
};

using InsertQueue = concurrent_queue<InsertRequest*>;
using InsertBatchQueue = concurrent_queue<InsertBatch*>;
using QueryQueue = concurrent_queue<QueryRequest*>;

static HNSWCoroutine dummy_service_coroutine() {
  co_return;
}

inline void reset_service_coroutine_state(HNSWCoroutine& coroutine) {
  coroutine.cached_ep_ptr = {};
  coroutine.visited_nodes.clear();
  coroutine.top_candidates.clear();
  coroutine.next_candidates.clear();
}

/**
 * Service-mode insert scheduler. Runs continuously, pulling InsertRequests from the queue.
 * Each coroutine slot has a staging buffer slot and an associated promise.
 * When shutdown is set, drains remaining work and exits.
 */
template <class Distance>
void service_build_insert_batches(InsertQueue& request_queue,
                                  InsertBatchQueue& batch_queue,
                                  std::atomic<bool>& shutdown,
                                  u32 max_batch_size,
                                  std::chrono::microseconds max_batch_wait,
                                  std::atomic<bool>& paused,
                                  std::atomic<u32>& idle_count,
                                  std::atomic<u32>& completed_batchers) {
  vec<InsertRequest*> pending;
  pending.reserve(max_batch_size);

  auto first_enqueue = std::chrono::steady_clock::time_point{};
  const auto flush_batch = [&]() {
    if (pending.empty()) return;
    auto* batch = new InsertBatch{};
    batch->requests = std::move(pending);
    pending.clear();
    pending.reserve(max_batch_size);
    first_enqueue = {};
    batch_queue.enqueue(batch);
  };

  for (;;) {
    bool did_work = false;

    const bool paused_now = paused.load(std::memory_order_relaxed);
    const bool shutting_down = shutdown.load(std::memory_order_relaxed);

    if (!paused_now || shutting_down) {
      InsertRequest* request = nullptr;
      while (pending.size() < max_batch_size && request_queue.try_dequeue(request)) {
        if (pending.empty()) {
          first_enqueue = std::chrono::steady_clock::now();
        }
        pending.push_back(request);
        did_work = true;
      }

      if (pending.size() >= max_batch_size) {
        flush_batch();
        did_work = true;
      } else if (!pending.empty() && std::chrono::steady_clock::now() - first_enqueue >= max_batch_wait) {
        flush_batch();
        did_work = true;
      }
    }

    if (shutting_down) {
      InsertRequest* request = nullptr;
      while (request_queue.try_dequeue(request)) {
        if (pending.empty()) {
          first_enqueue = std::chrono::steady_clock::now();
        }
        pending.push_back(request);
        if (pending.size() >= max_batch_size) {
          flush_batch();
        }
      }
      flush_batch();
      completed_batchers.fetch_add(1, std::memory_order_acq_rel);
      break;
    }

    if (paused_now) {
      flush_batch();
      idle_count.fetch_add(1, std::memory_order_release);
      while (paused.load(std::memory_order_relaxed) && !shutdown.load(std::memory_order_relaxed)) {
        std::this_thread::yield();
      }
      idle_count.fetch_sub(1, std::memory_order_release);
      continue;
    }

    if (!did_work) {
      std::this_thread::yield();
    }
  }
}

template <class Distance>
void service_schedule_inserts(hnsw::HNSW<Distance>& hnsw,
                              InsertBatchQueue& queue,
                              std::atomic<bool>& shutdown,
                              u32 num_coroutines,
                              const u_ptr<ComputeThread>& thread,
                              u32 dim,
                              std::atomic<bool>& paused,
                              std::atomic<u32>& idle_count,
                              u32 total_batchers,
                              std::atomic<u32>& completed_batchers) {
  thread->reset();
  // per-coroutine staging database: each coroutine gets one slot
  io::Database<element_t> staging;
  staging.allocate(dim, num_coroutines);

  // per-coroutine promise tracking
  vec<InsertRequest*> active_requests(num_coroutines, nullptr);
  InsertBatch* current_batch = nullptr;
  idx_t next_request_idx = 0;

  // initialize coroutines
  thread->coroutines.reserve(num_coroutines);
  for (u32 i = 0; i < num_coroutines; ++i) {
    thread->coroutines.emplace_back(std::make_unique<HNSWCoroutine>(dummy_service_coroutine()));
    thread->set_current_coroutine(i);
    thread->coroutines.back()->handle.resume();
  }

  for (;;) {
    bool all_idle = true;

    for (u32 cid = 0; cid < thread->coroutines.size(); ++cid) {
      auto& coroutine = *thread->coroutines[cid];
      thread->poll_cq();

      if (coroutine.handle.done()) {
        // deliver result for completed request
        if (active_requests[cid]) {
          active_requests[cid]->result.set_value(true);
          active_requests[cid] = nullptr;
        }

        if (!current_batch) {
          queue.try_dequeue(current_batch);
          next_request_idx = 0;
        }

        if (current_batch && next_request_idx < current_batch->requests.size()) {
          all_idle = false;
          InsertRequest* req = current_batch->requests[next_request_idx++];

          // copy vector into staging buffer
          auto slot_components = staging.get_components(cid);
          std::copy(req->components.begin(), req->components.end(), slot_components.begin());
          staging.set_id(cid, req->id);
          active_requests[cid] = req;

          coroutine.handle.destroy();
          reset_service_coroutine_state(coroutine);
          thread->set_current_coroutine(cid);
          coroutine.handle = hnsw.insert(req->id, slot_components, thread).handle;
        }

      } else if (thread->is_ready(cid)) {
        all_idle = false;
        thread->set_current_coroutine(cid);
        coroutine.handle.resume();
      } else {
        all_idle = false;
      }
    }

    if (current_batch) {
      bool batch_complete = next_request_idx >= current_batch->requests.size();
      if (batch_complete) {
        batch_complete = std::all_of(active_requests.begin(), active_requests.end(), [](const auto* request) {
          return request == nullptr;
        });
      }

      if (batch_complete) {
        delete current_batch;
        current_batch = nullptr;
        next_request_idx = 0;
      } else {
        all_idle = false;
      }
    }

    if (all_idle) {
      if (shutdown.load(std::memory_order_relaxed) && !current_batch) {
        InsertBatch* shutdown_batch = nullptr;
        if (queue.try_dequeue(shutdown_batch)) {
          current_batch = shutdown_batch;
          next_request_idx = 0;
          continue;
        }
        if (completed_batchers.load(std::memory_order_acquire) >= total_batchers) {
          break;
        }
      }
      if (paused.load(std::memory_order_relaxed)) {
        idle_count.fetch_add(1, std::memory_order_release);
        while (paused.load(std::memory_order_relaxed)) {
          std::this_thread::yield();
        }
        idle_count.fetch_sub(1, std::memory_order_release);
        continue;
      }
      std::this_thread::yield();
    }
  }

  // cleanup coroutines
  for (const auto& coroutine : thread->coroutines) {
    if (!coroutine->handle.done()) {
      // should not happen in normal shutdown, but be safe
    }
    coroutine->handle.destroy();
  }
  delete current_batch;
}

/**
 * Service-mode query scheduler. Runs continuously, pulling QueryRequests from the queue.
 * Results are collected from thread->query_results and delivered via promise.
 */
template <class Distance>
void service_schedule_queries(hnsw::HNSW<Distance>& hnsw,
                              QueryQueue& queue,
                              std::atomic<bool>& shutdown,
                              u32 num_coroutines,
                              const u_ptr<ComputeThread>& thread,
                              u32 dim,
                              std::atomic<bool>& paused,
                              std::atomic<u32>& idle_count) {
  thread->reset();
  // per-coroutine staging database
  io::Database<element_t> staging;
  staging.allocate(dim, num_coroutines);

  // per-coroutine request tracking
  vec<QueryRequest*> active_requests(num_coroutines, nullptr);
  // use coroutine slot id as query_id in thread->query_results
  vec<node_t> slot_ids(num_coroutines);
  for (u32 i = 0; i < num_coroutines; ++i) {
    slot_ids[i] = i;  // use slot index as temporary query id
  }

  // initialize coroutines
  thread->coroutines.reserve(num_coroutines);
  for (u32 i = 0; i < num_coroutines; ++i) {
    thread->coroutines.emplace_back(std::make_unique<HNSWCoroutine>(dummy_service_coroutine()));
    thread->set_current_coroutine(i);
    thread->coroutines.back()->handle.resume();
  }

  for (;;) {
    bool all_idle = true;

    for (u32 cid = 0; cid < thread->coroutines.size(); ++cid) {
      auto& coroutine = *thread->coroutines[cid];
      thread->poll_cq();

      if (coroutine.handle.done()) {
        // deliver result for completed query
        if (active_requests[cid]) {
          node_t q_id = slot_ids[cid];
          auto it = thread->query_results.find(q_id);
          if (it != thread->query_results.end()) {
            active_requests[cid]->result.set_value(std::move(it->second));
            thread->query_results.erase(it);
          } else {
            active_requests[cid]->result.set_value(vec<node_t>{});
          }
          active_requests[cid] = nullptr;
        }

        // try to dequeue a new query request
        QueryRequest* req = nullptr;
        if (queue.try_dequeue(req)) {
          all_idle = false;

          // copy vector into staging buffer
          auto slot_components = staging.get_components(cid);
          std::copy(req->components.begin(), req->components.end(), slot_components.begin());
          active_requests[cid] = req;

          coroutine.handle.destroy();
          reset_service_coroutine_state(coroutine);
          thread->set_current_coroutine(cid);
          coroutine.handle = hnsw.knn(slot_ids[cid], slot_components, thread).handle;
        }

      } else if (thread->is_ready(cid)) {
        all_idle = false;
        thread->set_current_coroutine(cid);
        coroutine.handle.resume();
      } else {
        all_idle = false;
      }
    }

    if (all_idle) {
      if (shutdown.load(std::memory_order_relaxed)) {
        break;
      }
      if (paused.load(std::memory_order_relaxed)) {
        idle_count.fetch_add(1, std::memory_order_release);
        while (paused.load(std::memory_order_relaxed)) {
          std::this_thread::yield();
        }
        idle_count.fetch_sub(1, std::memory_order_release);
        continue;
      }
      std::this_thread::yield();
    }
  }

  // cleanup coroutines
  for (const auto& coroutine : thread->coroutines) {
    coroutine->handle.destroy();
  }
}

}  // namespace service
