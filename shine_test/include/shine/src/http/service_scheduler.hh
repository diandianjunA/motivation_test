#pragma once

#include <atomic>
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

struct QueryRequest {
  vec<element_t> components;
  u32 k;
  std::promise<vec<node_t>> result;
};

using InsertQueue = concurrent_queue<InsertRequest*>;
using QueryQueue = concurrent_queue<QueryRequest*>;

static HNSWCoroutine dummy_service_coroutine() {
  co_return;
}

enum class ServiceRequestKind : u8 {
  none = 0,
  insert = 1,
  query = 2,
};

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
void service_schedule_inserts(hnsw::HNSW<Distance>& hnsw,
                              InsertQueue& queue,
                              std::atomic<bool>& shutdown,
                              u32 num_coroutines,
                              const u_ptr<ComputeThread>& thread,
                              u32 dim,
                              std::atomic<bool>& paused,
                              std::atomic<u32>& idle_count) {
  thread->reset();
  // per-coroutine staging database: each coroutine gets one slot
  io::Database<element_t> staging;
  staging.allocate(dim, num_coroutines);

  // per-coroutine promise tracking
  vec<InsertRequest*> active_requests(num_coroutines, nullptr);

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

        // try to dequeue a new insert request
        InsertRequest* req = nullptr;
        if (queue.try_dequeue(req)) {
          all_idle = false;

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
    if (!coroutine->handle.done()) {
      // should not happen in normal shutdown, but be safe
    }
    coroutine->handle.destroy();
  }
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

/**
 * Unified service scheduler. Every worker can execute inserts and queries.
 * This matches the original shine-hnsw-index model more closely than a fixed
 * one-writer / N-query split: all workers share the same queues and pick the
 * next available operation when a coroutine slot becomes free.
 */
template <class Distance>
void service_schedule_mixed(hnsw::HNSW<Distance>& hnsw,
                            InsertQueue& insert_queue,
                            QueryQueue& query_queue,
                            std::atomic<bool>& shutdown,
                            u32 num_coroutines,
                            const u_ptr<ComputeThread>& thread,
                            u32 dim,
                            std::atomic<bool>& paused,
                            std::atomic<u32>& idle_count) {
  thread->reset();

  io::Database<element_t> staging;
  staging.allocate(dim, num_coroutines);

  vec<ServiceRequestKind> active_kinds(num_coroutines, ServiceRequestKind::none);
  vec<InsertRequest*> active_insert_requests(num_coroutines, nullptr);
  vec<QueryRequest*> active_query_requests(num_coroutines, nullptr);
  vec<node_t> slot_ids(num_coroutines);
  for (u32 i = 0; i < num_coroutines; ++i) {
    slot_ids[i] = i;
  }

  thread->coroutines.reserve(num_coroutines);
  for (u32 i = 0; i < num_coroutines; ++i) {
    thread->coroutines.emplace_back(std::make_unique<HNSWCoroutine>(dummy_service_coroutine()));
    thread->set_current_coroutine(i);
    thread->coroutines.back()->handle.resume();
  }

  bool prefer_insert = true;

  for (;;) {
    bool all_idle = true;

    for (u32 cid = 0; cid < thread->coroutines.size(); ++cid) {
      auto& coroutine = *thread->coroutines[cid];
      thread->poll_cq();

      if (coroutine.handle.done()) {
        if (active_kinds[cid] == ServiceRequestKind::insert && active_insert_requests[cid]) {
          active_insert_requests[cid]->result.set_value(true);
          active_insert_requests[cid] = nullptr;
          active_kinds[cid] = ServiceRequestKind::none;
        } else if (active_kinds[cid] == ServiceRequestKind::query && active_query_requests[cid]) {
          node_t q_id = slot_ids[cid];
          auto it = thread->query_results.find(q_id);
          if (it != thread->query_results.end()) {
            active_query_requests[cid]->result.set_value(std::move(it->second));
            thread->query_results.erase(it);
          } else {
            active_query_requests[cid]->result.set_value(vec<node_t>{});
          }
          active_query_requests[cid] = nullptr;
          active_kinds[cid] = ServiceRequestKind::none;
        }

        InsertRequest* insert_req = nullptr;
        QueryRequest* query_req = nullptr;

        bool got_work = false;
        if (prefer_insert) {
          got_work = insert_queue.try_dequeue(insert_req);
          if (!got_work) {
            got_work = query_queue.try_dequeue(query_req);
          }
        } else {
          got_work = query_queue.try_dequeue(query_req);
          if (!got_work) {
            got_work = insert_queue.try_dequeue(insert_req);
          }
        }

        if (got_work) {
          all_idle = false;
          auto slot_components = staging.get_components(cid);

          coroutine.handle.destroy();
          reset_service_coroutine_state(coroutine);
          thread->set_current_coroutine(cid);

          if (insert_req) {
            std::copy(insert_req->components.begin(), insert_req->components.end(), slot_components.begin());
            staging.set_id(cid, insert_req->id);
            active_kinds[cid] = ServiceRequestKind::insert;
            active_insert_requests[cid] = insert_req;
            coroutine.handle = hnsw.insert(insert_req->id, slot_components, thread).handle;
            prefer_insert = false;
          } else {
            std::copy(query_req->components.begin(), query_req->components.end(), slot_components.begin());
            active_kinds[cid] = ServiceRequestKind::query;
            active_query_requests[cid] = query_req;
            coroutine.handle = hnsw.knn(slot_ids[cid], slot_components, thread).handle;
            prefer_insert = true;
          }
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

  for (const auto& coroutine : thread->coroutines) {
    coroutine->handle.destroy();
  }
}

}  // namespace service
