#pragma once

#include <atomic>
#include <future>
#include <thread>

#include "gpu/index.hh"

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

inline void gpu_schedule_inserts(gpu::GpuIndex& index,
                                 InsertQueue& queue,
                                 std::atomic<bool>& shutdown,
                                 u32 max_batch_size,
                                 const u_ptr<ComputeThread>& thread,
                                 std::atomic<bool>& paused,
                                 std::atomic<u32>& idle_count) {
  for (;;) {
    vec<InsertRequest*> batch;
    batch.reserve(max_batch_size);

    InsertRequest* request = nullptr;
    while (batch.size() < max_batch_size && queue.try_dequeue(request)) {
      batch.push_back(request);
    }

    if (!batch.empty()) {
      vec<gpu::PendingInsert> pending;
      pending.reserve(batch.size());
      for (const auto* req : batch) {
        pending.push_back({req->id, req->components.data()});
      }

      const vec<bool> results = index.insert_batch(pending, thread);
      for (idx_t i = 0; i < batch.size(); ++i) {
        batch[i]->result.set_value(i < results.size() ? results[i] : false);
      }
      continue;
    }

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

inline void gpu_schedule_queries(gpu::GpuIndex& index,
                                 QueryQueue& queue,
                                 std::atomic<bool>& shutdown,
                                 u32 max_batch_size,
                                 const u_ptr<ComputeThread>& thread,
                                 std::atomic<bool>& paused,
                                 std::atomic<u32>& idle_count) {
  for (;;) {
    vec<QueryRequest*> batch;
    batch.reserve(max_batch_size);

    QueryRequest* request = nullptr;
    while (batch.size() < max_batch_size && queue.try_dequeue(request)) {
      batch.push_back(request);
    }

    if (!batch.empty()) {
      vec<gpu::PendingQuery> pending;
      pending.reserve(batch.size());
      for (const auto* req : batch) {
        pending.push_back({req->components.data(), req->k});
      }

      vec<vec<node_t>> results = index.search_batch(pending, thread);
      for (idx_t i = 0; i < batch.size(); ++i) {
        batch[i]->result.set_value(i < results.size() ? std::move(results[i]) : vec<node_t>{});
      }
      continue;
    }

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

}  // namespace service
