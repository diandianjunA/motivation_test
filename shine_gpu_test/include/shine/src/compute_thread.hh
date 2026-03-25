#pragma once

#include <library/hugepage.hh>
#include <library/thread.hh>

#include "buffer_allocator.hh"
#include "shared_context.hh"

class ComputeThread : public Thread {
public:
  ComputeThread(u32 id,
                u32 compute_node_id,
                i32 max_send_queue_wr,
                BufferAllocator& buffer_allocator,
                u32 num_coroutines)
      : Thread(id),
        node_id(compute_node_id),
        send_wcs(max_send_queue_wr),
        buffer_allocator(buffer_allocator),
        post_balances(num_coroutines),
        max_send_queue_wr_(max_send_queue_wr) {
    // allocate single pointer slot (for RDMA requests) per coroutine
    for (idx_t i = 0; i < num_coroutines; ++i) {
      pointer_slots_.push_back(buffer_allocator.allocate_pointer());
    }
  }

  void poll_cq() {
    Context::poll_send_cq(send_wcs.data(), max_send_queue_wr_, ctx->get_cq(), [&](u64 wr_id) {
      auto [ctx_offset, coroutine_id] = decode_64bit(wr_id);
      --ctx->registered_threads[ctx_offset]->post_balances[coroutine_id];
    });
  }

  u64 create_wr_id() const { return encode_64bit(ctx_tid, running_coroutine_); }
  bool is_ready(u32 coroutine_id) const { return post_balances[coroutine_id] == 0; }

  void track_post() { ++post_balances[running_coroutine_]; }
  void set_current_coroutine(u32 id) { running_coroutine_ = id; }
  u64* coros_pointer_slot() const { return pointer_slots_[running_coroutine_]; }

public:
  const u32 node_id;
  vec<ibv_wc> send_wcs;

  BufferAllocator& buffer_allocator;  // global per compute node

  SharedContext<ComputeThread>* ctx{nullptr};  // initialized by WorkerPool
  u32 ctx_tid{};

  vec<std::atomic<i32>> post_balances;  // per coroutine

private:
  const i32 max_send_queue_wr_;
  u32 running_coroutine_{};  // tracks the id of the currently running coroutine
  vec<u64*> pointer_slots_;  // memory region for a single pointer per coroutine
};
