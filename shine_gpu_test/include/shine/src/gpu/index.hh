#pragma once

#include <memory>

#include "common/configuration.hh"
#include "common/types.hh"
#include "gpu/common.hh"

class BufferAllocator;
class ComputeThread;

namespace gpu {

class GpuIndex {
public:
  GpuIndex(const configuration::IndexConfiguration& config,
           BufferAllocator& buffer_allocator,
           u32 num_memory_nodes,
           size_t cache_bytes);
  ~GpuIndex();

  GpuIndex(const GpuIndex&) = delete;
  GpuIndex& operator=(const GpuIndex&) = delete;
  GpuIndex(GpuIndex&&) noexcept;
  GpuIndex& operator=(GpuIndex&&) noexcept;

  void initialize_or_bootstrap(const u_ptr<ComputeThread>& thread, bool is_initiator);
  void reload_from_remote(const u_ptr<ComputeThread>& thread);
  void clear_cache();

  vec<bool> insert_batch(const vec<PendingInsert>& batch, const u_ptr<ComputeThread>& thread);
  vec<vec<node_t>> search_batch(const vec<PendingQuery>& batch, const u_ptr<ComputeThread>& thread);

  u32 size() const;
  u32 dim() const;
  u32 pad_dim() const;
  u32 max_vectors() const;
  vec<element_t> routing_centroid() const;

  struct ImplBase;

private:
  std::unique_ptr<ImplBase> impl_;
};

}  // namespace gpu
