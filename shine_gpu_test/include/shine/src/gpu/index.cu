#ifndef JSON_HAS_RANGES
#define JSON_HAS_RANGES 0
#endif

#include "gpu/index.hh"

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <deque>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <queue>
#include <random>
#include <shared_mutex>
#include <stdexcept>
#include <thread>
#include <unordered_map>

#include <library/batched_read.hh>

#include "buffer_allocator.hh"
#include "compute_thread.hh"
#include "gpu/device_vector.cuh"

namespace cg = cooperative_groups;

namespace gpu {

namespace {

template <u32 PadDim>
using GpuVector = DeviceVector<float, PadDim>;

template <u32 PadDim>
using Record = VertexRecord<PadDim>;

template <u32 PadDim>
using RecordHandle = std::shared_ptr<Record<PadDim>>;

template <u32 PadDim>
struct RabitqQueryFactor;

template <u32 PadDim>
struct QuantizedQueryBatch {
  float* rotated_queries{nullptr};
  RabitqQueryFactor<PadDim>* factors{nullptr};

  QuantizedQueryBatch() = default;
  QuantizedQueryBatch(const QuantizedQueryBatch&) = delete;
  QuantizedQueryBatch& operator=(const QuantizedQueryBatch&) = delete;

  QuantizedQueryBatch(QuantizedQueryBatch&& other) noexcept
      : rotated_queries(other.rotated_queries), factors(other.factors) {
    other.rotated_queries = nullptr;
    other.factors = nullptr;
  }

  QuantizedQueryBatch& operator=(QuantizedQueryBatch&& other) noexcept {
    if (this != &other) {
      if (rotated_queries) cudaFree(rotated_queries);
      if (factors) cudaFree(factors);
      rotated_queries = other.rotated_queries;
      factors = other.factors;
      other.rotated_queries = nullptr;
      other.factors = nullptr;
    }
    return *this;
  }

  ~QuantizedQueryBatch() {
    if (rotated_queries) cudaFree(rotated_queries);
    if (factors) cudaFree(factors);
  }
};

template <u32 PadDim>
struct RabitqQueryFactor {
  float add{};
  float k1xSumq{};
  float kBxSumq{};
};

inline void ensure_gpu_log_init() {
  static std::once_flag init_flag;
  std::call_once(init_flag, []() {});
}

inline double best_rescale_factor(const float* values, size_t dim, size_t bits) {
  constexpr double kEps = 1e-5;
  constexpr int kNumEnum = 10;

  const double max_value = *std::max_element(values, values + dim);
  if (max_value <= kEps) return 1.0;

  constexpr std::array<float, 9> kTightStart = {0.0f, 0.15f, 0.20f, 0.52f, 0.59f, 0.71f, 0.75f, 0.77f, 0.81f};
  const double t_end = static_cast<double>(((1u << bits) - 1u) + kNumEnum) / max_value;
  const double t_start = t_end * kTightStart[bits];

  vec<int> cur_codes(dim);
  double sqr_denominator = static_cast<double>(dim) * 0.25;
  double numerator = 0.0;
  for (size_t i = 0; i < dim; ++i) {
    const int current = static_cast<int>((t_start * values[i]) + kEps);
    cur_codes[i] = current;
    sqr_denominator += current * current + current;
    numerator += (current + 0.5) * values[i];
  }

  std::priority_queue<std::pair<double, size_t>, vec<std::pair<double, size_t>>, std::greater<>> next_t;
  for (size_t i = 0; i < dim; ++i) {
    if (values[i] > kEps) {
      next_t.emplace(static_cast<double>(cur_codes[i] + 1) / values[i], i);
    }
  }

  double max_ip = 0.0;
  double best_t = t_start;
  while (!next_t.empty()) {
    const auto [current_t, update_id] = next_t.top();
    next_t.pop();

    ++cur_codes[update_id];
    const int updated = cur_codes[update_id];
    sqr_denominator += 2.0 * updated;
    numerator += values[update_id];

    const double current_ip = numerator / std::sqrt(sqr_denominator);
    if (current_ip > max_ip) {
      max_ip = current_ip;
      best_t = current_t;
    }

    if (updated < (1 << bits) - 1 && values[update_id] > kEps) {
      const double next_value = static_cast<double>(updated + 1) / values[update_id];
      if (next_value < t_end) {
        next_t.emplace(next_value, update_id);
      }
    }
  }

  return best_t;
}

inline float compute_rabitq_t_const(size_t dim, size_t bits) {
  constexpr size_t kSamples = 128;
  std::mt19937_64 rng(1234ULL);
  std::normal_distribution<float> normal(0.0f, 1.0f);

  vec<float> sample(dim);
  vec<float> abs_normalized(dim);
  double total = 0.0;
  for (size_t sample_id = 0; sample_id < kSamples; ++sample_id) {
    double norm_sqr = 0.0;
    for (size_t i = 0; i < dim; ++i) {
      sample[i] = normal(rng);
      norm_sqr += static_cast<double>(sample[i]) * sample[i];
    }

    const double norm = std::sqrt(std::max(norm_sqr, 1e-12));
    for (size_t i = 0; i < dim; ++i) {
      abs_normalized[i] = std::abs(static_cast<float>(sample[i] / norm));
    }

    total += best_rescale_factor(abs_normalized.data(), dim, bits);
  }

  return static_cast<float>(total / static_cast<double>(kSamples));
}

template <u32 PadDim>
GpuVector<PadDim> compute_centroid_host(const GpuVector<PadDim>* vectors, size_t count) {
  GpuVector<PadDim> centroid;
  if (count == 0) return centroid;

  for (size_t i = 0; i < count; ++i) {
    for (u32 dim = 0; dim < PadDim; ++dim) {
      centroid.data[dim] += vectors[i].data[dim];
    }
  }

  const float inv_count = 1.0f / static_cast<float>(count);
  for (u32 dim = 0; dim < PadDim; ++dim) {
    centroid.data[dim] *= inv_count;
  }
  return centroid;
}

template <u32 PadDim>
int compute_medoid_via_centroid_host(const GpuVector<PadDim>* vectors,
                                     size_t count,
                                     const GpuVector<PadDim>& centroid) {
  if (count == 0) return -1;

  int best_idx = 0;
  float best_dist = std::numeric_limits<float>::max();
  for (size_t i = 0; i < count; ++i) {
    float dist = 0.0f;
    for (u32 dim = 0; dim < PadDim; ++dim) {
      const float diff = vectors[i].data[dim] - centroid.data[dim];
      dist += diff * diff;
    }
    if (dist < best_dist) {
      best_dist = dist;
      best_idx = static_cast<int>(i);
    }
  }
  return best_idx;
}

template <u32 PadDim>
void build_signed_permutation(u64 seed, vec<u32>& permutation, vec<float>& sign) {
  permutation.resize(PadDim);
  std::iota(permutation.begin(), permutation.end(), 0u);
  std::mt19937_64 rng(seed);
  std::shuffle(permutation.begin(), permutation.end(), rng);

  sign.resize(PadDim);
  for (u32 dim = 0; dim < PadDim; ++dim) {
    sign[dim] = (rng() & 1ULL) ? 1.0f : -1.0f;
  }
}

template <u32 BlockSize>
__device__ float block_reduce_sum(float* scratch, float value) {
  scratch[threadIdx.x] = value;
  __syncthreads();

  for (u32 stride = BlockSize / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      scratch[threadIdx.x] += scratch[threadIdx.x + stride];
    }
    __syncthreads();
  }
  return scratch[0];
}

constexpr u32 kQuantizeThreads = 256;
constexpr float kRabitqEpsilon = 1e-6f;

template <u32 PadDim>
__global__ void rabitq_quantize_batch_kernel(const GpuVector<PadDim>* vectors,
                                             u32 num_vectors,
                                             const u32* permutation,
                                             const float* sign,
                                             const float* centroid,
                                             float t_const,
                                             Rabitq4Code<PadDim>* out_codes) {
  const u32 vector_id = blockIdx.x;
  if (vector_id >= num_vectors) return;

  extern __shared__ unsigned char shared_bytes[];
  auto* residual = reinterpret_cast<float*>(shared_bytes);
  auto* code = reinterpret_cast<std::uint8_t*>(shared_bytes + sizeof(float) * PadDim);
  __shared__ float reduce_scratch[kQuantizeThreads];

  float l2_sqr_local = 0.0f;
  for (u32 dim = threadIdx.x; dim < PadDim; dim += blockDim.x) {
    const u32 source_dim = permutation[dim];
    const float rotated_centroid = sign[dim] * centroid[source_dim];
    const float rotated_value = sign[dim] * vectors[vector_id].data[source_dim];
    const float residual_value = rotated_value - rotated_centroid;
    residual[dim] = residual_value;
    l2_sqr_local += residual_value * residual_value;
  }
  const float l2_sqr = block_reduce_sum<kQuantizeThreads>(reduce_scratch, l2_sqr_local);
  const float l2_norm = sqrtf(fmaxf(l2_sqr, kRabitqEpsilon));

  float ip_norm_local = 0.0f;
  for (u32 dim = threadIdx.x; dim < PadDim; dim += blockDim.x) {
    const float abs_residual = fabsf(residual[dim] / l2_norm);
    int value = static_cast<int>((t_const * abs_residual) + 1e-5f);
    value = min(value, (1 << (kRabitqBits - 1)) - 1);
    code[dim] = static_cast<std::uint8_t>(value);
    ip_norm_local += (static_cast<float>(value) + 0.5f) * abs_residual;
  }
  const float ip_norm = block_reduce_sum<kQuantizeThreads>(reduce_scratch, ip_norm_local);
  const float ipnorm_inv = ip_norm > kRabitqEpsilon ? (1.0f / ip_norm) : 1.0f;

  const float center_bias = -(static_cast<float>(1 << (kRabitqBits - 1)) - 0.5f);
  float ip_residual_code_local = 0.0f;
  float ip_centroid_code_local = 0.0f;
  for (u32 dim = threadIdx.x; dim < PadDim; dim += blockDim.x) {
    std::uint8_t encoded = code[dim];
    if (residual[dim] >= 0.0f) {
      encoded = static_cast<std::uint8_t>(encoded + (1 << (kRabitqBits - 1)));
    } else {
      encoded = static_cast<std::uint8_t>((~encoded) & ((1 << (kRabitqBits - 1)) - 1));
    }
    code[dim] = encoded;

    const u32 source_dim = permutation[dim];
    const float rotated_centroid = sign[dim] * centroid[source_dim];
    const float shifted_code = static_cast<float>(encoded) + center_bias;
    ip_residual_code_local += residual[dim] * shifted_code;
    ip_centroid_code_local += rotated_centroid * shifted_code;
  }
  float ip_residual_code = block_reduce_sum<kQuantizeThreads>(reduce_scratch, ip_residual_code_local);
  const float ip_centroid_code = block_reduce_sum<kQuantizeThreads>(reduce_scratch, ip_centroid_code_local);

  if (threadIdx.x == 0) {
    if (fabsf(ip_residual_code) < kRabitqEpsilon) {
      ip_residual_code = copysignf(kRabitqEpsilon, ip_residual_code == 0.0f ? 1.0f : ip_residual_code);
    }
    out_codes[vector_id].add = l2_sqr + (2.0f * l2_sqr * ip_centroid_code / ip_residual_code);
    out_codes[vector_id].rescale = ipnorm_inv * -2.0f * l2_norm;
  }

  constexpr u32 bytes_per_code = (kRabitqBits * PadDim + 7) / 8;
  static_assert(PadDim % 2 == 0, "4-bit packing requires even PadDim");
  for (u32 byte_idx = threadIdx.x; byte_idx < bytes_per_code; byte_idx += blockDim.x) {
    const u32 dim0 = byte_idx * 2;
    const u32 dim1 = dim0 + 1;
    out_codes[vector_id].data[byte_idx] =
      static_cast<std::uint8_t>((code[dim0] & 0xF) | ((code[dim1] & 0xF) << 4));
  }
}

template <u32 PadDim>
__global__ void rabitq_prepare_queries_kernel(const GpuVector<PadDim>* queries,
                                              u32 num_queries,
                                              const u32* permutation,
                                              const float* sign,
                                              const float* centroid,
                                              float* rotated_queries,
                                              RabitqQueryFactor<PadDim>* factors) {
  const u32 query_id = blockIdx.x;
  if (query_id >= num_queries) return;

  __shared__ float reduce_scratch[kQuantizeThreads];

  float sqr_norm_local = 0.0f;
  float sum_local = 0.0f;
  for (u32 dim = threadIdx.x; dim < PadDim; dim += blockDim.x) {
    const u32 source_dim = permutation[dim];
    const float residual = sign[dim] * (queries[query_id].data[source_dim] - centroid[source_dim]);
    rotated_queries[static_cast<size_t>(query_id) * PadDim + dim] = residual;
    sqr_norm_local += residual * residual;
    sum_local += residual;
  }

  const float sqr_norm = block_reduce_sum<kQuantizeThreads>(reduce_scratch, sqr_norm_local);
  const float sum_value = block_reduce_sum<kQuantizeThreads>(reduce_scratch, sum_local);

  if (threadIdx.x == 0) {
    factors[query_id].add = sqr_norm;
    factors[query_id].k1xSumq = -0.5f * sum_value;
    factors[query_id].kBxSumq = -0.5f * static_cast<float>((1 << kRabitqBits) - 1) * sum_value;
  }
}

template <u32 PadDim>
struct SearchCandidate {
  u32 vertex_id{};
  float distance{};
  bool expanded{};
  RecordHandle<PadDim> record;
};

template <u32 PadDim>
struct SearchState {
  u32 k{};
  u32 beam_width{};
  vec<SearchCandidate<PadDim>> frontier;
  vec<SearchCandidate<PadDim>> discovered;
  hashset_t<u32> visited;
};

template <u32 PadDim>
__global__ void exact_distance_pairs_kernel(const GpuVector<PadDim>* queries,
                                            const GpuVector<PadDim>* candidates,
                                            const u32* query_indices,
                                            u32 num_pairs,
                                            float* out_distances) {
  auto cta = cg::this_thread_block();
  auto tile = cg::tiled_partition<4>(cta);
  const u32 tiles_per_block = blockDim.x / tile.size();
  const u32 pair_id = blockIdx.x * tiles_per_block + tile.meta_group_rank();
  if (pair_id >= num_pairs) return;

  const auto* query_ptr = reinterpret_cast<const float4*>(queries[query_indices[pair_id]].data);
  const auto* candidate_ptr = reinterpret_cast<const float4*>(candidates[pair_id].data);

  float distance_partial = 0.0f;
  for (u32 chunk = tile.thread_rank(); chunk < PadDim / 4; chunk += tile.size()) {
    const float4 q = query_ptr[chunk];
    const float4 c = candidate_ptr[chunk];
    const float dx = q.x - c.x;
    const float dy = q.y - c.y;
    const float dz = q.z - c.z;
    const float dw = q.w - c.w;
    distance_partial += dx * dx + dy * dy + dz * dz + dw * dw;
  }
  const float distance = cg::reduce(tile, distance_partial, cg::plus<float>());
  if (tile.thread_rank() == 0) {
    out_distances[pair_id] = distance;
  }
}

template <u32 PadDim>
__global__ void rabitq_distance_pairs_kernel(const float* rotated_queries,
                                             const RabitqQueryFactor<PadDim>* query_factors,
                                             const u32* query_indices,
                                             const Rabitq4Code<PadDim>* candidates,
                                             u32 num_pairs,
                                             float* out_distances) {
  auto cta = cg::this_thread_block();
  auto tile = cg::tiled_partition<4>(cta);
  const u32 tiles_per_block = blockDim.x / tile.size();
  const u32 pair_id = blockIdx.x * tiles_per_block + tile.meta_group_rank();
  if (pair_id >= num_pairs) return;

  const u32 query_id = query_indices[pair_id];
  const float* rotated_query = rotated_queries + query_id * PadDim;
  const auto& query_factor = query_factors[query_id];
  const auto* packed_code = reinterpret_cast<const std::uint16_t*>(candidates[pair_id].data);
  const auto* query_ptr = reinterpret_cast<const float4*>(rotated_query);

  float dot_partial = 0.0f;
  for (u32 chunk = tile.thread_rank(); chunk < PadDim / 4; chunk += tile.size()) {
    const float4 query_chunk = query_ptr[chunk];
    const std::uint16_t packed = packed_code[chunk];
    const float c0 = static_cast<float>(packed & 0xF);
    const float c1 = static_cast<float>((packed >> 4) & 0xF);
    const float c2 = static_cast<float>((packed >> 8) & 0xF);
    const float c3 = static_cast<float>((packed >> 12) & 0xF);
    dot_partial += c0 * query_chunk.x + c1 * query_chunk.y + c2 * query_chunk.z + c3 * query_chunk.w;
  }
  const float dot = cg::reduce(tile, dot_partial, cg::plus<float>());
  const float distance =
    candidates[pair_id].add + query_factor.add +
    candidates[pair_id].rescale * (dot + query_factor.k1xSumq * static_cast<float>((1 << kRabitqBits) - 1));
  if (tile.thread_rank() == 0) {
    out_distances[pair_id] = distance;
  }
}

template <typename T>
T* cuda_malloc_array(size_t count) {
  T* ptr = nullptr;
  cudaError_t status = cudaMalloc(&ptr, sizeof(T) * count);
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string("cudaMalloc failed: ") + cudaGetErrorString(status));
  }
  return ptr;
}

template <typename T>
T* cuda_malloc_host_array(size_t count) {
  T* ptr = nullptr;
  const cudaError_t status = cudaMallocHost(reinterpret_cast<void**>(&ptr), sizeof(T) * count);
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string("cudaMallocHost failed: ") + cudaGetErrorString(status));
  }
  return ptr;
}

template <typename T>
void cuda_copy_to_device(T* dst, const T* src, size_t count) {
  const cudaError_t status = cudaMemcpy(dst, src, sizeof(T) * count, cudaMemcpyHostToDevice);
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string("cudaMemcpy H2D failed: ") + cudaGetErrorString(status));
  }
}

template <typename T>
void cuda_copy_to_host(T* dst, const T* src, size_t count) {
  const cudaError_t status = cudaMemcpy(dst, src, sizeof(T) * count, cudaMemcpyDeviceToHost);
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string("cudaMemcpy D2H failed: ") + cudaGetErrorString(status));
  }
}

template <typename T>
void cuda_copy_to_device_async(T* dst, const T* src, size_t count, cudaStream_t stream) {
  const cudaError_t status = cudaMemcpyAsync(dst, src, sizeof(T) * count, cudaMemcpyHostToDevice, stream);
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string("cudaMemcpyAsync H2D failed: ") + cudaGetErrorString(status));
  }
}

template <typename T>
void cuda_copy_to_host_async(T* dst, const T* src, size_t count, cudaStream_t stream) {
  const cudaError_t status = cudaMemcpyAsync(dst, src, sizeof(T) * count, cudaMemcpyDeviceToHost, stream);
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string("cudaMemcpyAsync D2H failed: ") + cudaGetErrorString(status));
  }
}

template <typename RecordT>
class RecordCache {
public:
  RecordCache(size_t cache_bytes, bool enabled)
      : enabled_(enabled),
        max_entries_(enabled ? std::max<size_t>(1, cache_bytes / sizeof(RecordT)) : 0) {}

  std::shared_ptr<RecordT> get(u32 key) {
    if (!enabled_) return {};
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = entries_.find(key);
    return it == entries_.end() ? std::shared_ptr<RecordT>{} : it->second;
  }

  void insert(u32 key, const std::shared_ptr<RecordT>& value) {
    if (!enabled_) return;
    std::lock_guard<std::mutex> lock(mutex_);
    auto [it, inserted] = entries_.emplace(key, value);
    if (!inserted) {
      it->second = value;
      return;
    }

    fifo_.push_back(key);
    while (entries_.size() > max_entries_) {
      const u32 victim = fifo_.front();
      fifo_.pop_front();
      auto victim_it = entries_.find(victim);
      if (victim_it != entries_.end() && victim_it->first == victim) {
        entries_.erase(victim_it);
      }
    }
  }

  void clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    entries_.clear();
    fifo_.clear();
  }

private:
  const bool enabled_;
  const size_t max_entries_;
  std::mutex mutex_;
  std::unordered_map<u32, std::shared_ptr<RecordT>> entries_;
  std::deque<u32> fifo_;
};

class SlotPool {
public:
  SlotPool(BufferAllocator& allocator, u32 num_threads, size_t slot_size)
      : allocator_(allocator), slot_size_(slot_size), freelists_(num_threads) {}

  byte_t* allocate(u32 thread_id) {
    byte_t* ptr = nullptr;
    if (!freelists_[thread_id].try_dequeue(ptr)) {
      ptr = allocator_.allocate_bytes(slot_size_);
    }
    return ptr;
  }

  void free(byte_t* ptr, u32 thread_id) { freelists_[thread_id].enqueue(ptr); }

private:
  BufferAllocator& allocator_;
  const size_t slot_size_;
  vec<concurrent_queue<byte_t*>> freelists_;
};

inline void cuda_check(cudaError_t status, const char* action) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(action) + ": " + cudaGetErrorString(status));
  }
}

}  // namespace

struct GpuIndex::ImplBase {
  virtual ~ImplBase() = default;
  virtual void initialize_or_bootstrap(const u_ptr<ComputeThread>& thread, bool is_initiator) = 0;
  virtual void reload_from_remote(const u_ptr<ComputeThread>& thread) = 0;
  virtual void clear_cache() = 0;
  virtual vec<bool> insert_batch(const vec<PendingInsert>& batch, const u_ptr<ComputeThread>& thread) = 0;
  virtual vec<vec<node_t>> search_batch(const vec<PendingQuery>& batch, const u_ptr<ComputeThread>& thread) = 0;
  virtual u32 size() const = 0;
  virtual u32 dim() const = 0;
  virtual u32 pad_dim() const = 0;
  virtual u32 max_vectors() const = 0;
  virtual vec<element_t> routing_centroid() const = 0;
};

template <u32 PadDim>
class GpuIndexImpl final : public GpuIndex::ImplBase {
public:
  GpuIndexImpl(const configuration::IndexConfiguration& config,
               BufferAllocator& buffer_allocator,
               u32 num_memory_nodes,
               size_t cache_bytes)
      : config_(config),
        buffer_allocator_(buffer_allocator),
        num_memory_nodes_(num_memory_nodes),
        r_(config.m),
        ef_search_(std::max(config.ef_search, config.k)),
        ef_construction_(config.ef_construction),
        gpu_device_(config.gpu_device),
        record_bytes_(sizeof(Record<PadDim>)),
        rabitq_t_const_(compute_rabitq_t_const(PadDim, kRabitqBits)),
        record_pool_(buffer_allocator, config.num_threads, sizeof(Record<PadDim>)),
        cache_(cache_bytes, config.use_cache) {
    select_device();
    streams_.resize(config.num_threads, nullptr);
    for (auto& stream : streams_) {
      cuda_check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "cudaStreamCreateWithFlags");
    }
    if (r_ == 0 || r_ > kMaxR) {
      throw std::runtime_error("GPU baseline requires 1 <= m <= 128");
    }
    metadata_.magic = kIndexMagic;
    metadata_.version = kIndexVersion;
    metadata_.dim = config_.dim;
    metadata_.pad_dim = PadDim;
    metadata_.max_vectors = config_.max_vectors;
    metadata_.num_memory_nodes = num_memory_nodes_;
    metadata_.r = r_;
    metadata_.record_bytes = record_bytes_;
    metadata_.shard_base = align_up(sizeof(IndexMetadata), 4096);
    metadata_.rotation_seed = config_.seed == -1 ? 1234 : static_cast<u64>(config_.seed);
  }

  ~GpuIndexImpl() override {
    select_device();
    for (auto& stream : streams_) {
      if (stream) cudaStreamDestroy(stream);
    }
    if (d_permutation_) cudaFree(d_permutation_);
    if (d_sign_) cudaFree(d_sign_);
    if (d_centroid_) cudaFree(d_centroid_);
  }

  void initialize_or_bootstrap(const u_ptr<ComputeThread>& thread, bool is_initiator) override {
    select_device();
    ensure_gpu_log_init();

    IndexMetadata remote = read_metadata(thread);
    if (remote.magic != kIndexMagic || remote.version != kIndexVersion || remote.pad_dim != PadDim) {
      if (!is_initiator) {
        throw std::runtime_error("GPU index metadata missing or incompatible on non-initiator");
      }
      bootstrap_empty(thread);
      return;
    }

    metadata_ = remote;
    maybe_expand_loaded_capacity(thread, is_initiator);
    rebuild_gpu_state();
    rebuild_user_map(thread, is_initiator);
  }

  void reload_from_remote(const u_ptr<ComputeThread>& thread) override {
    std::unique_lock<std::shared_mutex> lock(rw_lock_);
    select_device();
    IndexMetadata remote = read_metadata(thread);
    if (remote.magic != kIndexMagic || remote.version != kIndexVersion || remote.pad_dim != PadDim) {
      throw std::runtime_error("Loaded GPU index metadata is invalid or incompatible");
    }
    metadata_ = remote;
    maybe_expand_loaded_capacity(thread, true);
    cache_.clear();
    rebuild_gpu_state();
    rebuild_user_map(thread, true);
  }

  void clear_cache() override { cache_.clear(); }

  vec<bool> insert_batch(const vec<PendingInsert>& batch, const u_ptr<ComputeThread>& thread) override {
    std::unique_lock<std::shared_mutex> index_lock(rw_lock_);
    select_device();
    const cudaStream_t stream = stream_for_thread(thread);

    vec<bool> results(batch.size(), false);
    if (batch.empty()) return results;

    vec<idx_t> accepted_indices;
    accepted_indices.reserve(batch.size());
    hashset_t<node_t> seen_user_ids;
    for (idx_t i = 0; i < batch.size(); ++i) {
      if (user_to_vertex_.find(batch[i].user_id) == user_to_vertex_.end() &&
          seen_user_ids.insert(batch[i].user_id).second) {
        accepted_indices.push_back(i);
      }
    }
    if (accepted_indices.empty()) return results;

    auto* pinned_vectors = cuda_malloc_host_array<GpuVector<PadDim>>(accepted_indices.size());
    vec<node_t> accepted_user_ids;
    accepted_user_ids.reserve(accepted_indices.size());
    for (idx_t out = 0; out < accepted_indices.size(); ++out) {
      const idx_t src = accepted_indices[out];
      fill_vector(pinned_vectors[out], batch[src].components);
      accepted_user_ids.push_back(batch[src].user_id);
    }

    if (metadata_.n_vertices == 0) {
      const auto centroid = compute_centroid_host(pinned_vectors, accepted_indices.size());
      for (u32 d = 0; d < PadDim; ++d) {
        metadata_.centroid[d] = centroid.data[d];
      }
      metadata_.rotation_seed = config_.seed == -1 ? 1234 : static_cast<u64>(config_.seed);
      rebuild_gpu_state();
    } else if (!d_permutation_ || !d_sign_ || !d_centroid_) {
      rebuild_gpu_state();
    }

    vec<Rabitq4Code<PadDim>> host_codes = quantize_vectors(pinned_vectors, accepted_indices.size(), stream);

    vec<idx_t> insertion_order;
    insertion_order.reserve(accepted_indices.size());
    if (metadata_.n_vertices == 0) {
      const auto centroid = compute_centroid_host(pinned_vectors, accepted_indices.size());
      const int medoid_local = compute_medoid_via_centroid_host(pinned_vectors, accepted_indices.size(), centroid);
      insertion_order.push_back(static_cast<idx_t>(medoid_local));
      for (idx_t i = 0; i < accepted_indices.size(); ++i) {
        if (i != static_cast<idx_t>(medoid_local)) insertion_order.push_back(i);
      }
    } else {
      for (idx_t i = 0; i < accepted_indices.size(); ++i) insertion_order.push_back(i);
    }

    bool wrote_new_data = false;
    for (idx_t local_idx : insertion_order) {
      if (metadata_.n_vertices >= metadata_.max_vectors) break;

      const u32 internal_id = metadata_.n_vertices;
      const node_t user_id = accepted_user_ids[local_idx];

      auto record = allocate_record(thread->get_id());
      record->user_id = user_id;
      record->degree = 0;
      std::memcpy(record->vector, pinned_vectors[local_idx].data, sizeof(record->vector));
      record->rabitq = host_codes[local_idx];

      if (metadata_.n_vertices == 0) {
        metadata_.medoid_id = 0;
      } else {
        vec<GpuVector<PadDim>> single_query{pinned_vectors[local_idx]};
        const u32 construction_width = std::max(ef_construction_, r_);
        vec<u32> widths{construction_width};
        const auto states = run_search_exact(single_query, widths, widths, false, thread);
        const auto neighbors = robust_prune(record.get(), states[0].discovered);
        record->degree = static_cast<std::uint8_t>(neighbors.size());
        for (idx_t i = 0; i < neighbors.size(); ++i) {
          record->neighbors[i] = neighbors[i];
        }
      }

      write_record(internal_id, record, thread);
      cache_.insert(internal_id, record);
      user_to_vertex_[user_id] = internal_id;
      ++metadata_.n_vertices;
      wrote_new_data = true;
      results[accepted_indices[local_idx]] = true;

      if (record->degree > 0) {
        for (u32 i = 0; i < record->degree; ++i) {
          const u32 neighbor_id = record->neighbors[i];
          vec<u32> single_neighbor_id{neighbor_id};
          auto neighbor_record = fetch_records(single_neighbor_id, thread)[0];
          auto merged = merge_neighbor_candidates(neighbor_record, internal_id, record, thread);
          const auto updated_neighbors = robust_prune(neighbor_record.get(), merged);
          neighbor_record->degree = static_cast<std::uint8_t>(updated_neighbors.size());
          for (idx_t j = 0; j < updated_neighbors.size(); ++j) {
            neighbor_record->neighbors[j] = updated_neighbors[j];
          }
          for (idx_t j = updated_neighbors.size(); j < kMaxR; ++j) {
            neighbor_record->neighbors[j] = 0;
          }
          write_record(neighbor_id, neighbor_record, thread);
          cache_.insert(neighbor_id, neighbor_record);
        }
      }
    }

    cudaFreeHost(pinned_vectors);

    if (wrote_new_data) {
      write_metadata_all(thread);
    }

    return results;
  }

  vec<vec<node_t>> search_batch(const vec<PendingQuery>& batch, const u_ptr<ComputeThread>& thread) override {
    std::shared_lock<std::shared_mutex> index_lock(rw_lock_);
    select_device();
    const cudaStream_t stream = stream_for_thread(thread);

    vec<vec<node_t>> results(batch.size());
    if (batch.empty() || metadata_.n_vertices == 0) return results;

    vec<GpuVector<PadDim>> queries(batch.size());
    vec<u32> beam_widths(batch.size());
    for (idx_t i = 0; i < batch.size(); ++i) {
      fill_vector(queries[i], batch[i].components);
      beam_widths[i] = std::max(batch[i].k, ef_search_);
    }

    vec<u32> requested_ks(batch.size());
    for (idx_t i = 0; i < batch.size(); ++i) {
      requested_ks[i] = std::max<u32>(1, batch[i].k);
    }

    const auto states = run_search_rabitq(queries, beam_widths, requested_ks, thread);
    for (idx_t qid = 0; qid < states.size(); ++qid) {
      vec<SearchCandidate<PadDim>> rerank_candidates = states[qid].discovered;
      std::sort(rerank_candidates.begin(), rerank_candidates.end(), by_distance);

      vec<u32> pair_query_indices;
      vec<RecordHandle<PadDim>> pair_records;
      pair_query_indices.reserve(rerank_candidates.size());
      pair_records.reserve(rerank_candidates.size());
      for (const auto& candidate : rerank_candidates) {
        pair_query_indices.push_back(0);
        pair_records.push_back(candidate.record);
      }

      vec<GpuVector<PadDim>> single_query{queries[qid]};
      vec<float> exact_distances = compute_exact_distances(single_query, pair_query_indices, pair_records, stream);

      vec<std::pair<float, node_t>> ordered;
      ordered.reserve(rerank_candidates.size());
      for (idx_t i = 0; i < rerank_candidates.size(); ++i) {
        ordered.push_back({exact_distances[i], rerank_candidates[i].record->user_id});
      }
      std::sort(ordered.begin(), ordered.end(), [](const auto& lhs, const auto& rhs) { return lhs.first < rhs.first; });

      const u32 k = std::min<u32>(batch[qid].k, ordered.size());
      results[qid].reserve(k);
      for (u32 i = 0; i < k; ++i) {
        results[qid].push_back(ordered[i].second);
      }
    }

    return results;
  }

  u32 size() const override { return metadata_.n_vertices; }
  u32 dim() const override { return metadata_.dim; }
  u32 pad_dim() const override { return metadata_.pad_dim; }
  u32 max_vectors() const override { return metadata_.max_vectors; }
  vec<element_t> routing_centroid() const override {
    return vec<element_t>(metadata_.centroid, metadata_.centroid + metadata_.dim);
  }

private:
  static bool by_distance(const SearchCandidate<PadDim>& lhs, const SearchCandidate<PadDim>& rhs) {
    if (lhs.distance == rhs.distance) return lhs.vertex_id < rhs.vertex_id;
    return lhs.distance < rhs.distance;
  }

  void select_device() const { cuda_check(cudaSetDevice(gpu_device_), "cudaSetDevice"); }
  cudaStream_t stream_for_thread(const u_ptr<ComputeThread>& thread) const {
    return streams_[thread->get_id()];
  }

  void bootstrap_empty(const u_ptr<ComputeThread>& thread) {
    metadata_ = {};
    metadata_.shard_base = align_up(sizeof(IndexMetadata), 4096);
    metadata_.used_bytes = metadata_.shard_base;
    metadata_.magic = kIndexMagic;
    metadata_.version = kIndexVersion;
    metadata_.dim = config_.dim;
    metadata_.pad_dim = PadDim;
    metadata_.max_vectors = config_.max_vectors;
    metadata_.num_memory_nodes = num_memory_nodes_;
    metadata_.r = r_;
    metadata_.record_bytes = record_bytes_;
    metadata_.rotation_seed = config_.seed == -1 ? 1234 : static_cast<u64>(config_.seed);
    rebuild_gpu_state();
    cache_.clear();
    user_to_vertex_.clear();
    write_metadata_all(thread);
  }

  void rebuild_gpu_state() {
    if (d_permutation_) {
      cudaFree(d_permutation_);
      d_permutation_ = nullptr;
    }
    if (d_sign_) {
      cudaFree(d_sign_);
      d_sign_ = nullptr;
    }
    if (d_centroid_) {
      cudaFree(d_centroid_);
      d_centroid_ = nullptr;
    }

    vec<u32> permutation;
    vec<float> sign;
    build_signed_permutation<PadDim>(metadata_.rotation_seed, permutation, sign);

    d_permutation_ = cuda_malloc_array<u32>(PadDim);
    d_sign_ = cuda_malloc_array<float>(PadDim);
    d_centroid_ = cuda_malloc_array<float>(PadDim);
    cuda_copy_to_device(d_permutation_, permutation.data(), PadDim);
    cuda_copy_to_device(d_sign_, sign.data(), PadDim);
    cuda_copy_to_device(d_centroid_, metadata_.centroid, PadDim);
  }

  void rebuild_user_map(const u_ptr<ComputeThread>& thread, bool allow_remote_write) {
    user_to_vertex_.clear();
    if (metadata_.n_vertices == 0) return;

    u32 best_medoid = metadata_.medoid_id;
    float best_distance = std::numeric_limits<float>::infinity();
    for (u32 vertex_id = 0; vertex_id < metadata_.n_vertices; ++vertex_id) {
      auto record = read_record_uncached(vertex_id, thread);
      user_to_vertex_[record->user_id] = vertex_id;

      float distance = 0.0f;
      for (u32 d = 0; d < PadDim; ++d) {
        const float diff = record->vector[d] - metadata_.centroid[d];
        distance += diff * diff;
      }

      if (distance < best_distance) {
        best_distance = distance;
        best_medoid = vertex_id;
      }
    }

    if (best_medoid != metadata_.medoid_id) {
      metadata_.medoid_id = best_medoid;
      if (allow_remote_write) {
        write_metadata_all(thread);
      }
    }
  }

  void maybe_expand_loaded_capacity(const u_ptr<ComputeThread>& thread, bool allow_remote_write) {
    const u32 desired_capacity = std::max(metadata_.n_vertices, config_.max_vectors);
    if (desired_capacity <= metadata_.max_vectors) return;

    metadata_.max_vectors = desired_capacity;
    if (allow_remote_write) {
      write_metadata_all(thread);
    }
  }

  u32 local_index(u32 vertex_id) const { return vertex_id / num_memory_nodes_; }
  u32 memory_node(u32 vertex_id) const { return vertex_id % num_memory_nodes_; }
  u64 record_offset(u32 vertex_id) const {
    return metadata_.shard_base + static_cast<u64>(local_index(vertex_id)) * record_bytes_;
  }

  u64 records_on_node(u32 node, u32 n_vertices) const {
    if (n_vertices <= node) return 0;
    return ((n_vertices - node) + num_memory_nodes_ - 1) / num_memory_nodes_;
  }

  u64 used_bytes_for_node(u32 node) const { return metadata_.shard_base + records_on_node(node, metadata_.n_vertices) * record_bytes_; }

  void wait_slot(const u_ptr<ComputeThread>& thread, u32 slot_id = 0) const {
    while (!thread->is_ready(slot_id)) {
      thread->poll_cq();
      std::this_thread::yield();
    }
  }

  IndexMetadata read_metadata(const u_ptr<ComputeThread>& thread) {
    thread->set_current_coroutine(0);
    byte_t* raw = buffer_allocator_.allocate_bytes(sizeof(IndexMetadata));
    thread->track_post();

    const QP& qp = thread->ctx->qps[0]->qp;
    qp->post_send(reinterpret_cast<u64>(raw),
                  sizeof(IndexMetadata),
                  thread->ctx->get_lkey(),
                  IBV_WR_RDMA_READ,
                  true,
                  false,
                  thread->ctx->get_remote_mrt(0),
                  0,
                  0,
                  thread->create_wr_id());
    wait_slot(thread);

    IndexMetadata metadata;
    std::memcpy(&metadata, raw, sizeof(metadata));
    return metadata;
  }

  void write_metadata_all(const u_ptr<ComputeThread>& thread) {
    IndexMetadata metadata_copy = metadata_;
    byte_t* raw = buffer_allocator_.allocate_bytes(sizeof(IndexMetadata));
    thread->set_current_coroutine(0);

    for (u32 mn = 0; mn < num_memory_nodes_; ++mn) {
      metadata_copy.used_bytes = used_bytes_for_node(mn);
      std::memcpy(raw, &metadata_copy, sizeof(metadata_copy));
      thread->track_post();
      const QP& qp = thread->ctx->qps[mn]->qp;
      qp->post_send(reinterpret_cast<u64>(raw),
                    sizeof(IndexMetadata),
                    thread->ctx->get_lkey(),
                    IBV_WR_RDMA_WRITE,
                    true,
                    false,
                    thread->ctx->get_remote_mrt(mn),
                    0,
                    0,
                    thread->create_wr_id());
      wait_slot(thread);
    }
  }

  RecordHandle<PadDim> allocate_record(u32 thread_id) {
    byte_t* raw = record_pool_.allocate(thread_id);
    auto deleter = [this, thread_id](Record<PadDim>* record) {
      record_pool_.free(reinterpret_cast<byte_t*>(record), thread_id);
    };
    return RecordHandle<PadDim>(reinterpret_cast<Record<PadDim>*>(raw), std::move(deleter));
  }

  RecordHandle<PadDim> read_record_uncached(u32 vertex_id, const u_ptr<ComputeThread>& thread) {
    auto record = allocate_record(thread->get_id());
    thread->set_current_coroutine(0);
    thread->track_post();

    const u32 mn = memory_node(vertex_id);
    const QP& qp = thread->ctx->qps[mn]->qp;
    qp->post_send(reinterpret_cast<u64>(record.get()),
                  record_bytes_,
                  thread->ctx->get_lkey(),
                  IBV_WR_RDMA_READ,
                  true,
                  false,
                  thread->ctx->get_remote_mrt(mn),
                  record_offset(vertex_id),
                  0,
                  thread->create_wr_id());
    wait_slot(thread);
    return record;
  }

  vec<RecordHandle<PadDim>> fetch_records(const vec<u32>& vertex_ids, const u_ptr<ComputeThread>& thread) {
    vec<RecordHandle<PadDim>> records(vertex_ids.size());
    if (vertex_ids.empty()) return records;

    std::unordered_map<u32, vec<idx_t>> positions;
    positions.reserve(vertex_ids.size());
    for (idx_t i = 0; i < vertex_ids.size(); ++i) {
      positions[vertex_ids[i]].push_back(i);
    }

    struct PendingRead {
      u32 vertex_id{};
      RecordHandle<PadDim> record;
    };

    std::unordered_map<u32, PendingRead> pending_by_id;
    vec<PendingRead> pending_reads;
    for (const auto& [vertex_id, pos] : positions) {
      if (auto cached = cache_.get(vertex_id)) {
        for (idx_t idx : pos) {
          records[idx] = cached;
        }
        continue;
      }

      auto handle = allocate_record(thread->get_id());
      pending_reads.push_back({vertex_id, handle});
      pending_by_id.emplace(vertex_id, PendingRead{vertex_id, handle});
    }

    if (!pending_reads.empty()) {
      vec<std::unique_ptr<BatchedREAD>> batches(num_memory_nodes_);
      for (u32 mn = 0; mn < num_memory_nodes_; ++mn) {
        batches[mn] = std::make_unique<BatchedREAD>(pending_reads.size());
      }

      thread->set_current_coroutine(0);
      for (const auto& pending : pending_reads) {
        const u32 mn = memory_node(pending.vertex_id);
        thread->track_post();
        batches[mn]->add_to_batch(reinterpret_cast<u64>(pending.record.get()),
                                  thread->ctx->get_remote_mrt(mn)->address + record_offset(pending.vertex_id),
                                  record_bytes_,
                                  thread->ctx->get_lkey(),
                                  thread->ctx->get_remote_mrt(mn)->rkey,
                                  thread->create_wr_id(),
                                  true);
      }

      for (u32 mn = 0; mn < num_memory_nodes_; ++mn) {
        if (batches[mn]->requests > 0) {
          batches[mn]->post_batch(thread->ctx->qps[mn]->qp);
        }
      }
      wait_slot(thread);

      for (const auto& [vertex_id, pending] : pending_by_id) {
        cache_.insert(vertex_id, pending.record);
        for (idx_t idx : positions[vertex_id]) {
          records[idx] = pending.record;
        }
      }
    }

    return records;
  }

  void write_record(u32 vertex_id, const RecordHandle<PadDim>& record, const u_ptr<ComputeThread>& thread) {
    thread->set_current_coroutine(0);
    thread->track_post();

    const u32 mn = memory_node(vertex_id);
    const QP& qp = thread->ctx->qps[mn]->qp;
    qp->post_send(reinterpret_cast<u64>(record.get()),
                  record_bytes_,
                  thread->ctx->get_lkey(),
                  IBV_WR_RDMA_WRITE,
                  true,
                  false,
                  thread->ctx->get_remote_mrt(mn),
                  record_offset(vertex_id),
                  0,
                  thread->create_wr_id());
    wait_slot(thread);
  }

  void fill_vector(GpuVector<PadDim>& dst, const element_t* src) const {
    std::fill(std::begin(dst.data), std::end(dst.data), 0.0f);
    std::memcpy(dst.data, src, sizeof(float) * config_.dim);
  }

  vec<float> compute_exact_distances(const vec<GpuVector<PadDim>>& queries,
                                     const vec<u32>& pair_query_indices,
                                     const vec<RecordHandle<PadDim>>& pair_records,
                                     cudaStream_t stream) {
    if (pair_records.empty()) return {};

    vec<GpuVector<PadDim>> candidates(pair_records.size());
    for (idx_t i = 0; i < pair_records.size(); ++i) {
      std::memcpy(candidates[i].data, pair_records[i]->vector, sizeof(candidates[i].data));
    }

    auto* d_queries = cuda_malloc_array<GpuVector<PadDim>>(queries.size());
    auto* d_candidates = cuda_malloc_array<GpuVector<PadDim>>(candidates.size());
    auto* d_pair_query_indices = cuda_malloc_array<u32>(pair_query_indices.size());
    auto* d_distances = cuda_malloc_array<float>(pair_records.size());

    cuda_copy_to_device_async(d_queries, queries.data(), queries.size(), stream);
    cuda_copy_to_device_async(d_candidates, candidates.data(), candidates.size(), stream);
    cuda_copy_to_device_async(d_pair_query_indices, pair_query_indices.data(), pair_query_indices.size(), stream);

    constexpr u32 threads_per_block = 128;
    constexpr u32 tile_size = 4;
    const u32 pairs_per_block = threads_per_block / tile_size;
    const u32 blocks = (pair_records.size() + pairs_per_block - 1) / pairs_per_block;
    exact_distance_pairs_kernel<PadDim><<<blocks, threads_per_block, 0, stream>>>(
      d_queries, d_candidates, d_pair_query_indices, pair_records.size(), d_distances);

    vec<float> host_distances(pair_records.size());
    cuda_copy_to_host_async(host_distances.data(), d_distances, host_distances.size(), stream);
    cuda_check(cudaStreamSynchronize(stream), "exact_distance_pairs_kernel");

    cudaFree(d_queries);
    cudaFree(d_candidates);
    cudaFree(d_pair_query_indices);
    cudaFree(d_distances);

    return host_distances;
  }

  vec<Rabitq4Code<PadDim>> quantize_vectors(const GpuVector<PadDim>* vectors, size_t count, cudaStream_t stream) {
    vec<Rabitq4Code<PadDim>> host_codes(count);
    if (count == 0) return host_codes;

    auto* d_vectors = cuda_malloc_array<GpuVector<PadDim>>(count);
    auto* d_codes = cuda_malloc_array<Rabitq4Code<PadDim>>(count);
    cuda_copy_to_device_async(d_vectors, vectors, count, stream);

    const size_t shared_bytes = sizeof(float) * PadDim + sizeof(std::uint8_t) * PadDim;
    rabitq_quantize_batch_kernel<PadDim><<<count, kQuantizeThreads, shared_bytes, stream>>>(
      d_vectors, count, d_permutation_, d_sign_, d_centroid_, rabitq_t_const_, d_codes);
    cuda_copy_to_host_async(host_codes.data(), d_codes, count, stream);
    cuda_check(cudaStreamSynchronize(stream), "rabitq_quantize_batch_kernel");

    cudaFree(d_vectors);
    cudaFree(d_codes);
    return host_codes;
  }

  QuantizedQueryBatch<PadDim> quantize_queries(const vec<GpuVector<PadDim>>& queries, cudaStream_t stream) {
    QuantizedQueryBatch<PadDim> batch;
    if (queries.empty()) return batch;

    auto* d_queries = cuda_malloc_array<GpuVector<PadDim>>(queries.size());
    batch.rotated_queries = cuda_malloc_array<float>(queries.size() * PadDim);
    batch.factors = cuda_malloc_array<RabitqQueryFactor<PadDim>>(queries.size());

    cuda_copy_to_device_async(d_queries, queries.data(), queries.size(), stream);
    rabitq_prepare_queries_kernel<PadDim><<<queries.size(), kQuantizeThreads, 0, stream>>>(
      d_queries, queries.size(), d_permutation_, d_sign_, d_centroid_, batch.rotated_queries, batch.factors);
    cuda_check(cudaStreamSynchronize(stream), "rabitq_prepare_queries_kernel");
    cudaFree(d_queries);
    return batch;
  }

  vec<float> compute_rabitq_distances(const QuantizedQueryBatch<PadDim>& queries,
                                      const vec<u32>& pair_query_indices,
                                      const vec<RecordHandle<PadDim>>& pair_records,
                                      cudaStream_t stream) {
    if (pair_records.empty()) return {};

    vec<Rabitq4Code<PadDim>> host_codes(pair_records.size());
    for (idx_t i = 0; i < pair_records.size(); ++i) {
      host_codes[i] = pair_records[i]->rabitq;
    }

    auto* d_codes = cuda_malloc_array<Rabitq4Code<PadDim>>(host_codes.size());
    auto* d_pair_query_indices = cuda_malloc_array<u32>(pair_query_indices.size());
    auto* d_distances = cuda_malloc_array<float>(pair_records.size());

    cuda_copy_to_device_async(d_codes, host_codes.data(), host_codes.size(), stream);
    cuda_copy_to_device_async(d_pair_query_indices, pair_query_indices.data(), pair_query_indices.size(), stream);

    constexpr u32 threads_per_block = 128;
    constexpr u32 tile_size = 4;
    const u32 pairs_per_block = threads_per_block / tile_size;
    const u32 blocks = (pair_records.size() + pairs_per_block - 1) / pairs_per_block;
    rabitq_distance_pairs_kernel<PadDim><<<blocks, threads_per_block, 0, stream>>>(
      queries.rotated_queries, queries.factors, d_pair_query_indices, d_codes, pair_records.size(), d_distances);

    vec<float> host_distances(pair_records.size());
    cuda_copy_to_host_async(host_distances.data(), d_distances, host_distances.size(), stream);
    cuda_check(cudaStreamSynchronize(stream), "rabitq_distance_pairs_kernel");

    cudaFree(d_codes);
    cudaFree(d_pair_query_indices);
    cudaFree(d_distances);

    return host_distances;
  }

  template <bool UseRabitq>
  vec<SearchState<PadDim>> run_beam_search(const vec<GpuVector<PadDim>>& queries,
                                           const vec<u32>& beam_widths,
                                           const vec<u32>& requested_ks,
                                           bool apply_cut,
                                           const u_ptr<ComputeThread>& thread) {
    vec<SearchState<PadDim>> states(queries.size());
    if (queries.empty() || metadata_.n_vertices == 0) return states;
    const cudaStream_t stream = stream_for_thread(thread);

    vec<u32> medoid_ids(queries.size(), metadata_.medoid_id);
    auto medoid_record = fetch_records({metadata_.medoid_id}, thread).front();
    vec<RecordHandle<PadDim>> medoid_records(queries.size(), medoid_record);
    vec<u32> medoid_qids(queries.size());
    for (idx_t i = 0; i < queries.size(); ++i) medoid_qids[i] = i;

    vec<float> medoid_distances;
    QuantizedQueryBatch<PadDim> quantized_queries;
    if constexpr (UseRabitq) {
      quantized_queries = quantize_queries(queries, stream);
      medoid_distances = compute_rabitq_distances(quantized_queries, medoid_qids, medoid_records, stream);
    } else {
      medoid_distances = compute_exact_distances(queries, medoid_qids, medoid_records, stream);
    }

    for (idx_t qid = 0; qid < queries.size(); ++qid) {
      states[qid].k = std::max<u32>(1, requested_ks[qid]);
      states[qid].beam_width = std::max<u32>(1, beam_widths[qid]);
      states[qid].visited.insert(metadata_.medoid_id);
      states[qid].frontier.push_back({metadata_.medoid_id, medoid_distances[qid], false, medoid_record});
      states[qid].discovered.push_back(states[qid].frontier.back());
    }

    for (u32 iter = 0; iter < kDefaultSearchLimit; ++iter) {
      vec<u32> pair_query_indices;
      vec<u32> pending_vertex_ids;

      for (idx_t qid = 0; qid < states.size(); ++qid) {
        auto& state = states[qid];
        std::sort(state.frontier.begin(), state.frontier.end(), by_distance);

        u32 expanded = 0;
        for (auto& candidate : state.frontier) {
          if (candidate.expanded) continue;
          candidate.expanded = true;
          ++expanded;

          for (u32 ni = 0; ni < candidate.record->degree && ni < r_; ++ni) {
            const u32 neighbor_id = candidate.record->neighbors[ni];
            if (state.visited.insert(neighbor_id).second) {
              pair_query_indices.push_back(qid);
              pending_vertex_ids.push_back(neighbor_id);
            }
          }

          if (expanded >= kDefaultNodesExploredPerIteration) break;
        }
      }

      if (pending_vertex_ids.empty()) break;

      auto pending_records = fetch_records(pending_vertex_ids, thread);
      vec<float> distances;
      if constexpr (UseRabitq) {
        distances = compute_rabitq_distances(quantized_queries, pair_query_indices, pending_records, stream);
      } else {
        distances = compute_exact_distances(queries, pair_query_indices, pending_records, stream);
      }

      for (idx_t i = 0; i < pending_vertex_ids.size(); ++i) {
        auto& state = states[pair_query_indices[i]];
        SearchCandidate<PadDim> candidate{pending_vertex_ids[i], distances[i], false, pending_records[i]};
        state.frontier.push_back(candidate);
        state.discovered.push_back(candidate);
      }

      for (auto& state : states) {
        std::sort(state.frontier.begin(), state.frontier.end(), by_distance);
        if (state.frontier.size() > state.beam_width) {
          state.frontier.resize(state.beam_width);
        }
        if (apply_cut && state.frontier.size() > state.k) {
          const float reference_distance = state.frontier[state.k - 1].distance;
          // RabitQ scores are only used as an ordering heuristic and can become negative.
          // A multiplicative cutoff is only valid for positive distances; otherwise it can
          // prune the full frontier and make search return zero results.
          if (reference_distance > 0.0f) {
            const float cutoff = reference_distance * kDefaultCut;
            state.frontier.erase(
              std::remove_if(state.frontier.begin(), state.frontier.end(), [&](const auto& entry) {
                return entry.distance > cutoff;
              }),
              state.frontier.end());
          }
        }
      }
    }

    return states;
  }

  vec<SearchState<PadDim>> run_search_exact(const vec<GpuVector<PadDim>>& queries,
                                            const vec<u32>& beam_widths,
                                            const vec<u32>& requested_ks,
                                            bool apply_cut,
                                            const u_ptr<ComputeThread>& thread) {
    return run_beam_search<false>(queries, beam_widths, requested_ks, apply_cut, thread);
  }

  vec<SearchState<PadDim>> run_search_rabitq(const vec<GpuVector<PadDim>>& queries,
                                             const vec<u32>& beam_widths,
                                             const vec<u32>& requested_ks,
                                             const u_ptr<ComputeThread>& thread) {
    return run_beam_search<true>(queries, beam_widths, requested_ks, true, thread);
  }

  float host_squared_distance(const float* lhs, const float* rhs) const {
    float sum = 0.0f;
    for (u32 i = 0; i < config_.dim; ++i) {
      const float diff = lhs[i] - rhs[i];
      sum += diff * diff;
    }
    return sum;
  }

  vec<u32> robust_prune(const Record<PadDim>* target, vec<SearchCandidate<PadDim>> candidates) const {
    candidates.erase(
      std::remove_if(candidates.begin(), candidates.end(), [&](const auto& candidate) {
        return candidate.vertex_id >= metadata_.n_vertices || candidate.record.get() == nullptr;
      }),
      candidates.end());
    std::sort(candidates.begin(), candidates.end(), by_distance);

    vec<u32> selected;
    selected.reserve(std::min<size_t>(r_, candidates.size()));
    vec<RecordHandle<PadDim>> selected_records;
    selected_records.reserve(selected.capacity());

    for (const auto& candidate : candidates) {
      if (selected.size() >= r_) break;
      bool pruned = false;
      for (const auto& selected_record : selected_records) {
        if (kDefaultAlpha * host_squared_distance(selected_record->vector, candidate.record->vector) <= candidate.distance) {
          pruned = true;
          break;
        }
      }
      if (!pruned) {
        selected.push_back(candidate.vertex_id);
        selected_records.push_back(candidate.record);
      }
    }

    return selected;
  }

  vec<SearchCandidate<PadDim>> merge_neighbor_candidates(const RecordHandle<PadDim>& neighbor_record,
                                                         u32 new_vertex_id,
                                                         const RecordHandle<PadDim>& new_record,
                                                         const u_ptr<ComputeThread>& thread) {
    vec<u32> candidate_ids;
    candidate_ids.reserve(neighbor_record->degree + 1);
    for (u32 i = 0; i < neighbor_record->degree; ++i) {
      candidate_ids.push_back(neighbor_record->neighbors[i]);
    }
    candidate_ids.push_back(new_vertex_id);

    auto candidate_records = fetch_records(candidate_ids, thread);
    vec<SearchCandidate<PadDim>> merged;
    merged.reserve(candidate_ids.size());
    for (idx_t i = 0; i < candidate_ids.size(); ++i) {
      merged.push_back(
        {candidate_ids[i], host_squared_distance(neighbor_record->vector, candidate_records[i]->vector), false, candidate_records[i]});
    }
    merged.back().record = new_record;
    merged.back().distance = host_squared_distance(neighbor_record->vector, new_record->vector);
    return merged;
  }

private:
  const configuration::IndexConfiguration config_;
  BufferAllocator& buffer_allocator_;
  const u32 num_memory_nodes_;
  const u32 r_;
  const u32 ef_search_;
  const u32 ef_construction_;
  const u32 gpu_device_;
  const u64 record_bytes_;
  const float rabitq_t_const_;

  mutable std::shared_mutex rw_lock_;

  IndexMetadata metadata_{};
  SlotPool record_pool_;
  RecordCache<Record<PadDim>> cache_;
  std::unordered_map<node_t, u32> user_to_vertex_;
  vec<cudaStream_t> streams_;

  u32* d_permutation_{nullptr};
  float* d_sign_{nullptr};
  float* d_centroid_{nullptr};
};

GpuIndex::GpuIndex(const configuration::IndexConfiguration& config,
                   BufferAllocator& buffer_allocator,
                   u32 num_memory_nodes,
                   size_t cache_bytes) {
  switch (choose_pad_dim(config.dim)) {
    case 128:
      impl_ = std::make_unique<GpuIndexImpl<128>>(config, buffer_allocator, num_memory_nodes, cache_bytes);
      break;
    case 256:
      impl_ = std::make_unique<GpuIndexImpl<256>>(config, buffer_allocator, num_memory_nodes, cache_bytes);
      break;
    case 512:
      impl_ = std::make_unique<GpuIndexImpl<512>>(config, buffer_allocator, num_memory_nodes, cache_bytes);
      break;
    case 1024:
      impl_ = std::make_unique<GpuIndexImpl<1024>>(config, buffer_allocator, num_memory_nodes, cache_bytes);
      break;
    default:
      throw std::runtime_error("Unsupported padded dimension");
  }
}

GpuIndex::~GpuIndex() = default;
GpuIndex::GpuIndex(GpuIndex&&) noexcept = default;
GpuIndex& GpuIndex::operator=(GpuIndex&&) noexcept = default;

void GpuIndex::initialize_or_bootstrap(const u_ptr<ComputeThread>& thread, bool is_initiator) {
  impl_->initialize_or_bootstrap(thread, is_initiator);
}

void GpuIndex::reload_from_remote(const u_ptr<ComputeThread>& thread) { impl_->reload_from_remote(thread); }
void GpuIndex::clear_cache() { impl_->clear_cache(); }
vec<bool> GpuIndex::insert_batch(const vec<PendingInsert>& batch, const u_ptr<ComputeThread>& thread) {
  return impl_->insert_batch(batch, thread);
}
vec<vec<node_t>> GpuIndex::search_batch(const vec<PendingQuery>& batch, const u_ptr<ComputeThread>& thread) {
  return impl_->search_batch(batch, thread);
}
u32 GpuIndex::size() const { return impl_->size(); }
u32 GpuIndex::dim() const { return impl_->dim(); }
u32 GpuIndex::pad_dim() const { return impl_->pad_dim(); }
u32 GpuIndex::max_vectors() const { return impl_->max_vectors(); }
vec<element_t> GpuIndex::routing_centroid() const { return impl_->routing_centroid(); }

}  // namespace gpu
