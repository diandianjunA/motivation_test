#pragma once

#include <cuda.h>
#include <cuda_runtime_api.h>

namespace gpu_ann {
namespace bool_bloom_filter {

// number of hash functions
constexpr uint32_t N_HASHES = 3;

// Multiply-Shift hash: h(x) = (a * x) >> (w - r)
__device__ __forceinline__ uint32_t multiply_shift_hash(uint32_t key,
                                                        uint64_t seeds,
                                                        uint32_t bits = 32) {
  uint64_t product = (uint64_t)seeds * (uint64_t)key;
  return (uint32_t)(product >> (64 - bits));
}

__device__ uint64_t splitmix64(uint64_t& state) {
  uint64_t z = (state += 0x9E3779B97F4A7C15ull);
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  return z ^ (z >> 31);
}

// Clear the bloom filter
__device__ void clear(bool* addr, uint32_t size) {
  for (uint32_t i = threadIdx.x; i < size; i += blockDim.x) {
    addr[i] = false;
  }
}

// Generate hash seeds
__device__ void generate_hash(uint64_t* seeds) {
  if (threadIdx.x == 0) {
    uint64_t state = clock64() ^ ((uint64_t)blockIdx.x << 32);
#pragma unroll
    for (uint32_t i = 0; i < N_HASHES; i++) {
      seeds[i] = splitmix64(state);
    }
  }
}

// Insert key - no atomics needed with smem sync
__device__ void insert(bool* addr, uint64_t* seeds, uint32_t size, uint32_t key) {
#pragma unroll
  for (uint32_t i = 0; i < N_HASHES; i++) {
    uint32_t slot = multiply_shift_hash(key, seeds[i], 32) % size;
    addr[slot] = true;
  }
}

// Query key
__device__ bool query(bool* addr, uint64_t* seeds, uint32_t size, uint32_t key) {
#pragma unroll
  for (uint32_t i = 0; i < N_HASHES; i++) {
    uint32_t slot = multiply_shift_hash(key, seeds[i], 32) % size;
    if (!addr[slot]) return false;
  }
  return true;
}

// Query and insert - returns true if key already existed
__device__ bool query_and_insert_many(bool* addr, uint64_t* seeds, uint32_t size, uint32_t key) {
  bool existed = true;
#pragma unroll
  for (uint32_t i = 0; i < N_HASHES; i++) {
    uint32_t slot = multiply_shift_hash(key, seeds[i], 32) % size;
    if (!addr[slot]) {
      existed = false;
      addr[slot] = true;
    }
  }
  return existed;
}

}  // namespace bool_bloom_filter
}  // namespace gpu_ann
