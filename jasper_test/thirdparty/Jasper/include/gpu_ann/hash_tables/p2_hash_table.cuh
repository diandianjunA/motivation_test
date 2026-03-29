#pragma once

#include <cuda.h>
#include <cuda_runtime_api.h>

namespace gpu_ann {
namespace p2_hash_table {

constexpr uint32_t EMPTY_KEY = 0xFFFFFFFF;

// Simple hash function for power-of-2 table
__device__ __forceinline__ uint32_t hash_key(uint32_t key, uint32_t mask) {
  key ^= key >> 16;
  key *= 0x85ebca6b;
  key ^= key >> 13;
  key *= 0xc2b2ae35;
  key ^= key >> 16;
  return key & mask;
}

// Clear the hash table
__device__ void clear(uint32_t* addr, uint32_t size) {
  for (uint32_t i = threadIdx.x; i < size; i += blockDim.x) {
    addr[i] = EMPTY_KEY;
  }
}

// Dummy function for compatibility
__device__ void generate_hash(uint64_t* seeds) {
  // No-op for hash table
}

// Insert key in hash table
__device__ void insert(uint32_t* addr, uint64_t* seeds, uint32_t size, uint32_t key) {
  if (key == EMPTY_KEY) return;
  
  uint32_t mask = size - 1;
  uint32_t slot = hash_key(key, mask);
  
  // Linear probing
  while (true) {
    uint32_t old = atomicCAS(&addr[slot], EMPTY_KEY, key);
    if (old == EMPTY_KEY || old == key) break;
    slot = (slot + 1) & mask;
  }
}

// Query key in hash table
__device__ bool query(uint32_t* addr, uint64_t* seeds, uint32_t size, uint32_t key) {
  if (key == EMPTY_KEY) return false;
  
  uint32_t mask = size - 1;
  uint32_t slot = hash_key(key, mask);
  
  // Linear probing
  while (addr[slot] != EMPTY_KEY) {
    if (addr[slot] == key) return true;
    slot = (slot + 1) & mask;
  }
  return false;
}

// Query and insert - returns true if key already existed
__device__ bool query_and_insert_many(uint32_t* addr, uint64_t* seeds, uint32_t size, uint32_t key) {
  if (key == EMPTY_KEY) return false;
  
  uint32_t mask = size - 1;
  uint32_t slot = hash_key(key, mask);
  
  // Linear probing
  while (true) {
    uint32_t old = atomicCAS(&addr[slot], EMPTY_KEY, key);
    if (old == EMPTY_KEY) return false; // Inserted new
    if (old == key) return true;        // Already existed
    slot = (slot + 1) & mask;
  }
}

}  // namespace p2_hash_table
}  // namespace gpu_ann
