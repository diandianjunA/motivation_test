#pragma once

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cooperative_groups/scan.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

#include <fstream>
#include <gallatin/allocators/alloc_utils.cuh>
#include <gallatin/data_structs/ds_utils.cuh>
#include <gpu_ann/_proj_metadata.cuh>
#include <iostream>
#include <vector>

#include "assert.h"
#include "stdio.h"

namespace cg = cooperative_groups;

namespace gpu_ann {
namespace array {

// A small array resides in shared memory.
//
// we bound its size to number of thread in the block (512)
// so that both insertion and querying are constant time.

constexpr uint32_t default_key = ~static_cast<uint32_t>(0);

__device__ void init(uint32_t* src, size_t n_elements) {
  assert(n_elements == 512);
  for (uint i = threadIdx.x; i < n_elements; i += blockDim.x) {
    src[i] = default_key;
  }
}

__device__ uint32_t insert(uint32_t* src, size_t n_elements, const uint32_t key) {
  // We will never use the default key (~4 billion)
  // since we use this array to store vector index.
  assert(key != default_key);
  assert(n_elements == blockDim.x);

  uint32_t previous = 1;
  if (threadIdx.x != 0) {
    previous = src[threadIdx.x-1];
  }

  uint32_t my_slot = src[threadIdx.x];

  if (previous != default_key && my_slot == default_key) {
    src[threadIdx.x] = key;
  }

  return 1;
}

__device__ uint32_t search(uint32_t* src, size_t n_elements, const uint32_t key) {
  assert(key != default_key);
  assert(n_elements == blockDim.x);

  return static_cast<uint32_t>(key == src[threadIdx.x]);
}

}
}