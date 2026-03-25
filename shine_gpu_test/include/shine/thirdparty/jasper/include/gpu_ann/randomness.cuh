#ifndef RANDOM_GEN
#define RANDOM_GEN

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cooperative_groups/scan.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>

#include <fstream>
#include <gallatin/allocators/alloc_utils.cuh>
#include <gallatin/data_structs/ds_utils.cuh>
#include <gpu_ann/_proj_metadata.cuh>
#include <iostream>
#include <random>
#include <type_traits>
#include <vector>

#include "assert.h"
#include "stdio.h"

namespace cg = cooperative_groups;

// helper_macro
// define macros
// #define MAX_VALUE(nbits) ((1ULL << (nbits)) - 1)
// #define BITMASK(nbits) ((nbits) == 64 ? 0xffffffffffffffff :
// MAX_VALUE(nbits))

// #define SET_BIT_MASK(index) ((1ULL << index))

namespace gpu_ann {

// fill a host buffer with random data
template <typename T>
__host__ T *random_data_host(uint64_t nitems) {
  T *vals = gallatin::utils::get_host_version<T>(nitems);
  std::mt19937_64 rng(std::random_device{}());

  for (uint64_t i = 0; i < nitems; ++i) {
    if constexpr (std::is_integral_v<T> || std::is_enum_v<T>) {
      vals[i] = static_cast<T>(rng());
    } else if constexpr (std::is_floating_point_v<T>) {
      vals[i] = static_cast<T>(std::generate_canonical<double, 64>(rng));
    } else {
      auto *bytes = reinterpret_cast<unsigned char *>(&vals[i]);
      for (size_t byte = 0; byte < sizeof(T); ++byte) {
        bytes[byte] = static_cast<unsigned char>(rng() & 0xFFu);
      }
    }
  }

  return vals;
}

// generate on host with openSSL RAND_BYTES then copy.
// returns cudaMalloc pointer, must be released with cudaFree.
template <typename T>
__host__ T *random_data_device(uint64_t nitems) {
  T *host_data = random_data_host<T>(nitems);

  T *device_ptr = gallatin::utils::move_to_device<T>(host_data, nitems);

  return device_ptr;
}

// helper to generate ids[x] = x;
//  used for shuffle sampling kernel.
static __global__ void populate_with_id(uint32_t *ids, uint32_t n_keys) {
  uint64_t tid = gallatin::utils::get_tid();

  if (tid >= n_keys) return;

  ids[tid] = tid;
}

// generates samples based on shuffled ID
template <typename T>
static __global__ void sampling_kernel(T *host_data, T *samples, uint32_t *ids,
                                       uint32_t n_to_sample) {
  uint64_t tid = gallatin::utils::get_tid();

  if (tid >= n_to_sample) return;

  samples[tid] = host_data[ids[tid]];

  return;
}

// returns an object in device memory containing n_to_sample sampled objects of
// type T
//  uniformly randomly  chosen from the target set.
// this is done with a thrust sort over random data.
template <typename T>
__host__ T *sample(T *host_data, uint64_t n_keys, uint64_t n_to_sample) {
  uint32_t *weightings = random_data_device<uint32_t>(n_keys);

  uint32_t *ids = gallatin::utils::get_device_version<uint32_t>(n_keys);

  populate_with_id<<<(n_keys - 1) / 512 + 1, 512>>>(ids, n_keys);

  cudaDeviceSynchronize();

  thrust::sort_by_key(thrust::device, weightings, weightings + n_keys, ids);

  cudaFree(weightings);

  // ids are now sorted.
  // use the first n_to_sample ids to copy.

  T *sampled = gallatin::utils::get_device_version<T>(n_to_sample);

  sampling_kernel<T><<<(n_to_sample - 1) / 256 + 1, 256>>>(host_data, sampled,
                                                           ids, n_to_sample);

  cudaDeviceSynchronize();

  cudaFree(ids);

  return sampled;
}

}  // namespace gpu_ann

#endif  // GPU_BLOCK_
