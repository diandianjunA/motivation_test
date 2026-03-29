#ifndef SET_CACHE_JASPER
#define SET_CACHE_JASPER

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cooperative_groups/scan.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

#include <cfloat>
#include <fstream>
#include <gallatin/allocators/alloc_utils.cuh>
#include <gallatin/data_structs/ds_utils.cuh>
#include <gpu_ann/_proj_metadata.cuh>
#include <gpu_ann/randomness.cuh>
#include <gpu_error/loads.cuh>
#include <iostream>
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

template <typename T, uint n>
struct cache_data_block {
  T data[n];

  __host__ __device__ T& get_data(int idx) { return data[idx]; }
  __host__ __device__ void set_data(int idx, const T& value) {

    gpu_error::store_release(&data[idx], value);
    //data[idx] = value;
  }
};

template <uint n>
struct cache_metadata_block_aos {
  uint32_t ids[n];
  float dists[n];

  template <uint tile_size>
  __device__ int query_id(uint32_t id, cg::thread_block_tile<tile_size>& tile) {
    int lane = tile.thread_rank();
    int found_idx = -1;

    for (int i = lane; i < n; i += tile_size) {
      if (ids[i] == id) found_idx = i;
    }

    return cg::reduce(tile, found_idx, cg::greater<int>());
  }

  __host__ __device__ void set_entry(int idx, uint32_t id, float dist) {
    ids[idx] = id;
    dists[idx] = dist;
  }

  __device__ float get_dist(int idx) { return dists[idx]; }
  __device__ float* get_dist_ptr(int idx) { return &dists[idx]; }

  __device__ bool try_lock_and_update(int idx, float old_dist, uint32_t new_id,
                                      float new_dist) {
    uint32_t* lock_ptr = reinterpret_cast<uint32_t*>(&dists[idx]);
    uint32_t new_dist_locked = __float_as_uint(new_dist) | 0x80000000;
    uint32_t old_val =
        atomicCAS(lock_ptr, __float_as_uint(old_dist), new_dist_locked);

    if (old_val == __float_as_uint(old_dist)) {
      ids[idx] = new_id;
      return true;
    }
    return false;
  }

  __device__ void unlock(int idx) {
    uint32_t* lock_ptr = reinterpret_cast<uint32_t*>(&dists[idx]);
    atomicAnd(lock_ptr, 0x7FFFFFFF);
  }
};

struct fused_cache_metadata {
  uint32_t id;
  float dist;
};

template <uint n>
struct cache_metadata_block_soa {
  fused_cache_metadata md[n];

  template <uint tile_size>
  __device__ int query_id(uint32_t id, cg::thread_block_tile<tile_size>& tile) {
    int lane = tile.thread_rank();
    int found_idx = -1;

    for (int i = lane; i < n; i += tile_size) {
      if (md[i].id == id) found_idx = i;
    }

    return cg::reduce(tile, found_idx, cg::greater<int>());
  }

  __host__ __device__ void set_entry(int idx, uint32_t id, float dist) {
    md[idx].id = id;
    md[idx].dist = dist;
  }

  __device__ float get_dist(int idx) { return md[idx].dist; }
  __device__ float* get_dist_ptr(int idx) { return &md[idx].dist; }

  __device__ bool try_lock_and_update(int idx, float old_dist, uint32_t new_id,
                                      float new_dist) {
    uint32_t* lock_ptr = reinterpret_cast<uint32_t*>(&md[idx].dist);
    uint32_t new_dist_locked = __float_as_uint(new_dist) | 0x80000000;
    uint32_t old_val =
        atomicCAS(lock_ptr, __float_as_uint(old_dist), new_dist_locked);

    if (old_val == __float_as_uint(old_dist)) {
      md[idx].id = new_id;
      return true;
    }
    return false;
  }

  __device__ void unlock(int idx) {
    uint32_t* lock_ptr = reinterpret_cast<uint32_t*>(&md[idx].dist);
    atomicAnd(lock_ptr, 0x7FFFFFFF);
  }
};

template <typename T, uint n_ways, typename MetadataBlock>
struct set_cache;

template <typename T, uint n_ways, typename MetadataBlock, uint tile_size>
__global__ void compete_kernel(uint32_t* keys, float* dists, int count,
                               set_cache<T, n_ways, MetadataBlock> cache);

template <typename T, uint n_ways, typename MetadataBlock, uint tile_size>
__global__ void write_winners_kernel(uint32_t* keys, float* dists, T* values,
                                     int count,
                                     set_cache<T, n_ways, MetadataBlock> cache);

template <typename T, uint n_ways, typename MetadataBlock, uint tile_size>
__global__ void insert_kernel(uint32_t* keys, float* dists, T* values,
                                     int count,
                                     set_cache<T, n_ways, MetadataBlock> cache);

template <typename T, uint n_ways,
          typename MetadataBlock = cache_metadata_block_soa<n_ways>>
struct set_cache {
  static_assert(n_ways == 8,
                "Cache must be 8-way associative to match tile size");

  MetadataBlock* metadata;
  cache_data_block<T, n_ways>* data;  // Storage blocks matching metadata
  uint32_t n_sets;
  bool owns_memory;  // Track if this instance owns the memory

  set_cache(uint32_t num_sets) : n_sets(num_sets), owns_memory(true) {
    cudaMalloc(&metadata, n_sets * sizeof(MetadataBlock));
    cudaMalloc(&data, n_sets * sizeof(cache_data_block<T, n_ways>));

    // Initialize metadata on host then copy to device
    MetadataBlock* host_metadata = new MetadataBlock[n_sets];
    for (uint32_t s = 0; s < n_sets; s++) {
      for (uint32_t w = 0; w < n_ways; w++) {
        host_metadata[s].set_entry(w, ~0U, FLT_MAX);
      }
    }
    cudaMemcpy(metadata, host_metadata, n_sets * sizeof(MetadataBlock),
               cudaMemcpyHostToDevice);
    delete[] host_metadata;
  }

  // Copy constructor - copies don't own memory
  __host__ __device__ set_cache(const set_cache& other)
      : metadata(other.metadata),
        data(other.data),
        n_sets(other.n_sets),
        owns_memory(false) {}

  ~set_cache() {
    if (owns_memory && metadata) {
      cudaFree(metadata);
      cudaFree(data);
      metadata = nullptr;
      data = nullptr;
    }
  }

  __device__ uint32_t hash(uint32_t key) {
    key ^= key >> 16;
    key *= 0x85ebca6b;
    key ^= key >> 13;
    key *= 0xc2b2ae35;
    key ^= key >> 16;
    return key % n_sets;
  }

  template <uint tile_size>
  __device__ T* query(uint32_t key, cg::thread_block_tile<tile_size>& tile) {
    uint32_t set_idx = hash(key);
    int way_idx = metadata[set_idx].template query_id<tile_size>(key, tile);
    return (way_idx >= 0) ? &data[set_idx].get_data(way_idx) : nullptr;
  }

  template <uint tile_size>
  void insert_batch(uint32_t* keys, float* dists, T* values, int count) {
    printf("insert_batch: count=%d, tile_size=%d\n", count, tile_size);
    dim3 block(256);
    dim3 grid((count * tile_size + block.x - 1) / block.x);
    printf("Grid: %d blocks, Block: %d threads\n", grid.x, block.x);

    // // Phase 1: Compete for cache slots
    // compete_kernel<T, n_ways, MetadataBlock, tile_size>
    //     <<<grid, block>>>(keys, dists, count, *this);

    // // Phase 2: Check if won and write data
    // write_winners_kernel<T, n_ways, MetadataBlock, tile_size>
    //     <<<grid, block>>>(keys, dists, values, count, *this);

    insert_kernel<T, n_ways, MetadataBlock, tile_size>
        <<<grid, block>>>(keys, dists, values, count, *this);


  }

  template <uint tile_size>
  __device__ void insert(uint32_t key, float dist, const T& value,
                         cg::thread_block_tile<tile_size>& tile) {
    uint32_t set_idx = hash(key);
    auto& set_meta = metadata[set_idx];

    while (true) {
      int lane = tile.thread_rank();

      // Check if key already exists
      int existing_idx = set_meta.template query_id<tile_size>(key, tile);
      if (existing_idx >= 0) return;  // Hit found, exit

      // Find index with largest distance > current distance using coherent
      // reads
      float max_dist = -1.0f;
      int local_idx = -1;

      for (int i = lane; i < n_ways; i += tile_size) {
        float cached_dist;
        asm volatile("ld.global.acquire.f32 %0, [%1];"
                     : "=f"(cached_dist)
                     : "l"(set_meta.get_dist_ptr(i)));

        cached_dist = __abs(cached_dist);

        if (cached_dist > dist && cached_dist > max_dist) {
          max_dist = cached_dist;
          local_idx = i;
        }
      }

      struct {
        float dist;
        int idx;
      } local_max = {max_dist, local_idx};
      auto max_result = cg::reduce(tile, local_max, [](auto a, auto b) {
        return (a.dist > b.dist) ? a : b;
      });

      if (max_result.idx < 0) return;  // No smaller distances, exit

      if (tile.thread_rank() == 0) {
        if (set_meta.try_lock_and_update(max_result.idx, max_result.dist, key,
                                         dist)) {
          data[set_idx].set_data(max_result.idx, value);
          __threadfence();
          set_meta.unlock(max_result.idx);
          return;  // Successfully inserted, exit
        }
      }

      tile.sync();  // Sync before retry
    }
  }

  template <uint tile_size>
  __device__ void compete(uint32_t key, float dist,
                          cg::thread_block_tile<tile_size>& tile) {
    static_assert(tile_size == n_ways,
                  "Tile size must match cache associativity");

    uint32_t set_idx = hash(key);
    auto& set_meta = metadata[set_idx];
    int lane = tile.thread_rank();

    while (true) {
      // Each thread loads its corresponding cache entry
      float cached_dist;
      asm volatile("ld.global.acquire.gpu.f32 %0, [%1];"
                   : "=f"(cached_dist)
                   : "l"(set_meta.get_dist_ptr(lane)));

      // Skip locked entries (MSB set)
      bool can_replace = (cached_dist >= 0) && (cached_dist > dist);

      // Ballot to find threads that can replace their entry
      uint64_t ballot = tile.ballot(can_replace);

      if (ballot == 0) {
        // No one can replace, exit
        return;
      }

      // Try CAS in order of lowest lane ID first
      while (ballot != 0) {
        int first_lane = __ffsll(ballot) - 1;  // Find first set bit
        bool success = false;

        if (lane == first_lane) {
          success = set_meta.try_lock_and_update(lane, cached_dist, key, dist);
        }

        // Ballot on success - if anyone succeeded, everyone exits
        if (tile.ballot(success)) {
          return;
        }

        ballot &= ~(1ULL << first_lane);  // Clear this bit and try next
      }

      // All CAS attempts failed, reload and try again
    }
  }

  template <uint tile_size>
  __device__ bool check_winner(uint32_t key,
                               cg::thread_block_tile<tile_size>& tile) {
    uint32_t set_idx = hash(key);
    return metadata[set_idx].template query_id<tile_size>(key, tile) >= 0;
  }

  template <uint tile_size>
  __device__ void write_data(uint32_t key, const T& value,
                             cg::thread_block_tile<tile_size>& tile) {
    uint32_t set_idx = hash(key);
    int way_idx = metadata[set_idx].template query_id<tile_size>(key, tile);
    if (way_idx >= 0) {
      data[set_idx].set_data(way_idx, value);
      metadata[set_idx].unlock(way_idx);
    }
  }
};

template <typename T, uint n_ways, typename MetadataBlock, uint tile_size>
__global__ void compete_kernel(uint32_t* keys, float* dists, int count,
                               set_cache<T, n_ways, MetadataBlock> cache) {
  cg::thread_block thread_block = cg::this_thread_block();
  cg::thread_block_tile<tile_size> my_tile =
      cg::tiled_partition<tile_size>(thread_block);
  uint64_t tid = gallatin::utils::get_tile_tid(my_tile);

  if (tid >= count) return;

  cache.template compete<tile_size>(keys[tid], dists[tid], my_tile);
}

template <typename T, uint n_ways, typename MetadataBlock, uint tile_size>
__global__ void insert_kernel(
    uint32_t* keys, float* dists, T* values, int count,
    set_cache<T, n_ways, MetadataBlock> cache) {
  cg::thread_block thread_block = cg::this_thread_block();
  cg::thread_block_tile<tile_size> my_tile =
      cg::tiled_partition<tile_size>(thread_block);
  uint64_t tid = gallatin::utils::get_tile_tid(my_tile);

  if (tid >= count) return;

  cache.template insert<tile_size>(keys[tid], dists[tid], values[tid], my_tile);
}

template <typename T, uint n_ways, typename MetadataBlock, uint tile_size>
__global__ void write_winners_kernel(
    uint32_t* keys, float* dists, T* values, int count,
    set_cache<T, n_ways, MetadataBlock> cache) {
  cg::thread_block thread_block = cg::this_thread_block();
  cg::thread_block_tile<tile_size> my_tile =
      cg::tiled_partition<tile_size>(thread_block);
  uint64_t tid = gallatin::utils::get_tile_tid(my_tile);

  if (tid >= count) return;

  if (cache.template check_winner<tile_size>(keys[tid], my_tile)) {
    cache.template write_data<tile_size>(keys[tid], values[tid], my_tile);
  }
}

}  // namespace gpu_ann

#endif  // GPU_BLOCK_