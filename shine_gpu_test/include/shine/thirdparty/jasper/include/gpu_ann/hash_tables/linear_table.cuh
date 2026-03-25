#ifndef LINEAR_TABLE
#define LINEAR_TABLE

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

// helper_macro
// define macros
// #define MAX_VALUE(nbits) ((1ULL << (nbits)) - 1)
// #define BITMASK(nbits) ((nbits) == 64 ? 0xffffffffffffffff :
// MAX_VALUE(nbits))

// #define SET_BIT_MASK(index) ((1ULL << index))

namespace gpu_ann {

template <typename Key, Key defaultKey, uint tile_size>
struct tile_bucket {
  volatile Key slots[tile_size];

  __device__ void init(cg::thread_block_tile<tile_size>& my_tile) {
    slots[my_tile.thread_rank()] = defaultKey;
  }

  __device__ bool query(cg::thread_block_tile<tile_size>& my_tile, Key queryKey,
                        bool& early_stop) {
    bool ballot = false;

    bool early_stop_ballot = false;

    Key read_key = slots[my_tile.thread_rank()];

    if (read_key == queryKey) {
      ballot = true;
    }

    if (read_key == defaultKey) {
      early_stop_ballot = true;
    }

    early_stop = my_tile.ballot(early_stop_ballot);

    bool result = my_tile.ballot(ballot);

    return result;
  }

  __device__ bool insert(cg::thread_block_tile<tile_size>& my_tile,
                         Key insertKey) {
    bool found = (slots[my_tile.thread_rank()] == defaultKey);

    int leader = __ffs(my_tile.ballot(found)) - 1;

    if (leader == -1) return false;

    if (my_tile.thread_rank() == leader) {
      slots[my_tile.thread_rank()] = insertKey;
      __threadfence();
    }

    my_tile.sync();
    __threadfence();
    my_tile.sync();

    return true;
  }
};

template <typename Key, Key defaultKey, uint n_keys, uint32_t tile_size>
struct linear_table {
  using bucket_type = tile_bucket<Key, defaultKey, tile_size>;

  static constexpr uint n_buckets = (n_keys - 1) / tile_size + 1;

  bucket_type buckets[n_buckets];

  __device__ void init(cg::thread_block_tile<tile_size>& my_tile) {
    for (int i = 0; i < n_buckets; i++) {
      buckets[i].init(my_tile);
    }
    __threadfence();
    my_tile.sync();
  }

  __device__ uint32_t hash(Key hashKey) {
    // simple mult hash from Knuth
    return hashKey * (hashKey + 3);
  }

  __device__ bool insert(cg::thread_block_tile<tile_size>& my_tile,
                         Key insertKey) {
    uint32_t hash_val = hash(insertKey);

    for (uint i = 0; i < n_buckets; i++) {
      uint32_t bucket_id = (hash_val + i) % n_buckets;

      if (buckets[bucket_id].insert(my_tile, insertKey)) return true;
    }

    return false;
  }

  __device__ bool query(cg::thread_block_tile<tile_size>& my_tile,
                        Key queryKey) {
    __threadfence();
    my_tile.sync();

    bool early_exit = false;

    uint32_t hash_val = hash(queryKey);

    for (uint i = 0; i < n_buckets; i++) {
      uint32_t bucket_id = (hash_val + i) % n_buckets;

      if (buckets[bucket_id].query(my_tile, queryKey, early_exit)) return true;

      if (early_exit) return false;
    }

    return false;
  }

  __device__ Key access(uint index) {
    uint32_t bucket_id = index / tile_size;

    uint32_t internal_id = index % tile_size;

    return buckets[bucket_id].slots[internal_id];
  }
};

}  // namespace gpu_ann

#endif  // GPU_BLOCK_