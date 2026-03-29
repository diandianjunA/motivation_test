#ifndef EDGE_LIST
#define EDGE_LIST

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cooperative_groups/scan.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <thrust/execution_policy.h>

#include <fstream>
#include <gallatin/allocators/alloc_utils.cuh>
#include <gallatin/data_structs/ds_utils.cuh>
#include <gpu_ann/_proj_metadata.cuh>
#include <gpu_error/log.cuh>
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

// 0 is default for edge list - no one points to the source.
template <typename data_type, uint32_t R>
struct edge_list {
  data_type edges[R];

  void print() const {
    std::cout << "Edges: ";
    for (uint32_t i = 0; i < R; ++i) {
      std::cout << edges[i] << " ";
    }
    std::cout << std::endl;
  }
};

// in-memory edge list
//  allows for distances to be computed
//  and then merges into priority queue
template <typename data_type, uint32_t R, typename dist_type>
struct smem_edge_list {
  edge_list<data_type, R> loaded_edge_list;

  dist_type distances[R];

  __device__ data_type& operator[](uint64_t id) {
    return loaded_edge_list.edges[R];
  }

  __device__ void set_dist(uint loc, dist_type ext_dist) {
    distances[loc] = ext_dist;
  }

  __device__ void swap(uint left, uint right) {
    dist_type temp_dist;

    temp_dist = distances[left];
    distances[left] = distances[right];
    distances[right] = temp_dist;

    data_type temp_edge;
    temp_edge = loaded_edge_list.edges[left];
    loaded_edge_list.edges[left] = loaded_edge_list.edges[right];
    loaded_edge_list.edges[right] = temp_edge;
  }
  // sort the first "n_keys" keys
  template <uint32_t tile_size>
  __device__ void sort(cg::thread_block_tile<tile_size>& my_tile,
                       uint32_t n_keys) {
    int tid = my_tile.thread_rank();

    for (int p = 0; p < n_keys; ++p) {
      int phase = p % 2;
      for (int i = tid; i < n_keys / 2; i += tile_size) {
        int ix = 2 * i + phase;
        int ixj = ix + 1;
        if (ixj < n_keys) {
          if (distances[ix] > distances[ixj]) {
            swap(ix, ixj);
          }
        }
      }
      my_tile.sync();
    }
  }

  template <typename pair_type>
  __device__ pair_type get_pair(uint address) const {
    pair_type return_pair(loaded_edge_list.edges[address], distances[address]);
    return return_pair;
  }
};

}  // namespace gpu_ann

#endif  // GPU_BLOCK_