#pragma once

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cub/cub.cuh>

#include <iostream>
#include <vector>
#include <gallatin/allocators/alloc_utils.cuh>
#include <gallatin/data_structs/ds_utils.cuh>
#include <gpu_ann/edge_list.cuh>
#include <gpu_ann/edge_sorting.cuh>
#include <gpu_ann/vector.cuh>
#include <gpu_ann/edge_list.cuh>
#include <gpu_ann/beam_search.cuh>

#include "assert.h"
#include "stdio.h"

namespace gpu_ann {

template <typename INDEX_T, typename DISTANCE_T>
__device__ thrust::pair<uint32_t, bool> choose_edge_to_prune(
  thrust::pair<INDEX_T, DISTANCE_T> *entries,
  uint32_t entry_count,
  uint32_t offset
){
  constexpr INDEX_T INVALID_INDEX = std::numeric_limits<INDEX_T>::max();
  __shared__ uint32_t first_index;
  if (threadIdx.x == 0) first_index = ~0u;
  __syncthreads();

  uint32_t is_avaliable;
  if (threadIdx.x + offset < entry_count) {
    if (entries[offset + threadIdx.x].first != INVALID_INDEX) {
      is_avaliable = 1;
    } else {
      is_avaliable = 0;
    }
  } else {
    is_avaliable = 0;
  }

  unsigned mask = __ballot_sync(0xffffffff, is_avaliable);
  if (mask != 0) {
    int lane_id = threadIdx.x % warpSize;
    int first_lane = __ffs(mask) - 1;
    if (lane_id == first_lane) {
      atomicMin(&first_index, threadIdx.x+offset); 
    }
  }
  __syncthreads();
  return {first_index, first_index != ~0u};
}

template <typename INDEX_T,
          typename DATA_T,
          uint16_t DATA_DIM,
          typename DISTANCE_T,
          typename EDGE_LIST_T,
          uint32_t R,
          template <typename, typename, uint, uint> class distance_functor,
          uint32_t BLOCK_SIZE,
          uint32_t MAX_PRUNE_SIZE=256>
__global__ void prune_and_add_single(
  EDGE_LIST_T *graph,
  uint8_t *edge_count,
  data_vector<DATA_T, DATA_DIM> *data_vectors,
  uint64_t query_offset,
  uint64_t n_query_vectors,
  thrust::pair<INDEX_T, DISTANCE_T> *visited_lists,
  uint32_t *visited_counts,
  float alpha
) {
  using ENTRY_T = thrust::pair<INDEX_T, DISTANCE_T>;

  // each block responsible for one target to prune
  uint32_t query_idx = query_offset + blockIdx.x;
  data_vector<DATA_T, DATA_DIM> query_vec = data_vectors[query_idx];

  // get the visited list + count and existing edges + count
  uint32_t visited_count = visited_counts[blockIdx.x];
  uint32_t offset = 1024 * blockIdx.x;
  ENTRY_T *visited_list = visited_lists+offset;
  if (visited_count == 0) return;

  uint8_t current_count = edge_count[query_idx];
  EDGE_LIST_T current_edges = graph[query_idx];

  // allocate shared memory
  assert(visited_count + current_count <= MAX_PRUNE_SIZE);
  __shared__ ENTRY_T prune_list[MAX_PRUNE_SIZE];

  // copy visited list to shared memory
  for (uint i=threadIdx.x; i<visited_count; i+=blockDim.x) {
    prune_list[i] = visited_list[i];
  }

  uint32_t total_entries = visited_count + current_count;

  // If there are existing edges, append and sort by distance
  if (current_count > 0) {
    // append the current edges to the list
    for (uint i=threadIdx.x; i<current_count; i++) {
      prune_list[visited_count+i] = {current_edges.edges[i], 0};
    }
  }
  __syncthreads();

  // DEBUG: just print all the stuff in prune_list
  // if (threadIdx.x == 0 && blockIdx.x == 0) {
  //   printf("PRUNE_LIST: visited=%u, current=%u ", visited_count, current_count);
  //   for (uint i=0; i<total_entries; ++i) {
  //     printf("[%u,%f] ", prune_list[i].first, prune_list[i].second);
  //   }
  //   printf("\n");
  // }
  // __syncthreads();

  populate_distances<DATA_T, DATA_DIM, INDEX_T, DISTANCE_T, 4, distance_functor>(
      query_vec, data_vectors, prune_list, &total_entries, 0);

  // while edge list size > R, choose a newer index to prune
  // until we have edge list size <= R
  __shared__ uint32_t cur_index;         // next index to prune
  __shared__ uint32_t cur_total_entries; // entries size
  if (threadIdx.x == 0) {
    cur_index = 0;
    cur_total_entries = total_entries;
  }
  __syncthreads();

  using distance_type = distance_functor<DATA_T, DATA_T, DATA_DIM, 1>;
  auto thread_block = cg::this_thread_block();
  cg::thread_block_tile<1> my_tile = cg::tiled_partition<1>(thread_block);
  constexpr INDEX_T INVALID_INDEX = std::numeric_limits<INDEX_T>::max();

  while (true) {
    // get the next valid p* to prune
    auto [ps_idx, is_valid] = choose_edge_to_prune(prune_list, total_entries, cur_index);

    if (!is_valid) break;
    if (threadIdx.x == 0) cur_index = ps_idx + 1;
    INDEX_T ps = prune_list[ps_idx].first;
    data_vector<DATA_T, DATA_DIM> ps_vec = data_vectors[ps];
    __syncthreads();

    for (uint i=threadIdx.x; i<total_entries; i+=blockDim.x) {
      if (i == ps_idx || prune_list[i].first == INVALID_INDEX) continue;
      // get the entry distance d(p*-p')
      auto dest = prune_list[i].first;
      auto dist = distance_type::distance(ps_vec, data_vectors[dest], my_tile);

      // if alpha * d(p*,p') <= d(p, p')
      // reduce cur_total_entries by one and replace the entry with invalid.
      if (alpha * dist <= prune_list[i].second) {
        
          // printf("KERNEL: block=%u, pruned=%u, alpha*dist=%f, prune_list[i]=[%u,%f]\n", 
          //   blockIdx.x, 
          //   i,
          //   alpha * dist,
          //   prune_list[i].first,
          //   prune_list[i].second
          // );
        
        
        prune_list[i].first = INVALID_INDEX;
        atomicSub(&cur_total_entries, 1);
      }
    }
    __syncthreads();
    if (cur_index >= total_entries) break;
    if (cur_total_entries <= R) break;
  }

  // set the pruned edges to graph using prefix sum
  using BlockScanT = cub::BlockScan<uint32_t, BLOCK_SIZE>;
  constexpr uint32_t ELEMENTS_PER_THREAD = (MAX_PRUNE_SIZE-1) / BLOCK_SIZE + 1;
  __shared__ typename BlockScanT::TempStorage temp_storage;

  ENTRY_T this_entry[ELEMENTS_PER_THREAD];
  bool is_valid[ELEMENTS_PER_THREAD];
  uint32_t thread_data[ELEMENTS_PER_THREAD];

  #pragma unroll
  for (uint i=0; i<ELEMENTS_PER_THREAD; i++) {
    uint32_t element_id = threadIdx.x * ELEMENTS_PER_THREAD + i;
    this_entry[i] = prune_list[element_id];
    if (element_id < total_entries && this_entry[i].first != INVALID_INDEX) {
      is_valid[i] = true;
      thread_data[i] = 1;
    } else {
      is_valid[i] = false;
      thread_data[i] = 0;
    }
  }
  __syncthreads();

  BlockScanT(temp_storage).ExclusiveSum(thread_data, thread_data);

  #pragma unroll
  for (uint i=0; i<ELEMENTS_PER_THREAD; i++) {
    if (is_valid[i]) {
      visited_list[thread_data[i]] = this_entry[i];
      graph[query_idx].edges[i] = this_entry[i].first;
    }
  }
  if (threadIdx.x == blockDim.x - 1) {
    visited_counts[blockIdx.x] = thread_data[ELEMENTS_PER_THREAD-1];
    edge_count[query_idx] = thread_data[ELEMENTS_PER_THREAD-1];
    //if (blockIdx.x == 0) printf("KERNEL: final length is %u\n", thread_data[ELEMENTS_PER_THREAD-1]);
  }
  __syncthreads();
}

}