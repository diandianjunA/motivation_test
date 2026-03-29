#pragma once

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cooperative_groups/scan.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <thrust/pair.h>

#include <cub/cub.cuh>
#include <fstream>
#include <gallatin/allocators/alloc_utils.cuh>
#include <gallatin/data_structs/ds_utils.cuh>
#include <gpu_ann/_proj_metadata.cuh>
#include <gpu_ann/beam_search.cuh>
#include <gpu_ann/edge_list.cuh>
#include <gpu_ann/edge_sorting.cuh>
#include <gpu_ann/hash_tables/hashmap.cuh>
#include <gpu_ann/rabitq_quantizer.cuh>
#include <gpu_ann/vector.cuh>
#include <gpu_error/log.cuh>
#include <iostream>
#include <vector>

#include "assert.h"
#include "stdio.h"

namespace cg = cooperative_groups;

namespace gpu_ann {
namespace rabitq {

// distance calculation using rabitq
template <typename DATA_T, uint16_t DATA_DIM, typename INDEX_T,
          typename DISTANCE_T, uint32_t tile_size, uint16_t SIZE_PER_DIM = 1>
__device__ void populate_distances_rabitq(
    // Query vector
    const float * __restrict__ query_vec, const RabitqQueryFactor & query_factor,
    // Data vector
    const RabitqDataVec<SIZE_PER_DIM, DATA_DIM> * __restrict__ rabitq_vectors,
    // out
    ENTRY_T *result_buffer,
    uint32_t *result_buffer_count, const uint32_t & offset) {
  auto thread_block = cg::this_thread_block();
  cg::thread_block_tile<tile_size> my_tile =
      cg::tiled_partition<tile_size>(thread_block);
  uint64_t tid = my_tile.meta_group_rank();
  uint64_t tile_offset = blockDim.x / tile_size;
  constexpr INDEX_T INVALID_INDEX = std::numeric_limits<INDEX_T>::max();

  uint32_t count = result_buffer_count[0];
  
  // Process in batches to reduce divergence
  for (unsigned base = offset; base < count; base += tile_offset) {
    unsigned i = base + tid;
    if (i < count) {
      auto dest = get_index(result_buffer[i]);
      if (dest != INVALID_INDEX) {
        // L2 distance
        const RabitqDataVec<SIZE_PER_DIM, DATA_DIM> *data_vec = rabitq_vectors + dest;

        // compute <q,o>
        float dist = one_distance_device(query_factor, query_vec, data_vec, my_tile);

        if (my_tile.thread_rank() == 0) {
          result_buffer[i] = set_distance(result_buffer[i], dist);
        }
      }
    }
  }
  __syncthreads();
}

// alternate way to force registers down
// __launch_bounds__(BLOCK_SIZE, 2)

// main search kernel using rabitq

template <typename INDEX_T, typename DATA_T, uint16_t DATA_DIM,
          typename DISTANCE_T, typename EDGE_LIST_T, uint32_t BLOCK_SIZE,
          uint32_t MAX_SEARCH_WIDTH, bool GET_VISITED, uint32_t TILE_SIZE = 4,
          uint16_t SIZE_PER_DIM = 1>
// __maxnreg__(40)
__global__ void beam_search_kernel(
    const EDGE_LIST_T * __restrict__ graph, const uint8_t * __restrict__ edge_count,
    thrust::pair<INDEX_T, DISTANCE_T> *frontier_results,
    thrust::pair<INDEX_T, DISTANCE_T> *visited_results,
    uint32_t *visited_counts,
    const RabitqDataVec<SIZE_PER_DIM, DATA_DIM> * __restrict__ rabitq_vectors,
    const uint64_t n_data_vectors,
    const float * __restrict__ rabitq_query_vectors, const RabitqQueryFactor * __restrict__ rabitq_query_factors,
    const uint64_t n_query_vectors, const INDEX_T medoid, const uint32_t k, const uint32_t beam_width,
    const float cut, const uint32_t limit, const uint32_t hashmap_bitlen) {
  // get the query vector for this block
  const auto query_id = blockIdx.x;
  const auto query_vec = rabitq_query_vectors + query_id * DATA_DIM;
  const auto query_factor = rabitq_query_factors[query_id];
  uint32_t visited_counter = 0;

  assert(beam_width + 64 <= MAX_SEARCH_WIDTH);

  // allocate shared memory
  const uint32_t result_buffer_size = beam_width + 64;
  extern __shared__ __align__(128) uint32_t smem[];
  auto *__restrict__ result_buffer = reinterpret_cast<ENTRY_T *>(smem);
  auto *__restrict__ result_buffer_count = reinterpret_cast<uint32_t *>(result_buffer + result_buffer_size);

  // cub temporary workspace
  constexpr uint32_t ELEMENTS_PER_THREAD = (MAX_SEARCH_WIDTH - 1) / BLOCK_SIZE + 1;
  using BlockMergeSortT = cub::BlockMergeSort<ENTRY_T, BLOCK_SIZE, ELEMENTS_PER_THREAD>;
  using BlockScanT = cub::BlockScan<uint32_t, BLOCK_SIZE>;
  union TempStorage {
    typename cub::BlockMergeSort<ENTRY_T, BLOCK_SIZE, ELEMENTS_PER_THREAD>::TempStorage sort_storage;
    typename cub::BlockScan<uint32_t, BLOCK_SIZE>::TempStorage scan_storage;
  };
  __shared__ TempStorage temp_storage;

  // load query vector to shared memory
  __shared__ float __align__(16) smem_query_vec[DATA_DIM];
  for (unsigned i = threadIdx.x; i < DATA_DIM; i += blockDim.x) {
    smem_query_vec[i] = query_vec[i];
  }

  // initialize shared memory to default (invalid) key
  for (unsigned i = threadIdx.x; i < result_buffer_size; i += blockDim.x) {
    result_buffer[i] = empty_entry();
  }

  // populate frontier in shared memory
  if (threadIdx.x == 0) {
    result_buffer[0] = set_index(empty_entry(), medoid);
    result_buffer_count[0] = 1;
  }
  __syncthreads();

  populate_distances_rabitq<DATA_T, DATA_DIM, INDEX_T, DISTANCE_T, TILE_SIZE,
                            SIZE_PER_DIM>(
      smem_query_vec, query_factor, rabitq_vectors,
      result_buffer, result_buffer_count, 0);

  // loop
  uint32_t offset;
  uint32_t loop_count = 0;
  while (loop_count <= limit) {
    loop_count += 1;
    // choose a new frontier to explore
    auto [frontierIdx, found] =
        choose_new_frontier<INDEX_T, DISTANCE_T, BLOCK_SIZE, MAX_SEARCH_WIDTH>(
          result_buffer, result_buffer_count);
    if (!found) {
      break;  // we converged
    };

    __syncthreads();

    if constexpr (BLOCK_SIZE > 33) {
      if (threadIdx.x == 33) {
        result_buffer[frontierIdx] = set_visited(result_buffer[frontierIdx]);
      }
    } else {
      if (threadIdx.x == 1) {
        result_buffer[frontierIdx] = set_visited(result_buffer[frontierIdx]);
      }
    }

    // Add frontier to visited list
    // we do not add the first point
    if (threadIdx.x == 0 && GET_VISITED) {
      visited_results[query_id * 1024 + visited_counter].first =
          get_index(result_buffer[frontierIdx]);
      visited_results[query_id * 1024 + visited_counter].second =
          get_distance(result_buffer[frontierIdx]);
      visited_counter += 1;
      assert(visited_counter < 1024);
    }

    // record offset so that we know where to start calculating distances
    // (all the previous ones are calculated)
    offset = result_buffer_count[0];
    __syncthreads();

    // Add frontier's neighbor to results
    add_frontier_out<INDEX_T, DISTANCE_T>(graph, edge_count, result_buffer, result_buffer_count,
                     get_index(result_buffer[frontierIdx]), k, beam_width);
    __syncthreads();

    // populate distance
    populate_distances_rabitq<DATA_T, DATA_DIM, INDEX_T, DISTANCE_T, TILE_SIZE,
                              SIZE_PER_DIM>(
        smem_query_vec, query_factor, rabitq_vectors,
        result_buffer, result_buffer_count, offset);

    // sort the result buffer
    merge_sort<INDEX_T, DISTANCE_T, BLOCK_SIZE, MAX_SEARCH_WIDTH, BlockMergeSortT>(
        result_buffer, result_buffer_count, temp_storage.sort_storage);
    dedup_results<INDEX_T, DISTANCE_T, BLOCK_SIZE, MAX_SEARCH_WIDTH, BlockScanT>(
        result_buffer, result_buffer_count, temp_storage.scan_storage);
    clip_k(result_buffer_count, beam_width);
    filter_frontier<INDEX_T, DISTANCE_T>(result_buffer, result_buffer_count, k, cut);
  }

  // copy to the external result
  for (uint i=threadIdx.x; i<k; i += blockDim.x) {
    frontier_results[query_id * k + i].first =
        get_index(result_buffer[i]);
    frontier_results[query_id * k + i].second =
        get_distance(result_buffer[i]);
  }
  if (threadIdx.x == 0 && GET_VISITED) {
    visited_counts[query_id] = visited_counter;
  }
}

#define RABITQ_TILE_SIZE 8

// Beam search with rerank at the end
// while using rabitq.
template <
    typename INDEX_T, typename DATA_T, uint16_t DATA_DIM, typename DISTANCE_T,
    typename EDGE_LIST_T, uint32_t BLOCK_SIZE,
    template <typename, typename, uint, uint> class exact_distance_functor,
    bool GET_VISITED = true, uint16_t SIZE_PER_DIM = 1, uint32_t MAX_SEARCH_WIDTH = 512>
__host__
    thrust::pair<thrust::pair<INDEX_T, DISTANCE_T> *,
                 thrust::pair<thrust::pair<INDEX_T, DISTANCE_T> *, uint32_t *>>
    beam_search_rerank(
        // graph
        EDGE_LIST_T *graph, uint8_t *edge_count,
        // original
        data_vector<DATA_T, DATA_DIM> *data_vectors, uint64_t n_data_vectors,
        data_vector<DATA_T, DATA_DIM> *query_vectors, uint64_t n_query_vectors,
        // quantized
        RabitqDataVec<SIZE_PER_DIM, DATA_DIM> *rabitq_vectors,
        float *rabitq_query_vectors,
        RabitqQueryFactor *rabitq_query_factors,
        // misc
        INDEX_T medoid, uint32_t k, uint32_t beam_width, float cut,
        uint32_t limit) {
  if (k > beam_width) {
    std::cerr << "BeamSearch: k is larger than beam width (needs to be equal "
                 "or smaller)\n";
    exit(0);
  }

  constexpr uint32_t hashmap_bitlen = 9;
  dim3 thread_dims(BLOCK_SIZE, 1, 1);
  dim3 block_dims(n_query_vectors, 1, 1);
  uint32_t smem_size = get_smem_size<INDEX_T, DISTANCE_T>(
      beam_width, BLOCK_SIZE, hashmap_bitlen, k);

  // allocate location for result
  constexpr uint32_t MAX_RESULT_SIZE = 1024;
  using ENTRY_T = thrust::pair<INDEX_T, DISTANCE_T>;
  ENTRY_T *frontier_results = gallatin::utils::get_device_version<ENTRY_T>(
      n_query_vectors * beam_width);
  
  ENTRY_T *visited_results;
  uint32_t *visited_counts;
  if (GET_VISITED) {
    visited_results = gallatin::utils::get_device_version<ENTRY_T>(
      n_query_vectors * MAX_RESULT_SIZE);
    visited_counts =
      gallatin::utils::get_device_version<uint32_t>(n_query_vectors);
  }
  
  //int sharedMemBytes = 65536; // 64kb
  cudaError_t err = cudaFuncSetAttribute(
    (void *)beam_search_kernel<INDEX_T, DATA_T, DATA_DIM, DISTANCE_T, EDGE_LIST_T,
                     BLOCK_SIZE, MAX_SEARCH_WIDTH, GET_VISITED, RABITQ_TILE_SIZE, SIZE_PER_DIM>,
    cudaFuncAttributePreferredSharedMemoryCarveout,
    cudaSharedmemCarveoutMaxShared);

  if (err != cudaSuccess) {
      std::cerr << "cudaFuncSetAttribute failed: "
                << cudaGetErrorString(err) << std::endl;
  }

  beam_search_kernel<INDEX_T, DATA_T, DATA_DIM, DISTANCE_T, EDGE_LIST_T,
                     BLOCK_SIZE, MAX_SEARCH_WIDTH, GET_VISITED, RABITQ_TILE_SIZE, SIZE_PER_DIM>
      <<<block_dims, thread_dims, smem_size>>>(
          graph, edge_count, frontier_results, visited_results, visited_counts,
          rabitq_vectors, n_data_vectors, rabitq_query_vectors,
          rabitq_query_factors, n_query_vectors, medoid, beam_width, beam_width,
          cut, limit, hashmap_bitlen);

  err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    std::cerr << "beam_search_kernel launch failed: " << cudaGetErrorString(err)
              << std::endl;
  }

  // reranking
  ENTRY_T *reranked_frontier_results =
      gallatin::utils::get_device_version<ENTRY_T>(n_query_vectors * k);
  rerank_frontiers<INDEX_T, DATA_T, DATA_DIM, DISTANCE_T, BLOCK_SIZE,
                   exact_distance_functor, MAX_SEARCH_WIDTH>
      <<<block_dims, thread_dims>>>(frontier_results, reranked_frontier_results,
                                    data_vectors, n_data_vectors, query_vectors,
                                    n_query_vectors, k, beam_width);
  err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    std::cerr << "rerank_frontiers kernel launch failed: " << cudaGetErrorString(err)
              << std::endl;
  }

  thrust::pair<ENTRY_T *, uint32_t *> visited_pack = {visited_results,
                                                      visited_counts};
  return {reranked_frontier_results, visited_pack};
}

}  // namespace rabitq
}  // namespace gpu_ann