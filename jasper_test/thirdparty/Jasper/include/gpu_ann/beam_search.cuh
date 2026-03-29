#pragma once

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cooperative_groups/scan.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <thrust/pair.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <cub/cub.cuh>
#include <fstream>
#include <gallatin/allocators/alloc_utils.cuh>
#include <gallatin/data_structs/ds_utils.cuh>
#include <gpu_ann/_proj_metadata.cuh>
#include <gpu_ann/edge_list.cuh>
#include <gpu_ann/edge_sorting.cuh>
#include <gpu_ann/hash_tables/array.cuh>
#include <gpu_ann/hash_tables/hashmap.cuh>
#include <gpu_ann/vector.cuh>
#include <gpu_error/log.cuh>
#include <iostream>
#include <vector>

#include "assert.h"
#include "stdio.h"

namespace cg = cooperative_groups;

namespace gpu_ann {

#define BEAM_SEARCH_TILE_SIZE 4
// #define _CLK_BREAKDOWN

#define MEASURE_WASTED_CALCS 0

#if MEASURE_WASTED_CALCS

__managed__ int wasted_distance_calcs = 0; 

#endif

// Helper function to read an entry.
// the structure of the entry is
// - 1 bits:  visited
// - 31 bits: index (uint32_t)
// - 32 bits: distance
typedef uint64_t ENTRY_T;

__device__ __forceinline__ ENTRY_T create_entry(uint32_t index, float distance) {
  uint64_t entry = 0;
  entry |= (uint64_t(index) & 0x7FFFFFFFul) << 32;
  uint32_t dist_bits = __float_as_uint(distance);
  entry |= (static_cast<uint64_t>(dist_bits));
  return entry;
}

__device__ __forceinline__ ENTRY_T empty_entry() {
  return create_entry(
    std::numeric_limits<uint32_t>::max(),
    std::numeric_limits<float>::max()
  );
}

__device__ __forceinline__ bool get_visited(ENTRY_T i) {
  return (i >> 63) & 1;
}

__device__ __forceinline__ ENTRY_T set_visited(ENTRY_T i) {
  constexpr uint64_t MASK = 1ull << 63;
  return i | MASK;
}

__device__ __forceinline__ uint32_t get_index(ENTRY_T entry) {
  return static_cast<uint32_t>((entry >> 32) & 0x7FFFFFFF);
}

__device__ __forceinline__ ENTRY_T set_index(ENTRY_T entry, uint32_t index) {
  uint64_t idx64 = static_cast<uint64_t>(index) << 32;
  uint64_t dist64 = entry & 0xFFFFFFFFull;
  return dist64 | idx64;
}

__device__ __forceinline__ float get_distance(ENTRY_T entry) {
  uint32_t dist_bits = static_cast<uint32_t>(entry & 0xFFFFFFFF);
  return __uint_as_float(dist_bits);
}

__device__ __forceinline__ ENTRY_T set_distance(ENTRY_T entry, float distance) {
  uint32_t dist_bits = __float_as_uint(static_cast<float>(distance));
  uint64_t dist64 = uint64_t(dist_bits);
  uint64_t idx64 = entry & 0xFFFFFFFF00000000ull;
  return idx64 | dist64;
}

template <typename INDEX_T, typename DISTANCE_T>
__device__ void debug_print_result_buffer(
    ENTRY_T *result_buffer,
    uint32_t *result_buffer_count, uint32_t k, uint32_t beam_width) {
  const uint32_t total_entries = beam_width + 64;

  constexpr INDEX_T INVALID_INDEX = std::numeric_limits<INDEX_T>::max();

  // Only one thread prints to avoid interleaving
  if (threadIdx.x == 0 && blockIdx.x == 0) {

    printf("[b=%u]---- Result Buffer (result_buffer_count=%u)----\n",
         blockIdx.x, result_buffer_count[0]);
    for (uint32_t i = 0; i < total_entries && i < result_buffer_count[0]; ++i) {
      printf("[b=%u] i=%u index=%lld, visited=%lld, distance=%f\n", blockIdx.x, i,
            static_cast<long long>(get_index(result_buffer[i])),
            static_cast<long long>(get_visited(result_buffer[i])),
            static_cast<double>(get_distance(result_buffer[i])));
    }
    
  }
  __syncthreads();
}

// distance calculation
template <typename DATA_T, uint16_t DATA_DIM, typename INDEX_T,
          typename DISTANCE_T, uint32_t tile_size,
          template <typename, typename, uint, uint> class distance_functor>
__device__ void populate_distances(
    const data_vector<DATA_T, DATA_DIM> &query_vec,
    data_vector<DATA_T, DATA_DIM> *data_vectors,
    ENTRY_T *result_buffer,
    uint32_t *result_buffer_count, uint32_t offset) {
  auto thread_block = cg::this_thread_block();
  cg::thread_block_tile<tile_size> my_tile =
      cg::tiled_partition<tile_size>(thread_block);
  uint64_t tid = my_tile.meta_group_rank();
  constexpr INDEX_T INVALID_INDEX = std::numeric_limits<INDEX_T>::max();
  using distance_type = distance_functor<DATA_T, DATA_T, DATA_DIM, tile_size>;

  uint32_t count = result_buffer_count[0];
  for (unsigned i = tid + offset; i < count; i += my_tile.meta_group_size()) {
    auto dest = get_index(result_buffer[i]);
    if (dest != INVALID_INDEX) {
      float dist =
          distance_type::distance(&query_vec, &data_vectors[dest], my_tile);
      if (my_tile.thread_rank() == 0) {
        result_buffer[i] = set_distance(result_buffer[i], dist);
      }
    }
  }
  __threadfence();
  __syncthreads();
}

// select and return the new frontier
template <typename INDEX_T, typename DISTANCE_T, uint32_t BLOCK_SIZE, uint32_t MAX_SEARCH_WIDTH>
__device__ thrust::pair<INDEX_T, bool> choose_new_frontier(
    ENTRY_T *result_buffer,
    uint32_t *result_buffer_count) {

  constexpr uint32_t ELEMENTS_PER_THREAD = (MAX_SEARCH_WIDTH - 1) / BLOCK_SIZE + 1;

  // first_index is the index we want to find that we haven't traversed yet.
  __shared__ uint32_t first_index;
  if (threadIdx.x == 0) first_index = ~0u;
  __syncthreads();

  for (uint i=0; i<ELEMENTS_PER_THREAD; i++) {
    uint32_t visited;
    uint32_t index_to_visit = i * BLOCK_SIZE + threadIdx.x;
    if (index_to_visit < *result_buffer_count) {
      visited = static_cast<uint32_t>(get_visited(result_buffer[index_to_visit]));
    } else {
      visited = 1;
    }
    unsigned mask = __ballot_sync(0xffffffff, !visited);
    if (mask != 0) {
      int lane_id = threadIdx.x % warpSize;
      int first_lane = __ffs(mask) - 1;
      if (lane_id == first_lane) {
        atomicMin(&first_index, index_to_visit);
      }
    }
  }
  __syncthreads();
  
  return {first_index, first_index != ~0u};
}

// adding the newly selected frontier's neighbors to the frontier list
template <typename INDEX_T, typename DISTANCE_T, typename EDGE_LIST_T>
__device__ void add_frontier_out(
    const EDGE_LIST_T * __restrict__ graph, const uint8_t * __restrict__ edge_count,
    ENTRY_T *result_buffer,
    uint32_t *result_buffer_count, const INDEX_T & frontier, const uint32_t & k,
    const uint32_t & beam_width) {
  constexpr uint32_t max_edge = 64;
  uint8_t n_edges = edge_count[frontier];
  uint32_t offset = result_buffer_count[0];
  __syncthreads();
  constexpr DISTANCE_T INVALID_DISTANCE =
      std::numeric_limits<DISTANCE_T>::max();

  const uint4* l_ptr = reinterpret_cast<const uint4*>(&graph[frontier].edges);

  for (uint i = threadIdx.x*4; i < n_edges; i += blockDim.x*4){
    uint4 loaded_edges = l_ptr[i/4];
    const uint32_t * loaded_edges_ptr = (uint32_t *) &loaded_edges;
    for (uint j = 0; j < 4; j++){
      result_buffer[offset+i+j] = set_index(empty_entry(), loaded_edges_ptr[j]);
    }
  }

  __syncthreads();
  if (threadIdx.x == 0) {
    result_buffer_count[0] += n_edges;
  }
  __syncthreads();
}

// Custom comparison for {index, distance} pairs
struct CustomPairLess {
  __device__ __forceinline__ bool operator()(
      const ENTRY_T &a,
      const ENTRY_T &b) {
    float da = __uint_as_float(a & 0xFFFFFFFFu);
    float db = __uint_as_float(b & 0xFFFFFFFFu);

    // distance comparison
    uint32_t dist_lt = static_cast<uint32_t>(da < db);
    uint32_t dist_eq = static_cast<uint32_t>(da == db);

    // index comparison. (31 bits)
    uint32_t ia = static_cast<uint32_t>((a >> 32) & 0x7FFFFFFFul);
    uint32_t ib = static_cast<uint32_t>((b >> 32) & 0x7FFFFFFFul);
    uint32_t idx_lt = static_cast<uint32_t>(ia < ib);

    // final result: (dist_lt) OR (dist_eq AND idx_lt)
    return (dist_lt | (dist_eq & idx_lt)) != 0;
  }
};

// sort the result
template <typename INDEX_T, typename DISTANCE_T, uint32_t BLOCK_SIZE,
          uint32_t MAX_SEARCH_WIDTH, typename BlockMergeSortT>
__device__ void merge_sort(ENTRY_T *result_buffer,
                           uint32_t *result_buffer_count,
                           typename BlockMergeSortT::TempStorage &temp_storage) {
  uint32_t count = result_buffer_count[0];
  constexpr uint32_t ELEMENTS_PER_THREAD = (MAX_SEARCH_WIDTH - 1) / BLOCK_SIZE + 1;

  ENTRY_T thread_item[ELEMENTS_PER_THREAD];
#pragma unroll
  for (unsigned i = 0; i < ELEMENTS_PER_THREAD; i++) {
    uint32_t element_id = threadIdx.x * ELEMENTS_PER_THREAD + i;
    if (element_id < count) {
      thread_item[i] = result_buffer[element_id];
    } else {
      thread_item[i] = empty_entry();
    }
  }

  // sort by distance
  BlockMergeSortT(temp_storage).Sort(thread_item, CustomPairLess());

#pragma unroll
  for (unsigned i = 0; i < ELEMENTS_PER_THREAD; i++) {
    uint32_t element_id = threadIdx.x * ELEMENTS_PER_THREAD + i;
    if (element_id < count) {
      result_buffer[element_id] = thread_item[i];
    }
  }

  __syncthreads();
}

// clip the search results to beam_width
__device__ void clip_k(uint32_t* result_buffer_count, const uint32_t & k) {
  if (threadIdx.x == 0) {
    result_buffer_count[0] = min(result_buffer_count[0], k);
  }
  __syncthreads();
}

// deduplicate the frontier list
// assume the list is already sorted
template <typename INDEX_T, typename DISTANCE_T, uint32_t BLOCK_SIZE,
          uint32_t MAX_SEARCH_WIDTH, typename BlockScanT>
__device__ void dedup_results(ENTRY_T *result_buffer,
                              uint32_t *result_buffer_count,
                              typename BlockScanT::TempStorage &temp_storage) {
  constexpr uint32_t ELEMENTS_PER_THREAD =
      (MAX_SEARCH_WIDTH - 1) / BLOCK_SIZE + 1;

  uint32_t count = result_buffer_count[0];

  ENTRY_T this_entry[ELEMENTS_PER_THREAD];
  bool is_unique[ELEMENTS_PER_THREAD];

#pragma unroll
  for (unsigned i = 0; i < ELEMENTS_PER_THREAD; i++) {
    is_unique[i] = true;
    uint32_t element_id = threadIdx.x * ELEMENTS_PER_THREAD + i;
    if (element_id < count) {
      this_entry[i] = result_buffer[element_id];
      INDEX_T this_index = get_index(this_entry[i]);
      INDEX_T last_index = get_index(result_buffer[element_id - 1]);
      if (element_id > 0 && (this_index == last_index)) {
        is_unique[i] = false;

        #if MEASURE_WASTED_CALCS
        atomicAdd(&wasted_distance_calcs, 1);
        #endif

      }
    } else {
      is_unique[i] = false;
    }
  }
  __syncthreads();

  uint32_t thread_data[ELEMENTS_PER_THREAD];
#pragma unroll
  for (unsigned i = 0; i < ELEMENTS_PER_THREAD; i++) {
    thread_data[i] = is_unique[i] ? 1 : 0;
  }

  BlockScanT(temp_storage).ExclusiveSum(thread_data, thread_data);

#pragma unroll
  for (unsigned i = 0; i < ELEMENTS_PER_THREAD; i++) {
    if (is_unique[i]) {
      result_buffer[thread_data[i]] = this_entry[i];
    }
  }
  if (threadIdx.x == blockDim.x - 1) {
    result_buffer_count[0] = thread_data[ELEMENTS_PER_THREAD - 1]
                           + (is_unique[ELEMENTS_PER_THREAD - 1] ? 1 : 0);
  }
  __syncthreads();
}

// Cut
// Assume result_buffer is sorted
template <typename INDEX_T, typename DISTANCE_T, bool DISABLE_K_FILTERING = false>
__device__ void filter_frontier(
    ENTRY_T *result_buffer,
    uint32_t *result_buffer_count, const uint32_t & k, const float & cut) {
  if constexpr (DISABLE_K_FILTERING) return;
  
  const uint32_t count = result_buffer_count[0];
  if (count <= k) return;

  __shared__ DISTANCE_T cutoff;
  __shared__ uint32_t new_count;
  if (threadIdx.x == 0) {
    cutoff = get_distance(result_buffer[k - 1]) * cut;
    new_count = std::numeric_limits<uint32_t>::max();
  }
  __syncthreads();

  for (uint i=threadIdx.x; i<count; i+=blockDim.x) {
    if (get_distance(result_buffer[i]) > cutoff) {
      atomicMin(&new_count, i);
      break;
    }
  }
  __syncthreads();
  if (threadIdx.x == 0 && new_count != std::numeric_limits<uint32_t>::max()) {
    result_buffer_count[0] = new_count;
  }
  __syncthreads();
}

// main search kernel
template <typename INDEX_T, typename DATA_T, uint16_t DATA_DIM,
          typename DISTANCE_T, typename EDGE_LIST_T, uint32_t BLOCK_SIZE,
          template <typename, typename, uint, uint> class distance_functor,
          uint32_t MAX_SEARCH_WIDTH, bool GET_VISITED, uint32_t TILE_SIZE = 4, bool DISABLE_K_FILTERING = false>
//__maxnreg__(50)
__global__ void beam_search_single_kernel(
    EDGE_LIST_T *graph, uint8_t *edge_count,
    thrust::pair<INDEX_T, DISTANCE_T> *frontier_results,
    thrust::pair<INDEX_T, DISTANCE_T> *visited_results,
    uint32_t *visited_counts, data_vector<DATA_T, DATA_DIM> *data_vectors,
    uint64_t n_data_vectors, data_vector<DATA_T, DATA_DIM> *query_vectors,
    uint64_t n_query_vectors, INDEX_T medoid, uint32_t k, uint32_t beam_width,
    float cut, uint32_t limit, const uint32_t hashmap_bitlen) {

  #ifdef _CLK_BREAKDOWN
    std::uint64_t clk_init = 0;
    std::uint64_t clk_1st_opulate_distance = 0;
    std::uint64_t clk_choose_new_frontier = 0;
    std::uint64_t clk_insert_hash = 0;
    std::uint64_t clk_add_frontier_out = 0;
    std::uint64_t clk_populate_distances = 0;
    std::uint64_t clk_merge_sort = 0;
    std::uint64_t clk_clip_k = 0;
    std::uint64_t clk_dedup = 0;
    std::uint64_t clk_filter_frontier = 0;
    std::uint64_t clk_start;
    #define _CLK_START() clk_start = clock64();
    #define _CLK_REC(V) V += clock64() - clk_start;
  #else
    #define _CLK_START();
    #define _CLK_REC(V);
  #endif

  
  // get the query vector for this block
  const auto query_id = blockIdx.x;
  uint32_t visited_counter = 0;

  assert(beam_width + 64 <= MAX_SEARCH_WIDTH);

  // allocate shared memory
  _CLK_START();
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

  // load query vector into shared memory
  __shared__ data_vector<DATA_T, DATA_DIM> smem_query_vec;
  if (threadIdx.x == 0) {
    smem_query_vec = query_vectors[query_id];
  }

  // initialize shared memory to default (invalid) key
  for (unsigned i = threadIdx.x; i < result_buffer_size; i += blockDim.x) {
    result_buffer[i] = empty_entry();
  }

  // populate frontier in shared memory
  // TODO: we only put the mediod in there right now,
  //       maybe it is more efficient to start with multiple points.
  if (threadIdx.x == 0) {
    result_buffer[0] = set_index(empty_entry(), medoid);
    result_buffer_count[0] = 1;
  }
  __syncthreads();
  _CLK_REC(clk_init);

  _CLK_START();
  populate_distances<DATA_T, DATA_DIM, INDEX_T, DISTANCE_T, TILE_SIZE,
                     distance_functor>(smem_query_vec, data_vectors,
                                       result_buffer, result_buffer_count, 0);
  _CLK_REC(clk_1st_opulate_distance);

  // loop
  uint32_t offset;
  uint32_t loop_count = 0;
  while (loop_count <= limit) {

    loop_count += 1;
    // choose a new frontier to explore
    _CLK_START();
    auto [frontierIdx, found] =
        choose_new_frontier<INDEX_T, DISTANCE_T, BLOCK_SIZE, MAX_SEARCH_WIDTH>(
          result_buffer, result_buffer_count);
    if (!found) {
      break;  // we converged
    };
    _CLK_REC(clk_choose_new_frontier);

    _CLK_START();
    if constexpr (BLOCK_SIZE > 33) {
      if (threadIdx.x == 33) {
        result_buffer[frontierIdx] = set_visited(result_buffer[frontierIdx]);
      }
    } else {
      if (threadIdx.x == 1) {
        result_buffer[frontierIdx] = set_visited(result_buffer[frontierIdx]);
      }
    }
    __syncthreads();
    _CLK_REC(clk_insert_hash);
    
    // Add frontier to visited list
    // we do not add the first point
    if (threadIdx.x == 0 && GET_VISITED) {
      visited_results[query_id * 1024 + visited_counter].first =
          get_index(result_buffer[frontierIdx]);
      visited_results[query_id * 1024 + visited_counter].second =
          get_distance(result_buffer[frontierIdx]);
      assert(visited_counter < 1024);
    }
    visited_counter += 1;
    __syncthreads();

    // record offset so that we know where to start calculating distances
    // (all the previous ones are calculated)
    offset = result_buffer_count[0];
    __syncthreads();

    // Add frontier's neighbor to results
    _CLK_START();
    add_frontier_out<INDEX_T, DISTANCE_T>(graph, edge_count, result_buffer, result_buffer_count,
                     get_index(result_buffer[frontierIdx]), k, beam_width);
    __syncthreads();
    _CLK_REC(clk_add_frontier_out);

    // populate distance
    _CLK_START();
    populate_distances<DATA_T, DATA_DIM, INDEX_T, DISTANCE_T, TILE_SIZE,
                       distance_functor>(smem_query_vec, data_vectors,
                                         result_buffer, result_buffer_count,
                                         offset);
    __syncthreads();
    _CLK_REC(clk_populate_distances);

    //if (threadIdx.x == 0) printf("After populate dist result count is %u\n", result_buffer_count[0]);

    // sort the result buffer
    _CLK_START();
    merge_sort<INDEX_T, DISTANCE_T, BLOCK_SIZE, MAX_SEARCH_WIDTH, BlockMergeSortT>(
        result_buffer, result_buffer_count, temp_storage.sort_storage);
    _CLK_REC(clk_merge_sort);

    //if (threadIdx.x == 0) printf("After merge sort result count is %u\n", result_buffer_count[0]);


    _CLK_START();
    dedup_results<INDEX_T, DISTANCE_T, BLOCK_SIZE, MAX_SEARCH_WIDTH, BlockScanT>(
        result_buffer, result_buffer_count, temp_storage.scan_storage);
    _CLK_REC(clk_dedup);

    //if (threadIdx.x == 0) printf("After dedup result count is %u\n", result_buffer_count[0]);


    _CLK_START();
    clip_k(result_buffer_count, beam_width);
    _CLK_REC(clk_clip_k);

    //if (threadIdx.x == 0) printf("After clip result count is %u\n", result_buffer_count[0]);

    _CLK_START();
    filter_frontier<INDEX_T, DISTANCE_T, DISABLE_K_FILTERING>(result_buffer, result_buffer_count, k, cut);
    _CLK_REC(clk_filter_frontier);

    //if (threadIdx.x == 0) printf("After filter result count is %u\n", result_buffer_count[0]);

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

  #ifdef _CLK_BREAKDOWN
  if (threadIdx.x == 0 && blockIdx.x == 20) {
    printf(
      "%s:%d "
      "query %d thread %d visited=%u\n"
      " - init,                %lu\n"
      " - 1st_distance,        %lu\n"
      " - choose_new_frontier, %lu\n"
      " - mark_visited,        %lu\n"
      " - add_frontier_out,    %lu\n"
      " - populate_distances,  %lu\n"
      " - merge_sort,          %lu\n"
      " - deduplication,       %lu\n"
      " - clip_k,              %lu\n"
      " - filter_frontier,     %lu\n"
      "\n",
      __FILE__,
      __LINE__,
      blockIdx.x,
      threadIdx.x,
      visited_counter,
      clk_init,
      clk_1st_opulate_distance,
      clk_choose_new_frontier,
      clk_insert_hash,
      clk_add_frontier_out,
      clk_populate_distances,
      clk_merge_sort,
      clk_dedup,
      clk_clip_k,
      clk_filter_frontier
    );
  }
  #endif
}

// Get how big the shared memory needs to be
// in bytes.
template <typename INDEX_T, typename DISTANCE_T>
__host__ uint32_t get_smem_size(const uint32_t beam_width,
                                const uint32_t block_size,
                                const uint32_t hashmap_bitlen,
                                const uint32_t k) {
  // using ENTRY_T = thrust::pair<INDEX_T, DISTANCE_T>;
  uint32_t smem_size = 0;
  smem_size += sizeof(ENTRY_T) * (beam_width + 64);
  //smem_size += sizeof(uint32_t) * hashmap::get_size_host(hashmap_bitlen);
  smem_size += sizeof(uint32_t);
  return smem_size;
}

// Rerank a pq beam search result using the actual vectors
template <typename INDEX_T, typename DATA_T, uint16_t DATA_DIM,
          typename DISTANCE_T, uint32_t BLOCK_SIZE,
          template <typename, typename, uint, uint> class distance_functor,
          uint32_t MAX_SEARCH_WIDTH = 512, uint32_t TILE_SIZE = 4>
__global__ void rerank_frontiers(
    thrust::pair<INDEX_T, DISTANCE_T> *frontiers,
    thrust::pair<INDEX_T, DISTANCE_T> *reranked_frontiers,
    data_vector<DATA_T, DATA_DIM> *data_vectors, uint64_t n_data_vectors,
    data_vector<DATA_T, DATA_DIM> *query_vectors, uint64_t n_query_vectors,
    uint32_t k, uint32_t beam_width) {
  // each block responsible for one query vector
  const auto query_id = blockIdx.x;
  //const auto query_vec = query_vectors[query_id];

  __shared__ ENTRY_T local_frontiers[MAX_SEARCH_WIDTH];
  for (unsigned i = threadIdx.x; i < MAX_SEARCH_WIDTH; i += blockDim.x) {
    if (i < beam_width) {
      local_frontiers[i] = create_entry(
        frontiers[query_id * beam_width + i].first, 
        frontiers[query_id * beam_width + i].second);
    } else {
      local_frontiers[i] = empty_entry();
    }
  }
  __syncthreads();

  // populate distance
  populate_distances<DATA_T, DATA_DIM, INDEX_T, DISTANCE_T, TILE_SIZE,
                     distance_functor>(query_vectors[query_id], data_vectors, local_frontiers,
                                       &beam_width, 0);

  // sort this segment
  // using ENTRY_T = thrust::pair<INDEX_T, DISTANCE_T>;
  constexpr uint32_t ELEMENTS_PER_THREAD = (MAX_SEARCH_WIDTH - 1) / BLOCK_SIZE + 1;
  using BlockMergeSortT = cub::BlockMergeSort<ENTRY_T, BLOCK_SIZE, ELEMENTS_PER_THREAD>;
  __shared__ typename BlockMergeSortT::TempStorage sort_storage;
  merge_sort<INDEX_T, DISTANCE_T, BLOCK_SIZE, MAX_SEARCH_WIDTH, BlockMergeSortT>(local_frontiers,
                                                                &beam_width,
                                                                sort_storage);

  // copy result top k to reranked_frontiers
  for (unsigned i = threadIdx.x; i < MAX_SEARCH_WIDTH; i += blockDim.x) {
    if (i < k) {
      reranked_frontiers[query_id * k + i].first =
          get_index(local_frontiers[i]);
      reranked_frontiers[query_id * k + i].second =
          get_distance(local_frontiers[i]);
    }
  }
}

// host function call
template <typename INDEX_T, typename DATA_T, uint16_t DATA_DIM,
          typename DISTANCE_T, typename EDGE_LIST_T, uint32_t BLOCK_SIZE,
          template <typename, typename, uint, uint> class distance_functor,
          bool GET_VISITED=true,
          uint32_t MAX_SEARCH_WIDTH = 512,
          uint32_t TILE_SIZE = 4,
          bool DISABLE_K_FILTERING = false>
__host__ thrust::pair<thrust::pair<INDEX_T, DISTANCE_T> *,  // frontier
                      thrust::pair<thrust::pair<INDEX_T, DISTANCE_T> *,
                                   uint32_t *>  // visited + count
                      >
beam_search_single(EDGE_LIST_T *graph, uint8_t *edge_count,
                   data_vector<DATA_T, DATA_DIM> *data_vectors,
                   uint64_t n_data_vectors,
                   data_vector<DATA_T, DATA_DIM> *query_vectors,
                   uint64_t n_query_vectors, INDEX_T medoid, uint32_t k,
                   uint32_t beam_width, float cut, uint32_t limit) {

  constexpr uint32_t hashmap_bitlen = 9;
  dim3 thread_dims(BLOCK_SIZE, 1, 1);
  dim3 block_dims(n_query_vectors, 1, 1);
  uint32_t smem_size = get_smem_size<INDEX_T, DISTANCE_T>(
      beam_width, BLOCK_SIZE, hashmap_bitlen, k);

  // allocate location for result
  constexpr uint32_t MAX_RESULT_SIZE = 1024;
  using ENTRY_T = thrust::pair<INDEX_T, DISTANCE_T>;
  ENTRY_T *frontier_results =
      gallatin::utils::get_device_version<ENTRY_T>(n_query_vectors * k);

  ENTRY_T *visited_results;
  uint32_t *visited_counts;
  if (GET_VISITED) {
    visited_results = gallatin::utils::get_device_version<ENTRY_T>(
      n_query_vectors * MAX_RESULT_SIZE);
    visited_counts =
      gallatin::utils::get_device_version<uint32_t>(n_query_vectors);
  }

  beam_search_single_kernel<INDEX_T, DATA_T, DATA_DIM, DISTANCE_T, EDGE_LIST_T,
                            BLOCK_SIZE, distance_functor, MAX_SEARCH_WIDTH, GET_VISITED, TILE_SIZE, DISABLE_K_FILTERING>
      <<<block_dims, thread_dims, smem_size>>>(
          graph, edge_count, frontier_results, visited_results, visited_counts,
          data_vectors, n_data_vectors, query_vectors, n_query_vectors, medoid,
          DISABLE_K_FILTERING ? 0 : k, beam_width, cut, limit, hashmap_bitlen);

  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    std::cerr << "Kernel launch failed: " << cudaGetErrorString(err)
              << std::endl;
  }

  thrust::pair<ENTRY_T *, uint32_t *> visited_pack = {visited_results,
                                                      visited_counts};
  return {frontier_results, visited_pack};
}

// Beam search with rerank at the end.
// when using quantized vectors.
template <
    typename INDEX_T, typename DATA_T, uint16_t DATA_DIM, typename PQ_DATA_T,
    uint16_t PQ_DATA_DIM, typename DISTANCE_T, typename EDGE_LIST_T,
    uint32_t BLOCK_SIZE,
    template <typename, typename, uint, uint> class exact_distance_functor,
    template <typename, typename, uint, uint> class pq_distance_functor,
    uint32_t MAX_SEARCH_WIDTH = 512>
__host__
    thrust::pair<thrust::pair<INDEX_T, DISTANCE_T> *,
                 thrust::pair<thrust::pair<INDEX_T, DISTANCE_T> *, uint32_t *>>
    beam_search_single_rerank(
        EDGE_LIST_T *graph, uint8_t *edge_count,
        data_vector<DATA_T, DATA_DIM> *data_vectors,
        data_vector<PQ_DATA_T, PQ_DATA_DIM> *pq_data_vectors,
        uint64_t n_data_vectors, data_vector<DATA_T, DATA_DIM> *query_vectors,
        data_vector<PQ_DATA_T, PQ_DATA_DIM> *pq_query_vectors,
        uint64_t n_query_vectors, INDEX_T medoid, uint32_t k,
        uint32_t beam_width, float cut, uint32_t limit) {
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
  ENTRY_T *visited_results = gallatin::utils::get_device_version<ENTRY_T>(
      n_query_vectors * MAX_RESULT_SIZE);
  uint32_t *visited_counts =
      gallatin::utils::get_device_version<uint32_t>(n_query_vectors);

  beam_search_single_kernel<INDEX_T, PQ_DATA_T, PQ_DATA_DIM, DISTANCE_T,
                            EDGE_LIST_T, BLOCK_SIZE, pq_distance_functor,
                            MAX_SEARCH_WIDTH, false, 4>
      <<<block_dims, thread_dims, smem_size>>>(
          graph, edge_count, frontier_results, visited_results, visited_counts,
          pq_data_vectors, n_data_vectors, pq_query_vectors, n_query_vectors,
          medoid,
          beam_width,  // instead of k
          beam_width, cut, limit, hashmap_bitlen);

  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    std::cerr << "Kernel launch failed: " << cudaGetErrorString(err)
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

  thrust::pair<ENTRY_T *, uint32_t *> visited_pack = {visited_results,
                                                      visited_counts};
  return {frontier_results, visited_pack};
}

// debug print
template <typename INDEX_T, typename DISTANCE_T>
__global__ void print_visited_kernel(
    const thrust::pair<thrust::pair<INDEX_T, DISTANCE_T> *, uint32_t *>
        visited_pack,
    uint32_t n_queries, uint32_t max_result_size) {
  using ENTRY_T = thrust::pair<INDEX_T, DISTANCE_T>;

  ENTRY_T *visited_results = visited_pack.first;
  uint32_t *visited_counts = visited_pack.second;

  for (uint32_t qid = 0; qid < n_queries; qid++) {
    uint32_t count = visited_counts[qid];
    printf("Query %u: count = %u\n", qid, count);

    for (uint32_t i = 0; i < count; ++i) {
      ENTRY_T entry = visited_results[qid * max_result_size + i];
      printf("  [Q=%u, %u] -> index=%u, dist=%f\n", qid, i,
             (uint32_t)entry.first, (float)entry.second);
    }
  }
}

template <typename vertex_data_type, typename distance_type,
          typename edge_pair_type>
__global__ void convert_to_edges_kernel(
    const uint32_t *visited_counts,
    const thrust::pair<vertex_data_type, distance_type> *visited_results,
    const thrust::pair<vertex_data_type, distance_type> *frontier_result,
    edge_pair_type *visited_edges, edge_pair_type *frontier_edges,
    uint32_t n_vectors, uint32_t start, uint32_t max_result_size,
    uint32_t *visited_output_offsets  // exclusive scan of visited_counts
) {
  uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n_vectors) return;

  uint32_t source = idx;
  edge_pair_type f_edge;
  f_edge.source = start + source;
  f_edge.sink = frontier_result[idx].first;
  f_edge.distance = frontier_result[idx].second;
  frontier_edges[idx] = f_edge;

  // Write visited edges
  uint32_t count = visited_counts[idx];
  uint32_t offset = visited_output_offsets[idx];
  for (uint32_t j = 0; j < count; ++j) {
    edge_pair_type v_edge;
    v_edge.source = start + source;
    v_edge.sink = visited_results[idx * max_result_size + j].first;
    v_edge.distance = visited_results[idx * max_result_size + j].second;
    visited_edges[offset + j] = v_edge;
  }
}

// helper function to convert frontiers from thrust distance pairs
// into thrust vectors for calculating recall.
template <typename INDEX_T, typename DISTANCE_T>
thrust::device_vector<gpu_ann::edge_pair<INDEX_T, DISTANCE_T>> __host__
convert_frontier_pairs_to_vectors(const uint64_t n_query_vectors,
                                  const uint32_t k,
                                  thrust::pair<INDEX_T, DISTANCE_T> *frontiers) {
  auto h_frontiers =
      gallatin::utils::move_to_host<thrust::pair<INDEX_T, DISTANCE_T>>(
          frontiers, n_query_vectors * k);
  using h_edges_t = thrust::host_vector<gpu_ann::edge_pair<INDEX_T, DISTANCE_T>>;
  using d_edges_t = thrust::device_vector<gpu_ann::edge_pair<INDEX_T, DISTANCE_T>>;
  h_edges_t h_edges(n_query_vectors * k);
  for (unsigned i = 0; i < n_query_vectors; i++) {
    for (unsigned j = 0; j < k; j++) {
      h_edges[i * k + j] = {i, h_frontiers[i * k + j].first,
                            h_frontiers[i * k + j].second};
    }
  }
  d_edges_t d_edges = h_edges;
  return d_edges;
}

// helper function to convert frontiers from thrust distance pairs
// into thrust vectors for calculating recall, while also consider 
// the reordering maps. so it recreates the original vector's index.
template <typename INDEX_T, typename DISTANCE_T>
thrust::device_vector<gpu_ann::edge_pair<INDEX_T, DISTANCE_T>> __host__
convert_frontier_pairs_to_vectors(const uint64_t n_query_vectors,
                                  const uint32_t k,
                                  thrust::pair<INDEX_T, DISTANCE_T> *frontiers,
                                  std::vector<INDEX_T> mapping) {
  auto h_frontiers =
      gallatin::utils::move_to_host<thrust::pair<INDEX_T, DISTANCE_T>>(
          frontiers, n_query_vectors * k);
  using h_edges_t = thrust::host_vector<gpu_ann::edge_pair<INDEX_T, DISTANCE_T>>;
  using d_edges_t = thrust::device_vector<gpu_ann::edge_pair<INDEX_T, DISTANCE_T>>;
  h_edges_t h_edges(n_query_vectors * k);
  for (unsigned i = 0; i < n_query_vectors; i++) {
    for (unsigned j = 0; j < k; j++) {
      h_edges[i * k + j] = {i, mapping[h_frontiers[i * k + j].first],
                            h_frontiers[i * k + j].second};
    }
  }
  d_edges_t d_edges = h_edges;
  return d_edges;
}

}  // namespace gpu_ann