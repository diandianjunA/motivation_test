#ifndef ANN_BEAM_KERNELS
#define ANN_BEAM_KERNELS

#include <cooperative_groups.h>
#include <gpu_ann/cg_compat.cuh>
#include <gpu_ann/vector.cuh>

namespace cg = cooperative_groups;

namespace ann_kernels {

// calculate the new frontier
// this selects BEAM_WIDTH
template <typename edge_pair_type>
__global__ void select_new_frontier(edge_pair_type *frontier,
                                    edge_pair_type *live_frontier,
                                    uint64_t frontier_size,
                                    uint64_t *live_frontier_count) {
  uint64_t tid = gallatin::utils::get_tid();

  if (tid >= frontier_size) return;

  bool is_frontier =
      (tid == 0 || frontier[tid - 1].source != frontier[tid].source);

  if (!is_frontier) return;

  uint64_t addr =
      atomicAdd((unsigned long long int *)&live_frontier_count[0], 1ULL);

  live_frontier[addr] = frontier[tid];
}

template <typename edge_pair_type>
__global__ void find_beam_start_kernel(edge_pair_type *frontier,
                                       uint64_t nodes_in_frontier,
                                       uint64_t *start_addrs,
                                       uint64_t node_start_idx) {
  uint64_t tid = gallatin::utils::get_tid();
  if (tid >= nodes_in_frontier) return;

  bool is_frontier =
      (tid == 0 || frontier[tid - 1].source != frontier[tid].source);

  if (!is_frontier) return;

  uint64_t my_write_addr = frontier[tid].source - node_start_idx;

  start_addrs[my_write_addr] = tid;
}

template <typename edge_pair_type>
__global__ void find_beam_start_kernel_u32(edge_pair_type *frontier,
                                           uint64_t nodes_in_frontier,
                                           uint32_t *start_addrs,
                                           uint32_t node_start_idx) {
  uint64_t tid = gallatin::utils::get_tid();

  if (tid >= nodes_in_frontier) return;

  bool is_frontier =
      (tid == 0 || frontier[tid - 1].source != frontier[tid].source);

  if (!is_frontier) return;

  uint64_t my_write_addr = frontier[tid].source - node_start_idx;

  start_addrs[my_write_addr] = tid;
}

template <typename edge_pair_type>
__global__ void count_frontier_kernel(edge_pair_type *frontier,
                                      uint64_t nodes_in_frontier,
                                      uint64_t *sizes,
                                      uint64_t node_start_idx) {
  uint64_t tid = gallatin::utils::get_tid();

  if (tid >= nodes_in_frontier) return;

  uint64_t my_write_addr = frontier[tid].source - node_start_idx;

  atomicAdd((unsigned long long int *)&sizes[my_write_addr], 1ULL);
}

__device__ float atomicMinFloat(float *address, float val) {
  int *address_as_int = (int *)address;
  int old = *address_as_int, assumed;

  do {
    assumed = old;
    if (__int_as_float(assumed) <= val) break;
    old = atomicCAS(address_as_int, assumed, __float_as_int(val));
  } while (assumed != old);

  return __int_as_float(old);
}

template <typename edge_pair_type>
__global__ void clip_beam_kernel(edge_pair_type *frontier,
                                 uint64_t nodes_in_frontier,
                                 uint64_t *start_addrs, uint64_t node_start_idx,
                                 uint64_t *n_cut, uint64_t beam_width,
                                 edge_pair_type dead_node) {
  uint64_t tid = gallatin::utils::get_tid();

  if (tid >= nodes_in_frontier) return;

  uint64_t my_write_addr = frontier[tid].source - node_start_idx;

  // if (closest_boundary[my_write_addr] < frontier[tid].distance){
  //    frontier[tid] = dead_node;
  //    atomicAdd((unsigned long long int *)&n_cut[0], 1ULL);
  // }

  if (tid - start_addrs[my_write_addr] >= beam_width) {
    // atomicMin((int *)&closest_boundary[my_write_addr],
    //__float_as_int(frontier[tid].distance));

    frontier[tid] = dead_node;

    atomicAdd((unsigned long long int *)&n_cut[0], 1ULL);
  }
}

template <typename edge_pair_type, typename dist_type>
__global__ void frontier_cutoff_kernel(edge_pair_type *frontier,
                                       uint64_t frontier_size, uint64_t start,
                                       dist_type *worst_distances,
                                       uint32_t beam_width) {
  uint64_t tid = gallatin::utils::get_tid();

  if (tid >= frontier_size) return;

  bool is_first =
      (tid == 0 || frontier[tid - 1].source != frontier[tid].source);

  if (!is_first) return;

  auto source = frontier[tid].source;

  uint64_t next = tid + beam_width - 1;

  if (next >= frontier_size || (frontier[next].source != source)) {
    worst_distances[source - start] =
        cuda::std::numeric_limits<dist_type>::max();
  } else {
    worst_distances[source - start] = frontier[next].distance;
  }
}

template <typename edge_pair_type, typename dist_type>
__global__ void prune_candidates_cutoff_kernel(edge_pair_type *candidates,
                                               uint64_t candidates_size,
                                               uint64_t start,
                                               dist_type *worst_distances,
                                               const edge_pair_type dead_edge) {
  uint64_t tid = gallatin::utils::get_tid();

  if (tid >= candidates_size) return;

  auto source = candidates[tid].source;

  if (candidates[tid].distance >= worst_distances[source - start]) {
    candidates[tid] = dead_edge;
  }
}

template <typename edge_pair_type, typename dist_type>
__global__ void prune_candidates_cutoff_kernel_dead(
    edge_pair_type *candidates, uint64_t candidates_size, uint64_t start,
    dist_type *worst_distances, const edge_pair_type dead_edge) {
  uint64_t tid = gallatin::utils::get_tid();

  if (tid >= candidates_size) return;

  if (candidates[tid] == dead_edge) return;

  auto source = candidates[tid].source;

  if (candidates[tid].distance >= worst_distances[source - start]) {
    candidates[tid] = dead_edge;
  }
}

// frontier has been gathered, read all simultaneously.
template <uint tile_size, typename edge_list_type, typename edge_pair_type,
          typename vertex_data_type>
__global__ void populate_all_candidates_kernel(
    edge_list_type *graph, uint8_t *edge_counts, edge_pair_type *frontier,
    uint64_t frontier_count, edge_pair_type *candidates, uint64_t *n_candidates,
    uint64_t max_candidate_size) {
  auto thread_block = cg::this_thread_block();

  cg::thread_block_tile<tile_size> my_tile =
      cg::tiled_partition<tile_size>(thread_block);

  uint64_t tid = gallatin::utils::get_tile_tid(my_tile);

  if (tid >= frontier_count) return;

  // source in frontier is current node (to make parallel expansion easier)
  vertex_data_type next = frontier[tid].sink;

  uint64_t n_edges = edge_counts[next];

  // one thread in team populates candidate values
  //  (this densely packs candidates into array even when live frontier is
  //  shrinking)
  uint64_t offset = invoke_one_broadcast_compat(my_tile, [&]() {
    return atomicAdd((unsigned long long int *)&n_candidates[0],
                     (unsigned long long int)n_edges);
  });

  for (uint i = my_tile.thread_rank(); i < n_edges; i += tile_size) {
    vertex_data_type node_to_add = graph[next].edges[i];

    candidates[offset + i] = {frontier[tid].source, node_to_add, 0.0};
  }
}

// kernel to populate the candidates based on a static frontier
// with width passed in.
//  fills unused slots with dead_edge.
template <uint tile_size, uint n_tiles, typename edge_list_type,
          typename edge_pair_type, typename vertex_data_type>
__global__ void populate_candidates_width_kernel(
    edge_list_type *graph, uint8_t *edge_counts, edge_pair_type *frontier,
    uint64_t frontier_size, edge_pair_type *candidates,
    edge_pair_type *live_frontier, uint32_t *starting_indices,
    uint32_t *candidate_counter, uint32_t *live_frontier_counter, uint32_t R,
    uint32_t L_cap, uint32_t width, uint32_t start, uint32_t n_vectors_in_batch,
    edge_pair_type dead_edge) {
  auto thread_block = cg::this_thread_block();

  cg::thread_block_tile<tile_size> my_tile =
      cg::tiled_partition<tile_size>(thread_block);

  uint64_t tid = gallatin::utils::get_tile_tid(my_tile);

  __shared__ edge_list_type local_edges[n_tiles];

  uint shared_index = my_tile.meta_group_rank();

  if (tid >= width * n_vectors_in_batch) return;

  uint64_t my_width = tid % width;

  uint32_t lookup_id = (tid / width);

  uint32_t my_source = lookup_id + start;

  uint32_t start_index = starting_indices[lookup_id];

  if (start_index + my_width >= frontier_size) {
    return;
  }

  vertex_data_type loaded_source = frontier[start_index + my_width].source;

  if (loaded_source != my_source) return;

  if (my_tile.thread_rank() == 0) {
    uint32_t frontier_offset =
        atomicAdd((unsigned int *)&live_frontier_counter[0], (unsigned int)1U);
    live_frontier[frontier_offset] = frontier[start_index + my_width];
  }

  // source in frontier is current node (to make parallel expansion easier)
  vertex_data_type next = frontier[start_index + my_width].sink;

  uint64_t n_edges = edge_counts[next];

  uint32_t offset = invoke_one_broadcast_compat(my_tile, [&]() {
    return atomicAdd((unsigned int *)&candidate_counter[0],
                     (unsigned int)n_edges);
  });

  if (my_tile.thread_rank() == 0) {
    local_edges[shared_index] = graph[next];
  }

  my_tile.sync();
  // edge_list_type local_edges = invoke_one_broadcast_compat(my_tile, [&]() {
  //   return graph[next];
  // });

  for (uint i = my_tile.thread_rank(); i < n_edges; i += tile_size) {
    vertex_data_type node_to_add = local_edges[shared_index].edges[i];

    candidates[offset + i] = {my_source, node_to_add, 0.0};
  }
}

template <uint tile_size, uint n_tiles, typename edge_list_type,
          typename edge_pair_type, typename vertex_data_type>
__global__ void populate_candidates_prefix_sum(
    edge_list_type *graph, uint8_t *edge_counts, edge_pair_type *frontier,
    uint64_t frontier_size, edge_pair_type *candidates,
    edge_pair_type *live_frontier, uint32_t *starting_indices,
    uint32_t *candidate_counters, uint32_t *live_frontier_counters, uint32_t R,
    uint32_t L_cap, uint32_t width, uint32_t start, uint32_t n_vectors_in_batch,
    edge_pair_type dead_edge) {
  auto thread_block = cg::this_thread_block();
  cg::thread_block_tile<tile_size> my_tile =
      cg::tiled_partition<tile_size>(thread_block);
  uint64_t tid = gallatin::utils::get_tile_tid(my_tile);

  __shared__ edge_list_type local_edges[n_tiles];
  uint shared_index = my_tile.meta_group_rank();
  if (tid >= width * n_vectors_in_batch) return;

  uint64_t my_width = tid % width;
  uint32_t lookup_id = (tid / width);
  uint32_t my_source = lookup_id + start;
  uint32_t start_index = starting_indices[lookup_id];

  if (start_index + my_width >= frontier_size) {
    return;
  }

  vertex_data_type loaded_source = frontier[start_index + my_width].source;

  if (loaded_source != my_source) return;

  if (my_tile.thread_rank() == 0) {
    uint32_t frontier_offset = atomicAdd(
        (unsigned int *)&live_frontier_counters[lookup_id], (unsigned int)1U);
    live_frontier[frontier_offset] = frontier[start_index + my_width];
  }

  // source in frontier is current node (to make parallel expansion easier)
  vertex_data_type next = frontier[start_index + my_width].sink;

  uint64_t n_edges = edge_counts[next];

  uint32_t offset = invoke_one_broadcast_compat(my_tile, [&]() {
    return atomicAdd((unsigned int *)&candidate_counters[lookup_id],
                     (unsigned int)n_edges);
  });

  if (my_tile.thread_rank() == 0) {
    local_edges[shared_index] = graph[next];
  }

  my_tile.sync();
  // edge_list_type local_edges = invoke_one_broadcast_compat(my_tile, [&]() {
  //   return graph[next];
  // });

  for (uint i = my_tile.thread_rank(); i < n_edges; i += tile_size) {
    vertex_data_type node_to_add = local_edges[shared_index].edges[i];

    candidates[offset + i] = {my_source, node_to_add, 0.0};
  }
}

// kernel to populate the candidates based on a static frontier
// with width passed in.
//  fills unused slots with dead_edge.
template <typename edge_list_type, typename edge_pair_type>
__global__ void find_candidate_bounds(
    edge_list_type *graph, uint8_t *edge_counts, edge_pair_type *live_frontier,
    uint64_t frontier_size, uint32_t *starting_indices,
    uint32_t *unvisited_frontier_counter, uint32_t *live_frontier_counts,
    uint32_t width, uint32_t start, uint32_t n_vectors_in_batch) {
  uint64_t tid = gallatin::utils::get_tid();

  if (tid >= n_vectors_in_batch) return;

  uint32_t my_source = tid + start;

  uint32_t my_start = starting_indices[tid];

  uint32_t my_end = my_start + width;

  uint32_t sum = 0;

  uint32_t n_width = 0;

  while (my_start < my_end && live_frontier[my_start].source == my_source &&
         my_start < frontier_size) {
    sum += edge_counts[live_frontier[my_start].sink];

    my_start++;

    n_width++;
  }

  unvisited_frontier_counter[tid] = sum;
  live_frontier_counts[tid] = n_width;
}

template <typename edge_pair_type>
__global__ void calculate_accuracy_kernel(edge_pair_type *query_results_ptr,
                                          edge_pair_type *ground_truth,
                                          uint64_t n_query_vectors,
                                          uint64_t *accuracy_count,
                                          uint32_t k) {
  uint64_t tid = gallatin::utils::get_tid();

  if (tid >= (n_query_vectors * k)) return;

  // outgoing edge is the result
  auto my_value = query_results_ptr[tid].sink;

  // floor
  uint64_t my_query_id = (tid / k) * k;

  for (uint i = 0; i < k; i++) {
    if (ground_truth[my_query_id + i].sink == my_value) {
      atomicAdd((unsigned long long int *)accuracy_count, 1ULL);
      return;
    }
  }
}

// populate vector with vertices for bulk distance calculations.
template <typename vertex_data_type, typename edge_pair_type>
__global__ void populate_edges_from_vector_kernel(
    edge_pair_type *distances_ptr, vertex_data_type query_start,
    vertex_data_type n_query_vectors, vertex_data_type start,
    vertex_data_type end) {
  uint64_t tid = gallatin::utils::get_tid();

  if (tid >= (n_query_vectors) * (end - start)) return;

  vertex_data_type regular_id = tid / n_query_vectors;

  vertex_data_type query_vector_id = tid % n_query_vectors;

  distances_ptr[tid] = {query_vector_id + query_start, regular_id + start, 0.0};
}

}  // namespace ann_kernels

#endif