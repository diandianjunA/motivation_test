#ifndef ANN_PRUNE_KERNELS
#define ANN_PRUNE_KERNELS

#include <gpu_ann/edge_list.cuh>
#include <gpu_ann/vector.cuh>

namespace ann_kernels {

using namespace gpu_ann;

template <uint32_t tile_size, uint32_t n_tiles, typename vertex_data_type,
          typename distance_type, typename vector_data_type,
          uint32_t vector_degree,
          template <typename, typename, uint, uint> class distance_functor,
          typename edge_output_type, uint32_t R, bool can_merge>
__global__ void robust_prune_thrust(
    edge_list<vertex_data_type, R> *graph, uint8_t *edge_counts,
    data_vector<vector_data_type, vector_degree> *vertices,
    edge_output_type *accumulated_edges, edge_output_type dead_edge,
    uint32_t *starting_indices, uint32_t n_starting_indices, float alpha,
    uint64_t n_edges) {
  auto thread_block = cg::this_thread_block();

  cg::thread_block_tile<tile_size> my_tile =
      cg::tiled_partition<tile_size>(thread_block);

  uint64_t tile_tid = gallatin::utils::get_tile_tid(my_tile);

  uint shared_index = my_tile.meta_group_rank();

  if (tile_tid >= n_starting_indices) return;

  uint64_t tid = starting_indices[tile_tid];

  uint32_t source = accumulated_edges[tid].source;

  uint32_t start = tid;
  uint32_t end;

  if (tile_tid == n_starting_indices - 1) {
    end = n_edges;
  } else {
    end = starting_indices[tile_tid + 1];
  }

  // while (end < n_edges && accumulated_edges[end].source == source) {
  //   if (my_tile.thread_rank() == 0) {
  //     count_event(6);
  //   }

  //   end++;
  // }

  __threadfence();
  my_tile.sync();

  // use the prio queue for this.

  // load all edges into queue, then start dumping them out!

  using smem_edge_list_type = edge_list<vertex_data_type, R>;

  using dist_type = distance_functor<vector_data_type, vector_data_type,
                                     vector_degree, tile_size>;

  __shared__ smem_edge_list_type local_edges[n_tiles];

  uint32_t write_idx = 0;

  if (end - start < R && can_merge) {
    // map to parlayANN `add_neighbors_without_repeats`
    //  https://github.com/cmuparlay/ParlayANN/blob/393188145dfdf432092624de16a7c0ed15b0f06d/algorithms/vamana/index.h#L140C3-L146C4
    if (my_tile.thread_rank() == 0) {
      while (start < end) {
        auto edge = accumulated_edges[start];

        vertex_data_type p_star = edge.sink;

        if (p_star == source) {
          start++;
          continue;
        }

        local_edges[shared_index].edges[write_idx] = p_star;

        start++;

        write_idx++;
      }

      graph[source] = local_edges[shared_index];
      edge_counts[source] = write_idx;

      gpu_assert(write_idx < R, "Bad write output\n");
    }

    my_tile.sync();

    return;

  } else {
    while (start < end && write_idx < R) {
      auto edge = accumulated_edges[start];

      start++;

      vertex_data_type p_star = edge.sink;

      if (edge == dead_edge || p_star == source) {
        continue;
      }

      local_edges[shared_index].edges[write_idx] = p_star;

      write_idx++;

      if (write_idx == R) {
        break;
      }

      // else get dist
      auto dist_p_p_star = edge.distance;

      // iterate over next keys.
      for (uint i = start; i < end; i++) {
        auto next_edge = accumulated_edges[i];

        if (next_edge == dead_edge) {
          my_tile.sync();
          continue;
        }

        vertex_data_type p_prime = next_edge.sink;

        // else live for consideration.

        auto disk_p_star_p_prime =
            dist_type::distance(vertices[p_star], vertices[p_prime], my_tile);

        auto dist_p_p_prime = next_edge.distance;

        if (alpha * disk_p_star_p_prime <= dist_p_p_prime) {
          accumulated_edges[i] = dead_edge;
        }
      }
    }

    // final write.

    __threadfence();
    my_tile.sync();

    if (my_tile.thread_rank() == 0) {
      graph[source] = local_edges[shared_index];

      edge_counts[source] = write_idx;
    }
  }
}

// robust prune thrust kernel
//  modified to use one thread block per vertex
template <uint32_t tile_size, typename vertex_data_type, typename distance_type,
          typename vector_data_type, uint32_t vector_degree,
          template <typename, typename, uint, uint> class distance_functor,
          typename edge_output_type, uint32_t R, bool can_merge>
__global__ void robust_prune_block(
    edge_list<vertex_data_type, R> *graph, uint8_t *edge_counts,
    data_vector<vector_data_type, vector_degree> *vertices,
    edge_output_type *accumulated_edges, edge_output_type dead_edge,
    uint32_t *starting_indices, uint32_t n_starting_indices, double alpha,
    uint64_t n_edges) {
  namespace cg = cooperative_groups;
  cg::thread_block cta = cg::this_thread_block();
  cg::thread_block_tile<tile_size> my_tile =
      cg::tiled_partition<tile_size>(cta);

  // warp/lane_id are per-thread objects
  const uint32_t warp_id = threadIdx.x / tile_size;
  const uint32_t block_id = blockIdx.x;
  const uint32_t threads_per_block = blockDim.x;
  const uint32_t warps_per_block = blockDim.x / tile_size;

  if (block_id >= n_starting_indices) return;

  uint64_t tid = starting_indices[block_id];
  uint32_t source = accumulated_edges[tid].source;

  uint32_t start = tid;
  uint32_t end = (block_id == n_starting_indices - 1)
                     ? n_edges
                     : starting_indices[block_id + 1];

  // at this point all threads know which key range / position in thread block.

  // need to start iterating.
  using smem_edge_list_type = edge_list<vertex_data_type, R>;
  using dist_type = distance_functor<vector_data_type, vector_data_type,
                                     vector_degree, tile_size>;

  __shared__ smem_edge_list_type local_edges;
  __shared__ uint32_t write_idx;

  __shared__ bool is_valid[256];

  if (threadIdx.x == 0) {
    write_idx = 0;
  }

  if (end - start >= 256) {
    end = start + 256;
  }

  for (uint32_t i = threadIdx.x; i < (end - start); i += threads_per_block) {
    is_valid[i] = true;
  }

  __threadfence();
  cta.sync();

  if (end - start < R && can_merge) {
    // map to parlayANN `add_neighbors_without_repeats`
    //  https://github.com/cmuparlay/ParlayANN/blob/393188145dfdf432092624de16a7c0ed15b0f06d/algorithms/vamana/index.h#L140C3-L146C4
    if (threadIdx.x == 0) {
      while (start < end) {
        auto edge = accumulated_edges[start];

        vertex_data_type p_star = edge.sink;

        if (p_star == source) {
          start++;
          continue;
        }

        local_edges.edges[write_idx] = p_star;

        start++;

        write_idx++;
      }

      graph[source] = local_edges;
      edge_counts[source] = write_idx;

      gpu_assert(write_idx < R, "Bad write output\n");
    }

    __threadfence();
    cta.sync();
    return;

  } else {
    // now iterate

    uint32_t original = start;

    while (start < end && write_idx < R) {
      uint32_t index = start - original;

      // loop.
      if (!is_valid[index]) {
        start++;

        __threadfence();
        cta.sync();
        continue;
      }

      auto working_edge = accumulated_edges[start];

      vertex_data_type p_star = working_edge.sink;

      if (p_star == source){
        accumulated_edges[start] = dead_edge;
        start++;
        __threadfence();
        cta.sync();
        continue;
      }

      // add edge.
      if (threadIdx.x == 0) {
        local_edges.edges[write_idx] = p_star;
        write_idx++;
      }

      start++;

      for (uint32_t i = start + warp_id; i < end; i += warps_per_block) {
        if (!is_valid[i - original]) continue;

        auto next_edge = accumulated_edges[i];

        vertex_data_type p_prime = next_edge.sink;

        auto disk_p_star_p_prime =
            dist_type::distance(&vertices[p_star], &vertices[p_prime], my_tile);

        auto dist_p_p_prime = next_edge.distance;

        if (alpha * disk_p_star_p_prime <= dist_p_p_prime) {
          if (my_tile.thread_rank() == 0) {
            accumulated_edges[i] = dead_edge;
            is_valid[i - original] = false;
          }
        }
      }

      __threadfence();
      cta.sync();
    }

    for (uint32_t i = start+threadIdx.x; i < end; i+= blockDim.x){
      accumulated_edges[i] = dead_edge;
    }

    // if (start < end){
    //   printf("Bad edges being passed!\n");
    // }

    // Final write
    if (threadIdx.x == 0) {
      graph[source] = local_edges;
      edge_counts[source] = write_idx;
      
    }
  }
}

// variant of robust prune
// uses old edge counts to more efficiently process
template <uint32_t tile_size, typename vertex_data_type, typename distance_type,
          typename vector_data_type, uint32_t vector_degree,
          template <typename, typename, uint, uint> class distance_functor,
          typename edge_output_type, uint32_t R, bool can_merge>
__global__ void robust_prune_old_edges(
    edge_list<vertex_data_type, R> *graph, uint8_t *edge_counts,
    data_vector<vector_data_type, vector_degree> *vertices,
    edge_output_type *accumulated_edges, edge_output_type dead_edge,
    uint32_t *starting_indices, uint32_t n_starting_indices, double alpha,
    uint64_t n_edges) {
  namespace cg = cooperative_groups;
  cg::thread_block cta = cg::this_thread_block();
  cg::thread_block_tile<tile_size> my_tile =
      cg::tiled_partition<tile_size>(cta);

  // warp/lane_id are per-thread objects
  const uint32_t warp_id = threadIdx.x / tile_size;
  const uint32_t block_id = blockIdx.x;
  const uint32_t threads_per_block = blockDim.x;
  const uint32_t warps_per_block = blockDim.x / tile_size;

  if (block_id >= n_starting_indices) return;

  uint64_t tid = starting_indices[block_id];
  uint32_t source = accumulated_edges[tid].source;

  uint32_t start = tid;
  uint32_t end = (block_id == n_starting_indices - 1)
                     ? n_edges
                     : starting_indices[block_id + 1];

  // at this point all threads know which key range / position in thread block.

  // need to start iterating.
  using smem_edge_list_type = edge_list<vertex_data_type, R>;
  using dist_type = distance_functor<vector_data_type, vector_data_type,
                                     vector_degree, tile_size>;

  __shared__ smem_edge_list_type local_edges;
  __shared__ uint32_t write_idx;

  __shared__ bool is_valid[256];

  if (threadIdx.x == 0) {
    write_idx = 0;
  }

  if (end - start >= 256) {
    end = start + 256;
  }

  for (uint32_t i = threadIdx.x; i < (end - start); i += threads_per_block) {
    is_valid[i] = true;
  }

  __threadfence();
  cta.sync();

  if (end - start < R && can_merge) {
    // VARIANT
    //  must merge old and new edges together.
    //  direct append

    // map to parlayANN `add_neighbors_without_repeats`
    //  https://github.com/cmuparlay/ParlayANN/blob/393188145dfdf432092624de16a7c0ed15b0f06d/algorithms/vamana/index.h#L140C3-L146C4
    if (threadIdx.x == 0) {
      local_edges = graph[source];

      uint32_t write_idx = edge_counts[source];

      while (start < end) {
        auto edge = accumulated_edges[start];

        vertex_data_type p_star = edge.sink;

        if (p_star == source) {
          start++;
          continue;
        }

        local_edges.edges[write_idx] = p_star;

        start++;

        write_idx++;
      }

      graph[source] = local_edges;
      edge_counts[source] = write_idx;

      gpu_assert(write_idx <= R, "Bad write output\n");
    }

    return;

  } else {
    // now iterate

    uint32_t original = start;

    while (start < end && write_idx < R) {
      uint32_t index = start - original;

      // loop.
      if (!is_valid[index]) {
        start++;

        __threadfence();
        cta.sync();
        continue;
      }

      auto working_edge = accumulated_edges[start];

      vertex_data_type p_star = working_edge.sink;

      if (p_star == source) {
          start++;
          continue;
      }

      // add edge.
      if (threadIdx.x == 0) {
        local_edges.edges[write_idx] = p_star;
        write_idx++;
      }

      start++;

      for (uint32_t i = start + warp_id; i < end; i += warps_per_block) {
        if (!is_valid[i - original]) continue;

        auto next_edge = accumulated_edges[i];

        vertex_data_type p_prime = next_edge.sink;

        //prevent duplicate edges from being admitted.
        if (p_prime == p_star){
          if (my_tile.thread_rank() == 0){
            accumulated_edges[i] = dead_edge;
            is_valid[i - original] = false;
          }
          //printf("Double edge added!\n");
        }


        auto disk_p_star_p_prime =
            dist_type::distance(&vertices[p_star], &vertices[p_prime], my_tile);

        auto dist_p_p_prime = next_edge.distance;

        if (alpha * disk_p_star_p_prime <= dist_p_p_prime) {
          if (my_tile.thread_rank() == 0) {
            accumulated_edges[i] = dead_edge;
            is_valid[i - original] = false;
          }
        }
      }

      __threadfence();
      cta.sync();
    }

    for (uint32_t i = start+threadIdx.x; i < end; i+= blockDim.x){
      accumulated_edges[i] = dead_edge;
    }

    // Final write
    if (threadIdx.x == 0) {
      graph[source] = local_edges;
      edge_counts[source] = write_idx;
    }
  }
}

// helper kernel to accumulate # of existing edges in graph
// needed for robustPrune Merge
template <typename edge_pair_type, typename edge_list_type,
          typename vertex_data_type, vertex_data_type dead_key>
__global__ void count_existing_edges_kernel(edge_pair_type *edges,
                                            edge_list_type *graph,
                                            uint8_t *vertex_degrees,
                                            uint64_t n_existing_edges,
                                            uint64_t *edge_count_tracker) {
  uint64_t tid = gallatin::utils::get_tid();

  if (tid >= n_existing_edges) return;

  // 1. kill threads that point to dead keys in *either* direction
  if (edges[tid].source == dead_key || edges[tid].sink == dead_key) {
    asm volatile("trap;");
    return;
  }

  bool first = (tid == 0) || (edges[tid - 1].source != edges[tid].source);

  if (!first) return;

  vertex_data_type source = edges[tid].source;

  uint64_t edge_count = vertex_degrees[source];

  atomicAdd((unsigned long long int *)&edge_count_tracker[0],
            (unsigned long long int)edge_count);
}

template <typename edge_pair_type, uint R>
__global__ void count_existing_edges_kernel_with_count(
    edge_pair_type *unique_edges, uint64_t n_unique_edges,
    uint32_t *new_edge_counts, uint32_t *old_edge_counts,
    uint32_t *edge_count_tracker) {
  uint64_t tid = gallatin::utils::get_tid();

  if (tid >= n_unique_edges) return;

  uint32_t new_count = new_edge_counts[tid];
  uint32_t old_count = old_edge_counts[tid];

  if (new_count + old_count > R) {
    atomicAdd((unsigned int *)edge_count_tracker, (unsigned int)old_count);
  }
}

// write out edges
//  space has been reserved in underlying thrust vector
// distance calculation is called later.
template <uint32_t tile_size, uint32_t n_tiles, typename edge_pair_type,
          typename edge_list_type, typename vector_type,
          typename vertex_data_type, vertex_data_type dead_key>
__global__ void add_existing_edges_kernel(edge_pair_type *edges,
                                          vector_type *vectors,
                                          edge_list_type *graph,
                                          uint8_t *edge_counts,
                                          uint64_t n_existing_edges,
                                          uint64_t *edge_count_tracker) {
  auto thread_block = cg::this_thread_block();
  cg::thread_block_tile<tile_size> my_tile =
      cg::tiled_partition<tile_size>(thread_block);
  uint64_t tid = gallatin::utils::get_tile_tid(my_tile);

  if (tid >= n_existing_edges) return;

  // 1. kill threads that point to dead keys
  if (edges[tid].source == dead_key || edges[tid].sink == dead_key) return;
  bool first = (tid == 0) || (edges[tid - 1].source != edges[tid].source);
  if (!first) return;

  // uint shared_index = my_tile.meta_group_rank();
  // __shared__ edge_list_type local_edge_lists[n_tiles];

  vertex_data_type source = edges[tid].source;
  uint64_t output_count = edge_counts[source];

  uint64_t start_output_idx;
  if (my_tile.thread_rank() == 0) {
    //local_edge_list[shared_index] = graph[source];
    start_output_idx = atomicAdd(
        (unsigned long long int *)&edge_count_tracker[0], output_count);
  }

  __threadfence();

  my_tile.sync();

  start_output_idx = my_tile.shfl(start_output_idx, 0);
  for (uint i = my_tile.thread_rank(); i < output_count; i += tile_size) {
    edges[n_existing_edges + start_output_idx + i] = {
        //source, local_edge_lists[shared_index].edges[i]};
        source, graph[source].edges[i]};
  }

  my_tile.sync();
}

template <uint32_t tile_size, uint32_t n_tiles, typename edge_pair_type,
          typename edge_list_type, uint R>
__global__ void add_existing_edges_kernel_with_count(
    edge_pair_type *new_edges, edge_pair_type *unique_edges,
    uint64_t unique_edges_size, edge_list_type *graph, uint32_t *new_counts,
    uint32_t *old_counts, uint64_t n_existing_edges,
    uint32_t *edge_count_tracker) {
  auto thread_block = cg::this_thread_block();

  cg::thread_block_tile<tile_size> my_tile =
      cg::tiled_partition<tile_size>(thread_block);

  uint64_t tid = gallatin::utils::get_tile_tid(my_tile);

  if (tid >= unique_edges_size) return;

  uint shared_index = my_tile.meta_group_rank();

  __shared__ edge_list_type local_edge_lists[n_tiles];

  auto source = unique_edges[tid].source;

  uint32_t output_count = old_counts[tid];

  // keys with fulfilled counts don't bother merging.
  if (output_count + new_counts[tid] <= R) return;

  uint64_t start_output_idx;

  if (my_tile.thread_rank() == 0) {
    local_edge_lists[shared_index] = graph[source];

    start_output_idx =
        atomicAdd((unsigned int *)&edge_count_tracker[0], output_count);
  }

  __threadfence();

  my_tile.sync();

  start_output_idx = my_tile.shfl(start_output_idx, 0);

  for (uint i = my_tile.thread_rank(); i < output_count; i += tile_size) {
    new_edges[n_existing_edges + start_output_idx + i] = {
        source, local_edge_lists[shared_index].edges[i]};
  }

  my_tile.sync();
}

// helper kernel to swap over the edges
template <typename edge_type>
__global__ void flip_edges_kernel(edge_type *edges, uint64_t n_edges) {
  uint64_t tid = gallatin::utils::get_tid();

  if (tid >= n_edges) return;

  auto temp = edges[tid].source;

  edges[tid].source = edges[tid].sink;

  edges[tid].sink = temp;
}

// count # of unique edges in vector
template <typename edge_pair_type>
__global__ void count_unique_kernel(uint32_t *unique_count,
                                    edge_pair_type *edges, uint64_t size) {
  uint64_t tid = gallatin::utils::get_tid();

  if (tid >= size) return;

  bool first = (tid == 0) || (edges[tid - 1].source != edges[tid].source);

  if (first) {
    atomicAdd(unique_count, 1U);
  }
}

template <typename edge_pair_type>
__global__ void populate_old_counts(edge_pair_type *edges, uint32_t *counts,
                                    uint8_t *existing_edge_counts,
                                    uint64_t size) {
  uint64_t tid = gallatin::utils::get_tid();

  if (tid >= size) return;

  counts[tid] = existing_edge_counts[edges[tid].source];
}

}  // namespace ann_kernels

#endif