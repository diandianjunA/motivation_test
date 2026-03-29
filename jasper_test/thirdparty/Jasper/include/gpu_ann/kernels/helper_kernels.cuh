#ifndef ANN_HELPER_KERNELS
#define ANN_HELPER_KERNELS

#include <gpu_ann/vector.cuh>

namespace ann_kernels {

__global__ void generate_edge_stats_kernel(uint8_t *edge_counts,
                                           uint64_t n_vertices,
                                           uint64_t *statistics) {
  uint64_t tid = gallatin::utils::get_tid();

  if (tid >= n_vertices) return;

  atomicMin((unsigned long long int *)&statistics[0],
            (unsigned long long int)edge_counts[tid]);
  atomicAdd((unsigned long long int *)&statistics[1],
            (unsigned long long int)edge_counts[tid]);
  atomicMax((unsigned long long int *)&statistics[2],
            (unsigned long long int)edge_counts[tid]);
}

// calculate distances between all objects in edge kernel.
// this prunes dead edges
template <uint32_t tile_size, typename edge_pair_type, typename pq_vector_type,
          typename vertex_data_type, vertex_data_type dead_key,
          template <typename, typename, uint, uint> class distance_functor>
__global__ void edge_distance_kernel(edge_pair_type *edges,
                                     pq_vector_type *vectors, uint64_t n_edges,
                                     uint64_t offset) {
  auto thread_block = cg::this_thread_block();

  cg::thread_block_tile<tile_size> my_tile =
      cg::tiled_partition<tile_size>(thread_block);

  uint64_t tid = gallatin::utils::get_tile_tid(my_tile);

  if (tid >= n_edges) return;

  const uint vector_size = gpu_ann::data_vector_traits<pq_vector_type>::size;
  using vector_data_type =
      typename gpu_ann::data_vector_traits<pq_vector_type>::type;

  using distance_type = distance_functor<vector_data_type, vector_data_type,
                                         vector_size, tile_size>;

  auto source = edges[tid + offset].source;
  auto sink = edges[tid + offset].sink;

  if (source != dead_key && sink != dead_key) {
    auto dist =
        distance_type::distance(&vectors[source], &vectors[sink], my_tile);

    if (my_tile.thread_rank() == 0) edges[tid + offset].distance = dist;
  } else {
    // printf("Edge prune called?\n");
    if (my_tile.thread_rank() == 0) {
      edges[tid + offset].distance = 0.0;
      edges[tid + offset].source = dead_key;
      edges[tid + offset].sink = dead_key;
    }
  }

  return;
}

template <uint32_t tile_size, typename edge_pair_type, typename pq_vector_type,
          typename vertex_data_type, vertex_data_type dead_key,
          template <typename, typename, uint, uint> class distance_functor>
__global__ void edge_distance_kernel_dead(edge_pair_type *edges,
                                          pq_vector_type *vectors,
                                          uint64_t n_edges, uint64_t offset,
                                          edge_pair_type dead_edge,
                                          uint32_t workload_per_tile) {
  auto thread_block = cg::this_thread_block();

  cg::thread_block_tile<tile_size> my_tile =
      cg::tiled_partition<tile_size>(thread_block);

  uint64_t tid = gallatin::utils::get_tile_tid(my_tile);

  if (tid >= (n_edges / workload_per_tile) + 1) return;

  const uint vector_size = gpu_ann::data_vector_traits<pq_vector_type>::size;
  using vector_data_type =
      typename gpu_ann::data_vector_traits<pq_vector_type>::type;
  using distance_type = distance_functor<vector_data_type, vector_data_type,
                                         vector_size, tile_size>;

  for (uint i = 0; i < workload_per_tile; ++i) {
    uint edge_idx = offset + tid * workload_per_tile + i;

    if (edges[edge_idx] == dead_edge) return;

    auto source = edges[edge_idx].source;
    auto sink = edges[edge_idx].sink;

    auto dist =
        distance_type::distance(&vectors[source], &vectors[sink], my_tile);
    if (my_tile.thread_rank() == 0) edges[edge_idx].distance = dist;
  }
}

template <typename edge_pair_type, typename pq_vector_type,
          typename vertex_data_type, vertex_data_type dead_key>
__global__ void edge_distance_kernel_dead_query(
    edge_pair_type *__restrict__ edges, pq_vector_type *__restrict__ vectors,
    uint64_t n_edges, uint32_t edges_per_partition) {
  // const uint vector_size = gpu_ann::data_vector_traits<pq_vector_type>::size;
  using vector_data_type =
      typename gpu_ann::data_vector_traits<pq_vector_type>::type;
  using distance_lookup_singleton = gpu_error::gpu_singleton<float *>;
  float *distances = distance_lookup_singleton::instance();

  const int bid = blockIdx.x;
  // const int tid = blockIdx.x * blockDim.x + threadIdx.x;

  // load distance table to shared memory of this block
  const uint32_t num_centroids = 256;
  const uint32_t distance_table_size = 32 * 16 * 16;
  extern __shared__ float loaded_distance_table[];

  for (uint i = threadIdx.x; i < distance_table_size; i += blockDim.x) {
    loaded_distance_table[i] = distances[i];
  }
  __syncthreads();

  // calculate corresponding edge's distance
  const uint32_t edge_start = bid * edges_per_partition;
  const uint32_t edge_end =
      min(static_cast<uint32_t>(edge_start + edges_per_partition),
          static_cast<uint32_t>(n_edges));

  uint8_t source_vec[32];
  uint8_t sink_vec[32];
  for (uint i = edge_start + threadIdx.x; i < edge_end; i += blockDim.x) {
    const auto edge = edges[i];

#pragma unroll
    for (int j = 0; j < 32; ++j) {
      source_vec[j] = __ldg(&vectors[edge.source][j]);
      sink_vec[j] = __ldg(&vectors[edge.sink][j]);
    }

    float distance = 0;
#pragma unroll
    for (uint codebook = 0; codebook < 32; codebook++) {
      auto sourceCode = source_vec[codebook];
      auto sinkCode = sink_vec[codebook];
      if (sourceCode != dead_key && sinkCode != dead_key) {
        distance += loaded_distance_table[(codebook << 8) + (sourceCode << 4) +
                                          sinkCode];
      }
    }
    edges[i].distance = distance;
  }
  return;
}

template <typename edge_list_type, uint32_t R>
__global__ void populate_graph_kernel(edge_list_type *edges,
                                      uint8_t *edge_counts, uint64_t n_vectors,
                                      uint32_t *random_data) {
  uint64_t tid = gallatin::utils::get_tid();

  if (tid >= n_vectors) return;

  edge_counts[tid] = R;

  for (uint i = 0; i < R; i++) {
    edges[tid].edges[i] = (random_data[R * tid + i] % n_vectors);
  }
}

template <typename edge_list_type>
__global__ void ann_bfs_kernel(edge_list_type *edges, uint64_t n_vertices,
                               uint8_t *edge_counts, uint32_t *set_round,
                               uint32_t round, uint64_t *round_discovered,
                               uint64_t *total_discovered) {
  uint64_t tid = gallatin::utils::get_tid();

  if (tid >= n_vertices) return;

  if (set_round[tid] == round) {
    edge_list_type loaded_edges = edges[tid];

    for (uint i = 0; i < edge_counts[tid]; i++) {
      uint32_t edge = loaded_edges.edges[i];

      if (atomicCAS(&set_round[edge], 0, round + 1) == 0) {
        atomicAdd((unsigned long long int *)round_discovered,
                  (unsigned long long int)1);
        atomicAdd((unsigned long long int *)total_discovered,
                  (unsigned long long int)1);
      }
    }
  }
}

template <typename edge_list_type>
__global__ void print_medoid_kernel(edge_list_type *edges, uint8_t *edge_counts,
                                    uint32_t medoid) {
  uint64_t tid = gallatin::utils::get_tid();

  if (tid != 0) return;

  edge_list_type medoid_edges = edges[medoid];

  uint32_t count = edge_counts[medoid];

  printf("Medoid has %u outgoing edges\n", count);

  for (uint i = 0; i < count; i++) {
    printf("%u: %u -> %u\n", i, medoid, medoid_edges.edges[i]);
  }
}

}  // namespace ann_kernels

#endif