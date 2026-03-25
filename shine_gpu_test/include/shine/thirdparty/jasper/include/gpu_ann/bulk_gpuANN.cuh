#ifndef GPU_GRAPH
#define GPU_GRAPH

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cooperative_groups/scan.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/memory.h>
#include <thrust/merge.h>
#include <thrust/remove.h>
#include <thrust/set_operations.h>
#include <thrust/unique.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <gallatin/allocators/alloc_utils.cuh>
#include <gallatin/data_structs/ds_utils.cuh>
#include <gpu_ann/_proj_metadata.cuh>
#include <gpu_ann/beam_search.cuh>
#include <gpu_ann/edge_list.cuh>
#include <gpu_ann/edge_sorting.cuh>
#include <gpu_ann/hash_tables/linear_table.cuh>
#include <gpu_ann/kernels/beam_search_kernels.cuh>
#include <gpu_ann/kernels/helper_kernels.cuh>
#include <gpu_ann/kernels/prune_kernels.cuh>
#include <gpu_ann/priority_queue.cuh>
#include <gpu_ann/prune.cuh>
#include <gpu_ann/vector.cuh>
#include <gallatin/allocators/timer.cuh>
#include <gpu_error/progress_bar.cuh>
#include <gpu_error/timer.cuh>
#include <iostream>
#include <limits>
#include <sstream>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include "assert.h"
#include "stdio.h"

namespace cg = cooperative_groups;

const uint16_t EDGE_DISTANCE_TILE_SIZE = 4;
const uint16_t EDGE_DISTANCE_LOAD_PER_TILE = 1;

#define MEASURE_BATCH_TIME 1

#define THRUST_SAFE_CALL(thrust_expr)                                \
    try {                                                             \
        thrust_expr;                                                  \
    } catch (thrust::system_error &e) {                               \
        std::cerr << "Thrust error: " << e.what() << std::endl;       \
        std::cerr << "CUDA error code: " << e.code().value() << std::endl; \
        std::exit(EXIT_FAILURE);                                      \
    } catch (std::exception &e) {                                     \
        std::cerr << "Standard exception: " << e.what() << std::endl; \
        std::exit(EXIT_FAILURE);                                      \
    } catch (...) {                                                   \
        std::cerr << "Unknown error during Thrust call." << std::endl; \
        std::exit(EXIT_FAILURE);                                      \
    }

#define CUDA_CHECK(call)                                                      \
    {                                                                         \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__      \
                      << " code=" << err << " \"" << cudaGetErrorString(err) \
                      << "\"\n";                                              \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    }

namespace gpu_ann {

void checkGpuMem(){
  size_t free_bytes, total_bytes;

  cudaError_t err = cudaMemGetInfo(&free_bytes, &total_bytes);
  if (err != cudaSuccess) {
      fprintf(stderr, "cudaMemGetInfo failed: %s\n", cudaGetErrorString(err));
  }

  const double MB = 1024.0 * 1024.0;
  double free_mb  = free_bytes  / MB;
  double total_mb = total_bytes / MB;
  double used_mb  = total_mb - free_mb;

  printf("GPU Memory:");
  printf(" Total: %.2f MB", total_mb);
  printf(" Free : %.2f MB", free_mb);
  printf(" Used : %.2f MB\n", used_mb);
}

inline std::string format_progress_duration(std::chrono::steady_clock::duration duration) {
  const auto total_seconds =
      std::chrono::duration_cast<std::chrono::seconds>(duration).count();
  const auto hours = total_seconds / 3600;
  const auto minutes = (total_seconds % 3600) / 60;
  const auto seconds = total_seconds % 60;

  std::ostringstream os;
  if (hours > 0) {
    os << hours << "h" << minutes << "m" << seconds << "s";
  } else if (minutes > 0) {
    os << minutes << "m" << seconds << "s";
  } else {
    os << seconds << "s";
  }
  return os.str();
}

// bulk_gpuANN is the first modified version of the diskANN construction
// algorithm this uses thrust sort:ng over distances to efficiently compute
// neighbors for robustPrune.
//
// Template arguments:
//   - tile_size
//   - batch_size
//   - vertex_data_type: typename for vertex index
//   - distance_type: typename for distance
//   - vector_data_type: typename for each vector field
//   - vector_degree: degree of each vector
//   - pq_data_type: typename for each quantized vector field
//   - pq_degree: degree of each quantized vector
//   - R: Upper bound of maximum outgoing edges in the final graph
//   - L_cap:
//   - full_distance_functor: distance function for full vectors.
//   - pq_distance_functor: distance function for pq vectors using lookup table.
//   - on_host: whether the edge lists are stored on host or device

template <uint tile_size, uint batch_size, typename vertex_data_type,
          typename distance_type, typename vector_data_type, uint vector_degree,
          typename pq_data_type, uint pq_degree, uint32_t R, uint32_t L_cap,
          template <typename, typename, uint, uint> class full_distance_functor,
          template <typename, typename, uint, uint> class pq_distance_functor,
          bool on_host = true>
struct bulk_gpuANN {
  using edge_list_type = edge_list<vertex_data_type, R>;

  using vector_type = data_vector<vector_data_type, vector_degree>;
  using pq_vector_type = data_vector<pq_data_type, pq_degree>;

  using edge_pair_type = edge_pair<vertex_data_type, distance_type>;

  using thrust_vector_type = thrust::device_vector<edge_pair_type>;

  // each edge is of type vertex_data_type
  // as it is an unweighted directed edge to one vertex
  // every vertex stores only its outgoing neighbor list.

  vertex_data_type n_vertices;
  vertex_data_type n_vertices_max;
  vertex_data_type current_batch_size;
  vertex_data_type max_batch_size;
  vertex_data_type medoid;

  uint8_t *edge_counts;
  edge_list_type *edges;

  vector_type *vectors;
  pq_vector_type *pq_vectors;

  vertex_data_type *working_edges;

  std::vector<cudaStream_t> streams;

  uint64_t total_visited;
  uint64_t tail_visited;

  static constexpr edge_pair_type dead_edge =
      gpu_ann::get_dead_edge<vertex_data_type, distance_type>();

  // DEBUG MAP FOR COUNTING VISITED VERTICES.
  // std::unordered_map<vertex_data_type, uint64_t> visited_counter;

  // intialize bulk_gpuANN with # of vertices
  bulk_gpuANN(vertex_data_type n_vectors, double max_batch_ratio = .02) {
    n_vertices_max = n_vectors;

    // params = ext_params;

    // create cuda streams
    streams.resize(8);
    for (int i = 0; i < 8; ++i) {
      cudaError_t err = cudaStreamCreate(&streams[i]);
      if (err != cudaSuccess) {
        std::cerr << "Stream creation failed " << i << std::endl;
      }
    }

    n_vertices = 0;
    current_batch_size = 1;
    max_batch_size = std::min(static_cast<vertex_data_type>(max_batch_ratio * n_vectors),
                              static_cast<vertex_data_type>(1000000));

    if (max_batch_size == 0) max_batch_size = n_vectors;
    // max_batch_size = 10;

    if (on_host) {
      edges = gallatin::utils::get_host_version<edge_list_type>(n_vertices_max);
    } else {
      edges =
          gallatin::utils::get_device_version<edge_list_type>(n_vertices_max);
    }

    edge_counts = gallatin::utils::get_device_version<uint8_t>(n_vertices_max);

    if (on_host) {
      vectors = gallatin::utils::get_host_version<vector_type>(n_vertices_max);
    } else {
      vectors =
          gallatin::utils::get_device_version<vector_type>(n_vertices_max);
    }

    pq_vectors =
        gallatin::utils::get_device_version<pq_vector_type>(n_vertices_max);

    cudaMemset(edges, 0, sizeof(edge_list_type) * (n_vertices_max));
    cudaMemset(edge_counts, 0, sizeof(uint8_t) * (n_vertices_max));
    cudaMemset(vectors, 0, sizeof(vector_type) * (n_vertices_max));
    cudaMemset(pq_vectors, 0, sizeof(pq_vector_type) * (n_vertices_max));

    total_visited = 0;
    tail_visited = 0;
  }

  ~bulk_gpuANN() {
    if (on_host) {
      cudaFreeHost(edges);
      cudaFreeHost(vectors);
    } else {
      cudaFree(edges);
      cudaFree(vectors);
    }

    cudaFree(edge_counts);
    cudaFree(pq_vectors);

    for (int i = 0; i < 8; ++i) {
      if (streams[i] != nullptr)
        cudaStreamDestroy(streams[i]);
      else
        std::cerr << " Stream not found " << i << std::endl;
    }
  }

  // one time pass over data
  //  this uses the distance functor to calculate per-vertex distances.
  __host__ void populate_edge_distances(edge_pair_type *new_edges,
                                        uint64_t n_edges) {
    if (n_edges == 0) return;

    ann_kernels::edge_distance_kernel<tile_size, edge_pair_type, vector_type,
                                      vertex_data_type, (vertex_data_type)~0ULL,
                                      full_distance_functor>
        <<<(n_edges * tile_size - 1) / batch_size + 1, batch_size>>>(
            new_edges, vectors, n_edges, 0ULL);

    cudaDeviceSynchronize();
  }

  // Arguments:
  //   - new_edges_vector: frontier during search
  //   - search_vectors: vectors allocated on device memory.
  template <typename search_vector_type,
            template <typename, typename, uint, uint> class distance_functor>
  __host__ void populate_vector_edge_distances(
      thrust_vector_type &new_edges_vector,
      search_vector_type *search_vectors) {
    // get the number of edges in the given frontier.
    uint64_t n_edges = new_edges_vector.size();

    if (n_edges == 0) return;

    edge_pair_type *new_edges_ptr =
        thrust::raw_pointer_cast(&new_edges_vector[0]);

    ann_kernels::edge_distance_kernel<16, edge_pair_type, search_vector_type,
                                      vertex_data_type, (vertex_data_type)~0ULL,
                                      distance_functor>
        <<<(n_edges * 16 - 1) / batch_size + 1, batch_size>>>(
            new_edges_ptr, search_vectors, n_edges, 0ULL);

    cudaDeviceSynchronize();
  }

  template <typename search_vector_type,
            template <typename, typename, uint, uint> class distance_functor>
  __host__ void populate_vector_edge_distances_dead(
      thrust_vector_type &new_edges_vector,
      search_vector_type *search_vectors) {
    uint64_t n_edges = new_edges_vector.size();

    if (n_edges == 0) return;

    edge_pair_type *new_edges_ptr =
        thrust::raw_pointer_cast(&new_edges_vector[0]);

    uint32_t n_tiles = n_edges / EDGE_DISTANCE_LOAD_PER_TILE + 1;
    uint32_t grid_size =
        (n_tiles * EDGE_DISTANCE_TILE_SIZE - 1) / batch_size + 1;

    ann_kernels::edge_distance_kernel_dead<
        EDGE_DISTANCE_TILE_SIZE, edge_pair_type, search_vector_type,
        vertex_data_type, (vertex_data_type)~0ULL, distance_functor>
        <<<grid_size, batch_size>>>(new_edges_ptr, search_vectors, n_edges,
                                    0ULL, dead_edge,
                                    EDGE_DISTANCE_LOAD_PER_TILE);

    cudaDeviceSynchronize();
  }

  void debug_print_edges(const thrust_vector_type &device_vec) {
    thrust::host_vector<edge_pair_type> host_vec = device_vec;
    std::cout << "debug_print_edges() with n=" << host_vec.size() << std::endl;

    if (host_vec.empty()) return;

    uint32_t current_source = host_vec[0].source;
    size_t count = 1;

    for (size_t i = 1; i < host_vec.size(); ++i) {
      if (host_vec[i].source == current_source) {
        ++count;
      } else {
        std::cout << "source = " << current_source << ", count = " << count
                  << "\n";
        current_source = host_vec[i].source;
        count = 1;
      }
    }

    // Print the final group
    std::cout << "source = " << current_source << ", count = " << count << "\n";
  }

  template <typename search_vector_type,
            template <typename, typename, uint, uint> class distance_functor>
  __host__ void populate_vector_edge_distances_dead_query(
      thrust_vector_type &new_edges_vector,
      search_vector_type *search_vectors) {
    uint64_t n_edges = new_edges_vector.size();

    if (n_edges == 0) return;

    // debug_print_edges(new_edges_vector);

    edge_pair_type *new_edges_ptr =
        thrust::raw_pointer_cast(&new_edges_vector[0]);

    uint32_t n_tiles = n_edges / EDGE_DISTANCE_LOAD_PER_TILE + 1;
    uint32_t grid_size =
        (n_tiles * EDGE_DISTANCE_TILE_SIZE - 1) / batch_size + 1;

    ann_kernels::edge_distance_kernel_dead<
        EDGE_DISTANCE_TILE_SIZE, edge_pair_type, search_vector_type,
        vertex_data_type, (vertex_data_type)~0ULL, distance_functor>
        <<<grid_size, batch_size>>>(new_edges_ptr, search_vectors, n_edges,
                                    0ULL, dead_edge,
                                    EDGE_DISTANCE_LOAD_PER_TILE);

    cudaDeviceSynchronize();

    // uint64_t n_edges = new_edges_vector.size();

    // if (n_edges == 0) return;

    // edge_pair_type *new_edges_ptr =
    //     thrust::raw_pointer_cast(&new_edges_vector[0]);

    // //const uint32_t edges_per_partition = 1000;
    // //const uint32_t num_partition = n_edges / edges_per_partition + 1;
    // const uint32_t num_partition = 168;
    // const uint32_t edges_per_partition = (n_edges + num_partition - 1) /
    // num_partition;

    // const uint32_t block_size = 768;
    // const uint32_t grid_size = num_partition;
    // const uint32_t shared_mem_size = 32 * 16 * 16 * sizeof(float);

    // // Ask cuda what is the best setup
    // auto kernel_ptr = ann_kernels::edge_distance_kernel_dead_query<
    //     edge_pair_type, search_vector_type, vertex_data_type,
    //     (vertex_data_type)~0ULL>;
    // //int minGridSize, blockSize;
    // //cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize,
    // kernel_ptr, 16 * 16 * 4, 0);
    // //std::cout << "cudaOccupancyMaxPotentialBlockSize: minGridSize=" <<
    // minGridSize <<" blockSize=" << blockSize << std::endl;

    // cudaFuncSetCacheConfig(kernel_ptr, cudaFuncCachePreferShared);
    // ann_kernels::edge_distance_kernel_dead_query<
    //     edge_pair_type, search_vector_type, vertex_data_type,
    //     (vertex_data_type)~0ULL><<<grid_size, block_size, shared_mem_size>>>(
    //     new_edges_ptr, search_vectors, n_edges, edges_per_partition);
    // cudaError_t err = cudaGetLastError();
    // if (err != cudaSuccess) {
    //   std::cerr << "Kernel launch failed: " << cudaGetErrorString(err)
    //             << std::endl;
    // }
    // cudaError_t sync_err = cudaDeviceSynchronize();
    // if (sync_err != cudaSuccess) {
    //   std::cerr << "Sync failed: " << cudaGetErrorString(sync_err) <<
    //   std::endl;
    // }
  }

  __host__ void flip_edges(thrust_vector_type &new_edges_vector) {
    uint64_t n_edges = new_edges_vector.size();

    edge_pair_type *flip_edges = thrust::raw_pointer_cast(&new_edges_vector[0]);

    ann_kernels::flip_edges_kernel<edge_pair_type>
        <<<(n_edges - 1) / batch_size + 1, batch_size>>>(flip_edges, n_edges);

    cudaDeviceSynchronize();
  }

  __host__ void assert_unique(std::string name,
                              thrust_vector_type new_edges_vector) {
    thrust_vector_type copy = new_edges_vector;

    thrust::sort(thrust::device, copy.begin(), copy.end());

    auto uniq_end = thrust::unique(copy.begin(), copy.end());

    if (uniq_end - copy.begin() != new_edges_vector.size()) {
      std::cout << name << " is not unique" << std::endl;
    } else {
      std::cout << name << " is unique!" << std::endl;
    }
  }

  // get vectors representing unique # of edges
  //  and size of those edges
  __host__ thrust::device_vector<uint32_t> generate_unique_starts(
      thrust_vector_type edge_vector) {
    thrust::device_vector<uint32_t> starting_tid(edge_vector.size());

    thrust::sequence(starting_tid.begin(), starting_tid.end());

    auto end = thrust::unique_by_key(
        thrust::device, edge_vector.begin(), edge_vector.end(),
        starting_tid.begin(),
        pruneEqualityComparator<vertex_data_type, distance_type>());

    starting_tid.resize(end.second - starting_tid.begin());

    return starting_tid;
  }

  // generic function to add new edges to the graph
  // populated by distance already.
  // returns the outgoing neighbors of the nodes.
  template <bool can_merge>
  __host__ thrust_vector_type
  add_new_edges(thrust_vector_type &new_edges_vector, double alpha) {
    uint64_t n_edges = new_edges_vector.size();

    gpu_error::static_timer<11>::start("Total Add time");

    edge_pair_type *new_edges = thrust::raw_pointer_cast(&new_edges_vector[0]);

    // given a set of new edges, merge into graph
    // original edges are populated. Determine how many new edges to add.

    gpu_error::static_timer<12>::start("Dumb malloc call");

    uint64_t *existing_edges_count;
    cudaMallocManaged((void **)&existing_edges_count, sizeof(uint64_t));
    existing_edges_count[0] = 0;
    cudaDeviceSynchronize();

    gpu_error::static_timer<12>::stop();
    gpu_error::static_timer<13>::start("Edge count kernel");

    ann_kernels::count_existing_edges_kernel<edge_pair_type, edge_list_type,
                                             vertex_data_type,
                                             (vertex_data_type)~0ULL>
        <<<(n_edges - 1) / batch_size + 1, batch_size>>>(
            new_edges, edges, edge_counts, n_edges, existing_edges_count);
    cudaDeviceSynchronize();

    gpu_error::static_timer<13>::stop();
    gpu_error::static_timer<14>::start("Old Edge Merge");

    uint64_t new_edges_added = existing_edges_count[0];
    existing_edges_count[0] = 0;
    thrust_vector_type full_new_edges(new_edges_vector);
    cudaDeviceSynchronize();

    full_new_edges.resize(n_edges + new_edges_added);
    cudaDeviceSynchronize();

    edge_pair_type *full_new_edges_ptr =
        thrust::raw_pointer_cast(&full_new_edges[0]);

    constexpr uint n_tiles = batch_size / tile_size;

    ann_kernels::add_existing_edges_kernel<
        tile_size, n_tiles, edge_pair_type, edge_list_type, vector_type,
        vertex_data_type, (vertex_data_type)~0ULL>
        <<<(tile_size * n_edges - 1) / batch_size + 1, batch_size>>>(
            full_new_edges_ptr, vectors, edges, edge_counts, n_edges,
            existing_edges_count);
    cudaDeviceSynchronize();

    if (new_edges_added != 0) {
      ann_kernels::edge_distance_kernel<
          tile_size, edge_pair_type, vector_type, vertex_data_type,
          (vertex_data_type)~0ULL, full_distance_functor>
          <<<(new_edges_added * tile_size - 1) / batch_size + 1, batch_size>>>(
              full_new_edges_ptr, vectors, new_edges_added, n_edges);
      cudaDeviceSynchronize();
    }
    cudaDeviceSynchronize();

    thrust::sort(thrust::device, full_new_edges.begin(), full_new_edges.end());
    cudaDeviceSynchronize();

    gpu_error::static_timer<14>::stop();
    gpu_error::static_timer<15>::start("Get unique starts");

    auto start_indices_vector = generate_unique_starts(full_new_edges);
    uint32_t *start_indices_ptr =
        thrust::raw_pointer_cast(&start_indices_vector[0]);
    full_new_edges_ptr = thrust::raw_pointer_cast(&full_new_edges[0]);
    cudaDeviceSynchronize();

    gpu_error::static_timer<15>::stop();
    gpu_error::static_timer<16>::start("Robust prune thrust kernel");

    // new block_based kernel
    ann_kernels::robust_prune_block<
        16, vertex_data_type, distance_type, vector_data_type, vector_degree,
        full_distance_functor, edge_pair_type, R, can_merge>
        <<<start_indices_vector.size(), 512>>>(
            edges, edge_counts, vectors, full_new_edges_ptr, dead_edge,
            start_indices_ptr, start_indices_vector.size(), alpha,
            n_edges + new_edges_added);

    cudaFree(existing_edges_count);
    cudaDeviceSynchronize();

    gpu_error::static_timer<16>::stop();
    gpu_error::static_timer<3>::start("Thrust remove dead");
    // clip dead edges from full_set
    remove_dead(full_new_edges);

    gpu_error::static_timer<3>::stop();
    gpu_error::static_timer<11>::stop();

    return full_new_edges;
  }

  __host__ uint32_t count_unique(thrust_vector_type &edge_vector) {
    uint32_t *unique_count;
    cudaMallocManaged((void **)&unique_count, sizeof(uint32_t));

    cudaMemset(unique_count, 0, sizeof(uint32_t));

    edge_pair_type *edge_ptr = thrust::raw_pointer_cast(&edge_vector[0]);

    ann_kernels::count_unique_kernel<<<
        (edge_vector.size() - 1) / batch_size + 1, batch_size>>>(
        unique_count, edge_ptr, edge_vector.size());

    cudaDeviceSynchronize();

    uint32_t result = unique_count[0];

    cudaFree(unique_count);

    return result;
  }

  // helper function - count the number of times each key appears!
  // requires keys to be sorted.
  // populates copy - original is unchanged
  std::pair<thrust_vector_type, thrust::device_vector<uint32_t>>
  count_and_reduce(thrust_vector_type edge_vector) {
    thrust::device_vector<uint32_t> counts(edge_vector.size());

    thrust::fill(counts.begin(), counts.end(), 1);

    thrust_vector_type output_keys(edge_vector.size());
    thrust::device_vector<uint32_t> output_vals(edge_vector.size());

    auto new_end = thrust::reduce_by_key(
        edge_vector.begin(), edge_vector.end(), counts.begin(),
        output_keys.begin(), output_vals.begin(),
        pruneEqualityComparator<vertex_data_type, distance_type>());

    output_keys.resize(new_end.first - output_keys.begin());
    output_vals.resize(new_end.second - output_vals.begin());
    return {output_keys, output_vals};
  }

  // given a set of incoming edges, populate return vector with degree of
  // existing edges
  __host__ thrust::device_vector<uint32_t> populate_edge_sizes(
      thrust_vector_type &unique_edges) {
    thrust::device_vector<uint32_t> existing_sizes(unique_edges.size());

    thrust::fill(existing_sizes.begin(), existing_sizes.end(), 0);

    edge_pair_type *edge_ptr = thrust::raw_pointer_cast(&unique_edges[0]);

    uint32_t *counts = thrust::raw_pointer_cast(&existing_sizes[0]);

    ann_kernels::populate_old_counts<edge_pair_type>
        <<<(unique_edges.size() - 1) / batch_size + 1, batch_size>>>(
            edge_ptr, counts, edge_counts, unique_edges.size());

    cudaDeviceSynchronize();

    return existing_sizes;
  }

  // helper function that encapsulates data transfer of new edges
  // calls kernel to determine new edge size
  //  then resizes and repopulates with edges.
  //  original vector (new_edges_vector) is modified in-place.
  //  and size is trimmed.
  void copy_existing_edges_based_on_size(
      thrust_vector_type &new_edges_vector, thrust_vector_type &unique_edges,
      thrust::device_vector<uint32_t> &new_counts,
      thrust::device_vector<uint32_t> &existing_counts) {
    // number of edges to add into array.
    uint32_t *append_edge_count;

    cudaMallocManaged((void **)&append_edge_count, sizeof(uint32_t));

    cudaMemset(append_edge_count, 0, sizeof(uint32_t));

    // get edges
    edge_pair_type *unique_edges_ptr =
        thrust::raw_pointer_cast(&unique_edges[0]);

    // and counters.
    uint32_t *new_counts_ptr = thrust::raw_pointer_cast(&new_counts[0]);

    uint32_t *existing_counts_ptr =
        thrust::raw_pointer_cast(&existing_counts[0]);

    // collection kernel.
    ann_kernels::count_existing_edges_kernel_with_count<edge_pair_type, R>
        <<<(unique_edges.size() - 1) / batch_size + 1, batch_size>>>(
            unique_edges_ptr, unique_edges.size(), new_counts_ptr,
            existing_counts_ptr, append_edge_count);

    cudaDeviceSynchronize();

    uint64_t old_size = new_edges_vector.size();

    uint64_t new_edges_added = append_edge_count[0];

    new_edges_vector.resize(old_size + new_edges_added);

    edge_pair_type *new_edges_ptr =
        thrust::raw_pointer_cast(&new_edges_vector[0]);

    cudaMemset(append_edge_count, 0, sizeof(uint32_t));

    // and full append kernel.
    //  size is based on unique counts.

    constexpr uint n_tiles = batch_size / 16;

    if (new_edges_added != 0) {
      ann_kernels::add_existing_edges_kernel_with_count<
          16, n_tiles, edge_pair_type, edge_list_type, R>
          <<<(16 * unique_edges.size() - 1) / batch_size + 1, batch_size>>>(
              new_edges_ptr, unique_edges_ptr, unique_edges.size(), edges,
              new_counts_ptr, existing_counts_ptr, old_size, append_edge_count);

      ann_kernels::edge_distance_kernel<
          tile_size, edge_pair_type, vector_type, vertex_data_type,
          (vertex_data_type)~0ULL, full_distance_functor>
          <<<(new_edges_added * tile_size - 1) / batch_size + 1, batch_size>>>(
              new_edges_ptr, vectors, new_edges_added, old_size);
    }

    cudaDeviceSynchronize();

    cudaFree(append_edge_count);

    return;
  }

  // generic function to add new edges to the graph
  // populated by distance already.
  // returns set of new neighbors that have been pruned.
  // operational steps.
  // 1. generate # of existing verts
  // 2. count # of new edges
  // 3. count # of old edges
  // 3. count # where new_edges+old_edge > R
  // 4. copy valid old edges.
  // 5. merge must account for old edges being nonzero (copy kernel wipes.)
  template <bool can_merge>
  __host__ void add_new_edges_in_place(thrust_vector_type &new_edges_vector,
                                       double alpha) {
    uint64_t n_edges = new_edges_vector.size();

    gpu_error::static_timer<11>::start("Total Add time");

    auto [unique_edges, counts] = count_and_reduce(new_edges_vector);
    auto existing_sizes = populate_edge_sizes(unique_edges);

    // populate new edges_vector with size.
    copy_existing_edges_based_on_size(new_edges_vector, unique_edges, counts,
                                      existing_sizes);

    // sort for final install.
    thrust::sort(thrust::device, new_edges_vector.begin(),
                 new_edges_vector.end());
    cudaDeviceSynchronize();

    gpu_error::static_timer<15>::start("Get unique starts");

    // this can be replaced with reduce across sizes.
    auto start_indices_vector = generate_unique_starts(new_edges_vector);
    uint32_t *start_indices_ptr =
        thrust::raw_pointer_cast(&start_indices_vector[0]);
    auto full_new_edges_ptr = thrust::raw_pointer_cast(&new_edges_vector[0]);
    cudaDeviceSynchronize();

    gpu_error::static_timer<15>::stop();

    gpu_error::static_timer<16>::start("Robust prune thrust kernel");

    ann_kernels::robust_prune_old_edges<
        16, vertex_data_type, distance_type, vector_data_type, vector_degree,
        full_distance_functor, edge_pair_type, R, can_merge>
        <<<start_indices_vector.size(), 256>>>(
            edges, edge_counts, vectors, full_new_edges_ptr, dead_edge,
            start_indices_ptr, start_indices_vector.size(), alpha,
            new_edges_vector.size());

    cudaDeviceSynchronize();

    gpu_error::static_timer<16>::stop();

    gpu_error::static_timer<3>::start("Thrust remove dead");
    // clip dead edges from full_set
    // remove_dead(full_new_edges);

    gpu_error::static_timer<3>::stop();

    gpu_error::static_timer<11>::stop();

    // return full_new_edges;
  }

  __host__ void make_sorted_and_unique(thrust_vector_type &vector) {
    // thrust_vector_type new_vector = vector;
    thrust::sort(
        thrust::device, vector.begin(), vector.end(),
        gpu_ann::beamSearchComparator<vertex_data_type, distance_type>());

    auto vector_unique = thrust::unique(vector.begin(), vector.end());

    vector.resize(vector_unique - vector.begin());
  }

  __host__ void remove_dead(thrust_vector_type &vector) {
    if (vector.size() == 0) return;

    auto new_end = thrust::remove_if(
        vector.begin(), vector.end(),
        pruneBadEdgeComparator<vertex_data_type, distance_type>());

    vector.resize(new_end - vector.begin());
  }

  __host__ void make_sorted(thrust_vector_type &vector) {
    // thrust_vector_type new_vector = vector;
    thrust::sort(
        thrust::device, vector.begin(), vector.end(),
        gpu_ann::beamSearchComparator<vertex_data_type, distance_type>());
  }

  __host__ void make_unique(thrust_vector_type &vector) {
    // thrust_vector_type new_vector = vector;
    auto vector_unique = thrust::unique(vector.begin(), vector.end());

    vector.resize(vector_unique - vector.begin());
  }

  __host__ void thrust_merge(thrust_vector_type &l_vector,
                             thrust_vector_type &r_vector,
                             thrust_vector_type &output) {
    // thrust_vector_type new_vector = vector;
    output.resize(l_vector.size() + r_vector.size());

    auto output_end = thrust::merge(
        thrust::device, l_vector.begin(), l_vector.end(), r_vector.begin(),
        r_vector.end(), output.begin(),
        gpu_ann::beamSearchComparator<vertex_data_type, distance_type>());

    output.resize(output_end - output.begin());
  }

  // assume a sorted frontier.
  __host__ thrust_vector_type clip_to_beam_width(
      thrust_vector_type ext_frontier, uint64_t node_start_idx,
      uint64_t n_vectors_in_batch, uint beam_width, bool is_sorted = false) {
    if (ext_frontier.size() == 0) {
      return ext_frontier;
    }

    gpu_error::static_timer<24>::start("Clip beam Malloc");
    thrust_vector_type frontier = ext_frontier;
    uint64_t *starts;
    cudaMallocManaged((void **)&starts, sizeof(uint64_t) * n_vectors_in_batch);
    uint64_t *n_cut;
    cudaMallocManaged((void **)&n_cut, sizeof(uint64_t));
    n_cut[0] = 0;
    gpu_error::static_timer<24>::stop();

    gpu_error::static_timer<25>::start("Clip beam sort frontier");
    if (!is_sorted) make_sorted_and_unique(frontier);
    gpu_error::static_timer<25>::stop();
    cudaDeviceSynchronize();

    gpu_error::static_timer<26>::start("Clip beam find start");
    edge_pair_type *frontier_ptr = thrust::raw_pointer_cast(&frontier[0]);
    ann_kernels::find_beam_start_kernel<edge_pair_type>
        <<<(frontier.size() - 1) / batch_size + 1, batch_size>>>(
            frontier_ptr, frontier.size(), starts, node_start_idx);
    cudaDeviceSynchronize();
    gpu_error::static_timer<26>::stop();

    // and clip beam;

    gpu_error::static_timer<27>::start("Clip beam kernel");
    ann_kernels::clip_beam_kernel<edge_pair_type>
        <<<(frontier.size() - 1) / batch_size + 1, batch_size>>>(
            frontier_ptr, frontier.size(), starts, node_start_idx, n_cut,
            beam_width, dead_edge);
    cudaDeviceSynchronize();

    gpu_error::static_timer<27>::stop();
    gpu_error::static_timer<28>::start("Clip beam remove dead");

    // sort and clip.
    remove_dead(frontier);

    cudaDeviceSynchronize();

    cudaFree(n_cut);
    cudaFree(starts);

    gpu_error::static_timer<28>::stop();

    return frontier;
  }

  // BSP version of beam search
  //
  // Arguments:
  //   - start: index of the starting vertex (n_vertices)
  //   - n_vectors_in_batch: number of vectors this batch need to process
  //   - search_vectors: allocated vectors in device memory
  // Returns:
  //   - frontier
  //   - visited
  template <typename search_vector_type>
  __host__ std::pair<thrust_vector_type, thrust_vector_type> beam_search(
      vertex_data_type start, vertex_data_type n_vectors_in_batch,
      search_vector_type *search_vectors) {
    // const uint beam_width = 1;

    // process each vertex independently.
    // all ops are read-only so this is fine.
    //  candidates are atomically added all at once.
    //  in kernel using managed memory
    //  so host knows how to size vector.

    // comparisons drawn with beamSearch.h

    // line 64 - init frontier
    uint64_t *candidates_count;

    cudaMallocManaged((void **)&candidates_count, sizeof(uint64_t));

    uint64_t *live_frontier_count;

    cudaMallocManaged((void **)&live_frontier_count, sizeof(uint64_t));

    float *largest_in_frontier;

    cudaMallocManaged((void **)&largest_in_frontier,
                      sizeof(float) * n_vectors_in_batch);

    for (uint i = 0; i < n_vectors_in_batch; i++) {
      largest_in_frontier[i] = std::numeric_limits<float>::max();
    }

    thrust_vector_type frontier(n_vectors_in_batch);

    thrust_vector_type live_frontier(n_vectors_in_batch);

    thrust_vector_type candidates(R * n_vectors_in_batch);

    thrust_vector_type new_frontier(0);
    thrust_vector_type new_visited(0);

    thrust_vector_type visited(0);
    thrust_vector_type temp_frontier(0);

    thrust::host_vector<edge_pair_type> setup;

    // line 66-68 - populate frontier
    for (uint i = 0; i < n_vectors_in_batch; i++) {
      edge_pair_type to_fill;

      to_fill.source = start + i;
      to_fill.sink = medoid;
      to_fill.distance = 0.0;

      setup.push_back(to_fill);
    }

    frontier = setup;

    cudaDeviceSynchronize();

    edge_pair_type *frontier_ptr = thrust::raw_pointer_cast(&frontier[0]);

    populate_vector_edge_distances<search_vector_type, full_distance_functor>(
        frontier, search_vectors);

    // lines 72-77
    new_frontier = frontier;

    // start loop

    uint64_t n_loops = 0;

    // start of main loop
    while (new_frontier.size() != 0) {
      n_loops++;

      // if (n_loops >= 1024){
      //    break;
      // }

      // print_edge_head("frontier", frontier);

      // std::cout << "Loop " << n_loops << ", Current size "
      //           << new_frontier.size() << "/" << frontier.size()
      //           << ", visisted " << visited.size() << std::endl;

      live_frontier.resize(n_vectors_in_batch);

      candidates.resize(R * n_vectors_in_batch);

      frontier_ptr = thrust::raw_pointer_cast(&frontier[0]);
      edge_pair_type *new_frontier_ptr =
          thrust::raw_pointer_cast(&new_frontier[0]);
      edge_pair_type *live_frontier_ptr =
          thrust::raw_pointer_cast(&live_frontier[0]);
      edge_pair_type *candidates_ptr = thrust::raw_pointer_cast(&candidates[0]);

      cudaMemset(live_frontier_count, 0, sizeof(uint64_t));
      cudaMemset(candidates_count, 0, sizeof(uint64_t));
      // live_frontier_count[0] = 0;

      // candidates_count[0] = 0;

      cudaDeviceSynchronize();

      // gather frontier

      ann_kernels::select_new_frontier<edge_pair_type>
          <<<(frontier.size() - 1) / batch_size + 1, batch_size>>>(
              new_frontier_ptr, live_frontier_ptr, new_frontier.size(),
              live_frontier_count);

      cudaDeviceSynchronize();

      live_frontier.resize(live_frontier_count[0]);

      cudaDeviceSynchronize();

      ann_kernels::populate_all_candidates_kernel<
          tile_size, edge_list_type, edge_pair_type, vertex_data_type>
          <<<(live_frontier.size() * tile_size - 1) / batch_size + 1,
             batch_size>>>(edges, edge_counts, live_frontier_ptr,
                           live_frontier_count[0], candidates_ptr,
                           candidates_count, R * n_vectors_in_batch);

      cudaDeviceSynchronize();

      candidates.resize(candidates_count[0]);

      // print_edge_vector("Candidates", candidates);

      cudaDeviceSynchronize();

      populate_vector_edge_distances<vector_type, full_distance_functor>(
          candidates, search_vectors);

      cudaDeviceSynchronize();

      thrust::sort(
          thrust::device, live_frontier.begin(), live_frontier.end(),
          gpu_ann::beamSearchComparator<vertex_data_type, distance_type>());

      make_sorted_and_unique(candidates);

      cudaDeviceSynchronize();

      // clip candidates - gather current best cutoff and clip candidates.
      get_frontier_cutoff(frontier, start, largest_in_frontier, L_cap);

      prune_candidates(candidates, start, largest_in_frontier);

      // union is at most candidates + frontier;
      uint64_t frontier_size = frontier.end() - frontier.begin();

      temp_frontier.resize(frontier_size + candidates.size());

      // print_edge_vector("Frontier before merge", frontier);

      thrust_merge(frontier, candidates, temp_frontier);

      // print_edge_vector("New Frontier before", temp_frontier);

      // print_edge_vector("New Frontier After", temp_frontier);

      make_unique(temp_frontier);

      cudaDeviceSynchronize();

      thrust_merge(visited, live_frontier, new_visited);

      cudaDeviceSynchronize();

      visited = new_visited;

      cudaDeviceSynchronize();

      frontier = clip_to_beam_width(temp_frontier, start, n_vectors_in_batch,
                                    L_cap, true);

      cudaDeviceSynchronize();

      new_frontier.resize(frontier.size());

      auto result_end = thrust::set_difference(
          thrust::device, frontier.begin(), frontier.end(), visited.begin(),
          visited.end(), new_frontier.begin(),
          gpu_ann::beamSearchComparator<vertex_data_type, distance_type>());

      cudaDeviceSynchronize();

      // this resize is bad?
      uint64_t new_frontier_size = result_end - new_frontier.begin();

      // print_edge_vector("set diff before resize", new_frontier);

      new_frontier.resize(new_frontier_size);

      cudaDeviceSynchronize();
    }

    cudaFree(largest_in_frontier);

    return {frontier, visited};
  }

  __host__ void sort_partitioned(thrust_vector_type &candidates,
                                 uint32_t *candidate_partitions,
                                 uint32_t n_vectors_in_batch) {
    uint32_t step = 1000;

    if (n_vectors_in_batch / 8 > step) step = n_vectors_in_batch / 8 + 1;

    uint start = 0;

    uint stream_id = 0;

    while (start + step < n_vectors_in_batch) {
      uint32_t candidate_start = candidate_partitions[start];
      uint32_t candidate_end = candidate_partitions[start + step];

      thrust::sort(
          thrust::cuda::par.on(streams[stream_id]),
          candidates.begin() + candidate_start,
          candidates.begin() + candidate_end,
          gpu_ann::beamSearchComparator<vertex_data_type, distance_type>());

      start += step;

      stream_id = (stream_id + 1) % 8;
    }

    uint32_t candidate_start = candidate_partitions[start];
    uint32_t candidate_end = candidate_partitions[n_vectors_in_batch];

    thrust::sort(
        thrust::cuda::par.on(streams[stream_id]),
        candidates.begin() + candidate_start,
        candidates.begin() + candidate_end,
        gpu_ann::beamSearchComparator<vertex_data_type, distance_type>());

    for (uint i = 0; i < 8; i++) {
      cudaStreamSynchronize(streams[i]);
    }

    make_unique(candidates);
  }

  // one iteration of beam search with a set width.
  template <typename search_vector_type,
            template <typename, typename, uint, uint> class distance_functor>
  __host__ void beam_search_one_round(

      vertex_data_type &start, vertex_data_type &n_vectors_in_batch,
      search_vector_type *search_vectors, thrust_vector_type &frontier,
      thrust_vector_type &visited, thrust_vector_type &unvisited_frontier,
      thrust_vector_type &candidates, thrust_vector_type &live_frontier,
      thrust_vector_type &temp_visited, thrust_vector_type &temp_frontier,
      uint32_t *unvisited_starts, uint32_t *unvisited_candidate_counters,
      uint32_t *host_unvisited_candidate_counters,
      uint32_t *live_frontier_counters, uint32_t *candidate_counter,
      float *frontier_cutoffs, uint32_t nodes_explored_per_iteration,
      uint32_t beam_width) {
    // assert univisited is large enough

    gpu_error::static_timer<5>::start("Vector resize + fill");

    // cudaMemset(candidate_counter, 0, sizeof(uint32_t));
    cudaMemset(live_frontier_counters, 0,
               sizeof(uint32_t) * (n_vectors_in_batch));
    cudaMemset(unvisited_candidate_counters, 0,
               sizeof(uint32_t) * n_vectors_in_batch);

    unvisited_frontier.resize(beam_width * n_vectors_in_batch);

    // use thrust fill to populate with dead edge
    //  so candidates only write valid.

    live_frontier.resize(n_vectors_in_batch * nodes_explored_per_iteration);
    candidates.resize(n_vectors_in_batch * nodes_explored_per_iteration * R);

    thrust::fill(live_frontier.begin(), live_frontier.end(), dead_edge);
    thrust::fill(candidates.begin(), candidates.end(), dead_edge);

    gpu_error::static_timer<5>::stop();
    // and populate

    gpu_error::static_timer<6>::start("Set diff");

    auto unvisited_end = thrust::set_difference(
        thrust::device, frontier.begin(), frontier.end(), visited.begin(),
        visited.end(), unvisited_frontier.begin(),
        gpu_ann::beamSearchComparator<vertex_data_type, distance_type>());

    uint64_t unvisited_size = unvisited_end - unvisited_frontier.begin();

    unvisited_frontier.resize(unvisited_size);

    gpu_error::static_timer<6>::stop();
    if (unvisited_size == 0) return;

    // set sizes of frontier
    edge_pair_type *unvisited_frontier_ptr =
        thrust::raw_pointer_cast(&unvisited_frontier[0]);
    edge_pair_type *candidates_ptr = thrust::raw_pointer_cast(&candidates[0]);
    edge_pair_type *live_frontier_ptr =
        thrust::raw_pointer_cast(&live_frontier[0]);

    gpu_error::static_timer<7>::start("Find unvisited start");

    ann_kernels::find_beam_start_kernel_u32<edge_pair_type>
        <<<(unvisited_size - 1) / batch_size + 1, batch_size>>>(
            unvisited_frontier_ptr, unvisited_size, unvisited_starts, start);

    gpu_error::static_timer<7>::stop();
    // populate with best in frontier

    gpu_error::static_timer<8>::start("get frontier cutoff");
    get_frontier_cutoff(frontier, start, frontier_cutoffs, beam_width);

    gpu_error::static_timer<8>::stop();
    // and cheaper candidate population - fill all slots.
    //  gather live frontier as well.

    gpu_error::static_timer<9>::start("Candidate population");

    // new version - find bounds of candidates
    //  issues one thread per vertex.
    ann_kernels::find_candidate_bounds<edge_list_type, edge_pair_type>
        <<<(n_vectors_in_batch - 1) / batch_size + 1, batch_size>>>(
            edges, edge_counts, unvisited_frontier_ptr,
            unvisited_frontier.size(), unvisited_starts,
            unvisited_candidate_counters, live_frontier_counters,
            nodes_explored_per_iteration, start, n_vectors_in_batch);

    // reduce candidate counters.
    // this returns the start
    // these need to be one larger.
    thrust::exclusive_scan(
        thrust::device, unvisited_candidate_counters,
        unvisited_candidate_counters + n_vectors_in_batch + 1,
        unvisited_candidate_counters);
    thrust::exclusive_scan(thrust::device, live_frontier_counters,
                           live_frontier_counters + n_vectors_in_batch + 1,
                           live_frontier_counters);

    cudaMemcpy(host_unvisited_candidate_counters, unvisited_candidate_counters,
               sizeof(uint32_t) * (n_vectors_in_batch + 1), cudaMemcpyDefault);

    // candidate population

    constexpr uint n_tiles = batch_size / 16;

    uint64_t n_teams_launched =
        n_vectors_in_batch * nodes_explored_per_iteration;

    ann_kernels::populate_candidates_prefix_sum<
        16, n_tiles, edge_list_type, edge_pair_type, vertex_data_type>
        <<<(n_teams_launched * 16 - 1) / batch_size + 1, batch_size>>>(
            edges, edge_counts, unvisited_frontier_ptr, unvisited_size,
            candidates_ptr, live_frontier_ptr, unvisited_starts,
            unvisited_candidate_counters, live_frontier_counters, R, beam_width,
            nodes_explored_per_iteration, start, n_vectors_in_batch, dead_edge);

    cudaDeviceSynchronize();

    candidates.resize(unvisited_candidate_counters[n_vectors_in_batch]);

    live_frontier.resize(live_frontier_counters[n_vectors_in_batch]);

    gpu_error::static_timer<9>::stop();
    gpu_error::static_timer<10>::start("Candidate distance pop");
    // populate all distances, even dead ones, in candidates
    populate_vector_edge_distances_dead<search_vector_type, distance_functor>(
        candidates, search_vectors);

    // prune non-dead distances based on cutoffs

    gpu_error::static_timer<10>::stop();

    gpu_error::static_timer<19>::start("Candidate prune");

    prune_candidates_with_dead(candidates, start, frontier_cutoffs);

    gpu_error::static_timer<19>::stop();

    gpu_error::static_timer<20>::start("Candidate cleanup");

    gpu_error::static_timer<29>::start("Candidate sort");

    // print_edge_vector("Candidates before sort", candidates);

    // fast sort.
    sort_partitioned(candidates, host_unvisited_candidate_counters,
                     n_vectors_in_batch);
    // make_sorted_and_unique(candidates);
    gpu_error::static_timer<29>::stop();

    gpu_error::static_timer<30>::start("Candidate remove");
    remove_dead(candidates);
    gpu_error::static_timer<30>::stop();

    // live frontier is "sorted.", as it is pulled from unvisited frontier.
    gpu_error::static_timer<31>::start("Live frontier sort");
    make_sorted_and_unique(live_frontier);
    gpu_error::static_timer<31>::stop();

    // remove_dead(live_frontier);

    gpu_error::static_timer<20>::stop();

    // at this point, we have a list of new candidates (maybe overlapping with
    // frontier)

    // a set of live frontier that is sorted

    // frontier is sorted

    // visited is sorted.

    // let's merge via set diff?

    // can avoid sort on merge by doing diff.

    // or merge and make unique

    // now merge candidates into frontier.

    gpu_error::static_timer<21>::start("Frontier modification");
    temp_frontier.resize(candidates.size() + frontier.size());
    thrust_merge(frontier, candidates, temp_frontier);
    make_unique(temp_frontier);
    frontier = clip_to_beam_width(temp_frontier, start, n_vectors_in_batch,
                                  beam_width, true);
    gpu_error::static_timer<21>::stop();
    // and process visited.

    gpu_error::static_timer<22>::start("Visited modification");

    temp_visited.resize(visited.size() + live_frontier.size());

    thrust_merge(visited, live_frontier, temp_visited);

    visited = temp_visited;

    gpu_error::static_timer<22>::stop();

    // done! visited + frontier both contain results of round.

    return;
  }

  template <typename search_vector_type,
            template <typename, typename, uint, uint> class distance_functor>
  __host__ void beam_search_one_round_query(
      vertex_data_type &start, vertex_data_type &n_vectors_in_batch,
      search_vector_type *search_vectors, thrust_vector_type &frontier,
      thrust_vector_type &visited, thrust_vector_type &unvisited_frontier,
      thrust_vector_type &candidates, thrust_vector_type &live_frontier,
      thrust_vector_type &temp_visited, thrust_vector_type &temp_frontier,
      uint32_t *unvisited_starts, uint32_t *unvisited_candidate_counters,
      uint32_t *host_unvisited_candidate_counters,
      uint32_t *live_frontier_counters, uint32_t *candidate_counter,
      float *frontier_cutoffs, uint32_t nodes_explored_per_iteration,
      uint32_t beam_width) {
    gpu_error::static_timer<5>::start("Vector resize + fill");

    cudaMemset(live_frontier_counters, 0,
               sizeof(uint32_t) * (n_vectors_in_batch));
    cudaMemset(unvisited_candidate_counters, 0,
               sizeof(uint32_t) * n_vectors_in_batch);

    unvisited_frontier.resize(beam_width * n_vectors_in_batch);

    // use thrust fill to populate with dead edge
    //  so candidates only write valid.

    live_frontier.resize(n_vectors_in_batch * nodes_explored_per_iteration);
    candidates.resize(n_vectors_in_batch * nodes_explored_per_iteration * R);

    thrust::fill(live_frontier.begin(), live_frontier.end(), dead_edge);
    thrust::fill(candidates.begin(), candidates.end(), dead_edge);

    gpu_error::static_timer<5>::stop();
    // and populate

    gpu_error::static_timer<6>::start("Set diff");

    auto unvisited_end = thrust::set_difference(
        thrust::device, frontier.begin(), frontier.end(), visited.begin(),
        visited.end(), unvisited_frontier.begin(),
        gpu_ann::beamSearchComparator<vertex_data_type, distance_type>());

    uint64_t unvisited_size = unvisited_end - unvisited_frontier.begin();

    unvisited_frontier.resize(unvisited_size);

    gpu_error::static_timer<6>::stop();
    if (unvisited_size == 0) return;

    // set sizes of frontier
    edge_pair_type *unvisited_frontier_ptr =
        thrust::raw_pointer_cast(&unvisited_frontier[0]);
    edge_pair_type *candidates_ptr = thrust::raw_pointer_cast(&candidates[0]);
    edge_pair_type *live_frontier_ptr =
        thrust::raw_pointer_cast(&live_frontier[0]);

    gpu_error::static_timer<7>::start("Find unvisited start");

    ann_kernels::find_beam_start_kernel_u32<edge_pair_type>
        <<<(unvisited_size - 1) / batch_size + 1, batch_size>>>(
            unvisited_frontier_ptr, unvisited_size, unvisited_starts, start);

    gpu_error::static_timer<7>::stop();
    // populate with best in frontier

    gpu_error::static_timer<8>::start("get frontier cutoff");
    get_frontier_cutoff(frontier, start, frontier_cutoffs, beam_width);

    gpu_error::static_timer<8>::stop();
    // and cheaper candidate population - fill all slots.
    //  gather live frontier as well.

    gpu_error::static_timer<9>::start("Candidate population");

    // new version - find bounds of candidates
    //  issues one thread per vertex.
    ann_kernels::find_candidate_bounds<edge_list_type, edge_pair_type>
        <<<(n_vectors_in_batch - 1) / batch_size + 1, batch_size>>>(
            edges, edge_counts, unvisited_frontier_ptr,
            unvisited_frontier.size(), unvisited_starts,
            unvisited_candidate_counters, live_frontier_counters,
            nodes_explored_per_iteration, start, n_vectors_in_batch);

    // reduce candidate counters.
    // this returns the start
    // these need to be one larger.
    thrust::exclusive_scan(
        thrust::device, unvisited_candidate_counters,
        unvisited_candidate_counters + n_vectors_in_batch + 1,
        unvisited_candidate_counters);
    thrust::exclusive_scan(thrust::device, live_frontier_counters,
                           live_frontier_counters + n_vectors_in_batch + 1,
                           live_frontier_counters);

    cudaMemcpy(host_unvisited_candidate_counters, unvisited_candidate_counters,
               sizeof(uint32_t) * (n_vectors_in_batch + 1), cudaMemcpyDefault);

    // candidate population

    constexpr uint n_tiles = batch_size / 16;

    uint64_t n_teams_launched =
        n_vectors_in_batch * nodes_explored_per_iteration;

    ann_kernels::populate_candidates_prefix_sum<
        16, n_tiles, edge_list_type, edge_pair_type, vertex_data_type>
        <<<(n_teams_launched * 16 - 1) / batch_size + 1, batch_size>>>(
            edges, edge_counts, unvisited_frontier_ptr, unvisited_size,
            candidates_ptr, live_frontier_ptr, unvisited_starts,
            unvisited_candidate_counters, live_frontier_counters, R, beam_width,
            nodes_explored_per_iteration, start, n_vectors_in_batch, dead_edge);

    cudaDeviceSynchronize();

    candidates.resize(unvisited_candidate_counters[n_vectors_in_batch]);

    live_frontier.resize(live_frontier_counters[n_vectors_in_batch]);

    gpu_error::static_timer<9>::stop();
    gpu_error::static_timer<10>::start("Candidate distance pop");
    // populate all distances, even dead ones, in candidates
    populate_vector_edge_distances_dead_query<search_vector_type,
                                              distance_functor>(candidates,
                                                                search_vectors);

    // prune non-dead distances based on cutoffs

    gpu_error::static_timer<10>::stop();

    gpu_error::static_timer<19>::start("Candidate prune");

    prune_candidates_with_dead(candidates, start, frontier_cutoffs);

    gpu_error::static_timer<19>::stop();

    gpu_error::static_timer<20>::start("Candidate cleanup");

    gpu_error::static_timer<29>::start("Candidate sort");

    // print_edge_vector("Candidates before sort", candidates);

    // fast sort.
    sort_partitioned(candidates, host_unvisited_candidate_counters,
                     n_vectors_in_batch);
    // make_sorted_and_unique(candidates);
    gpu_error::static_timer<29>::stop();

    gpu_error::static_timer<30>::start("Candidate remove");
    remove_dead(candidates);
    gpu_error::static_timer<30>::stop();

    // print_edge_vector("Candidates after sort", candidates);

    // live frontier is "sorted.", as it is pulled from unvisited frontier.

    gpu_error::static_timer<31>::start("Live frontier sort");
    make_sorted_and_unique(live_frontier);
    gpu_error::static_timer<31>::stop();

    // remove_dead(live_frontier);

    gpu_error::static_timer<20>::stop();

    gpu_error::static_timer<21>::start("Frontier modification");
    temp_frontier.resize(candidates.size() + frontier.size());
    thrust_merge(frontier, candidates, temp_frontier);
    make_unique(temp_frontier);
    frontier = clip_to_beam_width(temp_frontier, start, n_vectors_in_batch,
                                  beam_width, true);
    gpu_error::static_timer<21>::stop();

    gpu_error::static_timer<22>::start("Visited modification");
    temp_visited.resize(visited.size() + live_frontier.size());
    thrust_merge(visited, live_frontier, temp_visited);
    visited = temp_visited;
    gpu_error::static_timer<22>::stop();

    return;
  }

  // version two of the beam search.
  //  planned improvements.
  //  live frontier and live candidates do not need to be sorted.
  //  frontier and candidate bounds are therefore selected by threads during
  //  runtime. automatic integration for wider beams and setup for adjustable
  //  beam width.

  // at every step, keep track of the current size of the frontier and don't
  // allow for resizes
  //  populate with special dead_edges that keep space.

  // use clip to only keep better candidates?
  template <typename search_vector_type,
            template <typename, typename, uint, uint> class distance_functor>
  __host__ std::pair<thrust_vector_type, thrust_vector_type> beam_search_v2(
      vertex_data_type start, vertex_data_type n_vectors_in_batch,
      search_vector_type *search_vectors, uint32_t nodes_explored_per_iteration,
      uint32_t beam_width,
      uint32_t limit = std::numeric_limits<uint32_t>::max()) {
    // const uint beam_width = 1;

    // using edge_pair_type = edge_pair_type;
    //  process each vertex independently.
    //  all ops are read-only so this is fine.
    //   candidates are atomically added all at once.
    //   in kernel using managed memory
    //   so host knows how to size vector.

    // comparisons drawn with beamSearch.h

    // line 64 - init frontier
    // set the size of each live frontier to be one to start?
    // calculate this as first call.

    uint32_t *candidate_counter;
    cudaMallocManaged((void **)&candidate_counter, sizeof(uint32_t));

    uint32_t *live_frontier_counters;
    cudaMallocManaged((void **)&live_frontier_counters,
                      sizeof(uint32_t) * (n_vectors_in_batch + 1));

    uint32_t *unvisited_starts;
    cudaMalloc((void **)&unvisited_starts,
               sizeof(uint64_t) * n_vectors_in_batch);

    uint32_t *unvisited_candidate_counters;
    cudaMallocManaged((void **)&unvisited_candidate_counters,
                      sizeof(uint64_t) * (n_vectors_in_batch + 1));

    uint32_t *host_unvisited_candidate_counters =
        gallatin::utils::get_host_version<uint32_t>(n_vectors_in_batch + 1);

    uint32_t *visited_total;
    cudaMallocManaged((void **)&visited_total, sizeof(uint32_t));

    float *frontier_cutoffs;
    cudaMallocManaged((void **)&frontier_cutoffs,
                      sizeof(float) * n_vectors_in_batch);

    for (uint i = 0; i < n_vectors_in_batch; i++) {
      frontier_cutoffs[i] = std::numeric_limits<float>::max();
    }

    thrust_vector_type frontier(0);
    thrust_vector_type unvisited_frontier(n_vectors_in_batch);
    thrust_vector_type temp_frontier(0);

    thrust_vector_type live_frontier(nodes_explored_per_iteration *
                                     n_vectors_in_batch);

    thrust_vector_type candidates(R * n_vectors_in_batch *
                                  nodes_explored_per_iteration);

    thrust_vector_type visited(0);
    thrust_vector_type temp_visited(0);

    thrust::host_vector<edge_pair_type> setup;

    // line 66-68 - populate frontier
    for (uint i = 0; i < n_vectors_in_batch; i++) {
      edge_pair_type to_fill;

      to_fill.source = start + i;
      to_fill.sink = medoid;
      to_fill.distance = 0.0;

      setup.push_back(to_fill);
    }

    frontier = setup;

    cudaDeviceSynchronize();

    populate_vector_edge_distances<search_vector_type, distance_functor>(
        frontier, search_vectors);
    // populate_edge_distances(frontier_ptr, n_vectors_in_batch);

    // start loop

    uint64_t n_loops = 0;

    do {
      // std::cout << "Round " << n_loops << ": frontier " << frontier.size()
      //           << " visited " << visited.size() << " unvisited "
      //           << unvisited_frontier.size() << std::endl;

      // if (n_loops >= 10) width = 16;
      gpu_error::static_timer<4>::start("beam round");

      beam_search_one_round<search_vector_type, distance_functor>(
          start, n_vectors_in_batch, search_vectors, frontier, visited,
          unvisited_frontier, candidates, live_frontier, temp_visited,
          temp_frontier, unvisited_starts, unvisited_candidate_counters,
          host_unvisited_candidate_counters, live_frontier_counters,
          candidate_counter, frontier_cutoffs, nodes_explored_per_iteration,
          beam_width);

      gpu_error::static_timer<4>::stop();

      n_loops++;

      // artificial cutoff enforcement.
      if (n_loops > 36) break;
      if (visited.size() > limit) break;

    } while (unvisited_frontier.size() != 0);

    if (tail_visited < n_loops) tail_visited = n_loops;

    // free memory
    cudaFree(live_frontier_counters);
    cudaFree(unvisited_starts);
    cudaFree(unvisited_candidate_counters);
    cudaFree(candidate_counter);
    cudaFreeHost(host_unvisited_candidate_counters);
    cudaFree(frontier_cutoffs);

    total_visited += visited.size();

    return {frontier, visited};
  }

  // Beam search use for querying, uses batching edge distance calculation.
  template <typename search_vector_type,
            template <typename, typename, uint, uint> class distance_functor>
  __host__ std::pair<thrust_vector_type, thrust_vector_type>
  beam_search_v2_query(vertex_data_type start,
                       vertex_data_type n_vectors_in_batch,
                       search_vector_type *search_vectors,
                       uint32_t nodes_explored_per_iteration,
                       uint32_t beam_width) {
    uint32_t *candidate_counter;
    uint32_t *live_frontier_counters;
    uint32_t *unvisited_starts;
    uint32_t *unvisited_candidate_counters;
    float *frontier_cutoffs;
    uint32_t *host_unvisited_candidate_counters =
        gallatin::utils::get_host_version<uint32_t>(n_vectors_in_batch + 1);
    cudaMallocManaged((void **)&candidate_counter, sizeof(uint32_t));
    cudaMallocManaged((void **)&live_frontier_counters,
                      sizeof(uint32_t) * (n_vectors_in_batch + 1));
    cudaMalloc((void **)&unvisited_starts,
               sizeof(uint64_t) * n_vectors_in_batch);
    cudaMallocManaged((void **)&unvisited_candidate_counters,
                      sizeof(uint64_t) * (n_vectors_in_batch + 1));
    cudaMallocManaged((void **)&frontier_cutoffs,
                      sizeof(float) * n_vectors_in_batch);

    for (uint i = 0; i < n_vectors_in_batch; i++) {
      frontier_cutoffs[i] = std::numeric_limits<float>::max();
    }

    thrust_vector_type frontier(0);
    thrust_vector_type unvisited_frontier(n_vectors_in_batch);
    thrust_vector_type temp_frontier(0);
    thrust_vector_type live_frontier(nodes_explored_per_iteration *
                                     n_vectors_in_batch);
    thrust_vector_type candidates(R * n_vectors_in_batch *
                                  nodes_explored_per_iteration);
    thrust_vector_type visited(0);
    thrust_vector_type temp_visited(0);

    thrust::host_vector<edge_pair_type> setup;

    // line 66-68 - populate frontier
    for (uint i = 0; i < n_vectors_in_batch; i++) {
      edge_pair_type to_fill;
      to_fill.source = start + i;
      to_fill.sink = medoid;
      to_fill.distance = 0.0;
      setup.push_back(to_fill);
    }
    frontier = setup;
    cudaDeviceSynchronize();

    populate_vector_edge_distances<search_vector_type, distance_functor>(
        frontier, search_vectors);

    uint64_t n_loops = 0;
    do {
      gpu_error::static_timer<4>::start("beam round");
      beam_search_one_round_query<search_vector_type, distance_functor>(
          start, n_vectors_in_batch, search_vectors, frontier, visited,
          unvisited_frontier, candidates, live_frontier, temp_visited,
          temp_frontier, unvisited_starts, unvisited_candidate_counters,
          host_unvisited_candidate_counters, live_frontier_counters,
          candidate_counter, frontier_cutoffs, nodes_explored_per_iteration,
          beam_width);
      gpu_error::static_timer<4>::stop();
      n_loops++;
    } while (unvisited_frontier.size() != 0);

    if (tail_visited < n_loops) tail_visited = n_loops;

    // free memory
    cudaFree(live_frontier_counters);
    cudaFree(unvisited_starts);
    cudaFree(unvisited_candidate_counters);
    cudaFree(candidate_counter);
    cudaFreeHost(host_unvisited_candidate_counters);
    cudaFree(frontier_cutoffs);

    total_visited += visited.size();

    return {frontier, visited};
  }

  __host__ thrust_vector_type beam_search_rerank(
      vertex_data_type start, vertex_data_type n_vectors_in_batch, int K_cutoff,
      uint16_t nodes_explored_per_iteration, uint32_t beam_width) {
    // gather top-L from frontier based on PQ_vectors
    auto [frontier, visited] =
        beam_search_v2_query<pq_vector_type, pq_distance_functor>(
            start, n_vectors_in_batch, pq_vectors, nodes_explored_per_iteration,
            beam_width);

    // rerank based on true vectors
    populate_vector_edge_distances<vector_type, full_distance_functor>(frontier,
                                                                       vectors);

    // sort and clip to top-K
    frontier =
        clip_to_beam_width(frontier, start, n_vectors_in_batch, K_cutoff);

    return frontier;
  }

  __host__ void get_frontier_cutoff(thrust_vector_type &frontier,
                                    uint64_t start,
                                    distance_type *worst_distances,
                                    uint32_t beam_width) {
    edge_pair_type *frontier_ptr = thrust::raw_pointer_cast(&frontier[0]);

    ann_kernels::frontier_cutoff_kernel<edge_pair_type, distance_type>
        <<<(frontier.size() - 1) / batch_size + 1, batch_size>>>(
            frontier_ptr, frontier.size(), start, worst_distances, beam_width);
  }

  __host__ void prune_candidates(thrust_vector_type &candidates, uint64_t start,
                                 distance_type *worst_distances) {
    uint64_t current_size = candidates.size();

    if (current_size == 0) return;
    edge_pair_type *candidates_ptr = thrust::raw_pointer_cast(&candidates[0]);

    ann_kernels::prune_candidates_cutoff_kernel<edge_pair_type, distance_type>
        <<<(candidates.size() - 1) / batch_size + 1, batch_size>>>(
            candidates_ptr, candidates.size(), start, worst_distances,
            dead_edge);

    cudaDeviceSynchronize();

    remove_dead(candidates);

    // std::cout << "pruned from " << current_size << " to " << new_size <<
    // std::endl;
  }

  // variant that allows for dead edges to already exist in the graph.
  //  also does not prune dead.
  __host__ void prune_candidates_with_dead(thrust_vector_type &candidates,
                                           uint64_t start,
                                           distance_type *worst_distances) {
    uint64_t current_size = candidates.size();

    if (current_size == 0) return;
    edge_pair_type *candidates_ptr = thrust::raw_pointer_cast(&candidates[0]);

    ann_kernels::prune_candidates_cutoff_kernel_dead<edge_pair_type,
                                                     distance_type>
        <<<(candidates.size() - 1) / batch_size + 1, batch_size>>>(
            candidates_ptr, candidates.size(), start, worst_distances,
            dead_edge);

    cudaDeviceSynchronize();
  }

  __host__ void print_edge_vector(std::string vec_name,
                                  thrust_vector_type target_vec) {
    thrust::host_vector<edge_pair_type> host_verstion = target_vec;

    std::cout << "printing " << vec_name << std::endl;
    for (uint64_t c = 0; c < host_verstion.size(); c++) {
      std::cout << host_verstion[c].source << " " << host_verstion[c].sink
                << " " << host_verstion[c].distance << std::endl;
    }
  }

  __host__ void print_last_edge(std::string vec_name,
                                thrust_vector_type target_vec) {
    thrust::host_vector<edge_pair_type> host_verstion = target_vec;

    if (target_vec.size() == 0) return;

    std::cout << "printing " << vec_name << std::endl;
    for (uint64_t c = host_verstion.size() - 1; c < host_verstion.size(); c++) {
      std::cout << host_verstion[c].source << " " << host_verstion[c].sink
                << " " << host_verstion[c].distance << std::endl;
    }
  }

  __host__ void print_edge_head(std::string vec_name,
                                thrust_vector_type target_vec) {
    if (target_vec.size() < 10) return;

    thrust::host_vector<edge_pair_type> host_verstion = target_vec;

    std::cout << "printing " << vec_name << std::endl;

    for (uint64_t c = 0; c < 10; c++) {
      std::cout << host_verstion[c].source << " " << host_verstion[c].sink
                << " " << host_verstion[c].distance << std::endl;
    }
  }

  // execute beam search for a large # of nodes in parallel.
  __host__ void beam_search_and_prune(vertex_data_type &n_vectors_in_batch,
                                      uint32_t nodes_explored_per_iteration,
                                      double alpha) {
    cudaDeviceSynchronize();

    gpu_error::static_timer<100>::start("Construct: Beam search time");
    auto [frontier_result, visited_pack] =
        beam_search_single<vertex_data_type, vector_data_type, vector_degree,
                           distance_type, edge_list_type, 64,
                           full_distance_functor, true, L_cap+64, 4, true>(
            edges, edge_counts, vectors, n_vertices, vectors + n_vertices,
            n_vectors_in_batch, medoid, L_cap, L_cap, 10.0, 512);
    auto [visited_results, visited_counts] = visited_pack;
    CUDA_CHECK(cudaDeviceSynchronize());
    // std::cout << "1. beam search            ";
    // checkGpuMem();
    gpu_error::static_timer<100>::stop();

    gpu_error::static_timer<101>::start("Construct: Move to vector time");
    thrust::device_vector<uint32_t> d_visited_counts(
        visited_counts, visited_counts + n_vectors_in_batch);
    thrust::device_vector<uint32_t> d_visited_offsets(n_vectors_in_batch);
    thrust::exclusive_scan(d_visited_counts.begin(), d_visited_counts.end(),
                           d_visited_offsets.begin());
    uint32_t total_visited = d_visited_counts.back() + d_visited_offsets.back();
    thrust::device_vector<edge_pair_type> d_visited_edges(total_visited);
    thrust::device_vector<edge_pair_type> d_frontier_edges(n_vectors_in_batch);
    CUDA_CHECK(cudaDeviceSynchronize());

    int threads = 128;
    int blocks = (n_vectors_in_batch + threads - 1) / threads;
    convert_to_edges_kernel<<<blocks, threads>>>(
        thrust::raw_pointer_cast(d_visited_counts.data()), visited_results,
        frontier_result, thrust::raw_pointer_cast(d_visited_edges.data()),
        thrust::raw_pointer_cast(d_frontier_edges.data()), n_vectors_in_batch,
        n_vertices, 1024, thrust::raw_pointer_cast(d_visited_offsets.data()));
    CUDA_CHECK(cudaDeviceSynchronize());
    // std::cout << "2. move to vector         ";
    cudaFree(frontier_result);
    cudaFree(visited_results);
    cudaFree(visited_counts);
    // checkGpuMem();
    gpu_error::static_timer<101>::stop();

    gpu_error::static_timer<102>::start("Construct: add_new_edges");
    thrust_vector_type out_neighbors_of_batch =
        add_new_edges<false>(d_visited_edges, alpha);
    CUDA_CHECK(cudaDeviceSynchronize());
    gpu_error::static_timer<102>::stop();

    if (out_neighbors_of_batch.size() != 0){

      gpu_error::static_timer<103>::start("Construct: Flip edges time");
      flip_edges(out_neighbors_of_batch);
      gpu_error::static_timer<103>::stop();

      gpu_error::static_timer<104>::start("Construct: make_sorted_and_unique");
      make_sorted_and_unique(out_neighbors_of_batch);
      gpu_error::static_timer<104>::stop();

      gpu_error::static_timer<105>::start("Construct: add_new_edges_in_place");
      add_new_edges_in_place<true>(out_neighbors_of_batch, alpha);
      cudaDeviceSynchronize();
      gpu_error::static_timer<105>::stop();

    }

    n_vertices += n_vectors_in_batch;

    return;
  }

  __host__ void print_medoid() {
    ann_kernels::print_medoid_kernel<edge_list_type>
        <<<1, 1>>>(edges, edge_counts, medoid);
    cudaDeviceSynchronize();
  }

  // process a batch - performs setup of batch size before moving to kernel
  // launch.
  struct batch_metrics {
    vertex_data_type processed;
    double elapsed_seconds;
  };

  __host__ batch_metrics process_batch(vertex_data_type &n_vectors,
                                       uint32_t nodes_explored_per_iteration,
                                       double alpha) {
    vertex_data_type n_vectors_to_process = 0;
    if (current_batch_size > n_vectors) {
      // Process all n_vectors in a single batch if they fit
      // in current_batch_size
      n_vectors_to_process = n_vectors;
      n_vectors = 0;

    } else {
      n_vectors_to_process = current_batch_size;
      current_batch_size *= 2;
      if (current_batch_size > max_batch_size)
        current_batch_size = max_batch_size;
      n_vectors -= n_vectors_to_process;
    }


    // std::cout << "Batch: processing=" << n_vectors_to_process
    //       << " next_batch_size=" << current_batch_size << std::endl;

    // current head is n_vertices.
    // incrementing from n_vertices to n_vertices_max.

    gpu_error::static_timer<0>::start("Total batch time");
    const auto batch_start = std::chrono::steady_clock::now();
    beam_search_and_prune(n_vectors_to_process, nodes_explored_per_iteration,
                          alpha);
    const auto batch_end = std::chrono::steady_clock::now();

    gpu_error::static_timer<0>::stop();

    const auto batch_duration = std::chrono::duration_cast<std::chrono::duration<double>>(batch_end - batch_start);
    return {n_vectors_to_process, batch_duration.count()};
  }

  __host__ bool BFS() {
    uint32_t round = 1;

    uint32_t *set_round;

    cudaMallocManaged((void **)&set_round, sizeof(uint32_t) * n_vertices);

    uint64_t *round_discovered;

    cudaMallocManaged((void **)&round_discovered, sizeof(uint64_t));

    uint64_t *total_discovered;

    cudaMallocManaged((void **)&total_discovered, sizeof(uint64_t));

    cudaMemset(set_round, 0, sizeof(uint32_t) * n_vertices);

    cudaDeviceSynchronize();

    round_discovered[0] = 1;
    total_discovered[0] = 1;

    set_round[medoid] = round;

    cudaDeviceSynchronize();

    while (round_discovered[0] != 0 && total_discovered[0] != n_vertices) {
      std::cout << "starting round " << round << std::endl;
      round_discovered[0] = 0;

      cudaDeviceSynchronize();

      ann_kernels::ann_bfs_kernel<edge_list_type>
          <<<(n_vertices - 1) / 256 + 1, 256>>>(
              edges, n_vertices, edge_counts, set_round, round,
              round_discovered, total_discovered);

      cudaDeviceSynchronize();

      std::cout << "Round " << round << std::endl;

      std::cout << "New nodes " << round_discovered[0] << ", total so far "
                << total_discovered[0] << "/" << n_vertices << std::endl;

      round++;
    }

    std::cout << "BFS done, total nodes discovered " << total_discovered[0]
              << "/" << n_vertices << std::endl;

    bool result = total_discovered[0] == n_vertices;
    cudaFree(total_discovered);
    cudaFree(round_discovered);
    cudaFree(set_round);

    return result;
  }

  // Set the index of the starting node for graph traversal.
  __host__ void set_medoid(vertex_data_type ext_medoid) { medoid = ext_medoid; }

  // Set the maximum batch size for incremental construction.
  __host__ void set_max_batch_size(vertex_data_type new_batch_size) { 
    max_batch_size = new_batch_size; 
  }

  // Construct the graph for one round. This function divides the vectors
  // to process into batches and call process_batch() on them.
  // Argument:
  //   - ext_n_vertices: number of vertices to process.
  //   - alpha: pruning alpha
  __host__ void construction_round(vertex_data_type ext_n_vertices,
                                   uint32_t nodes_explored_per_iteration,
                                   double alpha) {
    n_vertices = 0;
    const auto round_start = std::chrono::steady_clock::now();

    uint32_t inc = 0;
    uint32_t base = 2;
    uint32_t count = 0;

    while (count < ext_n_vertices) {
      uint32_t floor, ceiling;
      
      if (pow(base, inc) <= max_batch_size) {
        floor = static_cast<uint32_t>(pow(base, inc)) - 1;
        ceiling = std::min(static_cast<uint32_t>(pow(base, inc + 1)) - 1, ext_n_vertices);
        count = std::min(static_cast<uint32_t>(pow(base, inc + 1)) - 1, ext_n_vertices);
      } else {
        floor = count;
        ceiling = std::min(count + max_batch_size, ext_n_vertices);
        count += max_batch_size;
      }
      
      current_batch_size = ceiling - floor;
      vertex_data_type n_vectors_left = ext_n_vertices - floor;

      if (current_batch_size > 0) {
        const auto metrics = process_batch(n_vectors_left, nodes_explored_per_iteration, alpha);
        const auto now = std::chrono::steady_clock::now();
        const auto elapsed = now - round_start;
        const double elapsed_seconds =
            std::max(std::chrono::duration_cast<std::chrono::duration<double>>(elapsed).count(), 1e-9);
        const double ratio =
            ext_n_vertices == 0 ? 1.0 : static_cast<double>(n_vertices) / static_cast<double>(ext_n_vertices);
        const auto estimated_total =
            ratio > 0.0 ? std::chrono::duration_cast<std::chrono::steady_clock::duration>(elapsed / ratio)
                        : std::chrono::steady_clock::duration::zero();
        const auto eta =
            ratio > 0.0 ? estimated_total - elapsed : std::chrono::steady_clock::duration::zero();
        const double batch_qps =
            metrics.elapsed_seconds > 0.0 ? static_cast<double>(metrics.processed) / metrics.elapsed_seconds : 0.0;
        const double total_qps = static_cast<double>(n_vertices) / elapsed_seconds;

        std::cout << "Constructing graph "
                  << std::setw(3) << static_cast<int>(ratio * 100.0) << "% "
                  << "(" << n_vertices << "/" << ext_n_vertices << ") "
                  << "elapsed " << format_progress_duration(elapsed) << ' ';
        if (ratio > 0.0 && n_vertices < ext_n_vertices) {
          std::cout << "eta " << format_progress_duration(eta) << ' ';
        } else {
          std::cout << "eta 0s ";
        }
        std::cout << "last_batch=" << metrics.processed << ' '
                  << "batch_qps=" << std::fixed << std::setprecision(1) << batch_qps << ' '
                  << "total_qps=" << total_qps << std::defaultfloat << std::endl;
      }
      
      inc++;
    }

    assert(n_vertices == ext_n_vertices);
  }

  __host__ thrust_vector_type query_exact(vector_type *query_vectors,
                                          vertex_data_type n_query_vectors,
                                          uint64_t k,
                                          uint16_t nodes_explored_per_iteration,
                                          uint32_t beam_width, uint32_t limit) {
    // 1 - copy into vector memory and PQ vector memory
    cudaMemcpy(vectors + n_vertices, query_vectors,
               sizeof(vector_type) * n_query_vectors, cudaMemcpyDefault);

    auto [frontier, visited] =
        beam_search_v2<vector_type, full_distance_functor>(
            n_vertices, n_query_vectors, vectors, nodes_explored_per_iteration,
            beam_width, limit);

    frontier = clip_to_beam_width(frontier, n_vertices, n_query_vectors, k);

    return frontier;
  }

  // return the top K
  __host__ thrust_vector_type query(vector_type *query_vectors,
                                    pq_vector_type *query_pq_vectors,
                                    vertex_data_type n_query_vectors,
                                    uint64_t k,
                                    uint16_t nodes_explored_per_iteration,
                                    uint32_t beam_width) {
    // 1 - copy into vector memory and PQ vector memory
    cudaMemcpy(vectors + n_vertices, query_vectors,
               sizeof(vector_type) * n_query_vectors, cudaMemcpyDefault);

    cudaMemcpy(pq_vectors + n_vertices, query_pq_vectors,
               sizeof(pq_vector_type) * n_query_vectors, cudaMemcpyDefault);

    auto frontier =
        beam_search_rerank(n_vertices, n_query_vectors, k,
                           nodes_explored_per_iteration, beam_width);

    return frontier;
  }

  // add n_vectors vectors to the graph.
  __host__ void construct(vector_type *new_vectors, vertex_data_type n_vectors,
                          uint32_t n_rounds,
                          uint32_t nodes_explored_per_iteration,
                          bool random_init = true,
                          double alpha = 1.2) {
    if (n_vectors <= medoid) {
      std::cerr << "Medoid must be in range\n" << std::endl;
    }

    // first - copy vectors into memory, and start processing.
    cudaMemcpy(vectors, new_vectors, sizeof(vector_type) * n_vectors,
               cudaMemcpyDefault);

    // cudaMemcpy(pq_vectors, new_pq_vectors, sizeof(pq_vector_type) *
    // n_vectors,
    //            cudaMemcpyDefault);

    // initialize graph randomly
    if (random_init) {
      uint32_t *random_bytes =
          gpu_ann::random_data_device<uint32_t>(R * n_vectors);
      printf("Data generated, populating...\n");
      ann_kernels::populate_graph_kernel<edge_list_type, R>
          <<<(n_vectors - 1) / batch_size + 1, batch_size>>>(
              edges, edge_counts, n_vectors, random_bytes);
      cudaDeviceSynchronize();
      cudaFree(random_bytes);
      printf("Graph populated.\n");
    }

    for (uint i = 0; i < n_rounds - 1; i++) {
      construction_round(n_vectors, nodes_explored_per_iteration, 1.0);
    }

    construction_round(n_vectors, nodes_explored_per_iteration, alpha);
  }

  __host__ void print_edge_statistics() {
    uint64_t *statistics;

    cudaMallocManaged((void **)&statistics, sizeof(uint64_t) * 5);

    statistics[0] = ~0ULL;
    for (uint i = 1; i < 5; i++) {
      statistics[i] = 0;
    }

    ann_kernels::generate_edge_stats_kernel<<<(n_vertices - 1) / batch_size + 1,
                                              batch_size>>>(
        edge_counts, n_vertices, statistics);

    cudaDeviceSynchronize();

    printf("N_verts: %u, Min %lu Max %lu avg: %f\n", n_vertices, statistics[0],
           statistics[2], 1.0 * statistics[1] / n_vertices);

    cudaFree(statistics);

    // std::cout << "Total visited " << total_visited
    //           << " avg: " << 1.0 * total_visited / n_vertices
    //           << ", tail: " << tail_visited << std::endl;
  }

  // persist the graph to memory.
  __host__ void write_out(std::string output_fname, vector_type *originals) {
    // precalculations:
    // 1. size of each vector + graph
    // 2. n_vectors per sector
    // 3. n_sectors
    // 4. total file size.

    std::string md_filename = output_fname + "_metadata.bin";
    std::string vector_filename = output_fname + ".bin";

    // helper buffer
    std::vector<char> zeros(4096, 0);

    // and move edge counts;

    uint8_t *host_edge_counts =
        gallatin::utils::get_host_version<uint8_t>(n_vertices);

    cudaMemcpy(host_edge_counts, edge_counts, sizeof(uint8_t) * n_vertices,
               cudaMemcpyDeviceToHost);

    uint64_t bytes_per_vector = sizeof(vector_type);

    uint64_t bytes_per_neighbor_list = sizeof(edge_list_type);

    uint64_t bytes_per_node = bytes_per_vector + bytes_per_neighbor_list + 4;

    uint64_t nodes_per_sector = (4096) / bytes_per_node;

    uint64_t n_sectors = (n_vertices - 1) / nodes_per_sector + 1;

    uint64_t total_file_size = 4096ULL * (n_sectors + 1);

    uint64_t big_n_vectors = n_vertices;

    uint32_t full_vector_degree = data_vector_traits<vector_type>::size;

    std::cout << "Writing out " << total_file_size << " bytes"
              << ", " << n_sectors << " sectors each containing "
              << nodes_per_sector << " vectors using " << bytes_per_node
              << " bytes." << std::endl;

    std::ofstream mdFile(md_filename);
    std::ofstream outputFile(vector_filename);

    if (!mdFile.is_open()) {
      std::cerr << "Failed to open " << md_filename << " for graph write\n";
    }

    if (!outputFile.is_open()) {
      std::cerr << "Failed to open " << vector_filename << " for graph write\n";
    }

    // BANG output
    // 1. medoid
    uint64_t medoid_as_uint64_t = medoid;
    mdFile.write(reinterpret_cast<const char *>(&medoid_as_uint64_t),
                 sizeof(uint64_t));

    // entry len - bytes
    mdFile.write(reinterpret_cast<const char *>(&bytes_per_node),
                 sizeof(uint64_t));

    // 3. datatype
    if (std::is_same<vector_data_type, int8_t>::value) {
      int32_t value = 0;

      mdFile.write(reinterpret_cast<const char *>(&value), sizeof(int32_t));

    } else if (std::is_same<vector_data_type, uint8_t>::value) {
      int32_t value = 1;

      mdFile.write(reinterpret_cast<const char *>(&value), sizeof(int32_t));

    } else if (std::is_same<vector_data_type, float>::value) {
      int32_t value = 2;

      mdFile.write(reinterpret_cast<const char *>(&value), sizeof(int32_t));

    } else {
      std::cerr << " BANG will not recognize type" << std::endl;
    }

    // 4. Dimension of vector
    mdFile.write(reinterpret_cast<const char *>(&full_vector_degree),
                 sizeof(uint32_t));

    // 5 . N_data_points
    uint32_t R_clone = R;

    mdFile.write(reinterpret_cast<const char *>(&R_clone), sizeof(uint32_t));

    mdFile.write(reinterpret_cast<const char *>(&n_vertices), sizeof(uint32_t));

    gpu_error::progress_bar bar("Writing Sectors", n_sectors, .01);

    uint64_t n_vectors_written = 0;

    for (uint64_t i = 0; i < n_sectors; i++) {
      for (uint i = 0; i < nodes_per_sector; i++) {
        if (n_vectors_written < n_vertices) {
          outputFile.write(
              reinterpret_cast<const char *>(&originals[n_vectors_written]),
              sizeof(vector_type));

          uint32_t n_neighbors = host_edge_counts[n_vectors_written];

          outputFile.write(reinterpret_cast<const char *>(&n_neighbors),
                           sizeof(uint32_t));

          std::sort(edges[n_vectors_written].edges,
                    edges[n_vectors_written].edges + n_neighbors);

          outputFile.write(
              reinterpret_cast<const char *>(&edges[n_vectors_written]),
              sizeof(edge_list_type));

          n_vectors_written++;

        } else {
          outputFile.write(zeros.data(), bytes_per_node);
        }
      }

      // pad with zeros.
      outputFile.write(zeros.data(), 4096 - bytes_per_node * nodes_per_sector);

      bar.increment();
    }
  }

  __host__ void write_out_diskANN(std::string output_fname,
                                  vector_type *originals) {
    // precalculations:
    // 1. size of each vector + graph
    // 2. n_vectors per sector
    // 3. n_sectors
    // 4. total file size.

    std::string vector_filename = output_fname + ".index";

    // helper buffer
    std::vector<char> zeros(4096, 0);

    // and move edge counts;
    uint8_t *host_edge_counts =
        gallatin::utils::get_host_version<uint8_t>(n_vertices);
    cudaMemcpy(host_edge_counts, edge_counts, sizeof(uint8_t) * n_vertices,
               cudaMemcpyDeviceToHost);

    // If edges are on device, create a host version
    edge_list_type *host_edges;
    if (on_host) {
      host_edges = edges;
    } else {
      host_edges =
          gallatin::utils::get_host_version<edge_list_type>(n_vertices_max);
      cudaError_t cuda_status =
          cudaMemcpy(host_edges, edges, sizeof(edge_list_type) * n_vertices_max,
                     cudaMemcpyDeviceToHost);
      if (cuda_status != cudaSuccess) {
        std::cerr << "CUDA memcpy failed: " << cudaGetErrorString(cuda_status)
                  << std::endl;
        return;
      }
    }

    const uint64_t header_size = 4 * sizeof(uint64_t);
    uint64_t bytes_per_vector = sizeof(vector_type);
    uint64_t bytes_per_neighbor_list = sizeof(edge_list_type);
    uint64_t bytes_per_node = bytes_per_vector + bytes_per_neighbor_list + 1;

    uint64_t total_file_size = header_size + bytes_per_node * n_vertices;

    uint64_t big_n_vectors = n_vertices;

    std::cout << "Writing out " << total_file_size << " bytes"
      << " n_vertices=" << big_n_vectors 
      << " medoid=" << medoid 
      << " bytes_per_node=" << bytes_per_node
      << std::endl;

    std::ofstream outputFile(vector_filename, std::ios::binary);

    if (!outputFile.is_open()) {
      std::cerr << "Failed to open " << vector_filename << " for graph write\n";
    }

    // 1. total file size:
    outputFile.write(reinterpret_cast<const char *>(&total_file_size),
                     sizeof(uint64_t));
    // 2 n_nodes
    outputFile.write(reinterpret_cast<const char *>(&big_n_vectors),
                     sizeof(uint64_t));
    // 3. medoid
    uint64_t medoid_as_uint64_t = medoid;
    outputFile.write(reinterpret_cast<const char *>(&medoid_as_uint64_t),
                     sizeof(uint64_t));
    // 4. entry len in bytes.
    outputFile.write(reinterpret_cast<const char *>(&bytes_per_node),
                     sizeof(uint64_t));

    // gpu_error::progress_bar bar("Writing Sectors", n_sectors, .01);

    for (uint64_t i = 0; i < big_n_vectors; i++) {
      outputFile.write(reinterpret_cast<const char *>(&originals[i]),
                       sizeof(vector_type));

      uint8_t n_neighbors = host_edge_counts[i];

      outputFile.write(reinterpret_cast<const char *>(&n_neighbors),
                       sizeof(uint8_t));

      std::sort(host_edges[i].edges, host_edges[i].edges + n_neighbors);

      outputFile.write(reinterpret_cast<const char *>(&host_edges[i]),
                       sizeof(edge_list_type));

      // bar.increment();
    }
  }

  __host__ void load_from_diskANN(std::string input_fname,
                                          vertex_data_type n_vectors) {
    std::string vector_filename = input_fname + ".index";

    std::ifstream inputFile(vector_filename, std::ios::binary);
    if (!inputFile.is_open()) {
      std::cerr << "Failed to open " << vector_filename << " for graph load\n";
      exit(EXIT_FAILURE);
    }

    uint64_t total_file_size;
    uint64_t big_n_vectors;
    uint64_t medoid_as_uint64;
    uint64_t bytes_per_node;

    // Read the metadata
    inputFile.read(reinterpret_cast<char *>(&total_file_size),
                   sizeof(uint64_t));
    inputFile.read(reinterpret_cast<char *>(&big_n_vectors), sizeof(uint64_t));
    inputFile.read(reinterpret_cast<char *>(&medoid_as_uint64),
                   sizeof(uint64_t));
    inputFile.read(reinterpret_cast<char *>(&bytes_per_node), sizeof(uint64_t));

    medoid = static_cast<uint32_t>(medoid_as_uint64);
    n_vertices = big_n_vectors;
    n_vertices_max = big_n_vectors;

    std::cout << "total_file_size: " << total_file_size << "\n";
    std::cout << "big_n_vectors: " << big_n_vectors << "\n";
    std::cout << "medoid: " << medoid << "\n";
    std::cout << "bytes_per_node: " << bytes_per_node << "\n";

    uint8_t *host_edge_counts =
        gallatin::utils::get_host_version<uint8_t>(big_n_vectors);
    edge_list_type *host_edges =
        gallatin::utils::get_host_version<edge_list_type>(big_n_vectors);
    vector_type *host_vectors =
        gallatin::utils::get_host_version<vector_type>(big_n_vectors);

    for (uint i = 0; i < big_n_vectors; i++) {
      // Read vector
      inputFile.read(reinterpret_cast<char *>(&host_vectors[i]),
                     sizeof(vector_type));

      // Read neighbor count
      uint8_t n_neighbors;
      inputFile.read(reinterpret_cast<char *>(&n_neighbors), sizeof(uint8_t));
      host_edge_counts[i] = n_neighbors;

      // Read neighbor list
      inputFile.read(reinterpret_cast<char *>(&host_edges[i]),
                     sizeof(edge_list_type));
    }

    if (on_host) {
      edge_counts = host_edge_counts;
      edges = host_edges;
      vectors = host_vectors;
    } else {
      // move edge counts and edges to the ann struct
      cudaMemcpy(edge_counts, host_edge_counts, sizeof(uint8_t) * n_vertices,
                 cudaMemcpyHostToDevice);
      cudaMemcpy(edges, host_edges, sizeof(edge_list_type) * n_vertices_max,
                 cudaMemcpyHostToDevice);
      cudaMemcpy(vectors, host_vectors, sizeof(vector_type) * n_vertices,
                 cudaMemcpyHostToDevice);
      cudaError_t err = cudaFreeHost(host_edge_counts);
      if (err != cudaSuccess) {
        printf("cudaFree error: %s\n", cudaGetErrorString(err));
      }
      err = cudaFreeHost(host_edges);
      if (err != cudaSuccess) {
        printf("cudaFree error: %s\n", cudaGetErrorString(err));
      }
      err = cudaFreeHost(host_vectors);
      if (err != cudaSuccess) {
        printf("cudaFree error: %s\n", cudaGetErrorString(err));
      }
    }

    cudaDeviceSynchronize();
    return;
  }

  __host__ float calculate_accuracy(thrust_vector_type query_results,
                                   thrust_vector_type ground_truth,
                                   uint64_t n_query_vectors, uint k) {
    if (query_results.size() != n_query_vectors * k) {
      std::cerr << "Bad query result selection" << std::endl;
    }

    edge_pair_type *query_results_ptr =
        thrust::raw_pointer_cast(&query_results[0]);

    edge_pair_type *ground_truth_ptr =
        thrust::raw_pointer_cast(&ground_truth[0]);

    uint64_t *n_correct;

    cudaMallocManaged((void **)&n_correct, sizeof(uint64_t));

    n_correct[0] = 0;

    cudaDeviceSynchronize();

    ann_kernels::calculate_accuracy_kernel<edge_pair_type>
        <<<(query_results.size() - 1) / batch_size + 1, batch_size>>>(
            query_results_ptr, ground_truth_ptr, n_query_vectors, n_correct, k);

    cudaDeviceSynchronize();

    uint64_t correct_amount = n_correct[0];

    cudaFree(n_correct);

    std::cout << "Found " << correct_amount << "/" << n_query_vectors * k
              << " total correct, avg. accuracy is "
              << 1.0 * correct_amount / (n_query_vectors * k) << "/" << k
              << std::endl;

    return 1.0 * correct_amount / (n_query_vectors * k);
  }

  thrust_vector_type generate_bulk_distances(uint64_t query_start,
                                             uint64_t n_query_vectors,
                                             uint64_t start, uint64_t end) {
    thrust_vector_type distances((end - start) * n_query_vectors);

    edge_pair_type *distances_ptr = thrust::raw_pointer_cast(&distances[0]);

    ann_kernels::populate_edges_from_vector_kernel<vertex_data_type,
                                                   edge_pair_type>
        <<<(distances.size() - 1) / batch_size + 1, batch_size>>>(
            distances_ptr, query_start, n_query_vectors, start, end);

    cudaDeviceSynchronize();

    populate_vector_edge_distances<vector_type, full_distance_functor>(
        distances, vectors);

    make_sorted_and_unique(distances);

    // print_edge_vector("Bulk dist", distances);

    return distances;
  }
  // bulk generation.
  __host__ thrust_vector_type generate_ground_truth(std::string fname,
                                                    vector_type *query_vectors,
                                                    uint64_t n_query_vectors,
                                                    uint k) {
    using host_vector_type = thrust::host_vector<edge_pair_type>;
    // NEW - memoization
    namespace fs = std::filesystem;

    fs::path input_path(fname);
    std::string filename_only = input_path.filename().string();

    std::string outname = std::to_string(n_query_vectors) + "_" +
                          std::to_string(k) + "_" + filename_only;

    // cache outside of build
    fs::path dir = "../gt";
    fs::create_directories(dir);

    fs::path cache_path = dir / outname;

    if (fs::exists(cache_path)) {
      std::ifstream infile(cache_path, std::ios::binary);
      if (!infile) {
        throw std::runtime_error("Failed to open cache file: " +
                                 cache_path.string());
      }

      std::cout << "Reading ground truth from " << cache_path << std::endl;

      // host_vector_type vec;
      // edge_pair_type value;

      // Get file size
      infile.seekg(0, std::ios::end);
      size_t num_bytes = infile.tellg();
      infile.seekg(0, std::ios::beg);

      // Resize and read
      host_vector_type vec(num_bytes / sizeof(edge_pair_type));

      infile.read(reinterpret_cast<char *>(vec.data()), num_bytes);

      thrust_vector_type device_vector = vec;
      return device_vector;

    } else {
      std::cout << "Generating from fresh " << outname << std::endl;
      // copy into shared
      cudaMemcpy(vectors + n_vertices, query_vectors,
                 sizeof(vector_type) * n_query_vectors, cudaMemcpyDefault);

      uint64_t gt_batch_size = 100;

      uint64_t start = 0;

      thrust_vector_type visited(0);

      thrust_vector_type temp_visited(0);

      while (n_vertices - start > gt_batch_size) {
        auto distances = generate_bulk_distances(n_vertices, n_query_vectors,
                                                 start, start + gt_batch_size);

        thrust_merge(visited, distances, temp_visited);

        distances.resize(0);

        visited = temp_visited;

        visited = clip_to_beam_width(visited, n_vertices, n_query_vectors, k);

        start += gt_batch_size;
      }

      auto distances = generate_bulk_distances(n_vertices, n_query_vectors,
                                               start, n_vertices);

      thrust_merge(visited, distances, temp_visited);

      distances.resize(0);

      visited = temp_visited;

      visited = clip_to_beam_width(visited, n_vertices, n_query_vectors, k);

      std::ofstream outfile(cache_path, std::ios::binary);
      if (!outfile) {
        throw std::runtime_error("Failed to write to cache file: " +
                                 cache_path.string());
      }

      host_vector_type vec = visited;

      outfile.write(reinterpret_cast<const char *>(vec.data()),
                    vec.size() * sizeof(edge_pair_type));

      return visited;
    }
  }
};

}  // namespace gpu_ann

#endif  // GPU_BLOCK_
