#pragma once

// Jasper: Embeddable GPU-Accelerated Approximate Nearest Neighbor Index
//
// This header provides a clean C++ API wrapping the Jasper GPU ANN system.
// It supports build, search, insert, save, and load operations.
//
// Usage:
//   jasper::JasperIndex<128> index;          // 128-dim uint8 vectors
//   index.build(vectors, n, params);
//   index.search(queries, n_queries, k, results_ids, results_dists);
//   index.save("my_index");
//   index.load("my_index");

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <thrust/pair.h>

#include <gallatin/allocators/alloc_utils.cuh>
#include <gpu_ann/beam_search.cuh>
#include <gpu_ann/bulk_gpuANN.cuh>
#include <gpu_ann/centroid.cuh>
#include <gpu_ann/distance.cuh>
#include <gpu_ann/vector.cuh>
#include <gpu_error/log.cuh>

namespace jasper {

// Build parameters for index construction
struct BuildParams {
  uint32_t n_rounds = 1;
  uint32_t nodes_explored_per_iteration = 4;
  bool random_init = false;
  double alpha = 1.2;
  double max_batch_ratio = 0.02;
};

// Search parameters
struct SearchParams {
  uint32_t beam_width = 64;
  float cut = 10.0f;
  uint32_t limit = 512;
};

// Search result for a single query
struct SearchResult {
  uint32_t *ids;       // [n_queries * k] neighbor indices (device memory)
  float *distances;    // [n_queries * k] distances (device memory)
};

namespace detail {

#define JASPER_CUDA_CHECK(call)                                    \
  do {                                                             \
    cudaError_t err = (call);                                      \
    if (err != cudaSuccess) {                                      \
      throw std::runtime_error(                                    \
          std::string("CUDA error: ") + cudaGetErrorString(err) +  \
          " at " + __FILE__ + ":" + std::to_string(__LINE__));     \
    }                                                              \
  } while (0)

// Initialize gpu_error log (idempotent via static flag)
inline void ensure_gpu_log_init() {
  static bool initialized = false;
  if (!initialized) {
    gpu_error::init_gpu_log(256ULL * 1024 * 1024);
    initialized = true;
  }
}

}  // namespace detail

// JasperIndex: GPU-accelerated graph-based ANN index
//
// Template parameters:
//   VECTOR_DIM   - dimensionality of vectors (e.g. 128)
//   DATA_T       - element type of vectors (default: uint8_t)
//   R            - max out-degree per vertex (default: 64)
//   L_CAP        - beam search capacity (default: 64)
//   ON_HOST      - store graph on host memory (default: false = GPU)
template <uint32_t VECTOR_DIM,
          typename DATA_T = uint8_t,
          uint32_t R = 64,
          uint32_t L_CAP = 64,
          bool ON_HOST = false>
class JasperIndex {
 public:
  // Internal type aliases
  using index_t = uint32_t;
  using distance_t = float;
  using vector_type = gpu_ann::data_vector<DATA_T, VECTOR_DIM>;
  using pq_data_t = uint8_t;
  static constexpr uint32_t PQ_SIZE = 8;
  using pq_vector_type = gpu_ann::data_vector<pq_data_t, PQ_SIZE>;
  using edge_list_type = gpu_ann::edge_list<index_t, R>;

  using ann_type = gpu_ann::bulk_gpuANN<
      1, 512, index_t, distance_t, DATA_T, VECTOR_DIM,
      pq_data_t, PQ_SIZE, R, L_CAP,
      gpu_ann::euclidean_distance_no_sqrt_chunked,
      gpu_ann::safe_lookup_distance_no_sqrt, ON_HOST>;

  JasperIndex() = default;

  ~JasperIndex() {
    if (ann_) {
      delete ann_;
      ann_ = nullptr;
    }
    free_host_vectors();
  }

  // Non-copyable
  JasperIndex(const JasperIndex &) = delete;
  JasperIndex &operator=(const JasperIndex &) = delete;

  // Movable
  JasperIndex(JasperIndex &&other) noexcept
      : ann_(other.ann_),
        host_vectors_(other.host_vectors_),
        n_vectors_(other.n_vectors_),
        built_(other.built_) {
    other.ann_ = nullptr;
    other.host_vectors_ = nullptr;
    other.n_vectors_ = 0;
    other.built_ = false;
  }

  JasperIndex &operator=(JasperIndex &&other) noexcept {
    if (this != &other) {
      if (ann_) delete ann_;
      free_host_vectors();
      ann_ = other.ann_;
      host_vectors_ = other.host_vectors_;
      n_vectors_ = other.n_vectors_;
      built_ = other.built_;
      other.ann_ = nullptr;
      other.host_vectors_ = nullptr;
      other.n_vectors_ = 0;
      other.built_ = false;
    }
    return *this;
  }

  // Build an index from raw vectors.
  //
  // vectors: pointer to n_vectors * VECTOR_DIM elements of DATA_T
  //          in row-major layout, stored in host (CPU) memory.
  // n_vectors: number of vectors
  // params: build parameters (optional)
  void build(const DATA_T *vectors, uint64_t n_vectors,
             const BuildParams &params = {}) {
    detail::ensure_gpu_log_init();

    if (ann_) {
      delete ann_;
      ann_ = nullptr;
    }
    free_host_vectors();

    n_vectors_ = n_vectors;

    // Copy raw data into pinned host vector_type array
    host_vectors_ =
        gallatin::utils::get_host_version<vector_type>(n_vectors);
    memcpy(host_vectors_, vectors,
           sizeof(DATA_T) * VECTOR_DIM * n_vectors);

    // Compute medoid
    auto centroid =
        gpu_ann::compute_centroid<DATA_T, VECTOR_DIM>(
            host_vectors_, n_vectors);
    auto medoid =
        gpu_ann::compute_medoid_via_centroid<DATA_T, VECTOR_DIM>(
            host_vectors_, n_vectors, centroid);

    // Create ANN and build graph
    ann_ = new ann_type(n_vectors, params.max_batch_ratio);
    ann_->set_medoid(medoid);
    ann_->construct(host_vectors_, n_vectors, params.n_rounds,
                    params.nodes_explored_per_iteration,
                    params.random_init, params.alpha);

    built_ = true;
  }

  // Search for k nearest neighbors.
  //
  // queries: pointer to n_queries * VECTOR_DIM elements of DATA_T
  //          in row-major layout, stored in host (CPU) memory.
  // n_queries: number of query vectors
  // k: number of nearest neighbors to return
  // out_ids: output array of size n_queries * k (host memory, caller-allocated)
  // out_distances: output array of size n_queries * k (host memory,
  //               caller-allocated, optional — pass nullptr to skip)
  // params: search parameters (optional)
  void search(const DATA_T *queries, uint64_t n_queries, uint32_t k,
              uint32_t *out_ids, float *out_distances = nullptr,
              const SearchParams &params = {}) const {
    if (!built_) {
      throw std::runtime_error("JasperIndex: index not built or loaded");
    }

    // Copy query vectors to device
    vector_type *h_queries =
        gallatin::utils::get_host_version<vector_type>(n_queries);
    memcpy(h_queries, queries,
           sizeof(DATA_T) * VECTOR_DIM * n_queries);

    vector_type *d_queries;
    JASPER_CUDA_CHECK(
        cudaMalloc(&d_queries, sizeof(vector_type) * n_queries));
    JASPER_CUDA_CHECK(cudaMemcpy(
        d_queries, h_queries, sizeof(vector_type) * n_queries,
        cudaMemcpyHostToDevice));

    // Select beam search instantiation based on beam_width
    using entry_t = thrust::pair<index_t, distance_t>;
    entry_t *frontier_results = nullptr;

    uint32_t bw = params.beam_width;
    if (bw <= 64) {
      frontier_results = do_beam_search<136>(
          d_queries, n_queries, k, params);
    } else if (bw <= 128) {
      frontier_results = do_beam_search<208>(
          d_queries, n_queries, k, params);
    } else if (bw <= 256) {
      frontier_results = do_beam_search<352>(
          d_queries, n_queries, k, params);
    } else {
      frontier_results = do_beam_search<1024>(
          d_queries, n_queries, k, params);
    }

    // Copy results back to host
    std::vector<entry_t> h_results(n_queries * k);
    JASPER_CUDA_CHECK(cudaMemcpy(
        h_results.data(), frontier_results,
        sizeof(entry_t) * n_queries * k, cudaMemcpyDeviceToHost));

    for (uint64_t i = 0; i < n_queries * k; i++) {
      out_ids[i] = h_results[i].first;
      if (out_distances) {
        out_distances[i] = h_results[i].second;
      }
    }

    // Cleanup
    cudaFree(frontier_results);
    cudaFree(d_queries);
    cudaFreeHost(h_queries);
  }

  // Save the index to disk.
  // Creates file at path + ".index"
  void save(const std::string &path) const {
    if (!built_) {
      throw std::runtime_error("JasperIndex: index not built or loaded");
    }
    if (!host_vectors_) {
      throw std::runtime_error(
          "JasperIndex: host vectors not available for save");
    }
    ann_->write_out_diskANN(path, host_vectors_);
  }

  // Load an index from disk.
  // Reads from path + ".index"
  // n_vectors: number of vectors in the index (must match the saved index)
  void load(const std::string &path, uint64_t n_vectors) {
    detail::ensure_gpu_log_init();

    if (ann_) {
      delete ann_;
      ann_ = nullptr;
    }
    free_host_vectors();

    n_vectors_ = n_vectors;
    ann_ = new ann_type(n_vectors, 0.02);
    ann_->load_from_diskANN(path, n_vectors);

    // If on_host or we need host_vectors_ for future save, extract them
    if constexpr (ON_HOST) {
      host_vectors_ = ann_->vectors;
    } else {
      host_vectors_ =
          gallatin::utils::get_host_version<vector_type>(n_vectors);
      JASPER_CUDA_CHECK(cudaMemcpy(
          host_vectors_, ann_->vectors,
          sizeof(vector_type) * n_vectors, cudaMemcpyDeviceToHost));
    }

    built_ = true;
  }

  // Insert additional vectors into an already-built index.
  //
  // new_vectors: pointer to n_new * VECTOR_DIM elements of DATA_T
  //              in row-major layout, stored in host (CPU) memory.
  // n_new: number of new vectors to insert
  // params: build parameters for the incremental construction
  //
  // Note: The index must have been built with enough capacity.
  //       Total vectors after insert must not exceed the original
  //       n_vectors passed to build().
  void insert(const DATA_T *new_vectors, uint64_t n_new,
              const BuildParams &params = {}) {
    if (!built_) {
      throw std::runtime_error("JasperIndex: index not built or loaded");
    }

    uint64_t old_n = ann_->n_vertices;
    uint64_t new_total = old_n + n_new;

    if (new_total > ann_->n_vertices_max) {
      throw std::runtime_error(
          "JasperIndex: insert would exceed max capacity (" +
          std::to_string(ann_->n_vertices_max) + ")");
    }

    // Prepare combined host vectors
    vector_type *combined =
        gallatin::utils::get_host_version<vector_type>(new_total);
    if (host_vectors_) {
      memcpy(combined, host_vectors_, sizeof(vector_type) * old_n);
      free_host_vectors();
    }
    memcpy(combined + old_n, new_vectors,
           sizeof(DATA_T) * VECTOR_DIM * n_new);
    host_vectors_ = combined;

    // Recompute medoid on the full set
    auto centroid =
        gpu_ann::compute_centroid<DATA_T, VECTOR_DIM>(
            host_vectors_, new_total);
    auto medoid =
        gpu_ann::compute_medoid_via_centroid<DATA_T, VECTOR_DIM>(
            host_vectors_, new_total, centroid);
    ann_->set_medoid(medoid);

    // Run incremental construction
    ann_->construct(host_vectors_, new_total, params.n_rounds,
                    params.nodes_explored_per_iteration,
                    params.random_init, params.alpha);

    n_vectors_ = new_total;
  }

  // Get the number of vectors in the index
  uint64_t size() const { return n_vectors_; }

  // Check if the index has been built or loaded
  bool is_built() const { return built_; }

  // Access the underlying ann object (advanced usage)
  ann_type *raw() { return ann_; }
  const ann_type *raw() const { return ann_; }

 private:
  template <uint32_t MAX_SEARCH_WIDTH>
  thrust::pair<index_t, distance_t> *do_beam_search(
      vector_type *d_queries, uint64_t n_queries, uint32_t k,
      const SearchParams &params) const {
    auto beam_fn = gpu_ann::beam_search_single<
        index_t, DATA_T, VECTOR_DIM, distance_t,
        edge_list_type, 64,
        gpu_ann::euclidean_distance_no_sqrt_chunked,
        false, MAX_SEARCH_WIDTH, 4>;

    auto result = beam_fn(
        ann_->edges, ann_->edge_counts, ann_->vectors,
        n_vectors_, d_queries, n_queries, ann_->medoid, k,
        params.beam_width, params.cut, params.limit);
    return result.first;
  }

  void free_host_vectors() {
    if (host_vectors_ && !(ON_HOST && ann_)) {
      cudaFreeHost(host_vectors_);
    }
    host_vectors_ = nullptr;
  }

  ann_type *ann_ = nullptr;
  vector_type *host_vectors_ = nullptr;
  uint64_t n_vectors_ = 0;
  bool built_ = false;
};

// Convenience type aliases for common configurations
using JasperIndex128 = JasperIndex<128, uint8_t, 64, 64, false>;
using JasperIndex96 = JasperIndex<96, uint8_t, 64, 64, false>;
using JasperIndex256 = JasperIndex<256, uint8_t, 64, 64, false>;
using JasperIndexFloat128 = JasperIndex<128, float, 64, 64, false>;

}  // namespace jasper
