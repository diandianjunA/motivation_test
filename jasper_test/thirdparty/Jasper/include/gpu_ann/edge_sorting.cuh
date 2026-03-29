#ifndef EDGE_SORTING
#define EDGE_SORTING

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cooperative_groups/scan.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>

#include <fstream>
#include <gallatin/allocators/alloc_utils.cuh>
#include <gallatin/data_structs/ds_utils.cuh>
#include <gpu_ann/_proj_metadata.cuh>
#include <iostream>
#include <vector>

#include "assert.h"
#include "stdio.h"

namespace cg = cooperative_groups;

namespace gpu_ann {

template <typename data_type, typename dist_type>
struct edge_pair {
  using my_type = edge_pair<data_type, dist_type>;
  data_type source;
  data_type sink;
  dist_type distance;

  __host__ __device__ friend bool operator<(const my_type& l,
                                            const my_type& r) {
    if (l.source != r.source) return l.source < r.source;

    return l.distance < r.distance;
  }

  __host__ __device__ friend bool operator==(const my_type& l,
                                             const my_type& r) {
    return l.source == r.source && l.sink == r.sink;
  }

  // input/output io helpers
  friend std::ostream& operator<<(std::ostream& os, const my_type& edges) {
    os << edges.source << " " << edges.sink << " " << edges.distance;
    return os;
  }

  friend std::istream& operator>>(std::istream& input, my_type& obj) {
    input >> obj.source;
    input >> obj.sink;
    input >> obj.distance;
    return input;
  }
};

template <typename data_type, typename dist_type>
struct beamSearchComparator {
  using edge_pair_type = edge_pair<data_type, dist_type>;
  __host__ __device__ bool operator()(const edge_pair_type& l,
                                      const edge_pair_type& r) {
    if (l.source != r.source) return l.source < r.source;

    if (l.distance != r.distance) return l.distance < r.distance;

    return l.sink < r.sink;
  }
};

template <typename data_type, typename dist_type>
struct beamSearchGreaterComparator {
  using edge_pair_type = edge_pair<data_type, dist_type>;
  __device__ bool operator()(const edge_pair_type& l, const edge_pair_type& r) {
    if (l.source != r.source) return l.source > r.source;
    if (l.sink != r.sink) return l.sink > r.sink;

    return l.distance > r.distance;
  }
};

template <typename data_type, typename dist_type>
__host__ __device__ static constexpr edge_pair<data_type, dist_type>
get_dead_edge() {
  return {(data_type)~0ULL, (data_type)~0ULL,
          cuda::std::numeric_limits<dist_type>::max()};
}

template <typename data_type, typename dist_type>
struct pruneBadEdgeComparator {
  __host__ __device__ bool operator()(edge_pair<data_type, dist_type> x) const {
    return x == get_dead_edge<data_type, dist_type>();
  }
};

template <typename data_type, typename dist_type>
struct pruneEqualityComparator {
  using edge_pair_type = edge_pair<data_type, dist_type>;
  __device__ bool operator()(const edge_pair_type& l, const edge_pair_type& r) {
    return l.source == r.source;
  }
};

// v1 of sort - use thrust edges
template <typename data_type, typename dist_type>
edge_pair<data_type, dist_type>* semisort_edge_pairs_thrust(
    edge_pair<data_type, dist_type>* pairs, uint64_t n_edge_pairs) {
  thrust::sort(thrust::device, pairs, pairs + n_edge_pairs);

  return pairs;
}

}  // namespace gpu_ann

#endif  // GPU_BLOCK_