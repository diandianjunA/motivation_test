#ifndef TOP_K
#define TOP_K

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cooperative_groups/scan.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

#include <fstream>
#include <gallatin/allocators/alloc_utils.cuh>
#include <gallatin/data_structs/ds_utils.cuh>
#include <gpu_ann/_proj_metadata.cuh>
#include <gpu_ann/distance.cuh>
#include <gpu_ann/randomness.cuh>
#include <gpu_ann/vector.cuh>
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

// generate random vectors in a range
template <typename vector_data_type, uint vector_dim, uint K,
          template <typename, typename, uint, uint> class distance_functor>
__host__ data_vector<uint32_t, K>* generate_random_bounded_vectors(
    vector_data_type min, vector_data_type max, uint64_t n_vectors) {
  uint64_t dim = (max - min);

  uint64_t n_random_points_to_generate = vector_dim * n_vectors;

  using vector_type = data_vector<vector_data_type, vector_dim>;

  vector_type* output_vectors =
      gallatin::utils::get_host_version<vector_type>(n_vectors);

  int32_t* data = random_data_host<int32_t>(n_random_points_to_generate);

  for (uint i = 0; i < n_vectors; i++) {
    for (uint j = 0; j < vector_dim; j++) {
      output_vectors[i][j] = (data[i * vector_dim + j] % dim) + min;
    }
  }

  cudaFreeHost(data);

  return output_vectors;
}

}  // namespace gpu_ann

#endif  // GPU_BLOCK_