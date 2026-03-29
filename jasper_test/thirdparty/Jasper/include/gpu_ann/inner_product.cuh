#pragma once

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cooperative_groups/scan.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

#include <cmath>
#include <cuda/std/cmath>
#include <cuda/std/cstdint>
#include <fstream>
#include <gallatin/allocators/alloc_utils.cuh>
#include <gallatin/data_structs/ds_utils.cuh>
#include <gpu_ann/_proj_metadata.cuh>
#include <gpu_ann/edge_list.cuh>
#include <gpu_error/log.cuh>
#include <iostream>
#include <limits>
#include <vector>

#include "assert.h"
#include "stdio.h"

namespace cg = cooperative_groups;

// To search for inner product on our graph index, we have to reduce it from
// MIPS problem to NNS problem. The technique here we are using is xbox mapping.
// Originally proposed by this paper:
// https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/XboxInnerProduct.pdf

namespace gpu_ann {

// To reduce a data vector x from MIPS to NNS, we append one dimension
// with value sqrt(M^2-||x||^2) to it, where M = max(||x||) for x in
// all data vectors.
// TODO: optimize reduction on gpu.
template <typename VECTOR_DATA_TYPE, uint32_t VECTOR_SIZE>
data_vector<VECTOR_DATA_TYPE, VECTOR_SIZE + 1>*
inner_product_reduce_data_vectors( 
    data_vector<VECTOR_DATA_TYPE, VECTOR_SIZE> *input, uint64_t n_vectors) {
  // find M
  float M = 0;
  for (uint i = 0; i < n_vectors; i++) {
    float dist = 0;
    for (uint j = 0; j < VECTOR_SIZE; j++) {
      dist += static_cast<float>(input[i].data[j]) *
              static_cast<float>(input[i].data[j]);
    }
    M = max(dist, M);
  }

  // allocate and assign new vectors
  data_vector<VECTOR_DATA_TYPE, VECTOR_SIZE + 1> *new_vectors;
  cudaMallocHost(
      &new_vectors,
      sizeof(data_vector<VECTOR_DATA_TYPE, VECTOR_SIZE + 1>) * n_vectors);

  for (uint i = 0; i < n_vectors; i++) {
    float dist = 0;
    for (uint j = 0; j < VECTOR_SIZE; j++) {
      new_vectors[i].data[j] = input[i].data[j];
      dist += static_cast<float>(input[i].data[j]) *
              static_cast<float>(input[i].data[j]);
    }
    // sqrt(M^2-||x||^2)
    new_vectors[i].data[VECTOR_SIZE] = std::sqrt(M - dist);
  }

  return new_vectors;
}

// To reduce a query vector from MIPS to NNS, we append an empty dimension with
// value 0 to it.
template <typename VECTOR_DATA_TYPE, uint32_t VECTOR_SIZE>
data_vector<VECTOR_DATA_TYPE, VECTOR_SIZE + 1>*
inner_product_reduce_query_vectors(
  data_vector<VECTOR_DATA_TYPE, VECTOR_SIZE> *input, 
  uint64_t n_vectors
) {
  // allocate and assign new vectors
  data_vector<VECTOR_DATA_TYPE, VECTOR_SIZE + 1> *new_vectors;
  cudaMallocHost(
      &new_vectors,
      sizeof(data_vector<VECTOR_DATA_TYPE, VECTOR_SIZE + 1>) * n_vectors);
  for (uint i = 0; i < n_vectors; i++) {
    for (uint j = 0; j < VECTOR_SIZE; j++) {
      new_vectors[i].data[j] = input[i].data[j];
    }
    new_vectors[i].data[VECTOR_SIZE] = 0;
  }

  return new_vectors;
}

}  // namespace gpu_ann
