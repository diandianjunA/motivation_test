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
#include <limits>

#include "assert.h"
#include "stdio.h"

namespace cg = cooperative_groups;

namespace gpu_ann {

template <typename DATA_T, uint16_t DIM>
__global__ void centroid_kernel(const data_vector<DATA_T, DIM> *vectors,
                                size_t n_vectors, double *partial_sums) {
  // Each thread handles one dimension
  int d = blockIdx.x * blockDim.x + threadIdx.x;
  if (d >= DIM) return;

  double sum = 0;
  for (size_t i = 0; i < n_vectors; i++) {
    sum += static_cast<double>(vectors[i].data[d]);
  }

  partial_sums[d] = sum;
}

template <typename DATA_T, uint16_t DIM, bool FROM_HOST=true>
data_vector<DATA_T, DIM> compute_centroid(
    data_vector<DATA_T, DIM> *vectors, size_t n_vectors) {
  data_vector<DATA_T, DIM> *d_vectors;
  if (FROM_HOST) {
    cudaMalloc(&d_vectors, n_vectors * sizeof(data_vector<DATA_T, DIM>));
    cudaMemcpy(d_vectors, vectors, n_vectors * sizeof(data_vector<DATA_T, DIM>), cudaMemcpyHostToDevice);
  } else {
    d_vectors = vectors;
  }
  
  // Allocate buffer for sums
  double *d_sums, *h_sums;
  cudaMalloc(&d_sums, DIM * sizeof(double));
  h_sums = new double[DIM];

  // One thread per dimension
  int threads = 256;
  int blocks = (DIM + threads - 1) / threads;
  centroid_kernel<DATA_T, DIM><<<blocks, threads>>>(d_vectors, n_vectors, d_sums);
  cudaDeviceSynchronize();

  cudaMemcpy(h_sums, d_sums, DIM * sizeof(double), cudaMemcpyDeviceToHost);

  data_vector<DATA_T, DIM> centroid;
  for (int d = 0; d < DIM; d++) {
    centroid.data[d] = static_cast<DATA_T>(h_sums[d] / n_vectors);
  }

  delete[] h_sums;
  cudaFree(d_sums);

  if (FROM_HOST) {
    cudaFree(d_vectors);
  }

  return centroid;
}

template <typename DATA_T, uint16_t DIM>
__global__ void distance_to_centroid_kernel(const data_vector<DATA_T, DIM> *vectors,
                                            size_t n_vectors, const DATA_T *centroid,
                                            float *distances) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n_vectors) return;

  float dist = 0.0f;
  for (int d = 0; d < DIM; d++) {
    float diff = (float)vectors[i].data[d] - (float)centroid[d];
    dist += diff * diff;  // squared L2 distance
  }
  distances[i] = dist;
}

template <typename DATA_T, uint16_t DIM>
int compute_medoid_via_centroid(const data_vector<DATA_T, DIM> *h_vectors,
                                size_t n_vectors,
                                const data_vector<DATA_T, DIM> &centroid_vec) {
  data_vector<DATA_T, DIM> *d_vectors;
  cudaMalloc(&d_vectors, n_vectors * sizeof(data_vector<DATA_T, DIM>));
  cudaMemcpy(d_vectors, h_vectors, n_vectors * sizeof(data_vector<DATA_T, DIM>),
             cudaMemcpyHostToDevice);

  // Copy centroid to device
  DATA_T *d_centroid;
  cudaMalloc(&d_centroid, DIM * sizeof(DATA_T));
  cudaMemcpy(d_centroid, centroid_vec.data, DIM * sizeof(DATA_T),
             cudaMemcpyHostToDevice);

  // Allocate distance array
  float *d_distances, *h_distances;
  cudaMalloc(&d_distances, n_vectors * sizeof(float));
  h_distances = new float[n_vectors];

  // Launch kernel
  int threads = 256;
  int blocks = (n_vectors + threads - 1) / threads;
  distance_to_centroid_kernel<DATA_T, DIM>
      <<<blocks, threads>>>(d_vectors, n_vectors, d_centroid, d_distances);
  cudaDeviceSynchronize();

  // Copy results back
  cudaMemcpy(h_distances, d_distances, n_vectors * sizeof(float),
             cudaMemcpyDeviceToHost);

  // Find argmin on host
  int best_idx = 0;
  float best_val = std::numeric_limits<float>::max();
  for (size_t i = 0; i < n_vectors; i++) {
    if (h_distances[i] < best_val) {
      best_val = h_distances[i];
      best_idx = (int)i;
    }
  }

  delete[] h_distances;
  cudaFree(d_distances);
  cudaFree(d_centroid);
  cudaFree(d_vectors);

  return best_idx;  // index of medoid
}

}  // namespace gpu_ann