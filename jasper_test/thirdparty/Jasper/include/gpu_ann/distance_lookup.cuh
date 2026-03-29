#pragma once

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <fstream>
#include <iterator>

// alloc utils needed for easy host_device transfer
// and the global allocator
#include <gallatin/allocators/alloc_utils.cuh>
#include <gpu_error/fixed_vector.cuh>
#include <gpu_error/gpu_singleton.cuh>
#include <gpu_error/progress_bar.cuh>
#include <gpu_ann/product_quantizer.cuh>
#include <string>

#define N_CENTROIDS 256

namespace gpu_ann {

// helper kernel to calculate distance of two quantized vectors
// using the distance lookup table
template <typename quantized_type, uint quantized_vector_size, uint tile_size>
__device__ float distance_lookup(
  float* distances,
  const quantized_type& vector1,
  const quantized_type& vector2,
  cg::thread_block_tile<tile_size> &my_tile
) {
  float distance = 0;
  for (uint i=my_tile.thread_rank(); i<quantized_vector_size; i+=my_tile.size()) {
    auto field1 = vector1[i];
    auto field2 = vector2[i];
    distance += distances[i*N_CENTROIDS*N_CENTROIDS+field1*N_CENTROIDS + field2];
  }
  float result = sqrt(
    cg::reduce(my_tile, distance, cg::plus<float>())
  );
  return result;
}

// helper kernel to calculate distance of two quantized vectors
// using the distance lookup table with no sqrt.
template <typename quantized_type, uint quantized_vector_size, uint tile_size>
__device__ float distance_lookup_nosqrt(
  float* distances,
  const quantized_type& vector1,
  const quantized_type& vector2,
  cg::thread_block_tile<tile_size> &my_tile
) {
  float distance = 0;
  for (uint i=my_tile.thread_rank(); i<quantized_vector_size; i+=my_tile.size()) {
    auto field1 = vector1[i];
    auto field2 = vector2[i];
    distance += distances[i*N_CENTROIDS*N_CENTROIDS+field1*N_CENTROIDS + field2];
  }
  float result = cg::reduce(my_tile, distance, cg::plus<float>());
  // Only if the distance is less than 1, we calculate the sqrt.
  if (result < 1) {
    result = sqrt((double)result);
  }
  return result;
}

// define global gpu lookup distance table
using distance_lookup_singleton = gpu_error::gpu_singleton<float *>;

static __host__ void init_distance_lookup_singleton(
  uint n_codebooks,
  uint n_centroids,
  float *device_distances
){
  uint64_t size = n_codebooks * n_centroids * n_centroids;
  float *distance_lookup = gallatin::utils::get_device_version<float>(
    size
  );
  cudaMemcpy(distance_lookup, device_distances, size*sizeof(float),
             cudaMemcpyDefault);
  distance_lookup_singleton::write_instance(distance_lookup);
}

static __host__ void free_distance_lookup_singleton(){
  float *distance_lookup = distance_lookup_singleton::read_instance();
  cudaFree(distance_lookup);
  distance_lookup_singleton::write_instance(nullptr);
}


// "safe" L2 distance using lookup table.
template <typename l_data_type, typename r_data_type, uint n_degrees,
          uint tile_size>
struct safe_lookup_distance {
  static_assert(
    std::is_same<l_data_type, r_data_type>::value, 
    "l_data_type and r_data_type must be the same type."
  );
  using acc_type = typename promote_to_float<l_data_type>::type;
  static __device__ acc_type
  distance(const data_vector<l_data_type, n_degrees>& l_vector,
           const data_vector<r_data_type, n_degrees>& r_vector,
           cg::thread_block_tile<tile_size>& work_tile) {
    
    acc_type result = distance_lookup<data_vector<l_data_type, n_degrees>, n_degrees, tile_size>(
      distance_lookup_singleton::instance(),
      l_vector,
      r_vector,
      work_tile
    );

    gpu_assert(!is_bad(result), "Result is corrupted\n");
    return result;
  }
};

// "safe" L2 distance using lookup table with no sqrt.
template <typename l_data_type, typename r_data_type, uint n_degrees,
          uint tile_size>
struct safe_lookup_distance_no_sqrt {
  static_assert(
    std::is_same<l_data_type, r_data_type>::value, 
    "l_data_type and r_data_type must be the same type."
  );
  using acc_type = typename promote_to_float<l_data_type>::type;
  static __device__ acc_type
  distance(const data_vector<l_data_type, n_degrees>& l_vector,
           const data_vector<r_data_type, n_degrees>& r_vector,
           cg::thread_block_tile<tile_size>& work_tile) {
    
    acc_type result = distance_lookup_nosqrt<data_vector<l_data_type, n_degrees>, n_degrees, tile_size>(
      distance_lookup_singleton::instance(),
      l_vector,
      r_vector,
      work_tile
    );

    gpu_assert(!is_bad(result), "Result is corrupted\n");
    return result;
  }
};

}