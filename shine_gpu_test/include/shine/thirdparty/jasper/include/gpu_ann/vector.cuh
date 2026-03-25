#ifndef DATA_VECTOR
#define DATA_VECTOR

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cooperative_groups/scan.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

#include <fstream>
#include <gallatin/allocators/alloc_utils.cuh>
#include <gallatin/data_structs/ds_utils.cuh>
#include <gpu_ann/_proj_metadata.cuh>
#include <gpu_ann/randomness.cuh>
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

// representation of a GPU vector
template <typename data_type, uint n_degrees>
struct __align__(16) data_vector {
  data_type data[n_degrees];

  // self populate based on ptr to raw data
  __host__ data_vector(data_type* ptr_to_data) {
    for (uint i = 0; i < n_degrees; i++) {
      data[i] = ptr_to_data[i];
    }
  }

  // Unified default constructor
  __host__ __device__ data_vector() {
  #ifdef __CUDA_ARCH__
      // Device: do nothing
  #else
      // Host: zero-init
      for (uint i = 0; i < n_degrees; i++) {
          data[i] = static_cast<data_type>(0);
      }
  #endif
  }

  __host__ __device__ data_type& operator[](uint index) { return data[index]; }

  __host__ __device__ const data_type& operator[](uint index) const {
    return data[index];
  }

  // print the vector to std::cout
  // requires vector to be on host.
  __host__ void print() {
    std::cout << "Vector: [";
    for (uint i = 0; i < n_degrees; i++) {
      std::cout << +data[i];

      if (i != n_degrees - 1) {
        std::cout << " ";
      }
    }
    std::cout << "]" << std::endl;
  }

  // copy data from one vector to another
  // starting at an offset.
  template <typename other_data_type, uint n_degrees_other>
  __device__ void copy_vector_offset(
      const data_vector<other_data_type, n_degrees_other>& other_vector,
      uint offset) {
    uint j = 0;
    for (uint i = offset; i < n_degrees_other; i++) {
      data[j++] = other_vector.data[i];

      if (j == n_degrees) return;
    }
  }

  //
  template <typename other_data_type>
  __device__ void copy_vector_one_thread(
      data_vector<other_data_type, n_degrees>* other_vector) {
    for (uint i = 0; i < n_degrees; i++) {
      data[i] = other_vector->data[i];
    }
    return;
  }

  // merge a (smaller) vector into some accumulation vector
  //  for correctness, type of this vector must be large enough to hold all
  //  data.
  template <typename other_data_type, uint tile_size>
  __device__ void merge(data_vector<other_data_type, n_degrees>* other_vector,
                        cg::thread_block_tile<tile_size> work_tile) {
    for (uint i = work_tile.thread_rank(); i < n_degrees;
         i += work_tile.size()) {
      atomicAdd(&data[i], (data_type)other_vector->data[i]);
    }

    work_tile.sync();
  }

  // Merge a new vector into an old vector - alpha controls how quickly
  // convergence towards new points occurs.
  template <uint tile_size>
  __device__ void EMS(data_vector<data_type, n_degrees>* other_vector,
                      double alpha,
                      cg::thread_block_tile<tile_size> work_tile) {
    for (uint i = work_tile.thread_rank(); i < n_degrees;
         i += work_tile.size()) {
      data[i] = ((data_type)data[i] * (1 - alpha)) +
                ((data_type)other_vector->data[i] * alpha);
    }

    work_tile.sync();
  }

  // normalize an existing vector.
  template <uint tile_size>
  __device__ void normalize(uint n_vectors,
                            cg::thread_block_tile<tile_size> work_tile) {
    if (n_vectors == 0) {
      // printf("Cluster has no items!\n");
      return;
    }
    for (uint i = work_tile.thread_rank(); i < n_degrees;
         i += work_tile.size()) {
      data[i] = data[i] / n_vectors;
    }

    __threadfence();
    work_tile.sync();
  }
};

// helper clas to load a file
// asserts typing matches.
template <typename data_type, uint n_degrees>
struct vector_loader {
  using vector_type = data_vector<data_type, n_degrees>;

  // load from a file and populate n_vectors.
  // data returned is pinned host memory.
  __host__ static vector_type* load_from_file(std::string filename,
                                              uint64_t& n_vectors) {
    std::ifstream file(filename, std::ios::in | std::ios::binary);
    if (!file.is_open()) {
      std::cerr << "Error opening file: " << filename << std::endl;
      return nullptr;
    }
    file.seekg(0, std::ios::end);  // go to end of file
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    size = size - 8;

    int n_data_points;
    int n_dimensions;
    file.read(reinterpret_cast<char*>(&n_data_points), 4);
    file.read(reinterpret_cast<char*>(&n_dimensions), 4);

    std::cout << "Read " << size << " bytes of data" << std::endl;

    std::cout << "Read value is " << n_data_points << " with dim "
              << n_dimensions << std::endl;

    if (sizeof(data_type) * n_data_points * n_dimensions != size) {
      std::cerr << "DIM mismatch: "
                << " sizeof(data_type)=" << sizeof(data_type)
                << " n_data_points=" << n_data_points
                << " n_dimensions=" << n_dimensions
                << sizeof(data_type) * n_data_points * n_dimensions
                << " != " << size << std::endl;
      return nullptr;
    } else {
      std::cout << "Dimensional calcs match" << std::endl;
    }

    if (size % sizeof(data_type) != 0) {
      std::cerr << "File size is not a multiple of data_type with size "
                << sizeof(data_type) << std::endl;
      return nullptr;
    }

    n_vectors = size / (n_degrees * sizeof(data_type));

    if ((size % n_degrees) != 0) {
      std::cerr << "vector stride does not align with data: "
                << size % n_degrees << " over.\n";
    }

    printf("Size of vector type is %lu, loading %lu vectors\n",
           sizeof(vector_type), n_vectors);
    vector_type* data =
        gallatin::utils::get_host_version<vector_type>(n_vectors);

    if (file.read(reinterpret_cast<char*>(data), size)) {
      std::cout << "Successfully read " << n_vectors << " vectors from "
                << filename << std::endl;
    } else {
      std::cerr << "Error while reading the file." << std::endl;
    }

    return data;
  }

  __host__ static void save_to_file(std::string filename, vector_type* vectors,
                                    uint32_t n_vectors) {
    std::ofstream file(filename, std::ios::out | std::ios::binary);
    if (!file.is_open()) {
      std::cerr << "Error opening file: " << filename << std::endl;
      return;
    }

    int n_data_points = n_vectors;
    int n_dimensions = n_degrees;
    file.write(reinterpret_cast<char*>(&n_data_points), 4);
    file.write(reinterpret_cast<char*>(&n_dimensions), 4);
    file.write(reinterpret_cast<char*>(vectors),
               sizeof(vector_type) * n_vectors);

    return;
  }
};

template <typename T>
struct data_vector_traits;

template <typename data_type, uint Size>
struct data_vector_traits<data_vector<data_type, Size>> {
  using type = data_type;
  static constexpr uint size = Size;
};

// generate random vectors in a range
template <typename vector_data_type, uint vector_dim>
__host__ data_vector<vector_data_type, vector_dim>*
generate_random_bounded_vectors(vector_data_type min, vector_data_type max,
                                uint64_t n_vectors) {
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