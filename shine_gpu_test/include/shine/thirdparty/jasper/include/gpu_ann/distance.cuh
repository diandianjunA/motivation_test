#ifndef DISTANCE
#define DISTANCE

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cooperative_groups/scan.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

#include <fstream>
#include <gallatin/allocators/alloc_utils.cuh>
#include <gallatin/data_structs/ds_utils.cuh>
#include <gpu_ann/_proj_metadata.cuh>
#include <gpu_ann/vector.cuh>
#include <gpu_error/loads.cuh>
#include <gpu_error/log.cuh>
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

enum MetricType : std::uint8_t { METRIC_L2, METRIC_IP };

// helpers for calculating distance
template <typename T>
__device__ T auto_pow(T data, float power) {
  if constexpr ((sizeof(data) > sizeof(float))) {
    return (T)pow((double)data, (double)power);
  } else {
    return (T)pow((float)data, power);
  }
}

template <typename T>
__device__ T auto_sqrt(T data) {
  if constexpr ((sizeof(data) > sizeof(float))) {
    return (T)sqrt((double)data);
  } else {
    return (T)sqrt((float)data);
  }
}

// isnan/isinf helper
template <typename T>
__device__ inline bool is_bad(T val) {
  if constexpr (std::is_same<T, float>::value) {
    unsigned int bits = __float_as_uint(val);
    return (bits & 0x7f800000) == 0x7f800000;  // exponent all 1s = NaN or Inf
  } else if constexpr (std::is_same<T, double>::value) {
    unsigned long long bits = __double_as_longlong(val);
    return (bits & 0x7ff0000000000000ULL) == 0x7ff0000000000000ULL;
  } else {
    static_assert(sizeof(T) == 0, "Unsupported type for is_bad()");
  }
}

template <typename T>
struct promote_to_float {
  using type = float;
};

template <>
struct promote_to_float<double> {
  using type = double;
};

template <>
struct promote_to_float<float> {
  using type = float;
};

// default template instantiation
template <typename data_type, uint n_degrees, uint tile_size>
__device__ data_type
dummy_distance(data_vector<data_type, n_degrees>& l_vector,
               data_vector<data_type, n_degrees>& r_vector,
               cg::thread_block_tile<tile_size>& work_tile) {
  // data_type dist = 0;
  return 0;
}

// taxicab is sum of absolute difference between two vectors, calculated
// dim-by-dim
//  so [1, 2] and [0,0] have dist of abs(1-0) + abs(2-0) = 3.
template <typename data_type, uint n_degrees, uint tile_size>
__device__ data_type
taxicab_distance(data_vector<data_type, n_degrees>& l_vector,
                 data_vector<data_type, n_degrees>& r_vector,
                 cg::thread_block_tile<tile_size>& work_tile) {
  data_type local_dist = 0;
  for (int i = work_tile.thread_rank(); i < n_degrees; i += work_tile.size()) {
    local_dist += abs(l_vector.data[i] - r_vector.data[i]);
  }

  return cg::reduce(work_tile, local_dist, cg::plus<data_type>());
}

// distance functor for Euclidean_dist
template <typename l_data_type, typename r_data_type, uint n_degrees,
          uint tile_size>
struct euclidean_distance {
  static __device__ l_data_type
  distance(data_vector<l_data_type, n_degrees>& l_vector,
           data_vector<r_data_type, n_degrees>& r_vector,
           cg::thread_block_tile<tile_size>& work_tile) {
    l_data_type local_dist = 0;
    for (int i = work_tile.thread_rank(); i < n_degrees;
         i += work_tile.size()) {
      local_dist += auto_pow(l_vector.data[i] - r_vector.data[i], 2);
    }

    return auto_sqrt(
        cg::reduce(work_tile, local_dist, cg::plus<l_data_type>()));
  }
};

//"safe" version that accounts for NAN
template <typename l_data_type, typename r_data_type, uint n_degrees,
          uint tile_size>
struct safe_euclidean_distance {
  // Promote to higher precision for safety if needed
  using acc_type = typename promote_to_float<l_data_type>::type;

  static __device__ acc_type
  distance(const data_vector<l_data_type, n_degrees>& l_vector,
           const data_vector<r_data_type, n_degrees>& r_vector,
           cg::thread_block_tile<tile_size>& work_tile) {
    acc_type local_sum = 0.0;

#if COUNT_DIST
    if (work_tile.thread_rank() == 0) {
      count_event(0);
    }
#endif

    for (int i = work_tile.thread_rank(); i < n_degrees;
         i += work_tile.size()) {
      acc_type diff = static_cast<acc_type>(l_vector.data[i]) -
                      static_cast<acc_type>(r_vector.data[i]);

      gpu_assert(!is_bad(diff), "Diff is corrupted\n");
      // if (is_bad(diff)) {
      //     // Optionally, print for debugging
      //     printf("Bad value at dim %d: l=%.4f, r=%.4f\n", i,
      //            static_cast<float>(l_vector.data[i]),
      //            static_cast<float>(r_vector.data[i]));
      //     diff = 0.0; // Neutralize this contribution
      // }

      local_sum += diff * diff;
    }

    // Use cooperative groups to reduce across threads
    acc_type total_sum = cg::reduce(work_tile, local_sum, cg::plus<acc_type>());

    acc_type result = sqrt((double)total_sum);  // cast back to original type

    // Final safety check

    gpu_assert(!is_bad(result), "Result is corrupted\n");

    return result;
  }
};

template <typename l_data_type, typename r_data_type, uint n_degrees,
          uint tile_size>
struct euclidean_distance_no_sqrt {
  // Promote to higher precision for safety if needed
  using acc_type = typename promote_to_float<l_data_type>::type;

  static __device__ acc_type
  distance(const data_vector<l_data_type, n_degrees>* l_vector,
           const data_vector<r_data_type, n_degrees>* r_vector,
           cg::thread_block_tile<tile_size>& work_tile) {
    acc_type local_sum = 0.0;

#if COUNT_DIST
    if (work_tile.thread_rank() == 0) {
      count_event(0);
    }
#endif

    for (int i = work_tile.thread_rank(); i < n_degrees;
         i += work_tile.size()) {
      acc_type diff = static_cast<acc_type>(l_vector->data[i]) -
                      static_cast<acc_type>(r_vector->data[i]);

      gpu_assert(!is_bad(diff), "Diff is corrupted\n");
      local_sum += diff * diff;
    }

    // Use cooperative groups to reduce across threads
    acc_type total_sum = cg::reduce(work_tile, local_sum, cg::plus<acc_type>());

    acc_type result;
    if (total_sum < 1) {
      result = sqrt((double)total_sum);
    } else {
      result = total_sum;
    }

    // Final safety check
    gpu_assert(!is_bad(result), "Result is corrupted\n");

    return result;
  }
};

template <typename l_data_type, typename r_data_type, uint n_degrees,
          uint tile_size>
struct euclidean_distance_no_sqrt_compiler {
  // Promote to higher precision for safety if needed
  using acc_type = typename promote_to_float<l_data_type>::type;

  static __device__ acc_type
  distance(const data_vector<l_data_type, n_degrees> l_vector,
           const data_vector<r_data_type, n_degrees> r_vector,
           cg::thread_block_tile<tile_size>& work_tile) {
    acc_type local_sum = 0.0;

#if COUNT_DIST
    if (work_tile.thread_rank() == 0) {
      count_event(0);
    }
#endif

    for (int i = work_tile.thread_rank(); i < n_degrees;
         i += work_tile.size()) {
      acc_type diff = static_cast<acc_type>(l_vector.data[i]) -
                      static_cast<acc_type>(r_vector.data[i]);

      gpu_assert(!is_bad(diff), "Diff is corrupted\n");
      local_sum += diff * diff;
    }

    // Use cooperative groups to reduce across threads
    acc_type total_sum = cg::reduce(work_tile, local_sum, cg::plus<acc_type>());

    acc_type result;
    if (total_sum < 1) {
      result = sqrt((double)total_sum);
    } else {
      result = total_sum;
    }

    // Final safety check
    gpu_assert(!is_bad(result), "Result is corrupted\n");

    return result;
  }
};

__device__ __forceinline__ void ldg_cg(uint4& x, const uint4* addr)
{
  asm volatile("ld.global.cg.v4.u32 {%0, %1, %2, %3}, [%4];"
               : "=r"(x.x), "=r"(x.y), "=r"(x.z), "=r"(x.w)
               : "l"(addr));
}

__device__ __forceinline__ void lds(uint4& x, const uint4* addr)
{
  asm volatile("ld.shared.v4.u32 {%0, %1, %2, %3}, [%4];"
               : "=r"(x.x), "=r"(x.y), "=r"(x.z), "=r"(x.w)
               : "l"(addr));
}

template <typename l_data_type, typename r_data_type, uint n_degrees,
          uint tile_size>
struct euclidean_distance_no_sqrt_chunked {
  // Promote to higher precision for safety if needed
  using acc_type = typename promote_to_float<l_data_type>::type;

  template <uint32_t loads_per_round=2>
  static __device__ acc_type
  distance(const data_vector<l_data_type, n_degrees>* l_vector,
           const data_vector<r_data_type, n_degrees>* r_vector,
           cg::thread_block_tile<tile_size>& work_tile) {
    acc_type local_sum = 0.0;

    const uint4* l_ptr = reinterpret_cast<const uint4*>(l_vector->data);
    const uint4* r_ptr = reinterpret_cast<const uint4*>(r_vector->data);

    // Check 16-byte alignment
    gpu_assert((reinterpret_cast<uintptr_t>(l_ptr) % 16) == 0,
               "Left vector not 16-byte aligned\n");
    gpu_assert((reinterpret_cast<uintptr_t>(r_ptr) % 16) == 0,
               "Right vector not 16-byte aligned\n");

    // Process 16 elements at a time using uint4 (16 bytes)
    constexpr int n_element_per_uint4 = 16 / sizeof(l_data_type);
    constexpr int vec4_elements = n_degrees / n_element_per_uint4;

    //number of sets of tiled loads that need to occur.

    constexpr int total_loads = vec4_elements;

    constexpr int load_rounds = (vec4_elements-1)/(loads_per_round*tile_size)+1;


    uint4 l_data[loads_per_round];
    uint4 r_data[loads_per_round];

    for (uint i = 0; i < load_rounds; i+= 1){

      for (uint j = 0; j < loads_per_round; j++){

        // 3 parts - thread_rank is local offset,
        // j*tile_size steps through tile_size loads
        // i handles each load round.
        int index = work_tile.thread_rank()+j*tile_size+i*tile_size*loads_per_round;

    
        if (index < total_loads){

          l_data[j] = l_ptr[index];


        }

      }


      for (uint j = 0; j < loads_per_round; j++){

        // 3 parts - thread_rank is local offset,
        // j*tile_size steps through tile_size loads
        // i handles each load round.
        int index = work_tile.thread_rank()+j*tile_size+i*tile_size*loads_per_round;

    
        if (index < total_loads){

          r_data[j] = r_ptr[index];


        }

      }

      for (uint j = 0; j < loads_per_round; j++){

        int index = work_tile.thread_rank()+j*tile_size+i*tile_size*loads_per_round;

        if (index < total_loads){

          l_data_type* l_bytes = (l_data_type*)&l_data[j];
          r_data_type* r_bytes = (r_data_type*)&r_data[j];

          for (int k = 0; k < n_element_per_uint4; k++) {

              acc_type diff = static_cast<acc_type>(l_bytes[k]) -
                            static_cast<acc_type>(r_bytes[k]);
              local_sum += diff * diff;
        
          }

        }

      }

    }


    // Use cooperative groups to reduce across threads
    acc_type total_sum = cg::reduce(work_tile, local_sum, cg::plus<acc_type>());

    // Final safety check
    gpu_assert(!is_bad(total_sum), "Result is corrupted\n");

    return total_sum;
  }
};

template <typename l_data_type, typename r_data_type, uint n_degrees,
          uint tile_size>
struct inner_product_chunked {
  // Promote to higher precision for safety if needed
  using acc_type = typename promote_to_float<l_data_type>::type;

  static __device__ acc_type
  distance(const data_vector<l_data_type, n_degrees>* l_vector,
           const data_vector<r_data_type, n_degrees>* r_vector,
           cg::thread_block_tile<tile_size>& work_tile) {
    acc_type local_sum = 0.0;

    const uint4* l_ptr = reinterpret_cast<const uint4*>(l_vector->data);
    const uint4* r_ptr = reinterpret_cast<const uint4*>(r_vector->data);

    // Check 16-byte alignment
    gpu_assert((reinterpret_cast<uintptr_t>(l_ptr) % 16) == 0,
               "Left vector not 16-byte aligned\n");
    gpu_assert((reinterpret_cast<uintptr_t>(r_ptr) % 16) == 0,
               "Right vector not 16-byte aligned\n");

    // Process 16 elements at a time using uint4 (16 bytes)
    constexpr int n_element_per_uint4 = 16 / sizeof(l_data_type);
    constexpr int vec4_elements = n_degrees / n_element_per_uint4;
    for (int i = work_tile.thread_rank(); i < vec4_elements;
         i += work_tile.size()) {
      uint4 l_vec4 = l_ptr[i];
      uint4 r_vec4 = r_ptr[i];

      // Process the 16 loaded uint8_t elements
      l_data_type* l_bytes = (l_data_type*)&l_vec4;
      r_data_type* r_bytes = (r_data_type*)&r_vec4;

      for (int j = 0; j < n_element_per_uint4; j++) {
        if (i*n_element_per_uint4+j < n_degrees) {
          local_sum += static_cast<acc_type>(r_bytes[j]) * static_cast<acc_type>(r_bytes[j]);
        }
      }
    }
    // Use cooperative groups to reduce across threads
    acc_type total_sum = cg::reduce(work_tile, local_sum, cg::plus<acc_type>());

    // Final safety check
    gpu_assert(!is_bad(total_sum), "Result is corrupted\n");

    return -total_sum;
  }
};

}  // namespace gpu_ann

#endif  // GPU_BLOCK_