#pragma once

#include <cstdint>

namespace gpu {

template <typename T, std::uint32_t Dim>
struct __align__(16) DeviceVector {
  T data[Dim];

  __host__ __device__ DeviceVector() {
#ifndef __CUDA_ARCH__
    for (std::uint32_t i = 0; i < Dim; ++i) {
      data[i] = static_cast<T>(0);
    }
#endif
  }

  __host__ explicit DeviceVector(const T* src) {
    for (std::uint32_t i = 0; i < Dim; ++i) {
      data[i] = src[i];
    }
  }

  __host__ __device__ T& operator[](std::uint32_t index) { return data[index]; }
  __host__ __device__ const T& operator[](std::uint32_t index) const { return data[index]; }
};

}  // namespace gpu
