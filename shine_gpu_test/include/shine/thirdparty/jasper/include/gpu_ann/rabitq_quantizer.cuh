#pragma once

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cooperative_groups/scan.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <curand.h>

#include <fstream>
#include <gallatin/allocators/alloc_utils.cuh>
#include <gallatin/data_structs/ds_utils.cuh>
#include <gpu_ann/_proj_metadata.cuh>
#include <gpu_ann/distance.cuh>
#include <gpu_ann/distance_lookup.cuh>
#include <gpu_ann/randomness.cuh>
#include <gpu_ann/vector.cuh>
#include <iostream>
#include <vector>
#include <math.h>
#include <cstdlib>
#include <Eigen/Dense>
#include <random>
#include <queue>

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "assert.h"
#include "stdio.h"

// 10.48550/arXiv.2405.12497: RaBitQ: Quantizing High-Dimensional Vectors 
// with a Theoretical Error Bound for Approximate Nearest Neighbor Search

namespace gpu_ann {
namespace rabitq {

template <class Scalar=float>
__host__ void set_rotation_matrix(
  int d, 
  Scalar *m, 
  Scalar *m_transpose, 
  uint64_t seed=0
) {
  using Mat = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
  std::mt19937_64 rng(seed);
  std::normal_distribution<Scalar> N(0, 1);
  Mat A(d, d);
  for (int i = 0; i < d; ++i){
    for (int j = 0; j < d; ++j){
      A(i, j) = N(rng);
    }
  }
  Eigen::HouseholderQR<Mat> qr(A);
  Mat Q = qr.householderQ() * Mat::Identity(d, d);
  Mat R = Q.transpose() * A;
  for (int i = 0; i < d; ++i) {
      if (R(i, i) < Scalar(0)) {
          Q.col(i) *= Scalar(-1);
      }
  }
  Mat Qt = Q.transpose();
  for (int i = 0; i < d; ++i){
    for (int j = 0; j < d; ++j){
      // Store in column major 
      m[i+j*d] = Q(i, j);
      m_transpose[i+j*d] = Qt(i, j);
    }
  }
}

template <class Scalar=float>
__host__ float* get_device_rotation_matrix(int d, uint64_t seed=0) {
  uint32_t rot_matrix_size = d * d;

  float *P = gallatin::utils::get_host_version<float>(rot_matrix_size);
  float *PTranspose = gallatin::utils::get_host_version<float>(rot_matrix_size);

  gpu_ann::rabitq::set_rotation_matrix(d, P, PTranspose);

  P = gallatin::utils::move_to_device<float>(P, rot_matrix_size);
  cudaFreeHost(PTranspose);

  return P;
}

__host__ void rotate_data_vec(
  cublasHandle_t handle,
  float *d_data_vectors,         // row major   
  float *d_rotated_data_vectors, // row major
  uint64_t n_data_vectors,
  uint32_t dim,
  float *d_P // (dim x dim) col major
) {
  const float alpha = 1.0f;
  const float beta  = 0.0f;
  // reference: https://leimao.github.io/blog/cuBLAS-Transpose-Column-Major-Relationship/
  cublasStatus_t stat = cublasSgemm(
      handle,
      CUBLAS_OP_T, CUBLAS_OP_N,
      dim,
      n_data_vectors,
      dim,
      &alpha,
      d_P, dim,
      d_data_vectors, dim,
      &beta,
      d_rotated_data_vectors, dim
  );
  if (stat != CUBLAS_STATUS_SUCCESS) {
    printf("cublasSgemm failed: %d\n", stat);
  }
  cudaDeviceSynchronize();
}

__host__ void rotate_single_data_vec(
    cublasHandle_t handle,
    const float *d_data_vector,       // (dim)
    float *d_rotated_data_vector,     // (dim)
    uint32_t dim,
    const float *d_P                  // (dim x dim)
) {
  const float alpha = 1.0f;
  const float beta  = 0.0f;
  cublasSgemv(
    handle,
    CUBLAS_OP_N,
    dim,              
    dim,              
    &alpha,
    d_P, dim,         
    d_data_vector, 1, 
    &beta,
    d_rotated_data_vector, 1
  );
  cudaDeviceSynchronize();
}

// SIZE_PER_DIM in number of bits
template <uint SIZE_PER_DIM, uint DATA_DIM>
struct __align__(16) RabitqDataVec {
  uint8_t data[(SIZE_PER_DIM * DATA_DIM + 7) / 8];
  float add = 0;
  float rescale = 0;
};

template <uint SIZE_PER_DIM, uint DATA_DIM>
void printRabitqDataVec(const RabitqDataVec<SIZE_PER_DIM, DATA_DIM>& vec) {
    constexpr size_t BYTES = (SIZE_PER_DIM * DATA_DIM + 7) / 8;
    printf("RabitqDataVec<%u, %u> {\n", SIZE_PER_DIM, DATA_DIM);
    printf("  data (%zu bytes):", BYTES);
    for (size_t i = 0; i < BYTES; ++i) {
        if (i % 16 == 0) printf("\n    ");
        printf("%02X ", vec.data[i]);
    }
    printf("\n  add = %f\n", vec.add);
    printf("  rescale = %f\n", vec.rescale);
    printf("}\n");
}

template <uint SIZE_PER_DIM, uint DATA_DIM>
__host__ __device__ inline uint8_t get_dimension(
  const RabitqDataVec<SIZE_PER_DIM, DATA_DIM>* vec,
  uint32_t i
){
  const uint32_t bit_idx = i * SIZE_PER_DIM;
  const uint32_t byte_idx = bit_idx / 8;
  const uint32_t bit_off  = bit_idx % 8;
  uint16_t chunk = vec->data[byte_idx];
  if (bit_off + SIZE_PER_DIM > 8 && byte_idx + 1 < sizeof(vec->data)) {
    chunk |= (uint16_t(vec->data[byte_idx + 1]) << 8);
  }
  uint8_t mask = (1u << SIZE_PER_DIM) - 1u;
  return (chunk >> bit_off) & mask;
}

template <uint DATA_DIM>
__host__ __device__ inline uint8_t get_dimension(
  const RabitqDataVec<1, DATA_DIM>* vec,
  uint32_t i
){
  return ((vec->data[i >> 3]) >> (i & 7)) & 1u;
}

template <uint DATA_DIM>
__host__ __device__ inline uint8_t get_dimension(
  const RabitqDataVec<2, DATA_DIM>* vec,
  uint32_t i
){
  size_t bitOffset = (i & 0x3) << 1;
  uint8_t byte = vec->data[i>>2];
  return (byte >> bitOffset) & 0b11;
}

template <uint DATA_DIM>
__host__ __device__ inline uint8_t get_dimension(
  const RabitqDataVec<4, DATA_DIM>* vec,
  uint32_t i
){
  size_t bitOffset = (i & 0x1) << 2;
  uint8_t byte = vec->data[i>>1];
  return (byte >> bitOffset) & 0b1111;
}

// Quantized query vector
struct RabitqQueryFactor {
  float add = 0;
  float k1xSumq = 0;
  float kBxSumq = 0;
};

constexpr std::array<float, 9> kTightStart = {
    0,   
    0.15,
    0.20,
    0.52,
    0.59,
    0.71,
    0.75,
    0.77,
    0.81
};

__global__ void abs_kernel(float* data, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
      data[idx] = fabsf(data[idx]);
  }
}

// Factor function from RabitQLib
//
// Given a unit data vector *o_abs (assume all dimension is positive), its dimension and
// the number of quantized bits we assign fo each dimension, what is the best rescaling
// factor t when quantizing this vector.
//
// NOTE: this part of the code is computationally intensive and hard to parallelize.
//       so we randomly sample this function in get_const_scaling_factors to get a constant t.
__host__ inline double best_rescale_factor(const float *o_abs, size_t dim, size_t ex_bits) {
  constexpr double kEps = 1e-5;
  constexpr int kNEnum = 10;
  double max_o = *std::max_element(o_abs, o_abs + dim);

  double t_end = static_cast<double>(((1 << ex_bits) - 1) + kNEnum) / max_o;
  double t_start = t_end * kTightStart[ex_bits];

  std::vector<int> cur_o_bar(dim);
  double sqr_denominator = static_cast<double>(dim) * 0.25;
  double numerator = 0;

  for (size_t i = 0; i < dim; ++i) {
    int cur = static_cast<int>((t_start * o_abs[i]) + kEps);
    cur_o_bar[i] = cur;
    sqr_denominator += cur * cur + cur;
    numerator += (cur + 0.5) * o_abs[i];
  }

  std::priority_queue<std::pair<double, size_t>,
                      std::vector<std::pair<double, size_t>>, std::greater<>>
      next_t;

  for (size_t i = 0; i < dim; ++i) {
    next_t.emplace(static_cast<double>(cur_o_bar[i] + 1) / o_abs[i], i);
  }

  double max_ip = 0;
  double t = 0;

  while (!next_t.empty()) {
    double cur_t = next_t.top().first;
    size_t update_id = next_t.top().second;
    next_t.pop();

    cur_o_bar[update_id]++;
    int update_o_bar = cur_o_bar[update_id];
    sqr_denominator += 2 * update_o_bar;
    numerator += o_abs[update_id];

    double cur_ip = numerator / std::sqrt(sqr_denominator);
    if (cur_ip > max_ip) {
      max_ip = cur_ip;
      t = cur_t;
    }

    if (update_o_bar < (1 << ex_bits) - 1) {
      double t_next = static_cast<double>(update_o_bar + 1) / o_abs[update_id];
      if (t_next < t_end) {
        next_t.emplace(t_next, update_id);
      }
    }
  }

  return t;
}

// Sampling random data vectors to calculate a t_const.
__host__ double get_const_scaling_factors(size_t dim, size_t ex_bits){
  // generate random vectors
  cublasHandle_t handle;
  cublasCreate(&handle);
  constexpr int32_t n_samples = 1000;
  float *h_vectors = gallatin::utils::get_host_version<float>(n_samples*dim);

  curandGenerator_t curandGen;
  curandCreateGenerator(&curandGen, CURAND_RNG_PSEUDO_DEFAULT);
  curandSetPseudoRandomGeneratorSeed(curandGen, 1234ULL);
  curandGenerateNormal(curandGen, h_vectors, n_samples * dim, 0.0f, 1.0f);
  curandDestroyGenerator(curandGen);

  for (int i = 0; i < n_samples; i++) {
    float* row_ptr = h_vectors + i * dim;

    // compute norm = ||row||
    float norm = 0.0f;
    cublasSnrm2(handle, dim, row_ptr, 1, &norm);

    // scale row by 1/norm
    float inv_norm = 1.0f / norm;
    cublasSscal(handle, dim, &inv_norm, row_ptr, 1);
  }

  // get the absolute valule for each vector's dimension
  abs_kernel<<<(n_samples*dim+256-1)/256, 256>>>(h_vectors, n_samples*dim);

  // get the average t as the t_const
  double sum_val = 0.0;
  for (uint i=0; i<n_samples; i++) {
    double t = best_rescale_factor(h_vectors+i*dim, dim, ex_bits);
    sum_val += t;
  }

  return sum_val / n_samples;
}

// Calculate quantized code based on the given residual vector.
// Output an array of uint8_t with length equal to dimension.
template <uint SIZE_PER_DIM,
          uint DATA_DIM,
          uint16_t tile_size,
          typename ParentT>
__device__ float calc_multi_code(
  float *residual_vec,
  uint8_t *uncompressed_code,
  double t_const,
  float l2_norm,
  cg::thread_block_tile<tile_size, ParentT>& my_tile
) {
  assert(1 <= SIZE_PER_DIM && SIZE_PER_DIM <= 8);
  constexpr double kEps = 1e-5;

  // normalize and abs residual for plus code
  int val;
  float abs_o;
  float ip_norm_tmp = 0;
  for (uint i=my_tile.thread_rank(); i<DATA_DIM; i+=my_tile.size()) {
    abs_o = abs(residual_vec[i] / l2_norm);
    val = static_cast<int>((t_const * abs_o) + kEps);
    if (val >= (1 << (SIZE_PER_DIM-1))) {
      val = (1 << (SIZE_PER_DIM-1))-1;
    }
    uncompressed_code[i] = static_cast<uint8_t>(val);
    ip_norm_tmp += (val + 0.5) * abs_o;
  }
  float ip_norm = cg::reduce(my_tile, ip_norm_tmp, cg::plus<float>());
  float ip_norm_inv = 1 / ip_norm;
  if (isnan(ip_norm_inv)) {
    ip_norm_inv = 1.F;
  }

  // reverse the the negative code, 
  // and add in the sign if the residual is positive for that dimension.
  uint32_t const mask = (1 << (SIZE_PER_DIM-1)) - 1;
  for (uint i=my_tile.thread_rank(); i<DATA_DIM; i+=my_tile.size()) {
    if (residual_vec[i] >= 0) {
      // if residual_vec[i] is positive, add one to the
      // most significant bit.
      uncompressed_code[i] += 1 << (SIZE_PER_DIM-1);
    } else {
      // if residual_vec[i] is negative, reverse all the bits
      uint8_t tmp = uncompressed_code[i];
      uncompressed_code[i] = (~tmp) & mask;
    }
  }
  my_tile.sync();

  return ip_norm_inv;
}

// Main kernel for quantize data vectors
template <uint SIZE_PER_DIM, 
          uint DATA_DIM,
          uint16_t tile_size=4>
__global__ void rabitq_quantize_kernel(
  float *d_rot_vectors, // (dim x n_vectors)
  uint64_t n_vectors,
  RabitqDataVec<SIZE_PER_DIM,DATA_DIM> *rabitq_vectors,
  float *d_centroid,
  double t_const,
  MetricType metric_type
) {
  // one coorperative group tile ->
  // one vector quantization calculation
  auto thread_block = cg::this_thread_block();
  auto my_tile = cg::tiled_partition<tile_size>(thread_block);
  uint64_t total_tiles = gridDim.x * (blockDim.x / tile_size);
  uint64_t tid = thread_block.thread_rank() / my_tile.size() + blockIdx.x * (blockDim.x / tile_size);

  const float inv_d_sqrt = (DATA_DIM == 0) ? 1.0f : (1.0f / std::sqrt((float)DATA_DIM));

  for (uint i=tid; i<n_vectors; i+=total_tiles) {
    // get reference for quantized data vector and data factor.
    RabitqDataVec<SIZE_PER_DIM,DATA_DIM> *vec = rabitq_vectors + i;

    // residual vector (or-c)
    float residual_vec[DATA_DIM] = {0};
    float l2_sqr_tmp = 0;
    for (uint j=my_tile.thread_rank(); j<DATA_DIM; j+=my_tile.size()) {
      residual_vec[j] = d_rot_vectors[i*DATA_DIM+j] - d_centroid[j];
      l2_sqr_tmp += residual_vec[j] * residual_vec[j];
    }
    float l2_sqr = cg::reduce(my_tile, l2_sqr_tmp, cg::plus<float>());
    float l2_norm = std::sqrt(l2_sqr);

    // calculate the uncompressed version of the code and store them
    // in the `uncompressed_code`. This version of the code will occupy
    // one uint8_t per dimension.
    uint8_t uncompressed_code[DATA_DIM] = {0};
    float ipnorm_inv = calc_multi_code<SIZE_PER_DIM, DATA_DIM, tile_size>(
      residual_vec, uncompressed_code, t_const, l2_norm, my_tile
    );

    // calculate xu_cb (decompressed version of the quantized vector)
    // and calculate its inner product with residual_vec and centroid.
    float ip_resi_xucb_tmp = 0;
    float ip_cent_xucb_tmp = 0;
    float ip_resi_cent_tmp = 0;
    float cb = -(static_cast<float>(1 << (SIZE_PER_DIM-1)) - 0.5F);
    for (uint j=my_tile.thread_rank(); j<DATA_DIM; j+=my_tile.size()) {
      uint8_t tmp = uncompressed_code[j];
      float xu_cb_jth = static_cast<float>(tmp) + cb;
      ip_resi_xucb_tmp += residual_vec[j] * xu_cb_jth;
      ip_cent_xucb_tmp += d_centroid[j] * xu_cb_jth;
      ip_resi_cent_tmp += residual_vec[j] * d_centroid[j];
    }
    float ip_resi_xucb = cg::reduce(my_tile, ip_resi_xucb_tmp, cg::plus<float>());
    float ip_cent_xucb = cg::reduce(my_tile, ip_cent_xucb_tmp, cg::plus<float>());
    float ip_resi_cent = cg::reduce(my_tile, ip_resi_cent_tmp, cg::plus<float>());
    if (ip_resi_xucb == 0) {
      ip_resi_xucb = std::numeric_limits<float>::infinity();
    }

    // compute factors
    if (my_tile.thread_rank() == 0) {
      if (metric_type == METRIC_L2) {
        vec->add = l2_sqr + 2 * l2_sqr * ip_cent_xucb / ip_resi_xucb;
        vec->rescale = ipnorm_inv * -2 * l2_norm;
      } else if (metric_type == METRIC_IP) {
        vec->add = 1 - ip_resi_cent + l2_sqr * ip_cent_xucb / ip_resi_xucb;
        vec->rescale = ipnorm_inv * -l2_norm;
      } else {
        printf("[ERROR] Unsupported metric type=%u\n", static_cast<uint32_t>(metric_type));
      }
    }

    // store the uncompressed code to the RabitqDataVec
    // as compressed version. (TODO: potential optimization)
    uint32_t compressed_size = (SIZE_PER_DIM * DATA_DIM + 7) / 8; // byte
    uint8_t local_code;
    for (uint j=0; j<compressed_size; j++) {

      local_code = 0;
      for (uint k=my_tile.thread_rank(); k<DATA_DIM; k+=my_tile.size()) {
        uint32_t start = k * SIZE_PER_DIM;
        uint32_t shift = start % 8;
        if (start>=j*8 && start<(j+1)*8) {
          local_code |= uncompressed_code[k] << shift;
        }
      }
      uint8_t code = cg::reduce(my_tile, local_code, cg::bit_or<uint8_t>());

      if (my_tile.thread_rank() == 0) {
        vec->data[j] = code;
      }
    }
  }
}

template <typename T, uint N>
__global__ void flatten_to_float(data_vector<T, N>* __restrict__ in,
                                 float* __restrict__ out,
                                 size_t n_vecs) {
    size_t vec_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (vec_id >= n_vecs) return;
    for (uint i = 0; i < N; i++) {
      T d = in[vec_id].data[i];
      out[vec_id * N + i] = static_cast<float>(d);
    }
}

// Quantize code
template <typename DATA_T,
          uint DATA_DIM,
          uint SIZE_PER_DIM=1>
__host__ RabitqDataVec<SIZE_PER_DIM, DATA_DIM>* rabitq_quantize(
  data_vector<DATA_T, DATA_DIM> *original_vectors,
  uint64_t n_vectors,
  float *d_P, 
  float *d_centroid,
  const bool on_host,
  const MetricType metric_type
) {
  cublasHandle_t handle;
  cublasCreate(&handle);

  // data vectors to array of floats for later calculation
  float *d_data_vectors;
  d_data_vectors = gallatin::utils::get_device_version<float>(n_vectors*DATA_DIM); 
  flatten_to_float<DATA_T, DATA_DIM><<<(n_vectors+256-1)/256, 256>>>(
    original_vectors, d_data_vectors, n_vectors
  );
  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
      std::cerr << "Kernel launch failed: " 
                << cudaGetErrorString(err) << std::endl;
  }

  // Rotate data vectors
  float *d_rot_vectors;
  if (on_host) {
    d_rot_vectors = gallatin::utils::get_host_version<float>(n_vectors*DATA_DIM);
  } else {
    d_rot_vectors = gallatin::utils::get_device_version<float>(n_vectors*DATA_DIM);
  }
  rotate_data_vec(handle, d_data_vectors, d_rot_vectors, n_vectors, DATA_DIM, d_P);

  // Rotate centroid
  float *h_rot_centroid;
  h_rot_centroid = gallatin::utils::get_host_version<float>(DATA_DIM);
  rotate_single_data_vec(handle, d_centroid, h_rot_centroid, DATA_DIM, d_P);

  // Compute quantized vectors
  using quantized_t = RabitqDataVec<SIZE_PER_DIM, DATA_DIM>;
  quantized_t *rabitq_vectors;
  if (on_host) {
    rabitq_vectors = gallatin::utils::get_host_version<quantized_t>(n_vectors);
  } else {
    rabitq_vectors = gallatin::utils::get_device_version<quantized_t>(n_vectors);
  }

  // get t_const
  double t_const = get_const_scaling_factors(DATA_DIM, SIZE_PER_DIM);
  //t_const = 42.005;
  std::cout << "t_const=" << t_const << std::endl;

  constexpr uint32_t block_size = 256;
  constexpr uint32_t tile_size = 4;
  dim3 block_dims(block_size, 1, 1);
  dim3 grid_dims((n_vectors*tile_size-1)/block_size+1, 1, 1);
  rabitq_quantize_kernel<SIZE_PER_DIM, DATA_DIM, tile_size>
    <<<grid_dims, block_dims>>>(
      d_rot_vectors,
      n_vectors,
      rabitq_vectors,
      h_rot_centroid, //d_centroid,
      t_const,
      metric_type
  );
  err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
      std::cerr << "Kernel launch failed: " 
                << cudaGetErrorString(err) << std::endl;
  }

  if (on_host) {
    cudaFreeHost(d_data_vectors);
    cudaFreeHost(d_rot_vectors);
  } else {
    cudaFree(d_data_vectors);
    cudaFree(d_rot_vectors);
  }

  cublasDestroy(handle);
  return rabitq_vectors;
}

template <typename DATA_T,
          uint DATA_DIM,
          uint SIZE_PER_DIM = 1>
__host__ RabitqDataVec<SIZE_PER_DIM, DATA_DIM>* rabitq_quantize_batched(
  data_vector<DATA_T, DATA_DIM>* h_original_vectors,
  uint64_t n_vectors,
  float* d_P,
  float* d_centroid,
  uint64_t batch_size,
  const MetricType metric_type
) {
  using quantized_t = RabitqDataVec<SIZE_PER_DIM, DATA_DIM>;

  // =========================
  // cuBLAS
  // =========================
  cublasHandle_t handle;
  cublasCreate(&handle);

  // =========================
  // Final OUTPUT (DEVICE)
  // =========================
  quantized_t* d_rabitq_all =
    gallatin::utils::get_device_version<quantized_t>(n_vectors);

  // =========================
  // Rotate centroid ONCE
  // =========================
  float* h_rot_centroid =
    gallatin::utils::get_host_version<float>(DATA_DIM);

  rotate_single_data_vec(
    handle,
    d_centroid,
    h_rot_centroid,
    DATA_DIM,
    d_P
  );

  // =========================
  // Constants
  // =========================
  double t_const = get_const_scaling_factors(DATA_DIM, SIZE_PER_DIM);
  std::cout << "t_const=" << t_const << std::endl;

  constexpr uint32_t block_size = 256;
  constexpr uint32_t tile_size  = 4;

  // =========================
  // Batch DEVICE buffers
  // =========================
  float* d_data_vectors =
    gallatin::utils::get_device_version<float>(batch_size * DATA_DIM);

  float* d_rot_vectors =
    gallatin::utils::get_device_version<float>(batch_size * DATA_DIM);

  quantized_t* d_rabitq_batch =
    gallatin::utils::get_device_version<quantized_t>(batch_size);

  // =========================
  // Batched loop
  // =========================
  for (uint64_t base = 0; base < n_vectors; base += batch_size) {
    uint64_t cur_batch =
      std::min(batch_size, n_vectors - base);

    // ---- flatten ----
    flatten_to_float<DATA_T, DATA_DIM>
      <<< (cur_batch + 255) / 256, 256 >>>(
        h_original_vectors + base,
        d_data_vectors,
        cur_batch
      );
    cudaDeviceSynchronize();

    // ---- rotate ----
    rotate_data_vec(
      handle,
      d_data_vectors,
      d_rot_vectors,
      cur_batch,
      DATA_DIM,
      d_P
    );

    // ---- quantize ----
    dim3 block(block_size, 1, 1);
    dim3 grid((cur_batch * tile_size + block_size - 1) / block_size, 1, 1);

    rabitq_quantize_kernel<SIZE_PER_DIM, DATA_DIM, tile_size>
      <<<grid, block>>>(
        d_rot_vectors,
        cur_batch,
        d_rabitq_batch,
        h_rot_centroid,
        t_const,
        metric_type
      );
    cudaDeviceSynchronize();

    // ---- copy batch result INTO final GPU array ----
    cudaMemcpy(
      d_rabitq_all + base,
      d_rabitq_batch,
      cur_batch * sizeof(quantized_t),
      cudaMemcpyDeviceToDevice
    );
  }

  // =========================
  // Cleanup
  // =========================
  cudaFree(d_data_vectors);
  cudaFree(d_rot_vectors);
  cudaFree(d_rabitq_batch);
  cudaFreeHost(h_rot_centroid);

  cublasDestroy(handle);

  return d_rabitq_all;
}

// Calculate query vector's factor
template <uint SIZE_PER_DIM, uint DATA_DIM, uint tile_size=4>
__global__ void rabitq_query_quantize_kernel(
  float *d_rot_query_vectors,
  uint64_t n_query_vectors,
  RabitqQueryFactor *query_factors,
  float *d_centroid
) {
  // one coorperative group tile ->
  // one vector quantization calculation
  auto thread_block = cg::this_thread_block();
  auto my_tile = cg::tiled_partition<tile_size>(thread_block);
  uint64_t total_tiles = gridDim.x * (blockDim.x / tile_size);
  uint64_t tid = thread_block.thread_rank() / my_tile.size() + blockIdx.x * (blockDim.x / tile_size);

  float c_b = -static_cast<float>((1 << SIZE_PER_DIM) - 1) / 2.F;
  float c_1 = -static_cast<float>((1 << 1) - 1) / 2.F;

  for (uint i=tid; i<n_query_vectors; i+=total_tiles) {
    RabitqQueryFactor *fac = query_factors + i;

    float sqr_norm_tmp = 0; // |qr-c|^2
    float sumq_tmp = 0;
    for (uint j=my_tile.thread_rank(); j<DATA_DIM; j+=my_tile.size()) {
      float diff = d_rot_query_vectors[i*DATA_DIM+j]-d_centroid[j];
      sqr_norm_tmp += diff * diff;
      sumq_tmp += d_rot_query_vectors[i*DATA_DIM+j];
    }
    float sqr_norm = cg::reduce(my_tile, sqr_norm_tmp, cg::plus<float>());
    float sumq = cg::reduce(my_tile, sumq_tmp, cg::plus<float>());

    if (my_tile.thread_rank() == 0) {
      fac->add = sqr_norm;
      fac->k1xSumq = c_1 * sumq;
      fac->kBxSumq = c_b * sumq;
      //printf("query[%u] fac->add=%f, fac->kBxSumq=%f, fac->k1xSumq=%f\n", i, sqr_norm, fac->kBxSumq, fac->k1xSumq);
    }
  }
}

template <typename DATA_T, 
          uint SIZE_PER_DIM, 
          uint DATA_DIM>
__host__ thrust::pair<float*, RabitqQueryFactor*> rabitq_query_quantize(
  data_vector<DATA_T, DATA_DIM> *query_vectors,
  uint64_t n_query_vectors,
  float *d_P,
  float *d_centroid
) {
  cublasHandle_t handle;
  cublasCreate(&handle);

  // query vectors to array in column major
  float *d_query_vectors;
  d_query_vectors = gallatin::utils::get_host_version<float>(n_query_vectors*DATA_DIM);  
  for (uint j = 0; j < DATA_DIM; ++j) {
    for (uint i = 0; i < n_query_vectors; ++i) {
      d_query_vectors[j + i*DATA_DIM] = static_cast<float>(query_vectors[i][j]);
    }
  }
  d_query_vectors = gallatin::utils::move_to_device<float>(
    d_query_vectors, 
    n_query_vectors*DATA_DIM);

  // rotate using P
  float *d_rot_query_vectors;
  d_rot_query_vectors = gallatin::utils::get_device_version<float>(
    n_query_vectors*DATA_DIM
  );
  rotate_data_vec(
    handle, 
    d_query_vectors, 
    d_rot_query_vectors, 
    n_query_vectors, 
    DATA_DIM, 
    d_P
  );

  // Rotate centroid
  float *h_rot_centroid;
  h_rot_centroid = gallatin::utils::get_host_version<float>(DATA_DIM);
  rotate_single_data_vec(handle, d_centroid, h_rot_centroid, DATA_DIM, d_P);

  // Compute quantized query vectors
  RabitqQueryFactor *query_factors;
  query_factors =  gallatin::utils::get_device_version<RabitqQueryFactor>(
    n_query_vectors
  );

  constexpr uint32_t block_size = 512;
  constexpr uint32_t tile_size = 4;
  dim3 block_dims(block_size, 1, 1);
  dim3 grid_dims((n_query_vectors*tile_size-1)/block_size + 1, 1, 1);
  rabitq_query_quantize_kernel<SIZE_PER_DIM, DATA_DIM><<<grid_dims, block_dims>>>(
    d_rot_query_vectors,
    n_query_vectors,
    query_factors,
    h_rot_centroid//d_centroid
  );
  
  cudaFree(d_query_vectors);
  return {d_rot_query_vectors, query_factors};
}

// one distance call on device using tile group
//assumes divisible by 4.
template <uint DATA_DIM, uint tile_size, typename ParentT>
__device__ float one_distance_device(
  const RabitqQueryFactor & query_fac,
  const float * __restrict__ rot_qvec,
  const RabitqDataVec<8, DATA_DIM> * __restrict__ data_vec,
  const cg::thread_block_tile<tile_size, ParentT>& my_tile
) {
  const float data_add = data_vec->add;
  const float data_rescale = data_vec->rescale;

  const char4* data_ptr = reinterpret_cast<const char4*>(data_vec->data);
  const float4* query_ptr = reinterpret_cast<const float4*>(rot_qvec);

  float dot_tmp = 0;

  for (uint i = my_tile.thread_rank(); i < DATA_DIM/4; i+= my_tile.size()){
    float4 q = query_ptr[i];
    char4 d = data_ptr[i];

    dot_tmp += static_cast<uint8_t>(d.x) * q.x + 
      static_cast<uint8_t>(d.y) * q.y + 
      static_cast<uint8_t>(d.z) * q.z + 
      static_cast<uint8_t>(d.w) * q.w;
  }

  float dot = cg::reduce(my_tile, dot_tmp, cg::plus<float>());

  if (my_tile.thread_rank() == 0) {
    float dist = data_add + query_fac.add + data_rescale*(
      dot + query_fac.k1xSumq*static_cast<float>((1 << 8) - 1));
    return dist;
  }
  return 0;
}

// one distance call on device using tile group
// assumes divisible by 4.
template <uint DATA_DIM, uint tile_size, typename ParentT>
__device__ float one_distance_device(
  const RabitqQueryFactor & query_fac,
  const float * __restrict__ rot_qvec,
  const RabitqDataVec<4, DATA_DIM> * __restrict__ data_vec,
  const cg::thread_block_tile<tile_size, ParentT>& my_tile
) {
  const float data_add = data_vec->add;
  const float data_rescale = data_vec->rescale;

   const uint16_t * data_ptr = reinterpret_cast<const uint16_t*>(data_vec->data);
  const float4* query_ptr = reinterpret_cast<const float4*>(rot_qvec);

  float dot_tmp = 0;

  for (uint i = my_tile.thread_rank(); i < DATA_DIM/4; i+= my_tile.size()){
    float4 q = query_ptr[i];
    uint16_t packed = data_ptr[i];

    uint8_t d0 =  packed        & 0xF;
    uint8_t d1 = (packed >> 4)  & 0xF;
    uint8_t d2 = (packed >> 8)  & 0xF;
    uint8_t d3 = (packed >> 12) & 0xF;

    dot_tmp += d0 * q.x + d1 * q.y + d2 * q.z + d3 * q.w;
  }

  float dot = cg::reduce(my_tile, dot_tmp, cg::plus<float>());

  if (my_tile.thread_rank() == 0) {
    float dist = data_add + query_fac.add + data_rescale*(
      dot + query_fac.k1xSumq*static_cast<float>((1 << 4) - 1));
    return dist;
  }
  return 0;
}

// Calculate the l2 distance between query vector and data vector
template <uint SIZE_PER_DIM, uint DATA_DIM, uint tile_size=4>
__device__ void rabitq_l2_distance_device(
  // Query vector
  float *d_rot_qvec_minus_c,
  RabitqQueryFactor *query_factors,
  // Data vector to calculate distance with
  RabitqDataVec<SIZE_PER_DIM,DATA_DIM> *rabitq_vectors,
  uint64_t n_data_vectors,
  // result
  float *distances
){
  // We calculate one query per block.
  auto thread_block = cg::this_thread_block();
  auto my_tile = cg::tiled_partition<tile_size>(thread_block);
  uint64_t tiles_per_block = blockDim.x / tile_size;
  uint64_t tid = thread_block.thread_rank() / my_tile.size();

  RabitqQueryFactor qfac = query_factors[0];
  float *qvec = d_rot_qvec_minus_c;

  // one tile per distance
  for (uint i=tid; i<n_data_vectors; i += tiles_per_block) {
    RabitqDataVec<SIZE_PER_DIM,DATA_DIM> *dvec = rabitq_vectors + i;
    
    float pre_dist = one_distance_device(
      qfac, qvec, dvec, my_tile
    );

    if (my_tile.thread_rank() == 0) {
      distances[i] = pre_dist;
    }
    my_tile.sync();
  }
}


template <uint SIZE_PER_DIM, uint DATA_DIM, uint tile_size=4>
__global__ void rabitq_l2_distance_kernel(
  // Query vector
  float *d_rot_qvec_minus_c,
  RabitqQueryFactor *query_factors,
  // Data vector to calculate distance with
  RabitqDataVec<SIZE_PER_DIM,DATA_DIM> *rabitq_vectors,
  uint64_t n_data_vectors,
  // result
  float *distances
){
  rabitq_l2_distance_device(
    d_rot_qvec_minus_c,
    query_factors,
    rabitq_vectors,
    n_data_vectors,
    distances
  );
}

template <uint SIZE_PER_DIM, uint DATA_DIM, uint tile_size=4>
__host__ void rabitq_l2_distance(
  // Query vector
  float *d_rot_qvec_minus_c,
  RabitqQueryFactor *query_factors,
  // Data vector to calculate distance with
  RabitqDataVec<SIZE_PER_DIM,DATA_DIM> *rabitq_vectors,
  uint64_t n_data_vectors,
  // out
  float *distances
) {
  rabitq_l2_distance_kernel<<<1, 512>>>(
    d_rot_qvec_minus_c,
    query_factors,
    rabitq_vectors,
    n_data_vectors,
    distances
  );
}

// only calculate one distance pair between qvec and dvec
template <uint SIZE_PER_DIM, uint DATA_DIM, uint tile_size=4>
__global__ void rabitq_l2_distance_single_kernel(
  // Query vector
  float *d_rot_qvec_minus_c,
  RabitqQueryFactor *query_factors,
  // Data vector 
  RabitqDataVec<SIZE_PER_DIM,DATA_DIM> *rabitq_vectors,
  // result
  float *distances
){
  auto thread_block = cg::this_thread_block();
  auto my_tile = cg::tiled_partition<tile_size>(thread_block);
  uint64_t tiles_per_block = blockDim.x / tile_size;
  uint64_t tid = thread_block.thread_rank() / my_tile.size();

  if (tid != 0) return;

  RabitqQueryFactor qfac = query_factors[0];
  float pre_dist = one_distance_device(
    query_factors,
    d_rot_qvec_minus_c,
    rabitq_vectors,
    my_tile
  );
  if (my_tile.thread_rank() == 0) {
    distances[0] = pre_dist;
  }
}

template <uint SIZE_PER_DIM, uint DATA_DIM, uint tile_size=4>
__host__ void rabitq_l2_distance_single(
  // Query vector
  float *d_rot_qvec_minus_c,
  RabitqQueryFactor *query_factors,
  // Data vector
  RabitqDataVec<SIZE_PER_DIM,DATA_DIM> *rabitq_vectors,
  // out
  float *distances
) {
  rabitq_l2_distance_single_kernel<<<1, tile_size>>>(
    d_rot_qvec_minus_c,
    query_factors,
    rabitq_vectors,
    distances
  );
}


// ================== DEBUG ================= //

// Test function to compare the original l2 distance
// with the quantized version.
template <typename DATA_T, uint DATA_DIM>
__host__ void debug_rabitq_l2_distance(
  float *quantized_distances,
  uint32_t n_distances,
  data_vector<DATA_T, DATA_DIM> query_vector,
  data_vector<DATA_T, DATA_DIM> *data_vectors
) {
  // calculate the exact distances
  float *exact_distances = gallatin::utils::get_host_version<float>(n_distances);
  for (uint i=0; i<n_distances; i++) {
    // l2
    float tmp = 0;
    for (uint j=0; j<DATA_DIM; j++) {
      float diff = query_vector.data[j] - data_vectors[i].data[j];
      tmp += diff * diff;
    }
    exact_distances[i] = tmp;

    std::cout << "exact=" << exact_distances[i]
      << " rabitq=" << quantized_distances[i] << std::endl;
  }
}

template <uint SIZE_PER_DIM, uint DATA_DIM, typename VEC_T, template <typename, typename, uint, uint> class distance_functor, uint tile_size=4>
__global__ void check_relative_error_kernel(
  int64_t n_query_vecs,
  VEC_T *original_query_vecs,
  float *query_vecs,
  RabitqQueryFactor *query_facs,
  int64_t n_data_vecs,
  VEC_T *original_data_vecs,
  RabitqDataVec<SIZE_PER_DIM,DATA_DIM> *data_vecs,
  uint32_t *first_vec,
  uint32_t *second_vec,
  uint32_t *third_vec,
  uint64_t *matches
) {
  auto thread_block = cg::this_thread_block();
  cg::thread_block_tile<tile_size> my_tile = cg::tiled_partition<tile_size>(thread_block);
  uint64_t tid = gallatin::utils::get_tile_tid(my_tile);

  uint32_t first_id = first_vec[tid] % n_query_vecs;
  uint32_t second_id = second_vec[tid] % n_data_vecs;
  uint32_t third_id = third_vec[tid] % n_data_vecs;

  // calculate exact l2
  using vector_data_type = typename data_vector_traits<VEC_T>::type;
  float dist_a_b = distance_functor<vector_data_type, vector_data_type, DATA_DIM,
                       tile_size>::distance(original_query_vecs[first_id],
                                            original_data_vecs[second_id],
                                            my_tile);
  float dist_a_c = distance_functor<vector_data_type, vector_data_type, DATA_DIM,
                       tile_size>::distance(original_query_vecs[first_id],
                                            original_data_vecs[third_id],
                                            my_tile);

  // calculate quantized l2
  float *a_vec = query_vecs + first_id*DATA_DIM;
  RabitqQueryFactor a_fac = query_facs[first_id];
  RabitqDataVec<SIZE_PER_DIM,DATA_DIM> *b_vec = data_vecs + second_id;
  RabitqDataVec<SIZE_PER_DIM,DATA_DIM> *c_vec = data_vecs + third_id;

  float r_dist_a_b = one_distance_device(
    a_fac, a_vec, b_vec, my_tile
  );
  float r_dist_a_c = one_distance_device(
    a_fac, a_vec, c_vec, my_tile
  );

  if (my_tile.thread_rank() == 0) {
    bool dist_smaller = (dist_a_b <= dist_a_c);
    bool r_smaller = (r_dist_a_b <= r_dist_a_c);
    if (dist_smaller == r_smaller) {
      atomicAdd((unsigned long long int *)matches, 1ULL);
    }
  }
}

// Correctness test
// Sample distances and return the average relative distance error.relative distance error.
// if exact_dist(a, b) > exact_dist(a, c)
// then quantized_dist(a, b) > quantized_dist(a, c)
template <uint SIZE_PER_DIM, uint DATA_DIM, typename VEC_T, template <typename, typename, uint, uint> class distance_functor, uint tile_size=4>
__host__ void check_relative_error(
  int64_t n_tests,
  int64_t n_query_vecs,
  VEC_T *original_query_vecs,
  float *query_vecs,
  RabitqQueryFactor *query_facs,
  int64_t n_data_vecs,
  VEC_T *original_data_vecs,
  RabitqDataVec<SIZE_PER_DIM,DATA_DIM> *data_vecs
) {
  uint64_t *matches;
  cudaMallocManaged((void **)&matches, sizeof(uint64_t));
  matches[0] = 0;
  cudaDeviceSynchronize();

  uint32_t *first_vec = random_data_device<uint32_t>(n_tests);
  uint32_t *second_vec = random_data_device<uint32_t>(n_tests);
  uint32_t *third_vec = random_data_device<uint32_t>(n_tests);

  check_relative_error_kernel<SIZE_PER_DIM, DATA_DIM, VEC_T, distance_functor, tile_size>
    <<<(tile_size * n_tests - 1) / 256 + 1, 256>>>(
    n_query_vecs, original_query_vecs, query_vecs, query_facs,
    n_data_vecs, original_data_vecs, data_vecs, 
    first_vec, second_vec, third_vec,
    matches
  );
  cudaDeviceSynchronize();

  uint64_t n_matches = matches[0];
  printf("%lu/%lu distance comps match, %f percent match\n", n_matches, n_tests,
         100.0 * n_matches / n_tests);
  cudaFree(matches);
  cudaFree(first_vec);
  cudaFree(second_vec);
  cudaFree(third_vec);
}

}
}