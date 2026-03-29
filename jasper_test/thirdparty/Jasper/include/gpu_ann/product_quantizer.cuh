#ifndef PRODUCT_QUANTIZER
#define PRODUCT_QUANTIZER

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
#include <gpu_ann/distance_lookup.cuh>
#include <gpu_ann/randomness.cuh>
#include <gpu_ann/vector.cuh>
#include <iostream>
#include <vector>

#include "assert.h"
#include "stdio.h"

namespace cg = cooperative_groups;

#define CACHE_PRINT 0

// helper #defines - mark # of centroids and size of operational buffers
#define N_CENTROIDS 256

#define COPY_BUFFER_SIZE 1000000
#define ENCODE_BUFFER_SIZE 1000000

namespace gpu_ann {

// update the centroids by calculating centroid values for all keys and
// atomically applying updates.
template <typename centroid_type, typename vector_type, uint tile_size>
__global__ void train_centroids(centroid_type *centroids, uint n_centroids,
                                vector_type *inputs, uint8_t *centroid_ids,
                                uint n_inputs) {
  auto thread_block = cg::this_thread_block();

  cg::thread_block_tile<tile_size> my_tile =
      cg::tiled_partition<tile_size>(thread_block);

  uint64_t tid = gallatin::utils::get_tile_tid(my_tile);

  if (tid >= n_inputs) return;

  uint8_t my_centroid = centroid_ids[tid];

  centroids[my_centroid].merge(&inputs[tid], my_tile);
}

// normalize centroids by # of vectors in their cluster
template <typename vector_type, uint tile_size>
__global__ void normalize_centroids(vector_type *centroids,
                                    uint32_t *cluster_counts,
                                    uint n_centroids) {
  auto thread_block = cg::this_thread_block();

  cg::thread_block_tile<tile_size> my_tile =
      cg::tiled_partition<tile_size>(thread_block);

  uint64_t tid = gallatin::utils::get_tile_tid(my_tile);

  if (tid >= n_centroids) return;

  centroids[tid].normalize(cluster_counts[tid], my_tile);
}

// merge old and new centroids together.
// alpha < 1 does not currently work.
template <typename vector_type, uint tile_size>
__global__ void merge_centroids(vector_type *centroids,
                                vector_type *acc_centroids, uint n_centroids,
                                double alpha) {
  auto thread_block = cg::this_thread_block();

  cg::thread_block_tile<tile_size> my_tile =
      cg::tiled_partition<tile_size>(thread_block);

  uint64_t tid = gallatin::utils::get_tile_tid(my_tile);

  if (tid >= n_centroids) return;

  centroids[tid].EMS(acc_centroids + tid, alpha, my_tile);
  __threadfence();
}

// helper kernel
// given the current centroids and a distance function - update which cluster
// each thread belongs to.
template <typename centroid_data_type, typename data_type, uint n_degrees,
          uint tile_size,
          template <typename, typename, uint, uint> class distance_functor>
__global__ void update_clusters(
    data_vector<centroid_data_type, n_degrees> *centroids,
    uint32_t *cluster_counts, uint n_centroids,
    data_vector<data_type, n_degrees> *inputs, uint8_t *centroid_ids,
    uint n_inputs, uint32_t *converged) {
  auto thread_block = cg::this_thread_block();

  cg::thread_block_tile<tile_size> my_tile =
      cg::tiled_partition<tile_size>(thread_block);

  uint64_t tid = gallatin::utils::get_tile_tid(my_tile);

  if (tid >= n_inputs) return;

  auto min_dist = distance_functor<centroid_data_type, data_type, n_degrees,
                                   tile_size>::distance(&centroids[0],
                                                        &inputs[tid], my_tile);

  uint min_centroid = 0;

  for (uint i = 1; i < n_centroids; i++) {
    auto current_dist =
        distance_functor<centroid_data_type, data_type, n_degrees,
                         tile_size>::distance(&centroids[i], &inputs[tid],
                                              my_tile);

    // if (isnan(current_dist)){
    //    printf("Distance NAN\n");
    // }
    if (current_dist <= min_dist) {
      min_dist = current_dist;
      min_centroid = i;
    }
  }

  if (my_tile.thread_rank() == 0) {
    if (centroid_ids[tid] != min_centroid) {
      atomicAdd(converged, 1);
    }
    centroid_ids[tid] = min_centroid;

    atomicAdd(&cluster_counts[min_centroid], 1U);
  }

  return;
}

// helper kernel - count how many vectors belong to each cluster.
__global__ void set_cluster_counts(uint32_t *cluster_counts,
                                   uint8_t *centroid_ids, uint32_t n_vectors) {
  uint64_t tid = gallatin::utils::get_tid();

  if (tid >= n_vectors) return;

  atomicAdd(&cluster_counts[centroid_ids[tid]], 1);
}

// helper kernel to set centroid to existing vector.
template <typename centroid_data_type, typename data_type, uint n_degrees>
__global__ void copy_centroid_kernel(
    data_vector<centroid_data_type, n_degrees> *centroids, int n_centroids,
    data_vector<data_type, n_degrees> *inputs, uint32_t n_inputs,
    uint64_t *ids_to_pull) {
  uint64_t tid = gallatin::utils::get_tid();

  if (tid >= n_centroids) return;

  uint64_t id = ids_to_pull[tid] % n_inputs;

  centroids[tid].copy_vector_one_thread(inputs + id);
}

// kernel to check if centroids are dead - if se, re-init to a randow vector.
template <typename centroid_type, typename vector_type>
__global__ void reinit_empty_centroids(centroid_type *centroids,
                                       vector_type *input_vectors,
                                       uint32_t *cluster_counts,
                                       uint32_t n_centroids, uint32_t n_vectors,
                                       uint64_t *randomness) {
  uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid >= n_centroids) return;

  if (cluster_counts[tid] == 0) {
    // Pick a random vector index (use tid just to keep it deterministic, or
    // improve with randomness later)
    uint32_t new_idx = randomness[tid] % n_vectors;

    centroids[tid].copy_vector_one_thread(input_vectors + new_idx);
  }
}

template <typename centroid_data_type, uint tile_size,
          uint degrees_per_codebook, typename codebook_type,
          template <typename, typename, uint, uint> class distance_functor>
__global__ void populate_distance_lookup_kernel(float *distances,
                                                codebook_type **codebooks,
                                                uint n_codebooks) {
  // use coorporate group
  auto thread_block = cg::this_thread_block();
  cg::thread_block_tile<tile_size> my_tile =
      cg::tiled_partition<tile_size>(thread_block);

  // tile id, one tile per distance lookup slot.
  uint64_t tid = gallatin::utils::get_tile_tid(my_tile);
  if (tid >= n_codebooks * N_CENTROIDS * N_CENTROIDS) {
    return;
  }

  // which codebook, each codebook has lookup of size
  // N_CENTROIDS * N_CENTROIDS
  int codebook_id = tid / (N_CENTROIDS * N_CENTROIDS);
  codebook_type *codebook = codebooks[codebook_id];

  // which two centroids that lookup belongs to.
  int codebook_offset = tid % (N_CENTROIDS * N_CENTROIDS);
  int centroid_x_id = codebook_offset / N_CENTROIDS;
  int centroid_y_id = codebook_offset % N_CENTROIDS;
  using centroid_type = typename codebook_type::centroid_type;
  centroid_type centroids_x = codebook->centroids[centroid_x_id];
  centroid_type centroids_y = codebook->centroids[centroid_y_id];

  float distance =
      distance_functor<centroid_data_type, centroid_data_type,
                       degrees_per_codebook, tile_size>::distance(&centroids_x,
                                                                  &centroids_y,
                                                                  my_tile);
  distances[tid] = distance * distance;
}

// a codebook takes in a list of vectors
//  and produces a GPU codebook that can be used to cast those vectors
//  into a lower-dimensional representation
//  each codebook returns one uint8_t.
template <typename centroid_data_type, typename data_type, uint n_degrees>
struct codebook {
  using my_type = codebook<centroid_data_type, data_type, n_degrees>;

  using centroid_type = data_vector<centroid_data_type, n_degrees>;
  using vector_type = data_vector<data_type, n_degrees>;

  centroid_type *centroids;

  // initialize the codebook
  static my_type *generate_on_host() {
    my_type *host_version = gallatin::utils::get_host_version<my_type>();
    host_version->centroids = random_data_device<centroid_type>(N_CENTROIDS);
    return host_version;
  }

  // return the centroids in host memory
  __host__ centroid_type *get_centroids() {
    centroid_type *host_version =
        gallatin::utils::get_host_version<centroid_type>(N_CENTROIDS);
    cudaMemcpy(host_version, centroids, sizeof(centroid_type) * N_CENTROIDS,
               cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    return host_version;
  }

  // move the data to host memory
  __host__ void move_to_host() {
    centroid_type *host_version =
        gallatin::utils::get_host_version<centroid_type>(N_CENTROIDS);
    cudaMemcpy(host_version, centroids, sizeof(centroid_type) * N_CENTROIDS,
               cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    cudaFree(centroids);
    centroids = host_version;
  }

  // initialize the centroids to be random input vectors
  __host__ void set_random_centroids(vector_type *input_vectors,
                                     uint32_t n_vectors) {
    uint64_t *ids_to_pull = random_data_device<uint64_t>(N_CENTROIDS);

    copy_centroid_kernel<centroid_data_type, data_type, n_degrees>
        <<<(N_CENTROIDS - 1) / 256 + 1, 256>>>(
            centroids, N_CENTROIDS, input_vectors, n_vectors, ids_to_pull);

    cudaDeviceSynchronize();
    cudaFree(ids_to_pull);
  }

  // host function - train the k-means of this codebook.
  // takes in the list of vectors
  //  returns their encoding and leaves the codebook initialized for future
  //  operations.
  template <uint tile_size,
            template <typename, typename, uint, uint> class distance_functor>
  __host__ uint8_t *train(vector_type *input_vectors, uint32_t n_vectors,
                          double epsilon = .001) {
    const uint64_t n_threads = n_vectors * tile_size;

    uint32_t *converged;
    cudaMallocManaged((void **)&converged, sizeof(uint32_t));

    uint8_t *centroid_ids = random_data_device<uint8_t>(n_vectors);

    cudaMemset(centroid_ids, 0, sizeof(uint8_t) * n_vectors);

    uint32_t *cluster_counts;

    cudaMalloc((void **)&cluster_counts, sizeof(uint32_t) * N_CENTROIDS);

    cudaMemset(cluster_counts, 0, sizeof(uint32_t) * N_CENTROIDS);

    set_random_centroids(input_vectors, n_vectors);

    update_clusters<centroid_data_type, data_type, n_degrees, tile_size,
                    distance_functor><<<(n_threads - 1) / 512 + 1, 512>>>(
        centroids, cluster_counts, N_CENTROIDS, input_vectors, centroid_ids,
        n_vectors, converged);

    cudaDeviceSynchronize();

    converged[0] = n_vectors;

    centroid_type *acc_centroids =
        gallatin::utils::get_device_version<centroid_type>(N_CENTROIDS);

    cudaMemset(acc_centroids, 0, sizeof(centroid_type) * N_CENTROIDS);

    // printf("Starting training of centroids\n");

    uint n_rounds = 0;

    set_cluster_counts<<<(n_vectors - 1) / 512 + 1, 512>>>(
        cluster_counts, centroid_ids, n_vectors);

    while (1.0 * converged[0] / n_vectors > epsilon) {
      converged[0] = 0;

      cudaMemset(acc_centroids, 0, sizeof(centroid_type) * N_CENTROIDS);

      // update centroids.
      train_centroids<centroid_type, vector_type, tile_size>
          <<<(n_threads - 1) / 512 + 1, 512>>>(acc_centroids, N_CENTROIDS,
                                               input_vectors, centroid_ids,
                                               n_vectors);

      normalize_centroids<centroid_type, tile_size>
          <<<(tile_size * N_CENTROIDS - 1) / 512 + 1, 512>>>(
              acc_centroids, cluster_counts, N_CENTROIDS);

      merge_centroids<centroid_type, tile_size>
          <<<(tile_size * N_CENTROIDS - 1) / 512 + 1, 512>>>(
              centroids, acc_centroids, N_CENTROIDS, 1);
      // template function definition.
      // data_type (*euclidean)(vector_type&, vector_type&,
      // cg::thread_block_tile<tile_size>&) = euclidean_distance<data_type,
      // n_degrees, tile_size>;

      cudaMemset(cluster_counts, 0, sizeof(uint32_t) * N_CENTROIDS);

      update_clusters<centroid_data_type, data_type, n_degrees, tile_size,
                      euclidean_distance_no_sqrt>
          <<<(n_threads - 1) / 512 + 1, 512>>>(
              centroids, cluster_counts, N_CENTROIDS, input_vectors,
              centroid_ids, n_vectors, converged);

      uint64_t *randomness = random_data_device<uint64_t>(N_CENTROIDS);

      cudaDeviceSynchronize();

      reinit_empty_centroids<centroid_type, vector_type>
          <<<N_CENTROIDS - 1 / 256 + 1, 256>>>(centroids, input_vectors,
                                               cluster_counts, N_CENTROIDS,
                                               n_vectors, randomness);

      cudaFree(randomness);

      std::cout << "Round " << n_rounds++ << ": "
                << 1.0 * converged[0] / n_vectors << " unconverged \r";
      std::cout.flush();

      if (n_rounds >= 100) break;
    }

    std::cout << "Finished on round: " << n_rounds++ << ": "
              << 1.0 * converged[0] / n_vectors << " unconverged \n";
    std::cout.flush();

    cudaFree(cluster_counts);

    cudaFree(converged);

    cudaFree(acc_centroids);
    // cudaFree(centroid_ids);
    return centroid_ids;
  }

  // decode quantized vector
  // use the quantized vector with centroids to approximate the original vector
  centroid_type decode(uint8_t centroid_idx) { return centroids[centroid_idx]; }

  // device helper to encode data
  // generates a centroid for a given vector
  template <typename large_vector_type, uint tile_size,
            template <typename, typename, uint, uint> class distance_functor>
  __device__ uint8_t device_encode(cg::thread_block_tile<tile_size> &my_tile,
                                   uint64_t tid,  // tile id within the block
                                   const large_vector_type &input_vector,
                                   uint codebook_id) {
    // total number of tile per block = threads per block / threads per tile.
    const uint n_tiles = 256 / tile_size;

    // create a vector for each tile, and assign the vector that belongs to this
    // tile. with my_vector
    __shared__ vector_type temp_vectors[n_tiles];

    vector_type *my_vector = &temp_vectors[tid];

    my_tile.sync();

    // dimension of the input vector (128)
    constexpr uint large_vector_size =
        data_vector_traits<large_vector_type>::size;

    // dimension of the centroids (2 = 128 / 64)
    const uint small_vector_size = data_vector_traits<vector_type>::size;

    // copy vector with offset (starting from codebook_id * small_vector_size)
    my_vector->template copy_vector_offset<
        data_vector_traits<large_vector_type>::type, large_vector_size>(
        input_vector, codebook_id * small_vector_size);

    __threadfence();
    my_tile.sync();

    auto min_dist =
        distance_functor<centroid_data_type, data_type, n_degrees,
                         tile_size>::distance(&centroids[0], &my_vector[0],
                                              my_tile);

    uint min_centroid = 0;

    for (uint i = 1; i < N_CENTROIDS; i++) {
      auto current_dist =
          distance_functor<centroid_data_type, data_type, n_degrees,
                           tile_size>::distance(&centroids[i], &my_vector[0],
                                                my_tile);

      // if (isnan(current_dist)){
      //    printf("Distance NAN\n");
      // }
      if (current_dist <= min_dist) {
        min_dist = current_dist;
        min_centroid = i;
      }
    }

    return min_centroid;
  }
};

// helper kernel - copies data from vectors into intermediates.
template <typename data_type, uint n_degrees_in, uint n_degrees_intermediate>
__global__ void copy_vector_kernel(
    data_vector<data_type, n_degrees_intermediate> *intermediates,
    data_vector<data_type, n_degrees_in> *inputs, uint degree_offset,
    uint32_t array_offset, uint32_t n_keys) {
  uint64_t tid = gallatin::utils::get_tid();

  if (tid >= n_keys) return;

  intermediates[tid + array_offset].copy_vector_offset(inputs[tid],
                                                       degree_offset);
}

// helper kernel to merge intermediate results into one kernel.
template <typename output_vector_type, uint tile_size>
__global__ void combine_outputs(output_vector_type *outputs,
                                uint8_t **intermediates, uint n_codebooks,
                                uint32_t n_vectors) {
  auto thread_block = cg::this_thread_block();

  cg::thread_block_tile<tile_size> my_tile =
      cg::tiled_partition<tile_size>(thread_block);

  uint64_t tid = gallatin::utils::get_tile_tid(my_tile);

  if (tid >= n_vectors) return;

  for (uint i = my_tile.thread_rank(); i < n_codebooks; i += my_tile.size()) {
    outputs[tid].data[i] = intermediates[i][tid];
  }
}

template <typename output_vector_type, typename codebook_type,
          typename input_vector_type, uint tile_size,
          template <typename, typename, uint, uint> class distance_functor>
__global__ void buffered_encode_kernel(output_vector_type *outputs,
                                       uint32_t starting_offset,
                                       codebook_type **codebooks,
                                       uint n_codebooks,
                                       input_vector_type *inputs,
                                       uint32_t n_inputs) {
  // Which thread block you belong to.
  // compose of several threads.
  auto thread_block = cg::this_thread_block();

  // break the threads into tile_size.
  // it will return the same tile for the threads in the same tile.
  cg::thread_block_tile<tile_size> my_tile =
      cg::tiled_partition<tile_size>(thread_block);

  // What is the global id of your tile.
  uint64_t tid = gallatin::utils::get_tile_tid(my_tile);

  using codebook_vector_type = typename codebook_type::vector_type;

  if (tid >= n_inputs) return;

  for (uint i = 0; i < n_codebooks; i++) {
    outputs[starting_offset + tid].data[i] =
        codebooks[i]
            ->template device_encode<input_vector_type, tile_size,
                                     distance_functor>(
                my_tile, my_tile.meta_group_rank(), inputs[tid], i);
    // codebooks[i]->test_encode(inputs[tid]);

    // outputs[tid].data[i] = codebooks[i]->template device_encode<tile_size,
    // distance_functor>(my_tile, inputs[tid]);
  }
}

template <typename output_vector_type, typename centroid_type,
          typename codebook_type, typename input_vector_type, uint tile_size,
          template <typename, typename, uint, uint> class distance_functor>
__global__ void buffered_centroid_kernel(centroid_type *centroid,
                                         uint32_t starting_offset,
                                         output_vector_type *inputs,
                                         uint32_t n_inputs) {
  auto thread_block = cg::this_thread_block();

  cg::thread_block_tile<tile_size> my_tile =
      cg::tiled_partition<tile_size>(thread_block);

  uint64_t tid = gallatin::utils::get_tile_tid(my_tile);

  if (tid >= n_inputs) return;

  centroid->merge(inputs + starting_offset + tid, my_tile);
}

// helper kernel to encode inputs to output data type.
//  also records data to the centroid of the entire set.
template <typename output_vector_type, typename centroid_type,
          typename codebook_type, typename input_vector_type, uint tile_size,
          template <typename, typename, uint, uint> class distance_functor>
__global__ void buffered_encode_kernel_centroid(
    output_vector_type *outputs, centroid_type *centroid,
    uint32_t starting_offset, codebook_type **codebooks, uint n_codebooks,
    input_vector_type *inputs, uint32_t n_inputs) {
  auto thread_block = cg::this_thread_block();

  cg::thread_block_tile<tile_size> my_tile =
      cg::tiled_partition<tile_size>(thread_block);

  uint64_t tid = gallatin::utils::get_tile_tid(my_tile);

  using codebook_vector_type = typename codebook_type::vector_type;

  if (tid >= n_inputs) return;

  for (uint i = 0; i < n_codebooks; i++) {
    outputs[starting_offset + tid].data[i] =
        codebooks[i]
            ->template device_encode<input_vector_type, tile_size,
                                     distance_functor>(
                my_tile, my_tile.meta_group_rank(), inputs[tid], i);
    // codebooks[i]->test_encode(inputs[tid]);

    // outputs[tid].data[i] = codebooks[i]->template device_encode<tile_size,
    // distance_functor>(my_tile, inputs[tid]);
  }

  __threadfence();

  my_tile.sync();

  centroid->merge(outputs + starting_offset + tid, my_tile);
}

// helper kernel - downcasts centroid to a smaller output data type.
template <uint tile_size, typename centroid_vector_type,
          typename output_vector_type>
__global__ void normalize_and_downsize_kernel(centroid_vector_type *centroid,
                                              output_vector_type *output,
                                              uint32_t n_vectors) {
  auto thread_block = cg::this_thread_block();

  cg::thread_block_tile<tile_size> my_tile =
      cg::tiled_partition<tile_size>(thread_block);

  uint64_t tid = gallatin::utils::get_tile_tid(my_tile);

  if (tid != 0) return;

  centroid->normalize(n_vectors, my_tile);

  __threadfence();

  output->copy_vector_offset(centroid[0], 0);
}

template <uint tile_size, typename vector_type,
          template <typename, typename, uint, uint> class distance_functor>
__global__ void calculate_medoid_kernel(vector_type *vectors,
                                        vector_type *centroid,
                                        uint32_t n_vectors, uint64_t *output) {
  auto thread_block = cg::this_thread_block();

  cg::thread_block_tile<tile_size> my_tile =
      cg::tiled_partition<tile_size>(thread_block);

  uint64_t tid = gallatin::utils::get_tile_tid(my_tile);

  if (tid >= n_vectors) return;

  const uint vector_size = data_vector_traits<vector_type>::size;

  using vector_data_type = typename data_vector_traits<vector_type>::type;

  float my_dist =
      distance_functor<vector_data_type, vector_data_type, vector_size,
                       tile_size>::distance(&vectors[tid], &centroid[0],
                                            my_tile);

  uint32_t dist_cast = (uint32_t)my_dist;

  uint64_t dist_upscaled = ((uint64_t)dist_cast) << 32;

  dist_upscaled = dist_upscaled + tid;

  atomicMin((unsigned long long int *)&output[0],
            (unsigned long long int)dist_upscaled);
}

// product quantizer
// maps arbitrary input vectors into uint8_t in steps.
//  1. chop each vector into in/out degrees and place on GPU
//  2. run K-means clustering with K=256
//  3. map input vectors using k-means codebook and write back.
template <typename centroid_data_type, typename data_type, uint n_degrees_in,
          uint n_degrees_out>
struct product_quantizer {
  // needs to be over as
  static constexpr uint degrees_per_codebook =
      (n_degrees_in - 1) / n_degrees_out + 1;

  static constexpr uint n_codebooks =
      (n_degrees_in - 1) / degrees_per_codebook + 1;

  using input_vector_type = data_vector<data_type, n_degrees_in>;
  using intermediate_vector_type = data_vector<data_type, degrees_per_codebook>;
  using output_vector_type = data_vector<uint8_t, n_degrees_out>;

  using codebook_type =
      codebook<centroid_data_type, data_type, degrees_per_codebook>;
  using centroid_type = data_vector<centroid_data_type, degrees_per_codebook>;

  codebook_type **codebooks;

  // Lookup table of size (n_codebooks * N_CENTROIDS * N_CENTROIDS)
  // which stores the distance between each pair of centroids.
  float *device_distances;

  product_quantizer() {
    codebooks = gallatin::utils::get_host_version<codebook_type *>(n_codebooks);

    for (uint i = 0; i < n_codebooks; i++) {
      codebooks[i] = codebook_type::generate_on_host();
    }

    device_distances = gallatin::utils::get_device_version<float>(
        n_codebooks * N_CENTROIDS * N_CENTROIDS);
  }

  // given a set of vectors and a starting degree for extraction, generate
  // intermediates which are slices of the original vectors.
  intermediate_vector_type *generate_intermediates(int degree_start,
                                                   input_vector_type *inputs,
                                                   uint32_t n_vectors) {
    intermediate_vector_type *intermediates =
        gallatin::utils::get_device_version<intermediate_vector_type>(
            n_vectors);

    cudaMemset(intermediates, 0, sizeof(intermediate_vector_type) * n_vectors);

    input_vector_type *copy_vectors =
        gallatin::utils::get_device_version<input_vector_type>(
            COPY_BUFFER_SIZE);

    uint64_t offset = 0;

    while (offset + COPY_BUFFER_SIZE < n_vectors) {
      cudaMemcpy(copy_vectors, inputs + offset,
                 sizeof(input_vector_type) * COPY_BUFFER_SIZE,
                 cudaMemcpyHostToDevice);

      copy_vector_kernel<<<(COPY_BUFFER_SIZE - 1) / 256 + 1, 256>>>(
          intermediates, copy_vectors, degree_start, offset, COPY_BUFFER_SIZE);
      offset += COPY_BUFFER_SIZE;
    }

    uint64_t remaining = n_vectors - offset;

    cudaMemcpy(copy_vectors, inputs + offset,
               sizeof(input_vector_type) * remaining, cudaMemcpyHostToDevice);

    copy_vector_kernel<<<(COPY_BUFFER_SIZE - 1) / 256 + 1, 256>>>(
        intermediates, copy_vectors, degree_start, offset, remaining);

    cudaFree(copy_vectors);

    return intermediates;
  }

  // runs a kernel to combine individual codebook outputs back into output
  // vectors.
  template <uint tile_size>
  output_vector_type *recombine(uint8_t **aggregates, uint32_t n_vectors) {
    output_vector_type *outputs =
        gallatin::utils::get_device_version<output_vector_type>(n_vectors);

    combine_outputs<output_vector_type, tile_size>
        <<<(n_vectors * tile_size - 1) / 256 + 1, 256>>>(
            outputs, aggregates, n_codebooks, n_vectors);

    cudaDeviceSynchronize();

    return outputs;
  }

  // run k-means for each codebook to process keys.
  // for each code book, we generate a set of intermediate vectors on device
  //  and then run k-means on those vectors.
  template <uint tile_size,
            template <typename, typename, uint, uint> class distance_functor>
  output_vector_type *train(input_vector_type *inputs, uint32_t n_vectors) {
    uint8_t **outputs;

    cudaMallocManaged((void **)&outputs, sizeof(uint8_t *) * n_degrees_out);

    for (uint i = 0; i < n_codebooks; i++) {
      printf("Training %u\n", i);

      intermediate_vector_type *intermediates =
          generate_intermediates(i * degrees_per_codebook, inputs, n_vectors);

      outputs[i] = codebooks[i]->template train<tile_size, distance_functor>(
          intermediates, n_vectors);

      cudaFree(intermediates);
    }

    printf("All trained\n");

    // and convert into compressed form.
    auto final_vectors = recombine<tile_size>(outputs, n_vectors);

    for (uint i = 0; i < n_degrees_out; i++) {
      cudaFree(outputs[i]);
    }
    cudaFree(outputs);

    return final_vectors;
  }

  // using existing codebooks, decode a compressed vector
  // TODO: this is not a parallelized version
  input_vector_type *decode(output_vector_type *input, uint32_t n_vectors) {
    // std::vector<input_vector_type> outputs(n_vectors);
    input_vector_type *outputs =
        gallatin::utils::get_host_version<input_vector_type>(n_vectors);

    // for each output vector
    for (uint i = 0; i < n_vectors; ++i) {
      // for each codebook segment
      for (uint j = 0; j < n_codebooks; ++j) {
        auto centroid = codebooks[j]->decode(input[i][j]);
        for (uint k = 0; k < degrees_per_codebook; ++k) {
          outputs[i][j * degrees_per_codebook + k] =
              static_cast<uint8_t>(centroid[k]);
        }
      }
    }
    return outputs;
  }

  // using existing codebooks, convert a batch of vectors to the compressed
  // variant.
  template <uint tile_size>
  output_vector_type *encode(input_vector_type *inputs, uint32_t n_vectors) {
    output_vector_type *outputs =
        gallatin::utils::get_device_version<output_vector_type>(n_vectors);

    // thius encode kernel uses one

    uint64_t n_threads = ((uint64_t)tile_size) * n_vectors;
    buffered_encode_kernel<output_vector_type, codebook_type, input_vector_type,
                           tile_size, euclidean_distance_no_sqrt>
        <<<(n_threads - 1) / 256 + 1, 256>>>(outputs, 0, codebooks, n_codebooks,
                                             inputs, n_vectors);

    cudaDeviceSynchronize();
    return outputs;
  }

  // buffered variant, takes chunks of 1,000,000 vectors at a time.
  template <uint tile_size>
  output_vector_type *encode_buffered(input_vector_type *inputs,
                                      uint32_t n_vectors) {
    output_vector_type *outputs =
        gallatin::utils::get_device_version<output_vector_type>(n_vectors);

    input_vector_type *buffer =
        gallatin::utils::get_device_version<input_vector_type>(
            ENCODE_BUFFER_SIZE);

    uint32_t start = 0;

    // run buffers of MAX size
    while (start + ENCODE_BUFFER_SIZE < n_vectors) {
      cudaMemcpy(buffer, inputs + start,
                 sizeof(input_vector_type) * ENCODE_BUFFER_SIZE,
                 cudaMemcpyHostToDevice);

      uint64_t n_threads = ((uint64_t)tile_size) * ENCODE_BUFFER_SIZE;
      buffered_encode_kernel<output_vector_type, codebook_type,
                             input_vector_type, tile_size,
                             euclidean_distance_no_sqrt>
          <<<(n_threads - 1) / 256 + 1, 256>>>(outputs, start, codebooks,
                                               n_codebooks, buffer,
                                               ENCODE_BUFFER_SIZE);

      start += ENCODE_BUFFER_SIZE;
    }

    // one round to process remainder
    uint64_t remainder = n_vectors - start;

    cudaMemcpy(buffer, inputs + start, sizeof(input_vector_type) * remainder,
               cudaMemcpyHostToDevice);

    uint64_t n_threads = ((uint64_t)tile_size) * remainder;
    buffered_encode_kernel<output_vector_type, codebook_type, input_vector_type,
                           tile_size, euclidean_distance_no_sqrt>
        <<<(n_threads - 1) / 256 + 1, 256>>>(outputs, start, codebooks,
                                             n_codebooks, buffer, remainder);

    cudaFree(buffer);
    cudaDeviceSynchronize();
    return outputs;
  }

  // encodes host->host using two GPU buffers
  //  and calculates centroid during processing.
  template <uint tile_size>
  std::pair<output_vector_type *, output_vector_type *>
  encode_buffered_with_centroid(input_vector_type *inputs, uint32_t n_vectors) {
    // construct buffers for device operations.
    using centroid_vector_type = data_vector<centroid_data_type, n_degrees_out>;
    output_vector_type *outputs =
        gallatin::utils::get_host_version<output_vector_type>(n_vectors);

    output_vector_type *output_buffer =
        gallatin::utils::get_device_version<output_vector_type>(
            ENCODE_BUFFER_SIZE);

    centroid_vector_type *centroid =
        gallatin::utils::get_device_version<centroid_vector_type>(
            ENCODE_BUFFER_SIZE);

    input_vector_type *buffer =
        gallatin::utils::get_device_version<input_vector_type>(
            ENCODE_BUFFER_SIZE);

    // Extra centroid of the whole dataset is calculated.
    cudaMemset(centroid, 0, sizeof(output_vector_type));

    uint32_t start = 0;

    // run through all batches of max size.
    while (start + ENCODE_BUFFER_SIZE < n_vectors) {
      cudaMemcpy(buffer, inputs + start,
                 sizeof(input_vector_type) * ENCODE_BUFFER_SIZE,
                 cudaMemcpyHostToDevice);

      uint64_t n_threads = ((uint64_t)tile_size) * ENCODE_BUFFER_SIZE;
      buffered_encode_kernel_centroid<output_vector_type, centroid_vector_type,
                                      codebook_type, input_vector_type,
                                      tile_size, euclidean_distance_no_sqrt>
          <<<(n_threads - 1) / 256 + 1, 256>>>(output_buffer, centroid, 0,
                                               codebooks, n_codebooks, buffer,
                                               ENCODE_BUFFER_SIZE);

      cudaMemcpy(outputs + start, output_buffer,
                 sizeof(output_vector_type) * ENCODE_BUFFER_SIZE,
                 cudaMemcpyDeviceToHost);

      start += ENCODE_BUFFER_SIZE;
    }

    // final batch to cover stragglers.
    uint64_t remainder = n_vectors - start;

    cudaMemcpy(buffer, inputs + start, sizeof(input_vector_type) * remainder,
               cudaMemcpyHostToDevice);

    uint64_t n_threads = ((uint64_t)tile_size) * remainder;
    buffered_encode_kernel_centroid<output_vector_type, centroid_vector_type,
                                    codebook_type, input_vector_type, tile_size,
                                    euclidean_distance_no_sqrt>
        <<<(n_threads - 1) / 256 + 1, 256>>>(outputs, centroid, 0, codebooks,
                                             n_codebooks, buffer, remainder);

    cudaMemcpy(outputs + start, output_buffer,
               sizeof(output_vector_type) * remainder, cudaMemcpyDeviceToHost);

    output_vector_type *host_centroid =
        gallatin::utils::get_host_version<output_vector_type>();

    // centroid is much larger data type - need to downcast to
    // output_vector_type
    normalize_and_downsize_kernel<tile_size, centroid_vector_type,
                                  output_vector_type>
        <<<1, tile_size>>>(centroid, host_centroid, n_vectors);

    cudaFree(buffer);
    cudaFree(output_buffer);
    cudaFree(centroid);

    cudaDeviceSynchronize();
    return {outputs, host_centroid};
  }

  template <uint tile_size>
  static output_vector_type *determine_centroid(output_vector_type *inputs,
                                                uint32_t n_vectors) {
    // construct buffers for device operations.
    using centroid_vector_type = data_vector<centroid_data_type, n_degrees_out>;

    centroid_vector_type *centroid =
        gallatin::utils::get_device_version<centroid_vector_type>(
            ENCODE_BUFFER_SIZE);

    output_vector_type *buffer =
        gallatin::utils::get_device_version<output_vector_type>(
            ENCODE_BUFFER_SIZE);

    // Extra centroid of the whole dataset is calculated.
    cudaMemset(centroid, 0, sizeof(output_vector_type));

    uint32_t start = 0;

    // run through all batches of max size.
    while (start + ENCODE_BUFFER_SIZE < n_vectors) {
      cudaMemcpy(buffer, inputs + start,
                 sizeof(output_vector_type) * ENCODE_BUFFER_SIZE,
                 cudaMemcpyHostToDevice);

      uint64_t n_threads = ((uint64_t)tile_size) * ENCODE_BUFFER_SIZE;
      buffered_centroid_kernel<output_vector_type, centroid_vector_type,
                               codebook_type, input_vector_type, tile_size,
                               euclidean_distance_no_sqrt>
          <<<(n_threads - 1) / 256 + 1, 256>>>(centroid, 0, buffer,
                                               ENCODE_BUFFER_SIZE);

      start += ENCODE_BUFFER_SIZE;
    }

    // final batch to cover stragglers.
    uint64_t remainder = n_vectors - start;

    cudaMemcpy(buffer, inputs + start, sizeof(output_vector_type) * remainder,
               cudaMemcpyHostToDevice);

    uint64_t n_threads = ((uint64_t)tile_size) * remainder;
    buffered_centroid_kernel<output_vector_type, centroid_vector_type,
                             codebook_type, input_vector_type, tile_size,
                             euclidean_distance_no_sqrt>
        <<<(n_threads - 1) / 256 + 1, 256>>>(centroid, 0, buffer, remainder);

    output_vector_type *host_centroid =
        gallatin::utils::get_host_version<output_vector_type>();

    // centroid is much larger data type - need to downcast to
    // output_vector_type
    normalize_and_downsize_kernel<tile_size, centroid_vector_type,
                                  output_vector_type>
        <<<1, tile_size>>>(centroid, host_centroid, n_vectors);

    cudaDeviceSynchronize();

    cudaFree(buffer);
    cudaFree(centroid);

    cudaDeviceSynchronize();
    return host_centroid;
  }

  template <uint tile_size>
  static uint32_t calculate_medoid(output_vector_type *vectors,
                                   output_vector_type *centroid,
                                   uint32_t n_vectors) {
    // construct buffers for device operations.

    uint64_t *output;

    cudaMallocManaged((void **)&output, sizeof(uint64_t));

    output[0] = ~0ULL;

    calculate_medoid_kernel<tile_size, output_vector_type,
                            euclidean_distance_no_sqrt>
        <<<(n_vectors * tile_size - 1) / 256 + 1, 256>>>(vectors, centroid,
                                                         n_vectors, output);

    cudaDeviceSynchronize();

    uint32_t result = (output[0] & BITMASK(32));

    cudaFree(output);

    return result;
  }

  // Store the codebooks and distances to a binary file.
  void store_codebooks_and_distances(std::string filename) {
    std::ofstream file(filename, std::ios::out | std::ios::binary);
    if (!file.is_open()) {
      std::cerr << "Error opening file:" << filename << std::endl;
      return;
    }

    uint number_codebooks = n_codebooks;
    uint number_centroids = N_CENTROIDS;
    uint number_degrees_per_codebook = degrees_per_codebook;
    uint centroid_data_size = sizeof(centroid_data_type);

    file.write(reinterpret_cast<char *>(&number_codebooks), 4);
    file.write(reinterpret_cast<char *>(&number_centroids), 4);
    file.write(reinterpret_cast<char *>(&number_degrees_per_codebook), 4);
    file.write(reinterpret_cast<char *>(&centroid_data_size), 4);

    // Store centroids
    std::cout << "Storing codebook with dimension: n_codebooks="
              << number_codebooks << ", number_centroids=" << number_centroids
              << ", degrees_per_codebook=" << degrees_per_codebook
              << ", centroid_data_size=" << centroid_data_size << std::endl;

    uint centroids_size =
        N_CENTROIDS * number_degrees_per_codebook * centroid_data_size;
    for (int i = 0; i < n_codebooks; ++i) {
      centroid_type *centroids = codebooks[i]->get_centroids();
      file.write(reinterpret_cast<char *>(centroids), centroids_size);
      cudaFreeHost(centroids);
    }

    // Store distances
    std::cout << "Storing distance lookup table" << std::endl;
    uint64_t n_distances = n_codebooks * N_CENTROIDS * N_CENTROIDS;
    float *host_distances =
        gallatin::utils::get_host_version<float>(n_distances);
    cudaMemcpy(host_distances, device_distances, sizeof(float) * n_distances,
               cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    file.write(reinterpret_cast<char *>(host_distances),
               n_distances * sizeof(float));
    cudaFreeHost(host_distances);
  }

  // Load the codebooks from a binary file
  void load_codebooks_and_distances(std::string filename) {
    // Open file
    std::ifstream file(filename, std::ios::in | std::ios::binary);
    if (!file.is_open()) {
      std::cerr << "Error opening file: " << filename << std::endl;
      return;
    }

    // Get the size of the file
    file.seekg(0, std::ios::end);
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    // Error checking, the size of the file should equal to
    // 16 + n_codebooks * number_centroids * degrees_per_codebook *
    // centroid_data_size + n_codebooks * N_CENTROIDS * N_CENTROIDS
    // * sizeof(float)
    uint64_t target_size =
        16 +
        n_codebooks * N_CENTROIDS * degrees_per_codebook *
            sizeof(centroid_data_type) +
        n_codebooks * N_CENTROIDS * N_CENTROIDS * sizeof(float);
    if (size != target_size) {
      std::cerr << "Invalid codebook binary file size, expect=" << target_size
                << ", got=" << size << std::endl;
      return;
    }

    // Error check if the format of the codebook is correct.
    uint number_codebooks;
    uint number_centroids;
    uint number_degrees_per_codebook;
    uint centroid_data_size;
    file.read(reinterpret_cast<char *>(&number_codebooks), 4);
    file.read(reinterpret_cast<char *>(&number_centroids), 4);
    file.read(reinterpret_cast<char *>(&number_degrees_per_codebook), 4);
    file.read(reinterpret_cast<char *>(&centroid_data_size), 4);
    if (number_codebooks != n_codebooks || number_centroids != N_CENTROIDS ||
        number_degrees_per_codebook != degrees_per_codebook ||
        centroid_data_size != sizeof(centroid_data_type)) {
      std::cerr << "Invalid codebook format." << std::endl;
      return;
    }
    size -= 16;

    // Parse codebooks
    std::cout << "Parsing codebook with dimension: file_size=" << size
              << ", n_codebooks=" << number_codebooks
              << ", number_centroids=" << number_centroids
              << ", degrees_per_codebook=" << degrees_per_codebook
              << ", centroid_data_size=" << centroid_data_size << std::endl;

    uint centroids_size =
        N_CENTROIDS * degrees_per_codebook * centroid_data_size;
    for (int i = 0; i < n_codebooks; ++i) {
      centroid_type *centroids =
          gallatin::utils::get_host_version<centroid_type>(N_CENTROIDS);
      if (!file.read(reinterpret_cast<char *>(centroids), centroids_size)) {
        std::cerr << "Error while reading the file." << std::endl;
      }
      codebooks[i]->centroids = gallatin::utils::move_to_device<centroid_type>(
          centroids, N_CENTROIDS);
    }

    // Parse distances lookup table
    std::cout << "Parsing distance lookup table" << std::endl;
    uint64_t n_distances = n_codebooks * N_CENTROIDS * N_CENTROIDS;
    uint64_t distances_size = n_distances * sizeof(float);
    float *host_distances =
        gallatin::utils::get_host_version<float>(n_distances);
    if (!file.read(reinterpret_cast<char *>(host_distances), distances_size)) {
      std::cerr << "Error while reading the file." << std::endl;
    }
    cudaFree(device_distances);
    device_distances =
        gallatin::utils::move_to_device<float>(host_distances, n_distances);
  }

  // Populating the codebooks' distance lookup table for the centroids
  template <uint tile_size>
  void populate_distance_lookup() {
    uint64_t num_threads =
        ((uint64_t)tile_size) * n_codebooks * N_CENTROIDS * N_CENTROIDS;
    uint64_t num_block = (num_threads - 1) / 256 + 1;
    uint64_t num_threads_per_block = 256;

    populate_distance_lookup_kernel<centroid_data_type, tile_size,
                                    degrees_per_codebook, codebook_type,
                                    euclidean_distance_no_sqrt>
        <<<num_block, num_threads_per_block>>>(device_distances, codebooks,
                                               n_codebooks);
    cudaDeviceSynchronize();
  }
};

template <uint32_t tile_size, typename vector_type, typename quantized_type,
          template <typename, typename, uint, uint> class distance_functor>
__global__ void quantized_error_kernel(vector_type *original_vectors,
                                       quantized_type *quantized_vectors,
                                       uint32_t n_vectors, uint64_t *misses,
                                       uint32_t *first_vec,
                                       uint32_t *second_vec,
                                       uint32_t *third_vec) {
  auto thread_block = cg::this_thread_block();

  cg::thread_block_tile<tile_size> my_tile =
      cg::tiled_partition<tile_size>(thread_block);

  uint64_t tid = gallatin::utils::get_tile_tid(my_tile);

  uint32_t first_id = first_vec[tid] % n_vectors;
  uint32_t second_id = second_vec[tid] % n_vectors;
  uint32_t third_id = third_vec[tid] % n_vectors;

  // printf("%u->%u, %u->%u\n", first_id, second_id, first_id, third_id);

  constexpr uint32_t vector_size = data_vector_traits<vector_type>::size;
  using vector_data_type = typename data_vector_traits<vector_type>::type;

  auto dist_1_2 =
      distance_functor<vector_data_type, vector_data_type, vector_size,
                       tile_size>::distance(original_vectors[first_id],
                                            original_vectors[second_id],
                                            my_tile);

  auto dist_1_3 =
      distance_functor<vector_data_type, vector_data_type, vector_size,
                       tile_size>::distance(original_vectors[first_id],
                                            original_vectors[third_id],
                                            my_tile);

  constexpr uint32_t quantized_size = data_vector_traits<quantized_type>::size;
  using quantized_data_type = typename data_vector_traits<quantized_type>::type;

  auto q_dist_1_2 =
      distance_functor<quantized_data_type, quantized_data_type, quantized_size,
                       tile_size>::distance(quantized_vectors[first_id],
                                            quantized_vectors[second_id],
                                            my_tile);

  auto q_dist_1_3 =
      distance_functor<quantized_data_type, quantized_data_type, quantized_size,
                       tile_size>::distance(quantized_vectors[first_id],
                                            quantized_vectors[third_id],
                                            my_tile);

  bool dist_smaller = (dist_1_2 <= dist_1_3);
  bool q_smaller = (q_dist_1_2 <= q_dist_1_3);

  if (dist_smaller == q_smaller) {
    if (my_tile.thread_rank() == 0) {
      atomicAdd((unsigned long long int *)misses, 1ULL);
    }
  }
}

// for each, do the comparison.
template <typename vector_type, typename quantized_type,
          template <typename, typename, uint, uint> class distance_functor>
__host__ void sample_quantized_error(vector_type *original_vectors,
                                     quantized_type *quantized_vectors,
                                     uint32_t n_vectors, uint64_t n_sims) {
  uint64_t *matches;

  cudaMallocManaged((void **)&matches, sizeof(uint64_t));

  matches[0] = 0;

  cudaDeviceSynchronize();

  const uint32_t tile_size = 4;

  uint32_t *first_vec = random_data_device<uint32_t>(n_sims);
  uint32_t *second_vec = random_data_device<uint32_t>(n_sims);
  uint32_t *third_vec = random_data_device<uint32_t>(n_sims);

  quantized_error_kernel<tile_size, vector_type, quantized_type,
                         distance_functor>
      <<<(tile_size * n_sims - 1) / 256 + 1, 256>>>(
          original_vectors, quantized_vectors, n_vectors, matches, first_vec,
          second_vec, third_vec);

  cudaDeviceSynchronize();

  uint64_t n_matches = matches[0];
  printf("%lu/%lu distance comps match, %f match\n", n_matches, n_sims,
         100.0 * n_matches / n_sims);
  cudaFree(matches);

  cudaFree(first_vec);
  cudaFree(second_vec);
  cudaFree(third_vec);
}

template <typename vector_type, typename quantized_type, uint32_t tile_size,
          template <typename, typename, uint, uint> class full_distance_functor,
          template <typename, typename, uint, uint> class pq_distance_functor>
__global__ void lookup_error_kernel(quantized_type *quantized_vectors,
                                    vector_type *decoded_vectors,
                                    uint64_t *misses, uint32_t n_vectors,
                                    uint64_t n_sims, uint32_t *first_vec,
                                    uint32_t *second_vec, uint32_t *third_vec) {
  auto thread_block = cg::this_thread_block();
  cg::thread_block_tile<tile_size> my_tile =
      cg::tiled_partition<tile_size>(thread_block);
  uint64_t tid = gallatin::utils::get_tile_tid(my_tile);

  uint32_t first_id = first_vec[tid] % n_vectors;
  uint32_t second_id = second_vec[tid] % n_vectors;
  uint32_t third_id = third_vec[tid] % n_vectors;

  constexpr uint32_t vector_size = data_vector_traits<vector_type>::size;
  using vector_data_type = typename data_vector_traits<vector_type>::type;
  using quantized_data_type = typename data_vector_traits<quantized_type>::type;

  // distance of decoded_vectors
  auto decode_dist_1_2 =
      full_distance_functor<vector_data_type, vector_data_type, vector_size,
                            tile_size>::distance(decoded_vectors + first_id,
                                                 decoded_vectors + second_id,
                                                 my_tile);
  auto decode_dist_2_3 =
      full_distance_functor<vector_data_type, vector_data_type, vector_size,
                            tile_size>::distance(decoded_vectors + second_id,
                                                 decoded_vectors + third_id,
                                                 my_tile);

  // distance from lookup table
  constexpr uint32_t quantized_size = data_vector_traits<quantized_type>::size;
  auto pq_dist_1_2 =
      pq_distance_functor<quantized_data_type, quantized_data_type,
                          quantized_size,
                          tile_size>::distance(quantized_vectors+first_id,
                                               quantized_vectors+second_id,
                                               my_tile);
  auto pq_dist_2_3 =
      pq_distance_functor<quantized_data_type, quantized_data_type,
                          quantized_size,
                          tile_size>::distance(quantized_vectors+second_id,
                                               quantized_vectors+third_id,
                                               my_tile);

  // compare distance
  bool decoded_smaller = (decode_dist_1_2 <= decode_dist_2_3);
  bool pq_smaller = (pq_dist_1_2 <= pq_dist_2_3);
  if (my_tile.thread_rank() == 0) {
    if (decoded_smaller != pq_smaller) {
      printf("exact_dist= %f, %f. pq_dist=%f, %f\n", decode_dist_1_2, decode_dist_2_3, pq_dist_1_2, pq_dist_2_3);
      atomicAdd((unsigned long long int *)misses, 1ULL);
    }
  }
}

// Compare the distance lookup table with the decoded distance
// these two should be exactly the same.
template <typename vector_type, typename quantized_type,
          template <typename, typename, uint, uint> class full_distance_functor,
          template <typename, typename, uint, uint> class pq_distance_functor>
__host__ uint64_t lookup_error(quantized_type *quantized_vectors,
                               vector_type *decoded_vectors, uint n_vectors,
                               uint64_t n_sims) {
  uint64_t *misses;
  cudaMallocManaged((void **)&misses, sizeof(uint64_t));
  misses[0] = 0;
  cudaDeviceSynchronize();

  const uint32_t tile_size = 4;
  uint32_t *first_vec = random_data_device<uint32_t>(n_sims);
  uint32_t *second_vec = random_data_device<uint32_t>(n_sims);
  uint32_t *third_vec = random_data_device<uint32_t>(n_sims);

  uint64_t num_threads_per_block = 256;
  uint64_t num_block = (tile_size * n_sims - 1) / num_threads_per_block + 1;

  lookup_error_kernel<vector_type, quantized_type, tile_size,
                      full_distance_functor, pq_distance_functor>
      <<<num_block, num_threads_per_block>>>(quantized_vectors, decoded_vectors,
                                             misses, n_vectors, n_sims,
                                             first_vec, second_vec, third_vec);

  cudaDeviceSynchronize();

  uint64_t n_misses = misses[0];
  printf("%lu out of %lu distance missed (%f percent).\n", n_misses, n_sims,
         100.0 * n_misses / n_sims);

  cudaFree(misses);
  cudaFree(first_vec);
  cudaFree(second_vec);
  cudaFree(third_vec);
  return n_misses;
}

}  // namespace gpu_ann

#endif  // GPU_BLOCK_