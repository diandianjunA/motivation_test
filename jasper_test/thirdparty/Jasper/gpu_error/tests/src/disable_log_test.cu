/*
 * ============================================================================
 *
 *        Authors:
 *                  Hunter McCoy <hjmccoy@lbl.gov
 *
 * ============================================================================
 */

#define GPU_NDEBUG

#include <assert.h>
#include <stdio.h>

#include <chrono>
#include <gallatin/allocators/timer.cuh>
#include <gpu_error/log.cuh>
#include <iostream>

using namespace gallatin::allocators;

__global__ void write_to_log_kernel(uint64_t n_threads) {
  uint64_t tid = gallatin::utils::get_tid();

  if (tid >= n_threads) return;

  gpu_log("Logging from thread ", tid, "... but it won't happpen\n");

  gpu_assert(1 == 0, "This won't trigger\n");

  gpu_error("I am fatally crashing because I am a bad CUDA\n");
}

int main(int argc, char** argv) {
  uint64_t num_threads;

  // if (argc < 2){
  //    num_segments = 1000;
  // } else {
  //    num_segments = std::stoull(argv[1]);
  // }

  // if (argc < 3){
  //    num_threads = 1000000;
  // } else {
  //    num_threads = std::stoull(argv[2]);
  // }

  // if (argc < 4){
  //    num_rounds = 1;
  // } else {
  //    num_rounds = std::stoull(argv[3]);
  // }

  // if (argc < 5){
  //    min_size = 16;
  // } else {
  //    min_size = std::stoull(argv[4]);
  // }

  // if (argc < 6){
  //    max_size = 4096;
  // } else {
  //    max_size = std::stoull(argv[5]);
  // }

  if (argc < 2) {
    printf(
        "Test has each thread write one log. Assert all logs are present.\n");
    printf("Usage: ./tests/global_churn [num_threads]\n");
    return 0;
  }

  num_threads = std::stoull(argv[1]);

  // gpu_error::init_gpu_log();

  cudaDeviceSynchronize();

  gallatin::utils::timer log_timer;

  write_to_log_kernel<<<(num_threads - 1) / 512 + 1, 512>>>(num_threads);

  log_timer.sync_end();
  cudaDeviceSynchronize();

  log_timer.print_throughput("Logged", num_threads);

  gallatin::utils::timer export_timer;

  auto log_vector = gpu_error::export_log();

  export_timer.sync_end();

  export_timer.print_throughput("Exported", num_threads);

  // gpu_error::free_gpu_log();

  printf("%lu logs written\n", log_vector.size());

  cudaDeviceReset();
  return 0;
}
