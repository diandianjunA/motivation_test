/*
 * ============================================================================
 *
 *        Authors:
 *                  Hunter McCoy <hjmccoy@lbl.gov
 *
 * ============================================================================
 */

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

  gpu_assert(tid != (n_threads - 1), "Thread id ", tid, " == ", n_threads - 1,
             "\n");
  // log_error("Logging from thread ", tid, "\n");

  if (tid == 0) gpu_error::log("Logging from thread ", tid, "\n");
}

int main(int argc, char** argv) {
  uint64_t num_threads;

  if (argc < 2) {
    printf(
        "Test has each thread write one log. Assert all logs are present.\n");
    printf("Usage: ./tests/global_churn [num_threads]\n");
    return 0;
  }

  num_threads = std::stoull(argv[1]);

  gpu_error::init_gpu_log();

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

  gpu_error::free_gpu_log();

  printf("%lu logs written\n", log_vector.size());

  std::cout << log_vector[0] << std::endl;
  std::cout << log_vector[log_vector.size() - 1] << std::endl;

  cudaDeviceReset();
  return 0;
}
