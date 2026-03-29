#pragma once

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <fstream>
#include <iostream>
#include <filesystem>

#include <gpu_ann/edge_list.cuh>
#include <gpu_ann/edge_sorting.cuh>

#include "assert.h"
#include "stdio.h"

namespace gpu_ann {

__host__ thrust::device_vector<edge_pair<uint32_t, float>>
get_groundtruth_from_file(std::string filename, uint32_t target_k) {
  namespace fs = std::filesystem;
  using host_vector_type = thrust::host_vector<edge_pair<uint32_t, float>>;

  std::ifstream gtFile(filename, std::ios::binary);
  if (!gtFile.is_open()) {
    std::cerr << "Failed to open groundtruth file: " << filename << std::endl;
    exit(EXIT_FAILURE);
  }

  // read header
  uint32_t num_queries;
  uint32_t gt_k;
  gtFile.read(reinterpret_cast<char *>(&num_queries), sizeof(uint32_t));
  gtFile.read(reinterpret_cast<char *>(&gt_k), sizeof(uint32_t));

  if (gt_k < target_k) {
    std::cerr << "Given k is bigger than the k provided by the ground truth file.\n";
    exit(EXIT_FAILURE);
  }

  std::vector<uint32_t> ids(num_queries * gt_k);
  gtFile.read(reinterpret_cast<char *>(ids.data()), ids.size() * sizeof(uint32_t));

  std::vector<float> distances(num_queries * gt_k);
  gtFile.read(reinterpret_cast<char *>(distances.data()), distances.size() * sizeof(float));

  if (!gtFile.good()) {
    std::cerr << "Failed to read full groundtruth data.\n";
    std::exit(EXIT_FAILURE);
  }

  host_vector_type host_gt(num_queries * target_k);
  for (uint32_t q = 0; q < num_queries; ++q) {
    for (uint32_t k = 0; k < target_k; ++k) {
      size_t idx = q * gt_k + k;
      host_gt[q * target_k + k] = {0, ids[idx], distances[idx]};
    }
  }

  thrust::device_vector<edge_pair<uint32_t, float>> device_gt = host_gt;
  return device_gt;
}


}

