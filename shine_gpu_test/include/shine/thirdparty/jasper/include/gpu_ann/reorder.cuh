#pragma once

#pragma once

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cub/cub.cuh>

#include <iostream>
#include <vector>
#include <unordered_set>
#include <queue>
#include <random>
#include <algorithm>
#include <gallatin/allocators/alloc_utils.cuh>
#include <gallatin/data_structs/ds_utils.cuh>
#include <gpu_ann/edge_list.cuh>
#include <gpu_ann/edge_sorting.cuh>
#include <gpu_ann/vector.cuh>
#include <gpu_ann/edge_list.cuh>
#include <gpu_ann/beam_search.cuh>

#include "assert.h"
#include "stdio.h"

namespace gpu_ann {

// Helper function to calculate the number of breaks
// in continous index for each neighbor list
template <typename INDEX_T, uint16_t R>
__host__ void calc_breaks_in_neighbor_list(
  edge_list<INDEX_T, R> *edges,
  uint8_t *edge_counts,
  uint64_t n_vectors
){
  std::map<INDEX_T, uint64_t> freqs;
  for (uint i=0; i<n_vectors; ++i) {
    uint32_t num_breaks = 0;
    edge_list<INDEX_T, R> e = edges[i];
    for (uint j=0; j<edge_counts[i]-1; j++) {
      if (e.edges[j] != e.edges[j+1]-1) {
        num_breaks++;
      }
    }
    freqs[num_breaks]++;
  }

  for (const auto& pair : freqs) {
    std::cout << "freq: " << pair.first << ", Count: " << pair.second << std::endl;
  }
}

template <typename DATA_T, 
          uint16_t DATA_DIM, 
          typename INDEX_T,
          uint16_t R>
struct reordered_result {
  INDEX_T centroid_index;

  // on device
  data_vector<DATA_T, DATA_DIM> *d_vectors;
  edge_list<INDEX_T, R> *edges;
  uint8_t *edge_counts;

  // on host
  std::vector<INDEX_T> mapping;
};

template <typename DATA_T, 
          uint16_t DATA_DIM, 
          typename INDEX_T,
          uint16_t R>
__host__ reordered_result<DATA_T, DATA_DIM, INDEX_T, R> reorder(
  uint64_t n_vectors,
  edge_list<INDEX_T, R> *edges,
  uint8_t *edge_counts,
  data_vector<DATA_T, DATA_DIM> *vectors,
  INDEX_T centroid_index,
  bool on_host,
  bool use_rcm=true
){
  // move all to host space
  std::cout << "Moving to host" << std::endl;
  edges = gallatin::utils::move_to_host<edge_list<INDEX_T, R>>(edges, n_vectors);
  edge_counts = gallatin::utils::move_to_host<uint8_t>(edge_counts, n_vectors);
  vectors = gallatin::utils::move_to_host<data_vector<DATA_T, DATA_DIM>>(vectors, n_vectors);

  // If we are using rcm algorithm
  // sort each edge list by degree before performing the reordering
  if (use_rcm) {
    for (uint i=0; i<n_vectors; i++) {
      std::sort(edges[i].edges, edges[i].edges+edge_counts[i], [&edge_counts](uint32_t a, uint32_t b) {
        return edge_counts[a] > edge_counts[b];
      });
    }
  }

  std::vector<INDEX_T> map_ordered_to_original;
  map_ordered_to_original.push_back(centroid_index);

  std::vector<INDEX_T> map_original_to_ordered(n_vectors, 0);
  map_original_to_ordered[centroid_index] = 0;

  std::unordered_set<INDEX_T> visited;
  visited.insert(centroid_index);
  std::queue<INDEX_T> frontier;
  frontier.push(centroid_index);

  // BFS
  while (frontier.size() > 0) {
    INDEX_T vec_idx = frontier.front();
    frontier.pop();
    edge_list<INDEX_T, R>  cur_edges = edges[vec_idx];
    uint8_t cur_edge_count = edge_counts[vec_idx];

    for (uint i=0; i<cur_edge_count; i++) {
      // if we haven't seen this element
      INDEX_T e = cur_edges.edges[i];
      if (!visited.count(e)) {
        visited.insert(e);
        frontier.push(e);

        map_ordered_to_original.push_back(e);
        map_original_to_ordered[e] = map_ordered_to_original.size()-1;
      }
    }
  }

  std::cout << "BFS complete, map_ordered_to_original.size()=" << map_ordered_to_original.size() 
    << " visited.size()=" << visited.size() 
    << std::endl;

  // create a new vector list using the mapping
  data_vector<DATA_T, DATA_DIM> *ordered_vectors = 
    gallatin::utils::get_host_version<data_vector<DATA_T, DATA_DIM>>(n_vectors);
  for (uint i=0; i<n_vectors; i++) {
    if (i < map_ordered_to_original.size()) {
      ordered_vectors[i] = vectors[map_ordered_to_original[i]];
    } else {
      ordered_vectors[i] = vectors[centroid_index]; 
    }
  }
  
  std::cout << "Created new vector list." << std::endl;

  // replace edge list index with new ordered index
  for (uint i=0; i<n_vectors; i++) {
    uint8_t count = edge_counts[i];
    for (uint j=0; j<count; j++) {
      INDEX_T n = edges[i].edges[j];
      edges[i].edges[j] = map_original_to_ordered[n];
    }
  }
  std::cout << "Remapped out neighbor list." << std::endl;

  edge_list<INDEX_T, R> *ordered_edges = gallatin::utils::get_host_version<edge_list<INDEX_T, R>>(n_vectors);
  uint8_t *ordered_edge_counts = gallatin::utils::get_host_version<uint8_t>(n_vectors);
  for (uint i=0; i<n_vectors; i++) {
    if (i < map_ordered_to_original.size()) {
      ordered_edge_counts[i] = edge_counts[map_ordered_to_original[i]];
      ordered_edges[i] = edges[map_ordered_to_original[i]];
    } else {
      ordered_edges[i] = edges[centroid_index]; 
      ordered_edge_counts[i] = edge_counts[centroid_index]; 
    }
  }
  
  if (!on_host) {
    // move all to device space
    std::cout << "Moving to device" << std::endl;
    ordered_edges = gallatin::utils::move_to_device<edge_list<INDEX_T, R>>(
      ordered_edges, n_vectors
    );
    ordered_edge_counts = gallatin::utils::move_to_device<uint8_t>(
      ordered_edge_counts, n_vectors
    );
    ordered_vectors = gallatin::utils::move_to_device<data_vector<DATA_T, DATA_DIM>>(
     ordered_vectors, n_vectors
    );
  }
  
  cudaFree(edge_counts);
  cudaFree(edges);
  
  reordered_result<DATA_T, DATA_DIM, INDEX_T, R> result = {
    0,
    ordered_vectors,
    ordered_edges, 
    ordered_edge_counts, 
    map_ordered_to_original};
  return result;
}

// 
template <typename DATA_T, 
          uint16_t DATA_DIM, 
          typename INDEX_T,
          uint16_t R>
__host__ reordered_result<DATA_T, DATA_DIM, INDEX_T, R> reorder_random(
  uint64_t n_vectors,
  edge_list<INDEX_T, R> *edges,
  uint8_t *edge_counts,
  data_vector<DATA_T, DATA_DIM> *vectors,
  INDEX_T centroid_index,
  bool on_host
){
  // move all to host space
  std::cout << "Moving to host" << std::endl;
  edges = gallatin::utils::move_to_host<edge_list<INDEX_T, R>>(edges, n_vectors);
  edge_counts = gallatin::utils::move_to_host<uint8_t>(edge_counts, n_vectors);
  vectors = gallatin::utils::move_to_host<data_vector<DATA_T, DATA_DIM>>(vectors, n_vectors);

  std::vector<INDEX_T> map_ordered_to_original(n_vectors, 0);
  std::vector<INDEX_T> map_original_to_ordered(n_vectors, 0);

  for (size_t i = 0; i < n_vectors; i++) {
    map_ordered_to_original[i] = static_cast<INDEX_T>(i);
  }

  // shuffle
  std::random_device rd;
  std::mt19937 g(rd());
  std::shuffle(map_ordered_to_original.begin(), map_ordered_to_original.end(), g);

  // inverse
  for (size_t ordered = 0; ordered < n_vectors; ordered++) {
    INDEX_T original = map_ordered_to_original[ordered];
    map_original_to_ordered[original] = static_cast<INDEX_T>(ordered);
  }
  INDEX_T new_centroid_index = map_original_to_ordered[centroid_index];

  // create a new vector list using the mapping
  data_vector<DATA_T, DATA_DIM> *ordered_vectors = 
    gallatin::utils::get_host_version<data_vector<DATA_T, DATA_DIM>>(n_vectors);
  for (uint i=0; i<n_vectors; i++) {
    ordered_vectors[i] = vectors[map_ordered_to_original[i]];
  }
  
  std::cout << "Created new vector list." << std::endl;

  // replace edge list index with new ordered index
  for (uint i=0; i<n_vectors; i++) {
    uint8_t count = edge_counts[i];
    for (uint j=0; j<count; j++) {
      INDEX_T n = edges[i].edges[j];
      edges[i].edges[j] = map_original_to_ordered[n];
    }
  }
  std::cout << "Remapped out neighbor list." << std::endl;

  edge_list<INDEX_T, R> *ordered_edges = gallatin::utils::get_host_version<edge_list<INDEX_T, R>>(n_vectors);
  uint8_t *ordered_edge_counts = gallatin::utils::get_host_version<uint8_t>(n_vectors);
  for (uint i=0; i<n_vectors; i++) {
    ordered_edge_counts[i] = edge_counts[map_ordered_to_original[i]];
    ordered_edges[i] = edges[map_ordered_to_original[i]];
  }
  
  if (!on_host) {
    // move all to device space
    std::cout << "Moving to device" << std::endl;
    ordered_edges = gallatin::utils::move_to_device<edge_list<INDEX_T, R>>(
      ordered_edges, n_vectors
    );
    ordered_edge_counts = gallatin::utils::move_to_device<uint8_t>(
      ordered_edge_counts, n_vectors
    );
    ordered_vectors = gallatin::utils::move_to_device<data_vector<DATA_T, DATA_DIM>>(
     ordered_vectors, n_vectors
    );
  }
  
  reordered_result<DATA_T, DATA_DIM, INDEX_T, R> result = {
    new_centroid_index,
    ordered_vectors,
    ordered_edges, 
    ordered_edge_counts, 
    map_ordered_to_original};
  return result;
}

}