#ifndef GPU_GRAPH
#define GPU_GRAPH

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cooperative_groups/scan.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <fstream>
#include <gallatin/allocators/alloc_utils.cuh>
#include <gallatin/data_structs/ds_utils.cuh>
#include <gpu_ann/_proj_metadata.cuh>
#include <gpu_ann/edge_list.cuh>
#include <gpu_ann/edge_sorting.cuh>
#include <gpu_ann/hash_tables/linear_table.cuh>
#include <gpu_ann/priority_queue.cuh>
#include <gpu_ann/vector.cuh>
#include <gpu_error/progress_bar.cuh>
#include <iostream>
#include <vector>

#include "assert.h"
#include "stdio.h"

namespace cg = cooperative_groups;

namespace gpu_ann {

// helper struct that encapsulates the concept of the graph parameters
//  these include data necessary for the beam search.
struct graph_parameters {
  // # of threads involved in the beam.
  uint32_t threads_per_tile;

  uint32_t beam_width;

  uint32_t visited_cap;
};

__global__ void generate_edge_stats_kernel(uint8_t *edge_counts,
                                           uint64_t n_vertices,
                                           uint64_t *statistics) {
  uint64_t tid = gallatin::utils::get_tid();

  if (tid >= n_vertices) return;

  atomicMin((unsigned long long int *)&statistics[0],
            (unsigned long long int)edge_counts[tid]);
  atomicAdd((unsigned long long int *)&statistics[1],
            (unsigned long long int)edge_counts[tid]);
  atomicMax((unsigned long long int *)&statistics[2],
            (unsigned long long int)edge_counts[tid]);
}

template <uint32_t tile_size, uint32_t n_tiles, typename vertex_data_type,
          vertex_data_type dead_key, typename distance_type,
          typename vector_data_type, uint32_t vector_degree,
          template <typename, typename, uint, uint> class distance_functor,
          typename edge_output_type, uint32_t R, uint32_t V_cap>
__global__ void robust_prune_kernel(
    edge_list<vertex_data_type, R> *graph, uint8_t *edge_counts,
    data_vector<vector_data_type, vector_degree> *vertices,
    edge_output_type *accumulated_edges, double alpha, uint64_t n_edges,
    gpu_ann::priority_queue<vertex_data_type, distance_type, R + V_cap>
        *queues) {
  auto thread_block = cg::this_thread_block();

  cg::thread_block_tile<tile_size> my_tile =
      cg::tiled_partition<tile_size>(thread_block);

  uint64_t tid = gallatin::utils::get_tile_tid(my_tile);

  uint shared_index = my_tile.meta_group_rank();

  if (tid >= n_edges) return;

  bool first = (tid == 0) || (accumulated_edges[tid - 1].source !=
                              accumulated_edges[tid].source);

  if (!first) return;

  if (accumulated_edges[tid].source == dead_key) return;

  uint32_t source = accumulated_edges[tid].source;

  // use the prio queue for this.

  // load all edges into queue, then start dumping them out!

  using smem_edge_list_type =
      gpu_ann::smem_edge_list<vertex_data_type, R, distance_type>;

  __shared__ smem_edge_list_type local_edges[n_tiles];

  if (my_tile.thread_rank() == 0) {
    local_edges[shared_index].loaded_edge_list = graph[source];
  }

  uint8_t edge_count = edge_counts[source];

  queues[tid].init();

  my_tile.sync();

  for (uint i = 0; i < edge_count; i++) {
    vertex_data_type current_id =
        local_edges[shared_index].loaded_edge_list.edges[i];

#if COUNT_DIST
    if (my_tile.thread_rank() == 0) {
      count_event(6);
    }
#endif

    local_edges[shared_index].distances[i] =
        distance_functor<vector_data_type, vector_data_type, vector_degree,
                         tile_size>::distance(vertices[source],
                                              vertices[current_id], my_tile);
  }

  __threadfence();

  my_tile.sync();

  // next up - sort

  local_edges[shared_index].sort(my_tile, edge_count);

  __threadfence();

  my_tile.sync();

  queues[tid].template push<tile_size, R>(my_tile, local_edges[shared_index],
                                          edge_count);

  __threadfence();

  my_tile.sync();

  uint write_idx = 0;

  uint64_t read_idx = tid;
  while (read_idx < n_edges && accumulated_edges[read_idx].source == source) {
    vertex_data_type current_node = accumulated_edges[read_idx].sink;
    local_edges[shared_index].loaded_edge_list.edges[write_idx] =
        accumulated_edges[read_idx].sink;

#if COUNT_DIST
    if (my_tile.thread_rank() == 0) {
      count_event(7);
    }
#endif
    local_edges[shared_index].distances[write_idx] =
        distance_functor<vector_data_type, vector_data_type, vector_degree,
                         tile_size>::distance(vertices[source],
                                              vertices[current_node], my_tile);

    write_idx += 1;
    read_idx += 1;

    if (write_idx == R) {
      // push updates
      local_edges[shared_index].sort(my_tile, write_idx);

      __threadfence();

      my_tile.sync();

      queues[tid].template push<tile_size, R>(
          my_tile, local_edges[shared_index], write_idx);

      __threadfence();

      my_tile.sync();

      write_idx = 0;
    }

    my_tile.sync();
  }

  local_edges[shared_index].sort(my_tile, write_idx);

  __threadfence();

  my_tile.sync();

  queues[tid].template push<tile_size, R>(my_tile, local_edges[shared_index],
                                          write_idx);

  __threadfence();

  my_tile.sync();

  // pruning step.

  write_idx = 0;

  while (queues[tid].size() != 0 && write_idx < R) {
    auto pair = queues[tid].pop_both(my_tile);

    vertex_data_type head = pair.key;
    auto dist = pair.priority;

    bool failed = false;
    for (uint i = 0; i < write_idx; i++) {
      vertex_data_type comparison_vertex =
          local_edges[shared_index].loaded_edge_list.edges[i];

#if COUNT_DIST
      if (my_tile.thread_rank() == 0) {
        count_event(8);
      }
#endif

      auto comparison_dist =
          distance_functor<vector_data_type, vector_data_type, vector_degree,
                           tile_size>::distance(vertices[head],
                                                vertices[comparison_vertex],
                                                my_tile);

      if (comparison_dist * alpha <= dist) {
        failed = true;
        break;
      }
    }

    if (!failed) {
      local_edges[shared_index].loaded_edge_list.edges[write_idx] = head;
      local_edges[shared_index].distances[write_idx] = dist;

      write_idx++;
    }
  }

  // final write.

  __threadfence();
  my_tile.sync();

  if (my_tile.thread_rank() == 0) {
    graph[source] = local_edges[shared_index].loaded_edge_list;
    edge_counts[source] = write_idx;

    // if (write_idx > 1){
    //    printf("Node %u New # edges: %d\n", source, write_idx);
    // }
  }
}

template <typename vertex_data_type, vertex_data_type dead_key,
          typename edge_output_type>
__global__ void count_valid_edges(edge_output_type *edges, uint64_t n_edges,
                                  uint64_t *count) {
  uint64_t tid = gallatin::utils::get_tid();

  if (tid >= n_edges) return;

  if (edges[tid].source != dead_key) {
    atomicAdd((unsigned long long int *)count, 1ULL);

    printf("Valid edge %lu, %u -> %u\n", tid, edges[tid].source,
           edges[tid].sink);
  }
}

// helper kernel to set the neighbor list of new nodes.
// this is necessary
template <uint32_t tile_size, uint32_t n_tiles, typename vertex_data_type,
          vertex_data_type dead_key, typename distance_type,
          typename vector_data_type, uint32_t vector_degree,
          template <typename, typename, uint, uint> class distance_functor,
          typename edge_output_type, uint32_t R, uint32_t V_cap>
__global__ void attach_neighbors_kernel(
    edge_list<vertex_data_type, R> *graph, uint8_t *edge_counts,
    data_vector<vector_data_type, vector_degree> *vertices,
    edge_output_type *accumulated_edges, double alpha, uint64_t n_edges,
    gpu_ann::priority_queue<vertex_data_type, distance_type, R + V_cap>
        *queues) {
  auto thread_block = cg::this_thread_block();

  cg::thread_block_tile<tile_size> my_tile =
      cg::tiled_partition<tile_size>(thread_block);

  uint64_t tid = gallatin::utils::get_tile_tid(my_tile);

  uint shared_index = my_tile.meta_group_rank();

  if (tid >= n_edges) return;

  // make this V_cap.
  bool first = (tid == 0) ||
               (accumulated_edges[tid - 1].sink != accumulated_edges[tid].sink);

  if (!first) return;

  uint32_t source = accumulated_edges[tid].sink;

  // use the prio queue for this.

  // load all edges into queue, then start dumping them out!

  using smem_edge_list_type =
      gpu_ann::smem_edge_list<vertex_data_type, R, distance_type>;

  __shared__ smem_edge_list_type local_edges[n_tiles];

  if (my_tile.thread_rank() == 0) {
    local_edges[shared_index].loaded_edge_list = graph[source];
  }

  uint8_t edge_count = edge_counts[source];

  queues[tid].init();

  __threadfence();
  my_tile.sync();

  for (uint i = 0; i < edge_count; i++) {
    vertex_data_type current_id =
        local_edges[shared_index].loaded_edge_list.edges[i];

#if COUNT_DIST
    if (my_tile.thread_rank() == 0) {
      count_event(3);
    }
#endif
    local_edges[shared_index].distances[i] =
        distance_functor<vector_data_type, vector_data_type, vector_degree,
                         tile_size>::distance(vertices[source],
                                              vertices[current_id], my_tile);
  }

  __threadfence();

  my_tile.sync();

  // next up - sort

  local_edges[shared_index].sort(my_tile, edge_count);

  __threadfence();

  my_tile.sync();

  queues[tid].template push<tile_size, R>(my_tile, local_edges[shared_index],
                                          edge_count);

  __threadfence();

  my_tile.sync();

  uint write_idx = 0;

  uint64_t read_idx = tid;
  while (read_idx < n_edges && accumulated_edges[read_idx].sink == source) {
    vertex_data_type current_node = accumulated_edges[read_idx].source;

    if (current_node == dead_key) {
      read_idx += 1;
      continue;
    }
    local_edges[shared_index].loaded_edge_list.edges[write_idx] = current_node;

#if COUNT_DIST
    if (my_tile.thread_rank() == 0) {
      count_event(4);
    }
#endif
    local_edges[shared_index].distances[write_idx] =
        distance_functor<vector_data_type, vector_data_type, vector_degree,
                         tile_size>::distance(vertices[source],
                                              vertices[current_node], my_tile);

    write_idx += 1;
    read_idx += 1;

    if (write_idx == R) {
      // push updates
      local_edges[shared_index].sort(my_tile, write_idx);

      __threadfence();

      my_tile.sync();

      queues[tid].template push<tile_size, R>(
          my_tile, local_edges[shared_index], write_idx);

      __threadfence();

      my_tile.sync();

      write_idx = 0;
    }

    my_tile.sync();
  }

  local_edges[shared_index].sort(my_tile, write_idx);

  __threadfence();

  my_tile.sync();

  queues[tid].template push<tile_size, R>(my_tile, local_edges[shared_index],
                                          write_idx);

  __threadfence();

  my_tile.sync();

  // pruning step.

  // priority queue full of L_cap objects to connect to.

  write_idx = 0;

  while (queues[tid].size() != 0 && write_idx < R) {
    auto pair = queues[tid].pop_both(my_tile);

    vertex_data_type head = pair.key;
    auto dist = pair.priority;

    bool failed = false;
    for (uint i = 0; i < write_idx; i++) {
      vertex_data_type comparison_vertex =
          local_edges[shared_index].loaded_edge_list.edges[i];

#if COUNT_DIST
      if (my_tile.thread_rank() == 0) {
        count_event(5);
      }
#endif
      // dist between existing p* and new p`
      auto comparison_dist =
          distance_functor<vector_data_type, vector_data_type, vector_degree,
                           tile_size>::distance(vertices[head],
                                                vertices[comparison_vertex],
                                                my_tile);

      // dist[i] = dist(p,p*);
      // pair.priority = dist(p, p`);
      if (comparison_dist * alpha <= dist) {
        failed = true;
        break;
      }
    }

    if (!failed) {
      local_edges[shared_index].loaded_edge_list.edges[write_idx] = head;
      local_edges[shared_index].distances[write_idx] = dist;

      write_idx++;
    }
  }

  // final write.

  __threadfence();
  my_tile.sync();

  if (my_tile.thread_rank() == 0) {
    graph[source] = local_edges[shared_index].loaded_edge_list;
    edge_counts[source] = write_idx;

    // if (write_idx > 1){
    //    printf("Self attach: Node %u New # edges: %d\n", source, write_idx);
    // }
  }
}

// kernel to perform the beam search
// iteratively pop best node from queue, load neighbor list, and add to hash
// table
template <uint32_t tile_size, uint32_t n_tiles, typename vertex_data_type,
          vertex_data_type dead_key, typename distance_type,
          typename vector_data_type, uint32_t vector_degree,
          template <typename, typename, uint, uint> class distance_functor,
          typename edge_output_type, uint32_t R, uint32_t L_cap, uint32_t V_cap>
__global__ void beam_search_kernel(
    vertex_data_type medoid, edge_list<vertex_data_type, R> *graph,
    uint8_t *edge_counts,
    data_vector<vector_data_type, vector_degree> *vertices,
    uint64_t current_n_vertices, uint64_t n_to_processes,
    edge_output_type *outputs,
    gpu_ann::linear_table<vertex_data_type, dead_key, V_cap, tile_size>
        *hash_tables) {
  auto thread_block = cg::this_thread_block();

  cg::thread_block_tile<tile_size> my_tile =
      cg::tiled_partition<tile_size>(thread_block);

  uint64_t tid = gallatin::utils::get_tile_tid(my_tile);

  if (tid >= n_to_processes) return;

  using smem_edge_list_type =
      gpu_ann::smem_edge_list<vertex_data_type, R, distance_type>;

  __shared__ smem_edge_list_type local_edges[n_tiles];

  using priority_queue_type =
      gpu_ann::priority_queue<vertex_data_type, distance_type, L_cap>;

  __shared__ priority_queue_type local_queues[n_tiles];

  using ht_type =
      gpu_ann::linear_table<vertex_data_type, dead_key, V_cap, tile_size>;

  uint shared_index = my_tile.meta_group_rank();

  local_queues[shared_index].init_id(medoid);

  hash_tables[tid].init(my_tile);

  // wait here until my memory operations finish
  //  AND do not move memory operations over this line.
  __threadfence();
  my_tile.sync();

  gpu_assert(local_queues[shared_index].size() == 1,
             "Bad size, local queue contains ",
             local_queues[shared_index].size(), "\n");

  // main loop - iterate till queue drained.
  while (local_queues[shared_index].size() != 0) {
    // if (my_tile.thread_rank() == 0){
    //    printf("Queue size: %u\n", local_queues[shared_index].size());
    // }

    // first round size is exactly 1.
    // gpu_assert(local_queues[shared_index].size() == 1, "Bad size\n");

    __threadfence();
    my_tile.sync();

    // get closest item.
    vertex_data_type head = local_queues[shared_index].pop(my_tile);

    // __threadfence();
    // my_tile.sync();

    // if (my_tile.thread_rank() == 0){
    //    printf("Head is %u, new_size %u\n", head,
    //    local_queues[shared_index].size());
    // }

    __threadfence();

    my_tile.sync();

    // if item seen, return.
    if (hash_tables[tid].query(my_tile, head)) {
      __threadfence();
      my_tile.sync();
      continue;
    }

    // if the hash table is too full, continue on.
    if (!hash_tables[tid].insert(my_tile, head)) {
      // too full! drain?
      __threadfence();
      my_tile.sync();

#if COUNT_HT_FULL
      if (my_tile.thread_rank() == 0) {
        count_event(1);
      }
#endif

      break;
    }

    gpu_assert(hash_tables[tid].query(my_tile, head), "Key ", head,
               " is not queryable\n");

    // else, load!
    if (my_tile.thread_rank() == 0) {
      local_edges[shared_index].loaded_edge_list = graph[head];
    }

    uint8_t edge_count = edge_counts[head];

    __threadfence();

    my_tile.sync();

    // process distances.
    for (uint i = 0; i < edge_count; i++) {
      vertex_data_type current_id =
          local_edges[shared_index].loaded_edge_list.edges[i];

      if (hash_tables[tid].query(my_tile, current_id)) {
        // seen edges can be disregarded out of hand.
        local_edges[shared_index].distances[i] =
            cuda::std::numeric_limits<distance_type>::max();
      } else {
#if COUNT_DIST
        if (my_tile.thread_rank() == 0) {
          count_event(2);
        }
#endif
        local_edges[shared_index].distances[i] =
            distance_functor<vector_data_type, vector_data_type, vector_degree,
                             tile_size>::distance(vertices[current_id],
                                                  vertices[tid +
                                                           current_n_vertices],
                                                  my_tile);
      }
    }

    __threadfence();
    my_tile.sync();

    local_edges[shared_index].sort(my_tile, edge_count);

    __threadfence();

    my_tile.sync();

    // merge into hash table

    uint32_t size_before = local_queues[shared_index].size();

    local_queues[shared_index].template push<tile_size, R>(
        my_tile, local_edges[shared_index], edge_count);

    // assert that sizes match.
    gpu_assert(
        (local_queues[shared_index].size() == size_before + edge_count) ||
            local_queues[shared_index].size() == L_cap,
        "Bad size comparison after merge\n");
  }

  // process visited
  // write out hash table exactly.
  uint64_t my_start = tid * V_cap;

  // n_objs*V_cap;
  //  writing in range [tid*V_Cap. (tid+1)*V_Cap);

  for (uint i = my_tile.thread_rank(); i < V_cap; i += my_tile.size()) {
    outputs[my_start + i] = {(uint32_t)(tid + current_n_vertices),
                             hash_tables[tid].access(i), ~0};
  }
}

// gpuANN is the base version of the diskANN construction algorithm
template <uint tile_size, uint batch_size, typename vertex_data_type,
          typename distance_type, typename vector_data_type, uint vector_degree,
          uint32_t R, uint32_t L_cap, uint32_t V_cap,
          template <typename, typename, uint, uint> class distance_functor,
          bool on_host = true>
struct gpuANN {
  using edge_list_type = edge_list<vertex_data_type, R>;

  using pq_vector_type = data_vector<vector_data_type, vector_degree>;

  // each edge is of type data_type
  //  as it is an unweighted directed edge to one vertex
  // every vertex stores only its outgoing neighbor list.

  // allocator functor must implement "malloc"
  //  and "free" as static calls, and must be initialized before program start.

  vertex_data_type n_vertices;
  vertex_data_type n_vertices_max;
  vertex_data_type current_batch_size;
  vertex_data_type max_batch_size;
  vertex_data_type medoid;

  uint8_t *edge_counts;
  edge_list_type *edges;

  pq_vector_type *vectors;

  vertex_data_type *working_edges;

  // intialize gpuANN with # of vertices
  gpuANN(vertex_data_type n_vectors, double max_batch_ratio = .02) {
    n_vertices_max = n_vectors;

    // params = ext_params;

    n_vertices = 0;
    current_batch_size = 1;
    max_batch_size = n_vectors * max_batch_ratio;

    if (on_host) {
      edges =
          gallatin::utils::get_host_version<edge_list_type>(n_vertices_max + 1);
    } else {
      edges = gallatin::utils::get_device_version<edge_list_type>(
          n_vertices_max + 1);
    }

    edge_counts =
        gallatin::utils::get_device_version<uint8_t>(n_vertices_max + 1);

    // cudaMemset(edges, 0, sizeof(edge_list_type)*n_vertices_max);

    vectors =
        gallatin::utils::get_device_version<pq_vector_type>(n_vertices_max + 1);

    cudaMemset(edges, 0, sizeof(edge_list_type) * (n_vertices_max + 1));
    cudaMemset(edge_counts, 0, sizeof(uint8_t) * (n_vertices_max + 1));
    cudaMemset(vectors, 0, sizeof(pq_vector_type) * (n_vertices_max + 1));
  }

  ~gpuANN() {
    if (on_host) {
      cudaFreeHost(edges);
    } else {
      cudaFree(edges);
    }

    cudaFree(edge_counts);
    cudaFree(vectors);
  }

  // execute beam search for a large # of nodes in parallel.
  __host__ void run_beam_search(vertex_data_type &n_vectors_in_batch) {
    // printf("Executing beam search on %u vectors\n", n_vectors_in_batch);

    using edge_pair_type = edge_pair<vertex_data_type, distance_type>;

    // generate output edge list for sorting
    edge_pair_type *output_edges =
        gallatin::utils::get_device_version<edge_pair_type>(n_vectors_in_batch *
                                                            V_cap);

    // uint64_t * edge_count;

    // cudaMallocManaged ((void**)&edge_count, sizeof(uint64_t));

    // edge_count[0] = 0;

    using ht_type =
        gpu_ann::linear_table<vertex_data_type, (vertex_data_type)~0ULL, V_cap,
                              tile_size>;

    ht_type *tables =
        gallatin::utils::get_device_version<ht_type>(n_vectors_in_batch);

    using priority_queue_type =
        gpu_ann::priority_queue<vertex_data_type, distance_type, R + V_cap>;

    priority_queue_type *queues =
        gallatin::utils::get_device_version<priority_queue_type>(
            n_vectors_in_batch * V_cap);

    cudaDeviceSynchronize();

    constexpr uint n_tiles = batch_size / tile_size;

    // get ready, this is a big one
    //  launching the main kernel.
    beam_search_kernel<tile_size, n_tiles, vertex_data_type,
                       (vertex_data_type)~0ULL, distance_type, vector_data_type,
                       vector_degree, distance_functor, edge_pair_type, R,
                       L_cap,
                       V_cap>  // end of template args
        <<<(tile_size * n_vectors_in_batch - 1) / batch_size + 1,
           batch_size>>>  // kernel instantiation
        (medoid, edges, edge_counts, vectors, n_vertices, n_vectors_in_batch,
         output_edges, tables);  // actual args.

    cudaDeviceSynchronize();

    cudaFree(tables);

    uint64_t n_accumulated_edges = n_vectors_in_batch * V_cap;

    // count_valid_edges<vertex_data_type, (vertex_data_type) ~0ULL,
    // edge_pair_type><<<(n_accumulated_edges-1)/batch_size+1,batch_size>>>(output_edges,
    // n_accumulated_edges, edge_count);

    // cudaDeviceSynchronize();

    // printf("%lu valid edges in the round\n", edge_count[0]);

    // cudaFree(edge_count);

    // cudaDeviceSynchronize();

    // printf("Connecting neighbors\n");

    attach_neighbors_kernel<tile_size, n_tiles, vertex_data_type,
                            (vertex_data_type)~0ULL, distance_type,
                            vector_data_type, vector_degree, distance_functor,
                            edge_pair_type, R, V_cap>
        <<<(tile_size * n_accumulated_edges - 1) / batch_size + 1,
           batch_size>>>(edges, edge_counts, vectors, output_edges, 1.2,
                         n_accumulated_edges, queues);

    // xxwcudaDeviceSynchronize();

    // printf("Starting Sort\n");

    // whew! safely past the beam search. Now sort + update graph.
    output_edges =
        gpu_ann::semisort_edge_pairs_thrust(output_edges, n_accumulated_edges);

    // cudaDeviceSynchronize();

    // printf("Starting prune\n");

    robust_prune_kernel<tile_size, n_tiles, vertex_data_type,
                        (vertex_data_type)~0ULL, distance_type,
                        vector_data_type, vector_degree, distance_functor,
                        edge_pair_type, R, V_cap>
        <<<(tile_size * n_accumulated_edges - 1) / batch_size + 1,
           batch_size>>>(edges, edge_counts, vectors, output_edges, 1.2,
                         n_accumulated_edges, queues);

    // and finally prune.
    // robust_prune_kernel<tile_size,
    // <<<(tile_size*n_vectors_in_batch*V_cap-1)/batch_size+1,batch_size>>>()

    cudaDeviceSynchronize();

    cudaFree(output_edges);

    n_vertices += n_vectors_in_batch;

    return;
  }

  // process a batch - performs setup of batch size before moving to kernel
  // launch.
  __host__ void process_batch(vertex_data_type &n_vectors) {
    vertex_data_type n_vectors_to_process = 0;
    if (current_batch_size > n_vectors) {
      // process all vectors in batch.
      n_vectors_to_process = n_vectors;

      n_vectors = 0;

    } else {
      n_vectors_to_process = current_batch_size;

      current_batch_size *= 2;

      if (current_batch_size > max_batch_size)
        current_batch_size = max_batch_size;

      n_vectors -= n_vectors_to_process;
    }

    // current head is n_vertices.
    // incrementing from n_vertices to n_vertices_max.
    run_beam_search(n_vectors_to_process);
  }

  __host__ void set_medoid(vertex_data_type ext_medoid) { medoid = ext_medoid; }

  // add n_additions vectors to the graph.
  __host__ void construct(pq_vector_type *new_vectors,
                          vertex_data_type n_additions) {
    // first time initialization - pass the centroid.
    //  if (current_batch_size == 1){

    //    if (n_additions != 1){
    //       std::cerr << "First node supplied must be the centroid of the data
    //       set.\n" << std::endl; return;
    //    }

    //    //set centroid as is
    //    n_vertices = 1;
    //    current_batch_size = 2;
    //    cudaMemcpy(vectors, new_vectors, sizeof(pq_vector_type),
    //    cudaMemcpyDefault);

    //    return;

    // }

    if (n_vertices + n_additions <= medoid) {
      std::cerr << "Medoid must be in range\n" << std::endl;
    }

    // first - copy vectors into memory, and start processing.
    cudaMemcpy(vectors + n_vertices, new_vectors,
               sizeof(pq_vector_type) * n_additions, cudaMemcpyDefault);

    vertex_data_type n_vectors_left = n_additions;

    gpu_error::progress_bar bar("Building Index", n_additions, .01);

    while (n_vectors_left > current_batch_size) {
      uint64_t n_written = current_batch_size;

      process_batch(n_vectors_left);

      bar.increment(n_written);
    }

    uint64_t n_written = n_vectors_left;

    process_batch(n_vectors_left);

    bar.increment(n_written);
  }

  __host__ void print_edge_statistics() {
    uint64_t *statistics;

    cudaMallocManaged((void **)&statistics, sizeof(uint64_t) * 5);

    statistics[0] = ~0ULL;
    for (uint i = 1; i < 5; i++) {
      statistics[i] = 0;
    }

    generate_edge_stats_kernel<<<(n_vertices - 1) / batch_size + 1,
                                 batch_size>>>(edge_counts, n_vertices,
                                               statistics);

    cudaDeviceSynchronize();

    printf("N_verts: %u, Min %lu Max %lu avg: %f\n", n_vertices, statistics[0],
           statistics[2], 1.0 * statistics[1] / n_vertices);

    cudaFree(statistics);
  }

  // persist the graph to memory.
  template <typename host_vector_type>
  __host__ void write_out(std::string output_fname,
                          host_vector_type *originals) {
    // precalculations:
    // 1. size of each vector + graph
    // 2. n_vectors per sector
    // 3. n_sectors
    // 4. total file size.

    std::string md_filename = output_fname + "_metadata.bin";
    std::string vector_filename = output_fname + ".bin";

    // helper buffer
    std::vector<char> zeros(4096, 0);

    // and move edge counts;

    uint8_t *host_edge_counts =
        gallatin::utils::get_host_version<uint8_t>(n_vertices);

    cudaMemcpy(host_edge_counts, edge_counts, sizeof(uint8_t) * n_vertices,
               cudaMemcpyDeviceToHost);

    uint64_t bytes_per_vector = sizeof(host_vector_type);

    uint64_t bytes_per_neighbor_list = sizeof(edge_list_type);

    uint64_t bytes_per_node = bytes_per_vector + bytes_per_neighbor_list + 4;

    uint64_t nodes_per_sector = (4096) / bytes_per_node;

    uint64_t n_sectors = (n_vertices - 1) / nodes_per_sector + 1;

    uint64_t total_file_size = 4096ULL * (n_sectors + 1);

    uint64_t big_n_vectors = n_vertices;

    uint32_t full_vector_degree = data_vector_traits<host_vector_type>::size;

    std::cout << "Writing out " << total_file_size << " bytes"
              << ", " << n_sectors << " sectors each containing "
              << nodes_per_sector << " vectors using " << bytes_per_node
              << " bytes." << std::endl;

    std::ofstream mdFile(md_filename);
    std::ofstream outputFile(vector_filename);

    if (!mdFile.is_open()) {
      std::cerr << "Failed to open " << md_filename << " for graph write\n";
    }

    if (!outputFile.is_open()) {
      std::cerr << "Failed to open " << vector_filename << " for graph write\n";
    }

    // BANG output
    // 1. medoid
    uint64_t medoid_as_uint64_t = medoid;
    mdFile.write(reinterpret_cast<const char *>(&medoid_as_uint64_t),
                 sizeof(uint64_t));

    // entry len - bytes
    mdFile.write(reinterpret_cast<const char *>(&bytes_per_node),
                 sizeof(uint64_t));

    // 3. datatype
    if (std::is_same<vector_data_type, int8_t>::value) {
      int32_t value = 0;

      mdFile.write(reinterpret_cast<const char *>(&value), sizeof(int32_t));

    } else if (std::is_same<vector_data_type, uint8_t>::value) {
      int32_t value = 1;

      mdFile.write(reinterpret_cast<const char *>(&value), sizeof(int32_t));

    } else if (std::is_same<vector_data_type, float>::value) {
      int32_t value = 2;

      mdFile.write(reinterpret_cast<const char *>(&value), sizeof(int32_t));

    } else {
      std::cerr << " BANG will not recognize type" << std::endl;
    }

    // 4. Dimension of vector
    mdFile.write(reinterpret_cast<const char *>(&full_vector_degree),
                 sizeof(uint32_t));

    // 5 . N_data_points
    uint32_t R_clone = R;

    mdFile.write(reinterpret_cast<const char *>(&R_clone), sizeof(uint32_t));

    mdFile.write(reinterpret_cast<const char *>(&n_vertices), sizeof(uint32_t));

    gpu_error::progress_bar bar("Writing Sectors", n_sectors, .01);

    uint64_t n_vectors_written = 0;

    for (uint64_t i = 0; i < n_sectors; i++) {
      for (uint i = 0; i < nodes_per_sector; i++) {
        if (n_vectors_written < n_vertices) {
          outputFile.write(
              reinterpret_cast<const char *>(&originals[n_vectors_written]),
              sizeof(host_vector_type));

          uint32_t n_neighbors = host_edge_counts[n_vectors_written];

          outputFile.write(reinterpret_cast<const char *>(&n_neighbors),
                           sizeof(uint32_t));

          std::sort(edges[n_vectors_written].edges,
                    edges[n_vectors_written].edges + n_neighbors);

          outputFile.write(
              reinterpret_cast<const char *>(&edges[n_vectors_written]),
              sizeof(edge_list_type));

          n_vectors_written++;

        } else {
          outputFile.write(zeros.data(), bytes_per_node);
        }
      }

      // pad with zeros.
      outputFile.write(zeros.data(), 4096 - bytes_per_node * nodes_per_sector);

      bar.increment();
    }
  }

  template <typename host_vector_type>
  __host__ void write_out_diskANN(std::string output_fname,
                                  host_vector_type *originals) {
    // precalculations:
    // 1. size of each vector + graph
    // 2. n_vectors per sector
    // 3. n_sectors
    // 4. total file size.

    std::string vector_filename = output_fname + ".bin";

    // helper buffer
    std::vector<char> zeros(4096, 0);

    // and move edge counts;

    uint8_t *host_edge_counts =
        gallatin::utils::get_host_version<uint8_t>(n_vertices);

    cudaMemcpy(host_edge_counts, edge_counts, sizeof(uint8_t) * n_vertices,
               cudaMemcpyDeviceToHost);

    uint64_t bytes_per_vector = sizeof(host_vector_type);

    uint64_t bytes_per_neighbor_list = sizeof(edge_list_type);

    uint64_t bytes_per_node = bytes_per_vector + bytes_per_neighbor_list + 4;

    uint64_t nodes_per_sector = (4096) / bytes_per_node;

    uint64_t n_sectors = (n_vertices - 1) / nodes_per_sector + 1;

    uint64_t total_file_size = 4096ULL * (n_sectors + 1);

    uint64_t big_n_vectors = n_vertices;

    uint32_t full_vector_degree = data_vector_traits<host_vector_type>::size;

    std::cout << "Writing out " << total_file_size << " bytes"
              << ", " << n_sectors << " sectors each containing "
              << nodes_per_sector << " vectors using " << bytes_per_node
              << " bytes." << std::endl;

    std::ofstream outputFile(vector_filename);

    if (!outputFile.is_open()) {
      std::cerr << "Failed to open " << vector_filename << " for graph write\n";
    }

    // DiskANN output

    // 1. total file size:
    outputFile.write(reinterpret_cast<const char *>(&total_file_size),
                     sizeof(uint64_t));

    // 2 n_nodes
    outputFile.write(reinterpret_cast<const char *>(&big_n_vectors),
                     sizeof(uint64_t));

    // 3. medoid
    uint64_t medoid_as_uint64_t = medoid;
    outputFile.write(reinterpret_cast<const char *>(&medoid_as_uint64_t),
                     sizeof(uint64_t));

    // //4. Dimension of vector
    // outputFile.write(reinterpret_cast<const char*>(&full_vector_degree),
    // sizeof(uint32_t));

    // 4. entry len in bytes.
    outputFile.write(reinterpret_cast<const char *>(&bytes_per_node),
                     sizeof(uint64_t));

    // 5. nodes per sector.
    outputFile.write(reinterpret_cast<const char *>(&nodes_per_sector),
                     sizeof(uint64_t));

    // //5 . N_data_points
    // uint32_t R_clone = R;

    // outputFile.write(reinterpret_cast<const char*>(&R_clone),
    // sizeof(uint32_t));

    outputFile.write(zeros.data(), 4096 - 40);

    gpu_error::progress_bar bar("Writing Sectors", n_sectors, .01);

    uint64_t n_vectors_written = 0;

    for (uint64_t i = 0; i < n_sectors; i++) {
      for (uint i = 0; i < nodes_per_sector; i++) {
        if (n_vectors_written < n_vertices) {
          outputFile.write(
              reinterpret_cast<const char *>(&originals[n_vectors_written]),
              sizeof(host_vector_type));

          uint32_t n_neighbors = host_edge_counts[n_vectors_written];

          outputFile.write(reinterpret_cast<const char *>(&n_neighbors),
                           sizeof(uint32_t));

          std::sort(edges[n_vectors_written].edges,
                    edges[n_vectors_written].edges + n_neighbors);

          outputFile.write(
              reinterpret_cast<const char *>(&edges[n_vectors_written]),
              sizeof(edge_list_type));

          n_vectors_written++;

        } else {
          outputFile.write(zeros.data(), bytes_per_node);
        }
      }

      // pad with zeros.
      outputFile.write(zeros.data(), 4096 - bytes_per_node * nodes_per_sector);

      bar.increment();
    }
  }
};

}  // namespace gpu_ann

#endif  // GPU_BLOCK_