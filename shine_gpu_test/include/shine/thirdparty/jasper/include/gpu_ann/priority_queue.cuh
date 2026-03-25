#ifndef PRIORITY_QUEUE
#define PRIORITY_QUEUE

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cooperative_groups/scan.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

#include <gpu_ann/cg_compat.cuh>

namespace cg = cooperative_groups;

#include <cuda/std/cmath>
#include <cuda/std/cstdint>
#include <fstream>
#include <gallatin/allocators/alloc_utils.cuh>
#include <gallatin/data_structs/ds_utils.cuh>
#include <gpu_ann/_proj_metadata.cuh>
#include <gpu_ann/edge_list.cuh>
#include <gpu_error/log.cuh>
#include <iostream>
#include <limits>
#include <vector>

#include "assert.h"
#include "stdio.h"

namespace cg = cooperative_groups;

// This file defines a lightweight priority queue for use in shared memory.
//  the queue is operated on by a singular tile and synchronized via tile syncs.
//  operations are optimized for list merges controlled by the tile.
//  This allows for fast merges of incoming neighbor lists during the greedy
//  search.
namespace gpu_ann {

template <typename key_type, typename comparable_type>
struct priority_queue_pair {
  comparable_type priority;
  key_type key;

  __device__ priority_queue_pair() { return; }

  __device__ priority_queue_pair(key_type ext_key,
                                 comparable_type new_priority) {
    priority = new_priority;
    key = ext_key;
  }

  // version 1
  __device__ bool operator<(const priority_queue_pair& rhs) {
    return this->priority < rhs.priority;
  }
};

struct merge_pair {
  int first;
  int second;

  __host__ __device__ bool operator==(const merge_pair& other) const {
    return first == other.first && second == other.second;
  }

  __host__ __device__ bool operator!=(const merge_pair& other) const {
    return first != other.first || second != other.second;
  }
};

// implementation of a fixed-width priority queue
//  if more items are enqueued than have space, the largest-K are dropped.
template <typename key_type, typename comparable_type, uint n_slots>
struct priority_queue {
  using pair_type = priority_queue_pair<key_type, comparable_type>;

  volatile uint32_t head;
  volatile int32_t n_keys;

  pair_type queue[n_slots];

  __device__ priority_queue() { return; }

  __device__ void init() {
    head = 0;
    n_keys = 0;
  }

  __device__ void init_id(key_type init_key) {
    head = 0;
    n_keys = 1;

    read_from_head(0) = pair_type(init_key, 0.0);
  }

  __device__ pair_type& read_from_head(int address) {
    return queue[(head + address) % n_slots];
  }

  __device__ uint32_t size() { return n_keys; }

  template <uint tile_size>
  __device__ key_type pop(cg::thread_block_tile<tile_size>& my_tile) {
    __threadfence();
    my_tile.sync();

    gpu_assert(n_keys > 0, "Bad number of keys: ", n_keys, "\n");

    key_type key = invoke_one_broadcast_compat(my_tile, [&]() {
      key_type key = queue[head].key;
      n_keys--;
      head = (head + 1) % n_slots;

      return key;
    });

    my_tile.sync();

    return key;
  }

  template <uint tile_size>
  __device__ pair_type pop_both(cg::thread_block_tile<tile_size>& my_tile) {
    __threadfence();
    my_tile.sync();

    gpu_assert(n_keys > 0, "Bad number of keys: ", n_keys, "\n");

    pair_type return_pair = invoke_one_broadcast_compat(my_tile, [&]() {
      pair_type return_pair = queue[head];
      n_keys--;
      head = (head + 1) % n_slots;

      return return_pair;
    });

    my_tile.sync();

    return return_pair;
  }

  template <uint32_t R>
  __device__ bool good_pivots(
      int l_pivot, int r_pivot,
      const smem_edge_list<key_type, R, comparable_type>& list_to_merge,
      uint merge_list_size) {
    if (l_pivot < 0) return false;
    if (r_pivot < 0) return false;

    if (l_pivot > n_keys) return false;
    if (r_pivot > merge_list_size) return false;

    float a_smallest = cuda::std::numeric_limits<comparable_type>::lowest();
    float b_smallest = cuda::std::numeric_limits<comparable_type>::lowest();

    if (l_pivot != 0) {
      a_smallest = read_from_head(l_pivot - 1).priority;
    }

    if (r_pivot != 0) {
      b_smallest = list_to_merge.distances[r_pivot - 1];
    }

    // assert any items in both lists are larger.

    if (l_pivot != n_keys) {
      float a_next = read_from_head(l_pivot).priority;

      if (b_smallest > a_next) return false;
    }

    if (r_pivot != merge_list_size) {
      float b_next = list_to_merge.distances[r_pivot];

      if (a_smallest > b_next) return false;
    }

    return true;
  }

  // variant of pivot selection that *should* work
  //  assumes the list is bitonic, and finds the first pivot.
  template <uint32_t R>
  __device__ bool pivot_dir(
      int l_pivot, int r_pivot,
      const smem_edge_list<key_type, R, comparable_type>& list_to_merge,
      uint merge_list_size) {
    if (l_pivot < 0) return false;
    if (r_pivot < 0) return false;

    float a_smallest = cuda::std::numeric_limits<comparable_type>::lowest();
    float b_smallest = cuda::std::numeric_limits<comparable_type>::lowest();

    if (l_pivot != 0) {
      a_smallest = read_from_head(l_pivot - 1).priority;
    }

    if (r_pivot != 0) {
      b_smallest = list_to_merge.distances[r_pivot - 1];
    }

    // assert any items in both lists are larger.
    //  k = 4
    //| 1 2 3

    // 1 2 3 | bad

    // 1 | 2 3
    // 1 2 | 3 good

    // 1 2 | 3
    // 1 | 2 3 good

    // 1 2 3 |
    // | 1 2 3 (bad due to a_smallest > b_next)

    if (l_pivot != n_keys) {
      float a_next = read_from_head(l_pivot).priority;

      if (b_smallest > a_next) return false;
    }

    // if (r_pivot != merge_list_size){

    //    float b_next = list_to_merge.distances[r_pivot];

    //    if (a_smallest > b_next) return false;
    // }

    return true;
  }

  template <uint tile_size, uint32_t R>
  __device__ void assert_good_pivots(
      cg::thread_block_tile<tile_size>& my_tile,
      const smem_edge_list<key_type, R, comparable_type>& list_to_merge,
      uint merge_list_size) {
    for (uint i = my_tile.thread_rank(); i < n_keys; i += tile_size) {
      for (uint j = 1; j < merge_list_size; j++) {
        // assert
        if (pivot_dir(i, j, list_to_merge, merge_list_size)) {
          if (!pivot_dir(i + 1, j - 1, list_to_merge, merge_list_size)) {
            // pivot_dir(i+1,j-1, list_to_merge, merge_list_size);
            gpu_error("Bad pivots", i, " and ", j,
                      " are good but followed by a bad pivot.\n");
          }
        }
      }
    }

    my_tile.sync();
  }

  // push
  template <uint tile_size, uint32_t R>
  __device__ void push(
      cg::thread_block_tile<tile_size>& my_tile,
      const smem_edge_list<key_type, R, comparable_type>& list_to_merge,
      uint merge_list_size) {
    // rough idea
    // each thread demarcates region of memory it owns
    // spaces out new keys.
    // then merges in
    // to reserve space we will use the stack.

    __threadfence();
    my_tile.sync();

    merge_pair start_pair;
    merge_pair end_pair;

    int thread_id = my_tile.thread_rank();
    int m = n_keys, n = merge_list_size;
    int total = m + n;

    int start_k = thread_id * total / tile_size;
    int end_k = (thread_id + 1) * total / tile_size;

    __threadfence();
    my_tile.sync();

    // Early exit for invalid thread_id
    if (thread_id >= tile_size || tile_size == 0) {
      start_pair = {0, 0};
      end_pair = {0, 0};
    } else {
      // Determine output indices

      // want to find the smallest idx_a s.t. left[a] < right[a + k];
      //  for now, this uses a linear scan - I was having trouble getting the
      //  binary search to work due to incorrectly assuming monotonicity. I have
      //  a fix as the set is actually bitonic.

      // to prove this works, make the lambda scan.

      // this version is busted but I have a fix
      //  Helper lambda: find split point (a_idx, b_idx) for merged index "k"
      // needs to chosse the largest point in a, s.t.
      //  auto split = [&](int k) -> merge_pair {

      //     int low = (k >= n) ? (k - n) : 0;
      //     int high = min(k, m);

      //     int low = (k >= n) ? (k - n) : 0;
      //     int high = min(k, m);

      //     printf("Thread %d has low/high %d -> %d\n", thread_id, low, high);
      //     gpu_assert(0 <= k && k <= m + n, "Invalid merge index k, ", k,
      //     "\n");

      //     //we want to find the first item A
      //     //where

      //     while (low != high) {
      //         int a_idx = (low + high) / 2;
      //         int b_idx = k - a_idx;

      //         comparable_type a_left  = (a_idx == 0) ?
      //         cuda::std::numeric_limits<comparable_type>::lowest() :
      //         read_from_head(a_idx - 1).priority; comparable_type a_right =
      //         (a_idx == m) ?
      //         cuda::std::numeric_limits<comparable_type>::max()    :
      //         read_from_head(a_idx).priority;

      //         comparable_type b_left  = (b_idx == 0) ?
      //         cuda::std::numeric_limits<comparable_type>::lowest() :
      //         list_to_merge.distances[b_idx - 1]; comparable_type b_right =
      //         (b_idx == n) ?
      //         cuda::std::numeric_limits<comparable_type>::max()    :
      //         list_to_merge.distances[b_idx];

      //         if (a_left > b_right) {
      //            high = a_idx; // move left
      //         } else if (b_left > a_right) {
      //            low = a_idx + 1; // move right
      //         } else {
      //            low = a_idx;
      //            break;
      //         }
      //     }

      //     int a_idx = low;
      //     int b_idx = k - a_idx;

      //     return {a_idx, b_idx};
      // };

      // this version is slow but works...
      auto linear_split = [&](int k) -> merge_pair {
        // base case - no other list.
        if (n_keys == 0) return {0, k};

        if (k == 0) return {0, 0};

        for (int a_idx = 0; a_idx <= k; a_idx++) {
          int b_idx = k - a_idx;

          // this calculation clips with n_keys == 2
          //  and merge_list_size ==2
          //  due to outputs being {0,3}, {1,2}, {2,1}, {3,0}
          // if they are exactly on the boundary we gotta do something.

          // refactor how we are thinking about these.
          //  want to draw a line in the sand A / B
          //  where left[A-1] and right[B-1] are smaller than A and B.
          // lists are 1 2
          //  1 4

          if (a_idx > n_keys) continue;
          if (b_idx > merge_list_size) continue;

          float a_smallest =
              cuda::std::numeric_limits<comparable_type>::lowest();
          float b_smallest =
              cuda::std::numeric_limits<comparable_type>::lowest();

          if (a_idx != 0) {
            a_smallest = read_from_head(a_idx - 1).priority;
          }

          if (b_idx != 0) {
            b_smallest = list_to_merge.distances[b_idx - 1];
          }

          // assert any items in both lists are larger.

          if (a_idx != n_keys) {
            float a_next = read_from_head(a_idx).priority;

            if (b_smallest > a_next) continue;
          }

          if (b_idx != merge_list_size) {
            float b_next = list_to_merge.distances[b_idx];

            if (a_smallest > b_next) continue;
          }

          return {a_idx, b_idx};
        }

        // return maximally large a?

        gpu_error("Could not find bound in array\n");
      };

      // busted version - needs to be bitonic.
      auto split = [&](int k) -> merge_pair {
        // base case - no other list.
        int low = max(0, k - (int)merge_list_size);
        int high = min(k, n_keys);
        // tighten bounds to exact.

        while (low < high) {
          int a_idx = low + (high - low) / 2;
          // int a_idx = (low + high) / 2;
          int b_idx = k - a_idx;

          // if (a_idx > n_keys){
          //    high = a_idx-1;
          //    continue;
          // }

          // if (b_idx > merge_list_size){
          //    low = a_idx+1;
          //    continue;
          // }

          // assert any items in both lists are larger.

          bool valid = pivot_dir(a_idx, b_idx, list_to_merge, merge_list_size);

          // if (valid && !good_pivots(a_idx-1, b_idx+1, list_to_merge,
          // merge_list_size)){
          //    return {a_idx,b_idx};
          // }

          // is good pivot not monotonic?

          // it definitely is and can be proven to be so.
          // we have sub lists l and r.
          //  and keys left and right.
          //  left >> all l
          //  right >= all r
          //  to be a good pivot l must be
          //  2 | 5 5
          //  6 | 7 8

          if (valid) {
            // valid means this is a solution - tighten by moving down.
            high = a_idx;
          } else {
            // not a solution - move up
            low = a_idx + 1;
          }
        }

        int a_idx = low;
        int b_idx = k - a_idx;

        if (!good_pivots(a_idx, b_idx, list_to_merge, merge_list_size)) {
          good_pivots(a_idx, b_idx, list_to_merge, merge_list_size);
        }

        gpu_assert(good_pivots(a_idx, b_idx, list_to_merge, merge_list_size),
                   "returned value is bad pivot.\n");
        gpu_assert(
            !good_pivots(a_idx - 1, b_idx + 1, list_to_merge, merge_list_size),
            "moving lower could find better?\n");
        // while (good_pivots(a_idx-1, b_idx+1, list_to_merge,
        // merge_list_size)){
        //    a_idx-=1;
        //    b_idx+=1;
        // }

        return {a_idx, b_idx};
      };

      gpu_assert(split(start_k) == linear_split(start_k),
                 "split doesn't match\n");
      gpu_assert(split(end_k) == linear_split(end_k),
                 "end split doens't match\n");

      start_pair = split(start_k);
      end_pair = split(end_k);

      // if(split(start_k) != start_pair){

      //   split(start_k);

      // }

      // if (split(end_k) != end_pair){
      //   split(end_k);
      // }

      gpu_assert(start_pair.first <= end_pair.first,
                 "Invalid merge range: left start is greater than left end (",
                 start_pair.first, " > ", end_pair.first, ")\n");

      gpu_assert(start_pair.second <= end_pair.second,
                 "Invalid merge range: right start is greater than right end (",
                 start_pair.second, " > ", end_pair.second, ")\n");

      gpu_assert(start_pair.first + start_pair.second == start_k,
                 "Start pair sum mismatch: ", start_pair.first, "+",
                 start_pair.second, " != ", start_k, "\n");

      __threadfence();
      my_tile.sync();
      gpu_assert(end_pair.first + end_pair.second == end_k,
                 "End pair sum mismatch: ", end_pair.first, "+",
                 end_pair.second, " != ", end_k, "\n");
    }

    __threadfence();
    my_tile.sync();

    constexpr uint n_slots_per_thread = (n_slots + R - 1) / tile_size + 1;

    // int q = n_keys / tile_size;
    // int r = n_keys % tile_size;

    int start = start_pair.first;
    int end = end_pair.first;
    // int end = start + q + (my_tile.thread_rank() < r ? 1 : 0);

    gpu_assert(start <= end, "Start ", start, " larger than end ", end, "\n");

    gpu_assert(end - start <= n_slots_per_thread,
               "Size calculation is busted: Thread has space for ",
               n_slots_per_thread, " but needs ", end - start,
               " slots covering", start, " to ", end, "\n");

    // comparable_type start_priority = read_from_head(start).priority;

    // comparable_type end_priority = 1.7976931348623158e+308;

    // if (my_tile.thread_rank() != my_tile.size()-1){
    //    end_priority = read_from_head(end).priority;
    // }

    // //now every thread has it's start/end points for the main queue!

    // //first write to stack
    pair_type stack_space[n_slots_per_thread];

    int stack_write_addr = 0;
    for (int i = start; i < end; i++) {
      stack_space[stack_write_addr++] = read_from_head(i);
    }

    // //for (int )

    // //find the first and last keys in our range

    int merge_list_start = start_pair.second;
    int merge_list_end = end_pair.second;

    gpu_assert((start < n_keys) || (start == end), "Thread ",
               my_tile.thread_rank(), " sees start outside of key range\n");
    gpu_assert((end <= n_keys) || (start == end), "Thread ",
               my_tile.thread_rank(), " sees end outside of key range\n");

    gpu_assert((merge_list_start < merge_list_size) ||
                   (merge_list_start == merge_list_end),
               "Thread ", my_tile.thread_rank(),
               " sees start outside of key range\n");
    gpu_assert((merge_list_end <= merge_list_size) ||
                   (merge_list_start == merge_list_end),
               "Thread ", my_tile.thread_rank(),
               " sees end outside of key range\n");

    __threadfence();
    my_tile.sync();

    // if (end != start){

    //    for (int i = 0; i < merge_list_size; i++){

    //       if ((merge_list_start == -1)){

    //          if (list_to_merge.distances[i] >= start_priority){
    //             merge_list_start = i;
    //          } else {
    //             continue;
    //          }

    //       }

    //       if (list_to_merge.distances[i] < end_priority){
    //          merge_list_end = i;
    //       }

    //       //calculate back end

    //    }

    // } else {
    //    merge_list_start = 0;
    //    merge_list_end = 0;

    //    if (my_tile.thread_rank() == my_tile.size()-1){
    //       merge_list_end = merge_list_size;
    //    }
    // }

    // // then copy from other list. as well.

    uint n_original_keys = end - start;

    uint n_new_keys = (merge_list_end - merge_list_start);

    uint my_n_keys = n_original_keys + n_new_keys;

    // //and ballot;

    int final_start = start + merge_list_start;

    if (my_tile.thread_rank() == my_tile.size() - 1) {
      uint final_size = min(final_start + my_n_keys, n_slots);

      // printf("New N keys is %u\n", final_size);

      n_keys = final_size;
    }

    // my_tile.sync();

    // printf("Thread %u starts writing at %u, taking %u keys from %d (left) and
    // %u keys from %d (right)\n", my_tile.thread_rank(), final_start,
    // n_original_keys, start, n_new_keys, merge_list_start);

    // //my_tile.sync();

    int l_head = 0;
    int r_head = 0;

    // start merge based on start && end
    while (final_start < n_slots && l_head < n_original_keys &&
           r_head < n_new_keys) {
      if (stack_space[l_head].priority <
          list_to_merge.distances[merge_list_start + r_head]) {
        // pop from stack and increment both.
        read_from_head(final_start++) = stack_space[l_head++];

      } else {
        // pop from new_list.
        read_from_head(final_start++) =
            list_to_merge.template get_pair<pair_type>(merge_list_start +
                                                       r_head++);
      }
    }

    // either one list is done or final start has been exceeded.
    // assert checks that either 1 of the two lists is ended or final_start is
    // done.
    gpu_assert(final_start >= n_slots || l_head == n_original_keys ||
                   r_head == n_new_keys,
               "Thread in invalid state after first while loop: ", final_start,
               " ", n_slots, " ", l_head, " ", n_original_keys, " ", r_head,
               " ", n_new_keys, "\n");

    //"iterate" through both lists. If final_start exceeds bounds do not write.

    while (final_start < n_slots && l_head < n_original_keys) {
      read_from_head(final_start++) = stack_space[l_head++];
    }

    while (final_start < n_slots && r_head < n_new_keys) {
      read_from_head(final_start++) =
          list_to_merge.template get_pair<pair_type>(merge_list_start +
                                                     r_head++);
    }

    __threadfence();
    my_tile.sync();
  }

  // helper to assert that keys are sorted after merge
  template <uint tile_size>
  __device__ void assert_sorted(cg::thread_block_tile<tile_size>& my_tile) {
    for (uint i = my_tile.thread_rank(); i < n_keys - 1; i += tile_size) {
      auto current_key = read_from_head(i);
      auto next_key = read_from_head(i + 1);

      gpu_assert(current_key.priority <= next_key.priority,
                 "Bad sort at index ", i, "\n");
    }
  }
};

}  // namespace gpu_ann

#endif  // GPU_BLOCK_