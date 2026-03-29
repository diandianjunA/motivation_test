#pragma once

#include <cooperative_groups.h>

// Polyfill for cg::invoke_one_broadcast (requires CUDA 12.3+).
// Thread 0 executes the lambda and broadcasts the result to the tile.
template <typename TileT, typename Fn>
__device__ auto invoke_one_broadcast_compat(TileT &tile, Fn &&fn)
    -> decltype(fn()) {
  using R = decltype(fn());
  union { R val; char bytes[sizeof(R)]; };
  if (tile.thread_rank() == 0) {
    val = fn();
  }
  for (unsigned i = 0; i < sizeof(R); i++) {
    bytes[i] = tile.shfl(bytes[i], 0);
  }
  return val;
}
