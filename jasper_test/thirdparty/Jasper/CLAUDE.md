# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build Commands

```bash
# Configure (auto-detects GPU arch, or specify with -DGPU_ARCHS=86)
cmake -S . -B build

# If network is unavailable and deps are already cached:
cmake -S . -B build -DFETCHCONTENT_UPDATES_DISCONNECTED=ON

# Build all targets
cmake --build build -j

# Run example
./build/bin/jasper_example
```

Build targets: `jasper` (header-only library), `jasper_example`, `jasper_example2`.

There is no test suite for the main library. Tests exist only under `gpu_error/tests/`.

## Architecture

Jasper is a GPU-accelerated approximate nearest neighbor (ANN) library implementing the Vamana graph algorithm. It is header-only — all code lives in `include/`.

### Layer Structure

1. **Public API** (`include/jasper/jasper.cuh`) — `JasperIndex<VECTOR_DIM, DATA_T, R, L_CAP, ON_HOST>` wraps the internal engine with build/search/insert/save/load methods. This is the only header users include.

2. **Graph Engine** (`include/gpu_ann/bulk_gpuANN.cuh`, ~2200 lines) — The core `bulk_gpuANN` class manages graph construction via incremental batched insertion. Key flow:
   - `construction_round` → `process_batch` → `beam_search_and_prune`
   - `beam_search_and_prune`: runs beam search on new vectors, calls `add_new_edges<false>` (forward edges with `robust_prune_block`), then `flip_edges` + `add_new_edges_in_place<true>` (reverse edges with `robust_prune_old_edges`)
   - Batches grow exponentially (1, 2, 4, 8, ... up to `max_batch_size`)

3. **Beam Search** (`include/gpu_ann/beam_search.cuh`) — Single-kernel beam search using shared memory. Entry encoding packs visited bit + 31-bit index + 32-bit float distance into a 64-bit value. Key functions: `beam_search_single_kernel`, `add_frontier_out`, `populate_distances`, `merge_sort`, `dedup_results`, `filter_frontier`.

4. **CUDA Kernels** (`include/gpu_ann/kernels/`) — `prune_kernels.cuh` contains `robust_prune_block`, `robust_prune_old_edges`, `robust_prune_thrust`. `helper_kernels.cuh` and `beam_search_kernels.cuh` contain supporting kernels.

5. **Data Structures** — `edge_list.cuh` (fixed-size `edges[R]` array per vertex), `vector.cuh` (`data_vector<T, DIM>`, 16-byte aligned), `edge_sorting.cuh` (`edge_pair<data_type, dist_type>` with source/sink/distance).

### Key Type Relationships

```
JasperIndex<DIM>
  └─ ann_type = bulk_gpuANN<1, 512, uint32_t, float, uint8_t, DIM, uint8_t, 8, R=64, L_CAP=64, distance_functor, safe_lookup, ON_HOST>
       ├─ edges: edge_list<uint32_t, 64>*     (GPU array, one per vertex)
       ├─ edge_counts: uint8_t*                (GPU array, degree per vertex)
       ├─ vectors: data_vector<uint8_t, DIM>*  (GPU array)
       └─ medoid: uint32_t                     (entry point for search)
```

### Edge Sorting Comparators

- `operator<` on `edge_pair`: sorts by (source, distance)
- `beamSearchComparator`: sorts by (source, distance, sink)
- `pruneEqualityComparator`: groups by source only
- `operator==` on `edge_pair`: compares source AND sink (ignores distance)
- `dead_edge`: sentinel with source=sink=UINT32_MAX, distance=FLT_MAX

### Known Pitfalls

- CUDA kernel launches can fail silently with "too many resources requested for launch" if thread count is too high for the register pressure. Always check `cudaGetLastError()` after kernel launches during debugging.
- The `robust_prune_old_edges` kernel requires ≤256 threads per block (512 exceeds register limits on some GPUs). The `robust_prune_block` kernel can handle 512.
- `gpu_assert` is a no-op in release builds (`#define gpu_assert(...) ((void)0)`).
- `thrust::sort` with no comparator uses `edge_pair::operator<` (source, distance). `make_sorted_and_unique` uses `beamSearchComparator` (source, distance, sink). These are different orderings.

### Dependencies

- **Gallatin**: GPU memory allocator, provides `gallatin::utils::get_host_version<T>(n)` for pinned host memory and `get_device_version<T>(n)` for device memory.
- **Thrust/CUB**: Used extensively for sorting, reduction, scan, and device vectors.
- **Cooperative Groups**: Thread block tiles (tile_size=16) for collaborative distance computation.
- **gpu_error** (in-repo): GPU-side logging. `init_gpu_log(bytes)` must be called before use. Disabled with `GPU_NDEBUG`.
