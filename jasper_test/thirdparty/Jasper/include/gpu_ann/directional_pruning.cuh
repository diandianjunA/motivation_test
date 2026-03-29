#ifndef GPU_ANN_DIRECTIONAL_PRUNING_CUH
#define GPU_ANN_DIRECTIONAL_PRUNING_CUH

namespace gpu_ann {

// Compute directional signature relative to medoid using 32 hyperplanes
template <typename vector_type, uint32_t D>
__device__ uint32_t compute_directional_signature(const vector_type *vector,
                                                  const vector_type *medoid) {
  uint32_t signature = 0;

  uint32_t step = (D - 1) / 32 + 1;

  for (uint32_t i = 0; i < 32; i++) {
    uint32_t dim = (i * step) % D;  // Map 32 bits evenly across D dimensions
    if (vector[0][dim] > medoid[0][dim]) {
      signature |= (1u << i);
    }
  }

  return signature;
}

// Filter new candidates by their direectional signature
// Inputs are the two uint32_t signatures and a threshold
// Returns True if the signatures have >= threshold bits in common.
// XNOR returns 1 if bits are identical, then builtin popc.
__device__ bool hyperplane_signature_filter(uint32_t search_signature,
                                            uint32_t edge_signature,
                                            uint32_t threshold) {
  uint32_t common_bits = __popc(~(search_signature ^ edge_signature));

  // Debug: print first few comparisons
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    printf(
        "Signatures: search=0x%08x, edge=0x%08x, common=%u, threshold=%u, "
        "pass=%d\n",
        search_signature, edge_signature, common_bits, threshold,
        common_bits >= threshold);
  }

  return common_bits >= threshold;
}

// Initialize directional signatures for a batch of vectors
template <typename vector_type, typename vertex_data_type, uint32_t D>
__global__ void initialize_directional_signatures(
    const vector_type *all_vectors, vertex_data_type medoid_id,
    vertex_data_type start, uint32_t n_vectors,
    uint32_t *directional_signatures) {
  uint32_t tid = gallatin::utils::get_tid();
  if (tid >= n_vectors) return;

  const vector_type *search_vector = &all_vectors[tid + start];
  const vector_type *medoid = &all_vectors[medoid_id];

  directional_signatures[tid] =
      compute_directional_signature<vector_type, D>(search_vector, medoid);
}

// Debug function to print first 10 signatures
__host__ void debug_print_signatures(uint32_t *device_signatures,
                                     uint32_t n_vectors) {
  uint32_t n_to_print = min(10u, n_vectors);
  uint32_t *host_signatures = new uint32_t[n_to_print];

  cudaMemcpy(host_signatures, device_signatures, n_to_print * sizeof(uint32_t),
             cudaMemcpyDeviceToHost);

  printf("First %u directional signatures:\n", n_to_print);
  for (uint32_t i = 0; i < n_to_print; i++) {
    printf("Signature[%u]: 0x%08x (%u bits set)\n", i, host_signatures[i],
           __builtin_popcount(host_signatures[i]));
  }

  delete[] host_signatures;
}

}  // namespace gpu_ann

#endif
