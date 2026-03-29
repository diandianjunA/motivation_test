// Jasper GPU ANN - Usage Example
//
// This example demonstrates how to use the Jasper library to:
//   1. Build an index from random vectors
//   2. Search for nearest neighbors
//   3. Save the index to disk
//   4. Load the index from disk and search again

#include <jasper/jasper.cuh>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

// Generate random float vectors for demonstration
std::vector<float> generate_random_vectors(uint64_t n, uint32_t dim) {
  std::vector<float> data(n * dim);
  for (auto &v : data) {
    v = static_cast<float>(rand() % 256) / 256.0f;
  }
  return data;
}

int main() {
  constexpr uint32_t DIM = 1024;
  constexpr uint64_t N_VECTORS = 10000;
  constexpr uint64_t N_QUERIES = 10;
  constexpr uint32_t K = 10;

  printf("=== Jasper GPU ANN Example ===\n\n");

  // 1. Generate random data
  printf("[1] Generating %lu random %u-dim vectors...\n", N_VECTORS, DIM);
  auto base_data = generate_random_vectors(N_VECTORS, DIM);
  auto query_data = generate_random_vectors(N_QUERIES, DIM);

  // 2. Build index
  printf("[2] Building index...\n");
  jasper::JasperIndex<DIM, float> index;

  jasper::BuildParams build_params;
  build_params.n_rounds = 1;
  build_params.nodes_explored_per_iteration = 4;
  build_params.random_init = false;
  build_params.alpha = 1.2;

  index.build(base_data.data(), N_VECTORS, build_params);
  printf("    Index built with %lu vectors.\n", index.size());

  // 3. Search
  printf("[3] Searching for %u nearest neighbors...\n", K);
  std::vector<uint32_t> result_ids(N_QUERIES * K);
  std::vector<float> result_dists(N_QUERIES * K);

  jasper::SearchParams search_params;
  search_params.beam_width = 64;

  index.search(query_data.data(), N_QUERIES, K,
               result_ids.data(), result_dists.data(), search_params);

  // Print results for first query
  printf("    Results for query 0:\n");
  for (uint32_t i = 0; i < K; i++) {
    printf("      rank %2u: id=%6u  dist=%.2f\n",
           i, result_ids[i], result_dists[i]);
  }

  // 4. Save index
  printf("[4] Saving index to disk...\n");
  index.save("/tmp/jasper_example");
  printf("    Saved to /tmp/jasper_example.index\n");

  // 5. Load index into a new instance
  printf("[5] Loading index from disk...\n");
  jasper::JasperIndex<DIM, float> loaded_index;
  loaded_index.load("/tmp/jasper_example", N_VECTORS);
  printf("    Loaded index with %lu vectors.\n", loaded_index.size());

  // 6. Search on loaded index
  printf("[6] Searching on loaded index...\n");
  std::vector<uint32_t> loaded_ids(N_QUERIES * K);
  std::vector<float> loaded_dists(N_QUERIES * K);

  loaded_index.search(query_data.data(), N_QUERIES, K,
                      loaded_ids.data(), loaded_dists.data(), search_params);

  printf("    Results for query 0 (loaded index):\n");
  for (uint32_t i = 0; i < K; i++) {
    printf("      rank %2u: id=%6u  dist=%.2f\n",
           i, loaded_ids[i], loaded_dists[i]);
  }

  // 7. Verify results match
  bool match = (result_ids == loaded_ids);
  printf("\n[7] Results match between original and loaded index: %s\n",
         match ? "YES" : "NO");

  printf("\nDone.\n");
  return 0;
}
