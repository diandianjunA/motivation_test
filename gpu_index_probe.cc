#include <algorithm>
#include <array>
#include <numeric>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

using u32 = std::uint32_t;
using u64 = std::uint64_t;

constexpr u32 kPadDim = 1024;
constexpr u32 kMaxR = 128;
constexpr u32 kTopK = 10;
constexpr u32 kSearchLimit = 512;
constexpr u32 kNodesExploredPerIteration = 4;

struct alignas(16) Rabitq4Code {
  std::uint8_t data[512]{};
  float add{};
  float rescale{};
};

struct alignas(16) Record {
  u32 user_id{};
  std::uint8_t degree{};
  std::uint8_t reserved0{};
  std::uint16_t reserved1{};
  float vector[kPadDim]{};
  Rabitq4Code rabitq{};
  std::uint32_t neighbors[kMaxR]{};
};

struct alignas(64) IndexMetadata {
  u64 used_bytes{};
  u64 magic{};
  u32 version{};
  u32 dim{};
  u32 pad_dim{};
  u32 max_vectors{};
  u32 num_memory_nodes{};
  u32 r{};
  u32 n_vertices{};
  u32 medoid_id{};
  u64 rotation_seed{};
  u64 shard_base{};
  u64 record_bytes{};
  float centroid[kPadDim]{};
};

struct SearchCandidate {
  u32 vertex_id{};
  float distance{};
  bool expanded{};
  std::shared_ptr<Record> record;
};

static float l2_distance(const float* lhs, const float* rhs, u32 dim) {
  float sum = 0.0f;
  for (u32 i = 0; i < dim; ++i) {
    const float diff = lhs[i] - rhs[i];
    sum += diff * diff;
  }
  return sum;
}

static bool by_distance(const SearchCandidate& lhs, const SearchCandidate& rhs) {
  if (lhs.distance == rhs.distance) return lhs.vertex_id < rhs.vertex_id;
  return lhs.distance < rhs.distance;
}

static std::pair<float*, std::pair<u32, u32>> read_fbin(const std::string& path) {
  auto input = std::ifstream(path, std::ios::binary);
  if (!input) throw std::runtime_error("failed to open fbin: " + path);

  u32 num = 0;
  u32 dim = 0;
  input.read(reinterpret_cast<char*>(&num), sizeof(num));
  input.read(reinterpret_cast<char*>(&dim), sizeof(dim));
  if (!input) throw std::runtime_error("failed to read fbin header: " + path);

  auto* data = new float[static_cast<size_t>(num) * dim];
  input.read(reinterpret_cast<char*>(data), sizeof(float) * static_cast<size_t>(num) * dim);
  if (!input) throw std::runtime_error("failed to read fbin payload: " + path);
  return {data, {num, dim}};
}

static std::pair<u32*, std::pair<u32, u32>> read_bin(const std::string& path) {
  auto input = std::ifstream(path, std::ios::binary);
  if (!input) throw std::runtime_error("failed to open bin: " + path);

  u32 num = 0;
  u32 dim = 0;
  input.read(reinterpret_cast<char*>(&num), sizeof(num));
  input.read(reinterpret_cast<char*>(&dim), sizeof(dim));
  if (!input) throw std::runtime_error("failed to read bin header: " + path);

  auto* data = new u32[static_cast<size_t>(num) * dim];
  input.read(reinterpret_cast<char*>(data), sizeof(u32) * static_cast<size_t>(num) * dim);
  if (!input) throw std::runtime_error("failed to read bin payload: " + path);
  return {data, {num, dim}};
}

class IndexReader {
public:
  explicit IndexReader(const std::string& path) : input_(path, std::ios::binary) {
    if (!input_) throw std::runtime_error("failed to open index: " + path);
    input_.read(reinterpret_cast<char*>(&metadata_), sizeof(metadata_));
    if (!input_) throw std::runtime_error("failed to read metadata");
    if (metadata_.pad_dim != kPadDim) throw std::runtime_error("unexpected pad_dim");
    if (metadata_.num_memory_nodes != 1) throw std::runtime_error("probe currently expects one shard");
    if (metadata_.record_bytes != sizeof(Record)) {
      throw std::runtime_error("record_bytes mismatch: file=" + std::to_string(metadata_.record_bytes) +
                               " local=" + std::to_string(sizeof(Record)));
    }
  }

  const IndexMetadata& metadata() const { return metadata_; }

  std::shared_ptr<Record> read_record(u32 vertex_id) {
    auto it = cache_.find(vertex_id);
    if (it != cache_.end()) return it->second;

    const u64 offset = metadata_.shard_base + static_cast<u64>(vertex_id) * metadata_.record_bytes;
    auto record = std::make_shared<Record>();
    input_.seekg(static_cast<std::streamoff>(offset), std::ios::beg);
    input_.read(reinterpret_cast<char*>(record.get()), sizeof(Record));
    if (!input_) throw std::runtime_error("failed to read record " + std::to_string(vertex_id));
    cache_.emplace(vertex_id, record);
    return record;
  }

private:
  std::ifstream input_;
  IndexMetadata metadata_{};
  std::unordered_map<u32, std::shared_ptr<Record>> cache_;
};

static u32 reachable_count(IndexReader& reader) {
  const auto& metadata = reader.metadata();
  std::vector<std::uint8_t> visited(metadata.n_vertices, 0);
  std::queue<u32> q;
  visited[metadata.medoid_id] = 1;
  q.push(metadata.medoid_id);
  u32 count = 0;

  while (!q.empty()) {
    const u32 vertex_id = q.front();
    q.pop();
    ++count;

    const auto record = reader.read_record(vertex_id);
    for (u32 ni = 0; ni < record->degree && ni < metadata.r; ++ni) {
      const u32 neighbor_id = record->neighbors[ni];
      if (neighbor_id < metadata.n_vertices && !visited[neighbor_id]) {
        visited[neighbor_id] = 1;
        q.push(neighbor_id);
      }
    }
  }
  return count;
}

struct DisjointSet {
  std::vector<u32> parent;
  std::vector<u32> size;

  explicit DisjointSet(u32 n) : parent(n), size(n, 1) {
    std::iota(parent.begin(), parent.end(), 0u);
  }

  u32 find(u32 x) {
    while (parent[x] != x) {
      parent[x] = parent[parent[x]];
      x = parent[x];
    }
    return x;
  }

  void unite(u32 a, u32 b) {
    a = find(a);
    b = find(b);
    if (a == b) return;
    if (size[a] < size[b]) std::swap(a, b);
    parent[b] = a;
    size[a] += size[b];
  }

  u32 component_size(u32 x) { return size[find(x)]; }
};

static u32 undirected_component_size(IndexReader& reader) {
  const auto& metadata = reader.metadata();
  DisjointSet dsu(metadata.n_vertices);
  for (u32 vertex_id = 0; vertex_id < metadata.n_vertices; ++vertex_id) {
    const auto record = reader.read_record(vertex_id);
    for (u32 ni = 0; ni < record->degree && ni < metadata.r; ++ni) {
      const u32 neighbor_id = record->neighbors[ni];
      if (neighbor_id < metadata.n_vertices) {
        dsu.unite(vertex_id, neighbor_id);
      }
    }
  }
  return dsu.component_size(metadata.medoid_id);
}

static std::vector<std::vector<u32>> build_symmetric_adjacency(IndexReader& reader) {
  const auto& metadata = reader.metadata();
  std::vector<std::vector<u32>> adjacency(metadata.n_vertices);
  for (u32 vertex_id = 0; vertex_id < metadata.n_vertices; ++vertex_id) {
    const auto record = reader.read_record(vertex_id);
    auto& out = adjacency[vertex_id];
    out.reserve(record->degree);
    for (u32 ni = 0; ni < record->degree && ni < metadata.r; ++ni) {
      const u32 neighbor_id = record->neighbors[ni];
      if (neighbor_id >= metadata.n_vertices || neighbor_id == vertex_id) continue;
      out.push_back(neighbor_id);
      adjacency[neighbor_id].push_back(vertex_id);
    }
  }

  for (auto& neighbors : adjacency) {
    std::sort(neighbors.begin(), neighbors.end());
    neighbors.erase(std::unique(neighbors.begin(), neighbors.end()), neighbors.end());
  }
  return adjacency;
}

static std::vector<std::vector<u32>> build_reverse_adjacency(IndexReader& reader) {
  const auto& metadata = reader.metadata();
  std::vector<std::vector<u32>> adjacency(metadata.n_vertices);
  for (u32 vertex_id = 0; vertex_id < metadata.n_vertices; ++vertex_id) {
    const auto record = reader.read_record(vertex_id);
    for (u32 ni = 0; ni < record->degree && ni < metadata.r; ++ni) {
      const u32 neighbor_id = record->neighbors[ni];
      if (neighbor_id >= metadata.n_vertices || neighbor_id == vertex_id) continue;
      adjacency[neighbor_id].push_back(vertex_id);
    }
  }
  for (auto& neighbors : adjacency) {
    std::sort(neighbors.begin(), neighbors.end());
    neighbors.erase(std::unique(neighbors.begin(), neighbors.end()), neighbors.end());
  }
  return adjacency;
}

static u32 reachable_count(const std::vector<std::vector<u32>>& adjacency, u32 start_vertex) {
  std::vector<std::uint8_t> visited(adjacency.size(), 0);
  std::queue<u32> q;
  visited[start_vertex] = 1;
  q.push(start_vertex);
  u32 count = 0;

  while (!q.empty()) {
    const u32 vertex_id = q.front();
    q.pop();
    ++count;
    for (u32 neighbor_id : adjacency[vertex_id]) {
      if (!visited[neighbor_id]) {
        visited[neighbor_id] = 1;
        q.push(neighbor_id);
      }
    }
  }
  return count;
}

static std::vector<u32> search_exact(IndexReader& reader, const float* query, u32 beam_width) {
  const auto& metadata = reader.metadata();
  std::vector<SearchCandidate> frontier;
  std::vector<SearchCandidate> discovered;
  std::unordered_set<u32> visited;

  auto medoid = reader.read_record(metadata.medoid_id);
  const float medoid_distance = l2_distance(query, medoid->vector, metadata.dim);
  frontier.push_back({metadata.medoid_id, medoid_distance, false, medoid});
  discovered.push_back(frontier.back());
  visited.insert(metadata.medoid_id);

  for (u32 iter = 0; iter < kSearchLimit; ++iter) {
    std::sort(frontier.begin(), frontier.end(), by_distance);

    std::vector<u32> pending_vertex_ids;
    pending_vertex_ids.reserve(kNodesExploredPerIteration * metadata.r);

    u32 expanded = 0;
    for (auto& candidate : frontier) {
      if (candidate.expanded) continue;
      candidate.expanded = true;
      ++expanded;

      for (u32 ni = 0; ni < candidate.record->degree && ni < metadata.r; ++ni) {
        const u32 neighbor_id = candidate.record->neighbors[ni];
        if (visited.insert(neighbor_id).second) {
          pending_vertex_ids.push_back(neighbor_id);
        }
      }

      if (expanded >= kNodesExploredPerIteration) break;
    }

    if (pending_vertex_ids.empty()) break;

    for (u32 neighbor_id : pending_vertex_ids) {
      auto record = reader.read_record(neighbor_id);
      SearchCandidate candidate{neighbor_id, l2_distance(query, record->vector, metadata.dim), false, record};
      frontier.push_back(candidate);
      discovered.push_back(candidate);
    }

    std::sort(frontier.begin(), frontier.end(), by_distance);
    if (frontier.size() > beam_width) {
      frontier.resize(beam_width);
    }
  }

  std::sort(discovered.begin(), discovered.end(), by_distance);
  std::vector<u32> results;
  for (const auto& candidate : discovered) {
    results.push_back(candidate.record->user_id);
    if (results.size() == kTopK) break;
  }
  return results;
}

static std::vector<u32> search_exact_with_adjacency(IndexReader& reader,
                                                    const std::vector<std::vector<u32>>& adjacency,
                                                    const float* query,
                                                    u32 beam_width) {
  const auto& metadata = reader.metadata();
  std::vector<SearchCandidate> frontier;
  std::vector<SearchCandidate> discovered;
  std::unordered_set<u32> visited;

  auto medoid = reader.read_record(metadata.medoid_id);
  const float medoid_distance = l2_distance(query, medoid->vector, metadata.dim);
  frontier.push_back({metadata.medoid_id, medoid_distance, false, medoid});
  discovered.push_back(frontier.back());
  visited.insert(metadata.medoid_id);

  for (u32 iter = 0; iter < kSearchLimit; ++iter) {
    std::sort(frontier.begin(), frontier.end(), by_distance);

    std::vector<u32> pending_vertex_ids;
    pending_vertex_ids.reserve(kNodesExploredPerIteration * metadata.r * 2);

    u32 expanded = 0;
    for (auto& candidate : frontier) {
      if (candidate.expanded) continue;
      candidate.expanded = true;
      ++expanded;

      for (u32 neighbor_id : adjacency[candidate.vertex_id]) {
        if (visited.insert(neighbor_id).second) {
          pending_vertex_ids.push_back(neighbor_id);
        }
      }

      if (expanded >= kNodesExploredPerIteration) break;
    }

    if (pending_vertex_ids.empty()) break;

    for (u32 neighbor_id : pending_vertex_ids) {
      auto record = reader.read_record(neighbor_id);
      SearchCandidate candidate{neighbor_id, l2_distance(query, record->vector, metadata.dim), false, record};
      frontier.push_back(candidate);
      discovered.push_back(candidate);
    }

    std::sort(frontier.begin(), frontier.end(), by_distance);
    if (frontier.size() > beam_width) {
      frontier.resize(beam_width);
    }
  }

  std::sort(discovered.begin(), discovered.end(), by_distance);
  std::vector<u32> results;
  for (const auto& candidate : discovered) {
    results.push_back(candidate.record->user_id);
    if (results.size() == kTopK) break;
  }
  return results;
}

int main(int argc, char** argv) {
  if (argc != 4 && argc != 5) {
    std::cerr << "usage: " << argv[0] << " <index.dat> <query.fbin> <groundtruth.bin> [beam_width]\n";
    return 1;
  }
  const u32 beam_width = argc == 5 ? static_cast<u32>(std::stoul(argv[4])) : 27;

  IndexReader reader(argv[1]);
  const auto& metadata = reader.metadata();
  std::cerr << "metadata: n_vertices=" << metadata.n_vertices << " dim=" << metadata.dim
            << " medoid_id=" << metadata.medoid_id << " record_bytes=" << metadata.record_bytes << "\n";
  std::cerr << "reachable_from_medoid=" << reachable_count(reader) << "\n";
  std::cerr << "undirected_component_from_medoid=" << undirected_component_size(reader) << "\n";
  const auto reverse_adjacency = build_reverse_adjacency(reader);
  std::cerr << "reachable_from_medoid_on_reverse_graph="
            << reachable_count(reverse_adjacency, metadata.medoid_id) << "\n";
  const auto symmetric_adjacency = build_symmetric_adjacency(reader);
  std::cerr << "reachable_from_medoid_after_symmetrize="
            << reachable_count(symmetric_adjacency, metadata.medoid_id) << "\n";

  auto [queries, query_info] = read_fbin(argv[2]);
  auto [groundtruth, gt_info] = read_bin(argv[3]);
  if (query_info.first != gt_info.first || gt_info.second < kTopK) {
    throw std::runtime_error("query/groundtruth shape mismatch");
  }

  const u32 num_queries = std::min<u32>(query_info.first, 20);
  double recall_sum = 0.0;
  for (u32 qid = 0; qid < num_queries; ++qid) {
    const float* query = queries + static_cast<size_t>(qid) * query_info.second;
    auto result = search_exact(reader, query, beam_width);

    u32 hits = 0;
    for (u32 j = 0; j < kTopK; ++j) {
      const u32 truth = groundtruth[static_cast<size_t>(qid) * gt_info.second + j];
      if (std::find(result.begin(), result.end(), truth) != result.end()) {
        ++hits;
      }
    }
    recall_sum += static_cast<double>(hits) / static_cast<double>(kTopK);

    std::cerr << "query " << qid << " result:";
    for (u32 id : result) std::cerr << ' ' << id;
    std::cerr << "\nquery " << qid << " truth:";
    for (u32 j = 0; j < kTopK; ++j) {
      std::cerr << ' ' << groundtruth[static_cast<size_t>(qid) * gt_info.second + j];
    }
    std::cerr << "\nquery " << qid << " recall=" << static_cast<double>(hits) / static_cast<double>(kTopK) << "\n";

    if (qid == 0) {
      std::cerr << "query 0 result distances:";
      for (u32 id : result) {
        const auto record = reader.read_record(id);
        std::cerr << ' ' << l2_distance(query, record->vector, metadata.dim);
      }
      std::cerr << "\nquery 0 truth distances:";
      for (u32 j = 0; j < kTopK; ++j) {
        const u32 id = groundtruth[static_cast<size_t>(qid) * gt_info.second + j];
        const auto record = reader.read_record(id);
        std::cerr << ' ' << l2_distance(query, record->vector, metadata.dim);
      }
      std::cerr << "\n";
    }
  }

  std::cerr << "avg_recall@" << kTopK << " over " << num_queries << " queries (beam_width=" << beam_width
            << ") = " << (recall_sum / num_queries) << "\n";

  double reverse_recall_sum = 0.0;
  for (u32 qid = 0; qid < num_queries; ++qid) {
    const float* query = queries + static_cast<size_t>(qid) * query_info.second;
    const auto result = search_exact_with_adjacency(reader, reverse_adjacency, query, beam_width);
    u32 hits = 0;
    for (u32 j = 0; j < kTopK; ++j) {
      const u32 truth = groundtruth[static_cast<size_t>(qid) * gt_info.second + j];
      if (std::find(result.begin(), result.end(), truth) != result.end()) {
        ++hits;
      }
    }
    reverse_recall_sum += static_cast<double>(hits) / static_cast<double>(kTopK);
  }
  std::cerr << "avg_recall@" << kTopK << " on reverse graph over " << num_queries
            << " queries (beam_width=" << beam_width << ") = "
            << (reverse_recall_sum / num_queries) << "\n";

  double symmetric_recall_sum = 0.0;
  for (u32 qid = 0; qid < num_queries; ++qid) {
    const float* query = queries + static_cast<size_t>(qid) * query_info.second;
    const auto result = search_exact_with_adjacency(reader, symmetric_adjacency, query, beam_width);
    u32 hits = 0;
    for (u32 j = 0; j < kTopK; ++j) {
      const u32 truth = groundtruth[static_cast<size_t>(qid) * gt_info.second + j];
      if (std::find(result.begin(), result.end(), truth) != result.end()) {
        ++hits;
      }
    }
    symmetric_recall_sum += static_cast<double>(hits) / static_cast<double>(kTopK);
  }
  std::cerr << "avg_recall@" << kTopK << " after symmetrize over " << num_queries
            << " queries (beam_width=" << beam_width << ") = "
            << (symmetric_recall_sum / num_queries) << "\n";

  delete[] queries;
  delete[] groundtruth;
  return 0;
}
