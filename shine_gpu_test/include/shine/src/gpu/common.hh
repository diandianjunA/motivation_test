#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

#include "common/types.hh"

namespace gpu {

constexpr u64 kIndexMagic = 0x4750555348494E45ULL;  // "GPUSHINE"
constexpr u32 kIndexVersion = 1;
constexpr u32 kMaxPadDim = 1024;
constexpr u32 kRabitqBits = 4;
constexpr u32 kMaxR = 128;
constexpr u32 kDefaultNodesExploredPerIteration = 4;
constexpr u32 kDefaultSearchLimit = 512;
constexpr float kDefaultCut = 10.0f;
constexpr float kDefaultAlpha = 1.2f;

inline u64 align_up(u64 value, u64 alignment) {
  return ((value + alignment - 1) / alignment) * alignment;
}

inline u32 choose_pad_dim(u32 dim) {
  if (dim <= 128) return 128;
  if (dim <= 256) return 256;
  if (dim <= 512) return 512;
  if (dim <= 1024) return 1024;
  throw std::runtime_error("GPU baseline only supports dimensions up to 1024");
}

template <u32 PadDim>
struct alignas(16) Rabitq4Code {
  std::uint8_t data[(kRabitqBits * PadDim + 7) / 8]{};
  float add{};
  float rescale{};
};

template <u32 PadDim>
struct alignas(16) VertexRecord {
  node_t user_id{};
  std::uint8_t degree{};
  std::uint8_t reserved0{};
  std::uint16_t reserved1{};
  float vector[PadDim]{};
  Rabitq4Code<PadDim> rabitq{};
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
  float centroid[kMaxPadDim]{};
};

struct PendingInsert {
  node_t user_id{};
  const element_t* components{};
};

struct PendingQuery {
  const element_t* components{};
  u32 k{};
};

inline u64 estimate_record_bytes(u32 pad_dim) {
  const u64 header_bytes = sizeof(node_t) + sizeof(std::uint8_t) + sizeof(std::uint8_t) + sizeof(std::uint16_t);
  const u64 vector_bytes = static_cast<u64>(pad_dim) * sizeof(float);
  const u64 rabitq_bytes = ((static_cast<u64>(kRabitqBits) * pad_dim + 7) / 8) + 2 * sizeof(float);
  const u64 neighbor_bytes = static_cast<u64>(kMaxR) * sizeof(std::uint32_t);
  return align_up(header_bytes + vector_bytes + rabitq_bytes + neighbor_bytes, 16);
}

}  // namespace gpu
