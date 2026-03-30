#pragma once

#include <cstddef>

#include "node.hh"

namespace index_buffer {

struct SanitizeResult {
  bool ok{true};
  size_t cleared_headers{};
  str message;
};

inline SanitizeResult clear_runtime_locks(byte_t* buffer, size_t buffer_size, u32 dim, u32 m) {
  if (buffer_size < 2 * sizeof(u64)) {
    return {false, 0, "index buffer too small"};
  }

  Node::init_static_storage(dim, m, m * 2);

  const u64 free_ptr = *reinterpret_cast<u64*>(buffer);
  if (free_ptr < 2 * sizeof(u64) || free_ptr > buffer_size) {
    return {false, 0, "index free pointer out of bounds"};
  }

  size_t cleared_headers = 0;
  u64 offset = 2 * sizeof(u64);

  while (offset < free_ptr) {
    if (offset + Node::HEADER_SIZE + Node::META_SIZE > free_ptr) {
      return {false, cleared_headers, "truncated node header in index buffer"};
    }

    auto* header_ptr = reinterpret_cast<u64*>(buffer + offset);
    const u64 sanitized_header =
      *header_ptr & ~(static_cast<u64>(Node::HEADER_NODE_LOCK) | static_cast<u64>(Node::HEADER_NEW_LEVEL_LOCK));
    if (*header_ptr != sanitized_header) {
      *header_ptr = sanitized_header;
      ++cleared_headers;
    }

    const u32 level = *reinterpret_cast<u32*>(buffer + offset + Node::HEADER_SIZE + sizeof(u32));
    size_t node_size = Node::total_size(level);
    while (node_size % 8 != 0) {
      node_size += 4;
    }

    if (offset + node_size > free_ptr) {
      return {false, cleared_headers, "node size exceeds index buffer bounds"};
    }

    offset += node_size;
  }

  if (offset != free_ptr) {
    return {false, cleared_headers, "index buffer traversal ended at unexpected offset"};
  }

  return {true, cleared_headers, ""};
}

}  // namespace index_buffer
