#pragma once

#include <library/utils.hh>

#include "types.hh"

namespace gpu_index_path {

inline filepath_t base_directory(const filepath_t& data_path) {
  lib_assert(!data_path.empty(), "data path must not be empty");
  return data_path.has_extension() ? data_path.parent_path() : data_path;
}

inline filepath_t default_prefix(const filepath_t& data_path) {
  return base_directory(data_path) / "dump" / "gpu_index";
}

inline filepath_t resolve_prefix(const filepath_t& data_path, const filepath_t& explicit_prefix) {
  return explicit_prefix.empty() ? default_prefix(data_path) : explicit_prefix;
}

inline filepath_t shard_file(const filepath_t& prefix, size_t node_ordinal, size_t num_nodes) {
  return filepath_t(prefix.string() + "_node" + std::to_string(node_ordinal) + "_of" +
                    std::to_string(num_nodes) + ".dat");
}

inline filepath_t metadata_file(const filepath_t& prefix) { return filepath_t(prefix.string() + ".meta.json"); }

}  // namespace gpu_index_path
