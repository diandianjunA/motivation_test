#pragma once

#include <iomanip>
#include <iostream>
#include <library/configuration.hh>

#include "common/index_path.hh"
#include "types.hh"

namespace configuration {

// struct used for sending serialized from CN to MN
struct Parameters {
  u32 num_threads{};
  bool use_cache{};
  bool routing{};
};

class IndexConfiguration : public Configuration {
public:
  filepath_t data_path{};
  filepath_t index_prefix{};
  str query_suffix{};
  u32 num_threads{};
  u32 num_coroutines{};
  i32 seed{};
  bool disable_thread_pinning{};
  str label{};  // for labeling benchmarks

  // GPU graph index parameters
  u32 ef_search{};
  u32 ef_construction{};
  u32 k{};
  u32 m{};

  bool store_index{};  // memory servers store the GPU index; location is derived from data_path/index_prefix
  bool load_index{};  // memory servers load the GPU index; location is derived from data_path/index_prefix
  bool no_recall{};  // legacy benchmark flag
  bool ip_distance{};  // use the inner product distance rather than squared L2 norm

  u32 cache_size_ratio{};  // in %
  bool use_cache{};
  bool routing{};

  u32 dim{};
  u32 max_vectors{1000000};
  u32 gpu_device{0};

  // Memory size parameters (in GB)
  u32 cn_memory_gb{10};
  u32 mn_memory_gb{10};

public:
  IndexConfiguration(int argc, char** argv) {
    add_options();
    process_program_options(argc, argv);

    if (!is_server) {
      validate_compute_node_options(argv);
    }

    operator<<(std::cerr, *this);
  }

private:
  void add_options() {
    desc.add_options()("data-path,d",
                       po::value<filepath_t>(&data_path),
                       "Data directory used to derive the default GPU index prefix.")(
      "index-prefix",
      po::value<filepath_t>(&index_prefix),
      "Path prefix of GPU index shard files without the _nodeX_ofN.dat suffix.")(
      "threads,t", po::value<u32>(&num_threads), "Number of threads per compute node.")(
      "coroutines,C", po::value<u32>(&num_coroutines)->default_value(4), "Number of coroutines per compute thread.")(
      "disable-thread-pinning,p",
      po::bool_switch(&disable_thread_pinning)->default_value(false),
      "Disables pinning compute threads to physical cores if set.")(
      "seed", po::value<i32>(&seed)->default_value(1234), "Seed for PRNG; setting to -1 uses std::random_device.")(
      "label", po::value<str>(&label), "Optional label to identify benchmarks.")(
      "query-suffix,q", po::value<str>(&query_suffix), "Legacy benchmark option retained for compatibility.")(
      "store-index,s",
      po::bool_switch(&store_index),
      "After startup, ask the memory servers to persist the GPU index to files under --data-path.")(
      "load-index,l",
      po::bool_switch(&load_index),
      "During startup, ask the memory servers to load the GPU index from files under --data-path.")(
      "cache", po::bool_switch(&use_cache), "Activate the local record cache on the compute node.")(
      "routing", po::bool_switch(&routing), "Activate adaptive query routing across compute nodes.")(
      "cache-ratio",
      po::value<u32>(&cache_size_ratio)->default_value(5),
      "Cache size ratio relative to the index size in %.")(
      "no-recall", po::bool_switch(&no_recall), "Legacy benchmark option retained for compatibility.")(
      "ip-dist", po::bool_switch(&ip_distance), "Use the inner product distance rather than the squared L2 norm.")(
      "ef-search", po::value<u32>(&ef_search), "Beam width during search.")(
      "ef-construction",
      po::value<u32>(&ef_construction)->default_value(200),
      "Beam width during insertion / construction.")(
      "k,k", po::value<u32>(&k), "Number of k nearest neighbors.")(
      "m,m", po::value<u32>(&m)->default_value(32), "Vamana out-degree R.")(
      "dim", po::value<u32>(&dim), "Vector dimension")(
      "max-vectors", po::value<u32>(&max_vectors)->default_value(1000000), "Max vectors capacity")(
      "gpu-device", po::value<u32>(&gpu_device)->default_value(0), "CUDA device ordinal for this compute node")(
      "cn-memory", po::value<u32>(&cn_memory_gb)->default_value(10), "Compute node local buffer size in GB")(
      "mn-memory", po::value<u32>(&mn_memory_gb)->default_value(10), "Memory node buffer size in GB");
  }

  void validate_compute_node_options(char** argv) const {
    if (num_threads == 0 || ef_search == 0 || k == 0 || dim == 0) {
      std::cerr << "[ERROR]: Parameters threads, ef-search, k, and dim are required" << std::endl;
      exit_with_help_message(argv);
    }

    if (num_threads < 2) {
      std::cerr << "[ERROR]: GPU baseline requires at least 2 threads" << std::endl;
      exit_with_help_message(argv);
    }

    if (store_index && load_index) {
      std::cerr << "[ERROR]: --store-index and --load-index cannot be used in conjunction" << std::endl;
      exit_with_help_message(argv);
    }

    if ((store_index || load_index) && data_path.empty() && index_prefix.empty()) {
      std::cerr << "[ERROR]: --data-path or --index-prefix is required when --load-index or --store-index is set"
                << std::endl;
      exit_with_help_message(argv);
    }

    if (use_cache && cache_size_ratio == 0) {
      std::cerr << "[ERROR]: If --cache is set, --cache-ratio must be > 0" << std::endl;
      exit_with_help_message(argv);
    }

  }

public:
  filepath_t resolved_index_prefix() const { return gpu_index_path::resolve_prefix(data_path, index_prefix); }

  friend std::ostream& operator<<(std::ostream& os, const IndexConfiguration& config) {
    os << static_cast<const Configuration&>(config);

    if (config.is_initiator) {
      constexpr i32 width = 30;
      constexpr i32 max_width = width * 2;

      os << std::left << std::setfill(' ');
      os << std::setw(width) << "data path: " << config.data_path << std::endl;
      if (!config.index_prefix.empty()) {
        os << std::setw(width) << "index prefix: " << config.index_prefix << std::endl;
      }
      os << std::setw(width) << "query suffix (legacy): " << config.query_suffix << std::endl;
      os << std::setw(width) << "number of threads: " << config.num_threads << std::endl;
      os << std::setw(width) << "number of coroutines: " << config.num_coroutines << std::endl;
      os << std::setw(width) << "threads pinned: " << (config.disable_thread_pinning ? "false" : "true") << std::endl;
      os << std::setw(width) << "seed: " << config.seed << std::endl;
      os << std::setw(width) << "dimension: " << config.dim << std::endl;
      os << std::setw(width) << "max vectors: " << config.max_vectors << std::endl;
      os << std::setw(width) << "gpu device: " << config.gpu_device << std::endl;
      os << std::setw(width) << "CN memory (GB): " << config.cn_memory_gb << std::endl;
      os << std::setfill('-') << std::setw(max_width) << "" << std::endl;
      os << std::left << std::setfill(' ');
      os << std::setw(width) << "K: " << config.k << std::endl;
      os << std::setw(width) << "M: " << config.m << std::endl;
      os << std::setw(width) << "ef search: " << config.ef_search << std::endl;
      os << std::setw(width) << "ef construction: " << config.ef_construction << std::endl;
      os << std::setfill('=') << std::setw(max_width) << "" << std::endl;
    }
    return os;
  }
};

}  // namespace configuration
