#include "odinann_index.h"

#include "aux_utils.h"
#include "config.h"
#include "ssd_index.h"
#include <filesystem>
#include <fstream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <cstdlib>
#include <unistd.h>

namespace {

size_t getRequiredSize(const std::map<std::string, std::string>& config, const std::string& key) {
    const auto it = config.find(key);
    if (it == config.end()) {
        throw std::runtime_error("Config file missing " + key);
    }
    return static_cast<size_t>(std::stoull(it->second));
}

size_t getOptionalSize(
    const std::map<std::string, std::string>& config,
    const std::string& key,
    size_t default_value) {
    const auto it = config.find(key);
    if (it == config.end()) {
        return default_value;
    }
    return static_cast<size_t>(std::stoull(it->second));
}

float getOptionalFloat(
    const std::map<std::string, std::string>& config,
    const std::string& key,
    float default_value) {
    const auto it = config.find(key);
    if (it == config.end()) {
        return default_value;
    }
    return std::stof(it->second);
}

std::string getOptionalString(
    const std::map<std::string, std::string>& config,
    const std::string& key,
    const std::string& default_value) {
    const auto it = config.find(key);
    if (it == config.end()) {
        return default_value;
    }
    return it->second;
}

bool getOptionalBool(
    const std::map<std::string, std::string>& config,
    const std::string& key,
    bool default_value) {
    const auto it = config.find(key);
    if (it == config.end()) {
        return default_value;
    }
    return it->second == "true";
}

bool hasIdentityIds(const std::vector<uint32_t>& ids) {
    for (size_t i = 0; i < ids.size(); ++i) {
        if (ids[i] != static_cast<uint32_t>(i)) {
            return false;
        }
    }
    return true;
}

pipeann::Metric parseMetric(const std::map<std::string, std::string>& config) {
    const auto metric = getOptionalString(config, "metric", "l2");
    if (metric == "l2") {
        return pipeann::Metric::L2;
    }
    if (metric == "cosine") {
        return pipeann::Metric::COSINE;
    }
    throw std::runtime_error("Unsupported metric: " + metric);
}

std::string makeTempPath(const std::string& pattern) {
    std::vector<char> writable(pattern.begin(), pattern.end());
    writable.push_back('\0');
    const int fd = mkstemp(writable.data());
    if (fd == -1) {
        throw std::runtime_error("Failed to create temporary file");
    }
    close(fd);
    return std::string(writable.data());
}

}  // namespace

OdinANNIndex::OdinANNIndex(std::string conf, std::string runtime_conf) : config_path_(std::move(conf)) {
    config_ = readConfig(config_path_);
    if (!runtime_conf.empty()) {
        const auto runtime_config = readConfig(runtime_conf);
        for (const auto& [key, value] : runtime_config) {
            config_.insert_or_assign(key, value);
        }
    }

    dim_ = getRequiredSize(config_, "dim");
    max_points_ = getRequiredSize(config_, "max_points_to_insert");
    build_R_ = static_cast<uint32_t>(getOptionalSize(config_, "build_R", getOptionalSize(config_, "R", 48)));
    build_L_ = static_cast<uint32_t>(getOptionalSize(config_, "build_L", getOptionalSize(config_, "L", 150)));
    build_B_ = getOptionalString(config_, "build_B", "8");
    build_M_ = getOptionalString(config_, "build_M", "16");
    build_threads_ = static_cast<uint32_t>(getOptionalSize(config_, "build_threads", getOptionalSize(config_, "num_threads", 16)));
    search_L_ = static_cast<uint32_t>(getOptionalSize(config_, "search_L", build_L_));
    beamwidth_ = static_cast<uint32_t>(getOptionalSize(config_, "beamwidth", 4));
    runtime_threads_ = static_cast<uint32_t>(getOptionalSize(config_, "num_threads", 16));
    merge_L_disk_ = static_cast<uint32_t>(getOptionalSize(config_, "L_disk", build_L_));
    merge_R_disk_ = static_cast<uint32_t>(getOptionalSize(config_, "R_disk", 0));
    alpha_disk_ = getOptionalFloat(config_, "alpha_disk", 1.2F);
    merge_C_ = static_cast<uint32_t>(getOptionalSize(config_, "C", 384));
    mem_L_ = static_cast<uint32_t>(getOptionalSize(config_, "L_mem", 192));
    mem_R_ = static_cast<uint32_t>(getOptionalSize(config_, "R_mem", 64));
    mem_alpha_ = getOptionalFloat(config_, "alpha_mem", 1.2F);
    mem_C_ = static_cast<uint32_t>(getOptionalSize(config_, "C_mem", getOptionalSize(config_, "C", 100)));
    search_mem_L_ = static_cast<uint32_t>(getOptionalSize(config_, "search_mem_L", 0));
    use_mem_index_ = getOptionalBool(config_, "use_mem_index", search_mem_L_ > 0);
    single_file_index_ = getOptionalBool(config_, "single_file_index", false);
    force_sharded_build_ = getOptionalBool(config_, "force_sharded_build", false);
    partition_replication_factor_ =
        static_cast<uint32_t>(getOptionalSize(config_, "partition_replication_factor", 2));
    metric_ = parseMetric(config_);

    if (metric_ == pipeann::Metric::COSINE) {
        dist_ = std::make_unique<pipeann::DistanceCosineFloat>();
    } else {
        dist_ = std::make_unique<pipeann::DistanceL2>();
    }
}

void OdinANNIndex::clearPendingTempFiles() {
    if (pending_dataset_is_temp_ && !pending_dataset_path_.empty()) {
        std::filesystem::remove(pending_dataset_path_);
    }
    if (pending_tag_is_temp_ && !pending_tag_path_.empty()) {
        std::filesystem::remove(pending_tag_path_);
    }
    pending_dataset_path_.clear();
    pending_tag_path_.clear();
    pending_identity_tags_ = false;
    pending_dataset_is_temp_ = false;
    pending_tag_is_temp_ = false;
}

std::string OdinANNIndex::writeTempFbin(const std::vector<float>& vecs, size_t count) const {
    std::filesystem::create_directories("/tmp");
    const std::string path = makeTempPath("/tmp/odinann_data_XXXXXX");
    std::ofstream out(path, std::ios::binary);
    const uint32_t n = static_cast<uint32_t>(count);
    const uint32_t d = static_cast<uint32_t>(dim_);
    out.write(reinterpret_cast<const char*>(&n), sizeof(uint32_t));
    out.write(reinterpret_cast<const char*>(&d), sizeof(uint32_t));
    out.write(reinterpret_cast<const char*>(vecs.data()), static_cast<std::streamsize>(vecs.size() * sizeof(float)));
    return path;
}

std::string OdinANNIndex::writeTagFile(const std::vector<uint32_t>& ids, const std::string& path) const {
    std::ofstream out(path, std::ios::binary);
    const uint32_t n = static_cast<uint32_t>(ids.size());
    const uint32_t d = 1;
    out.write(reinterpret_cast<const char*>(&n), sizeof(uint32_t));
    out.write(reinterpret_cast<const char*>(&d), sizeof(uint32_t));
    out.write(reinterpret_cast<const char*>(ids.data()), static_cast<std::streamsize>(ids.size() * sizeof(uint32_t)));
    return path;
}

int OdinANNIndex::resolveSearchMode() const {
    const std::string mode = getOptionalString(config_, "search_mode", "pipe");
    if (mode == "beam") {
        return BEAM_SEARCH;
    }
    if (mode == "page") {
        return PAGE_SEARCH;
    }
    if (mode == "pipe") {
        return PIPE_SEARCH;
    }
    throw std::runtime_error("Unsupported search_mode: " + mode);
}

void OdinANNIndex::build(const std::vector<float>& vecs, const std::vector<uint32_t>& ids) {
    if (vecs.empty()) {
        return;
    }
    if (vecs.size() != ids.size() * dim_) {
        throw std::runtime_error("Input vector size does not match dim * ids.size()");
    }
    clearPendingTempFiles();
    pending_dataset_path_ = writeTempFbin(vecs, ids.size());
    pending_identity_tags_ = hasIdentityIds(ids);
    if (!pending_identity_tags_) {
        pending_tag_path_ = writeTagFile(ids, pending_dataset_path_ + ".tags");
        pending_tag_is_temp_ = true;
    }
    pending_dataset_is_temp_ = true;
}

void OdinANNIndex::build(const std::string& dataset_path) {
    uint64_t total_points = 0;
    uint64_t data_dim = 0;
    pipeann::get_bin_metadata(dataset_path, total_points, data_dim);
    if (data_dim != dim_) {
        throw std::runtime_error("Dataset dim does not match config dim");
    }

    clearPendingTempFiles();
    pending_dataset_path_ = dataset_path;
    pending_identity_tags_ = true;
    pending_dataset_is_temp_ = false;
}

void OdinANNIndex::buildDiskIndexToPrefix(const std::string& index_prefix) {
    if (pending_dataset_path_.empty()) {
        throw std::runtime_error("No pending dataset available for build");
    }

    std::filesystem::create_directories(std::filesystem::path(index_prefix).parent_path());
    const std::string params = std::to_string(build_R_) + " " + std::to_string(build_L_) + " " + build_B_ + " " +
                               build_M_ + " " + std::to_string(build_threads_);
    const char* tag_file = pending_identity_tags_ || pending_tag_path_.empty() ? nullptr : pending_tag_path_.c_str();
    if (force_sharded_build_) {
        setenv("PIPEANN_DISABLE_ONESHOT_BUILD", "1", 1);
    }
    if (partition_replication_factor_ > 0) {
        setenv(
            "PIPEANN_PARTITION_REPLICATION_FACTOR",
            std::to_string(partition_replication_factor_).c_str(),
            1);
    }
    const bool ok = pipeann::build_disk_index<float>(
        pending_dataset_path_.c_str(),
        index_prefix.c_str(),
        params.c_str(),
        metric_,
        single_file_index_,
        tag_file);
    if (force_sharded_build_) {
        unsetenv("PIPEANN_DISABLE_ONESHOT_BUILD");
    }
    unsetenv("PIPEANN_PARTITION_REPLICATION_FACTOR");
    if (!ok) {
        throw std::runtime_error("OdinANN build_disk_index failed");
    }

    if (use_mem_index_ && search_mem_L_ > 0) {
        const char* tag_file = pending_identity_tags_ || pending_tag_path_.empty() ? nullptr : pending_tag_path_.c_str();
        buildMemIndexToPrefix(index_prefix, pending_dataset_path_, tag_file);
    }
}

void OdinANNIndex::buildMemIndexToPrefix(
    const std::string& index_prefix,
    const std::string& dataset_path,
    const char* tag_file) const {
    uint64_t total_points = 0;
    uint64_t data_dim = 0;
    pipeann::get_bin_metadata(dataset_path, total_points, data_dim);
    if (data_dim != dim_) {
        throw std::runtime_error("Dataset dim does not match config dim for mem index build");
    }

    pipeann::Parameters mem_params;
    mem_params.Set<unsigned>("L", mem_L_);
    mem_params.Set<unsigned>("R", mem_R_);
    mem_params.Set<unsigned>("C", mem_C_);
    mem_params.Set<float>("alpha", mem_alpha_);
    mem_params.Set<bool>("saturate_graph", false);
    mem_params.Set<unsigned>("num_threads", build_threads_);

    pipeann::Index<float, uint32_t> mem_index(metric_, dim_, total_points, false, false, true);
    if (tag_file == nullptr) {
        mem_index.build(dataset_path.c_str(), total_points, mem_params, static_cast<const char*>(nullptr));
    } else {
        mem_index.build(dataset_path.c_str(), total_points, mem_params, tag_file);
    }
    mem_index.save((index_prefix + "_mem.index").c_str());
}

void OdinANNIndex::ensureMemIndexReady(const std::string& index_path) const {
    if (!use_mem_index_ || search_mem_L_ == 0) {
        return;
    }
    if (std::filesystem::exists(index_path + "_mem.index")) {
        return;
    }
    const auto dataset_it = config_.find("data_path");
    if (dataset_it == config_.end() || dataset_it->second.empty()) {
        throw std::runtime_error("use_mem_index=true but data_path is unavailable to build _mem.index");
    }
    buildMemIndexToPrefix(index_path, dataset_it->second, nullptr);
}

void OdinANNIndex::copyIndexPrefix(const std::string& src_prefix, const std::string& dst_prefix) const {
    std::filesystem::create_directories(std::filesystem::path(dst_prefix).parent_path());
    for (const std::string& suffix : {
             "_disk.index",
             "_disk.index.tags",
             "_pq_pivots.bin",
             "_pq_compressed.bin",
             "_partition.bin.aligned",
             "_mem.index",
             "_mem.index.data",
             "_mem.index.tags"}) {
        const std::filesystem::path src = src_prefix + suffix;
        const std::filesystem::path dst = dst_prefix + suffix;
        if (std::filesystem::exists(src)) {
            std::filesystem::copy_file(src, dst, std::filesystem::copy_options::overwrite_existing);
        }
    }
}

void OdinANNIndex::load(const std::string& index_path) {
    if (!std::filesystem::exists(index_path + "_disk.index")) {
        throw std::runtime_error("OdinANN disk index prefix does not exist: " + index_path);
    }

    ensureMemIndexReady(index_path);

    loaded_base_prefix_ = index_path;
    loaded_shadow_out_prefix_ = index_path + "_shadow2";

    pipeann::Parameters params;
    params.Set<unsigned>("L_disk", merge_L_disk_);
    params.Set<unsigned>("R_disk", merge_R_disk_);
    params.Set<float>("alpha_disk", alpha_disk_);
    params.Set<unsigned>("C", merge_C_);
    params.Set<unsigned>("beamwidth", beamwidth_);
    params.Set<unsigned>("nodes_to_cache", 0);
    params.Set<unsigned>("num_threads", runtime_threads_);

    const bool enable_mem_index = use_mem_index_ && search_mem_L_ > 0 &&
                                  std::filesystem::exists(index_path + "_mem.index");

    runtime_ = std::make_unique<pipeann::DynamicSSDIndex<float, uint32_t>>(
        params,
        index_path,
        loaded_shadow_out_prefix_,
        dist_.get(),
        metric_,
        resolveSearchMode(),
        enable_mem_index);
}

void OdinANNIndex::ensureRuntimeLoaded() const {
    if (runtime_ == nullptr) {
        throw std::runtime_error("OdinANN runtime is not loaded");
    }
}

void OdinANNIndex::insert(const std::vector<float>& vec, const std::vector<uint32_t>& ids) {
    ensureRuntimeLoaded();
    if (vec.size() != ids.size() * dim_) {
        throw std::runtime_error("Insert vector size does not match dim * ids.size()");
    }

    for (size_t i = 0; i < ids.size(); ++i) {
        const int result = runtime_->insert(vec.data() + i * dim_, ids[i]);
        if (result < 0) {
            throw std::runtime_error("OdinANN insert failed for id " + std::to_string(ids[i]));
        }
    }
}

void OdinANNIndex::search(
    const std::vector<float>& query,
    size_t top_k,
    std::vector<uint32_t>& ids,
    std::vector<float>& distances) const {
    ensureRuntimeLoaded();
    if (query.size() != dim_) {
        throw std::runtime_error("Query dim does not match config dim");
    }

    ids.assign(top_k, std::numeric_limits<uint32_t>::max());
    distances.assign(top_k, std::numeric_limits<float>::max());
    pipeann::QueryStats stats {};
    runtime_->search(query.data(), top_k, search_mem_L_, search_L_, beamwidth_, ids.data(), distances.data(), &stats, true);
}

void OdinANNIndex::save(const std::string& index_path) {
    if (!pending_dataset_path_.empty()) {
        buildDiskIndexToPrefix(index_path);
        clearPendingTempFiles();
        loaded_base_prefix_.clear();
        loaded_shadow_out_prefix_.clear();
        runtime_.reset();
        return;
    }

    if (runtime_ != nullptr) {
        runtime_->final_merge();
        copyIndexPrefix(runtime_->_disk_index_prefix_in, index_path);
        return;
    }

    throw std::runtime_error("Nothing to save");
}

std::string OdinANNIndex::getIndexType() const {
    return "OdinANN";
}
