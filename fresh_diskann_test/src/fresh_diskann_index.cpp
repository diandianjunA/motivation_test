#include "fresh_diskann_index.h"

#include "aux_utils.h"
#include "config.h"
#include <filesystem>
#include <fstream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <unistd.h>

namespace {

constexpr uint64_t kFreshDiskANNSectorLen = 8192;

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

diskann::Metric parseMetric(const std::map<std::string, std::string>& config) {
    const auto metric = getOptionalString(config, "metric", "l2");
    if (metric == "l2") {
        return diskann::Metric::L2;
    }
    if (metric == "cosine") {
        return diskann::Metric::COSINE;
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

std::string makeTempDir(const std::string& pattern) {
    std::vector<char> writable(pattern.begin(), pattern.end());
    writable.push_back('\0');
    char* result = mkdtemp(writable.data());
    if (result == nullptr) {
        throw std::runtime_error("Failed to create temporary directory");
    }
    return std::string(result);
}

}  // namespace

FreshDiskANNIndex::FreshDiskANNIndex(std::string conf) : config_path_(std::move(conf)) {
    config_ = readConfig(config_path_);

    dim_ = getRequiredSize(config_, "dim");
    max_points_ = getRequiredSize(config_, "max_points_to_insert");
    build_R_ = static_cast<uint32_t>(getOptionalSize(config_, "build_R", getOptionalSize(config_, "R", 64)));
    build_L_ = static_cast<uint32_t>(getOptionalSize(config_, "build_L", getOptionalSize(config_, "L", 100)));
    build_B_ = getOptionalString(config_, "build_B", "4");
    build_M_ = getOptionalString(config_, "build_M", "8");
    build_threads_ = static_cast<uint32_t>(getOptionalSize(config_, "build_threads", getOptionalSize(config_, "num_threads", 16)));
    search_L_ = static_cast<uint32_t>(getOptionalSize(config_, "search_L", build_L_));
    beamwidth_ = static_cast<uint32_t>(getOptionalSize(config_, "beamwidth", 4));
    num_search_threads_ = static_cast<uint32_t>(getOptionalSize(config_, "num_search_threads", getOptionalSize(config_, "num_threads", 16)));
    nodes_to_cache_ = static_cast<uint32_t>(getOptionalSize(config_, "nodes_to_cache", 0));
    mem_L_ = static_cast<uint32_t>(getOptionalSize(config_, "L_mem", 128));
    mem_R_ = static_cast<uint32_t>(getOptionalSize(config_, "R_mem", 48));
    alpha_mem_ = getOptionalFloat(config_, "alpha_mem", 1.2F);
    disk_L_ = static_cast<uint32_t>(getOptionalSize(config_, "L_disk", build_L_));
    disk_R_ = static_cast<uint32_t>(getOptionalSize(config_, "R_disk", build_R_));
    alpha_disk_ = getOptionalFloat(config_, "alpha_disk", 1.2F);
    build_C_ = static_cast<uint32_t>(getOptionalSize(config_, "C", 75));
    merge_threshold_ = static_cast<uint64_t>(getOptionalSize(config_, "merge_threshold", max_points_ + 1));
    single_file_index_ = getOptionalBool(config_, "single_file_index", false);
    metric_ = parseMetric(config_);

    const uint64_t max_node_len =
        (static_cast<uint64_t>(dim_) * sizeof(float)) + ((static_cast<uint64_t>(build_R_) + 1U) * sizeof(uint32_t));
    if (max_node_len > kFreshDiskANNSectorLen) {
        throw std::runtime_error(
            "FreshDiskANN SSD layout does not support this dim/build_R combination: "
            "max_node_len=" + std::to_string(max_node_len) + " exceeds sector size " +
            std::to_string(kFreshDiskANNSectorLen) +
            ". The baseline stores one full-precision vector per disk node.");
    }

    if (single_file_index_) {
        throw std::runtime_error("FreshDiskANN dynamic test currently expects multi-file disk index");
    }

    if (metric_ == diskann::Metric::COSINE) {
        dist_ = std::make_unique<diskann::DistanceCosineFloat>();
    } else {
        dist_ = std::make_unique<diskann::DistanceL2>();
    }
}

void FreshDiskANNIndex::clearPendingTempFiles() {
    if (pending_dataset_is_temp_ && !pending_dataset_path_.empty()) {
        std::filesystem::remove(pending_dataset_path_);
    }
    if (pending_tag_is_temp_ && !pending_tag_path_.empty()) {
        std::filesystem::remove(pending_tag_path_);
    }
    pending_dataset_path_.clear();
    pending_tag_path_.clear();
    pending_dataset_is_temp_ = false;
    pending_tag_is_temp_ = false;
}

std::string FreshDiskANNIndex::writeTempFbin(const std::vector<float>& vecs, size_t count) const {
    std::filesystem::create_directories("/tmp");
    const std::string path = makeTempPath("/tmp/fresh_diskann_data_XXXXXX");
    std::ofstream out(path, std::ios::binary);
    const uint32_t n = static_cast<uint32_t>(count);
    const uint32_t d = static_cast<uint32_t>(dim_);
    out.write(reinterpret_cast<const char*>(&n), sizeof(uint32_t));
    out.write(reinterpret_cast<const char*>(&d), sizeof(uint32_t));
    out.write(reinterpret_cast<const char*>(vecs.data()), static_cast<std::streamsize>(vecs.size() * sizeof(float)));
    return path;
}

std::string FreshDiskANNIndex::writeTagFile(const std::vector<uint32_t>& ids, const std::string& path) const {
    std::ofstream out(path, std::ios::binary);
    const uint32_t n = static_cast<uint32_t>(ids.size());
    const uint32_t d = 1;
    out.write(reinterpret_cast<const char*>(&n), sizeof(uint32_t));
    out.write(reinterpret_cast<const char*>(&d), sizeof(uint32_t));
    out.write(reinterpret_cast<const char*>(ids.data()), static_cast<std::streamsize>(ids.size() * sizeof(uint32_t)));
    return path;
}

void FreshDiskANNIndex::populateMergeParameters(diskann::Parameters& params) const {
    params.Set<unsigned>("L_mem", mem_L_);
    params.Set<unsigned>("R_mem", mem_R_);
    params.Set<float>("alpha_mem", alpha_mem_);
    params.Set<unsigned>("L_disk", disk_L_);
    params.Set<unsigned>("R_disk", disk_R_);
    params.Set<float>("alpha_disk", alpha_disk_);
    params.Set<unsigned>("C", build_C_);
    params.Set<unsigned>("nodes_to_cache", nodes_to_cache_);
    params.Set<unsigned>("num_search_threads", num_search_threads_);
    params.Set<unsigned>("beamwidth", beamwidth_);
}

std::string FreshDiskANNIndex::makeBuildParameterString() const {
    return std::to_string(build_R_) + " " + std::to_string(build_L_) + " " + build_B_ + " " + build_M_ + " " +
           std::to_string(build_threads_);
}

void FreshDiskANNIndex::build(const std::vector<float>& vecs, const std::vector<uint32_t>& ids) {
    if (vecs.empty()) {
        return;
    }
    if (vecs.size() != ids.size() * dim_) {
        throw std::runtime_error("Input vector size does not match dim * ids.size()");
    }
    clearPendingTempFiles();
    pending_dataset_path_ = writeTempFbin(vecs, ids.size());
    pending_tag_path_ = writeTagFile(ids, pending_dataset_path_ + ".tags");
    pending_dataset_is_temp_ = true;
    pending_tag_is_temp_ = true;
}

void FreshDiskANNIndex::build(const std::string& dataset_path) {
    uint64_t total_points = 0;
    uint64_t data_dim = 0;
    diskann::get_bin_metadata(dataset_path, total_points, data_dim);
    if (data_dim != dim_) {
        throw std::runtime_error("Dataset dim does not match config dim");
    }
    clearPendingTempFiles();
    pending_dataset_path_ = dataset_path;
    std::vector<uint32_t> ids(static_cast<size_t>(total_points));
    std::iota(ids.begin(), ids.end(), 0U);
    pending_tag_path_ = writeTagFile(ids, makeTempPath("/tmp/fresh_diskann_tags_XXXXXX"));
    pending_dataset_is_temp_ = false;
    pending_tag_is_temp_ = true;
}

void FreshDiskANNIndex::buildDiskIndexToPrefix(const std::string& index_prefix) {
    if (pending_dataset_path_.empty() || pending_tag_path_.empty()) {
        throw std::runtime_error("No pending dataset available for build");
    }
    std::filesystem::create_directories(std::filesystem::path(index_prefix).parent_path());
    const std::string params = makeBuildParameterString();
    const bool ok = diskann::build_disk_index<float>(
        pending_dataset_path_.c_str(),
        index_prefix.c_str(),
        params.c_str(),
        metric_,
        single_file_index_,
        pending_tag_path_.c_str());
    if (!ok) {
        throw std::runtime_error("FreshDiskANN build_disk_index failed");
    }
}

void FreshDiskANNIndex::copyIndexPrefix(const std::string& src_prefix, const std::string& dst_prefix) const {
    std::filesystem::create_directories(std::filesystem::path(dst_prefix).parent_path());
    for (const std::string& suffix : {"_disk.index", "_disk.index.tags", "_pq_pivots.bin", "_pq_compressed.bin", "_sample_data.bin"}) {
        const std::filesystem::path src = src_prefix + suffix;
        const std::filesystem::path dst = dst_prefix + suffix;
        if (std::filesystem::exists(src)) {
            std::filesystem::copy_file(src, dst, std::filesystem::copy_options::overwrite_existing);
        }
    }
}

std::string FreshDiskANNIndex::activeIndexPrefix() const {
    if (runtime_ == nullptr) {
        return loaded_base_prefix_;
    }
    return runtime_->ret_merge_prefix();
}

void FreshDiskANNIndex::load(const std::string& index_path) {
    if (!std::filesystem::exists(index_path + "_disk.index")) {
        throw std::runtime_error("FreshDiskANN disk index prefix does not exist: " + index_path);
    }
    loaded_base_prefix_ = index_path;
    loaded_shadow_out_prefix_ = index_path + "_shadow2";
    diskann::Parameters params;
    populateMergeParameters(params);
    runtime_ = std::make_unique<diskann::MergeInsert<float, uint32_t>>(
        params,
        dim_,
        index_path + "_mem",
        index_path,
        loaded_shadow_out_prefix_,
        dist_.get(),
        metric_,
        single_file_index_,
        std::filesystem::path(index_path).parent_path().string(),
        true,
        false,
        merge_threshold_);
}

void FreshDiskANNIndex::ensureRuntimeLoaded() const {
    if (runtime_ == nullptr) {
        throw std::runtime_error("FreshDiskANN runtime is not loaded");
    }
}

void FreshDiskANNIndex::insert(const std::vector<float>& vec, const std::vector<uint32_t>& ids) {
    ensureRuntimeLoaded();
    if (vec.size() != ids.size() * dim_) {
        throw std::runtime_error("Insert vector size does not match dim * ids.size()");
    }
    for (size_t i = 0; i < ids.size(); ++i) {
        const int result = runtime_->insert(vec.data() + i * dim_, ids[i]);
        if (result != 0) {
            throw std::runtime_error("FreshDiskANN insert failed for id " + std::to_string(ids[i]));
        }
    }
}

void FreshDiskANNIndex::search(
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
    diskann::QueryStats stats;
    runtime_->search_sync(query.data(), top_k, search_L_, ids.data(), distances.data(), &stats);
}

void FreshDiskANNIndex::save(const std::string& index_path) {
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
        copyIndexPrefix(activeIndexPrefix(), index_path);
        return;
    }

    throw std::runtime_error("Nothing to save");
}

std::string FreshDiskANNIndex::getIndexType() const {
    return "FreshDiskANN";
}
