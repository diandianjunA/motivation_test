#include "jasper_index.h"

#include "config.h"
#include "jasper/jasper.cuh"
#include "util.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>

struct JasperIndex::RuntimeBase {
    virtual ~RuntimeBase() = default;

    virtual void build(const float* vectors, uint64_t count, const jasper::BuildParams& params) = 0;
    virtual void load(const std::string& index_path, uint64_t count) = 0;
    virtual void save(const std::string& index_path) const = 0;
    virtual void search(
        const std::vector<float>& query,
        size_t top_k,
        std::vector<uint32_t>& ids,
        std::vector<float>& distances,
        const jasper::SearchParams& params) const = 0;
    virtual void exportVectors(std::vector<float>& vectors) const = 0;
};

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

double getOptionalDouble(
    const std::map<std::string, std::string>& config,
    const std::string& key,
    double default_value) {
    const auto it = config.find(key);
    if (it == config.end()) {
        return default_value;
    }
    return std::stod(it->second);
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

constexpr char kMetadataMagic[] = "JASPMETA";
constexpr uint32_t kMetadataVersion = 1;

struct MetadataHeader {
    char magic[8];
    uint32_t version;
    uint32_t dim;
    uint64_t count;
};

template <size_t Dim>
class JasperRuntimeImpl final : public JasperIndex::RuntimeBase {
public:
    void build(const float* vectors, uint64_t count, const jasper::BuildParams& params) override {
        index_.build(vectors, count, params);
    }

    void load(const std::string& index_path, uint64_t count) override {
        index_.load(index_path, count);
    }

    void save(const std::string& index_path) const override {
        index_.save(index_path);
    }

    void search(
        const std::vector<float>& query,
        size_t top_k,
        std::vector<uint32_t>& ids,
        std::vector<float>& distances,
        const jasper::SearchParams& params) const override {
        ids.assign(top_k, std::numeric_limits<uint32_t>::max());
        distances.assign(top_k, std::numeric_limits<float>::max());
        index_.search(query.data(), 1, static_cast<uint32_t>(top_k), ids.data(), distances.data(), params);
    }

    void exportVectors(std::vector<float>& vectors) const override {
        const auto* raw = index_.raw();
        if (raw == nullptr) {
            throw std::runtime_error("Jasper runtime has no loaded raw index");
        }

        vectors.resize(static_cast<size_t>(index_.size()) * Dim);
        cudaError_t err = cudaMemcpy(
            vectors.data(),
            raw->vectors,
            sizeof(float) * vectors.size(),
            cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            throw std::runtime_error("Failed to copy Jasper vectors from device: " + std::string(cudaGetErrorString(err)));
        }
    }

private:
    jasper::JasperIndex<Dim, float> index_;
};

std::unique_ptr<JasperIndex::RuntimeBase> createRuntime(size_t dim) {
    switch (dim) {
        case 1024:
            return std::make_unique<JasperRuntimeImpl<1024>>();
        default:
            throw std::runtime_error(
                "jasper_test currently supports dim = 1024; got " + std::to_string(dim));
    }
}

}  // namespace
JasperIndex::JasperIndex(std::string conf) : config_path_(std::move(conf)) {
    config_ = readConfig(config_path_);

    dim_ = getRequiredSize(config_, "dim");
    build_n_rounds_ = static_cast<uint32_t>(getOptionalSize(config_, "build_n_rounds", 1));
    build_nodes_explored_per_iteration_ =
        static_cast<uint32_t>(getOptionalSize(config_, "build_nodes_explored_per_iteration", 4));
    build_random_init_ = getOptionalBool(config_, "build_random_init", false);
    build_alpha_ = getOptionalDouble(config_, "build_alpha", 1.2);
    build_max_batch_ratio_ = getOptionalDouble(config_, "build_max_batch_ratio", 0.02);

    search_beam_width_ = static_cast<uint32_t>(getOptionalSize(config_, "search_beam_width", 64));
    search_cut_ = static_cast<float>(getOptionalDouble(config_, "search_cut", 10.0));
    search_limit_ = static_cast<uint32_t>(getOptionalSize(config_, "search_limit", 512));

    runtime_ = createRuntime(dim_);
}

JasperIndex::~JasperIndex() = default;

void JasperIndex::build(const std::vector<float>& vecs, const std::vector<uint32_t>& ids) {
    if (vecs.empty()) {
        host_vectors_.clear();
        external_ids_.clear();
        return;
    }
    if (vecs.size() != ids.size() * dim_) {
        throw std::runtime_error("Input vector size does not match dim * ids.size()");
    }

    host_vectors_ = vecs;
    external_ids_ = ids;
    rebuildIndex();
}

void JasperIndex::build(const std::string& dataset_path) {
    auto [data, info] = read_fbin(dataset_path);
    if (info.second != dim_) {
        delete[] data;
        throw std::runtime_error("Dataset dim does not match Jasper config dim");
    }

    host_vectors_.assign(data, data + info.first * info.second);
    delete[] data;

    external_ids_.resize(info.first);
    std::iota(external_ids_.begin(), external_ids_.end(), 0U);
    rebuildIndex();
}

void JasperIndex::insert(const std::vector<float>& vec, const std::vector<uint32_t>& ids) {
    if (vec.empty()) {
        return;
    }
    if (vec.size() != ids.size() * dim_) {
        throw std::runtime_error("Insert vector size does not match dim * ids.size()");
    }

    host_vectors_.insert(host_vectors_.end(), vec.begin(), vec.end());
    external_ids_.insert(external_ids_.end(), ids.begin(), ids.end());
    rebuildIndex();
}

void JasperIndex::search(
    const std::vector<float>& query,
    size_t top_k,
    std::vector<uint32_t>& ids,
    std::vector<float>& distances) const {
    ensureRuntime();
    if (query.size() != dim_) {
        throw std::runtime_error("Query dim does not match Jasper config dim");
    }

    jasper::SearchParams params;
    params.beam_width = search_beam_width_;
    params.cut = search_cut_;
    params.limit = search_limit_;

    std::vector<uint32_t> internal_ids;
    runtime_->search(query, top_k, internal_ids, distances, params);

    ids.resize(top_k, std::numeric_limits<uint32_t>::max());
    for (size_t i = 0; i < top_k; ++i) {
        const uint32_t internal_id = internal_ids[i];
        if (internal_id < external_ids_.size()) {
            ids[i] = external_ids_[internal_id];
        }
    }
}

void JasperIndex::load(const std::string& index_path) {
    loadMetadata(index_path);

    runtime_ = createRuntime(dim_);
    runtime_->load(index_path, external_ids_.size());
    runtime_->exportVectors(host_vectors_);
}

void JasperIndex::save(const std::string& index_path) {
    ensureRuntime();
    const auto parent = std::filesystem::path(index_path).parent_path();
    if (!parent.empty()) {
        std::filesystem::create_directories(parent);
    }
    runtime_->save(index_path);
    saveMetadata(index_path);
}

std::string JasperIndex::getIndexType() const {
    return "Jasper";
}

void JasperIndex::rebuildIndex() {
    if (external_ids_.empty()) {
        runtime_.reset();
        return;
    }

    jasper::BuildParams params;
    params.n_rounds = build_n_rounds_;
    params.nodes_explored_per_iteration = build_nodes_explored_per_iteration_;
    params.random_init = build_random_init_;
    params.alpha = build_alpha_;
    params.max_batch_ratio = build_max_batch_ratio_;

    runtime_ = createRuntime(dim_);
    runtime_->build(host_vectors_.data(), external_ids_.size(), params);
}

void JasperIndex::ensureRuntime() const {
    if (!runtime_) {
        throw std::runtime_error("Jasper runtime is not built or loaded");
    }
}

void JasperIndex::loadMetadata(const std::string& index_path) {
    std::ifstream input(index_path + ".meta", std::ios::binary);
    if (!input.is_open()) {
        throw std::runtime_error("Failed to open Jasper metadata file: " + index_path + ".meta");
    }

    MetadataHeader header{};
    input.read(reinterpret_cast<char*>(&header), sizeof(header));
    if (!input) {
        throw std::runtime_error("Failed to read Jasper metadata header");
    }
    if (std::string(header.magic, sizeof(header.magic)) != std::string(kMetadataMagic, sizeof(header.magic))) {
        throw std::runtime_error("Invalid Jasper metadata magic");
    }
    if (header.version != kMetadataVersion) {
        throw std::runtime_error("Unsupported Jasper metadata version: " + std::to_string(header.version));
    }
    if (header.dim != dim_) {
        throw std::runtime_error("Jasper metadata dim does not match config dim");
    }

    external_ids_.resize(static_cast<size_t>(header.count));
    input.read(reinterpret_cast<char*>(external_ids_.data()), static_cast<std::streamsize>(external_ids_.size() * sizeof(uint32_t)));
    if (!input) {
        throw std::runtime_error("Failed to read Jasper metadata ids");
    }
}

void JasperIndex::saveMetadata(const std::string& index_path) const {
    MetadataHeader header{};
    std::copy_n(kMetadataMagic, sizeof(header.magic), header.magic);
    header.version = kMetadataVersion;
    header.dim = static_cast<uint32_t>(dim_);
    header.count = external_ids_.size();

    std::ofstream output(index_path + ".meta", std::ios::binary | std::ios::trunc);
    if (!output.is_open()) {
        throw std::runtime_error("Failed to open Jasper metadata file for write: " + index_path + ".meta");
    }

    output.write(reinterpret_cast<const char*>(&header), sizeof(header));
    output.write(
        reinterpret_cast<const char*>(external_ids_.data()),
        static_cast<std::streamsize>(external_ids_.size() * sizeof(uint32_t)));
    if (!output) {
        throw std::runtime_error("Failed to write Jasper metadata");
    }
}
