#include "shine_gpu_index.h"

#include "vector_test/config.h"
#include "vector_test/util.h"

#include <algorithm>
#include <sstream>
#include <stdexcept>

namespace {

bool is_truthy(const std::string& value) {
    return value == "1" || value == "true" || value == "on" || value == "yes";
}

std::vector<std::string> split_tokens(const std::string& value) {
    std::string normalized = value;
    std::replace(normalized.begin(), normalized.end(), ',', ' ');

    std::stringstream ss(normalized);
    std::vector<std::string> tokens;
    std::string token;
    while (ss >> token) {
        tokens.push_back(token);
    }
    return tokens;
}

std::vector<char*> make_argv(std::vector<std::string>& args) {
    std::vector<char*> argv;
    argv.reserve(args.size());
    for (auto& arg : args) {
        argv.push_back(arg.data());
    }
    return argv;
}

}  // namespace

ShineGpuIndex::ShineGpuIndex(const std::string& service_config_path) {
    auto args = build_service_argv(service_config_path);
    auto argv = make_argv(args);
    configuration::IndexConfiguration config(static_cast<int>(argv.size()), argv.data());
    service_ = std::make_unique<GpuComputeService>(config, false);
}

ShineGpuIndex::~ShineGpuIndex() = default;

void ShineGpuIndex::build(const std::vector<float>& vecs, const std::vector<uint32_t>& ids) {
    insert(vecs, ids);
}

void ShineGpuIndex::build(const std::string& dataset_path) {
    auto [data, info] = read_fbin(dataset_path);
    const size_t num_vectors = info.first;
    const size_t vector_dim = info.second;

    if (vector_dim != dim()) {
        delete[] data;
        throw std::runtime_error("dataset dimension does not match service configuration");
    }

    std::vector<float> vectors(data, data + num_vectors * vector_dim);
    std::vector<uint32_t> ids(num_vectors);
    for (size_t i = 0; i < num_vectors; ++i) {
        ids[i] = static_cast<uint32_t>(i);
    }
    delete[] data;

    build(vectors, ids);
}

void ShineGpuIndex::insert(const std::vector<float>& vectors, const std::vector<uint32_t>& ids) {
    if (ids.empty()) {
        return;
    }
    if (vectors.size() != ids.size() * dim()) {
        throw std::runtime_error("insert vector count/dimension mismatch");
    }

    constexpr size_t batch_size = 100;
    for (size_t offset = 0; offset < ids.size(); offset += batch_size) {
        const size_t end = std::min(offset + batch_size, ids.size());
        vec<GpuComputeService::InsertItem> batch;
        batch.reserve(end - offset);

        for (size_t i = offset; i < end; ++i) {
            const auto begin = vectors.begin() + static_cast<std::ptrdiff_t>(i * dim());
            const auto finish = begin + static_cast<std::ptrdiff_t>(dim());
            batch.push_back({ids[i], vec<element_t>(begin, finish)});
        }

        service_->insert(batch);
    }
}

void ShineGpuIndex::search(const std::vector<float>& query,
                           size_t top_k,
                           std::vector<uint32_t>& ids,
                           std::vector<float>& distances) const {
    if (query.size() != dim()) {
        throw std::runtime_error("search dimension mismatch");
    }

    const auto results = service_->search(query, static_cast<u32>(top_k));
    ids.assign(results.begin(), results.end());
    distances.assign(ids.size(), 0.0f);
}

void ShineGpuIndex::load(const std::string& index_path) {
    str error_message;
    if (!service_->load_index(index_path, &error_message)) {
        throw std::runtime_error("failed to load index: " + error_message);
    }
}

void ShineGpuIndex::save(const std::string& index_path) {
    str error_message;
    if (!service_->store_index(index_path, &error_message)) {
        throw std::runtime_error("failed to save index: " + error_message);
    }
}

std::string ShineGpuIndex::getIndexType() const {
    return "ShineGPU";
}

size_t ShineGpuIndex::dim() const {
    return service_->config().dim;
}

std::vector<std::string> ShineGpuIndex::build_service_argv(const std::string& service_config_path) {
    const auto config = readConfig(service_config_path);
    std::vector<std::string> args;
    args.emplace_back("shine_gpu_vector_test");

    static const std::vector<std::string> multi_keys = {"servers", "clients"};
    static const std::vector<std::string> flag_keys = {
        "initiator",
        "cache",
        "routing",
        "load-index",
        "store-index",
        "disable-thread-pinning",
        "no-recall",
        "ip-dist",
    };

    for (const auto& [key, value] : config) {
        const std::string option = "--" + key;
        if (std::find(flag_keys.begin(), flag_keys.end(), key) != flag_keys.end()) {
            if (is_truthy(value)) {
                args.push_back(option);
            }
            continue;
        }

        if (std::find(multi_keys.begin(), multi_keys.end(), key) != multi_keys.end()) {
            const auto tokens = split_tokens(value);
            if (!tokens.empty()) {
                args.push_back(option);
                args.insert(args.end(), tokens.begin(), tokens.end());
            }
            continue;
        }

        args.push_back(option);
        args.push_back(value);
    }

    return args;
}
