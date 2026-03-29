#include "ground_truth_calculator.h"
#include <faiss/MetricType.h>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <omp.h>
#include <stdexcept>
#include <vector>
#include "progress_bar.hpp"
#include "faiss/IndexFlat.h"
#include "faiss/IndexIDMap.h"
#include <faiss/gpu/GpuIndexFlat.h>
#include <faiss/gpu/StandardGpuResources.h>
#include <faiss/gpu/impl/IndexUtils.h>
#include <faiss/gpu/utils/DeviceUtils.h>

GroundTruthCalculator::GroundTruthCalculator(
    int dimension,
    DistanceMetric metric,
    bool use_gpu,
    size_t gpu_shard_size,
    int gpu_device)
    : dimension_(dimension)
    , metric_(metric)
    , use_gpu_(use_gpu)
    , gpu_shard_size_(gpu_shard_size)
    , gpu_device_(gpu_device) {
    faiss_metric_ = faiss::MetricType::METRIC_L2;
    if (metric_ == DistanceMetric::L2) {
        faiss_metric_ = faiss::MetricType::METRIC_L2;
    } else if (metric_ == DistanceMetric::INNER_PRODUCT) {
        faiss_metric_ = faiss::MetricType::METRIC_INNER_PRODUCT;
    } else if (metric_ == DistanceMetric::COSINE) {
        faiss_metric_ = faiss::MetricType::METRIC_INNER_PRODUCT;
    }
    if (use_gpu_) {
        const int device_count = faiss::gpu::getNumDevices();
        if (device_count <= 0) {
            throw std::runtime_error("GPU mode requested but no CUDA device is available");
        }
        if (gpu_device_ < 0) {
            gpu_device_ = 0;
        }
        if (gpu_device_ >= device_count) {
            throw std::runtime_error("gpu_device is out of range");
        }

        gpu_resources_ = std::make_unique<faiss::gpu::StandardGpuResources>();
        gpu_resources_->noTempMemory();
        faiss::gpu::GpuIndexFlatConfig config;
        config.device = gpu_device_;
        config.useFloat16 = false;
        gpu_index_ = std::make_unique<faiss::gpu::GpuIndexFlat>(
            gpu_resources_.get(),
            dimension_,
            faiss_metric_,
            config);
    } else {
        index_ = std::make_unique<faiss::IndexIDMap>(
            new faiss::IndexFlat(dimension_, faiss_metric_));
    }
}

GroundTruthCalculator::~GroundTruthCalculator() = default;

size_t GroundTruthCalculator::resolve_gpu_shard_size() const {
    if (gpu_shard_size_ > 0) {
        return gpu_shard_size_;
    }
    constexpr size_t kTargetShardBytes = static_cast<size_t>(24) << 30;
    const size_t bytes_per_vector = dataset_dim_ * sizeof(float);
    return std::max<size_t>(1, kTargetShardBytes / std::max<size_t>(bytes_per_vector, 1));
}

bool GroundTruthCalculator::is_better(float lhs, float rhs) const {
    if (metric_ == DistanceMetric::L2) {
        return lhs < rhs;
    }
    return lhs > rhs;
}

void GroundTruthCalculator::merge_candidate(
    std::vector<Neighbor>& current,
    const Neighbor& candidate,
    size_t k) const {
    if (candidate.id == static_cast<size_t>(-1)) {
        return;
    }
    if (current.size() < k) {
        current.push_back(candidate);
        return;
    }

    size_t worst_index = 0;
    for (size_t i = 1; i < current.size(); ++i) {
        if (is_better(current[worst_index].distance, current[i].distance)) {
            worst_index = i;
        }
    }
    if (is_better(candidate.distance, current[worst_index].distance)) {
        current[worst_index] = candidate;
    }
}

void GroundTruthCalculator::init(const std::vector<std::vector<float>>& dataset) {
    size_t n = dataset.size();
    if (n == 0) return;

    dataset_num_vectors_ = n;
    dataset_dim_ = dataset[0].size();
    if (use_gpu_) {
        owned_dataset_.resize(n * dataset_dim_);
        for (size_t i = 0; i < n; ++i) {
            std::memcpy(
                owned_dataset_.data() + i * dataset_dim_,
                dataset[i].data(),
                dataset_dim_ * sizeof(float));
        }
        dataset_ptr_ = owned_dataset_.data();
        return;
    }

    size_t batch_size = 10000;
    
    ProgressBar bar("Building index", n, true, true);

    std::vector<faiss::idx_t> ids;
    std::vector<float> batch_vectors;
    ids.reserve(batch_size);
    batch_vectors.reserve(batch_size * dataset[0].size());
    
    for (size_t i = 0; i < n; ++i) {
        size_t dim = dataset[i].size();
        
        batch_vectors.insert(batch_vectors.end(), dataset[i].begin(), dataset[i].end());
        ids.push_back(static_cast<faiss::idx_t>(i));
        
        if (batch_vectors.size() >= batch_size * dim || i == n - 1) {
            size_t batch_n = ids.size();
            index_->add_with_ids(batch_n, batch_vectors.data(), ids.data());
            
            batch_vectors.clear();
            ids.clear();
        }
        
        if ((i + 1) % 1000 == 0 || i == n - 1) {
            bar.set_current(i + 1);
            bar.display();
        }
    }

    bar.finish();
}

void GroundTruthCalculator::init(float* dataset, size_t num_vectors, size_t dim) {
    dataset_ptr_ = dataset;
    dataset_num_vectors_ = num_vectors;
    dataset_dim_ = dim;
    if (use_gpu_) {
        owned_dataset_.clear();
        return;
    }

    size_t batch_size = 10000;
    
    ProgressBar bar("Building index", num_vectors, true, true);

    std::vector<faiss::idx_t> ids;
    ids.reserve(batch_size);

    for (size_t i = 0; i < num_vectors; i += batch_size) {
        const size_t batch_n = std::min(batch_size, num_vectors - i);
        
        ids.clear();
        ids.reserve(batch_n);
        for (size_t j = 0; j < batch_n; ++j) {
            ids.push_back(static_cast<faiss::idx_t>(i + j));
        }

        index_->add_with_ids(batch_n, dataset + i * dim, ids.data());
        
        bar.set_current(i + batch_n);
        bar.display();
    }

    bar.finish();
}

std::vector<GroundTruthCalculator::Neighbor> GroundTruthCalculator::compute_query_ground_truth(
    const std::vector<float>& query, size_t k) {
    std::vector<float> query_vec = query;
    std::vector<faiss::idx_t> ids(k);
    std::vector<float> distances(k, 0.0f);
    if (!index_) {
        throw std::runtime_error("single-query ground truth is only supported in CPU mode");
    }
    index_->search(1, query_vec.data(), k, distances.data(), ids.data());
    
    std::vector<Neighbor> neighbors;
    neighbors.reserve(k);
    for (size_t i = 0; i < k; ++i) {
        neighbors.emplace_back(ids[i], distances[i]);
    }
    return neighbors;
}

// 批量计算ground truth
std::vector<std::vector<GroundTruthCalculator::Neighbor>> GroundTruthCalculator::compute_all_ground_truth(
    const std::vector<std::vector<float>>& queries,
    size_t k, int num_threads, size_t batch_size) {
    size_t n_queries = queries.size();
    size_t dim = dimension_;
    if (n_queries == 0) {
        return {};
    }

    std::vector<float> query_buffer(n_queries * dim);
    for (size_t i = 0; i < n_queries; ++i) {
        std::memcpy(
            query_buffer.data() + i * dim,
            queries[i].data(),
            dim * sizeof(float));
    }

    return compute_all_ground_truth(query_buffer.data(), n_queries, dim, k, num_threads, batch_size);
}

std::vector<std::vector<GroundTruthCalculator::Neighbor>> GroundTruthCalculator::compute_all_ground_truth(
    const float* queries,
    size_t num_queries,
    size_t dim,
    size_t k,
    int num_threads,
    size_t batch_size) {
    if (queries == nullptr) {
        throw std::runtime_error("queries pointer is null");
    }
    if (dim != static_cast<size_t>(dimension_)) {
        throw std::runtime_error("query dimension does not match calculator dimension");
    }
    if (batch_size == 0) {
        throw std::runtime_error("batch_size must be positive");
    }
    if (num_threads <= 0) {
        num_threads = omp_get_max_threads();
    }
    omp_set_num_threads(num_threads);

    std::vector<std::vector<Neighbor>> ground_truth(num_queries);
    ProgressBar bar("Computing ground truth", num_queries, true, true);

    const size_t effective_batch_size = std::min(batch_size, num_queries);
    for (auto& neighbors : ground_truth) {
        neighbors.reserve(k);
    }

    if (use_gpu_) {
        if (dataset_ptr_ == nullptr || dataset_num_vectors_ == 0) {
            throw std::runtime_error("GPU ground truth requires dataset to be initialized");
        }

        const size_t shard_size = std::min(resolve_gpu_shard_size(), dataset_num_vectors_);
        const size_t shard_count = (dataset_num_vectors_ + shard_size - 1) / shard_size;
        const size_t total_work = num_queries * shard_count;
        ProgressBar shard_bar("Computing ground truth (GPU shards)", total_work, true, true);
        size_t completed_work = 0;

        std::cout << "GPU exact search with shard_size=" << shard_size
                  << ", shard_count=" << shard_count
                  << ", device=" << gpu_device_ << std::endl;

        for (size_t shard_start = 0; shard_start < dataset_num_vectors_; shard_start += shard_size) {
            const size_t current_shard_size = std::min(shard_size, dataset_num_vectors_ - shard_start);
            gpu_index_->reset();
            gpu_index_->add(
                current_shard_size,
                dataset_ptr_ + shard_start * dataset_dim_);

            std::vector<float> distances(effective_batch_size * k);
            std::vector<faiss::idx_t> ids(effective_batch_size * k);

            for (size_t i = 0; i < num_queries; i += effective_batch_size) {
                const size_t batch_n = std::min(effective_batch_size, num_queries - i);
                gpu_index_->search(
                    batch_n,
                    queries + i * dim,
                    k,
                    distances.data(),
                    ids.data());

                for (size_t query_offset = 0; query_offset < batch_n; ++query_offset) {
                    auto& neighbors = ground_truth[i + query_offset];
                    for (size_t rank = 0; rank < k; ++rank) {
                        const size_t result_index = query_offset * k + rank;
                        const faiss::idx_t local_id = ids[result_index];
                        if (local_id < 0) {
                            continue;
                        }
                        merge_candidate(
                            neighbors,
                            Neighbor(
                                shard_start + static_cast<size_t>(local_id),
                                distances[result_index]),
                            k);
                    }
                }

                completed_work += batch_n;
                shard_bar.set_current(completed_work);
                shard_bar.display();
            }
        }

        for (auto& neighbors : ground_truth) {
            std::sort(
                neighbors.begin(),
                neighbors.end(),
                [&](const Neighbor& lhs, const Neighbor& rhs) {
                    return is_better(lhs.distance, rhs.distance);
                });
        }
        shard_bar.finish();
        return ground_truth;
    }

    std::vector<float> distances(effective_batch_size * k);
    std::vector<faiss::idx_t> ids(effective_batch_size * k);
    for (size_t i = 0; i < num_queries; i += effective_batch_size) {
        const size_t batch_n = std::min(effective_batch_size, num_queries - i);

        index_->search(
            batch_n,
            queries + i * dim,
            k,
            distances.data(),
            ids.data());

        for (size_t j = 0; j < batch_n * k; ++j) {
            ground_truth[i + j / k].emplace_back(ids[j], distances[j]);
        }

        bar.set_current(i + batch_n);
        bar.display();
    }
    
    bar.finish();
    
    return ground_truth;
}

// 保存ground truth
void GroundTruthCalculator::save_ground_truth(const std::string& filename,
                        const std::vector<std::vector<GroundTruthCalculator::Neighbor>>& ground_truth) {
    std::ofstream fout(filename, std::ios::binary);
    if (!fout) {
        throw std::runtime_error("Cannot open file: " + filename);
    }
    
    size_t n_queries = ground_truth.size();
    size_t k = ground_truth.empty() ? 0 : ground_truth[0].size();
    
    // 验证所有查询的k值一致
    for (size_t i = 0; i < n_queries; ++i) {
        if (ground_truth[i].size() != k) {
            throw std::runtime_error("Inconsistent k value at query " + std::to_string(i));
        }
    }
    
    std::cout << "Saving ground truth for " << n_queries 
              << " queries (k=" << k << ") to " << filename << std::endl;
    
    ProgressBar bar("Saving ground truth", n_queries, true, false);

    // 写入num_queries（uint32_t）
    uint32_t n32 = static_cast<uint32_t>(n_queries);
    fout.write(reinterpret_cast<const char*>(&n32), sizeof(uint32_t));

    // 写入k值（uint32_t）
    uint32_t k32 = static_cast<uint32_t>(k);
    fout.write(reinterpret_cast<const char*>(&k32), sizeof(uint32_t));

    
    for (size_t i = 0; i < n_queries; ++i) {        
        // 写入所有邻居ID
        for (size_t j = 0; j < k; ++j) {
            uint32_t id = static_cast<uint32_t>(ground_truth[i][j].id);
            fout.write(reinterpret_cast<const char*>(&id), sizeof(uint32_t));
        }
        
        if (i % 1000 == 0 || i == n_queries - 1) {
            bar.set_current(i + 1);
            bar.display();
        }
    }
    
    bar.finish();
    fout.close();
    
    // 验证文件大小
    std::ifstream fin(filename, std::ios::binary | std::ios::ate);
    size_t file_size = fin.tellg();
    fin.close();
    
    size_t expected_size = n_queries * (sizeof(uint32_t) + k * sizeof(uint32_t));
    
    std::cout << "File size: " << file_size << " bytes" << std::endl;
    std::cout << "Expected size: " << expected_size << " bytes" << std::endl;
    
    if (file_size != expected_size) {
        std::cerr << "Warning: File size mismatch! Expected " << expected_size 
                  << " bytes, got " << file_size << " bytes" << std::endl;
    }
}

// 加载ground truth
std::vector<std::vector<GroundTruthCalculator::Neighbor>> GroundTruthCalculator::load_ground_truth(
    const std::string& filename) {
    
    std::ifstream fin(filename, std::ios::binary);
    if (!fin) {
        throw std::runtime_error("Cannot open file: " + filename);
    }
    
    std::vector<std::vector<Neighbor>> ground_truth;
    
    ProgressBar bar("Loading ground truth", 0, true, false);
    
    // 读取num_queries（uint32_t）
    uint32_t n32;
    fin.read(reinterpret_cast<char*>(&n32), sizeof(uint32_t));
    size_t n_queries = static_cast<size_t>(n32);
    
    // 读取k值
    uint32_t k32;
    fin.read(reinterpret_cast<char*>(&k32), sizeof(uint32_t));

    for (size_t i = 0; i < n_queries; ++i) {
        std::vector<Neighbor> neighbors;
        neighbors.reserve(static_cast<size_t>(k32));
        
        // 读取k个邻居ID
        for (size_t j = 0; j < k32; ++j) {
            uint32_t id;
            fin.read(reinterpret_cast<char*>(&id), sizeof(uint32_t));
            
            if (fin.gcount() != sizeof(uint32_t)) {
                throw std::runtime_error("Error reading neighbor ID for query " 
                                         + std::to_string(i) + ", neighbor " + std::to_string(j));
            }
            
            // 距离设置为0（因为bigANN格式不保存距离）
            neighbors.push_back({static_cast<size_t>(id), 0.0f});
        }
        
        ground_truth.push_back(std::move(neighbors));
        
        if (i % 1000 == 0 || i == n_queries - 1) {
            bar.set_current(i + 1);
            bar.display();
        }
    }
    
    fin.close();
    return ground_truth;
}

// 辅助函数：格式化时间
std::string GroundTruthCalculator::format_time(double ms) {
    if (ms < 1000) {
        return std::to_string(static_cast<int>(ms)) + "ms";
    } else if (ms < 60000) {
        return std::to_string(static_cast<int>(ms / 1000)) + "s";
    } else if (ms < 3600000) {
        int minutes = static_cast<int>(ms / 60000);
        int seconds = static_cast<int>((ms - minutes * 60000) / 1000);
        return std::to_string(minutes) + "m" + std::to_string(seconds) + "s";
    } else {
        int hours = static_cast<int>(ms / 3600000);
        int minutes = static_cast<int>((ms - hours * 3600000) / 60000);
        return std::to_string(hours) + "h" + std::to_string(minutes) + "m";
    }
}
