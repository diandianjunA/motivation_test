#include "ground_truth_calculator.h"
#include <faiss/MetricType.h>
#include <iostream>
#include <fstream>
#include <vector>
#include "progress_bar.hpp"
#include "faiss/IndexFlat.h"
#include "faiss/IndexIDMap.h"

GroundTruthCalculator::GroundTruthCalculator(int dimension, DistanceMetric metric)
    : dimension_(dimension), metric_(metric) {
    faiss::MetricType faiss_metric = faiss::MetricType::METRIC_L2;
    if (metric_ == DistanceMetric::L2) {
        faiss_metric = faiss::MetricType::METRIC_L2;
    } else if (metric_ == DistanceMetric::INNER_PRODUCT) {
        faiss_metric = faiss::MetricType::METRIC_INNER_PRODUCT;
    } else if (metric_ == DistanceMetric::COSINE) {
        faiss_metric = faiss::MetricType::METRIC_INNER_PRODUCT;
    }
    index = new faiss::IndexIDMap(new faiss::IndexFlat(dimension, faiss_metric));
}

void GroundTruthCalculator::init(const std::vector<std::vector<float>>& dataset) {
    size_t n = dataset.size();
    if (n == 0) return;

    size_t batch_size = 10000;
    
    ProgressBar bar("Building index", n, true, true);

    std::vector<faiss::idx_t> ids;
    std::vector<float> batch_vectors;
    
    for (size_t i = 0; i < n; ++i) {
        size_t dim = dataset[i].size();
        
        batch_vectors.insert(batch_vectors.end(), dataset[i].begin(), dataset[i].end());
        ids.push_back(static_cast<faiss::idx_t>(i));
        
        if (batch_vectors.size() >= batch_size * dim || i == n - 1) {
            size_t batch_n = ids.size();
            index->add_with_ids(batch_n, batch_vectors.data(), ids.data());
            
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

std::vector<GroundTruthCalculator::Neighbor> GroundTruthCalculator::compute_query_ground_truth(
    const std::vector<float>& query, size_t k) {
    std::vector<float> query_vec = query;
    std::vector<faiss::idx_t> ids(k);
    std::vector<float> distances(k, 0.0f);
    index->search(1, query_vec.data(), k, distances.data(), ids.data());
    
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
    size_t k, int num_threads) {
    std::vector<std::vector<Neighbor>> ground_truth;
    ground_truth.reserve(queries.size());
    
    ProgressBar bar("Computing ground truth", queries.size(), true, true);
    
    #pragma omp parallel for num_threads(num_threads) schedule(dynamic, 1000)
    for (int64_t i = 0; i < static_cast<int64_t>(queries.size()); ++i) {
        ground_truth.push_back(compute_query_ground_truth(queries[i], k));
        
        if (i % 10 == 0 || i == static_cast<int64_t>(queries.size()) - 1) {
            #pragma omp critical
            {
                bar.set_current(i + 1);
                bar.display();
            }
        }
    }
    
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
