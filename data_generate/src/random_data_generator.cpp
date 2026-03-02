#include "random_data_generator.h"
#include "progress_bar.hpp"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <cmath>

std::vector<std::vector<float>> RandomDataGenerator::generate_vectors(size_t n, size_t seed) {
    std::vector<std::vector<float>> vectors;
    vectors.reserve(n);

    std::cout << "Generating " << n << " vectors of dimension " << config_.dimension << "..." << std::endl;
    
    ProgressBar bar("Generating vectors", n, true, true);
    
    if (config_.distribution == "uniform") {
        std::uniform_real_distribution<float> dist(
            config_.data_min, config_.data_max);
        
        // 预分配内存并批量生成
        vectors.resize(n);
        for (auto& vec : vectors) {
            vec.resize(config_.dimension);
        }
        
        // 并行生成（使用OpenMP）
        #pragma omp parallel for schedule(dynamic, 1000)
        for (int64_t i = 0; i < static_cast<int64_t>(n); ++i) {
            std::mt19937_64 local_rng(seed + i);
            std::uniform_real_distribution<float> local_dist(config_.data_min, config_.data_max);
            
            auto& vec = vectors[i];
            for (size_t j = 0; j < config_.dimension; ++j) {
                vec[j] = local_dist(local_rng);
            }
            
            normalize_vector(vec);
            
            // 更新进度条（仅主线程）
            if (i % 1000 == 0 || i == static_cast<int64_t>(n) - 1) {
                #pragma omp critical
                {
                    bar.set_current(i + 1);
                    bar.display();
                }
            }
        }
    } 
    else if (config_.distribution == "normal") {
        std::normal_distribution<float> dist(0.0f, 1.0f);
        
        // 预分配内存
        vectors.resize(n);
        for (auto& vec : vectors) {
            vec.resize(config_.dimension);
        }
        
        // 并行生成
        #pragma omp parallel for schedule(dynamic, 1000)
        for (int64_t i = 0; i < static_cast<int64_t>(n); ++i) {
            std::mt19937_64 local_rng(seed + i);
            std::normal_distribution<float> local_dist(0.0f, 1.0f);
            
            auto& vec = vectors[i];
            for (size_t j = 0; j < config_.dimension; ++j) {
                vec[j] = local_dist(local_rng);
            }
            
            // 归一化到单位球面
            normalize_vector(vec);
            
            // 更新进度条
            if (i % 1000 == 0 || i == static_cast<int64_t>(n) - 1) {
                #pragma omp critical
                {
                    bar.set_current(i + 1);
                    bar.display();
                }
            }
        }
    }
    else if (config_.distribution == "clustered") {
        // 生成聚类数据
        return generate_clustered_vectors(n, seed);
    }
    
    bar.finish();
    
    return vectors;
}

// 生成聚类数据
std::vector<std::vector<float>> RandomDataGenerator::generate_clustered_vectors(size_t n, size_t seed) {
    std::vector<std::vector<float>> vectors;
    vectors.reserve(n);

    std::cout << "Generating " << n << " clustered vectors..." << std::endl;
    ProgressBar bar("Generating clusters", n, true, true);
    
    // 生成聚类中心
    const size_t num_clusters = std::max(size_t(1), n / 1000);
    std::vector<std::vector<float>> cluster_centers = 
        generate_vectors(num_clusters, seed);
    
    // 预分配内存
    vectors.resize(n);
    for (auto& vec : vectors) {
        vec.resize(config_.dimension);
    }
    
    // 并行生成聚类数据
    #pragma omp parallel for schedule(dynamic, 1000)
    for (int64_t i = 0; i < static_cast<int64_t>(n); ++i) {
        std::mt19937_64 local_rng(config_.seed + i);
        std::normal_distribution<float> local_cluster_dist(0.0f, 0.1f);
        std::uniform_int_distribution<size_t> local_cluster_idx_dist(0, num_clusters-1);
        
        size_t cluster_idx = local_cluster_idx_dist(local_rng);
        auto& vec = vectors[i];
        
        // 在聚类中心周围添加噪声
        for (size_t j = 0; j < config_.dimension; ++j) {
            vec[j] = cluster_centers[cluster_idx][j] + local_cluster_dist(local_rng);
        }
        
        normalize_vector(vec);
        
        // 更新进度条
        if (i % 1000 == 0 || i == static_cast<int64_t>(n) - 1) {
            #pragma omp critical
            {
                bar.set_current(i + 1);
                bar.display();
            }
        }
    }
    
    bar.finish();
    
    return vectors;
}

// 归一化向量（优化版本）
void RandomDataGenerator::normalize_vector(std::vector<float>& vec) {
    float norm = 0.0f;
    
    // 使用SIMD友好的循环展开
    size_t i = 0;
    const size_t unroll = 4;
    const size_t n = vec.size();
    
    for (; i + unroll <= n; i += unroll) {
        norm += vec[i] * vec[i];
        norm += vec[i+1] * vec[i+1];
        norm += vec[i+2] * vec[i+2];
        norm += vec[i+3] * vec[i+3];
    }
    
    // 处理剩余元素
    for (; i < n; ++i) {
        norm += vec[i] * vec[i];
    }
    
    norm = std::sqrt(norm);
    
    if (norm > 0) {
        // 归一化
        for (float& v : vec) {
            v /= norm;
        }
    }
}

// 保存向量到文件
void RandomDataGenerator::save_vectors(const std::string& filename, 
                    const std::vector<std::vector<float>>& vectors) {
    std::ofstream fout(filename, std::ios::binary);
    if (!fout) {
        throw std::runtime_error("Cannot open file: " + filename);
    }
    
    // 写入头信息
    size_t n = vectors.size();
    size_t d = vectors[0].size();
    // bigANN格式：前4字节是向量数量（uint32_t），后4字节是维度（uint32_t）
    if (n > UINT32_MAX) {
        throw std::runtime_error("Number of vectors exceeds uint32_t maximum");
    }
    if (d > UINT32_MAX) {
        throw std::runtime_error("Dimension exceeds uint32_t maximum");
    }
    
    uint32_t n32 = static_cast<uint32_t>(n);
    uint32_t d32 = static_cast<uint32_t>(d);
    
    // 写入头信息（little-endian）
    fout.write(reinterpret_cast<const char*>(&n32), sizeof(uint32_t));
    fout.write(reinterpret_cast<const char*>(&d32), sizeof(uint32_t));
    
    // 验证所有向量维度一致
    for (size_t i = 0; i < n; ++i) {
        if (vectors[i].size() != d) {
            throw std::runtime_error("Inconsistent vector dimensions at index " + std::to_string(i));
        }
    }
    
    // 写入数据
    ProgressBar bar("Saving vectors", n, true, true);
    for (size_t i = 0; i < n; ++i) {
        fout.write(reinterpret_cast<const char*>(vectors[i].data()), d * sizeof(float));
        
        if (i % 10000 == 0 || i == n - 1) {
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
    
    size_t expected_size = sizeof(uint32_t) * 2 + n * d * sizeof(float);
    
    std::cout << "Saved " << n << " vectors to " << filename 
              << " (dim=" << d << ")" << std::endl;
    std::cout << "File size: " << file_size << " bytes" << std::endl;
    std::cout << "Expected size: " << expected_size << " bytes" << std::endl;
    
    if (file_size != expected_size) {
        std::cerr << "Warning: File size mismatch! Expected " << expected_size 
                  << " bytes, got " << file_size << " bytes" << std::endl;
    }
}

// 从文件加载向量
std::vector<std::vector<float>> RandomDataGenerator::load_vectors(const std::string& filename) {
    std::ifstream fin(filename, std::ios::binary);
    if (!fin) {
        throw std::runtime_error("Cannot open file: " + filename);
    }
    
    // 读取头信息
    uint32_t n32, d32;
    fin.read(reinterpret_cast<char*>(&n32), sizeof(uint32_t));
    fin.read(reinterpret_cast<char*>(&d32), sizeof(uint32_t));
    
    size_t n = n32;
    size_t d = d32;
    
    std::cout << "Loading " << n << " vectors (dim=" << d << ") from " << filename << std::endl;
    
    std::vector<std::vector<float>> vectors;
    vectors.reserve(n);
    
    ProgressBar bar("Loading vectors", n, true, true);
    
    std::vector<float> buffer(d);
    for (size_t i = 0; i < n; ++i) {
        fin.read(reinterpret_cast<char*>(buffer.data()), d * sizeof(float));
        
        if (fin.gcount() != static_cast<std::streamsize>(d * sizeof(float))) {
            throw std::runtime_error("Unexpected end of file at vector " + std::to_string(i));
        }
        
        vectors.push_back(buffer);
        
        if (i % 10000 == 0 || i == n - 1) {
            bar.set_current(i + 1);
            bar.display();
        }
    }
    
    bar.finish();
    
    // 检查是否还有多余数据
    fin.peek();
    if (!fin.eof()) {
        std::cerr << "Warning: File contains extra data after reading all vectors" << std::endl;
    }
    
    fin.close();
    return vectors;
}
