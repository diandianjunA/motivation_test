#pragma once
#include <cstddef>
#include <vector>
#include <random>
#include <cmath>
#include <string>
#include <cassert>

// 配置结构体
struct DataConfig {
    size_t num_vectors;      // 数据库向量数量
    size_t num_queries;      // 查询向量数量
    size_t dimension;        // 向量维度
    size_t top_k;            // 返回的top-k数量
    float data_min;          // 数据最小值
    float data_max;          // 数据最大值
    int seed;                // 随机种子
    std::string distribution; // 数据分布类型
    std::string output_dir;  // 输出目录
};

// 距离度量类型
enum class DistanceMetric {
    L2,      // 欧氏距离
    INNER_PRODUCT, // 内积
    COSINE   // 余弦相似度
};

// 数据生成器
class RandomDataGenerator {
private:
    std::mt19937_64 rng_;
    DataConfig config_;
    
public:
    RandomDataGenerator(const DataConfig& config) 
        : rng_(config.seed), config_(config) {
    }
    
    // 生成随机向量
    std::vector<std::vector<float>> generate_vectors(size_t n, size_t seed);
    
    // 生成聚类数据
    std::vector<std::vector<float>> generate_clustered_vectors(size_t n, size_t seed);
    
    // 归一化向量
    void normalize_vector(std::vector<float>& vec);
    
    // 保存向量到文件
    void save_vectors(const std::string& filename, 
                     const std::vector<std::vector<float>>& vectors);
    
    // 从文件加载向量
    std::vector<std::vector<float>> load_vectors(const std::string& filename);
};