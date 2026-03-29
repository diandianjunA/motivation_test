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
    std::string preset = "medium";            // hard | medium | easy | custom
    std::string query_mode = "from_base_noise"; // independent | from_base_noise
    float query_noise_std = 0.12f;              // query总扰动强度，会按1/sqrt(dim)换算到每维
    float cluster_noise_std = 0.35f;            // cluster总扰动强度，会按1/sqrt(dim)换算到每维
    bool normalize_queries = true;              // 生成 query 后是否归一化
    size_t ground_truth_batch_size = 64;        // 计算ground truth时的query批量大小
    bool use_gpu = false;                       // 是否使用GPU加速
    size_t gpu_shard_size = 0;                  // GPU分片大小，0表示自动推导
    int gpu_device = -1;                        // GPU设备号，-1表示默认设备
};

void apply_data_preset(DataConfig& config);

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

    // 基于数据库向量生成更易搜索的 query
    std::vector<std::vector<float>> generate_queries_from_database(
        const std::vector<std::vector<float>>& database, size_t n, size_t seed);
    
    // 归一化向量
    void normalize_vector(std::vector<float>& vec);
    
    // 保存向量到文件
    void save_vectors(const std::string& filename, 
                     const std::vector<std::vector<float>>& vectors);
    
    // 从文件加载向量
    std::vector<std::vector<float>> load_vectors(const std::string& filename);
};
