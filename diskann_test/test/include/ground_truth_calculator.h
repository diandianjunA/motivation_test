#pragma once
#include "random_data_generator.h"
#include <faiss/IndexIDMap.h>

// Ground Truth计算器
class GroundTruthCalculator {
private:
    int dimension_;
    DistanceMetric metric_;
    faiss::IndexIDMap* index;

    // 辅助函数：格式化时间
    std::string format_time(double ms);
    
public:
    GroundTruthCalculator(int dimension, DistanceMetric metric = DistanceMetric::L2);
    
    // 计算单个查询的ground truth
    struct Neighbor {
        size_t id;
        float distance;

        Neighbor(size_t id, float distance) : id(id), distance(distance) {}
        
        bool operator<(const Neighbor& other) const {
            return distance < other.distance;
        }
    };
    
    void init(const std::vector<std::vector<float>>& dataset);

    std::vector<Neighbor> compute_query_ground_truth(
        const std::vector<float>& query,
        size_t k);
    
    // 批量计算ground truth
    std::vector<std::vector<Neighbor>> compute_all_ground_truth(
        const std::vector<std::vector<float>>& queries,
        size_t k, int num_threads = 8);
    
    // 保存ground truth
    void save_ground_truth(const std::string& filename,
                          const std::vector<std::vector<Neighbor>>& ground_truth);
    
    // 加载ground truth
    std::vector<std::vector<Neighbor>> load_ground_truth(
        const std::string& filename);
};