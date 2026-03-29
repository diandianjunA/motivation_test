#pragma once
#include "random_data_generator.h"
#include <faiss/Index.h>
#include <faiss/MetricType.h>
#include <memory>

namespace faiss {
namespace gpu {
class StandardGpuResources;
class GpuIndexFlat;
} // namespace gpu
} // namespace faiss

// Ground Truth计算器
class GroundTruthCalculator {
public:
    // 计算单个查询的ground truth
    struct Neighbor {
        size_t id;
        float distance;

        Neighbor(size_t id, float distance) : id(id), distance(distance) {}
        
        bool operator<(const Neighbor& other) const {
            return distance < other.distance;
        }
    };

private:
    int dimension_;
    DistanceMetric metric_;
    bool use_gpu_;
    size_t gpu_shard_size_;
    int gpu_device_;
    faiss::MetricType faiss_metric_;
    std::unique_ptr<faiss::Index> index_;
    std::unique_ptr<faiss::gpu::StandardGpuResources> gpu_resources_;
    std::unique_ptr<faiss::gpu::GpuIndexFlat> gpu_index_;
    std::vector<float> owned_dataset_;
    const float* dataset_ptr_ = nullptr;
    size_t dataset_num_vectors_ = 0;
    size_t dataset_dim_ = 0;

    // 辅助函数：格式化时间
    std::string format_time(double ms);

    size_t resolve_gpu_shard_size() const;

    bool is_better(float lhs, float rhs) const;

    void merge_candidate(std::vector<Neighbor>& current, const Neighbor& candidate, size_t k) const;

public:
    GroundTruthCalculator(
        int dimension,
        DistanceMetric metric = DistanceMetric::L2,
        bool use_gpu = false,
        size_t gpu_shard_size = 0,
        int gpu_device = -1);

    ~GroundTruthCalculator();
    
    void init(const std::vector<std::vector<float>>& dataset);

    void init(float* dataset, size_t num_vectors, size_t dim);

    std::vector<Neighbor> compute_query_ground_truth(
        const std::vector<float>& query,
        size_t k);
    
    // 批量计算ground truth
    std::vector<std::vector<Neighbor>> compute_all_ground_truth(
        const std::vector<std::vector<float>>& queries,
        size_t k, int num_threads = 8, size_t batch_size = 1024);

    std::vector<std::vector<Neighbor>> compute_all_ground_truth(
        const float* queries,
        size_t num_queries,
        size_t dim,
        size_t k,
        int num_threads = 8,
        size_t batch_size = 1024);
    
    // 保存ground truth
    void save_ground_truth(const std::string& filename,
                          const std::vector<std::vector<Neighbor>>& ground_truth);
    
    // 加载ground truth
    std::vector<std::vector<Neighbor>> load_ground_truth(
        const std::string& filename);
};
