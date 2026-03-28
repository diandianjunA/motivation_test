#pragma once

#include "hnswlib/hnswlib.h"
#include "progress_bar.hpp"
#include "vector_index.h"
#include <atomic>
#include <map>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <string>

class HNSWIndex : public VectorIndex {
public:
    explicit HNSWIndex(std::string conf);
    ~HNSWIndex() override = default;

    void build(const std::vector<float>& vecs, const std::vector<uint32_t>& ids) override;
    void build(const std::string& dataset_path) override;
    void insert(const std::vector<float>& vec, const std::vector<uint32_t>& ids) override;
    void search(const std::vector<float>& query, size_t top_k, std::vector<uint32_t>& ids, std::vector<float>& distances) const override;
    void load(const std::string& index_path) override;
    void save(const std::string& index_path) override;
    std::string getIndexType() const override;

private:
    void createIndex(size_t max_elements);
    void ensureCapacity(size_t required_elements);
    void addPoints(const float* data, size_t count, const std::vector<uint32_t>& ids);
    void addPointsRange(const float* data, size_t count, uint32_t first_id);
    void addPointsWithProgress(
        const float* data,
        size_t count,
        const std::vector<uint32_t>& ids,
        ProgressBar& progress_bar,
        size_t base_processed,
        const std::string& stage);
    void addPointsRangeWithProgress(
        const float* data,
        size_t count,
        uint32_t first_id,
        ProgressBar& progress_bar,
        size_t base_processed,
        const std::string& stage);

    std::string config_path_;
    std::map<std::string, std::string> config_;
    int dim_ = 0;
    size_t max_elements_ = 0;
    size_t m_ = 16;
    size_t ef_construction_ = 200;
    size_t ef_search_ = 50;
    size_t num_threads_ = 1;
    size_t random_seed_ = 100;
    size_t seed_build_size_ = 100000;
    size_t batch_size_ = 100000;
    bool allow_replace_deleted_ = false;

    std::unique_ptr<hnswlib::L2Space> space_;
    std::unique_ptr<hnswlib::HierarchicalNSW<float>> index_;
    mutable std::mutex resize_mutex_;
    mutable std::shared_mutex index_rw_mutex_;
};
