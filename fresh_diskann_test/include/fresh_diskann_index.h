#pragma once

#include "distance.h"
#include "vector_index.h"
#include "v2/merge_insert.h"
#include <map>
#include <memory>
#include <string>

class FreshDiskANNIndex : public VectorIndex {
public:
    explicit FreshDiskANNIndex(std::string conf);
    ~FreshDiskANNIndex() override = default;

    void build(const std::vector<float>& vecs, const std::vector<uint32_t>& ids) override;
    void build(const std::string& dataset_path) override;
    void insert(const std::vector<float>& vec, const std::vector<uint32_t>& ids) override;
    void search(const std::vector<float>& query, size_t top_k, std::vector<uint32_t>& ids, std::vector<float>& distances) const override;
    void load(const std::string& index_path) override;
    void save(const std::string& index_path) override;
    std::string getIndexType() const override;

private:
    void clearPendingTempFiles();
    void ensureRuntimeLoaded() const;
    void buildDiskIndexToPrefix(const std::string& index_prefix);
    void populateMergeParameters(diskann::Parameters& params) const;
    std::string makeBuildParameterString() const;
    std::string writeTempFbin(const std::vector<float>& vecs, size_t count) const;
    std::string writeTagFile(const std::vector<uint32_t>& ids, const std::string& path) const;
    void copyIndexPrefix(const std::string& src_prefix, const std::string& dst_prefix) const;
    std::string activeIndexPrefix() const;

    std::string config_path_;
    std::map<std::string, std::string> config_;
    size_t dim_ = 0;
    size_t max_points_ = 0;
    uint32_t build_R_ = 64;
    uint32_t build_L_ = 100;
    std::string build_B_ = "4";
    std::string build_M_ = "8";
    uint32_t build_threads_ = 16;
    uint32_t search_L_ = 128;
    uint32_t beamwidth_ = 4;
    uint32_t num_search_threads_ = 16;
    uint32_t nodes_to_cache_ = 0;
    uint32_t mem_L_ = 128;
    uint32_t mem_R_ = 48;
    float alpha_mem_ = 1.2F;
    uint32_t disk_L_ = 128;
    uint32_t disk_R_ = 48;
    float alpha_disk_ = 1.2F;
    uint32_t build_C_ = 75;
    uint64_t merge_threshold_ = 3001000;
    bool single_file_index_ = false;
    diskann::Metric metric_ = diskann::Metric::L2;

    std::string pending_dataset_path_;
    std::string pending_tag_path_;
    bool pending_dataset_is_temp_ = false;
    bool pending_tag_is_temp_ = false;

    std::string loaded_base_prefix_;
    std::string loaded_shadow_out_prefix_;

    std::unique_ptr<diskann::Distance<float>> dist_;
    std::unique_ptr<diskann::MergeInsert<float, uint32_t>> runtime_;
};
