#pragma once

#include "distance.h"
#include "vector_index.h"
#include "v2/dynamic_index.h"
#include <map>
#include <memory>
#include <string>

class OdinANNIndex : public VectorIndex {
public:
    explicit OdinANNIndex(std::string conf, std::string runtime_conf = "");
    ~OdinANNIndex() override = default;

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
    void buildMemIndexToPrefix(
        const std::string& index_prefix,
        const std::string& dataset_path,
        const char* tag_file) const;
    void ensureMemIndexReady(const std::string& index_path) const;
    std::string writeTempFbin(const std::vector<float>& vecs, size_t count) const;
    std::string writeTagFile(const std::vector<uint32_t>& ids, const std::string& path) const;
    void copyIndexPrefix(const std::string& src_prefix, const std::string& dst_prefix) const;
    int resolveSearchMode() const;

    std::string config_path_;
    std::map<std::string, std::string> config_;
    size_t dim_ = 0;
    size_t max_points_ = 0;
    uint32_t build_R_ = 48;
    uint32_t build_L_ = 150;
    std::string build_B_ = "8";
    std::string build_M_ = "16";
    uint32_t build_threads_ = 16;
    uint32_t search_L_ = 200;
    uint32_t beamwidth_ = 4;
    uint32_t runtime_threads_ = 16;
    uint32_t merge_L_disk_ = 150;
    uint32_t merge_R_disk_ = 0;
    float alpha_disk_ = 1.2F;
    uint32_t merge_C_ = 384;
    uint32_t mem_L_ = 192;
    uint32_t mem_R_ = 64;
    float mem_alpha_ = 1.2F;
    uint32_t mem_C_ = 100;
    uint32_t search_mem_L_ = 0;
    bool use_mem_index_ = false;
    bool single_file_index_ = false;
    bool force_sharded_build_ = false;
    uint32_t partition_replication_factor_ = 2;
    pipeann::Metric metric_ = pipeann::Metric::L2;

    std::string pending_dataset_path_;
    std::string pending_tag_path_;
    bool pending_identity_tags_ = false;
    bool pending_dataset_is_temp_ = false;
    bool pending_tag_is_temp_ = false;

    std::string loaded_base_prefix_;
    std::string loaded_shadow_out_prefix_;

    std::unique_ptr<pipeann::Distance<float>> dist_;
    std::unique_ptr<pipeann::DynamicSSDIndex<float, uint32_t>> runtime_;
};
