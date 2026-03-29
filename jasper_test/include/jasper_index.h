#pragma once

#include "vector_index.h"
#include <map>
#include <memory>
#include <string>
#include <vector>

class JasperIndex : public VectorIndex {
public:
    struct RuntimeBase;

    explicit JasperIndex(std::string conf);
    ~JasperIndex() override;

    void build(const std::vector<float>& vecs, const std::vector<uint32_t>& ids) override;
    void build(const std::string& dataset_path) override;
    void insert(const std::vector<float>& vec, const std::vector<uint32_t>& ids) override;
    void search(const std::vector<float>& query, size_t top_k, std::vector<uint32_t>& ids, std::vector<float>& distances) const override;
    void load(const std::string& index_path) override;
    void save(const std::string& index_path) override;
    std::string getIndexType() const override;

private:
    void rebuildIndex();
    void ensureRuntime() const;
    void loadMetadata(const std::string& index_path);
    void saveMetadata(const std::string& index_path) const;

    std::string config_path_;
    std::map<std::string, std::string> config_;
    size_t dim_ = 0;

    uint32_t build_n_rounds_ = 1;
    uint32_t build_nodes_explored_per_iteration_ = 4;
    bool build_random_init_ = false;
    double build_alpha_ = 1.2;
    double build_max_batch_ratio_ = 0.02;

    uint32_t search_beam_width_ = 64;
    float search_cut_ = 10.0F;
    uint32_t search_limit_ = 512;

    std::unique_ptr<RuntimeBase> runtime_;
    std::vector<float> host_vectors_;
    std::vector<uint32_t> external_ids_;
};
