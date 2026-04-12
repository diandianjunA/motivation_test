#pragma once

#include <memory>
#include <string>
#include <vector>

#include "common/distance.hh"
#include "service/compute_service.hh"
#include "vector_test/vector_index.h"

class DvstorIndex : public VectorIndex {
public:
    explicit DvstorIndex(const std::string& service_config_path);
    ~DvstorIndex() override;

    void build(const std::vector<float>& vecs, const std::vector<uint32_t>& ids) override;
    void build(const std::string& dataset_path) override;
    void insert(const std::vector<float>& vectors, const std::vector<uint32_t>& ids) override;
    void search(const std::vector<float>& query,
                size_t top_k,
                std::vector<uint32_t>& ids,
                std::vector<float>& distances) const override;
    void load(const std::string& index_path) override;
    void save(const std::string& index_path) override;
    std::string getIndexType() const override;

private:
    size_t dim() const;
    static std::vector<std::string> build_service_argv(const std::string& service_config_path);

private:
    bool ip_distance_{false};
    mutable std::unique_ptr<ComputeService<L2Distance>> l2_service_;
    mutable std::unique_ptr<ComputeService<IPDistance>> ip_service_;
};
