#pragma once
#include <vector>
#include <string>
#include <stdint.h>

class VectorIndex {
public:
    VectorIndex() {};
    virtual~VectorIndex() = default;

    virtual void build(const std::vector<float>& vecs, const std::vector<uint32_t>& ids) = 0;

    virtual void build(const std::string& dataset_path) = 0;

    virtual void insert(const std::vector<float>& vec, const std::vector<uint32_t>& ids) = 0;

    virtual void search(const std::vector<float>& query, size_t top_k, std::vector<uint32_t>& ids, std::vector<float>& distances) const = 0;

    virtual void load(const std::string& index_path) = 0;

    virtual void save(const std::string& index_path) = 0;

    virtual std::string getIndexType() const = 0;
};