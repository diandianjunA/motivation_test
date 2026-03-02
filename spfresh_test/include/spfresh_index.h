#include "vector_test/vector_index.h"
#include "inc/Core/VectorIndex.h"
#include "inc/Core/SPANN/Index.h"

class SPFreshIndex : public VectorIndex {
public:
    SPFreshIndex(std::string conf);

    ~SPFreshIndex();

    void build(const std::vector<float>& vecs, const std::vector<uint32_t>& ids) override;

    void build(const std::string& dataset_path) override;

    void insert(const std::vector<float>& vec, const std::vector<uint32_t>& ids) override;

    void search(const std::vector<float>& query, size_t top_k, std::vector<uint32_t>& ids, std::vector<float>& distances) const override;

    void load(const std::string& index_path) override;

    void save(const std::string& index_path) override;

    std::string getIndexType() const override;

private:
    std::string m_conf;
    int m_dimension;
    std::shared_ptr<SPTAG::VectorIndex> m_index;
    SPTAG::SPANN::Index<float>* m_spann_index;
};