// 添加 SIMD 头文件
#include <map>
#if defined(__SSE__) || defined(_M_IX86) || defined(_M_X64)
#include <xmmintrin.h>
#include <emmintrin.h>
#else
#pragma message("SSE not available, using software fallback")
#endif

#include "vector_test/vector_index.h"
#include "diskann/include/index_factory.h"

class DiskANNIndex : public VectorIndex {
public:
    DiskANNIndex(std::string conf);

    ~DiskANNIndex() override;

    void build(const std::vector<float>& vecs, const std::vector<uint32_t>& ids) override;

    void build(const std::string& dataset_path) override;

    void insert(const std::vector<float>& vec, const std::vector<uint32_t>& ids) override;

    void search(const std::vector<float>& query, size_t top_k, std::vector<uint32_t>& ids, std::vector<float>& distances) const override;

    void load(const std::string& index_path) override;

    void save(const std::string& index_path) override;

    std::string getIndexType() const override;

private:
    std::unique_ptr<diskann::IndexFactory> index_factory;
    std::unique_ptr<diskann::AbstractIndex> diskann_index;
    std::map<std::string, std::string> config;
    int dim;
    int max_points_to_insert;
    uint32_t L;
    uint32_t R;
    float alpha;
    int num_threads;
};