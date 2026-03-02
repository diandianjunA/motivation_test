#include "diskann_index.h"
#include "vector_test/config.h"
#include "vector_test/util.h"
#include <cstddef>
#include <cstdint>
#include <string>
#include <sys/mman.h>
#include <vector>

DiskANNIndex::DiskANNIndex(std::string conf) : index_factory(nullptr) {
    // 解析配置字符串，初始化索引参数
    config = readConfig(conf);

    this->dim = std::stoi(this->config["dim"]); // 维度
    this->max_points_to_insert = std::stoi(this->config["max_points_to_insert"]); // 最大插入点数量
    diskann::Metric metric = diskann::L2; // 距离度量
    this->L = std::stoi(this->config["L"]); // 搜索列表大小
    this->R = std::stoi(this->config["R"]); // 最大度数
    this->alpha = std::stof(this->config["alpha"]); // alpha参数
    this->num_threads = std::stoi(this->config["num_threads"]); // 线程数
        
    uint32_t Lf = 0; // 过滤列表大小
    const std::string index_path_prefix = "diskann_index";
    const std::string data_type = "float";
    const std::string label_type = "uint32";

    bool saturate_graph = true; // 是否饱和图

    // 创建DiskANN索引
    diskann::IndexWriteParameters params = diskann::IndexWriteParametersBuilder(L, R)
                                                   .with_max_occlusion_size(500)
                                                   .with_saturate_graph(saturate_graph)
                                                   .with_alpha(alpha)
                                                   .with_num_threads(num_threads)
                                                   .with_filter_list_size(Lf)
                                                   .build();
    auto index_search_params = diskann::IndexSearchParams(params.search_list_size, params.num_threads);
    bool enable_tags = true; // 是否启用标签

    auto index_build_params = diskann::IndexWriteParametersBuilder(L, R)
                                    .with_filter_list_size(Lf)
                                    .with_alpha(alpha)
                                    .with_saturate_graph(saturate_graph)
                                    .with_num_threads(num_threads)
                                    .build();

    auto config = diskann::IndexConfigBuilder()
                        .with_metric(metric)
                        .with_dimension(dim)
                        .with_max_points(max_points_to_insert)
                        .with_data_load_store_strategy(diskann::DataStoreStrategy::MEMORY)
                        .with_graph_load_store_strategy(diskann::GraphStoreStrategy::MEMORY)
                        .with_data_type(data_type)
                        .with_label_type(label_type)
                        .is_dynamic_index(true)
                        .with_index_search_params(index_search_params)
                        .with_index_write_params(index_build_params)
                        .is_enable_tags(enable_tags)
                        .build();

    index_factory = std::make_unique<diskann::IndexFactory>(config);
    diskann_index = std::move(index_factory->create_instance());
}

DiskANNIndex::~DiskANNIndex() {
    if (diskann_index) {
        diskann_index.reset();
    }
    if (index_factory) {
        index_factory.reset();
    }
}

void DiskANNIndex::build(const std::vector<float>& vecs, const std::vector<uint32_t>& ids) {
    diskann_index->build(vecs.data(), ids.size(), ids);
}

void DiskANNIndex::build(const std::string& dataset_path) {
    // 使用内存映射打开文件
    auto [mapped_data, sizes] = mmap_fbin(dataset_path);
    size_t num = sizes.first;
    size_t dim = sizes.second;
    
    const uint32_t* original_mapped_addr = reinterpret_cast<const uint32_t*>(mapped_data) - 2;
    
    // 生成所有向量的ID
    std::vector<uint32_t> all_ids(num);
    for (size_t i = 0; i < num; ++i) {
        all_ids[i] = static_cast<uint32_t>(i);
    }
    
    // 直接使用内存映射的数据构建整个索引
    std::cout << "Building index with " << num << " vectors..." << std::endl;
    diskann_index->build(mapped_data, num, all_ids);
    std::cout << "Index built successfully!" << std::endl;
    
    // 解除映射
    size_t file_size = 8 + num * dim * sizeof(float);
    munmap(const_cast<uint32_t*>(original_mapped_addr), file_size);
}

void DiskANNIndex::insert(const std::vector<float>& vec, const std::vector<uint32_t>& ids) {
    int count = ids.size();
    for (int i = 0; i < count; ++i) {
        diskann_index->insert_point(&vec[i * this->dim], ids[i]);
    }
}

void DiskANNIndex::search(const std::vector<float>& query, size_t top_k, std::vector<uint32_t>& ids, std::vector<float>& distances) const {
    ids.resize(top_k);
    distances.resize(top_k);
    diskann_index->search(query.data(), top_k, this->L, ids.data(), distances.data());
}

void DiskANNIndex::load(const std::string& index_path) {
    diskann_index->load(index_path.c_str(), 1, this->L);
}

void DiskANNIndex::save(const std::string& index_path) {
    diskann_index->save(index_path.c_str());
}

std::string DiskANNIndex::getIndexType() const {
    return "DiskANN";
}