#include "diskann_index.h"
#include "vector_test/config.h"
#include "vector_test/util.h"
#include "progress_bar.hpp"
#include <cstddef>
#include <cstdint>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <vector>
#include <omp.h>

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

std::pair<const float*, std::pair<size_t, size_t>> partial_mmap_fbin(
    const std::string& file_path, size_t start_vector, size_t num_vectors) {
    int fd = open(file_path.c_str(), O_RDONLY);
    if (fd == -1) {
        throw std::runtime_error("无法打开文件: " + file_path);
    }
    
    struct stat sb;
    if (fstat(fd, &sb) == -1) {
        close(fd);
        throw std::runtime_error("无法获取文件大小: " + file_path);
    }
    
    size_t file_size = sb.st_size;
    
    const size_t header_size = 8;
    const size_t page_size = 4096;
    
    uint32_t header[2];
    pread(fd, header, header_size, 0);
    uint32_t dim = header[1];
    
    size_t data_size = num_vectors * dim * sizeof(float);
    size_t data_offset = header_size + start_vector * dim * sizeof(float);
    
    size_t mmap_offset = (data_offset / page_size) * page_size;
    size_t extra_bytes = data_offset - mmap_offset;
    
    size_t mmap_size = data_size + extra_bytes;
    
    if (mmap_offset + mmap_size > file_size) {
        close(fd);
        throw std::runtime_error("偏移量超出文件大小");
    }
    
    void* addr = mmap(nullptr, mmap_size, PROT_READ, MAP_PRIVATE, fd, mmap_offset);
    if (addr == MAP_FAILED) {
        close(fd);
        throw std::runtime_error("部分内存映射失败");
    }
    
    close(fd);
    
    madvise(addr, mmap_size, MADV_WILLNEED);
    
    const float* data = reinterpret_cast<const float*>(static_cast<char*>(addr) + extra_bytes);
    return {data, {num_vectors, dim}};
}

void unmap_partial(const float* data, size_t num_vectors, size_t dim, size_t extra_bytes) {
    if (data != nullptr) {
        const char* addr = reinterpret_cast<const char*>(data) - extra_bytes;
        size_t mmap_size = num_vectors * dim * sizeof(float) + extra_bytes;
        munmap(const_cast<char*>(addr), mmap_size);
    }
}

void DiskANNIndex::build(const std::string& dataset_path) {
    int fd = open(dataset_path.c_str(), O_RDONLY);
    if (fd == -1) {
        throw std::runtime_error("无法打开文件: " + dataset_path);
    }
    
    struct stat sb;
    if (fstat(fd, &sb) == -1) {
        close(fd);
        throw std::runtime_error("无法获取文件大小: " + dataset_path);
    }
    size_t file_size = sb.st_size;
    
    uint32_t header[2];
    pread(fd, header, 8, 0);
    uint32_t total_num = header[0];
    uint32_t dim = header[1];
    close(fd);
    
    std::cout << "数据集: " << total_num << " vectors, " << dim << " dim" << std::endl;
    std::cout << "文件大小: " << file_size << " bytes (" << file_size / (1024*1024*1024.0) << " GB)" << std::endl;
    
    size_t batch_size = 100000;
    size_t num_batches = (total_num + batch_size - 1) / batch_size;
    
    std::cout << "分批构建索引: " << num_batches << " 批次, 每批 " << batch_size << " 向量" << std::endl;
    
    ProgressBar bar("Building index", total_num, true, true);
    
    size_t processed = 0;
    for (size_t batch = 0; batch < num_batches; ++batch) {
        size_t start = batch * batch_size;
        size_t current_batch_size = std::min(batch_size, total_num - start);
        
        auto [mapped_data, sizes] = partial_mmap_fbin(dataset_path, start, current_batch_size);
        
        std::vector<uint32_t> ids(current_batch_size);
        for (size_t i = 0; i < current_batch_size; ++i) {
            ids[i] = static_cast<uint32_t>(start + i);
        }
        
        size_t extra_bytes = 0;
        if (batch > 0) {
            size_t data_offset = 8 + start * dim * sizeof(float);
            const size_t page_size = 4096;
            extra_bytes = data_offset - (data_offset / page_size) * page_size;
        }
        
        if (batch == 0) {
            diskann_index->build(mapped_data, current_batch_size, ids);
        } else {
            #pragma omp parallel for schedule(dynamic, 1000)
            for (size_t i = 0; i < current_batch_size; ++i) {
                diskann_index->insert_point(mapped_data + i * dim, ids[i]);
            }
        }
        
        unmap_partial(mapped_data, current_batch_size, dim, extra_bytes);
        
        processed += current_batch_size;
        bar.set_current(processed);
        bar.display();
        
        std::cout << "  已处理 " << processed << "/" << total_num << " 向量" << std::endl;
    }
    
    bar.finish();
    std::cout << "索引构建完成!" << std::endl;
}

void DiskANNIndex::insert(const std::vector<float>& vec, const std::vector<uint32_t>& ids) {
    int count = ids.size();
    #pragma omp parallel for schedule(dynamic, 1000)
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