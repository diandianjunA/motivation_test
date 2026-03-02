#include "spfresh_index.h"
#include "vector_test/config.h"
#include "vector_test/util.h"
#include <iostream>
#include <sys/mman.h>

SPFreshIndex::SPFreshIndex(std::string conf) {
    // 解析配置字符串，初始化索引参数
    m_conf = conf;
    // 解析配置字符串，初始化索引参数
    auto config = readConfig(m_conf);
    // 初始化索引
    m_index = SPTAG::VectorIndex::CreateInstance(SPTAG::IndexAlgoType::SPANN, SPTAG::VectorValueType::Float);
    if (!m_index) {
        throw std::runtime_error("Failed to create SPTAG VectorIndex");
    }
    if (config.find("Dim") == config.end()) {
        throw std::runtime_error("Dim not found in config");
    }
    int dim = std::stoi(config["Dim"]);
    m_dimension = dim;
    if (config.find("VectorSize") == config.end()) {
        throw std::runtime_error("VectorSize not found in config");
    }
    int vector_size = std::stoi(config["VectorSize"]);
    if (config.find("IndexDirectory") == config.end()) {
        throw std::runtime_error("IndexDirectory not found in config");
    }
    std::string index_dir = config["IndexDirectory"];
    if (config.find("KVPath") == config.end()) {
        throw std::runtime_error("KVPath not found in config");
    }
    std::string kv_path = config["KVPath"];

    m_index->SetParameter("ValueType", "Float", "Base");
    m_index->SetParameter("DistCalcMethod", "L2", "Base");
    m_index->SetParameter("IndexAlgoType", "BKT", "Base");
    m_index->SetParameter("Dim", std::to_string(dim), "Base");
    m_index->SetParameter("VectorSize", std::to_string(vector_size), "Base");  // 预期10万个向量
    m_index->SetParameter("IndexDirectory", index_dir, "Base");
    m_index->SetParameter("DeletedIDs", index_dir + "/DeletedIDs.bin", "Base");  
    m_index->SetParameter("HeadVectors", "SPTAGHeadVectors.bin", "Base");  
    m_index->SetParameter("HeadVectorIDs", "SPTAGHeadVectorIDs.bin", "Base");  
    m_index->SetParameter("SsdInfoFile", index_dir + "/ssdinfo.bin", "BuildSSDIndex");  

    // 选择头节点参数设置
    m_index->SetParameter("isExecute", "true", "SelectHead");
    m_index->SetParameter("TreeNumber", "1", "SelectHead");
    m_index->SetParameter("BKTKmeansK", "32", "SelectHead");
    m_index->SetParameter("BKTLeafSize", "8", "SelectHead");
    m_index->SetParameter("SamplesNumber", "1000", "SelectHead");
    m_index->SetParameter("SelectThreshold", "10", "SelectHead");
    m_index->SetParameter("SplitFactor", "6", "SelectHead");
    m_index->SetParameter("SplitThreshold", "25", "SelectHead");
    m_index->SetParameter("Ratio", "0.12", "SelectHead");
    m_index->SetParameter("NumberOfThreads", "45", "SelectHead");

    // 构建头节点图参数设置
    m_index->SetParameter("isExecute", "true", "BuildHead");
    m_index->SetParameter("NeighborhoodSize", "32", "BuildHead");
    m_index->SetParameter("TPTNumber", "32", "BuildHead");
    m_index->SetParameter("TPTLeafSize", "2000", "BuildHead");
    m_index->SetParameter("MaxCheck", "16324", "BuildHead");
    m_index->SetParameter("MaxCheckForRefineGraph", "16324", "BuildHead");
    m_index->SetParameter("RefineIterations", "3", "BuildHead");
    m_index->SetParameter("NumberOfThreads", "45", "BuildHead");

    // 构建SSD索引参数设置
    m_index->SetParameter("isExecute", "true", "BuildSSDIndex");
    m_index->SetParameter("BuildSsdIndex", "true", "BuildSSDIndex");
    m_index->SetParameter("InternalResultNum", "64", "BuildSSDIndex");
    m_index->SetParameter("ReplicaCount", "8", "BuildSSDIndex");
    m_index->SetParameter("PostingPageLimit", "32", "BuildSSDIndex");
    m_index->SetParameter("NumberOfThreads", "45", "BuildSSDIndex");
    m_index->SetParameter("MaxCheck", "16324", "BuildSSDIndex");
    m_index->SetParameter("TmpDir", "/tmp/", "BuildSSDIndex");
    m_index->SetParameter("SearchInternalResultNum", "32", "BuildSSDIndex");
    m_index->SetParameter("SearchPostingPageLimit", "32", "BuildSSDIndex");
    m_index->SetParameter("SearchResult", "result.txt", "BuildSSDIndex");
    m_index->SetParameter("ResultNum", "10", "BuildSSDIndex");
    m_index->SetParameter("MaxDistRatio", "8.0", "BuildSSDIndex");
    m_index->SetParameter("UseKV", "true", "BuildSSDIndex");  
    m_index->SetParameter("KVPath", kv_path, "BuildSSDIndex");
    m_index->SetParameter("Update", "true", "BuildSSDIndex");
    m_index->SetParameter("Step", "100", "BuildSSDIndex");
    m_index->SetParameter("ExcludeHead", "false", "BuildSSDIndex");

    m_spann_index = (SPTAG::SPANN::Index<float>*)m_index.get();
}

SPFreshIndex::~SPFreshIndex() {
    m_spann_index->ForceCompaction();
    m_spann_index->ExitBlockController();
}

void SPFreshIndex::build(const std::vector<float>& vecs, const std::vector<uint32_t>& ids) {
    // 实现构建索引的逻辑
    m_spann_index->BuildIndex(vecs.data(), ids.size(), m_dimension);
}

void SPFreshIndex::build(const std::string& dataset_path) {
    // 实现从数据集路径构建索引的逻辑
    // 使用内存映射打开文件
    auto [mapped_data, sizes] = mmap_fvecs(dataset_path);
    size_t num = sizes.first;
    size_t dim = sizes.second;

    // diskann需要先build再insert
    // 因此先分出一部分向量来build
    // int build_num = std::min(num, (size_t)1000000);
    int build_num = num / 10;
    std::vector<float> build_vecs(build_num * dim);
    for (int i = 0; i < build_num; ++i) {
        std::copy(mapped_data + i * (dim + 1) + 1, mapped_data + (i + 1) * (dim + 1), build_vecs.begin() + i * dim);
    }
    std::vector<uint32_t> build_ids(build_num);
    for (int i = 0; i < build_num; ++i) {
        build_ids[i] = i + 1;
    }
    m_spann_index->BuildIndex(build_vecs.data(), build_num, m_dimension);
    // 提前释放build_vecs
    build_vecs.clear();
    build_vecs.shrink_to_fit();

    const size_t batch_size = 100000; // 每个批次处理10万个向量
    std::vector<float> insert_vecs(batch_size * dim);
    std::vector<int32_t> insert_ids(batch_size);

    for (size_t start = build_num; start < num; start += batch_size) {
        size_t end = std::min(start + batch_size, num);
        size_t batch_count = end - start;
        
        for (int i = 0; i < batch_count; ++i) {
            std::copy(mapped_data + (start + i) * (dim + 1) + 1, mapped_data + (start + i + 1) * (dim + 1), insert_vecs.begin() + i * dim);
            insert_ids[i] = start + i + 1;
        }

        // 输出第一个向量作为调试
        std::cout << "Inserting vector[0]: (" << insert_vecs[0] << ", " << insert_vecs[1] << ", ..., " << insert_vecs[dim-1] << ")" << std::endl;

        try {
            m_spann_index->AddIndexSPFresh(insert_vecs.data(), batch_count, m_dimension, insert_ids.data());
        } catch (const std::exception& e) {
            std::cerr << "Error inserting batch " << start << ": " << e.what() << std::endl;
        }
        
        // 打印进度
        std::cout << "Processed " << end << "/" << num << " vectors (" << 100.0 * end / num << "%)" << std::endl;
    }

    // 解除内存映射
    munmap(const_cast<float*>(mapped_data), num * (dim + 1) * sizeof(float) + sizeof(uint32_t));
}

void SPFreshIndex::insert(const std::vector<float>& vec, const std::vector<uint32_t>& ids) {
    // 实现插入向量的逻辑
    m_spann_index->AddIndexSPFresh(vec.data(), ids.size(), m_dimension, (int32_t*)ids.data());

    while(!m_spann_index->AllFinished()) {  
        std::this_thread::sleep_for(std::chrono::milliseconds(20));  
    }
}

void SPFreshIndex::search(const std::vector<float>& query, size_t top_k, std::vector<uint32_t>& ids, std::vector<float>& distances) const {
    // 实现搜索逻辑
    SPTAG::QueryResult result(query.data(), top_k, false);  
    m_spann_index->GetMemoryIndex()->SearchIndex(result);
    m_spann_index->SearchDiskIndex(result);

    // 提取结果
    ids.resize(top_k);
    distances.resize(top_k);
    for (int i = 0; i < top_k; ++i) {
        auto res = result.GetResult(i);
        ids[i] = res->VID;
        distances[i] = res->Dist;
    }
}

void SPFreshIndex::load(const std::string& index_path) {
    // 实现从索引路径加载索引的逻辑
    m_index->LoadIndex(index_path, m_index);
    m_spann_index = (SPTAG::SPANN::Index<float>*)m_index.get();
}

void SPFreshIndex::save(const std::string& index_path) {

    m_spann_index->ForceCompaction();
    m_spann_index->ExitBlockController();

    // 实现将索引保存到索引路径的逻辑
    m_index->SaveIndex(index_path);
}

std::string SPFreshIndex::getIndexType() const {
    return "SPFreshIndex";
}
