#include <cstddef>
#include <faiss/IndexFlat.h>
#include <faiss/IndexIDMap.h>
#include <faiss/MetricType.h>
#include <gtest/gtest.h>
#include "ground_truth_calculator.h"
#include "random_data_generator.h"

TEST(DataGenerateTest, FLATTest) {
    // 快速生成小型测试数据集
    DataConfig test_config = {
        /*num_vectors=*/100000,
        /*num_queries=*/1000,
        /*dimension=*/1024,
        /*top_k=*/10,
        /*data_min=*/-1.0f,
        /*data_max=*/1.0f,
        /*seed=*/42,
        /*distribution=*/"normal",
        /*output_dir=*/"/data/xjs/random_dataset/1024dim100K"
    };
    
    std::string data_path = test_config.output_dir + "/base.fbin";
    RandomDataGenerator generator(test_config);
    auto vectors = generator.load_vectors(data_path);
    
    std::string query_path = test_config.output_dir + "/queries/query-test.fbin";
    auto queries = generator.load_vectors(query_path);

    GroundTruthCalculator gt_calc(test_config.dimension, DistanceMetric::L2);
    auto ground_truth = gt_calc.load_ground_truth(test_config.output_dir + "/queries/groundtruth-test.bin");

    faiss::IndexIDMap* index = new faiss::IndexIDMap(new faiss::IndexFlat(test_config.dimension, faiss::MetricType::METRIC_L2));

    // 批量插入数据
    int batch_size = 10000;
    std::vector<faiss::idx_t> ids;
    std::vector<float> batch_vectors;
    
    for (size_t i = 0; i < test_config.num_vectors; ++i) {
        size_t dim = vectors[i].size();
        
        batch_vectors.insert(batch_vectors.end(), vectors[i].begin(), vectors[i].end());
        ids.push_back(static_cast<faiss::idx_t>(i));
        
        if (batch_vectors.size() >= batch_size * dim || i == test_config.num_vectors - 1) {
            size_t batch_n = ids.size();
            index->add_with_ids(batch_n, batch_vectors.data(), ids.data());
            
            batch_vectors.clear();
            ids.clear();
        }

        if (i % 1000 == 0) {
            std::cout << "Inserted " << i << " vectors" << std::endl;
        }
    }
    
    // 执行查询
    std::vector<std::vector<size_t>> new_ground_truth;
    new_ground_truth.resize(queries.size()); 

    auto compute_ground_truth = [&] (const std::vector<float>& query, size_t k) {
        std::vector<float> query_vec = query;
        std::vector<faiss::idx_t> ids(k);
        std::vector<float> distances(k, 0.0f);
        index->search(1, query_vec.data(), k, distances.data(), ids.data());
        
        std::vector<size_t> neighbors;
        neighbors.reserve(k);
        for (size_t i = 0; i < k; ++i) {
            neighbors.push_back(ids[i]);
        }
        return neighbors;
    };

    #pragma omp parallel for num_threads(32) schedule(dynamic, 1000)
    for (uint32_t i = 0; i < test_config.num_queries; i++) {
        new_ground_truth[i] = compute_ground_truth(queries[i], test_config.top_k);
    }

    for (uint32_t i = 0; i < test_config.num_queries; i++) {
        std::cout << "new_ground_truth[" << i << "]:" << std::endl;
        for (uint32_t j = 0; j < test_config.top_k; j++) {
            std::cout << "ID: " << new_ground_truth[i][j] << std::endl;
        }
        std::cout << "ground_truth[" << i << "]:" << std::endl;
        for (uint32_t j = 0; j < test_config.top_k; j++) {
            std::cout << "ID: " << ground_truth[i][j].id << std::endl;
        }
    }
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
