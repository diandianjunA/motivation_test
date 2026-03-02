#include <gtest/gtest.h>
#include <iomanip>
#include "diskann_index.h"
#include "random_data_generator.h"
#include "ground_truth_calculator.h"
#include <faiss/IndexFlat.h>
#include <faiss/MetricType.h>


TEST(DiskANNTest, CorrectTest) {
    std::cout << std::defaultfloat << std::setprecision(6);
    DiskANNIndex index("/home/xjs/experiment/motivation_test/diskann_test/config/index_conf/1024dim100K.ini");

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

    std::vector<float> dataset(vectors.size() * vectors[0].size());
    std::vector<uint32_t> ids(vectors.size());
    for (size_t i = 0; i < vectors.size(); ++i) {
        ids[i] = i;
        std::copy(vectors[i].begin(), vectors[i].end(), dataset.begin() + i * vectors[0].size());
    }
    index.build(dataset, ids);

    std::vector<float> distances(test_config.top_k);
    std::vector<uint32_t> results(test_config.top_k);

    float recall = 0.0f;
    for (size_t i = 0; i < queries.size(); ++i) {
        index.search(queries[i], test_config.top_k, results, distances);
        std::cout << " DiskANN 查询向量" << i <<" 前 " << test_config.top_k << " 个结果: " << std::endl;
        for (size_t j = 0; j < test_config.top_k; ++j) {
            std::cout << "ID: " << results[j] << ", Distance: " << std::fixed << std::setprecision(6) << distances[j] << std::endl;
        }
        std::cout << "  ground truth 前 " << test_config.top_k << " 个结果: " << std::endl;
        for (size_t j = 0; j < test_config.top_k; ++j) {
            std::cout << "ID: " << ground_truth[i][j].id << ", Distance: " << std::fixed << std::setprecision(6) << ground_truth[i][j].distance << std::endl;
        }
        float recall_at_k = 0.0f;
        for (size_t j = 0; j < test_config.top_k; ++j) {
            if (std::find(results.begin(), results.end(), ground_truth[i][j].id) != results.end()) {
                recall_at_k += 1.0f;
            }
        }
        recall_at_k /= test_config.top_k;
        recall += recall_at_k;
        std::cout << "Recall@" << test_config.top_k << " for query " << i << ": " << std::fixed << std::setprecision(6) << recall_at_k << std::endl;
    }
    recall /= queries.size();
    std::cout << "Recall@" << test_config.top_k << ": " << std::fixed << std::setprecision(6) << recall << std::endl;
}

int main(int argc, char** argv) {
    std::cout << std::defaultfloat << std::setprecision(6);
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}