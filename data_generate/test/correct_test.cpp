#include <cstddef>
#include <gtest/gtest.h>
#include "ground_truth_calculator.h"
#include "include/hnswlib/hnswlib.h"
#include <thread>

template <class Function>
inline void ParallelFor(size_t start, size_t end, size_t numThreads,
                        Function fn) {
    if (numThreads <= 0) {
        numThreads = std::thread::hardware_concurrency();
    }

    if (numThreads == 1) {
        for (size_t id = start; id < end; id++) {
            fn(id, 0);
        }
    } else {
        std::vector<std::thread> threads;
        std::atomic<size_t> current(start);

        // keep track of exceptions in threads
        // https://stackoverflow.com/a/32428427/1713196
        std::exception_ptr lastException = nullptr;
        std::mutex lastExceptMutex;

        for (size_t threadId = 0; threadId < numThreads; ++threadId) {
            threads.push_back(std::thread([&, threadId] {
                while (true) {
                    size_t id = current.fetch_add(1);

                    if (id >= end) {
                        break;
                    }

                    try {
                        fn(id, threadId);
                    } catch (...) {
                        std::unique_lock<std::mutex> lastExcepLock(
                            lastExceptMutex);
                        lastException = std::current_exception();
                        /*
                         * This will work even when current is the largest value
                         * that size_t can fit, because fetch_add returns the
                         * previous value before the increment (what will result
                         * in overflow and produce 0 instead of current + 1).
                         */
                        current = end;
                        break;
                    }
                }
            }));
        }
        for (auto &thread : threads) {
            thread.join();
        }
        if (lastException) {
            std::rethrow_exception(lastException);
        }
    }
}

TEST(DataGenerateTest, HNSWTest) {
    // 快速生成小型测试数据集
    DataConfig test_config = {
        /*num_vectors=*/1000000,
        /*num_queries=*/1000,
        /*dimension=*/1024,
        /*top_k=*/10,
        /*data_min=*/-1.0f,
        /*data_max=*/1.0f,
        /*seed=*/42,
        /*distribution=*/"normal",
        /*output_dir=*/"/data/xjs/random_dataset/1024dim1M"
    };

    std::string data_path = test_config.output_dir + "/base.fbin";
    RandomDataGenerator generator(test_config);
    auto vectors = generator.load_vectors(data_path);

    std::string query_path = test_config.output_dir + "/queries/query-test.fbin";
    auto queries = generator.load_vectors(query_path);

    GroundTruthCalculator gt_calc(test_config.dimension, DistanceMetric::L2);
    auto ground_truth = gt_calc.load_ground_truth(test_config.output_dir + "/queries/groundtruth-test.bin");

    hnswlib::L2Space space(test_config.dimension);
    hnswlib::HierarchicalNSW<float> hnsw(&space, test_config.num_vectors, 32, 200);
    hnsw.setEf(1000);
    
    // 插入数据
    int batch_size = 1000;
    std::vector<float> batch_vectors;
    std::vector<uint32_t> batch_ids;
    for (uint32_t i = 0; i < test_config.num_vectors; i++) {
        batch_vectors.insert(batch_vectors.end(), vectors[i].begin(), vectors[i].end());
        batch_ids.push_back(i);

        if (batch_vectors.size() >= batch_size * test_config.dimension || i == test_config.num_vectors - 1) {
            ParallelFor(0, batch_ids.size(), std::thread::hardware_concurrency(), [&](size_t id, size_t threadId) {
                (void)threadId;
                hnsw.addPoint(batch_vectors.data() + id * test_config.dimension, batch_ids[id]);
            });

            batch_vectors.clear();
            batch_ids.clear();

            std::cout << "已插入 " << i << " 个向量" << std::endl;
        }
    }

    float total_recall = 0.0f;
    // 执行查询
    for (uint32_t i = 0; i < test_config.num_queries; i++) {
        std::vector<float> query(test_config.dimension);
        std::copy(queries[i].begin(), queries[i].end(), query.begin());
        auto results = hnsw.searchKnn(query.data(), test_config.top_k);
        std::vector<size_t> hnsw_ids(test_config.top_k);
        while (!results.empty()) {
            hnsw_ids.push_back(results.top().second);
            results.pop();
        }
        std::reverse(hnsw_ids.begin(), hnsw_ids.end());

        std::cout << "HNSW 前10个邻居: " << std::endl;
        for (uint32_t j = 0; j < test_config.top_k; j++) {
            std::cout << "ID: " << hnsw_ids[j] << std::endl;
        }

        std::cout << "Ground Truth 前10个邻居: " << std::endl;
        for (uint32_t j = 0; j < test_config.top_k; j++) {
            std::cout << "ID: " << ground_truth[i][j].id << ", 距离: " << ground_truth[i][j].distance << std::endl;
        }

        uint32_t num_correct = 0;
        for (uint32_t j = 0; j < test_config.top_k; j++) {
            if (std::find(hnsw_ids.begin(), hnsw_ids.end(), ground_truth[i][j].id) != hnsw_ids.end()) {
                num_correct++;
            }
        }
        float recall = static_cast<float>(num_correct) / test_config.top_k;
        std::cout << "查询 " << i << " 的召回率: " << recall << std::endl;

        total_recall += recall;
    }
    total_recall /= test_config.num_queries;
    std::cout << "平均召回率: " << total_recall << std::endl;
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
