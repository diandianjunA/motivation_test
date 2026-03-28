#include <gtest/gtest.h>
#include "experiment_generator.h"

TEST(DataGenerateTest, BasicTest) {
    // 快速生成小型测试数据集
    DataConfig test_config = {
        /*num_vectors=*/100000,
        /*num_queries=*/1000,
        /*dimension=*/1024,
        /*top_k=*/10,
        /*data_min=*/-1.0f,
        /*data_max=*/1.0f,
        /*seed=*/42,
        /*distribution=*/"clustered",
        /*output_dir=*/"/data/xjs/random_dataset/1024dim100K"
    };
    test_config.preset = "medium";
    
    ExperimentGenerator exp_gen(test_config, DistanceMetric::L2);
    
    // 生成并验证
    exp_gen.generate_experiment("1024dim100K", true);
    
    // 加载并展示部分数据
    RandomDataGenerator data_gen(test_config);
    auto database = data_gen.load_vectors("/data/xjs/random_dataset/1024dim100K/base.fbin");
    
    std::cout << "\nSample vector (first 5 dimensions):" << std::endl;
    for (size_t i = 0; i < std::min(size_t(5), database[0].size()); ++i) {
        std::cout << std::fixed << std::setprecision(6) << database[0][i] << " ";
    }
    std::cout << std::endl;
}

// 读取向量文件，检查向量数量和维度
TEST(DataGenerateTest, ReadVectorsTest) {
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
    test_config.preset = "medium";
    
    RandomDataGenerator data_gen(test_config);
    auto database = data_gen.load_vectors("/data/xjs/random_dataset/1024dim100K/base.fbin");
    
    EXPECT_EQ(database.size(), test_config.num_vectors);
    EXPECT_EQ(database[0].size(), test_config.dimension);

    auto queries = data_gen.load_vectors("/data/xjs/random_dataset/1024dim100K/queries/query-test.fbin");
    EXPECT_EQ(queries.size(), test_config.num_queries);
    EXPECT_EQ(queries[0].size(), test_config.dimension);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
