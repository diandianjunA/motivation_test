#include <gtest/gtest.h>
#include "experiment_generator.h"

TEST(DataGenerateTest, BasicTest) {
    // 快速生成小型测试数据集
    DataConfig test_config = {
        /*num_vectors=*/1000,
        /*num_queries=*/100,
        /*dimension=*/64,
        /*top_k=*/10,
        /*data_min=*/-1.0f,
        /*data_max=*/1.0f,
        /*seed=*/42,
        /*distribution=*/"normal",
        /*output_dir=*/"./test_data"
    };
    
    ExperimentGenerator exp_gen(test_config, DistanceMetric::L2);
    
    // 生成并验证
    exp_gen.generate_experiment("quick_test", true);
    
    // 加载并展示部分数据
    RandomDataGenerator data_gen(test_config);
    auto database = data_gen.load_vectors("./test_data/base.fbin");
    
    std::cout << "\nSample vector (first 5 dimensions):" << std::endl;
    for (size_t i = 0; i < std::min(size_t(5), database[0].size()); ++i) {
        std::cout << std::fixed << std::setprecision(6) << database[0][i] << " ";
    }
    std::cout << std::endl;
}

// 读取向量文件，检查向量数量和维度
TEST(DataGenerateTest, ReadVectorsTest) {
    DataConfig test_config = {
        /*num_vectors=*/1000,
        /*num_queries=*/100,
        /*dimension=*/64,
        /*top_k=*/10,
        /*data_min=*/-1.0f,
        /*data_max=*/1.0f,
        /*seed=*/42,
        /*distribution=*/"normal",
        /*output_dir=*/"./test_data"
    };

    RandomDataGenerator data_gen(test_config);
    auto database = data_gen.load_vectors("./test_data/base.fbin");
    
    EXPECT_EQ(database.size(), test_config.num_vectors);
    EXPECT_EQ(database[0].size(), test_config.dimension);

    auto queries = data_gen.load_vectors("./test_data/queries/query-test.fbin");
    EXPECT_EQ(queries.size(), test_config.num_queries);
    EXPECT_EQ(queries[0].size(), test_config.dimension);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
