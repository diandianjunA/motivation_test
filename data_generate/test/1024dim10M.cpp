#include <gtest/gtest.h>
#include "experiment_generator.h"

TEST(DataGenerateTest, BasicTest) {
    // 快速生成小型测试数据集
    DataConfig test_config = {
        /*num_vectors=*/10000000,
        /*num_queries=*/1000,
        /*dimension=*/1024,
        /*top_k=*/10,
        /*data_min=*/-1.0f,
        /*data_max=*/1.0f,
        /*seed=*/42,
        /*distribution=*/"clustered",
        /*output_dir=*/"/data/xjs/random_dataset/1024dim10M"
    };
    test_config.preset = "medium";
    test_config.ground_truth_batch_size = 64;
    test_config.use_gpu = true;
    test_config.gpu_shard_size = 6000000;
    test_config.gpu_device = 0;
    
    ExperimentGenerator exp_gen(test_config, DistanceMetric::L2);
    
    // 生成并验证
    exp_gen.generate_experiment("1024dim10M", true);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
