#include "experiment_generator.h"
#include <filesystem>
#include <iostream>
#include <fstream>

ExperimentGenerator::ExperimentGenerator(const DataConfig& config, 
                    DistanceMetric metric)
    : config_(config)
    , data_gen_(config)
    , gt_calc_(config_.dimension, metric) {
    
    // 创建输出目录
    std::filesystem::create_directories(config.output_dir);
}

// 生成完整的数据集和ground truth
void ExperimentGenerator::generate_experiment(const std::string& exp_name, 
                        bool save_vectors) {
    
    std::cout << "\n=============================================" << std::endl;
    std::cout << "Generating experiment: " << exp_name << std::endl;
    std::cout << "=============================================" << std::endl;

    std::cout << "\n[Step 0/3] Preparing output directory..." << std::endl;
    std::filesystem::create_directories(config_.output_dir);
    std::filesystem::create_directories(config_.output_dir + "/queries");

    auto experiment_start = std::chrono::high_resolution_clock::now();
    
    // 1. 生成数据库向量
    std::cout << "\n[Step 1/3] Generating " << config_.num_vectors 
              << " database vectors (dim=" << config_.dimension << ")..." << std::endl;
    auto database = data_gen_.generate_vectors(config_.num_vectors, config_.seed);

    if (save_vectors) {
        std::cout << "\n[Saving] Writing database to disk..." << std::endl;
        data_gen_.save_vectors(config_.output_dir + "/" + "base.fbin", database);
    }
    
    // 2. 生成查询向量
    std::cout << "\n[Step 2/3] Generating " << config_.num_queries 
              << " query vectors (dim=" << config_.dimension << ")..." << std::endl;
    auto queries = data_gen_.generate_vectors(config_.num_queries, 0);

    if (save_vectors) {
        std::cout << "\n[Saving] Writing queries to disk..." << std::endl;
        data_gen_.save_vectors(config_.output_dir + "/queries/" + "query-test.fbin", queries);
    }
    
    // 3. 计算ground truth
    std::cout << "\n[Step 3/3] Computing ground truth for " << config_.num_queries 
              << " queries (top-" << config_.top_k << ")..." << std::endl;
    gt_calc_.init(database);
    auto ground_truth = gt_calc_.compute_all_ground_truth(
        queries, config_.top_k, 32);

    if (save_vectors) {
        std::cout << "\n[Saving] Writing ground truth to disk..." << std::endl;
        gt_calc_.save_ground_truth(config_.output_dir + "/queries/" + "groundtruth-test.bin", ground_truth);
    }
    
    auto experiment_end = std::chrono::high_resolution_clock::now();
    auto total_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        experiment_end - experiment_start);
    
    std::cout << "\n" << std::string(50, '=') << std::endl;
    std::cout << "Experiment '" << exp_name << "' completed!" << std::endl;
    std::cout << "Total time: " << format_time_detailed(total_time.count()) << std::endl;
    
    // 4. 保存数据
    if (save_vectors) {
        // 保存配置信息
        save_config(config_.output_dir + "/" + "config-test.txt");
        
        std::cout << "Data saved to: " << config_.output_dir << "/" << "*.fbin" << std::endl;
        std::cout << "Data saved to: " << config_.output_dir << "/queries/" << "*.bin" << std::endl;
    }
    
    // 5. 输出统计信息
    print_statistics(database, queries, ground_truth);
}

void ExperimentGenerator::save_config(const std::string& filename) {
    std::ofstream fout(filename);
    fout << "Experiment Configuration:" << std::endl;
    fout << "=========================" << std::endl;
    fout << "Database size: " << config_.num_vectors << std::endl;
    fout << "Query size: " << config_.num_queries << std::endl;
    fout << "Dimension: " << config_.dimension << std::endl;
    fout << "Top-K: " << config_.top_k << std::endl;
    fout << "Data range: [" << config_.data_min << ", " 
            << config_.data_max << "]" << std::endl;
    fout << "Distribution: " << config_.distribution << std::endl;
    fout << "Random seed: " << config_.seed << std::endl;
    fout.close();
}

void ExperimentGenerator::print_statistics(const std::vector<std::vector<float>>& database,
                        const std::vector<std::vector<float>>& queries,
                        const std::vector<std::vector<GroundTruthCalculator::Neighbor>>& ground_truth) {
    
    std::cout << "\nDataset Statistics:" << std::endl;
    std::cout << "===================" << std::endl;
    std::cout << "Database vectors: " << database.size() << std::endl;
    std::cout << "Query vectors: " << queries.size() << std::endl;
    std::cout << "Vector dimension: " << config_.dimension << std::endl;
    
    // 计算距离统计
    if (!ground_truth.empty() && !ground_truth[0].empty()) {
        float min_dist = ground_truth[0][0].distance;
        float max_dist = ground_truth[0].back().distance;
        float avg_dist = 0.0f;
        size_t count = 0;
        
        for (const auto& gt_list : ground_truth) {
            for (const auto& neighbor : gt_list) {
                min_dist = std::min(min_dist, neighbor.distance);
                max_dist = std::max(max_dist, neighbor.distance);
                avg_dist += neighbor.distance;
                count++;
            }
        }
        
        avg_dist /= count;
        
        std::cout << "\nDistance Statistics (top-" << config_.top_k << "):" << std::endl;
        std::cout << "Min distance: " << min_dist << std::endl;
        std::cout << "Max distance: " << max_dist << std::endl;
        std::cout << "Avg distance: " << avg_dist << std::endl;
    }
}

// 改进的时间格式化函数
std::string ExperimentGenerator::format_time_detailed(double ms) {
    if (ms < 1000) {
        return std::to_string(static_cast<int>(ms)) + "ms";
    } else if (ms < 60000) {
        double seconds = ms / 1000.0;
        return std::to_string(static_cast<int>(seconds)) + "." 
               + std::to_string(static_cast<int>((seconds - static_cast<int>(seconds)) * 10))
               + "s";
    } else if (ms < 3600000) {
        int minutes = static_cast<int>(ms / 60000);
        int seconds = static_cast<int>((ms - minutes * 60000) / 1000);
        return std::to_string(minutes) + "m " + std::to_string(seconds) + "s";
    } else {
        int hours = static_cast<int>(ms / 3600000);
        int minutes = static_cast<int>((ms - hours * 3600000) / 60000);
        int seconds = static_cast<int>((ms - hours * 3600000 - minutes * 60000) / 1000);
        return std::to_string(hours) + "h " + std::to_string(minutes) + "m " 
               + std::to_string(seconds) + "s";
    }
}
