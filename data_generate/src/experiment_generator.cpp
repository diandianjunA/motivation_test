#include "experiment_generator.h"
#include <filesystem>
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <thread>

namespace {

DataConfig resolved_config(DataConfig config) {
    apply_data_preset(config);
    return config;
}

std::vector<float> flatten_vectors(const std::vector<std::vector<float>>& vectors, size_t dim) {
    std::vector<float> flattened(vectors.size() * dim);
    for (size_t i = 0; i < vectors.size(); ++i) {
        std::copy(vectors[i].begin(), vectors[i].end(), flattened.begin() + i * dim);
    }
    return flattened;
}

} // namespace

ExperimentGenerator::ExperimentGenerator(const DataConfig& config, 
                    DistanceMetric metric)
    : config_(resolved_config(config))
    , data_gen_(config_)
    , gt_calc_(
          config_.dimension,
          metric,
          config_.use_gpu,
          config_.gpu_shard_size,
          config_.gpu_device) {
    // 创建输出目录
    std::filesystem::create_directories(config_.output_dir);
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
    std::vector<std::vector<float>> queries;
    if (config_.query_mode == "independent") {
        queries = data_gen_.generate_vectors(config_.num_queries, 0);
    } else if (config_.query_mode == "from_base_noise") {
        queries = data_gen_.generate_queries_from_database(database, config_.num_queries, config_.seed + 1000003);
    } else {
        throw std::runtime_error("unsupported query mode: " + config_.query_mode);
    }

    if (save_vectors) {
        std::cout << "\n[Saving] Writing queries to disk..." << std::endl;
        data_gen_.save_vectors(config_.output_dir + "/queries/" + "query-test.fbin", queries);
    }
    
    // 3. 计算ground truth
    std::cout << "\n[Step 3/3] Computing ground truth for " << config_.num_queries 
              << " queries (top-" << config_.top_k << ")..." << std::endl;
    gt_calc_.init(database);
    std::vector<float> flat_queries = flatten_vectors(queries, config_.dimension);
    const unsigned int hardware_threads = std::thread::hardware_concurrency();
    const int gt_threads = static_cast<int>(hardware_threads == 0 ? 8 : hardware_threads);
    auto ground_truth = gt_calc_.compute_all_ground_truth(
        flat_queries.data(),
        queries.size(),
        config_.dimension,
        config_.top_k,
        gt_threads,
        config_.ground_truth_batch_size);

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
    fout << "Preset: " << config_.preset << std::endl;
    fout << "Query mode: " << config_.query_mode << std::endl;
    fout << "Query noise std: " << config_.query_noise_std << std::endl;
    fout << "Normalize queries: " << (config_.normalize_queries ? "true" : "false") << std::endl;
    fout << "Ground truth batch size: " << config_.ground_truth_batch_size << std::endl;
    fout << "Use GPU: " << (config_.use_gpu ? "true" : "false") << std::endl;
    fout << "GPU shard size: " << config_.gpu_shard_size << std::endl;
    fout << "GPU device: " << config_.gpu_device << std::endl;
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
