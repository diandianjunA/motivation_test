#include "vector_test.h"
#include "config.h"
#include "logger.h"
#include "component/timer.h"
#include "component/memory_monitor.h"
#include "component/stat.h"
#include "util.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <sstream>
#include <stdexcept>
#include <thread>

namespace {

int getRequiredInt(const std::map<std::string, std::string>& config, const std::string& key) {
    const auto it = config.find(key);
    if (it == config.end()) {
        throw std::runtime_error("Config file missing " + key);
    }
    return std::stoi(it->second);
}

int getOptionalInt(const std::map<std::string, std::string>& config, const std::string& key, int defaultValue) {
    const auto it = config.find(key);
    if (it == config.end()) {
        return defaultValue;
    }
    return std::stoi(it->second);
}

size_t getOptionalSize(
    const std::map<std::string, std::string>& config,
    const std::string& key,
    size_t defaultValue) {
    const auto it = config.find(key);
    if (it == config.end()) {
        return defaultValue;
    }
    return static_cast<size_t>(std::stoull(it->second));
}

double getOptionalDouble(
    const std::map<std::string, std::string>& config,
    const std::string& key,
    double defaultValue) {
    const auto it = config.find(key);
    if (it == config.end()) {
        return defaultValue;
    }
    return std::stod(it->second);
}

bool getOptionalBool(const std::map<std::string, std::string>& config, const std::string& key, bool defaultValue) {
    const auto it = config.find(key);
    if (it == config.end()) {
        return defaultValue;
    }
    return it->second == "true";
}

void ensureRange(int value, int minValue, int maxValue, const std::string& key) {
    if (value < minValue || value > maxValue) {
        throw std::runtime_error(
            key + " must be in [" + std::to_string(minValue) + ", " + std::to_string(maxValue) + "]");
    }
}

void ensurePositive(int value, const std::string& key) {
    if (value <= 0) {
        throw std::runtime_error(key + " must be positive");
    }
}

void ensurePositiveSize(size_t value, const std::string& key) {
    if (value == 0) {
        throw std::runtime_error(key + " must be positive");
    }
}

} // namespace

std::string genetateLogFileName() {
    auto now = std::chrono::system_clock::now();
    std::time_t now_time = std::chrono::system_clock::to_time_t(now);
    std::tm local_time = *std::localtime(&now_time);

    std::ostringstream oss;
    oss << std::put_time(&local_time, "%Y%m%d-%H-%M.log");
    return oss.str();
}

VectorTest::VectorTest(const std::string& conf, std::shared_ptr<VectorIndex> index) {
    // 初始化全局日志记录器
    init_global_logger();
    set_log_level(spdlog::level::debug);
    GlobalLogger->info("Global logger initialized");

    // 读取配置文件
    this->config = readConfig(conf);

    const std::string log_path = config["log_path"];
    if (log_path.empty()) {
        throw std::runtime_error("Config file missing log_path");
    }
    // 日志名用xxxx年x月x日x时x分x秒的格式化字符串
    std::string log_name = log_path + "/" + genetateLogFileName();
    add_file_sink(log_name);

    this->index = index;
    for (const auto& [key, value] : config) {
        GlobalLogger->info("Config: {} = {}", key, value);
    }

    // 读取option
    if (config.find("option") == config.end()) {
        throw std::runtime_error("Config file missing option");
    }
    std::string option = config["option"];
    if (option == "build") {
        build();
    } else if (option == "storage") {
        storage_test();
    } else if (option == "dynamic") {
        dynamic_test();
    } else if (option == "recall") {
        recall_test();
    } else {
        throw std::runtime_error("Unknown option");
    }
}

VectorTest::~VectorTest() {
    
}

void VectorTest::build() {
    GlobalLogger->info("Build index {}", index->getIndexType());

    if (config.find("data_path") == config.end()) {
        throw std::runtime_error("Config file missing data_path");
    }
    if (config.find("index_path") == config.end()) {
        throw std::runtime_error("Config file missing index_path");
    }

    const std::string dataset_path = config["data_path"];
    Timer timer;
    timer.start();
    try {
        index->build(dataset_path);
    } catch (const std::exception& e) {
        GlobalLogger->error("Build index error: {}", e.what());
    }
    timer.stop();
    GlobalLogger->info("Build time: {} s", timer.elapsed());
    index->save(config["index_path"]);
}

void VectorTest::storage_test() {
    GlobalLogger->info("Storage test {}", index->getIndexType());
    MemoryMonitor mem_monitor;
    mem_monitor.start();
    try {
        GlobalLogger->info("Load index {}", config["index_path"]);
        index->load(config["index_path"]);
    } catch (const std::exception& e) {
        GlobalLogger->error("Load index error: {}", e.what());
    }

    int dim = 0;
    if (config.find("dim") != config.end()) {
        dim = std::stoi(config["dim"]);
    } else {
        throw std::runtime_error("Config file missing dim");
    }
    int topk = 10;
    if (config.find("topk") != config.end()) {
        topk = std::stoi(config["topk"]);
    }

    float* query = nullptr;
    size_t query_count = 0;
    if (config.find("query_data") != config.end()) {
        // 读取query_data
        auto [query_data, query_info] = read_fbin(config["query_data"]);
        query = query_data;
        query_count = query_info.first;
    } else {
        // 生成随机query
        query_count = std::stoi(config["test_scale"]);
        query = new float[query_count * dim];
        rand_vec(query, dim, query_count);
    }
    std::vector<uint32_t> ids_res(topk, 0);
    std::vector<float> distances_res(topk, 0.0);
    for (size_t i = 0; i < query_count; i++) {
        index->search(std::vector<float>(query + i * dim, query + (i + 1) * dim), topk, ids_res, distances_res);
    }
    std::this_thread::sleep_for(std::chrono::seconds(10));
    delete[] query;
    mem_monitor.stop();
    GlobalLogger->info("Storage test memory usage: {} %", mem_monitor.getAverageMemoryUsage());
}

void VectorTest::dynamic_test() {
    GlobalLogger->info("Dynamic test {}", index->getIndexType());
    if (config.find("index_path") == config.end()) {
        throw std::runtime_error("Config file missing index_path");
    }

    const int dim = getRequiredInt(config, "dim");
    const int topk = getOptionalInt(config, "topk", 10);
    const int thread_count = getOptionalInt(config, "threads", 4);
    const int read_batch_size = getOptionalInt(config, "read_batch_size", 1);
    const int insert_batch_size = getOptionalInt(config, "insert_batch_size", 1);
    const int warmup_seconds = getOptionalInt(config, "warmup_seconds", 30);
    const int run_seconds = getOptionalInt(config, "run_seconds", 60);
    const float read_ratio = static_cast<float>(getOptionalDouble(config, "read_ratio", 0.5));
    const bool total_test = getOptionalBool(config, "total_test", false);
    const uint32_t write_id_base = static_cast<uint32_t>(getOptionalSize(config, "write_id_base", 1000000));

    ensurePositive(dim, "dim");
    ensurePositive(topk, "topk");
    ensurePositive(thread_count, "threads");
    ensurePositive(read_batch_size, "read_batch_size");
    ensurePositive(insert_batch_size, "insert_batch_size");
    ensureRange(warmup_seconds, 30, 60, "warmup_seconds");
    ensureRange(run_seconds, 60, 120, "run_seconds");
    if (read_ratio < 0.0F || read_ratio > 1.0F) {
        throw std::runtime_error("read_ratio must be in [0, 1]");
    }

    GlobalLogger->info("使用 {} 线程进行动态测试", thread_count);
    GlobalLogger->info(
        "动态测试配置: read_batch_size={}, insert_batch_size={}, warmup_seconds={}, run_seconds={}",
        read_batch_size,
        insert_batch_size,
        warmup_seconds,
        run_seconds);

    size_t data_pool_size = 0;
    std::unique_ptr<float[]> vector_data_holder;
    if (config.find("vector_data") != config.end()) {
        auto [vector_data, vector_info] = read_fbin(config["vector_data"]);
        if (static_cast<int>(vector_info.second) != dim) {
            delete[] vector_data;
            throw std::runtime_error("vector_data dim does not match config dim");
        }
        data_pool_size = vector_info.first;
        vector_data_holder.reset(vector_data);
    } else {
        if (config.find("data_pool_size") != config.end()) {
            data_pool_size = getOptionalSize(config, "data_pool_size", 0);
        } else if (config.find("test_scale") != config.end()) {
            data_pool_size = getOptionalSize(config, "test_scale", 0);
            GlobalLogger->warn(
                "Dynamic mode no longer uses test_scale as operation count; using test_scale={} as data_pool_size for backward compatibility",
                data_pool_size);
        } else {
            throw std::runtime_error("Dynamic mode requires vector_data or data_pool_size");
        }
        ensurePositiveSize(data_pool_size, "data_pool_size");
        vector_data_holder = std::make_unique<float[]>(data_pool_size * static_cast<size_t>(dim));
        rand_vec(vector_data_holder.get(), dim, static_cast<int>(data_pool_size));
        GlobalLogger->info("Generated random dynamic data pool with {} vectors", data_pool_size);
    }
    ensurePositiveSize(data_pool_size, "data_pool_size");
    const float* vector_data = vector_data_holder.get();

    struct PhaseResult {
        double elapsed_seconds = 0.0;
        uint64_t read_vectors = 0;
        uint64_t write_vectors = 0;
        uint64_t read_batches = 0;
        uint64_t write_batches = 0;
        std::shared_ptr<Stat> stat;
    };

    auto run_scenario = [&](float scenario_read_ratio) {
        try {
            index->load(config["index_path"]);
        } catch (const std::exception& e) {
            GlobalLogger->error("Load index error: {}", e.what());
            throw;
        }

        std::atomic<size_t> read_cursor(0);
        std::atomic<size_t> write_cursor(0);
        std::atomic<uint64_t> next_write_id(write_id_base);

        auto run_phase = [&](const std::string& phase_name, int target_seconds, bool collect_stats) {
            PhaseResult result;
            result.stat = std::make_shared<Stat>(2);
            result.stat->setOperationName(OperationType::WRITE, "Write");
            result.stat->setOperationName(OperationType::READ, "Read");

            std::atomic<uint64_t> phase_read_vectors(0);
            std::atomic<uint64_t> phase_write_vectors(0);
            std::atomic<uint64_t> phase_read_batches(0);
            std::atomic<uint64_t> phase_write_batches(0);
            std::atomic<bool> phase_done(false);

            const auto phase_start = std::chrono::steady_clock::now();
            const auto deadline = phase_start + std::chrono::seconds(target_seconds);

            auto progress_display = [&]() {
                const int progress_bar_width = 50;
                while (!phase_done.load()) {
                    const double elapsed_seconds =
                        std::chrono::duration<double>(std::chrono::steady_clock::now() - phase_start).count();
                    const double progress = std::min(elapsed_seconds / static_cast<double>(target_seconds), 1.0);
                    const int current_progress = static_cast<int>(progress * progress_bar_width);

                    std::cout << "\r" << phase_name << " [";
                    for (int i = 0; i < progress_bar_width; ++i) {
                        std::cout << (i < current_progress ? '=' : ' ');
                    }
                    std::cout << "] " << static_cast<int>(progress * 100.0) << "% "
                              << "elapsed=" << std::fixed << std::setprecision(1) << elapsed_seconds << "s/"
                              << target_seconds << "s "
                              << "read_vectors=" << phase_read_vectors.load() << " "
                              << "write_vectors=" << phase_write_vectors.load();
                    std::cout.flush();
                    std::this_thread::sleep_for(std::chrono::milliseconds(200));
                }

                const double elapsed_seconds =
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - phase_start).count();
                std::cout << "\r" << phase_name << " [";
                for (int i = 0; i < progress_bar_width; ++i) {
                    std::cout << '=';
                }
                std::cout << "] 100% "
                          << "elapsed=" << std::fixed << std::setprecision(1) << elapsed_seconds << "s "
                          << "read_vectors=" << phase_read_vectors.load() << " "
                          << "write_vectors=" << phase_write_vectors.load() << "        \n";
                std::cout.flush();
            };

            auto worker = [&](int thread_index) {
                std::mt19937 rng(static_cast<uint32_t>(std::random_device{}()) + static_cast<uint32_t>(thread_index * 9973));
                std::uniform_real_distribution<float> choice_dist(0.0F, 1.0F);
                std::vector<float> query_buffer(static_cast<size_t>(dim));
                std::vector<uint32_t> ids_res(static_cast<size_t>(topk));
                std::vector<float> distances_res(static_cast<size_t>(topk));
                std::vector<float> insert_buffer(static_cast<size_t>(insert_batch_size) * static_cast<size_t>(dim));
                std::vector<uint32_t> insert_ids(static_cast<size_t>(insert_batch_size));

                while (true) {
                    if (std::chrono::steady_clock::now() >= deadline) {
                        break;
                    }

                    const bool is_read = choice_dist(rng) < scenario_read_ratio;
                    const auto op_start = std::chrono::steady_clock::now();

                    if (is_read) {
                        const size_t batch_start = read_cursor.fetch_add(static_cast<size_t>(read_batch_size));
                        for (int batch_index = 0; batch_index < read_batch_size; ++batch_index) {
                            const size_t source_index =
                                (batch_start + static_cast<size_t>(batch_index)) % data_pool_size;
                            std::copy_n(
                                vector_data + source_index * static_cast<size_t>(dim),
                                static_cast<size_t>(dim),
                                query_buffer.begin());
                            index->search(query_buffer, static_cast<size_t>(topk), ids_res, distances_res);
                        }
                        const double latency =
                            std::chrono::duration<double>(std::chrono::steady_clock::now() - op_start).count();
                        phase_read_vectors.fetch_add(static_cast<uint64_t>(read_batch_size));
                        phase_read_batches.fetch_add(1);
                        if (collect_stats) {
                            result.stat->addOperation(
                                OperationType::READ,
                                latency,
                                static_cast<uint64_t>(read_batch_size));
                        }
                    } else {
                        const size_t batch_start = write_cursor.fetch_add(static_cast<size_t>(insert_batch_size));
                        const uint64_t id_start = next_write_id.fetch_add(static_cast<uint64_t>(insert_batch_size));
                        for (int batch_index = 0; batch_index < insert_batch_size; ++batch_index) {
                            const size_t source_index =
                                (batch_start + static_cast<size_t>(batch_index)) % data_pool_size;
                            std::copy_n(
                                vector_data + source_index * static_cast<size_t>(dim),
                                static_cast<size_t>(dim),
                                insert_buffer.begin() +
                                    static_cast<size_t>(batch_index) * static_cast<size_t>(dim));
                            insert_ids[static_cast<size_t>(batch_index)] =
                                static_cast<uint32_t>(id_start + static_cast<uint64_t>(batch_index));
                        }
                        index->insert(insert_buffer, insert_ids);
                        const double latency =
                            std::chrono::duration<double>(std::chrono::steady_clock::now() - op_start).count();
                        phase_write_vectors.fetch_add(static_cast<uint64_t>(insert_batch_size));
                        phase_write_batches.fetch_add(1);
                        if (collect_stats) {
                            result.stat->addOperation(
                                OperationType::WRITE,
                                latency,
                                static_cast<uint64_t>(insert_batch_size));
                        }
                    }
                }
            };

            std::vector<std::thread> workers;
            workers.reserve(static_cast<size_t>(thread_count));
            for (int thread_index = 0; thread_index < thread_count; ++thread_index) {
                workers.emplace_back(worker, thread_index);
            }
            std::thread progress_thread(progress_display);

            for (auto& worker_thread : workers) {
                worker_thread.join();
            }
            phase_done.store(true);
            progress_thread.join();

            result.elapsed_seconds =
                std::chrono::duration<double>(std::chrono::steady_clock::now() - phase_start).count();
            result.read_vectors = phase_read_vectors.load();
            result.write_vectors = phase_write_vectors.load();
            result.read_batches = phase_read_batches.load();
            result.write_batches = phase_write_batches.load();
            return result;
        };

        GlobalLogger->info(
            "开始动态测试: read_ratio={:.0f}%, warmup={}s, run={}s",
            scenario_read_ratio * 100.0F,
            warmup_seconds,
            run_seconds);
        const auto warmup_result = run_phase("warmup", warmup_seconds, false);
        GlobalLogger->info(
            "预热完成: elapsed={:.2f}s, read_vectors={}, write_vectors={}",
            warmup_result.elapsed_seconds,
            warmup_result.read_vectors,
            warmup_result.write_vectors);

        const auto measured_result = run_phase("run", run_seconds, true);
        const double measured_seconds = measured_result.elapsed_seconds;
        const uint64_t total_vectors = measured_result.read_vectors + measured_result.write_vectors;
        const double total_throughput =
            measured_seconds > 0.0 ? static_cast<double>(total_vectors) / measured_seconds : 0.0;
        const double qps =
            measured_seconds > 0.0 ? static_cast<double>(measured_result.read_vectors) / measured_seconds : 0.0;
        const double ips =
            measured_seconds > 0.0 ? static_cast<double>(measured_result.write_vectors) / measured_seconds : 0.0;

        GlobalLogger->info("Dynamic test completed, measured time: {:.3f} s", measured_seconds);
        GlobalLogger->info(
            "Measured counts: read_batches={}, write_batches={}, read_vectors={}, write_vectors={}",
            measured_result.read_batches,
            measured_result.write_batches,
            measured_result.read_vectors,
            measured_result.write_vectors);
        GlobalLogger->info("Total Throughput: {:.2f} vectors/sec", total_throughput);
        GlobalLogger->info("QPS: {:.2f}", qps);
        GlobalLogger->info("IPS: {:.2f}", ips);

        const auto log_percentiles = [&](OperationType type, const std::string& name) {
            const auto p50 = measured_result.stat->getPercentile(type, 50.0);
            const auto p90 = measured_result.stat->getPercentile(type, 90.0);
            const auto p99 = measured_result.stat->getPercentile(type, 99.0);
            if (!p50.has_value() || !p90.has_value() || !p99.has_value()) {
                GlobalLogger->info("{} Latency Percentiles: P50=N/A, P90=N/A, P99=N/A", name);
                return;
            }
            GlobalLogger->info(
                "{} Latency Percentiles: P50={:.3f} ms, P90={:.3f} ms, P99={:.3f} ms",
                name,
                p50.value() * 1000.0,
                p90.value() * 1000.0,
                p99.value() * 1000.0);
        };

        log_percentiles(OperationType::READ, "Read");
        log_percentiles(OperationType::WRITE, "Write");
        measured_result.stat->printAll();
    };

    if (total_test) {
        for (int ratio_percent = 100; ratio_percent >= 0; ratio_percent -= 10) {
            run_scenario(static_cast<float>(ratio_percent) / 100.0F);
        }
    } else {
        run_scenario(read_ratio);
    }
}

void VectorTest::recall_test() {
    GlobalLogger->info("Recall test {}", index->getIndexType());
    try {
        index->load(config["index_path"]);
    } catch (const std::exception& e) {
        GlobalLogger->error("Load index error: {}", e.what());
    }

    if (config.find("query_data") == config.end()) {
        throw std::runtime_error("Config file missing query_data");
    }
    if (config.find("groundtruth") == config.end()) {
        throw std::runtime_error("Config file missing groundtruth");
    }

    // 读取query_data
    auto [query_data, query_info] = read_fbin(config["query_data"]);
    int dim = query_info.second;
    int topk = 10;
    if (config.find("topk") != config.end()) {
        topk = std::stoi(config["topk"]);
    }
    std::vector<std::vector<uint32_t>> ids_res(query_info.first, std::vector<uint32_t>(topk, 0));
    std::vector<std::vector<float>> distances_res(query_info.first, std::vector<float>(topk, 0.0));
    for (size_t i = 0; i < query_info.first; i++) {
        index->search(std::vector<float>(query_data + i * dim, query_data + (i + 1) * dim), topk, ids_res[i], distances_res[i]);
        if (i % 10 == 0) {
            GlobalLogger->info("Search query ({}/{}) done", i, query_info.first);
        }
    }
    delete[] query_data;

    // 读取groundtruth
    auto [groundtruth, groundtruth_info] = read_bin(config["groundtruth"]);
    int groundtruth_num = groundtruth_info.first;
    if (groundtruth_num != (int)query_info.first) {
        throw std::runtime_error("Groundtruth number does not match query number");
    }
    int groundtruth_topk = groundtruth_info.second;
    if (groundtruth_topk != topk) {
        throw std::runtime_error("Groundtruth topk does not match test topk");
    }
    
    // 计算recall
    double recall = 0.0;
    for (size_t i = 0; i < query_info.first; i++) {
        double recall_per_query = 0.0;
        for (int j = 0; j < topk; j++) {
            if (std::find(ids_res[i].begin(), ids_res[i].end(), groundtruth[i * topk + j]) != ids_res[i].end()) {
                recall_per_query++;
            }
        }
        // GlobalLogger->info("Groundtruth: ");
        // for (int j = 0; j < topk; j++) {
        //     GlobalLogger->info("{}", groundtruth[i * topk + j]);
        // }
        // GlobalLogger->info("DiskANN result: ");
        // for (int j = 0; j < topk; j++) {
        //     GlobalLogger->info("ids_res[{}][{}] = {}, distances_res[{}][{}] = {:.4f}", i, j, ids_res[i][j], i, j, distances_res[i][j]);
        // }
        GlobalLogger->info("Recall per query: {:.4f}", recall_per_query / topk);
        recall += recall_per_query / topk;
    }
    GlobalLogger->info("Recall: {:.4f}", recall / query_info.first);
    delete[] groundtruth;
}
