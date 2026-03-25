#include "vector_test.h"
#include "config.h"
#include "logger.h"
#include "component/timer.h"
#include "component/memory_monitor.h"
#include "component/stat.h"
#include "util.h"
#include <iomanip>
#include <iostream>
#include <stdexcept>

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

    try {
        index->load(config["index_path"]);
    } catch (const std::exception& e) {
        GlobalLogger->error("Load index error: {}", e.what());
    }

    // 读取配置参数
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
    int thread_count = 4;
    if (config.find("threads") != config.end()) {
        thread_count = std::stoi(config["threads"]);
    }
    GlobalLogger->info("使用 {} 线程进行测试", thread_count);

    float* vector_data = nullptr;
    size_t test_count = 0;
    if (config.find("vector_data") != config.end()) {
        // 读取vector_data
        auto [vector_data_, vector_info] = read_fbin(config["vector_data"]);
        vector_data = vector_data_;
        test_count = vector_info.first;
    } else {
        // 生成随机vector
        test_count = std::stoi(config["test_scale"]);
        vector_data = new float[test_count * dim];
        rand_vec(vector_data, dim, test_count);
    }

    // Dynamic writes should use fresh IDs beyond the preloaded index range.
    // The loaded benchmark indexes are built with sequential ids [0, base_count),
    // so reusing (i + 1) turns the workload into accidental updates/conflicts.
    uint32_t write_id_base = 1000000;
    if (config.find("write_id_base") != config.end()) {
        write_id_base = static_cast<uint32_t>(std::stoul(config["write_id_base"]));
    }

    auto test_core = [&](float read_ratio) {
        // 线程安全的统计对象
        Stat stat(2);
        stat.setOperationName(OperationType::WRITE, "Write");
        stat.setOperationName(OperationType::READ, "Read");
        
        // 原子计数器
        std::atomic<int> completed_ops(0);
        std::atomic<int> write_count(0);
        std::atomic<int> read_count(0);
        
        // 进度条参数
        const int progress_bar_width = 50;
        auto start_time = std::chrono::steady_clock::now();
        
        // 工作线程函数
        auto worker = [&](int start, int end) {
            Timer timer;
            for (int i = start; i < end; i++) {
                if (rand() / (float)RAND_MAX < read_ratio) {
                    read_count++;
                    timer.start();
                    std::vector<uint32_t> ids_res;
                    std::vector<float> distances_res;
                    index->search(std::vector<float>(vector_data + i * dim, vector_data + (i + 1) * dim), topk, ids_res, distances_res);
                    timer.stop();
                    stat.addOperation(OperationType::READ, timer.elapsed());
                } else {
                    write_count++;
                    timer.start();
                    index->insert(std::vector<float>(vector_data + i * dim, vector_data + (i + 1) * dim),
                                  {static_cast<uint32_t>(write_id_base + i)});
                    timer.stop();
                    stat.addOperation(OperationType::WRITE, timer.elapsed());
                }
                completed_ops++;
            }
        };
        
        // 进度条显示函数
        auto progress_display = [&]() {
            int last_progress = -1;
            while (completed_ops < static_cast<int>(test_count)) {
                int current_progress = static_cast<int>(
                    static_cast<float>(completed_ops) / test_count * progress_bar_width);
                
                if (current_progress != last_progress) {
                    last_progress = current_progress;
                    auto elapsed = std::chrono::steady_clock::now() - start_time;
                    auto seconds = std::chrono::duration_cast<std::chrono::seconds>(elapsed).count();
                    
                    std::cout << "\r[";
                    for (int j = 0; j < progress_bar_width; j++) {
                        std::cout << (j <= current_progress ? '=' : ' ');
                    }
                    std::cout << "] " 
                            << static_cast<int>(static_cast<float>(completed_ops) / test_count * 100) 
                            << "% (" << completed_ops << "/" << test_count 
                            << ") 已用时间: " << seconds << "秒";
                    std::cout.flush();
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
            
            // 完成进度条
            std::cout << "\r[";
            for (int j = 0; j < progress_bar_width; j++) {
                std::cout << '=';
            }
            std::cout << "] 100% (" << completed_ops << "/" << test_count << ") 已完成\n";
        };
        
        GlobalLogger->info("开始动态测试，总操作数: {}", test_count);
        
        // 创建并启动工作线程
        std::vector<std::thread> threads;
        int ops_per_thread = test_count / thread_count;
        int remainder = test_count % thread_count;
        int start_index = 0;
        
        for (int i = 0; i < thread_count; i++) {
            int end_index = start_index + ops_per_thread + (i < remainder ? 1 : 0);
            threads.emplace_back(worker , start_index, end_index);
            start_index = end_index;
        }
        
        // 启动进度条线程
        std::thread progress_thread(progress_display);
        
        // 等待所有工作线程完成
        for (auto& t : threads) {
            t.join();
        }
        
        // 等待进度条线程完成
        progress_thread.join();
        
        // 计算总耗时
        auto total_time = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - start_time).count();
        
        // 获取统计结果
        double write_total_time = stat.getTotalTime(OperationType::WRITE);
        int write_calls = stat.getCallCount(OperationType::WRITE);
        double read_total_time = stat.getTotalTime(OperationType::READ);
        int read_calls = stat.getCallCount(OperationType::READ);
        
        // 计算延迟和吞吐量
        double write_avg_latency = write_calls > 0 ? write_total_time / write_calls : 0;
        double read_avg_latency = read_calls > 0 ? read_total_time / read_calls : 0;
        
        double write_throughput = total_time > 0 ? static_cast<double>(write_calls) / total_time * 1000 : 0;
        double read_throughput = total_time > 0 ? static_cast<double>(read_calls) / total_time * 1000 : 0;
        double total_throughput = total_time > 0 ? static_cast<double>(read_calls + write_calls) / total_time * 1000 : 0;
        
        // 输出结果
        GlobalLogger->info("动态测试完成，总耗时: {} 毫秒", total_time);
        GlobalLogger->info("Write 操作次数: {}, Read 操作次数: {}", write_calls, read_calls);
        
        GlobalLogger->info("Write 平均延迟: {:.6f} 秒", write_avg_latency);
        GlobalLogger->info("Read 平均延迟: {:.6f} 秒", read_avg_latency);
        
        GlobalLogger->info("Write 吞吐量: {:.2f} 操作/秒", write_throughput);
        GlobalLogger->info("Read 吞吐量: {:.2f} 操作/秒", read_throughput);
        GlobalLogger->info("总吞吐量: {:.2f} 操作/秒", total_throughput);
        
        // 输出详细统计
        stat.printAll();
    };

    bool total_test = false;
    if (config.find("total_test") != config.end()) {
        total_test = config["total_test"] == "true";
    }

    if (total_test) {
        for (int i = 10; i >= 0; i--) {
            GlobalLogger->info("Dynamic test read ratio {}%", i * 10);
            config["read_ratio"] = std::to_string(i * 0.1);
            test_core(i * 0.1);
        }
    } else {
        float read_ratio = 0.5;
        if (config.find("read_ratio") != config.end()) {
            read_ratio = std::stof(config["read_ratio"]);
        }
        GlobalLogger->info("Dynamic test read ratio {}%", read_ratio * 100);
        test_core(read_ratio);
    }

    delete [] vector_data;
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
