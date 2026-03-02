#include <utility>
#include <cstdint>
#include <iostream>
#include <fstream>
#include <string>

std::pair<uint32_t*, std::pair<size_t, size_t>> read_bin(const std::string& file_path) {
    std::ifstream file(file_path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("无法打开文件: " + file_path);
    }

    // 读取头部: [num_queries][k]
    uint32_t num_queries, k;
    file.read(reinterpret_cast<char*>(&num_queries), sizeof(uint32_t));
    file.read(reinterpret_cast<char*>(&k), sizeof(uint32_t));
    
    std::cout << "读取ground truth文件: " << file_path << std::endl;
    std::cout << "查询数量: " << num_queries << ", 每个查询返回k: " << k << std::endl;
    
    // 计算数据大小
    size_t data_size = static_cast<size_t>(num_queries) * k;
    
    // 分配内存
    uint32_t* data = new uint32_t[data_size];
    
    // 读取所有邻居ID
    file.read(reinterpret_cast<char*>(data), data_size * sizeof(uint32_t));
    
    // 检查读取是否成功
    if (!file) {
        delete[] data;
        throw std::runtime_error("文件读取不完整: 期望 " + 
                                std::to_string(data_size * sizeof(uint32_t)) + 
                                " 字节, 实际读取了 " + 
                                std::to_string(file.gcount()) + " 字节");
    }
    
    // 检查是否还有多余数据
    char extra_byte;
    file.read(&extra_byte, 1);
    if (!file.eof()) {
        delete[] data;
        throw std::runtime_error("文件包含额外数据，格式不正确");
    }
    
    file.close();
    
    std::cout << "读取完成: " << num_queries << " 个查询, 每个查询 " << k << " 个邻居" << std::endl;
    
    return {data, {num_queries, k}};
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cout << "Usage: " << argv[0] << " <input_file>" << std::endl;
        std::cout << "Reads ground truth binary format and prints statistics" << std::endl;
        return 1;
    }
    
    try {
        auto [data, info] = read_bin(argv[1]);

        std::cout << "Groundtruth number: " << info.first << std::endl;
        std::cout << "Groundtruth topk: " << info.second << std::endl;

        for (size_t i = 0; i < info.first; i++) {
            std::cout << "Groundtruth " << i << ": ";
            for (size_t j = 0; j < info.second; j++) {
                std::cout << data[i * info.second + j] << " ";
            }
            std::cout << std::endl;
        }

        delete[] data;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}