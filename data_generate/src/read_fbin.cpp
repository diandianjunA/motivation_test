#include <cstddef>
#include <utility>
#include <cstdint>
#include <iostream>
#include <fstream>
#include <string>

std::pair<float*, std::pair<size_t, size_t>> read_bin(const std::string& file_path) {
    std::ifstream file(file_path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("无法打开文件: " + file_path);
    }

    // 读取头部: [num_vector][dim]
    uint32_t num_vectors, dim;
    file.read(reinterpret_cast<char*>(&num_vectors), sizeof(uint32_t));
    file.read(reinterpret_cast<char*>(&dim), sizeof(uint32_t));
    
    std::cout << "读取fbin文件: " << file_path << std::endl;
    std::cout << "向量数量: " << num_vectors << ", 每个向量维度: " << dim << std::endl;
    
    // 计算数据大小
    size_t data_size = static_cast<size_t>(num_vectors) * dim;
    
    // 分配内存
    float* data = new float[data_size];
    
    // 读取所有向量数据
    file.read(reinterpret_cast<char*>(data), data_size * sizeof(float));
    
    // 检查读取是否成功
    if (!file) {
        delete[] data;
        throw std::runtime_error("文件读取不完整: 期望 " + 
                                std::to_string(data_size * sizeof(float)) + 
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
    
    std::cout << "读取完成: " << num_vectors << " 个向量, 每个向量 " << dim << " 维" << std::endl;
    
    return {data, {num_vectors, dim}};
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cout << "Usage: " << argv[0] << " <input_file>" << std::endl;
        std::cout << "Reads fbin binary format and prints statistics" << std::endl;
        return 1;
    }
    
    size_t display = 5;
    try {
        auto [data, info] = read_bin(argv[1]);

        std::cout << "向量数量: " << info.first << std::endl;
        std::cout << "每个向量维度: " << info.second << std::endl;

        for (size_t i = 0; i < info.first; i++) {
            std::cout << "向量 " << i << ": [";
            if (display > info.second) {
                for (size_t j = 0; j < info.second; j++) {
                    std::cout << data[i * info.second + j] << ", ";
                }
                std::cout << "]";
            } else {
                for (size_t j = 0; j < display; j++) {
                    std::cout << data[i * info.second + j] << ", ";
                }
                std::cout << "..., ";
                for (size_t j = info.second - display; j < info.second; j++) {
                    std::cout << data[i * info.second + j] << ", ";
                }
                std::cout << "]";
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