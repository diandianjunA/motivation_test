#include "util.h"
#include <fstream>
#include <fcntl.h>
#include <sys/mman.h>
#include <cstring>
#include <unistd.h>
#include <sys/stat.h>
#include <iostream>

std::pair<float*, std::pair<size_t, size_t>> read_fbin(const std::string& file_path) {
    std::ifstream file(file_path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("无法打开文件: " + file_path);
    }

    // 读取头部: [uint32_t num_vectors][uint32_t dim]
    uint32_t num_vectors, dim;
    file.read(reinterpret_cast<char*>(&num_vectors), sizeof(uint32_t));
    file.read(reinterpret_cast<char*>(&dim), sizeof(uint32_t));
    
    std::cout << "读取bigANN .bin文件: " << file_path << std::endl;
    std::cout << "向量数量: " << num_vectors << ", 维度: " << dim << std::endl;
    
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
    
    std::cout << "读取完成: " << num_vectors << " 个向量" << std::endl;
    
    return {data, {num_vectors, dim}};
}

// 内存映射方式读取 fbin 文件
std::pair<const float*, std::pair<size_t, size_t>> mmap_fbin(const std::string& file_path) {
    int fd = open(file_path.c_str(), O_RDONLY);
    if (fd == -1) {
        throw std::runtime_error("无法打开文件: " + file_path);
    }
    
    // 获取文件大小
    struct stat sb;
    if (fstat(fd, &sb) == -1) {
        close(fd);
        throw std::runtime_error("无法获取文件大小: " + file_path);
    }
    
    size_t file_size = sb.st_size;
    
    std::cout << "内存映射bigANN .bin文件: " << file_path 
              << " (" << file_size << " 字节)" << std::endl;
    
    // 映射整个文件
    void* addr = mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (addr == MAP_FAILED) {
        close(fd);
        throw std::runtime_error("内存映射失败: " + file_path);
    }
    
    close(fd);
    
    // 解析头部
    const uint32_t* header = reinterpret_cast<const uint32_t*>(addr);
    uint32_t num_vectors = header[0];
    uint32_t dim = header[1];
    
    std::cout << "向量数量: " << num_vectors << ", 维度: " << dim << std::endl;
    
    // 验证文件大小
    size_t expected_size = 8 + static_cast<size_t>(num_vectors) * dim * sizeof(float);
    if (file_size != expected_size) {
        munmap(addr, file_size);
        throw std::runtime_error("文件大小不匹配: 期望 " + 
                                 std::to_string(expected_size) + 
                                 " 字节, 得到 " + std::to_string(file_size));
    }
    
    // 数据指针：跳过8字节的头部
    const float* data = reinterpret_cast<const float*>(header + 2);
    
    std::cout << "内存映射完成: " << num_vectors << " 个向量" << std::endl;
    
    return {data, {num_vectors, dim}};
}

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


std::vector<float> rand_vec(int dim, int count) {
    std::vector<float> vec(dim * count);
    for (int i = 0; i < dim * count; i++) {
        vec[i] = static_cast<float>(rand()) / RAND_MAX;
    }
    return vec;
}

void rand_vec(float* vec, int dim, int count) {
    for (int i = 0; i < dim * count; i++) {
        vec[i] = static_cast<float>(rand()) / RAND_MAX;
    }
}
