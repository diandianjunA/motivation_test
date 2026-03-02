#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

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


int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cout << "Usage: " << argv[0] << " <input_file>" << std::endl;
        std::cout << "Reads fbin binary format and prints statistics" << std::endl;
        return 1;
    }
    
    size_t display = 5;
    try {
        auto [data, shape] = mmap_fbin(argv[1]);
        auto [num_vectors, dim] = shape;

        std::cout << "向量数量: " << num_vectors << std::endl;
        std::cout << "每个向量维度: " << dim << std::endl;

        for (size_t i = 0; i < num_vectors; i++) {
            std::cout << "向量 " << i << ": [";
            if (display > dim) {
                for (size_t j = 0; j < dim; j++) {
                    std::cout << std::fixed << std::setprecision(6) << data[i * dim + j] << ", ";
                }
                std::cout << "]";
            } else {
                for (size_t j = 0; j < display; j++) {
                    std::cout << std::fixed << std::setprecision(6) << data[i * dim + j] << ", ";
                }
                std::cout << "..., ";
                for (size_t j = dim - display; j < dim; j++) {
                    std::cout << std::fixed << std::setprecision(6) << data[i * dim + j] << ", ";
                }
                std::cout << "]";
            }
            std::cout << std::endl;
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}