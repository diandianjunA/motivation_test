#include <string>
#include <iostream>
#include <fstream>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstring>
#include <omp.h>

#include "ground_truth_calculator.h"

struct MappedData {
    float* data;
    size_t num_vectors;
    size_t dim;
    int fd;
    void* mapped_addr;
    size_t mapped_size;
};

MappedData mmap_bin(const std::string& file_path) {
    int fd = open(file_path.c_str(), O_RDONLY);
    if (fd == -1) {
        throw std::runtime_error("无法打开文件: " + file_path);
    }

    struct stat sb;
    if (fstat(fd, &sb) == -1) {
        close(fd);
        throw std::runtime_error("无法获取文件大小: " + file_path);
    }

    size_t file_size = sb.st_size;
    void* mapped_addr = mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (mapped_addr == MAP_FAILED) {
        close(fd);
        throw std::runtime_error("mmap失败: " + file_path);
    }

    madvise(mapped_addr, file_size, MADV_WILLNEED);

    char* data = static_cast<char*>(mapped_addr);
    
    if (file_size < 2 * sizeof(uint32_t)) {
        munmap(mapped_addr, file_size);
        close(fd);
        throw std::runtime_error("文件太小，不是有效的fbin文件");
    }

    uint32_t num_vectors = *reinterpret_cast<uint32_t*>(data);
    uint32_t dim = *reinterpret_cast<uint32_t*>(data + sizeof(uint32_t));
    
    std::cout << "mmap fbin文件: " << file_path << std::endl;
    std::cout << "向量数量: " << num_vectors << ", 每个向量维度: " << dim << std::endl;

    size_t expected_size = 2 * sizeof(uint32_t) + static_cast<size_t>(num_vectors) * dim * sizeof(float);
    if (file_size < expected_size) {
        munmap(mapped_addr, file_size);
        close(fd);
        throw std::runtime_error("文件大小不匹配: 期望 " + std::to_string(expected_size) + 
                                " 字节, 实际 " + std::to_string(file_size) + " 字节");
    }

    float* float_data = reinterpret_cast<float*>(data + 2 * sizeof(uint32_t));
    
    MappedData result;
    result.data = float_data;
    result.num_vectors = num_vectors;
    result.dim = dim;
    result.fd = fd;
    result.mapped_addr = mapped_addr;
    result.mapped_size = file_size;
    
    std::cout << "mmap映射完成: " << num_vectors << " 个向量, 每个向量 " << dim << " 维" << std::endl;
    
    return result;
}

void unmap_bin(MappedData& mapped) {
    if (mapped.mapped_addr != nullptr && mapped.mapped_addr != MAP_FAILED) {
        munmap(mapped.mapped_addr, mapped.mapped_size);
    }
    if (mapped.fd != -1) {
        close(mapped.fd);
    }
    mapped.data = nullptr;
    mapped.fd = -1;
    mapped.mapped_addr = nullptr;
}

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0]
                  << " <base.fbin> <queries.fbin> <groundtruth.bin>"
                  << " [--gpu] [--gpu-device N] [--gpu-shard-size N] [--topk K] [--batch-size N]"
                  << std::endl;
        return 1;
    }
    
    std::string base_path = argv[1];
    std::string queries_path = argv[2];
    std::string groundtruth_path = argv[3];
    bool use_gpu = false;
    int gpu_device = -1;
    size_t gpu_shard_size = 0;
    size_t top_k = 10;
    size_t batch_size = 64;

    for (int i = 4; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--gpu") {
            use_gpu = true;
        } else if (arg == "--gpu-device" && i + 1 < argc) {
            gpu_device = std::stoi(argv[++i]);
        } else if (arg == "--gpu-shard-size" && i + 1 < argc) {
            gpu_shard_size = static_cast<size_t>(std::stoull(argv[++i]));
        } else if (arg == "--topk" && i + 1 < argc) {
            top_k = static_cast<size_t>(std::stoull(argv[++i]));
        } else if (arg == "--batch-size" && i + 1 < argc) {
            batch_size = static_cast<size_t>(std::stoull(argv[++i]));
        } else {
            std::cerr << "Unknown or incomplete argument: " << arg << std::endl;
            return 1;
        }
    }
    
    MappedData base_mapped = mmap_bin(base_path);
    
    GroundTruthCalculator calculator(
        base_mapped.dim,
        DistanceMetric::L2,
        use_gpu,
        gpu_shard_size,
        gpu_device);

    std::cout << "初始化Ground Truth计算器..." << std::endl;
    calculator.init(base_mapped.data, base_mapped.num_vectors, base_mapped.dim);

    if (!use_gpu) {
        std::cout << "释放base数据mmap..." << std::endl;
        unmap_bin(base_mapped);
    }

    MappedData queries_mapped = mmap_bin(queries_path);
    
    std::cout << "读取查询数据完成: " << queries_mapped.num_vectors << " 个查询, 每个查询 " << queries_mapped.dim << " 维" << std::endl;
    
    std::cout << "开始计算所有查询的Ground Truth..." << std::endl;
    auto ground_truths = calculator.compute_all_ground_truth(
        queries_mapped.data,
        queries_mapped.num_vectors,
        queries_mapped.dim,
        top_k,
        omp_get_max_threads(),
        batch_size);
    unmap_bin(queries_mapped);
    if (use_gpu) {
        std::cout << "释放base数据mmap..." << std::endl;
        unmap_bin(base_mapped);
    }
    
    std::cout << "Ground Truth计算完成: " << ground_truths.size() << " 个查询" << std::endl;
    
    std::cout << "开始保存Ground Truth..." << std::endl;
    calculator.save_ground_truth(groundtruth_path, ground_truths);

    std::cout << "Ground truth 计算完成并保存至: " << groundtruth_path << std::endl;

    return 0;
}
