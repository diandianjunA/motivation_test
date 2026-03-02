#include <cstddef>
#include <iomanip>
#include <iostream>
#include "random_data_generator.h"

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cout << "Usage: " << argv[0] << " <input_file>" << std::endl;
        std::cout << "Reads fbin binary format and prints statistics" << std::endl;
        return 1;
    }
    
    size_t display = 5;
    try {
        DataConfig test_config = {
            /*num_vectors=*/100000,
            /*num_queries=*/1000,
            /*dimension=*/1024,
            /*top_k=*/10,
            /*data_min=*/-1.0f,
            /*data_max=*/1.0f,
            /*seed=*/42,
            /*distribution=*/"normal",
            /*output_dir=*/"/data/xjs/random_dataset/1024dim100K"
        };
        RandomDataGenerator generator(test_config);
        auto database = generator.load_vectors(test_config.output_dir + "/base.fbin");

        std::cout << "向量数量: " << database.size() << std::endl;
        std::cout << "每个向量维度: " << database[0].size() << std::endl;

        for (size_t i = 0; i < database.size(); i++) {
            std::cout << "向量 " << i << ": [";
            if (display > database[0].size()) {
                for (size_t j = 0; j < database[0].size(); j++) {
                    std::cout << std::fixed << std::setprecision(6) << database[i][j] << ", ";
                }
                std::cout << "]";
            } else {
                for (size_t j = 0; j < display; j++) {
                    std::cout << std::fixed << std::setprecision(6) << database[i][j] << ", ";
                }
                std::cout << "..., ";
                for (size_t j = database[0].size() - display; j < database[0].size(); j++) {
                    std::cout << std::fixed << std::setprecision(6) << database[i][j] << ", ";
                }
                std::cout << "]";
            }
            std::cout << std::endl;
        }

        // delete[] data;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}