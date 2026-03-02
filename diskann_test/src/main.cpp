#include "diskann_index.h"
#include "vector_test/config.h"
#include "vector_test/vector_test.h"
#include <iostream>

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <config_file>" << std::endl;
        return EXIT_FAILURE;
    }

    std::string config_file = argv[1];
    auto config = readConfig(config_file);

    if (config.find("index_conf") == config.end()) {
        std::cerr << "index_conf not found in config file" << std::endl;
        return EXIT_FAILURE;
    }
    auto index_conf = config["index_conf"];

    std::shared_ptr<DiskANNIndex> diskann_index = std::make_shared<DiskANNIndex>(std::string(index_conf));

    VectorTest vector_test(config_file, diskann_index);

    return EXIT_SUCCESS;
}