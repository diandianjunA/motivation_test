#include "config.h"
#include "odinann_index.h"
#include "vector_test.h"
#include <iostream>
#include <memory>

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <config_file>" << std::endl;
        return EXIT_FAILURE;
    }

    const std::string config_file = argv[1];
    const auto config = readConfig(config_file);
    const auto it = config.find("index_conf");
    if (it == config.end()) {
        std::cerr << "index_conf not found in config file" << std::endl;
        return EXIT_FAILURE;
    }

    auto index = std::make_shared<OdinANNIndex>(it->second);
    VectorTest vector_test(config_file, index);
    (void) vector_test;

    return EXIT_SUCCESS;
}
