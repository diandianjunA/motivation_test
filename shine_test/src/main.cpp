#include "shine_index.h"

#include "vector_test/config.h"
#include "vector_test/vector_test.h"

#include <iostream>
#include <memory>

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <config_file>" << std::endl;
        return EXIT_FAILURE;
    }

    const std::string config_file = argv[1];
    const auto config = readConfig(config_file);

    if (config.find("index_conf") == config.end()) {
        std::cerr << "index_conf not found in config file" << std::endl;
        return EXIT_FAILURE;
    }

    auto index = std::make_shared<ShineIndex>(config.at("index_conf"));
    VectorTest vector_test(config_file, index);
    return EXIT_SUCCESS;
}
