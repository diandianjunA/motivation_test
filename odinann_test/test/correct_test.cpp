#include "odinann_index.h"
#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <numeric>

namespace {

std::filesystem::path writeConfigFile(
    const std::filesystem::path& path,
    const std::string& content) {
    std::filesystem::create_directories(path.parent_path());
    std::ofstream out(path);
    out << content;
    return path;
}

}  // namespace

TEST(OdinANNTest, BuildInsertSearchAndReload) {
    const std::filesystem::path temp_dir =
        std::filesystem::temp_directory_path() / "odinann_test_correct";
    std::filesystem::create_directories(temp_dir);

    const auto conf_path = writeConfigFile(
        temp_dir / "index_conf.ini",
        "dim = 64\n"
        "max_points_to_insert = 256\n"
        "build_R = 16\n"
        "build_L = 32\n"
        "build_B = 0.25\n"
        "build_M = 2\n"
        "build_threads = 1\n"
        "search_L = 32\n"
        "beamwidth = 1\n"
        "num_threads = 1\n"
        "L_disk = 32\n"
        "R_disk = 0\n"
        "alpha_disk = 1.2\n"
        "C = 64\n"
        "search_mem_L = 0\n"
        "use_mem_index = false\n"
        "single_file_index = false\n"
        "search_mode = beam\n"
        "metric = l2\n");

    OdinANNIndex index(conf_path.string());

    std::vector<float> flat_base;
    constexpr size_t dim = 64;
    constexpr size_t base_count = 64;
    flat_base.reserve(base_count * dim);
    for (size_t i = 0; i < base_count; ++i) {
        for (size_t d = 0; d < dim; ++d) {
            flat_base.push_back(static_cast<float>(i * 0.1) + static_cast<float>(d) * 0.01F);
        }
    }

    std::vector<uint32_t> ids(base_count);
    std::iota(ids.begin(), ids.end(), 0U);
    index.build(flat_base, ids);

    const auto prefix = (temp_dir / "tiny_index").string();
    index.save(prefix);
    index.load(prefix);

    std::vector<float> inserted(dim, 0.0F);
    for (size_t d = 0; d < dim; ++d) {
        inserted[d] = 100.0F + static_cast<float>(d) * 0.001F;
    }
    index.insert(inserted, std::vector<uint32_t>{200U});

    std::vector<uint32_t> result_ids;
    std::vector<float> result_distances;
    index.search(inserted, 5, result_ids, result_distances);

    ASSERT_FALSE(result_ids.empty());
    EXPECT_NE(std::find(result_ids.begin(), result_ids.end(), 200U), result_ids.end());
    EXPECT_TRUE(std::all_of(result_distances.begin(), result_distances.end(), [](float value) {
        return std::isfinite(value) || value == std::numeric_limits<float>::max();
    }));

    const auto merged_prefix = (temp_dir / "tiny_index_merged").string();
    index.save(merged_prefix);

    OdinANNIndex reloaded(conf_path.string());
    reloaded.load(merged_prefix);
    result_ids.clear();
    result_distances.clear();
    reloaded.search(inserted, 5, result_ids, result_distances);
    EXPECT_NE(std::find(result_ids.begin(), result_ids.end(), 200U), result_ids.end());
}
