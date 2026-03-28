#include "hnsw_index.h"
#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
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

float l2Distance(const std::vector<float>& lhs, const std::vector<float>& rhs) {
    float sum = 0.0F;
    for (size_t i = 0; i < lhs.size(); ++i) {
        const float diff = lhs[i] - rhs[i];
        sum += diff * diff;
    }
    return sum;
}

}  // namespace

TEST(HNSWTest, CorrectTest) {
    const std::filesystem::path temp_dir =
        std::filesystem::temp_directory_path() / "hnsw_test_correct";
    std::filesystem::create_directories(temp_dir);

    const auto conf_path = writeConfigFile(
        temp_dir / "index_conf.ini",
        "dim = 4\n"
        "max_elements_to_insert = 8\n"
        "M = 8\n"
        "ef_construction = 64\n"
        "ef_search = 64\n"
        "num_threads = 1\n");

    HNSWIndex index(conf_path.string());

    const std::vector<std::vector<float>> base = {
        {0.0F, 0.0F, 0.0F, 0.0F},
        {1.0F, 0.0F, 0.0F, 0.0F},
        {0.0F, 1.0F, 0.0F, 0.0F},
        {0.0F, 0.0F, 1.0F, 0.0F},
        {0.0F, 0.0F, 0.0F, 1.0F},
        {1.0F, 1.0F, 0.0F, 0.0F},
    };

    std::vector<float> flat_base;
    flat_base.reserve(base.size() * base[0].size());
    for (const auto& vec : base) {
        flat_base.insert(flat_base.end(), vec.begin(), vec.end());
    }

    std::vector<uint32_t> ids(base.size());
    std::iota(ids.begin(), ids.end(), 0U);
    index.build(flat_base, ids);

    const std::vector<float> query = {0.9F, 0.1F, 0.0F, 0.0F};
    std::vector<uint32_t> result_ids;
    std::vector<float> result_distances;
    index.search(query, 3, result_ids, result_distances);

    std::vector<std::pair<float, uint32_t>> expected;
    for (size_t i = 0; i < base.size(); ++i) {
        expected.emplace_back(l2Distance(query, base[i]), static_cast<uint32_t>(i));
    }
    std::sort(expected.begin(), expected.end(), [](const auto& lhs, const auto& rhs) {
        if (lhs.first != rhs.first) {
            return lhs.first < rhs.first;
        }
        return lhs.second < rhs.second;
    });

    ASSERT_EQ(result_ids.size(), 3U);
    EXPECT_EQ(result_ids[0], expected[0].second);
    EXPECT_EQ(result_ids[1], expected[1].second);
    EXPECT_EQ(result_ids[2], expected[2].second);
    EXPECT_NEAR(result_distances[0], expected[0].first, 1e-5F);
    EXPECT_NEAR(result_distances[1], expected[1].first, 1e-5F);
    EXPECT_NEAR(result_distances[2], expected[2].first, 1e-5F);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
