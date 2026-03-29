#include "hnsw_index.h"
#include <gtest/gtest.h>
#include <algorithm>
#include <atomic>
#include <cmath>
#include <exception>
#include <filesystem>
#include <fstream>
#include <limits>
#include <mutex>
#include <numeric>
#include <random>
#include <thread>

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

std::vector<float> makeDeterministicVectors(size_t count, size_t dim) {
    std::vector<float> data(count * dim);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0F, 1.0F);
    for (float& value : data) {
        value = dist(rng);
    }
    return data;
}

void runRawMixedReadWriteWithoutResizeCase(
    size_t dim,
    size_t initial_count,
    size_t write_count,
    size_t top_k,
    size_t reader_threads,
    size_t searches_per_reader) {
    const size_t total_count = initial_count + write_count;
    auto vectors = makeDeterministicVectors(total_count, dim);

    hnswlib::L2Space space(dim);
    hnswlib::HierarchicalNSW<float> index(&space, total_count, 16, 100, 17, false);
    index.setEf(64);

    for (size_t i = 0; i < initial_count; ++i) {
        index.addPoint(
            vectors.data() + i * dim,
            static_cast<hnswlib::labeltype>(i));
    }

    std::atomic<bool> start(false);
    std::atomic<bool> writer_done(false);
    std::exception_ptr thread_error;
    std::mutex error_mutex;

    auto capture_error = [&](std::exception_ptr eptr) {
        std::lock_guard<std::mutex> lock(error_mutex);
        if (!thread_error) {
            thread_error = eptr;
        }
    };

    std::thread writer([&]() {
        while (!start.load(std::memory_order_acquire)) {
            std::this_thread::yield();
        }
        try {
            for (size_t i = initial_count; i < total_count; ++i) {
                index.addPoint(
                    vectors.data() + i * dim,
                    static_cast<hnswlib::labeltype>(i));
            }
        } catch (...) {
            capture_error(std::current_exception());
        }
        writer_done.store(true, std::memory_order_release);
    });

    std::vector<std::thread> readers;
    readers.reserve(reader_threads);
    for (size_t thread_idx = 0; thread_idx < reader_threads; ++thread_idx) {
        readers.emplace_back([&, thread_idx]() {
            while (!start.load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }

            try {
                for (size_t i = 0; i < searches_per_reader; ++i) {
                    const size_t vector_id = (thread_idx * searches_per_reader + i) % initial_count;
                    const auto results =
                        index.searchKnnCloserFirst(vectors.data() + vector_id * dim, top_k);

                    if (results.empty()) {
                        throw std::runtime_error("search returned empty results during concurrent read/write test");
                    }
                    if (results.size() > top_k) {
                        throw std::runtime_error("search returned more than top_k results");
                    }
                    for (const auto& [distance, label] : results) {
                        if (!std::isfinite(distance)) {
                            throw std::runtime_error("search returned non-finite distance");
                        }
                        if (static_cast<size_t>(label) >= total_count) {
                            throw std::runtime_error("search returned label out of expected range");
                        }
                    }
                }
            } catch (...) {
                capture_error(std::current_exception());
            }
        });
    }

    start.store(true, std::memory_order_release);

    writer.join();
    for (auto& reader : readers) {
        reader.join();
    }

    if (thread_error) {
        std::rethrow_exception(thread_error);
    }

    if (!writer_done.load(std::memory_order_acquire)) {
        throw std::runtime_error("writer thread did not complete");
    }
    if (index.getCurrentElementCount() != total_count) {
        throw std::runtime_error("index element count mismatch after concurrent test");
    }
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

TEST(HNSWTest, RawHnswlibMixedReadWriteWithoutResize) {
    constexpr size_t dim = 16;
    constexpr size_t initial_count = 2000;
    constexpr size_t write_count = 1000;
    constexpr size_t top_k = 10;
    constexpr size_t reader_threads = 4;
    constexpr size_t searches_per_reader = 4000;

    EXPECT_NO_THROW(runRawMixedReadWriteWithoutResizeCase(
        dim,
        initial_count,
        write_count,
        top_k,
        reader_threads,
        searches_per_reader));
}

TEST(HNSWTest, RawHnswlibMixedReadWriteWithoutResizeSoak) {
    constexpr size_t rounds = 30;
    constexpr size_t dim = 16;
    constexpr size_t initial_count = 1500;
    constexpr size_t write_count = 750;
    constexpr size_t top_k = 10;
    constexpr size_t reader_threads = 4;
    constexpr size_t searches_per_reader = 1500;

    for (size_t round = 0; round < rounds; ++round) {
        SCOPED_TRACE(::testing::Message() << "round=" << round);
        EXPECT_NO_THROW(runRawMixedReadWriteWithoutResizeCase(
            dim,
            initial_count,
            write_count,
            top_k,
            reader_threads,
            searches_per_reader));
    }
}

TEST(HNSWTest, RawHnswlibMixedReadWriteQualityStaysReasonableWithoutResize) {
    constexpr size_t dim = 8;
    constexpr size_t initial_count = 400;
    constexpr size_t write_count = 200;
    constexpr size_t total_count = initial_count + write_count;
    constexpr size_t query_count = 32;
    constexpr size_t top_k = 1;
    constexpr size_t quality_reader_threads = 4;
    constexpr size_t per_thread_searches = 2000;
    constexpr float max_distance_error = 1e-4F;

    std::vector<float> vectors(total_count * dim, 0.0F);
    for (size_t i = 0; i < total_count; ++i) {
        vectors[i * dim] = static_cast<float>(i);
    }

    hnswlib::L2Space space(dim);
    hnswlib::HierarchicalNSW<float> index(&space, total_count, 16, 100, 17, false);
    index.setEf(64);

    for (size_t i = 0; i < initial_count; ++i) {
        index.addPoint(
            vectors.data() + i * dim,
            static_cast<hnswlib::labeltype>(i));
    }

    std::atomic<bool> start(false);
    std::exception_ptr thread_error;
    std::mutex error_mutex;
    std::atomic<uint64_t> mismatch_count(0);
    std::atomic<uint64_t> total_checks(0);

    auto capture_error = [&](std::exception_ptr eptr) {
        std::lock_guard<std::mutex> lock(error_mutex);
        if (!thread_error) {
            thread_error = eptr;
        }
    };

    std::thread writer([&]() {
        while (!start.load(std::memory_order_acquire)) {
            std::this_thread::yield();
        }
        try {
            for (size_t i = initial_count; i < total_count; ++i) {
                index.addPoint(
                    vectors.data() + i * dim,
                    static_cast<hnswlib::labeltype>(i));
            }
        } catch (...) {
            capture_error(std::current_exception());
        }
    });

    std::vector<std::thread> readers;
    readers.reserve(quality_reader_threads);
    for (size_t thread_idx = 0; thread_idx < quality_reader_threads; ++thread_idx) {
        readers.emplace_back([&, thread_idx]() {
            while (!start.load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }

            try {
                for (size_t i = 0; i < per_thread_searches; ++i) {
                    const size_t query_id = (thread_idx * per_thread_searches + i) % query_count;
                    const auto query_ptr = vectors.data() + query_id * dim;
                    const auto results = index.searchKnnCloserFirst(query_ptr, top_k);
                    if (results.empty()) {
                        throw std::runtime_error("quality test search returned empty results");
                    }

                    total_checks.fetch_add(1, std::memory_order_relaxed);
                    const auto& best = results.front();
                    if (best.second != static_cast<hnswlib::labeltype>(query_id) ||
                        std::fabs(best.first) > max_distance_error) {
                        mismatch_count.fetch_add(1, std::memory_order_relaxed);
                    }
                }
            } catch (...) {
                capture_error(std::current_exception());
            }
        });
    }

    start.store(true, std::memory_order_release);

    writer.join();
    for (auto& reader : readers) {
        reader.join();
    }

    if (thread_error) {
        std::rethrow_exception(thread_error);
    }

    ASSERT_GT(total_checks.load(std::memory_order_relaxed), 0U);
    const double mismatch_ratio =
        static_cast<double>(mismatch_count.load(std::memory_order_relaxed)) /
        static_cast<double>(total_checks.load(std::memory_order_relaxed));

    EXPECT_LT(mismatch_ratio, 0.01)
        << "Concurrent read/write introduced too many top-1 mismatches for fixed queries";
}

TEST(HNSWTest, DISABLED_RawHnswlibMixedReadWriteWithResizeStress) {
    constexpr size_t dim = 16;
    constexpr size_t initial_count = 256;
    constexpr size_t max_elements = 256;
    constexpr size_t resize_target = 1024;
    constexpr size_t write_count = resize_target - initial_count;
    constexpr size_t top_k = 10;
    constexpr size_t reader_threads = 4;
    constexpr size_t searches_per_reader = 4000;
    constexpr size_t rounds = 20;

    for (size_t round = 0; round < rounds; ++round) {
        SCOPED_TRACE(::testing::Message() << "round=" << round);

        const size_t total_count = initial_count + write_count;
        auto vectors = makeDeterministicVectors(total_count, dim);

        hnswlib::L2Space space(dim);
        hnswlib::HierarchicalNSW<float> index(&space, max_elements, 16, 100, 17, false);
        index.setEf(64);

        for (size_t i = 0; i < initial_count; ++i) {
            index.addPoint(
                vectors.data() + i * dim,
                static_cast<hnswlib::labeltype>(i));
        }

        std::atomic<bool> start(false);
        std::exception_ptr thread_error;
        std::mutex error_mutex;

        auto capture_error = [&](std::exception_ptr eptr) {
            std::lock_guard<std::mutex> lock(error_mutex);
            if (!thread_error) {
                thread_error = eptr;
            }
        };

        std::thread writer([&]() {
            while (!start.load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }
            try {
                // Force repeated realloc on the raw hnswlib index while readers are traversing it.
                for (size_t i = initial_count; i < total_count; ++i) {
                    if (i == initial_count || ((i - initial_count) % 128 == 0)) {
                        const size_t requested_capacity =
                            std::min(total_count, std::max(index.getMaxElements() * 2, i + 1));
                        index.resizeIndex(requested_capacity);
                    }
                    index.addPoint(
                        vectors.data() + i * dim,
                        static_cast<hnswlib::labeltype>(i));
                }
            } catch (...) {
                capture_error(std::current_exception());
            }
        });

        std::vector<std::thread> readers;
        readers.reserve(reader_threads);
        for (size_t thread_idx = 0; thread_idx < reader_threads; ++thread_idx) {
            readers.emplace_back([&, thread_idx]() {
                while (!start.load(std::memory_order_acquire)) {
                    std::this_thread::yield();
                }

                try {
                    for (size_t i = 0; i < searches_per_reader; ++i) {
                        const size_t vector_id = (thread_idx * searches_per_reader + i) % initial_count;
                        const auto results =
                            index.searchKnnCloserFirst(vectors.data() + vector_id * dim, top_k);

                        if (results.empty()) {
                            throw std::runtime_error("search returned empty results during resize stress test");
                        }
                        if (results.size() > top_k) {
                            throw std::runtime_error("search returned more than top_k during resize stress test");
                        }
                        for (const auto& [distance, label] : results) {
                            if (!std::isfinite(distance)) {
                                throw std::runtime_error("search returned non-finite distance during resize stress test");
                            }
                            if (static_cast<size_t>(label) >= total_count) {
                                throw std::runtime_error("search returned label out of range during resize stress test");
                            }
                        }
                    }
                } catch (...) {
                    capture_error(std::current_exception());
                }
            });
        }

        start.store(true, std::memory_order_release);

        writer.join();
        for (auto& reader : readers) {
            reader.join();
        }

        if (thread_error) {
            std::rethrow_exception(thread_error);
        }

        EXPECT_EQ(index.getCurrentElementCount(), total_count);
        EXPECT_GE(index.getMaxElements(), total_count);
    }
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
