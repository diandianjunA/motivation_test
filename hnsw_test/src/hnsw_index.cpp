#include "hnsw_index.h"

#include "config.h"
#include "util.h"
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fcntl.h>
#include <iostream>
#include <stdexcept>
#include <sys/mman.h>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>

namespace {

int getRequiredInt(const std::map<std::string, std::string>& config, const std::string& key) {
    const auto it = config.find(key);
    if (it == config.end()) {
        throw std::runtime_error("Config file missing " + key);
    }
    return std::stoi(it->second);
}

size_t getOptionalSize(
    const std::map<std::string, std::string>& config,
    const std::string& key,
    size_t default_value) {
    const auto it = config.find(key);
    if (it == config.end()) {
        return default_value;
    }
    return static_cast<size_t>(std::stoull(it->second));
}

bool getOptionalBool(
    const std::map<std::string, std::string>& config,
    const std::string& key,
    bool default_value) {
    const auto it = config.find(key);
    if (it == config.end()) {
        return default_value;
    }
    return it->second == "true";
}

std::pair<const float*, std::pair<size_t, size_t>> partialMmapFbin(
    const std::string& file_path,
    size_t start_vector,
    size_t num_vectors) {
    const int fd = open(file_path.c_str(), O_RDONLY);
    if (fd == -1) {
        throw std::runtime_error("无法打开文件: " + file_path);
    }

    struct stat sb;
    if (fstat(fd, &sb) == -1) {
        close(fd);
        throw std::runtime_error("无法获取文件大小: " + file_path);
    }

    const size_t file_size = sb.st_size;
    constexpr size_t header_size = 8;
    constexpr size_t page_size = 4096;

    uint32_t header[2];
    if (pread(fd, header, header_size, 0) != static_cast<ssize_t>(header_size)) {
        close(fd);
        throw std::runtime_error("无法读取文件头: " + file_path);
    }

    const size_t dim = header[1];
    const size_t data_size = num_vectors * dim * sizeof(float);
    const size_t data_offset = header_size + start_vector * dim * sizeof(float);
    const size_t mmap_offset = (data_offset / page_size) * page_size;
    const size_t extra_bytes = data_offset - mmap_offset;
    const size_t mmap_size = data_size + extra_bytes;

    if (mmap_offset + mmap_size > file_size) {
        close(fd);
        throw std::runtime_error("偏移量超出文件大小");
    }

    void* addr = mmap(nullptr, mmap_size, PROT_READ, MAP_PRIVATE, fd, mmap_offset);
    close(fd);
    if (addr == MAP_FAILED) {
        throw std::runtime_error("部分内存映射失败");
    }

    madvise(addr, mmap_size, MADV_WILLNEED);
    const float* data = reinterpret_cast<const float*>(static_cast<char*>(addr) + extra_bytes);
    return {data, {num_vectors, dim}};
}

void unmapPartial(const float* data, size_t num_vectors, size_t dim, size_t start_vector) {
    if (data == nullptr) {
        return;
    }

    constexpr size_t header_size = 8;
    constexpr size_t page_size = 4096;
    const size_t data_offset = header_size + start_vector * dim * sizeof(float);
    const size_t mmap_offset = (data_offset / page_size) * page_size;
    const size_t extra_bytes = data_offset - mmap_offset;
    const size_t mmap_size = num_vectors * dim * sizeof(float) + extra_bytes;
    const char* addr = reinterpret_cast<const char*>(data) - extra_bytes;
    munmap(const_cast<char*>(addr), mmap_size);
}

}  // namespace

HNSWIndex::HNSWIndex(std::string conf) : config_path_(std::move(conf)) {
    config_ = readConfig(config_path_);

    dim_ = getRequiredInt(config_, "dim");
    max_elements_ = getOptionalSize(
        config_,
        "max_elements_to_insert",
        getOptionalSize(config_, "max_elements", 0));
    if (max_elements_ == 0) {
        throw std::runtime_error("Config file missing max_elements_to_insert or max_elements");
    }

    m_ = getOptionalSize(config_, "M", 16);
    ef_construction_ = getOptionalSize(config_, "ef_construction", 200);
    ef_search_ = getOptionalSize(config_, "ef_search", 50);
    num_threads_ = std::max<size_t>(1, getOptionalSize(config_, "num_threads", 1));
    random_seed_ = getOptionalSize(config_, "random_seed", 100);
    seed_build_size_ = std::max<size_t>(1, getOptionalSize(config_, "seed_build_size", 100000));
    batch_size_ = std::max<size_t>(1, getOptionalSize(config_, "batch_size", 100000));
    allow_replace_deleted_ = getOptionalBool(config_, "allow_replace_deleted", false);

    space_ = std::make_unique<hnswlib::L2Space>(dim_);
    createIndex(max_elements_);
}

void HNSWIndex::createIndex(size_t max_elements) {
    index_ = std::make_unique<hnswlib::HierarchicalNSW<float>>(
        space_.get(),
        max_elements,
        m_,
        ef_construction_,
        random_seed_,
        allow_replace_deleted_);
    index_->setEf(ef_search_);
    max_elements_ = max_elements;
}

void HNSWIndex::ensureCapacity(size_t required_elements) {
    if (index_ == nullptr) {
        createIndex(std::max(required_elements, max_elements_));
        return;
    }
    if (required_elements <= max_elements_) {
        return;
    }

    std::lock_guard<std::mutex> lock(resize_mutex_);
    if (required_elements <= max_elements_) {
        return;
    }

    size_t new_capacity = max_elements_;
    while (new_capacity < required_elements) {
        new_capacity = std::max(new_capacity * 2, required_elements);
    }
    index_->resizeIndex(new_capacity);
    max_elements_ = new_capacity;
}

void HNSWIndex::addPoints(const float* data, size_t count, const std::vector<uint32_t>& ids) {
    if (ids.size() != count) {
        throw std::runtime_error("ids size does not match vector count");
    }
    ensureCapacity(index_->getCurrentElementCount() + count);
    ParallelFor(0, count, num_threads_, [&](size_t i, size_t) {
        index_->addPoint(data + i * static_cast<size_t>(dim_), static_cast<hnswlib::labeltype>(ids[i]));
    });
}

void HNSWIndex::addPointsWithProgress(
    const float* data,
    size_t count,
    const std::vector<uint32_t>& ids,
    ProgressBar& progress_bar,
    size_t base_processed,
    const std::string& stage) {
    if (ids.size() != count) {
        throw std::runtime_error("ids size does not match vector count");
    }

    ensureCapacity(index_->getCurrentElementCount() + count);

    std::atomic<size_t> local_processed(0);
    std::atomic<bool> done(false);

    std::thread display_thread([&]() {
        while (!done.load()) {
            progress_bar.set_current(base_processed + local_processed.load());
            progress_bar.display();
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
        }
        progress_bar.set_current(base_processed + count);
        progress_bar.display();
    });

    try {
        ParallelFor(0, count, num_threads_, [&](size_t i, size_t) {
            index_->addPoint(data + i * static_cast<size_t>(dim_), static_cast<hnswlib::labeltype>(ids[i]));
            local_processed.fetch_add(1, std::memory_order_relaxed);
        });
        done.store(true);
        display_thread.join();
        std::cout << std::endl
                  << stage << " 完成: " << count << " 向量, 当前总量 "
                  << index_->getCurrentElementCount() << std::endl;
    } catch (...) {
        done.store(true);
        display_thread.join();
        throw;
    }
}

void HNSWIndex::build(const std::vector<float>& vecs, const std::vector<uint32_t>& ids) {
    std::unique_lock<std::shared_mutex> index_lock(index_rw_mutex_);
    if (vecs.empty()) {
        return;
    }
    if (vecs.size() != ids.size() * static_cast<size_t>(dim_)) {
        throw std::runtime_error("Input vector size does not match dim * ids.size()");
    }
    if (ids.size() > max_elements_) {
        throw std::runtime_error("Input vector count exceeds configured max_elements_to_insert");
    }

    createIndex(std::max(max_elements_, ids.size()));

    const size_t total_count = ids.size();
    const size_t initial_build_size = std::min(total_count, seed_build_size_);
    size_t processed = 0;

    std::cout << "内存数据集: " << total_count << " vectors, " << dim_ << " dim" << std::endl;
    std::cout << "HNSW 构建参数: M=" << m_ << ", ef_construction=" << ef_construction_
              << ", ef_search=" << ef_search_ << ", num_threads=" << num_threads_
              << ", seed_build_size=" << seed_build_size_ << ", batch_size=" << batch_size_ << std::endl;

    ProgressBar progress_bar("Building HNSW index", total_count, true, true);

    if (initial_build_size > 0) {
        std::vector<uint32_t> initial_ids(ids.begin(), ids.begin() + static_cast<std::ptrdiff_t>(initial_build_size));
        std::cout << "开始首批建图: " << initial_build_size << " 向量" << std::endl;
        addPointsWithProgress(vecs.data(), initial_build_size, initial_ids, progress_bar, processed, "首批建图");
        processed += initial_build_size;
    }

    while (processed < total_count) {
        const size_t current_batch_size = std::min(batch_size_, total_count - processed);
        const float* batch_data = vecs.data() + processed * static_cast<size_t>(dim_);
        std::vector<uint32_t> batch_ids(
            ids.begin() + static_cast<std::ptrdiff_t>(processed),
            ids.begin() + static_cast<std::ptrdiff_t>(processed + current_batch_size));

        std::cout << "开始批次插入: offset=" << processed << ", batch_size=" << current_batch_size
                  << std::endl;
        addPointsWithProgress(batch_data, current_batch_size, batch_ids, progress_bar, processed, "批次插入");
        processed += current_batch_size;
    }

    progress_bar.finish();
    std::cout << "HNSW 索引构建完成" << std::endl;
}

void HNSWIndex::build(const std::string& dataset_path) {
    std::unique_lock<std::shared_mutex> index_lock(index_rw_mutex_);
    const int fd = open(dataset_path.c_str(), O_RDONLY);
    if (fd == -1) {
        throw std::runtime_error("无法打开文件: " + dataset_path);
    }
    uint32_t header[2];
    if (pread(fd, header, 8, 0) != 8) {
        close(fd);
        throw std::runtime_error("无法读取文件头: " + dataset_path);
    }
    close(fd);

    const size_t total_count = header[0];
    const size_t dim = header[1];
    if (static_cast<int>(dim) != dim_) {
        throw std::runtime_error("Dataset dim does not match config dim");
    }

    if (total_count > max_elements_) {
        throw std::runtime_error("Dataset size exceeds configured max_elements_to_insert");
    }

    std::cout << "数据集: " << total_count << " vectors, " << dim << " dim" << std::endl;
    std::cout << "HNSW 构建参数: M=" << m_ << ", ef_construction=" << ef_construction_
              << ", ef_search=" << ef_search_ << ", num_threads=" << num_threads_
              << ", seed_build_size=" << seed_build_size_ << ", batch_size=" << batch_size_ << std::endl;

    createIndex(std::max(max_elements_, total_count));

    ProgressBar progress_bar("Building HNSW index", total_count, true, true);
    size_t processed = 0;
    const size_t initial_build_size = std::min(total_count, seed_build_size_);

    if (initial_build_size > 0) {
        auto [mapped_data, sizes] = partialMmapFbin(dataset_path, 0, initial_build_size);
        std::vector<uint32_t> ids(initial_build_size);
        for (size_t i = 0; i < initial_build_size; ++i) {
            ids[i] = static_cast<uint32_t>(i);
        }
        std::cout << "开始首批建图: " << initial_build_size << " 向量" << std::endl;
        addPointsWithProgress(mapped_data, initial_build_size, ids, progress_bar, processed, "首批建图");
        processed += initial_build_size;
        unmapPartial(mapped_data, initial_build_size, dim, 0);
    }

    while (processed < total_count) {
        const size_t current_batch_size = std::min(batch_size_, total_count - processed);
        auto [mapped_data, sizes] = partialMmapFbin(dataset_path, processed, current_batch_size);
        std::vector<uint32_t> ids(current_batch_size);
        for (size_t i = 0; i < current_batch_size; ++i) {
            ids[i] = static_cast<uint32_t>(processed + i);
        }

        std::cout << "开始批次插入: offset=" << processed << ", batch_size=" << current_batch_size
                  << std::endl;
        addPointsWithProgress(mapped_data, current_batch_size, ids, progress_bar, processed, "批次插入");
        unmapPartial(mapped_data, current_batch_size, dim, processed);
        processed += current_batch_size;
    }

    progress_bar.finish();
    std::cout << "HNSW 索引构建完成" << std::endl;
}

void HNSWIndex::insert(const std::vector<float>& vec, const std::vector<uint32_t>& ids) {
    std::unique_lock<std::shared_mutex> index_lock(index_rw_mutex_);
    if (ids.empty()) {
        return;
    }
    if (vec.size() != ids.size() * static_cast<size_t>(dim_)) {
        throw std::runtime_error("Insert vector size does not match dim * ids.size()");
    }

    addPoints(vec.data(), ids.size(), ids);
}

void HNSWIndex::search(
    const std::vector<float>& query,
    size_t top_k,
    std::vector<uint32_t>& ids,
    std::vector<float>& distances) const {
    std::shared_lock<std::shared_mutex> index_lock(index_rw_mutex_);
    if (query.size() != static_cast<size_t>(dim_)) {
        throw std::runtime_error("Query dim does not match config dim");
    }
    if (index_ == nullptr) {
        throw std::runtime_error("Index is not initialized");
    }

    const auto results = index_->searchKnnCloserFirst(query.data(), top_k);
    ids.assign(top_k, 0);
    distances.assign(top_k, 0.0F);

    for (size_t i = 0; i < results.size(); ++i) {
        distances[i] = results[i].first;
        ids[i] = static_cast<uint32_t>(results[i].second);
    }
}

void HNSWIndex::load(const std::string& index_path) {
    std::unique_lock<std::shared_mutex> index_lock(index_rw_mutex_);
    index_ = std::make_unique<hnswlib::HierarchicalNSW<float>>(
        space_.get(),
        index_path,
        false,
        max_elements_,
        allow_replace_deleted_);
    index_->setEf(ef_search_);
    max_elements_ = std::max(max_elements_, index_->getMaxElements());
}

void HNSWIndex::save(const std::string& index_path) {
    std::shared_lock<std::shared_mutex> index_lock(index_rw_mutex_);
    if (index_ == nullptr) {
        throw std::runtime_error("Index is not initialized");
    }
    const std::filesystem::path path(index_path);
    if (path.has_parent_path()) {
        std::filesystem::create_directories(path.parent_path());
    }
    index_->saveIndex(index_path);
}

std::string HNSWIndex::getIndexType() const {
    return "HNSW";
}
