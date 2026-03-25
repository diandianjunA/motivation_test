#ifndef RDMA_LIBRARY_TYPES_HH
#define RDMA_LIBRARY_TYPES_HH

#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <vector>

#include "extern/concurrentqueue.hh"

using i8 = int8_t;
using u8 = uint8_t;

using i16 = int16_t;
using u16 = uint16_t;

using i32 = int32_t;
using u32 = uint32_t;

using i64 = int64_t;
using u64 = uint64_t;

using f32 = float;
using f64 = double;

using byte_t = uint8_t;
using str = std::string;
using size_t = std::size_t;
using idx_t = std::size_t;

using intptr_t = std::intptr_t;

template <typename T>
using u_ptr = std::unique_ptr<T>;

template <typename T>
using s_ptr = std::shared_ptr<T>;

template <typename T>
using func = std::function<T>;

template <typename T>
using vec = std::vector<T>;

template <typename T>
using span = std::span<T>;

template <typename T>
class concurrent_vec {
public:
  concurrent_vec() = default;
  concurrent_vec(const concurrent_vec&) = delete;
  concurrent_vec& operator=(const concurrent_vec&) = delete;
  concurrent_vec(concurrent_vec&&) = delete;
  concurrent_vec& operator=(concurrent_vec&&) = delete;

  void resize(size_t count) {
    std::lock_guard<std::mutex> lock(mutex_);
    data_.resize(count);
  }

  template <typename... Args>
  T* emplace_back(Args&&... args) {
    std::lock_guard<std::mutex> lock(mutex_);
    data_.emplace_back(std::forward<Args>(args)...);
    return &data_.back();
  }

  void push_back(T value) {
    std::lock_guard<std::mutex> lock(mutex_);
    data_.push_back(std::move(value));
  }

  size_t size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return data_.size();
  }

  T& operator[](size_t index) { return data_[index]; }
  const T& operator[](size_t index) const { return data_[index]; }

private:
  mutable std::mutex mutex_;
  std::vector<T> data_;
};

template <typename T>
using concurrent_queue = moodycamel::ConcurrentQueue<T>;
// using concurrent_queue = oneapi::tbb::concurrent_queue<T>;

#endif  // RDMA_LIBRARY_TYPES_HH
