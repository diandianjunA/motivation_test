#include "component/stat.h"
#include "logger.h"
#include <algorithm>
#include <cmath>

void OperationStat::add(double duration, uint64_t units) {
    std::lock_guard<std::mutex> lock(mutex);
    totalTime += duration;
    callCount++;
    unitCount += units;
    latencies.push_back(duration);
}

void OperationStat::reset() {
    std::lock_guard<std::mutex> lock(mutex);
    totalTime = 0;
    callCount = 0;
    unitCount = 0;
    latencies.clear();
}

void OperationStat::print(const std::string& operationName) const {
    std::lock_guard<std::mutex> lock(mutex);
    if (callCount > 0) {
        GlobalLogger->info("操作: {}", operationName);
        GlobalLogger->info("总时间: {} 秒", totalTime);
        GlobalLogger->info("批次数: {}", callCount);
        GlobalLogger->info("处理向量数: {}", unitCount);
        GlobalLogger->info("平均批次延迟: {:.3f} ms", (totalTime / callCount) * 1000.0);

        auto percentileValue = [&](double percentile) -> std::optional<double> {
            if (latencies.empty()) {
                return std::nullopt;
            }
            std::vector<double> sorted = latencies;
            std::sort(sorted.begin(), sorted.end());
            const double rank = percentile / 100.0 * static_cast<double>(sorted.size());
            const size_t index = static_cast<size_t>(std::ceil(rank <= 1.0 ? 1.0 : rank)) - 1;
            return sorted[std::min(index, sorted.size() - 1)];
        };

        const auto p50 = percentileValue(50.0);
        const auto p90 = percentileValue(90.0);
        const auto p99 = percentileValue(99.0);
        if (p50.has_value() && p90.has_value() && p99.has_value()) {
            GlobalLogger->info(
                "批次延迟百分位: P50={:.3f} ms, P90={:.3f} ms, P99={:.3f} ms",
                p50.value() * 1000.0,
                p90.value() * 1000.0,
                p99.value() * 1000.0);
        }
    } else {
        GlobalLogger->info("操作: {} 没有被调用。", operationName);
    }
}

double OperationStat::getTotalTime() const {
    std::lock_guard<std::mutex> lock(mutex);
    return totalTime;
}

int OperationStat::getCallCount() const {
    std::lock_guard<std::mutex> lock(mutex);
    return callCount;
}

uint64_t OperationStat::getUnitCount() const {
    std::lock_guard<std::mutex> lock(mutex);
    return unitCount;
}

std::optional<double> OperationStat::getPercentile(double percentile) const {
    std::lock_guard<std::mutex> lock(mutex);
    if (latencies.empty()) {
        return std::nullopt;
    }
    std::vector<double> sorted = latencies;
    std::sort(sorted.begin(), sorted.end());
    const double rank = percentile / 100.0 * static_cast<double>(sorted.size());
    const size_t index = static_cast<size_t>(std::ceil(rank <= 1.0 ? 1.0 : rank)) - 1;
    return sorted[std::min(index, sorted.size() - 1)];
}

void Stat::setOperationName(size_t index, const std::string& name) {
    if (index < operationNames.size()) {
        operationNames[index] = name;
    }
}

void Stat::addOperation(size_t operationIndex, double duration, uint64_t units) {
    if (operationIndex < stats.size()) {
        stats[operationIndex].add(duration, units);
    }
}

void Stat::reset(size_t operationIndex) {
    if (operationIndex < stats.size()) {
        stats[operationIndex].reset();
    }
}

void Stat::print(size_t operationIndex) const {
    if (operationIndex < stats.size()) {
        stats[operationIndex].print(operationNames[operationIndex]);
    }
}

void Stat::printAll() const {
    for (size_t i = 0; i < stats.size(); ++i) {
        print(i);
    }
}

size_t Stat::getOperationCount() const {
    return stats.size();
}

double Stat::getTotalTime(size_t operationIndex) const {
    if (operationIndex < stats.size()) {
        return stats[operationIndex].getTotalTime();
    }
    return 0;
}

int Stat::getCallCount(size_t operationIndex) const {
    if (operationIndex < stats.size()) {
        return stats[operationIndex].getCallCount();
    }
    return 0;
}

uint64_t Stat::getUnitCount(size_t operationIndex) const {
    if (operationIndex < stats.size()) {
        return stats[operationIndex].getUnitCount();
    }
    return 0;
}

std::optional<double> Stat::getPercentile(size_t operationIndex, double percentile) const {
    if (operationIndex < stats.size()) {
        return stats[operationIndex].getPercentile(percentile);
    }
    return std::nullopt;
}
