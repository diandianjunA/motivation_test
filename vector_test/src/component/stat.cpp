#include "component/stat.h"
#include "logger.h"

void OperationStat::add(double duration) {
    std::lock_guard<std::mutex> lock(mutex); // 保护对 totalTime 和 callCount 的访问
    totalTime += duration;
    callCount++;
}

void OperationStat::reset() {
    std::lock_guard<std::mutex> lock(mutex); // 保护对 totalTime 和 callCount 的访问
    totalTime = 0;
    callCount = 0;
}

void OperationStat::print(const std::string& operationName) const {
    if (callCount > 0) {
        GlobalLogger->info("操作: {}", operationName);
        GlobalLogger->info("总时间: {} 秒", totalTime);
        GlobalLogger->info("调用次数: {}", callCount);
        GlobalLogger->info("平均时间: {} 秒", (totalTime / callCount));
    } else {
        GlobalLogger->info("操作: {} 没有被调用。", operationName);
    }
}

double OperationStat::getTotalTime() const {
    return totalTime;
}

int OperationStat::getCallCount() const {
    return callCount;
}

void Stat::setOperationName(size_t index, const std::string& name) {
    if (index < operationNames.size()) {
        operationNames[index] = name;
    }
}

void Stat::addOperation(size_t operationIndex, double duration) {
    if (operationIndex < stats.size()) {
        stats[operationIndex].add(duration);
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
