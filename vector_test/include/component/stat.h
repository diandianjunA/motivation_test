#pragma once
#include <mutex>
#include <vector>
#include <string>

class OperationStat {
public:
    OperationStat() : totalTime(0), callCount(0) {}

    void add(double duration);

    void reset();

    void print(const std::string& operationName) const;

    double getTotalTime() const;

    int getCallCount() const;

private:
    mutable std::mutex mutex; // 互斥锁
    double totalTime; // 总时间
    int callCount;    // 调用次数
};

enum OperationType {
    WRITE,
    READ,
};

class Stat {
public:
    Stat(size_t operationCount) : stats(operationCount), operationNames(operationCount) {}

    void setOperationName(size_t index, const std::string& name);

    void addOperation(size_t operationIndex, double duration);

    void reset(size_t operationIndex);

    void print(size_t operationIndex) const;

    void printAll() const;

    size_t getOperationCount() const;

    double getTotalTime(size_t operationIndex) const;

    int getCallCount(size_t operationIndex) const;

private:
    std::vector<OperationStat> stats; // 操作统计信息
    std::vector<std::string> operationNames; // 操作名称
};