#pragma once
#include <cstdint>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

class OperationStat {
public:
    OperationStat() : totalTime(0), callCount(0), unitCount(0) {}

    void add(double duration, uint64_t units);

    void reset();

    void print(const std::string& operationName) const;

    double getTotalTime() const;

    int getCallCount() const;

    uint64_t getUnitCount() const;

    std::optional<double> getPercentile(double percentile) const;

private:
    mutable std::mutex mutex;
    double totalTime;
    int callCount;
    uint64_t unitCount;
    std::vector<double> latencies;
};

enum OperationType {
    WRITE,
    READ,
};

class Stat {
public:
    Stat(size_t operationCount) : stats(operationCount), operationNames(operationCount) {}

    void setOperationName(size_t index, const std::string& name);

    void addOperation(size_t operationIndex, double duration, uint64_t units);

    void reset(size_t operationIndex);

    void print(size_t operationIndex) const;

    void printAll() const;

    size_t getOperationCount() const;

    double getTotalTime(size_t operationIndex) const;

    int getCallCount(size_t operationIndex) const;

    uint64_t getUnitCount(size_t operationIndex) const;

    std::optional<double> getPercentile(size_t operationIndex, double percentile) const;

private:
    std::vector<OperationStat> stats;
    std::vector<std::string> operationNames;
};
