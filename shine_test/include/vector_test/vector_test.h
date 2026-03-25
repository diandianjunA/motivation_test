#pragma once
#include "vector_index.h"
#include <map>
#include <memory>

enum TestType {
    // 测试图构建后的存储容量
    STORAGE,
    // 测试动态读写过程中的读写性能
    DYNAMIC,
    // 测试召回率
    RECALL,
};

class VectorTest {
public:
    VectorTest(const std::string& conf, std::shared_ptr<VectorIndex> index);
    ~VectorTest();

    void build();

    void storage_test();

    void dynamic_test();

    void recall_test();

private:
    std::shared_ptr<VectorIndex> index;
    std::map<std::string, std::string> config;
};