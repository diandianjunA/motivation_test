cmake_minimum_required(VERSION 3.16)

find_package(GTest REQUIRED)

add_executable(correct_test
    correct_test.cpp
    ../src/hnsw_index.cpp
)

target_include_directories(correct_test
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/..
        ${CMAKE_CURRENT_SOURCE_DIR}/../include
        ${CMAKE_CURRENT_SOURCE_DIR}/../../vector_test/include
)

target_link_libraries(correct_test
    PRIVATE
        OpenMP::OpenMP_CXX
        VectorTest
        GTest::GTest
        GTest::Main
)

add_test(NAME hnsw_correct_test COMMAND correct_test)
