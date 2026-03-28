cmake_minimum_required(VERSION 3.16)

project(HNSWTest
    VERSION 1.0.0
    DESCRIPTION "A sample C++ project for HNSW test"
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(HNSW_TEST_BUILD_CORRECT_TEST "Build correctness test target" ON)

find_package(OpenMP REQUIRED)

add_subdirectory(../vector_test ${CMAKE_CURRENT_BINARY_DIR}/vector_test_build)

add_executable(${PROJECT_NAME}
    src/main.cpp
    src/hnsw_index.cpp
)

target_include_directories(${PROJECT_NAME}
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/../vector_test/include
)

target_link_libraries(${PROJECT_NAME}
    PRIVATE
        OpenMP::OpenMP_CXX
        VectorTest
)

if(HNSW_TEST_BUILD_CORRECT_TEST)
    enable_testing()
    add_subdirectory(test)
endif()
