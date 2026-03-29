cmake_minimum_required(VERSION 3.16)

project(FreshDiskANNTest
    VERSION 1.0.0
    DESCRIPTION "FreshDiskANN test project with VectorTest"
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(FRESH_DISKANN_BUILD_CORRECT_TEST "Build correctness test target" ON)

find_package(OpenMP REQUIRED)

add_subdirectory(../vector_test ${CMAKE_CURRENT_BINARY_DIR}/vector_test_build)

set(FRESH_DISKANN_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/FreshDiskANN-baseline)
set(FRESH_DISKANN_SOURCES
    ${FRESH_DISKANN_ROOT}/src/ann_exception.cpp
    ${FRESH_DISKANN_ROOT}/src/aux_utils.cpp
    ${FRESH_DISKANN_ROOT}/src/distance.cpp
    ${FRESH_DISKANN_ROOT}/src/index.cpp
    ${FRESH_DISKANN_ROOT}/src/linux_aligned_file_reader.cpp
    ${FRESH_DISKANN_ROOT}/src/logger.cpp
    ${FRESH_DISKANN_ROOT}/src/math_utils.cpp
    ${FRESH_DISKANN_ROOT}/src/memory_mapper.cpp
    ${FRESH_DISKANN_ROOT}/src/partition_and_pq.cpp
    ${FRESH_DISKANN_ROOT}/src/pq_flash_index.cpp
    ${FRESH_DISKANN_ROOT}/src/utils.cpp
    ${FRESH_DISKANN_ROOT}/src/v2/graph_delta.cpp
    ${FRESH_DISKANN_ROOT}/src/v2/index_merger.cpp
    ${FRESH_DISKANN_ROOT}/src/v2/merge_insert.cpp
)

add_library(FreshDiskANN STATIC ${FRESH_DISKANN_SOURCES})

target_include_directories(FreshDiskANN
    PUBLIC
        ${FRESH_DISKANN_ROOT}/include
        ${FRESH_DISKANN_ROOT}/include/tsl/include
        /usr/include/mkl
)

target_compile_definitions(FreshDiskANN
    PUBLIC
        MKL_ILP64
        USE_ACCELERATED_PQ
        USE_AVX2
)

target_compile_options(FreshDiskANN
    PRIVATE
        -march=native
        -fopenmp-simd
        -funroll-loops
)

target_link_libraries(FreshDiskANN
    PUBLIC
        OpenMP::OpenMP_CXX
        uring
        mkl_rt
        tcmalloc_minimal
        pthread
        dl
        m
)

add_executable(${PROJECT_NAME}
    src/main.cpp
    src/fresh_diskann_index.cpp
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
        FreshDiskANN
        VectorTest
)

if(FRESH_DISKANN_BUILD_CORRECT_TEST)
    enable_testing()
    add_subdirectory(test)
endif()
