cmake_minimum_required(VERSION 3.16)

project(OdinANNTest
    VERSION 1.0.0
    DESCRIPTION "OdinANN test project with VectorTest"
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(ODINANN_TEST_BUILD_CORRECT_TEST "Build correctness test target" ON)

find_package(OpenMP REQUIRED)
find_package(BLAS REQUIRED)

add_subdirectory(../vector_test ${CMAKE_CURRENT_BINARY_DIR}/vector_test_build)

set(ODINANN_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/26FAST-PipeANN)
file(GLOB ODINANN_SOURCES
    ${ODINANN_ROOT}/src/*.cpp
    ${ODINANN_ROOT}/src/search/*.cpp
    ${ODINANN_ROOT}/src/update/*.cpp
    ${ODINANN_ROOT}/src/utils/*.cpp
)

add_library(OdinANN STATIC ${ODINANN_SOURCES})

target_include_directories(OdinANN
    PUBLIC
        ${ODINANN_ROOT}/include
)

target_compile_definitions(OdinANN
    PUBLIC
        NDEBUG
        OVERLAP_INIT
        DYN_PIPE_WIDTH
)

target_compile_options(OdinANN
    PRIVATE
        -Wall
        -Wextra
        -Wfatal-errors
        -Wno-class-memaccess
        -Wno-unused-variable
        -Wno-unused-parameter
        -Wno-unused-but-set-variable
        -fopenmp-simd
        -O3
        -g
        -march=native
        -mtune=native
        -ftree-vectorize
        -funroll-loops
)

target_link_libraries(OdinANN
    PUBLIC
        OpenMP::OpenMP_CXX
        ${BLAS_LIBRARIES}
        aio
        uring
        jemalloc
)

add_executable(${PROJECT_NAME}
    src/main.cpp
    src/odinann_index.cpp
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
        OdinANN
        VectorTest
)

if(ODINANN_TEST_BUILD_CORRECT_TEST)
    enable_testing()
    add_subdirectory(test)
endif()
