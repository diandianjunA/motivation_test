cmake_minimum_required(VERSION 3.10)
project(DataGenerateTest VERSION 1.0.0 LANGUAGES CXX)

# 设置C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 设置编译选项
if(MSVC)
    add_compile_options(/W4 /WX)
else()
    add_compile_options(-Wall -Wextra -Werror -pedantic)
endif()

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# 查找依赖库
find_package(GTest REQUIRED)
find_package(OpenMP REQUIRED)

include_directories(../include)
include_directories(./include)
include_directories(${GTEST_INCLUDE_DIRS})
link_directories(./lib)
link_directories(../lib)
link_directories(../build)
link_directories(/usr/local/lib)

add_executable(correct_test correct_test.cpp ../src/diskann_index.cpp)

target_link_libraries(correct_test OpenMP::OpenMP_CXX
    diskann
    faiss
    openblas
    lapacke
    VectorTest
    GTest::GTest
    GTest::Main
    DataGenerate
)
