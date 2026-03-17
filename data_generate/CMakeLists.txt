cmake_minimum_required(VERSION 3.10)
project(DataGenerate VERSION 1.0.0 LANGUAGES CXX)

# 设置C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# 启用OpenMP
find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP found: ${OpenMP_CXX_VERSION}")
endif()

include_directories(include)
link_directories(/usr/local/lib)

add_library(${PROJECT_NAME}
    src/experiment_generator.cpp
    src/ground_truth_calculator.cpp
    src/random_data_generator.cpp
)

target_link_libraries(${PROJECT_NAME} OpenMP::OpenMP_CXX
    faiss
    faiss_avx2
    cublas cudart cuda curand
    openblas
)

add_executable(read_bin src/read_bin.cpp)

add_executable(read_fbin src/read_fbin.cpp)

add_executable(read_fbin2 src/read_fbin2.cpp src/random_data_generator.cpp)

add_executable(read_fbin3 src/read_fbin3.cpp)

add_executable(calculate_ground_truth src/calculate_ground_truth.cpp src/ground_truth_calculator.cpp)
target_link_libraries(calculate_ground_truth OpenMP::OpenMP_CXX
    faiss
    faiss_avx2
    cublas cudart cuda curand
    openblas
)
