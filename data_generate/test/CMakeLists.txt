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
link_directories(../build)
link_directories(/usr/local/lib)

add_executable(basic_test
    basic_test.cpp
)

target_link_libraries(basic_test DataGenerate ${GTEST_LIBRARIES} ${GTEST_MAIN_LIBRARIES} OpenMP::OpenMP_CXX
    faiss
    faiss_avx2
    cublas cudart cuda curand
    openblas
)

add_executable(1024dim100K_generator
    1024dim100K.cpp
)

target_link_libraries(1024dim100K_generator DataGenerate ${GTEST_LIBRARIES} ${GTEST_MAIN_LIBRARIES} OpenMP::OpenMP_CXX
    faiss
    faiss_avx2
    cublas cudart cuda curand
    openblas
)

add_executable(1024dim1M_generator
    1024dim1M.cpp
)

target_link_libraries(1024dim1M_generator DataGenerate ${GTEST_LIBRARIES} ${GTEST_MAIN_LIBRARIES} OpenMP::OpenMP_CXX
    faiss
    faiss_avx2
    cublas cudart cuda curand
    openblas
)

add_executable(1024dim10M_generator
    1024dim10M.cpp
)

target_link_libraries(1024dim10M_generator DataGenerate ${GTEST_LIBRARIES} ${GTEST_MAIN_LIBRARIES} OpenMP::OpenMP_CXX
    faiss
    faiss_avx2
    cublas cudart cuda curand
    openblas
)

add_executable(1024dim100M_generator
    1024dim100M.cpp
)

target_link_libraries(1024dim100M_generator DataGenerate ${GTEST_LIBRARIES} ${GTEST_MAIN_LIBRARIES} OpenMP::OpenMP_CXX
    faiss
    faiss_avx2
    cublas cudart cuda curand
    openblas
)

add_executable(correct_test
    correct_test.cpp
)

target_link_libraries(correct_test DataGenerate ${GTEST_LIBRARIES} ${GTEST_MAIN_LIBRARIES} OpenMP::OpenMP_CXX
    faiss
    faiss_avx2
    cublas cudart cuda curand
    openblas
)

add_executable(correct_test2
    correct_test2.cpp
)

target_link_libraries(correct_test2 DataGenerate ${GTEST_LIBRARIES} ${GTEST_MAIN_LIBRARIES} OpenMP::OpenMP_CXX
    faiss
    faiss_avx2
    cublas cudart cuda curand
    openblas
)

add_executable(correct_test3
    correct_test3.cpp
)

target_link_libraries(correct_test3 DataGenerate ${GTEST_LIBRARIES} ${GTEST_MAIN_LIBRARIES} OpenMP::OpenMP_CXX
    faiss
    faiss_avx2
    cublas cudart cuda curand
    openblas
)