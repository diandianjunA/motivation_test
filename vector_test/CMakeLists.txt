cmake_minimum_required(VERSION 3.10)
project(VectorTest VERSION 1.0.0 LANGUAGES CXX)

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

include_directories(include)

# 添加库目标
add_library(${PROJECT_NAME} STATIC
    src/vector_test.cpp
    src/util.cpp
    src/logger.cpp
    src/config.cpp
    src/component/memory_monitor.cpp
    src/component/stat.cpp
    src/component/timer.cpp
)