cmake_minimum_required(VERSION 3.18)

project(ShineGpuVectorTest LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

find_package(CUDAToolkit REQUIRED)
find_package(OpenSSL REQUIRED)
find_package(Boost COMPONENTS program_options REQUIRED)
find_package(Threads REQUIRED)
find_package(TBB REQUIRED)
find_library(IBVERBS_LIBRARY ibverbs REQUIRED)
find_library(NL_ROUTE_LIBRARY nl-route-3 REQUIRED)
find_library(NL_3_LIBRARY nl-3 REQUIRED)
find_library(SHINE_RUNTIME_LIBRARY shine_runtime PATHS ${CMAKE_CURRENT_SOURCE_DIR}/lib NO_DEFAULT_PATH REQUIRED)
find_library(RDMA_LIBRARY rdma_library PATHS ${CMAKE_CURRENT_SOURCE_DIR}/lib NO_DEFAULT_PATH REQUIRED)
find_library(VECTOR_TEST_LIBRARY VectorTest PATHS ${CMAKE_CURRENT_SOURCE_DIR}/lib NO_DEFAULT_PATH REQUIRED)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/include/shine/src
    ${CMAKE_CURRENT_SOURCE_DIR}/include/shine/rdma-library
    ${CMAKE_CURRENT_SOURCE_DIR}/include/shine/thirdparty
)

add_executable(${PROJECT_NAME}
    src/main.cpp
    src/shine_gpu_index.cpp
)

target_link_libraries(${PROJECT_NAME} PRIVATE
    ${SHINE_RUNTIME_LIBRARY}
    ${RDMA_LIBRARY}
    ${VECTOR_TEST_LIBRARY}
    CUDA::cudart
    OpenSSL::SSL
    OpenSSL::Crypto
    ${IBVERBS_LIBRARY}
    ${Boost_LIBRARIES}
    TBB::tbb
    Threads::Threads
    ${NL_ROUTE_LIBRARY}
    ${NL_3_LIBRARY}
    ${CMAKE_DL_LIBS}
)
