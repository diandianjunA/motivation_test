cmake_minimum_required(VERSION 3.18)

project(ShineGpuVectorTest LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

find_package(CUDAToolkit 11.8 REQUIRED)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(Boost COMPONENTS program_options REQUIRED)
find_package(Threads REQUIRED)
find_package(TBB REQUIRED)

# Force system rdma-core headers and shared libibverbs
foreach(_cache_var
    IBVERBS_INCLUDE_DIR
    IBVERBS_LIBRARY
    IBVerbs_INCLUDE_DIR
    IBVerbs_LIBRARY
)
    if(DEFINED ${_cache_var} AND "${${_cache_var}}" MATCHES "^/usr/local/")
        unset(${_cache_var} CACHE)
        unset(${_cache_var})
    endif()
endforeach()
set(_ibverbs_include_paths /usr/include)
set(_ibverbs_library_paths
    /usr/lib/${CMAKE_LIBRARY_ARCHITECTURE}
    /lib/${CMAKE_LIBRARY_ARCHITECTURE}
    /usr/lib64 /lib64 /usr/lib /lib
)
find_path(IBVERBS_INCLUDE_DIR
    NAMES infiniband/verbs.h
    PATHS ${_ibverbs_include_paths}
    NO_DEFAULT_PATH REQUIRED
)
find_library(IBVERBS_LIBRARY
    NAMES ibverbs
    PATHS ${_ibverbs_library_paths}
    NO_DEFAULT_PATH REQUIRED
)
foreach(_resolved_var IBVERBS_INCLUDE_DIR IBVERBS_LIBRARY)
    if(DEFINED ${_resolved_var} AND "${${_resolved_var}}" MATCHES "^/usr/local/")
        message(FATAL_ERROR "Refusing legacy RDMA artifact from /usr/local: ${_resolved_var}=${${_resolved_var}}")
    endif()
endforeach()
if(IBVERBS_LIBRARY MATCHES "\\.a$")
    message(FATAL_ERROR "Expected shared libibverbs, found static archive: ${IBVERBS_LIBRARY}")
endif()

find_library(NL_ROUTE_LIBRARY nl-route-3 REQUIRED)
find_library(NL_3_LIBRARY nl-3 REQUIRED)

# Pre-built libraries
find_library(SHINE_RUNTIME_LIBRARY shine_runtime
    PATHS ${CMAKE_CURRENT_SOURCE_DIR}/lib NO_DEFAULT_PATH REQUIRED)
find_library(SHINE_GPU_KERNELS_LIBRARY shine_gpu_kernels
    PATHS ${CMAKE_CURRENT_SOURCE_DIR}/lib NO_DEFAULT_PATH REQUIRED)
find_library(RDMA_LIBRARY rdma_library
    PATHS ${CMAKE_CURRENT_SOURCE_DIR}/lib NO_DEFAULT_PATH REQUIRED)
find_library(VECTOR_TEST_LIBRARY VectorTest
    PATHS ${CMAKE_CURRENT_SOURCE_DIR}/lib NO_DEFAULT_PATH REQUIRED)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/include/shine_gpu/src
    ${CMAKE_CURRENT_SOURCE_DIR}/include/shine_gpu/rdma-library
    ${CMAKE_CURRENT_SOURCE_DIR}/include/shine_gpu/thirdparty
)

add_executable(${PROJECT_NAME}
    src/main.cpp
    src/shine_gpu_index.cpp
)
target_include_directories(${PROJECT_NAME} SYSTEM BEFORE PRIVATE ${IBVERBS_INCLUDE_DIR})

target_link_libraries(${PROJECT_NAME} PRIVATE
    ${SHINE_RUNTIME_LIBRARY}
    ${SHINE_GPU_KERNELS_LIBRARY}
    ${RDMA_LIBRARY}
    ${VECTOR_TEST_LIBRARY}
    CUDA::cudart
    CUDA::cublas
    Eigen3::Eigen
    ${IBVERBS_LIBRARY}
    ${Boost_LIBRARIES}
    TBB::tbb
    Threads::Threads
    ${NL_ROUTE_LIBRARY}
    ${NL_3_LIBRARY}
)
