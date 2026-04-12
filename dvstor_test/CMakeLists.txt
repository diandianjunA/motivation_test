cmake_minimum_required(VERSION 3.18)

project(DvstorVectorTest LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(DVSTOR_ROOT "/home/xjs/experiment/dvstor" CACHE PATH "Path to the DVSTOR source tree")

function(dvstor_ignore_prefix prefix)
    if(prefix AND IS_DIRECTORY "${prefix}")
        list(APPEND CMAKE_IGNORE_PREFIX_PATH "${prefix}")
        list(APPEND CMAKE_IGNORE_PATH
            "${prefix}/bin"
            "${prefix}/include"
            "${prefix}/lib"
            "${prefix}/lib/cmake"
            "${prefix}/lib64"
        )
        set(CMAKE_IGNORE_PREFIX_PATH "${CMAKE_IGNORE_PREFIX_PATH}" PARENT_SCOPE)
        set(CMAKE_IGNORE_PATH "${CMAKE_IGNORE_PATH}" PARENT_SCOPE)
    endif()
endfunction()

dvstor_ignore_prefix("$ENV{CONDA_PREFIX}")
dvstor_ignore_prefix("/home/xjs/anaconda3")

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    execute_process(
        COMMAND ${CMAKE_CXX_COMPILER} -print-file-name=libstdc++.so
        OUTPUT_VARIABLE DVSTOR_LIBSTDCXX
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    get_filename_component(DVSTOR_LIBSTDCXX_REALPATH "${DVSTOR_LIBSTDCXX}" REALPATH)
    get_filename_component(DVSTOR_LIBSTDCXX_DIR "${DVSTOR_LIBSTDCXX}" DIRECTORY)
    get_filename_component(DVSTOR_LIBSTDCXX_RUNTIME_DIR "${DVSTOR_LIBSTDCXX_REALPATH}" DIRECTORY)
    if(DVSTOR_LIBSTDCXX_DIR AND IS_DIRECTORY "${DVSTOR_LIBSTDCXX_DIR}")
        link_directories(BEFORE "${DVSTOR_LIBSTDCXX_DIR}")
        add_link_options(
            "SHELL:-L${DVSTOR_LIBSTDCXX_DIR}"
            "SHELL:-Wl,--disable-new-dtags"
            "SHELL:-Wl,-rpath,${DVSTOR_LIBSTDCXX_DIR}"
        )
    endif()
    if(DVSTOR_LIBSTDCXX_RUNTIME_DIR AND IS_DIRECTORY "${DVSTOR_LIBSTDCXX_RUNTIME_DIR}")
        add_link_options("SHELL:-Wl,-rpath,${DVSTOR_LIBSTDCXX_RUNTIME_DIR}")
    endif()
endif()

find_package(CUDAToolkit 11.8 REQUIRED)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(Boost COMPONENTS program_options REQUIRED)
find_package(Threads REQUIRED)
find_package(TBB REQUIRED)

find_path(IBVERBS_INCLUDE_DIR
    NAMES infiniband/verbs.h
    PATHS /usr/include
    NO_DEFAULT_PATH REQUIRED
)
find_library(IBVERBS_LIBRARY
    NAMES ibverbs
    PATHS /usr/lib/${CMAKE_LIBRARY_ARCHITECTURE} /lib/${CMAKE_LIBRARY_ARCHITECTURE} /usr/lib64 /lib64 /usr/lib /lib
    NO_DEFAULT_PATH REQUIRED
)
find_library(NL_ROUTE_LIBRARY nl-route-3 REQUIRED)
find_library(NL_3_LIBRARY nl-3 REQUIRED)

set(DOCA_LIBRARY_DIR "/opt/mellanox/doca/lib/x86_64-linux-gnu")

find_library(DVSTOR_RUNTIME_LIBRARY dvstor_runtime
    PATHS ${CMAKE_CURRENT_SOURCE_DIR}/lib NO_DEFAULT_PATH REQUIRED)
find_library(DVSTOR_GPU_KERNELS_LIBRARY dvstor_gpu_kernels
    PATHS ${CMAKE_CURRENT_SOURCE_DIR}/lib NO_DEFAULT_PATH REQUIRED)
find_library(RDMA_LIBRARY rdma_library
    PATHS ${CMAKE_CURRENT_SOURCE_DIR}/lib NO_DEFAULT_PATH REQUIRED)
find_library(VECTOR_TEST_LIBRARY VectorTest
    PATHS ${CMAKE_CURRENT_SOURCE_DIR}/lib NO_DEFAULT_PATH REQUIRED)

add_executable(${PROJECT_NAME}
    src/main.cpp
    src/dvstor_index.cpp
)

target_include_directories(${PROJECT_NAME} SYSTEM BEFORE PRIVATE ${IBVERBS_INCLUDE_DIR})
target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${DVSTOR_ROOT}/src
    ${DVSTOR_ROOT}/rdma-library
    ${DVSTOR_ROOT}/thirdparty
    /opt/mellanox/doca/include
)
target_link_directories(${PROJECT_NAME} PRIVATE ${DOCA_LIBRARY_DIR})

target_link_libraries(${PROJECT_NAME} PRIVATE
    ${DVSTOR_RUNTIME_LIBRARY}
    ${DVSTOR_GPU_KERNELS_LIBRARY}
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
    doca_gpunetio
    doca_verbs
    doca_common
    doca_rdma
    mlx5
)
