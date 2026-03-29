cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

project(jasper VERSION 1.0 LANGUAGES CXX CUDA)

# --------------------------------------------------------------------------
# Options
# --------------------------------------------------------------------------
option(JASPER_BUILD_EXAMPLES "Build Jasper example programs" ON)

# --------------------------------------------------------------------------
# Build type
# --------------------------------------------------------------------------
set(DEFAULT_BUILD_TYPE "Release")
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE "${DEFAULT_BUILD_TYPE}" CACHE STRING "" FORCE)
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G -g")
endif()

# --------------------------------------------------------------------------
# GPU architecture
# --------------------------------------------------------------------------
if(DEFINED GPU_ARCHS)
  message(STATUS "Jasper: GPU_ARCHS = ${GPU_ARCHS}")
else()
  message(STATUS "Jasper: auto-detecting GPU architecture...")
  set(GPU_ARCHS "")
  include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/EvalGpuArchs.cmake)
  evaluate_gpu_archs(GPU_ARCHS)
endif()

# --------------------------------------------------------------------------
# CUDA & dependencies
# --------------------------------------------------------------------------
find_package(CUDAToolkit 11.8 REQUIRED)
set(CMAKE_CUDA_RUNTIME_LIBRARY SHARED)

# CPM (reuse parent cache if available)
if(NOT DEFINED CPM_SOURCE_CACHE)
  set(CPM_SOURCE_CACHE ${CMAKE_CURRENT_SOURCE_DIR}/cache)
endif()
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/CPM.cmake)

CPMAddPackage(
  NAME Eigen
  GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
  GIT_TAG 3.4.0
)
find_package(Eigen3 REQUIRED)

file(GLOB JASPER_LOCAL_GALLATIN_DIRS LIST_DIRECTORIES true
  "${CMAKE_CURRENT_SOURCE_DIR}/cache/gallatin/*")
list(LENGTH JASPER_LOCAL_GALLATIN_DIRS JASPER_LOCAL_GALLATIN_COUNT)

if(JASPER_LOCAL_GALLATIN_COUNT GREATER 0)
  list(GET JASPER_LOCAL_GALLATIN_DIRS 0 JASPER_LOCAL_GALLATIN_DIR)
  message(STATUS "Jasper: using vendored Gallatin at ${JASPER_LOCAL_GALLATIN_DIR}")
  add_subdirectory(${JASPER_LOCAL_GALLATIN_DIR}
                   ${CMAKE_CURRENT_BINARY_DIR}/gallatin_local_build)
else()
  CPMAddPackage(
    NAME gallatin
    GITHUB_REPOSITORY saltsystemslab/gallatin
    GIT_TAG origin/inlined_commands
  )
endif()

# gpu_error submodule (from parent project)
if(NOT TARGET gpu_error)
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/gpu_error
                   ${CMAKE_CURRENT_BINARY_DIR}/gpu_error)
endif()

# --------------------------------------------------------------------------
# C++ / CUDA standards
# --------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -rdc=true -lcudadevrt -lcudart")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --ptxas-options=-v")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")

# --------------------------------------------------------------------------
# Jasper interface library
# --------------------------------------------------------------------------
add_library(jasper INTERFACE)

target_include_directories(jasper INTERFACE
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
  $<INSTALL_INTERFACE:include>
)

target_link_libraries(jasper INTERFACE
  CUDA::cudart
  CUDA::cublas
  CUDA::curand
  ssl crypto
  gpu_error
  gallatin
  Eigen3::Eigen
)

target_compile_options(jasper INTERFACE
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)

# --------------------------------------------------------------------------
# Helper function for building Jasper executables
# --------------------------------------------------------------------------
function(ConfigureJasperExecutable EXE_NAME EXE_SRC)
  add_executable(${EXE_NAME} "${EXE_SRC}")
  set_target_properties(${EXE_NAME} PROPERTIES
    CUDA_ARCHITECTURES ${GPU_ARCHS}
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/bin"
  )
  target_link_libraries(${EXE_NAME} PRIVATE jasper)
  target_link_libraries(${EXE_NAME} PRIVATE dl)
endfunction()

# --------------------------------------------------------------------------
# Examples
# --------------------------------------------------------------------------
if(JASPER_BUILD_EXAMPLES)
  ConfigureJasperExecutable(jasper_example
    ${CMAKE_CURRENT_SOURCE_DIR}/examples/example.cu)
  ConfigureJasperExecutable(jasper_example2
    ${CMAKE_CURRENT_SOURCE_DIR}/examples/example2.cu)
endif()
