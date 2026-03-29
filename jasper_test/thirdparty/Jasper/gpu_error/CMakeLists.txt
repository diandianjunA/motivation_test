cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

project(gallatin VERSION 1.0 LANGUAGES CXX CUDA)



###################################################################################################
# - build type ------------------------------------------------------------------------------------

# Set a default build type if none was specified
set(DEFAULT_BUILD_TYPE "Release")

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to '${DEFAULT_BUILD_TYPE}' since none specified.")
  set(CMAKE_BUILD_TYPE "${DEFAULT_BUILD_TYPE}" CACHE STRING "Choose the type of build." FORCE)
  # Set the possible values of build type for cmake-gui
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Debug")

  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G -g")

endif()

if (DEFINED GPU_ARCHS)
  message("GPU_ARCHS is defined as ${GPU_ARCHS}")
else()
  message("GPU_ARCHS not specified, auto-detecting GPU...")
  set(GPU_ARCHS "")
  include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/EvalGpuArchs.cmake)
  evaluate_gpu_archs(GPU_ARCHS)
endif()

#set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda-12.1)

find_package(CUDAToolkit 11.8 REQUIRED)
set(CMAKE_CUDA_RUNTIME_LIBRARY SHARED)

#package manager
include(cmake/CPM.cmake)

#macro for configuring executable
include(cmake/ConfigureExecutable.cmake)

if(NOT TARGET gallatin)
  CPMAddPackage(
    NAME gallatin
    GITHUB_REPOSITORY saltsystemslab/gallatin
    GIT_TAG origin/inlined_commands
  )
endif()

## Package Target
add_library(gpu_error INTERFACE)

target_include_directories(gpu_error INTERFACE
                           "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
                           $<INSTALL_INTERFACE:include>)

target_link_libraries(gpu_error INTERFACE gallatin)


option(LINFO "Add lineInfo for NSight Compute profiling" OFF)

if (LINFO)
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo")
endif(LINFO)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_C_COMPILER $ENV{CC})
set(CMAKE_CXX_COMPILER $ENV{CXX})


set(GPU_ERR_TESTS_BINARY_DIR "${CMAKE_BINARY_DIR}/tests")

option(GPU_ERR_TESTS "Configure CMake to build tests for the allocator" OFF)

option(USE_ASSERTIONS "If enabled, assertions are present. Disabling this removes device-side asserts and errors." OFF)

# if (NOT USE_ASSERTIONS)

#   message("GPU-side Assertions are Disabled")
#   add_compile_definitions(GPU_NDEBUG)
#   add_definitions(-DGPU_NDEBUG) 
# endif()


if(GPU_ERR_TESTS)
    add_subdirectory(tests)
endif(GPU_ERR_TESTS)



