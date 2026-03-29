cmake_minimum_required(VERSION 3.18)

project(JasperTest
    VERSION 1.0.0
    DESCRIPTION "Jasper test project with VectorTest"
    LANGUAGES CXX CUDA
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

find_package(OpenMP REQUIRED)

add_subdirectory(../vector_test ${CMAKE_CURRENT_BINARY_DIR}/vector_test_build)

set(JASPER_BUILD_EXAMPLES OFF CACHE BOOL "Build Jasper examples" FORCE)
add_subdirectory(thirdparty/Jasper ${CMAKE_CURRENT_BINARY_DIR}/jasper_build)

add_executable(${PROJECT_NAME}
    src/main.cpp
    src/jasper_index.cu
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
        jasper
        VectorTest
        dl
)

target_compile_options(${PROJECT_NAME}
    PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)

set_target_properties(${PROJECT_NAME} PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/bin"
)

if(DEFINED GPU_ARCHS AND NOT "${GPU_ARCHS}" STREQUAL "")
    set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_ARCHITECTURES "${GPU_ARCHS}")
endif()
