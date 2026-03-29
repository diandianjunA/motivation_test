function(ConfigureGPUErrExecutable EXE_NAME EXE_SRC EXE_DEST)
    add_executable(${EXE_NAME} "${EXE_SRC}")
    set_target_properties(${EXE_NAME} PROPERTIES
                                          CUDA_ARCHITECTURES ${GPU_ARCHS}
                                          RUNTIME_OUTPUT_DIRECTORY "${EXE_DEST}")
    target_include_directories(${EXE_NAME} PRIVATE
                                             "${CMAKE_CURRENT_SOURCE_DIR}")
    target_link_libraries(${EXE_NAME} PRIVATE gpu_error)

    if (NOT USE_ASSERTIONS)
      target_compile_definitions(${EXE_NAME} PRIVATE GPU_NDEBUG)
    endif()

endfunction(ConfigureGPUErrExecutable)


function(ConfigureGPUErrStatus EXE_NAME)

    if (NOT USE_ASSERTIONS)

      message("Disabling assertions for ${EXE_NAME}")
      target_compile_definitions(${EXE_NAME} PRIVATE GPU_NDEBUG)
    endif()

endfunction(ConfigureGPUErrStatus)