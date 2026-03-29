cmake_minimum_required(VERSION 3.18 FATAL_ERROR)



#mainline tests

ConfigureGPUErrExecutable(log_test "${CMAKE_CURRENT_SOURCE_DIR}/src/log_test.cu" "${GPU_ERR_TESTS_BINARY_DIR}")

ConfigureGPUErrExecutable(error_test "${CMAKE_CURRENT_SOURCE_DIR}/src/error_test.cu" "${GPU_ERR_TESTS_BINARY_DIR}")
ConfigureGPUErrExecutable(assertion_test "${CMAKE_CURRENT_SOURCE_DIR}/src/assertion_test.cu" "${GPU_ERR_TESTS_BINARY_DIR}")

ConfigureGPUErrExecutable(assertion_throughput_test "${CMAKE_CURRENT_SOURCE_DIR}/src/assertion_throughput_test.cu" "${GPU_ERR_TESTS_BINARY_DIR}")

ConfigureGPUErrExecutable(disable_log_test "${CMAKE_CURRENT_SOURCE_DIR}/src/disable_log_test.cu" "${GPU_ERR_TESTS_BINARY_DIR}")

ConfigureGPUErrExecutable(logging_mode_test "${CMAKE_CURRENT_SOURCE_DIR}/src/logging_mode_test.cu" "${GPU_ERR_TESTS_BINARY_DIR}")
