
# The GPU Error Library

This is a little helper library to assist with building/debugging GPU applications. The main object exposed is the log in `<gpu_error/log.cuh>`, which exposes a logging object that is instantiated on host and used on device. A lightweight custom GPU string implementation based on variadic templates is used to construct log objects, and there is support for custom errors and assertions that correctly pause execution.


## String Implementation

custring is a lightweight-ish implementation of a string using the Gallatin allocator to perform dynamic memory management.

custrings can be defined per-object, but the fastest way to instantiate a string is with the `make_string` variadic template. This accepts any number of arguments where each argument is of type `const char *`, `char *`, `custring`, `uint64_t`, `double`, `int64_t`, or any type that can be safely typecast to the above. Arguments are passed in order to construct a string. For example, to produce the string:

`"Thread 0 reads value 12.5322 from address 15\n"`

you call:

`custring str = make_string("Thread ", tid, " reads value ", arr[address], " from address ", address, "\n");`

This variadic functionality is extended to all logging functions! The size, # of arguments, and argument types are determined at compile time, allowing for fast string construction during runtime.


## Host API

- `init_gpu_log(optional uint64_t n_bytes)`: intializes the log and backing allocator. This must be called before ANY log, assertion, or error calls are made. Allocator defaults to 8GB of host memory. To allocate a different amount of memory, specify the amount as `n_bytes`. Specifying a value that is too low many not allow the allocator to initialize correctly: I would recommend at least 1GB.
- `free_gpu_log`: Release underlying memory of the logging system. After this is called, all other log functionality is disabled. Call this last.
- `std::vector<std::string> export_log()` - writes all logs in device memory to a host vector as strings. This is not safe for concurrency with the GPU, i.e. call cudaDeviceSynchronize() and ensure no logging kernels are running.



## Device API

All logging functionality obeys strict serial ordering per-thread. It is not possible for one thread to write to the log out of order, but there is no enforced ordering between different threads.

-  `gpu_log(...)`: construct and store a custring in the log with arguments `...`, where `...` is a variadic template of castable types. Logs are collected in order via an atomicAdd, and the log operation is thread-safe. Logging does not pause execution.
-  `gpu_error(...)`: construct and store a custring representing an error. This logs the file and line info, similar to a CPU Exception call. In the default mode, this also triggers an `asm volatile(trap;);` call that will pause the debugger on the error and end execution of the kernel in regular programs. If you want errors to be non-fatal, you can use pass the flag `LOG_GPU_ERRORS` to log errors into the log instead. Errors that are logged to not trap and will not pause execution.
- `gpu_assert(boolean expression, ...)`: resolves a boolean expression. If the result is not true, construct a custring representing the assertion failure. Similar to `gpu_error`, the default behavior is to kill the kernel and pause execution on the line where the assertion was triggered, and can be converted to a log via the `LOG_GPU_ERRORS` flag. The assertion does no work besides the resolution of the boolean statement if the expression is true.

## Disabling Logging
By default, logging is enabled. When you've determined your code is bulletproof and you no longer need to log, it can be disabled by passing the `GPU_NDEBUG` flag. This will convert all logging, error, and assertion statements into `NOP`. When this behavior is enabled, there is no need to construct/destruct the log, so it is safe to call all functions even if the log has not been initialized. This means that it is safe to enable logging behavior in your own tests even if downstream users do not use the log. To speed up the variadic templatization, custring is disabled when `GPU_NDEBUG` is passed as well.


## Tests

There are a few sanity tests included that show the behavior. To enable these, pass the flag `-DGPU_ERR_TESTS=ON` to cmake during construction.

The following tests are present:
- `log_test`: tests that the log compiles and that basic strings can be written and collected.
- `error_test`: tests that errors trigger and kill the kernel
- `assertion_test`: tests that assertions successfully trigger and are accessible.
- `disable_log`: tests that a disabled log does not produce any output and does not affect the kernel runtime.
- `assertion_throughput_test`: measures the cost of untriggered assertions when used in a kernel.
- `logging_mode_test`: tests that with `LOG_GPU_ERRORS` passed, errors and assertions are logged like regular log objects.


# Other

Code is licensed with the BSD-3 license, and pull requests are welcome.



