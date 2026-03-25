# RDMA Library

"High-level" library to connect machines, connect queue pairs, register memory regions, post RDMA verbs, etc.
The goal of this library is to conveniently wrap
the [ibverbs library](https://github.com/linux-rdma/rdma-core/tree/master/libibverbs).

[TODO: public libary interface and namespaces...]

## Required C++ Libraries

* ibverbs
* Boost (for CLI parsing)
* pthreads (for multithreading)
* oneTBB (for concurrent data structures)

## In This Repository

`rdma-library/` is vendored directly into this repository and is built automatically by the top-level `CMakeLists.txt`. No extra checkout is needed when building SHINE.

## Using RDMA Library in Another Project

If you want to reuse it elsewhere, add the directory to your project and include it from CMake:

```
add_subdirectory(rdma-library)
```

Finally, link the executable with the library in `CMakeLists.txt`, e.g.,
```
add_executable(ping_pong src/ping_pong.cc)
target_link_libraries(ping_pong rdma_library)
```
