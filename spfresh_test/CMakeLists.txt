# CMake 最低版本要求
cmake_minimum_required(VERSION 3.16)

# 项目信息
project(SPFreshTest
    VERSION 1.0.0
    DESCRIPTION "A sample C++ project for SPFreshTest"
    LANGUAGES C CXX
)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 查找 OpenMP 包
find_package(OpenMP REQUIRED)
# 将 OpenMP 的编译标志和链接标志分别设置到变量中
if(OpenMP_CXX_FOUND) # 检查是否找到C++的OpenMP
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    # 将链接库路径和库名保存在一个变量中，供后续使用
    set(OPENMP_LIBRARIES ${OpenMP_CXX_LIBRARIES})
endif()

set(Spdk ${PROJECT_SOURCE_DIR}/lib/spdk)
set(Dpdk ${PROJECT_SOURCE_DIR}/lib/dpdk)
set(IsalLCrypto ${PROJECT_SOURCE_DIR}/lib/libisal_crypto.a)
set(SpdkLibPrefix ${Spdk}/lib/libspdk_)
set(DpdkLibPrefix ${Dpdk}/lib/librte_)
set(SPDK_LIBRARIES -Wl,--whole-archive ${SpdkLibPrefix}bdev_nvme.a ${SpdkLibPrefix}bdev.a ${SpdkLibPrefix}nvme.a ${SpdkLibPrefix}vfio_user.a ${SpdkLibPrefix}sock.a ${SpdkLibPrefix}dma.a ${SpdkLibPrefix}notify.a ${SpdkLibPrefix}accel.a ${SpdkLibPrefix}event_bdev.a ${SpdkLibPrefix}event_accel.a ${SpdkLibPrefix}vmd.a ${SpdkLibPrefix}event_vmd.a ${SpdkLibPrefix}event_sock.a ${SpdkLibPrefix}event_iobuf.a ${SpdkLibPrefix}event.a ${SpdkLibPrefix}env_dpdk.a ${SpdkLibPrefix}log.a ${SpdkLibPrefix}thread.a ${SpdkLibPrefix}rpc.a ${SpdkLibPrefix}init.a ${SpdkLibPrefix}jsonrpc.a ${SpdkLibPrefix}json.a ${SpdkLibPrefix}trace.a ${SpdkLibPrefix}util.a ${DpdkLibPrefix}mempool.a ${DpdkLibPrefix}mempool_ring.a ${DpdkLibPrefix}eal.a ${DpdkLibPrefix}ring.a ${DpdkLibPrefix}telemetry.a ${DpdkLibPrefix}bus_pci.a ${DpdkLibPrefix}kvargs.a ${DpdkLibPrefix}pci.a -Wl,--no-whole-archive dl rt isal ${IsalLCrypto} uuid)

# 构建类型配置
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# 包含目录
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/inc
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    
)

link_directories(lib)

# 源文件列表
set(SOURCES
    src/main.cpp
    src/spfresh_index.cpp
)

# 创建可执行文件
add_executable(${PROJECT_NAME} ${SOURCES})

# 链接库
target_link_libraries(${PROJECT_NAME}
    ${OPENMP_LIBRARIES}
    ${SPDK_LIBRARIES}
    numa
    SPTAGLibStatic
    tbb
    DistanceUtils
    ssdservingLib
    zstd
    pthread
    rocksdb
    VectorTest
)
