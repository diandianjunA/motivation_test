# Jasper - GPU-Accelerated Approximate Nearest Neighbor Library

Jasper 是一个可嵌入的 GPU 加速近似最近邻 (ANN) 索引库，基于 Vamana 图算法，支持在 NVIDIA GPU 上进行高性能的向量索引构建和搜索。

## 功能

- `build` - 从原始向量构建 ANN 索引
- `search` - 批量 k-NN 查询
- `insert` - 向已有索引中增量插入新向量
- `save` / `load` - 索引的序列化与反序列化

## 快速开始

```cpp
#include <jasper/jasper.cuh>

// 创建 128 维 uint8 向量索引
jasper::JasperIndex<128> index;

// 构建索引
jasper::BuildParams params;
params.n_rounds = 1;
params.alpha = 1.2;
index.build(vectors, n_vectors, params);

// 搜索 top-10 最近邻
std::vector<uint32_t> ids(n_queries * 10);
std::vector<float> dists(n_queries * 10);
index.search(queries, n_queries, 10, ids.data(), dists.data());

// 保存 / 加载
index.save("my_index");

jasper::JasperIndex<128> loaded;
loaded.load("my_index", n_vectors);
```

## 模板参数

```cpp
template <uint32_t VECTOR_DIM,        // 向量维度 (如 128)
          typename DATA_T = uint8_t,   // 向量元素类型
          uint32_t R = 64,             // 图最大出度
          uint32_t L_CAP = 64,         // Beam search 容量
          bool ON_HOST = false>        // 图存储位置 (false=GPU)
class JasperIndex;
```

### 预定义类型别名

| 别名 | 配置 |
|------|------|
| `JasperIndex128` | 128维 uint8, R=64, GPU |
| `JasperIndex96` | 96维 uint8, R=64, GPU |
| `JasperIndex256` | 256维 uint8, R=64, GPU |
| `JasperIndexFloat128` | 128维 float, R=64, GPU |

## API 参考

### BuildParams

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `n_rounds` | 1 | 图构建轮数 |
| `nodes_explored_per_iteration` | 4 | 每轮探索节点数 |
| `random_init` | false | 是否随机初始化图 |
| `alpha` | 1.2 | 剪枝参数 (越大图越稠密) |
| `max_batch_ratio` | 0.02 | 批处理大小比例 |

### SearchParams

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `beam_width` | 64 | Beam 宽度 (越大越精确但越慢) |
| `cut` | 10.0 | 距离截断系数 |
| `limit` | 512 | 搜索节点上限 |

### 方法

```cpp
void build(const DATA_T *vectors, uint64_t n_vectors,
           const BuildParams &params = {});

void search(const DATA_T *queries, uint64_t n_queries, uint32_t k,
            uint32_t *out_ids, float *out_distances = nullptr,
            const SearchParams &params = {}) const;

void insert(const DATA_T *new_vectors, uint64_t n_new,
            const BuildParams &params = {});

void save(const std::string &path) const;
void load(const std::string &path, uint64_t n_vectors);

uint64_t size() const;
bool is_built() const;
```

## 依赖

### 需要手动安装的依赖

| 依赖 | 最低版本 | 说明 | 安装方式 |
|------|----------|------|----------|
| CMake | 3.18 | 构建系统 | `apt install cmake` 或从 [cmake.org](https://cmake.org) 下载 |
| CUDA Toolkit | 11.8 | GPU 编译器和运行时，包含 Thrust、CUB | 从 [NVIDIA CUDA Downloads](https://developer.nvidia.com/cuda-downloads) 安装 |
| OpenSSL | — | 随机数生成 (`RAND_bytes`) | `apt install libssl-dev` (Ubuntu/Debian) |
| GCC / G++ | C++17 | 需要支持 C++17 的编译器 | `apt install build-essential` |

### 自动获取的依赖 (通过 CPM，构建时自动下载)

| 依赖 | 版本 | 说明 |
|------|------|------|
| Gallatin | `inlined_commands` 分支 | GPU 内存分配器 |
| Eigen3 | 3.4.0 | 线性代数库 (RaBitQ 量化器使用) |

### GPU 架构参考

| GPU | 架构代号 | SM 值 |
|-----|----------|-------|
| V100 | Volta | 70 |
| T4 | Turing | 75 |
| A100 | Ampere | 80 |
| A10/A30 | Ampere | 86 |
| H100 | Hopper | 90 |
| L40S | Ada Lovelace | 89 |

如果不指定 `GPU_ARCHS`，CMake 会自动检测当前机器上的 GPU 架构。

## 构建

### 独立构建

```bash
cd Jasper
mkdir -p build
cmake -S . -B build -DGPU_ARCHS=80   # 按你的 GPU 架构设置，或省略让 CMake 自动检测
cmake --build build -j
./build/bin/jasper_example
```

### CMake 选项

| 选项 | 默认值 | 说明 |
|------|--------|------|
| `GPU_ARCHS` | 自动检测 | GPU SM 架构 (如 `80`) |
| `JASPER_BUILD_EXAMPLES` | ON | 是否构建示例程序 |
| `CMAKE_BUILD_TYPE` | Release | 构建类型 (`Debug` 会启用 CUDA 调试标志 `-G -g`) |

### 在你的项目中使用 (add_subdirectory)

将 `Jasper/` 目录整体复制到你的项目中，然后：

```cmake
add_subdirectory(path/to/Jasper)
target_link_libraries(your_target PRIVATE jasper)
set_target_properties(your_target PROPERTIES
  CUDA_ARCHITECTURES ${GPU_ARCHS}
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_RESOLVE_DEVICE_SYMBOLS ON
)
```

## 数据格式

向量以行优先 (row-major) 的连续内存布局传入，每个向量占 `VECTOR_DIM * sizeof(DATA_T)` 字节。

索引文件格式 (`.index`):
```
Header (32 bytes):
  uint64_t total_file_size
  uint64_t n_vertices
  uint64_t medoid
  uint64_t bytes_per_node

Per-vertex (repeated n_vertices times):
  DATA_T[VECTOR_DIM]           向量数据
  uint8_t                      邻居数量
  uint32_t[R]                  邻居列表
```
