# Breakdown 分析报告

## 实验元信息

- **client\_threads**: 16
- **coroutines**: 4
- **dim**: 1,024
- **measure\_mixed**
  - **completed\_reads**: 1,472
  - **completed\_writes**: 317
  - **issued\_reads**: 1,472
  - **issued\_writes**: 317
- **measure\_ops**: 1,000
- **measure\_seconds**: 60
- **operation\_granularity**: single\_vector
- **read\_ratio**: 0.5
- **run\_mode**: time
- **search\_mode**: rabitq\_gpu
- **synthetic\_query\_vectors**: 65,536
- **threads**: 4
- **time\_completion\_policy**: drain
- **time\_issue\_policy**: bounded\_by\_observed\_call\_latency
- **warmup\_mixed**
  - **completed\_reads**: 981
  - **completed\_writes**: 247
  - **issued\_reads**: 981
  - **issued\_writes**: 247
- **warmup\_ops**: 100
- **warmup\_seconds**: 30
- **workload**: mixed

## Bottleneck Summary

### insert

```text
insert breakdown
  count: 317
  latency_ms: mean=1493.58 p50=1430.83 p95=2390.7 p99=2796.41
  top_categories:
    gpu_ns: 450676 ms (95.1879%)
    rdma_ns: 18244 ms (3.85334%)
    transfer_ns: 3372.96 ms (0.712408%)
    cpu_ns: 1166.21 ms (0.246318%)
```

### query

```text
query breakdown
  count: 1472
  latency_ms: mean=324.627 p50=308.236 p95=584.09 p99=728.269
  top_categories:
    gpu_ns: 452931 ms (94.7934%)
    rdma_ns: 20578.8 ms (4.30691%)
    transfer_ns: 2973.62 ms (0.622345%)
    cpu_ns: 1325.15 ms (0.277339%)
```

## INSERT 分析

- 操作数：**317**
- 平均端到端延迟：**1493.579 ms**
- P50 端到端延迟：**1430.832 ms**
- P95 端到端延迟：**2390.704 ms**
- P99 端到端延迟：**2796.405 ms**

### 一级 Breakdown 占比

| 部分           | 时间            | 占比     |
| ------------ | ------------- | ------ |
| gpu\_ns      | 450675.809 ms | 95.19% |
| rdma\_ns     | 18243.971 ms  | 3.85%  |
| transfer\_ns | 3372.960 ms   | 0.71%  |
| cpu\_ns      | 1166.213 ms   | 0.25%  |

- insert 一级热点：占比最高的几项是 `gpu_ns`（95.19%）、`rdma_ns`（3.85%）、`transfer_ns`（0.71%）。

### Sub Breakdown 细分占比

#### cpu\_ns

| 部分                                         | 时间         | 占比     |
| ------------------------------------------ | ---------- | ------ |
| cpu\_insert\_filter\_ns                    | 460.863 ms | 39.52% |
| cpu\_insert\_runtime\_overhead\_ns         | 219.745 ms | 18.84% |
| cpu\_insert\_stage\_candidates\_ns         | 168.985 ms | 14.49% |
| cpu\_insert\_beam\_update\_ns              | 117.807 ms | 10.10% |
| cpu\_insert\_select\_ns                    | 87.441 ms  | 7.50%  |
| cpu\_insert\_finalize\_ns                  | 81.032 ms  | 6.95%  |
| cpu\_insert\_overflow\_prepare\_ns         | 14.575 ms  | 1.25%  |
| cpu\_insert\_pruned\_neighbor\_collect\_ns | 3.680 ms   | 0.32%  |
| cpu\_insert\_neighbor\_collect\_ns         | 2.956 ms   | 0.25%  |
| cpu\_insert\_init\_ns                      | 2.809 ms   | 0.24%  |
| cpu\_insert\_prune\_prepare\_ns            | 2.438 ms   | 0.21%  |
| cpu\_insert\_preprune\_sort\_ns            | 2.266 ms   | 0.19%  |
| cpu\_insert\_candidate\_collect\_ns        | 1.035 ms   | 0.09%  |
| cpu\_cache\_lookup\_ns                     | 0.263 ms   | 0.02%  |
| cpu\_insert\_neighbor\_prepare\_ns         | 0.176 ms   | 0.02%  |
| cpu\_insert\_candidate\_sort\_ns           | 0.131 ms   | 0.01%  |
| cpu\_insert\_quantize\_prepare\_ns         | 0.009 ms   | 0.00%  |

- cpu\_ns 内部热点：占比最高的几项是 `cpu_insert_filter_ns`（39.52%）、`cpu_insert_runtime_overhead_ns`（18.84%）、`cpu_insert_stage_candidates_ns`（14.49%）。

#### gpu\_ns

| 部分                                  | 时间            | 占比     |
| ----------------------------------- | ------------- | ------ |
| gpu\_insert\_distance\_ns           | 332076.138 ms | 73.68% |
| gpu\_insert\_overflow\_prune\_ns    | 74343.698 ms  | 16.50% |
| gpu\_insert\_overflow\_distance\_ns | 35651.208 ms  | 7.91%  |
| gpu\_insert\_prune\_ns              | 7857.918 ms   | 1.74%  |
| gpu\_insert\_quantize\_ns           | 746.846 ms    | 0.17%  |

- gpu\_ns 内部热点：占比最高的几项是 `gpu_insert_distance_ns`（73.68%）、`gpu_insert_overflow_prune_ns`（16.50%）、`gpu_insert_overflow_distance_ns`（7.91%）。

#### rdma\_ns

| 部分                                | 时间          | 占比     |
| --------------------------------- | ----------- | ------ |
| rdma\_neighbor\_fetch\_ns         | 7386.562 ms | 40.49% |
| rdma\_vector\_fetch\_ns           | 7320.666 ms | 40.13% |
| rdma\_neighbor\_lock\_ns          | 1479.097 ms | 8.11%  |
| rdma\_overflow\_vec\_fetch\_ns    | 411.065 ms  | 2.25%  |
| rdma\_neighbor\_list\_read\_ns    | 392.065 ms  | 2.15%  |
| rdma\_neighbor\_node\_read\_ns    | 370.731 ms  | 2.03%  |
| rdma\_neighbor\_unlock\_ns        | 369.342 ms  | 2.02%  |
| rdma\_pruned\_neighbor\_write\_ns | 225.865 ms  | 1.24%  |
| rdma\_neighbor\_list\_write\_ns   | 179.739 ms  | 0.99%  |
| rdma\_candidate\_fetch\_ns        | 73.625 ms   | 0.40%  |
| rdma\_new\_node\_write\_ns        | 13.329 ms   | 0.07%  |
| rdma\_alloc\_ns                   | 12.718 ms   | 0.07%  |
| rdma\_medoid\_ptr\_ns             | 9.168 ms    | 0.05%  |

- rdma\_ns 内部热点：占比最高的几项是 `rdma_neighbor_fetch_ns`（40.49%）、`rdma_vector_fetch_ns`（40.13%）、`rdma_neighbor_lock_ns`（8.11%）。

#### transfer\_ns

| 部分                                         | 时间          | 占比     |
| ------------------------------------------ | ----------- | ------ |
| transfer\_distance\_d2h\_ns                | 2787.440 ms | 82.64% |
| transfer\_overflow\_prune\_d2h\_ns         | 165.619 ms  | 4.91%  |
| transfer\_overflow\_dist\_d2h\_ns          | 155.185 ms  | 4.60%  |
| transfer\_overflow\_prune\_inputs\_h2d\_ns | 83.396 ms   | 2.47%  |
| transfer\_quantize\_d2h\_ns                | 62.383 ms   | 1.85%  |
| transfer\_overflow\_query\_h2d\_ns         | 54.406 ms   | 1.61%  |
| transfer\_overflow\_candidate\_h2d\_ns     | 51.676 ms   | 1.53%  |
| transfer\_prune\_d2h\_ns                   | 7.820 ms    | 0.23%  |
| transfer\_insert\_query\_h2d\_ns           | 2.597 ms    | 0.08%  |
| transfer\_prune\_h2d\_ns                   | 2.438 ms    | 0.07%  |

- transfer\_ns 内部热点：占比最高的几项是 `transfer_distance_d2h_ns`（82.64%）、`transfer_overflow_prune_d2h_ns`（4.91%）、`transfer_overflow_dist_d2h_ns`（4.60%）。

### Latency

| 延迟字段                   | 值             |
| ---------------------- | ------------- |
| end\_to\_end\_ns       | 473464.634 ms |
| mean\_end\_to\_end\_ns | 1493.579 ms   |
| mean\_queue\_wait\_ns  | 0.018 ms      |
| mean\_service\_ns      | 1493.561 ms   |
| p50\_end\_to\_end\_ns  | 1430.832 ms   |
| p50\_service\_ns       | 1430.812 ms   |
| p95\_end\_to\_end\_ns  | 2390.704 ms   |
| p95\_service\_ns       | 2390.681 ms   |
| p99\_end\_to\_end\_ns  | 2796.405 ms   |
| p99\_service\_ns       | 2796.398 ms   |
| queue\_wait\_ns        | 5.681 ms      |
| service\_ns            | 473458.953 ms |

### Counters

| 字段                     | 值              |
| ---------------------- | -------------- |
| rdma\_read\_bytes      | 67,103,098,384 |
| vector\_rdma\_bytes    | 66,738,349,440 |
| h2d\_bytes             | 8,106,582,864  |
| neighbor\_rdma\_bytes  | 364,738,896    |
| d2h\_bytes             | 71,825,076     |
| rdma\_write\_bytes     | 33,951,559     |
| l2\_kernels            | 577,126        |
| lock\_attempts         | 56,374         |
| cas\_failures          | 42,998         |
| lock\_retries          | 42,998         |
| prune\_kernels         | 31,180         |
| overflow\_prunes       | 29,902         |
| remote\_allocations    | 1,266          |
| cache\_hits            | 1,256          |
| cache\_misses          | 0              |
| exact\_reranks         | 0              |
| rabitq\_kernels        | 0              |
| rabitq\_rdma\_bytes    | 0              |
| visited\_neighborlists | 0              |
| visited\_nodes         | 0              |

## QUERY 分析

- 操作数：**1,472**
- 平均端到端延迟：**324.627 ms**
- P50 端到端延迟：**308.236 ms**
- P95 端到端延迟：**584.090 ms**
- P99 端到端延迟：**728.269 ms**

### 一级 Breakdown 占比

| 部分           | 时间            | 占比     |
| ------------ | ------------- | ------ |
| gpu\_ns      | 452930.815 ms | 94.79% |
| rdma\_ns     | 20578.781 ms  | 4.31%  |
| transfer\_ns | 2973.617 ms   | 0.62%  |
| cpu\_ns      | 1325.147 ms   | 0.28%  |

- query 一级热点：占比最高的几项是 `gpu_ns`（94.79%）、`rdma_ns`（4.31%）、`transfer_ns`（0.62%）。

### Sub Breakdown 细分占比

#### cpu\_ns

| 部分                                | 时间         | 占比     |
| --------------------------------- | ---------- | ------ |
| cpu\_query\_filter\_ns            | 607.074 ms | 45.81% |
| cpu\_query\_runtime\_overhead\_ns | 234.879 ms | 17.72% |
| cpu\_query\_beam\_update\_ns      | 137.975 ms | 10.41% |
| cpu\_query\_finalize\_ns          | 100.166 ms | 7.56%  |
| cpu\_query\_result\_ids\_ns       | 99.823 ms  | 7.53%  |
| cpu\_cache\_lookup\_ns            | 95.283 ms  | 7.19%  |
| cpu\_query\_select\_ns            | 42.996 ms  | 3.24%  |
| cpu\_query\_beam\_sort\_ns        | 5.924 ms   | 0.45%  |
| cpu\_query\_rerank\_collect\_ns   | 0.799 ms   | 0.06%  |
| cpu\_query\_rerank\_update\_ns    | 0.228 ms   | 0.02%  |

- cpu\_ns 内部热点：占比最高的几项是 `cpu_query_filter_ns`（45.81%）、`cpu_query_runtime_overhead_ns`（17.72%）、`cpu_query_beam_update_ns`（10.41%）。

#### gpu\_ns

| 部分                       | 时间            | 占比     |
| ------------------------ | ------------- | ------ |
| gpu\_query\_distance\_ns | 445874.949 ms | 98.44% |
| gpu\_query\_rerank\_ns   | 3614.495 ms   | 0.80%  |
| gpu\_query\_prepare\_ns  | 3441.371 ms   | 0.76%  |

- gpu\_ns 内部热点：占比最高的几项是 `gpu_query_distance_ns`（98.44%）、`gpu_query_rerank_ns`（0.80%）、`gpu_query_prepare_ns`（0.76%）。

#### rdma\_ns

| 部分                        | 时间           | 占比     |
| ------------------------- | ------------ | ------ |
| rdma\_rabitq\_fetch\_ns   | 10452.067 ms | 50.79% |
| rdma\_neighbor\_fetch\_ns | 9950.517 ms  | 48.35% |
| rdma\_rerank\_fetch\_ns   | 133.311 ms   | 0.65%  |
| rdma\_medoid\_ptr\_ns     | 42.886 ms    | 0.21%  |

- rdma\_ns 内部热点：占比最高的几项是 `rdma_rabitq_fetch_ns`（50.79%）、`rdma_neighbor_fetch_ns`（48.35%）、`rdma_rerank_fetch_ns`（0.65%）。

#### transfer\_ns

| 部分                          | 时间          | 占比     |
| --------------------------- | ----------- | ------ |
| transfer\_distance\_d2h\_ns | 2930.656 ms | 98.56% |
| transfer\_rerank\_d2h\_ns   | 30.883 ms   | 1.04%  |
| transfer\_query\_h2d\_ns    | 12.078 ms   | 0.41%  |

- transfer\_ns 内部热点：占比最高的几项是 `transfer_distance_d2h_ns`（98.56%）、`transfer_rerank_d2h_ns`（1.04%）、`transfer_query_h2d_ns`（0.41%）。

### Latency

| 延迟字段                   | 值             |
| ---------------------- | ------------- |
| end\_to\_end\_ns       | 477850.902 ms |
| mean\_end\_to\_end\_ns | 324.627 ms    |
| mean\_queue\_wait\_ns  | 0.029 ms      |
| mean\_service\_ns      | 324.598 ms    |
| p50\_end\_to\_end\_ns  | 308.236 ms    |
| p50\_service\_ns       | 308.189 ms    |
| p95\_end\_to\_end\_ns  | 584.090 ms    |
| p95\_service\_ns       | 584.059 ms    |
| p99\_end\_to\_end\_ns  | 728.269 ms    |
| p99\_service\_ns       | 728.226 ms    |
| queue\_wait\_ns        | 42.542 ms     |
| service\_ns            | 477808.360 ms |

### Counters

| 字段                     | 值              |
| ---------------------- | -------------- |
| rdma\_read\_bytes      | 15,007,849,386 |
| rabitq\_rdma\_bytes    | 12,253,947,680 |
| vector\_rdma\_bytes    | 2,341,061,088  |
| neighbor\_rdma\_bytes  | 412,793,658    |
| d2h\_bytes             | 96,874,940     |
| h2d\_bytes             | 24,072,192     |
| visited\_neighborlists | 802,163        |
| rabitq\_kernels        | 704,637        |
| cache\_hits            | 56,911         |
| cache\_misses          | 7,708          |
| exact\_reranks         | 5,858          |
| cas\_failures          | 0              |
| l2\_kernels            | 0              |
| lock\_attempts         | 0              |
| lock\_retries          | 0              |
| overflow\_prunes       | 0              |
| prune\_kernels         | 0              |
| rdma\_write\_bytes     | 0              |
| remote\_allocations    | 0              |
| visited\_nodes         | 0              |

## System Counters

| 字段                 | 值              |
| ------------------ | -------------- |
| rdma\_read\_bytes  | 82,110,947,770 |
| h2d\_bytes         | 8,130,655,056  |
| d2h\_bytes         | 168,700,016    |
| rdma\_write\_bytes | 33,951,559     |

## Insert / Query 对比

| 类别           | Insert 占比 | Query 占比 |
| ------------ | --------- | -------- |
| cpu\_ns      | 0.25%     | 0.28%    |
| gpu\_ns      | 95.19%    | 94.79%   |
| rdma\_ns     | 3.85%     | 4.31%    |
| transfer\_ns | 0.71%     | 0.62%    |

- Insert 最大部分是 **gpu\_ns**，占 **95.19%**。
- Query 最大部分是 **gpu\_ns**，占 **94.79%**。
- Insert 更偏向 GPU 计算密集。
- Query 更偏向 RDMA / 远端访问受限。

