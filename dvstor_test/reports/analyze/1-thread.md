# Breakdown 分析报告

## 实验元信息

- **client_threads**: 1
- **coroutines**: 4
- **dim**: 1,024
- **measure_mixed**
  - **completed_reads**: 346
  - **completed_writes**: 346
  - **issued_reads**: 346
  - **issued_writes**: 346
- **measure_ops**: 1,000
- **measure_seconds**: 60
- **operation_granularity**: single_vector
- **read_ratio**: 0.5
- **run_mode**: time
- **search_mode**: rabitq_gpu
- **synthetic_query_vectors**: 4,096
- **threads**: 4
- **time_completion_policy**: drain
- **time_issue_policy**: bounded_by_observed_call_latency
- **warmup_mixed**
  - **completed_reads**: 397
  - **completed_writes**: 396
  - **issued_reads**: 397
  - **issued_writes**: 396
- **warmup_ops**: 100
- **warmup_seconds**: 30
- **workload**: mixed

## Bottleneck Summary

### insert

```text
insert breakdown
  count: 346
  latency_ms: mean=163.596 p50=190.252 p95=214.179 p99=222.652
  top_categories:
    gpu_ns: 50478.3 ms (89.1793%)
    rdma_ns: 2969.23 ms (5.24571%)
    cpu_ns: 1671.33 ms (2.95272%)
    transfer_ns: 1484.27 ms (2.62225%)
```

### query

```text
query breakdown
  count: 346
  latency_ms: mean=9.80485 p50=9.59815 p95=15.4064 p99=17.6685
  top_categories:
    gpu_ns: 1699.59 ms (50.1174%)
    rdma_ns: 944.5 ms (27.8513%)
    transfer_ns: 414.36 ms (12.2186%)
    cpu_ns: 332.772 ms (9.81272%)
```

## INSERT 分析

- 操作数：**346**
- 平均端到端延迟：**163.596 ms**
- P50 端到端延迟：**190.252 ms**
- P95 端到端延迟：**214.179 ms**
- P99 端到端延迟：**222.652 ms**

### 一级 Breakdown 占比

| 部分 | 时间 | 占比 |
|---|---|---|
| gpu_ns | 50478.257 ms | 89.18% |
| rdma_ns | 2969.232 ms | 5.25% |
| cpu_ns | 1671.333 ms | 2.95% |
| transfer_ns | 1484.274 ms | 2.62% |

- insert 一级热点：占比最高的几项是 `gpu_ns`（89.18%）、`rdma_ns`（5.25%）、`cpu_ns`（2.95%）。

### Sub Breakdown 细分占比

#### cpu_ns

| 部分 | 时间 | 占比 |
|---|---|---|
| cpu_insert_stage_candidates_ns | 985.115 ms | 58.94% |
| cpu_insert_filter_ns | 252.752 ms | 15.12% |
| cpu_insert_runtime_overhead_ns | 123.419 ms | 7.38% |
| cpu_insert_beam_update_ns | 97.347 ms | 5.82% |
| cpu_insert_select_ns | 77.625 ms | 4.64% |
| cpu_insert_candidate_sort_ns | 58.971 ms | 3.53% |
| cpu_insert_finalize_ns | 38.425 ms | 2.30% |
| cpu_insert_overflow_prepare_ns | 19.188 ms | 1.15% |
| cpu_insert_pruned_neighbor_collect_ns | 5.586 ms | 0.33% |
| cpu_insert_neighbor_collect_ns | 4.323 ms | 0.26% |
| cpu_insert_init_ns | 3.100 ms | 0.19% |
| cpu_insert_preprune_sort_ns | 2.283 ms | 0.14% |
| cpu_insert_prune_prepare_ns | 2.123 ms | 0.13% |
| cpu_insert_candidate_collect_ns | 0.693 ms | 0.04% |
| cpu_cache_lookup_ns | 0.267 ms | 0.02% |
| cpu_insert_neighbor_prepare_ns | 0.102 ms | 0.01% |
| cpu_insert_quantize_prepare_ns | 0.014 ms | 0.00% |

- cpu_ns 内部热点：占比最高的几项是 `cpu_insert_stage_candidates_ns`（58.94%）、`cpu_insert_filter_ns`（15.12%）、`cpu_insert_runtime_overhead_ns`（7.38%）。

#### gpu_ns

| 部分 | 时间 | 占比 |
|---|---|---|
| gpu_insert_overflow_prune_ns | 41710.865 ms | 82.63% |
| gpu_insert_prune_ns | 5094.248 ms | 10.09% |
| gpu_insert_distance_ns | 2804.277 ms | 5.56% |
| gpu_insert_overflow_distance_ns | 784.044 ms | 1.55% |
| gpu_insert_quantize_ns | 84.822 ms | 0.17% |

- gpu_ns 内部热点：占比最高的几项是 `gpu_insert_overflow_prune_ns`（82.63%）、`gpu_insert_prune_ns`（10.09%）、`gpu_insert_distance_ns`（5.56%）。

#### rdma_ns

| 部分 | 时间 | 占比 |
|---|---|---|
| rdma_vector_fetch_ns | 1517.967 ms | 51.12% |
| rdma_neighbor_fetch_ns | 703.999 ms | 23.71% |
| rdma_overflow_vec_fetch_ns | 384.316 ms | 12.94% |
| rdma_neighbor_node_read_ns | 71.351 ms | 2.40% |
| rdma_neighbor_list_read_ns | 68.110 ms | 2.29% |
| rdma_pruned_neighbor_write_ns | 66.360 ms | 2.23% |
| rdma_neighbor_lock_ns | 52.247 ms | 1.76% |
| rdma_candidate_fetch_ns | 48.601 ms | 1.64% |
| rdma_neighbor_unlock_ns | 34.427 ms | 1.16% |
| rdma_neighbor_list_write_ns | 16.222 ms | 0.55% |
| rdma_new_node_write_ns | 2.204 ms | 0.07% |
| rdma_alloc_ns | 1.842 ms | 0.06% |
| rdma_medoid_ptr_ns | 1.587 ms | 0.05% |

- rdma_ns 内部热点：占比最高的几项是 `rdma_vector_fetch_ns`（51.12%）、`rdma_neighbor_fetch_ns`（23.71%）、`rdma_overflow_vec_fetch_ns`（12.94%）。

#### transfer_ns

| 部分 | 时间 | 占比 |
|---|---|---|
| transfer_distance_d2h_ns | 838.043 ms | 56.46% |
| transfer_candidate_h2d_ns | 212.951 ms | 14.35% |
| transfer_overflow_prune_d2h_ns | 144.818 ms | 9.76% |
| transfer_overflow_dist_d2h_ns | 118.461 ms | 7.98% |
| transfer_overflow_prune_inputs_h2d_ns | 61.001 ms | 4.11% |
| transfer_overflow_query_h2d_ns | 36.485 ms | 2.46% |
| transfer_quantize_d2h_ns | 32.681 ms | 2.20% |
| transfer_overflow_candidate_h2d_ns | 31.764 ms | 2.14% |
| transfer_prune_d2h_ns | 3.927 ms | 0.26% |
| transfer_prune_h2d_ns | 2.125 ms | 0.14% |
| transfer_insert_query_h2d_ns | 2.019 ms | 0.14% |

- transfer_ns 内部热点：占比最高的几项是 `transfer_distance_d2h_ns`（56.46%）、`transfer_candidate_h2d_ns`（14.35%）、`transfer_overflow_prune_d2h_ns`（9.76%）。

### Latency

| 延迟字段 | 值 |
|---|---|
| end_to_end_ns | 56604.372 ms |
| mean_end_to_end_ns | 163.596 ms |
| mean_queue_wait_ns | 0.004 ms |
| mean_service_ns | 163.593 ms |
| p50_end_to_end_ns | 190.252 ms |
| p50_service_ns | 190.249 ms |
| p95_end_to_end_ns | 214.179 ms |
| p95_service_ns | 214.175 ms |
| p99_end_to_end_ns | 222.652 ms |
| p99_service_ns | 222.647 ms |
| queue_wait_ns | 1.276 ms |
| service_ns | 56603.096 ms |

### Counters

| 字段 | 值 |
|---|---|
| rdma_read_bytes | 14,400,881,695 |
| h2d_bytes | 14,358,027,320 |
| vector_rdma_bytes | 14,302,531,184 |
| neighbor_rdma_bytes | 98,347,743 |
| d2h_bytes | 17,494,412 |
| rdma_write_bytes | 11,163,891 |
| l2_kernels | 119,533 |
| lock_attempts | 18,151 |
| prune_kernels | 15,057 |
| overflow_prunes | 14,711 |
| cache_hits | 346 |
| remote_allocations | 346 |
| cache_misses | 0 |
| cas_failures | 0 |
| exact_reranks | 0 |
| lock_retries | 0 |
| rabitq_kernels | 0 |
| rabitq_rdma_bytes | 0 |
| visited_neighborlists | 0 |
| visited_nodes | 0 |

## QUERY 分析

- 操作数：**346**
- 平均端到端延迟：**9.805 ms**
- P50 端到端延迟：**9.598 ms**
- P95 端到端延迟：**15.406 ms**
- P99 端到端延迟：**17.669 ms**

### 一级 Breakdown 占比

| 部分 | 时间 | 占比 |
|---|---|---|
| gpu_ns | 1699.594 ms | 50.12% |
| rdma_ns | 944.500 ms | 27.85% |
| transfer_ns | 414.360 ms | 12.22% |
| cpu_ns | 332.772 ms | 9.81% |

- query 一级热点：占比最高的几项是 `gpu_ns`（50.12%）、`rdma_ns`（27.85%）、`transfer_ns`（12.22%）。

### Sub Breakdown 细分占比

#### cpu_ns

| 部分 | 时间 | 占比 |
|---|---|---|
| cpu_query_filter_ns | 115.892 ms | 34.83% |
| cpu_query_stage_candidates_ns | 113.449 ms | 34.09% |
| cpu_query_beam_update_ns | 31.517 ms | 9.47% |
| cpu_query_runtime_overhead_ns | 28.686 ms | 8.62% |
| cpu_query_rerank_prepare_ns | 15.405 ms | 4.63% |
| cpu_query_select_ns | 10.289 ms | 3.09% |
| cpu_query_finalize_ns | 5.752 ms | 1.73% |
| cpu_query_result_ids_ns | 5.677 ms | 1.71% |
| cpu_cache_lookup_ns | 4.576 ms | 1.38% |
| cpu_query_beam_sort_ns | 1.349 ms | 0.41% |
| cpu_query_rerank_collect_ns | 0.138 ms | 0.04% |
| cpu_query_rerank_update_ns | 0.041 ms | 0.01% |

- cpu_ns 内部热点：占比最高的几项是 `cpu_query_filter_ns`（34.83%）、`cpu_query_stage_candidates_ns`（34.09%）、`cpu_query_beam_update_ns`（9.47%）。

#### gpu_ns

| 部分 | 时间 | 占比 |
|---|---|---|
| gpu_query_distance_ns | 1668.665 ms | 98.18% |
| gpu_query_rerank_ns | 15.786 ms | 0.93% |
| gpu_query_prepare_ns | 15.144 ms | 0.89% |

- gpu_ns 内部热点：占比最高的几项是 `gpu_query_distance_ns`（98.18%）、`gpu_query_rerank_ns`（0.93%）、`gpu_query_prepare_ns`（0.89%）。

#### rdma_ns

| 部分 | 时间 | 占比 |
|---|---|---|
| rdma_rabitq_fetch_ns | 713.449 ms | 75.54% |
| rdma_neighbor_fetch_ns | 215.502 ms | 22.82% |
| rdma_rerank_fetch_ns | 13.703 ms | 1.45% |
| rdma_medoid_ptr_ns | 1.846 ms | 0.20% |

- rdma_ns 内部热点：占比最高的几项是 `rdma_rabitq_fetch_ns`（75.54%）、`rdma_neighbor_fetch_ns`（22.82%）、`rdma_rerank_fetch_ns`（1.45%）。

#### transfer_ns

| 部分 | 时间 | 占比 |
|---|---|---|
| transfer_distance_d2h_ns | 315.117 ms | 76.05% |
| transfer_rabitq_h2d_ns | 93.677 ms | 22.61% |
| transfer_rerank_d2h_ns | 2.716 ms | 0.66% |
| transfer_query_h2d_ns | 1.991 ms | 0.48% |
| transfer_rerank_h2d_ns | 0.859 ms | 0.21% |

- transfer_ns 内部热点：占比最高的几项是 `transfer_distance_d2h_ns`（76.05%）、`transfer_rabitq_h2d_ns`（22.61%）、`transfer_rerank_d2h_ns`（0.66%）。

### Latency

| 延迟字段 | 值 |
|---|---|
| end_to_end_ns | 3392.480 ms |
| mean_end_to_end_ns | 9.805 ms |
| mean_queue_wait_ns | 0.004 ms |
| mean_service_ns | 9.801 ms |
| p50_end_to_end_ns | 9.598 ms |
| p50_service_ns | 9.594 ms |
| p95_end_to_end_ns | 15.406 ms |
| p95_service_ns | 15.403 ms |
| p99_end_to_end_ns | 17.669 ms |
| p99_service_ns | 17.665 ms |
| queue_wait_ns | 1.254 ms |
| service_ns | 3391.225 ms |

### Counters

| 字段 | 值 |
|---|---|
| rdma_read_bytes | 852,561,636 |
| h2d_bytes | 828,428,792 |
| rabitq_rdma_bytes | 690,958,840 |
| vector_rdma_bytes | 137,051,952 |
| neighbor_rdma_bytes | 24,548,076 |
| d2h_bytes | 5,447,932 |
| visited_neighborlists | 47,852 |
| rabitq_kernels | 41,444 |
| cache_hits | 3,563 |
| exact_reranks | 346 |
| cache_misses | 243 |
| cas_failures | 0 |
| l2_kernels | 0 |
| lock_attempts | 0 |
| lock_retries | 0 |
| overflow_prunes | 0 |
| prune_kernels | 0 |
| rdma_write_bytes | 0 |
| remote_allocations | 0 |
| visited_nodes | 0 |

## System Counters

| 字段 | 值 |
|---|---|
| rdma_read_bytes | 15,253,443,331 |
| h2d_bytes | 15,186,456,112 |
| d2h_bytes | 22,942,344 |
| rdma_write_bytes | 11,163,891 |

## Insert / Query 对比

| 类别 | Insert 占比 | Query 占比 |
|---|---|---|
| cpu_ns | 2.95% | 9.81% |
| gpu_ns | 89.18% | 50.12% |
| rdma_ns | 5.25% | 27.85% |
| transfer_ns | 2.62% | 12.22% |

- Insert 最大部分是 **gpu_ns**，占 **89.18%**。
- Query 最大部分是 **gpu_ns**，占 **50.12%**。
- Insert 更偏向 GPU 计算密集。
- Query 更偏向 RDMA / 远端访问受限。
