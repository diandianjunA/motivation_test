# Breakdown 分析报告

## 实验元信息

- **client_threads**: 32
- **coroutines**: 4
- **dim**: 1,024
- **measure_mixed**
  - **completed_reads**: 10,093
  - **completed_writes**: 1,620
  - **issued_reads**: 10,093
  - **issued_writes**: 1,620
- **measure_ops**: 1,000
- **measure_seconds**: 60
- **operation_granularity**: single_vector
- **read_ratio**: 0.5
- **run_mode**: time
- **search_mode**: rabitq_gpu
- **synthetic_query_vectors**: 131,072
- **threads**: 4
- **time_completion_policy**: drain
- **time_issue_policy**: bounded_by_observed_call_latency
- **warmup_mixed**
  - **completed_reads**: 4,271
  - **completed_writes**: 819
  - **issued_reads**: 4,271
  - **issued_writes**: 819
- **warmup_ops**: 100
- **warmup_seconds**: 30
- **workload**: mixed

## Bottleneck Summary

### insert

```text
insert breakdown
  count: 1620
  latency_ms: mean=591.5 p50=344.947 p95=477.326 p99=752.707
  top_categories:
    gpu_ns: 336426 ms (69.9563%)
    rdma_ns: 104964 ms (21.8262%)
    cpu_ns: 24548.2 ms (5.10456%)
    transfer_ns: 14970.5 ms (3.11297%)
```

### query

```text
query breakdown
  count: 10093
  latency_ms: mean=94.8908 p50=50.6614 p95=93.6793 p99=235.738
  top_categories:
    rdma_ns: 283216 ms (59.0203%)
    gpu_ns: 137768 ms (28.7099%)
    cpu_ns: 33625.7 ms (7.00736%)
    transfer_ns: 25252.4 ms (5.26243%)
```

## INSERT 分析

- 操作数：**1,620**
- 平均端到端延迟：**591.500 ms**
- P50 端到端延迟：**344.947 ms**
- P95 端到端延迟：**477.326 ms**
- P99 端到端延迟：**752.707 ms**

### 一级 Breakdown 占比

| 部分 | 时间 | 占比 |
|---|---|---|
| gpu_ns | 336425.682 ms | 69.96% |
| rdma_ns | 104963.760 ms | 21.83% |
| cpu_ns | 24548.225 ms | 5.10% |
| transfer_ns | 14970.521 ms | 3.11% |

- insert 一级热点：占比最高的几项是 `gpu_ns`（69.96%）、`rdma_ns`（21.83%）、`cpu_ns`（5.10%）。

### Sub Breakdown 细分占比

#### cpu_ns

| 部分 | 时间 | 占比 |
|---|---|---|
| cpu_insert_stage_candidates_ns | 16611.956 ms | 67.67% |
| cpu_insert_runtime_overhead_ns | 4045.423 ms | 16.48% |
| cpu_insert_filter_ns | 1523.907 ms | 6.21% |
| cpu_insert_candidate_sort_ns | 1059.042 ms | 4.31% |
| cpu_insert_beam_update_ns | 489.489 ms | 1.99% |
| cpu_insert_select_ns | 355.651 ms | 1.45% |
| cpu_insert_finalize_ns | 203.945 ms | 0.83% |
| cpu_insert_overflow_prepare_ns | 132.942 ms | 0.54% |
| cpu_insert_pruned_neighbor_collect_ns | 39.902 ms | 0.16% |
| cpu_insert_neighbor_collect_ns | 35.348 ms | 0.14% |
| cpu_insert_prune_prepare_ns | 18.696 ms | 0.08% |
| cpu_insert_init_ns | 14.714 ms | 0.06% |
| cpu_insert_preprune_sort_ns | 11.181 ms | 0.05% |
| cpu_insert_candidate_collect_ns | 4.594 ms | 0.02% |
| cpu_cache_lookup_ns | 1.109 ms | 0.00% |
| cpu_insert_neighbor_prepare_ns | 0.234 ms | 0.00% |
| cpu_insert_quantize_prepare_ns | 0.090 ms | 0.00% |

- cpu_ns 内部热点：占比最高的几项是 `cpu_insert_stage_candidates_ns`（67.67%）、`cpu_insert_runtime_overhead_ns`（16.48%）、`cpu_insert_filter_ns`（6.21%）。

#### gpu_ns

| 部分 | 时间 | 占比 |
|---|---|---|
| gpu_insert_overflow_prune_ns | 266854.857 ms | 79.32% |
| gpu_insert_distance_ns | 36243.906 ms | 10.77% |
| gpu_insert_prune_ns | 25374.199 ms | 7.54% |
| gpu_insert_overflow_distance_ns | 7445.151 ms | 2.21% |
| gpu_insert_quantize_ns | 507.569 ms | 0.15% |

- gpu_ns 内部热点：占比最高的几项是 `gpu_insert_overflow_prune_ns`（79.32%）、`gpu_insert_distance_ns`（10.77%）、`gpu_insert_prune_ns`（7.54%）。

#### rdma_ns

| 部分 | 时间 | 占比 |
|---|---|---|
| rdma_vector_fetch_ns | 40232.223 ms | 38.33% |
| rdma_neighbor_fetch_ns | 36226.161 ms | 34.51% |
| rdma_overflow_vec_fetch_ns | 11110.394 ms | 10.58% |
| rdma_pruned_neighbor_write_ns | 3945.791 ms | 3.76% |
| rdma_neighbor_lock_ns | 3756.007 ms | 3.58% |
| rdma_neighbor_node_read_ns | 2957.142 ms | 2.82% |
| rdma_neighbor_list_read_ns | 2906.528 ms | 2.77% |
| rdma_neighbor_unlock_ns | 2497.953 ms | 2.38% |
| rdma_candidate_fetch_ns | 966.010 ms | 0.92% |
| rdma_neighbor_list_write_ns | 169.737 ms | 0.16% |
| rdma_alloc_ns | 72.677 ms | 0.07% |
| rdma_medoid_ptr_ns | 68.413 ms | 0.07% |
| rdma_new_node_write_ns | 54.724 ms | 0.05% |

- rdma_ns 内部热点：占比最高的几项是 `rdma_vector_fetch_ns`（38.33%）、`rdma_neighbor_fetch_ns`（34.51%）、`rdma_overflow_vec_fetch_ns`（10.58%）。

#### transfer_ns

| 部分 | 时间 | 占比 |
|---|---|---|
| transfer_distance_d2h_ns | 7564.025 ms | 50.53% |
| transfer_candidate_h2d_ns | 2736.760 ms | 18.28% |
| transfer_overflow_prune_d2h_ns | 1487.993 ms | 9.94% |
| transfer_overflow_dist_d2h_ns | 1191.945 ms | 7.96% |
| transfer_overflow_prune_inputs_h2d_ns | 787.746 ms | 5.26% |
| transfer_overflow_query_h2d_ns | 530.311 ms | 3.54% |
| transfer_overflow_candidate_h2d_ns | 446.488 ms | 2.98% |
| transfer_quantize_d2h_ns | 166.416 ms | 1.11% |
| transfer_prune_d2h_ns | 29.661 ms | 0.20% |
| transfer_prune_h2d_ns | 18.703 ms | 0.12% |
| transfer_insert_query_h2d_ns | 10.473 ms | 0.07% |

- transfer_ns 内部热点：占比最高的几项是 `transfer_distance_d2h_ns`（50.53%）、`transfer_candidate_h2d_ns`（18.28%）、`transfer_overflow_prune_d2h_ns`（9.94%）。

### Latency

| 延迟字段 | 值 |
|---|---|
| end_to_end_ns | 958230.146 ms |
| mean_end_to_end_ns | 591.500 ms |
| mean_queue_wait_ns | 294.643 ms |
| mean_service_ns | 296.857 ms |
| p50_end_to_end_ns | 344.947 ms |
| p50_service_ns | 309.347 ms |
| p95_end_to_end_ns | 477.326 ms |
| p95_service_ns | 408.941 ms |
| p99_end_to_end_ns | 752.707 ms |
| p99_service_ns | 468.923 ms |
| queue_wait_ns | 477321.958 ms |
| service_ns | 480908.188 ms |

### Counters

| 字段 | 值 |
|---|---|
| rdma_read_bytes | 267,241,201,210 |
| h2d_bytes | 266,887,375,944 |
| vector_rdma_bytes | 265,396,765,664 |
| neighbor_rdma_bytes | 1,844,383,770 |
| d2h_bytes | 340,253,544 |
| rdma_write_bytes | 214,800,197 |
| l2_kernels | 2,519,442 |
| prune_kernels | 339,054 |
| overflow_prunes | 332,577 |
| lock_attempts | 183,531 |
| cas_failures | 95,651 |
| lock_retries | 95,651 |
| remote_allocations | 6,477 |
| cache_hits | 6,476 |
| cache_misses | 0 |
| exact_reranks | 0 |
| rabitq_kernels | 0 |
| rabitq_rdma_bytes | 0 |
| visited_neighborlists | 0 |
| visited_nodes | 0 |

## QUERY 分析

- 操作数：**10,093**
- 平均端到端延迟：**94.891 ms**
- P50 端到端延迟：**50.661 ms**
- P95 端到端延迟：**93.679 ms**
- P99 端到端延迟：**235.738 ms**

### 一级 Breakdown 占比

| 部分 | 时间 | 占比 |
|---|---|---|
| rdma_ns | 283215.993 ms | 59.02% |
| gpu_ns | 137768.182 ms | 28.71% |
| cpu_ns | 33625.680 ms | 7.01% |
| transfer_ns | 25252.421 ms | 5.26% |

- query 一级热点：占比最高的几项是 `rdma_ns`（59.02%）、`gpu_ns`（28.71%）、`cpu_ns`（7.01%）。

### Sub Breakdown 细分占比

#### cpu_ns

| 部分 | 时间 | 占比 |
|---|---|---|
| cpu_query_stage_candidates_ns | 19507.131 ms | 58.01% |
| cpu_query_filter_ns | 5297.758 ms | 15.76% |
| cpu_query_runtime_overhead_ns | 3104.603 ms | 9.23% |
| cpu_query_finalize_ns | 1174.372 ms | 3.49% |
| cpu_query_result_ids_ns | 1171.815 ms | 3.48% |
| cpu_query_rerank_prepare_ns | 1142.456 ms | 3.40% |
| cpu_cache_lookup_ns | 1008.968 ms | 3.00% |
| cpu_query_beam_update_ns | 889.046 ms | 2.64% |
| cpu_query_select_ns | 282.344 ms | 0.84% |
| cpu_query_beam_sort_ns | 40.497 ms | 0.12% |
| cpu_query_rerank_collect_ns | 5.395 ms | 0.02% |
| cpu_query_rerank_update_ns | 1.295 ms | 0.00% |

- cpu_ns 内部热点：占比最高的几项是 `cpu_query_stage_candidates_ns`（58.01%）、`cpu_query_filter_ns`（15.76%）、`cpu_query_runtime_overhead_ns`（9.23%）。

#### gpu_ns

| 部分 | 时间 | 占比 |
|---|---|---|
| gpu_query_distance_ns | 135280.984 ms | 98.19% |
| gpu_query_rerank_ns | 1270.753 ms | 0.92% |
| gpu_query_prepare_ns | 1216.445 ms | 0.88% |

- gpu_ns 内部热点：占比最高的几项是 `gpu_query_distance_ns`（98.19%）、`gpu_query_rerank_ns`（0.92%）、`gpu_query_prepare_ns`（0.88%）。

#### rdma_ns

| 部分 | 时间 | 占比 |
|---|---|---|
| rdma_rabitq_fetch_ns | 158549.125 ms | 55.98% |
| rdma_neighbor_fetch_ns | 121568.229 ms | 42.92% |
| rdma_rerank_fetch_ns | 1996.115 ms | 0.70% |
| rdma_medoid_ptr_ns | 1102.524 ms | 0.39% |

- rdma_ns 内部热点：占比最高的几项是 `rdma_rabitq_fetch_ns`（55.98%）、`rdma_neighbor_fetch_ns`（42.92%）、`rdma_rerank_fetch_ns`（0.70%）。

#### transfer_ns

| 部分 | 时间 | 占比 |
|---|---|---|
| transfer_distance_d2h_ns | 17548.481 ms | 69.49% |
| transfer_rabitq_h2d_ns | 7435.848 ms | 29.45% |
| transfer_rerank_d2h_ns | 142.956 ms | 0.57% |
| transfer_query_h2d_ns | 63.315 ms | 0.25% |
| transfer_rerank_h2d_ns | 61.821 ms | 0.24% |

- transfer_ns 内部热点：占比最高的几项是 `transfer_distance_d2h_ns`（69.49%）、`transfer_rabitq_h2d_ns`（29.45%）、`transfer_rerank_d2h_ns`（0.57%）。

### Latency

| 延迟字段 | 值 |
|---|---|
| end_to_end_ns | 957732.540 ms |
| mean_end_to_end_ns | 94.891 ms |
| mean_queue_wait_ns | 47.347 ms |
| mean_service_ns | 47.544 ms |
| p50_end_to_end_ns | 50.661 ms |
| p50_service_ns | 43.455 ms |
| p95_end_to_end_ns | 93.679 ms |
| p95_service_ns | 80.189 ms |
| p99_end_to_end_ns | 235.738 ms |
| p99_service_ns | 101.523 ms |
| queue_wait_ns | 477870.264 ms |
| service_ns | 479862.276 ms |

### Counters

| 字段 | 值 |
|---|---|
| rdma_read_bytes | 90,326,364,758 |
| h2d_bytes | 87,663,654,104 |
| rabitq_rdma_bytes | 71,628,046,360 |
| vector_rdma_bytes | 16,032,177,664 |
| neighbor_rdma_bytes | 2,665,817,838 |
| d2h_bytes | 566,484,072 |
| visited_neighborlists | 5,196,518 |
| rabitq_kernels | 5,096,611 |
| cache_hits | 404,572 |
| exact_reranks | 40,360 |
| cache_misses | 39,392 |
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
| rdma_read_bytes | 357,567,565,968 |
| h2d_bytes | 354,551,030,048 |
| d2h_bytes | 906,737,616 |
| rdma_write_bytes | 214,800,197 |

## Insert / Query 对比

| 类别 | Insert 占比 | Query 占比 |
|---|---|---|
| cpu_ns | 5.10% | 7.01% |
| gpu_ns | 69.96% | 28.71% |
| rdma_ns | 21.83% | 59.02% |
| transfer_ns | 3.11% | 5.26% |

- Insert 最大部分是 **gpu_ns**，占 **69.96%**。
- Query 最大部分是 **rdma_ns**，占 **59.02%**。
- Insert 更偏向 GPU 计算密集。
- Query 更偏向 RDMA / 远端访问受限。
