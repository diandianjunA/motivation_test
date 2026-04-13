# Breakdown 分析报告

## 实验元信息

- **client_threads**: 16
- **coroutines**: 4
- **dim**: 1,024
- **measure_mixed**
  - **completed_reads**: 9,244
  - **completed_writes**: 1,647
  - **issued_reads**: 9,244
  - **issued_writes**: 1,647
- **measure_ops**: 1,000
- **measure_seconds**: 60
- **operation_granularity**: single_vector
- **read_ratio**: 0.5
- **run_mode**: time
- **search_mode**: rabitq_gpu
- **synthetic_query_vectors**: 65,536
- **threads**: 4
- **time_completion_policy**: drain
- **time_issue_policy**: bounded_by_observed_call_latency
- **warmup_mixed**
  - **completed_reads**: 3,896
  - **completed_writes**: 827
  - **issued_reads**: 3,896
  - **issued_writes**: 827
- **warmup_ops**: 100
- **warmup_seconds**: 30
- **workload**: mixed

## Bottleneck Summary

### insert

```text
insert breakdown
  count: 1647
  latency_ms: mean=290.576 p50=306.231 p95=386.63 p99=439.437
  top_categories:
    gpu_ns: 340981 ms (71.2569%)
    rdma_ns: 98829 ms (20.6529%)
    cpu_ns: 23582.6 ms (4.92821%)
    transfer_ns: 15130.9 ms (3.162%)
```

### query

```text
query breakdown
  count: 9244
  latency_ms: mean=51.7173 p50=48.4954 p95=82.0935 p99=100.23
  top_categories:
    rdma_ns: 279098 ms (58.4685%)
    gpu_ns: 139922 ms (29.3124%)
    cpu_ns: 33240.9 ms (6.96366%)
    transfer_ns: 25086.4 ms (5.25538%)
```

## INSERT 分析

- 操作数：**1,647**
- 平均端到端延迟：**290.576 ms**
- P50 端到端延迟：**306.231 ms**
- P95 端到端延迟：**386.630 ms**
- P99 端到端延迟：**439.437 ms**

### 一级 Breakdown 占比

| 部分 | 时间 | 占比 |
|---|---|---|
| gpu_ns | 340980.836 ms | 71.26% |
| rdma_ns | 98829.003 ms | 20.65% |
| cpu_ns | 23582.625 ms | 4.93% |
| transfer_ns | 15130.928 ms | 3.16% |

- insert 一级热点：占比最高的几项是 `gpu_ns`（71.26%）、`rdma_ns`（20.65%）、`cpu_ns`（4.93%）。

### Sub Breakdown 细分占比

#### cpu_ns

| 部分 | 时间 | 占比 |
|---|---|---|
| cpu_insert_stage_candidates_ns | 15803.987 ms | 67.02% |
| cpu_insert_runtime_overhead_ns | 3940.636 ms | 16.71% |
| cpu_insert_filter_ns | 1499.976 ms | 6.36% |
| cpu_insert_candidate_sort_ns | 1026.436 ms | 4.35% |
| cpu_insert_beam_update_ns | 492.900 ms | 2.09% |
| cpu_insert_select_ns | 354.970 ms | 1.51% |
| cpu_insert_finalize_ns | 203.630 ms | 0.86% |
| cpu_insert_overflow_prepare_ns | 134.102 ms | 0.57% |
| cpu_insert_pruned_neighbor_collect_ns | 39.451 ms | 0.17% |
| cpu_insert_neighbor_collect_ns | 35.256 ms | 0.15% |
| cpu_insert_prune_prepare_ns | 19.033 ms | 0.08% |
| cpu_insert_init_ns | 14.808 ms | 0.06% |
| cpu_insert_preprune_sort_ns | 11.256 ms | 0.05% |
| cpu_insert_candidate_collect_ns | 4.715 ms | 0.02% |
| cpu_cache_lookup_ns | 1.086 ms | 0.00% |
| cpu_insert_neighbor_prepare_ns | 0.304 ms | 0.00% |
| cpu_insert_quantize_prepare_ns | 0.077 ms | 0.00% |

- cpu_ns 内部热点：占比最高的几项是 `cpu_insert_stage_candidates_ns`（67.02%）、`cpu_insert_runtime_overhead_ns`（16.71%）、`cpu_insert_filter_ns`（6.36%）。

#### gpu_ns

| 部分 | 时间 | 占比 |
|---|---|---|
| gpu_insert_overflow_prune_ns | 271792.919 ms | 79.71% |
| gpu_insert_distance_ns | 34896.698 ms | 10.23% |
| gpu_insert_prune_ns | 26288.319 ms | 7.71% |
| gpu_insert_overflow_distance_ns | 7490.453 ms | 2.20% |
| gpu_insert_quantize_ns | 512.447 ms | 0.15% |

- gpu_ns 内部热点：占比最高的几项是 `gpu_insert_overflow_prune_ns`（79.71%）、`gpu_insert_distance_ns`（10.23%）、`gpu_insert_prune_ns`（7.71%）。

#### rdma_ns

| 部分 | 时间 | 占比 |
|---|---|---|
| rdma_vector_fetch_ns | 37481.234 ms | 37.93% |
| rdma_neighbor_fetch_ns | 33416.161 ms | 33.81% |
| rdma_overflow_vec_fetch_ns | 10746.432 ms | 10.87% |
| rdma_pruned_neighbor_write_ns | 3870.982 ms | 3.92% |
| rdma_neighbor_lock_ns | 3700.698 ms | 3.74% |
| rdma_neighbor_node_read_ns | 2934.433 ms | 2.97% |
| rdma_neighbor_list_read_ns | 2863.790 ms | 2.90% |
| rdma_neighbor_unlock_ns | 2481.675 ms | 2.51% |
| rdma_candidate_fetch_ns | 923.073 ms | 0.93% |
| rdma_neighbor_list_write_ns | 220.043 ms | 0.22% |
| rdma_alloc_ns | 72.823 ms | 0.07% |
| rdma_medoid_ptr_ns | 62.695 ms | 0.06% |
| rdma_new_node_write_ns | 54.964 ms | 0.06% |

- rdma_ns 内部热点：占比最高的几项是 `rdma_vector_fetch_ns`（37.93%）、`rdma_neighbor_fetch_ns`（33.81%）、`rdma_overflow_vec_fetch_ns`（10.87%）。

#### transfer_ns

| 部分 | 时间 | 占比 |
|---|---|---|
| transfer_distance_d2h_ns | 7658.983 ms | 50.62% |
| transfer_candidate_h2d_ns | 2761.159 ms | 18.25% |
| transfer_overflow_prune_d2h_ns | 1503.256 ms | 9.93% |
| transfer_overflow_dist_d2h_ns | 1205.286 ms | 7.97% |
| transfer_overflow_prune_inputs_h2d_ns | 792.784 ms | 5.24% |
| transfer_overflow_query_h2d_ns | 530.514 ms | 3.51% |
| transfer_overflow_candidate_h2d_ns | 449.611 ms | 2.97% |
| transfer_quantize_d2h_ns | 169.481 ms | 1.12% |
| transfer_prune_d2h_ns | 30.114 ms | 0.20% |
| transfer_prune_h2d_ns | 19.037 ms | 0.13% |
| transfer_insert_query_h2d_ns | 10.703 ms | 0.07% |

- transfer_ns 内部热点：占比最高的几项是 `transfer_distance_d2h_ns`（50.62%）、`transfer_candidate_h2d_ns`（18.25%）、`transfer_overflow_prune_d2h_ns`（9.93%）。

### Latency

| 延迟字段 | 值 |
|---|---|
| end_to_end_ns | 478579.248 ms |
| mean_end_to_end_ns | 290.576 ms |
| mean_queue_wait_ns | 0.034 ms |
| mean_service_ns | 290.542 ms |
| p50_end_to_end_ns | 306.231 ms |
| p50_service_ns | 306.166 ms |
| p95_end_to_end_ns | 386.630 ms |
| p95_service_ns | 386.592 ms |
| p99_end_to_end_ns | 439.437 ms |
| p99_service_ns | 439.319 ms |
| queue_wait_ns | 55.855 ms |
| service_ns | 478523.393 ms |

### Counters

| 字段 | 值 |
|---|---|
| rdma_read_bytes | 267,358,079,786 |
| h2d_bytes | 267,012,362,616 |
| vector_rdma_bytes | 265,495,983,872 |
| neighbor_rdma_bytes | 1,862,043,282 |
| d2h_bytes | 341,632,156 |
| rdma_write_bytes | 221,634,116 |
| l2_kernels | 2,569,311 |
| prune_kernels | 344,480 |
| overflow_prunes | 337,835 |
| lock_attempts | 180,023 |
| cas_failures | 89,145 |
| lock_retries | 89,145 |
| remote_allocations | 6,587 |
| cache_hits | 6,580 |
| cache_misses | 0 |
| exact_reranks | 0 |
| rabitq_kernels | 0 |
| rabitq_rdma_bytes | 0 |
| visited_neighborlists | 0 |
| visited_nodes | 0 |

## QUERY 分析

- 操作数：**9,244**
- 平均端到端延迟：**51.717 ms**
- P50 端到端延迟：**48.495 ms**
- P95 端到端延迟：**82.094 ms**
- P99 端到端延迟：**100.230 ms**

### 一级 Breakdown 占比

| 部分 | 时间 | 占比 |
|---|---|---|
| rdma_ns | 279098.008 ms | 58.47% |
| gpu_ns | 139922.181 ms | 29.31% |
| cpu_ns | 33240.878 ms | 6.96% |
| transfer_ns | 25086.428 ms | 5.26% |

- query 一级热点：占比最高的几项是 `rdma_ns`（58.47%）、`gpu_ns`（29.31%）、`cpu_ns`（6.96%）。

### Sub Breakdown 细分占比

#### cpu_ns

| 部分 | 时间 | 占比 |
|---|---|---|
| cpu_query_stage_candidates_ns | 19309.905 ms | 58.09% |
| cpu_query_filter_ns | 5785.456 ms | 17.40% |
| cpu_query_runtime_overhead_ns | 2555.513 ms | 7.69% |
| cpu_query_finalize_ns | 1118.755 ms | 3.37% |
| cpu_query_result_ids_ns | 1116.679 ms | 3.36% |
| cpu_cache_lookup_ns | 1085.267 ms | 3.26% |
| cpu_query_rerank_prepare_ns | 1013.050 ms | 3.05% |
| cpu_query_beam_update_ns | 924.640 ms | 2.78% |
| cpu_query_select_ns | 288.417 ms | 0.87% |
| cpu_query_beam_sort_ns | 37.215 ms | 0.11% |
| cpu_query_rerank_collect_ns | 4.728 ms | 0.01% |
| cpu_query_rerank_update_ns | 1.252 ms | 0.00% |

- cpu_ns 内部热点：占比最高的几项是 `cpu_query_stage_candidates_ns`（58.09%）、`cpu_query_filter_ns`（17.40%）、`cpu_query_runtime_overhead_ns`（7.69%）。

#### gpu_ns

| 部分 | 时间 | 占比 |
|---|---|---|
| gpu_query_distance_ns | 137623.685 ms | 98.36% |
| gpu_query_rerank_ns | 1176.894 ms | 0.84% |
| gpu_query_prepare_ns | 1121.602 ms | 0.80% |

- gpu_ns 内部热点：占比最高的几项是 `gpu_query_distance_ns`（98.36%）、`gpu_query_rerank_ns`（0.84%）、`gpu_query_prepare_ns`（0.80%）。

#### rdma_ns

| 部分 | 时间 | 占比 |
|---|---|---|
| rdma_rabitq_fetch_ns | 155579.369 ms | 55.74% |
| rdma_neighbor_fetch_ns | 120930.356 ms | 43.33% |
| rdma_rerank_fetch_ns | 1766.077 ms | 0.63% |
| rdma_medoid_ptr_ns | 822.206 ms | 0.29% |

- rdma_ns 内部热点：占比最高的几项是 `rdma_rabitq_fetch_ns`（55.74%）、`rdma_neighbor_fetch_ns`（43.33%）、`rdma_rerank_fetch_ns`（0.63%）。

#### transfer_ns

| 部分 | 时间 | 占比 |
|---|---|---|
| transfer_distance_d2h_ns | 17432.378 ms | 69.49% |
| transfer_rabitq_h2d_ns | 7409.735 ms | 29.54% |
| transfer_rerank_d2h_ns | 129.420 ms | 0.52% |
| transfer_query_h2d_ns | 59.054 ms | 0.24% |
| transfer_rerank_h2d_ns | 55.841 ms | 0.22% |

- transfer_ns 内部热点：占比最高的几项是 `transfer_distance_d2h_ns`（69.49%）、`transfer_rabitq_h2d_ns`（29.54%）、`transfer_rerank_d2h_ns`（0.52%）。

### Latency

| 延迟字段 | 值 |
|---|---|
| end_to_end_ns | 478074.846 ms |
| mean_end_to_end_ns | 51.717 ms |
| mean_queue_wait_ns | 0.079 ms |
| mean_service_ns | 51.639 ms |
| p50_end_to_end_ns | 48.495 ms |
| p50_service_ns | 48.424 ms |
| p95_end_to_end_ns | 82.094 ms |
| p95_service_ns | 82.027 ms |
| p99_end_to_end_ns | 100.230 ms |
| p99_service_ns | 100.144 ms |
| queue_wait_ns | 727.351 ms |
| service_ns | 477347.495 ms |

### Counters

| 字段 | 值 |
|---|---|
| rdma_read_bytes | 94,586,925,404 |
| h2d_bytes | 91,973,734,808 |
| rabitq_rdma_bytes | 77,305,333,040 |
| vector_rdma_bytes | 14,652,363,504 |
| neighbor_rdma_bytes | 2,628,934,164 |
| d2h_bytes | 609,072,096 |
| visited_neighborlists | 5,122,570 |
| rabitq_kernels | 5,052,313 |
| cache_hits | 362,805 |
| cache_misses | 42,545 |
| exact_reranks | 36,825 |
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
| rdma_read_bytes | 361,945,005,190 |
| h2d_bytes | 358,986,097,424 |
| d2h_bytes | 950,704,252 |
| rdma_write_bytes | 221,634,116 |

## Insert / Query 对比

| 类别 | Insert 占比 | Query 占比 |
|---|---|---|
| cpu_ns | 4.93% | 6.96% |
| gpu_ns | 71.26% | 29.31% |
| rdma_ns | 20.65% | 58.47% |
| transfer_ns | 3.16% | 5.26% |

- Insert 最大部分是 **gpu_ns**，占 **71.26%**。
- Query 最大部分是 **rdma_ns**，占 **58.47%**。
- Insert 更偏向 GPU 计算密集。
- Query 更偏向 RDMA / 远端访问受限。
