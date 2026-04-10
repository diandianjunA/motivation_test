# Breakdown 分析报告

## 实验元信息

- **client_threads**: 16
- **coroutines**: 4
- **dim**: 1,024
- **measure_mixed**
  - **completed_reads**: 13,171
  - **completed_writes**: 1,550
  - **issued_reads**: 13,171
  - **issued_writes**: 1,550
- **measure_ops**: 1,000
- **measure_seconds**: 60
- **operation_granularity**: single_vector
- **read_ratio**: 0.5
- **run_mode**: time
- **search_mode**: exact_gpu
- **synthetic_query_vectors**: 65,536
- **threads**: 4
- **time_completion_policy**: drain
- **time_issue_policy**: bounded_by_observed_call_latency
- **warmup_mixed**
  - **completed_reads**: 5,339
  - **completed_writes**: 729
  - **issued_reads**: 5,339
  - **issued_writes**: 729
- **warmup_ops**: 100
- **warmup_seconds**: 30
- **workload**: mixed

## Bottleneck Summary

### insert

```text
insert breakdown
  count: 1550
  latency_ms: mean=308.449 p50=319.872 p95=422.927 p99=474.657
  top_categories:
    gpu_ns: 319402 ms (66.817%)
    rdma_ns: 117635 ms (24.6085%)
    cpu_ns: 26589.2 ms (5.5623%)
    transfer_ns: 14399 ms (3.01218%)
```

### query

```text
query breakdown
  count: 13171
  latency_ms: mean=36.2874 p50=34.0625 p95=59.0615 p99=72.9247
  top_categories:
    rdma_ns: 275750 ms (57.9104%)
    gpu_ns: 131624 ms (27.6425%)
    cpu_ns: 53876.6 ms (11.3146%)
    transfer_ns: 14915.7 ms (3.13244%)
```

## INSERT 分析

- 操作数：**1,550**
- 平均端到端延迟：**308.449 ms**
- P50 端到端延迟：**319.872 ms**
- P95 端到端延迟：**422.927 ms**
- P99 端到端延迟：**474.657 ms**

### 一级 Breakdown 占比

| 部分 | 时间 | 占比 |
|---|---|---|
| gpu_ns | 319402.217 ms | 66.82% |
| rdma_ns | 117635.117 ms | 24.61% |
| cpu_ns | 26589.205 ms | 5.56% |
| transfer_ns | 14399.004 ms | 3.01% |

- insert 一级热点：占比最高的几项是 `gpu_ns`（66.82%）、`rdma_ns`（24.61%）、`cpu_ns`（5.56%）。

### Sub Breakdown 细分占比

#### cpu_ns

| 部分 | 时间 | 占比 |
|---|---|---|
| cpu_insert_stage_candidates_ns | 17727.539 ms | 66.67% |
| cpu_insert_runtime_overhead_ns | 4947.973 ms | 18.61% |
| cpu_insert_filter_ns | 1475.232 ms | 5.55% |
| cpu_insert_candidate_sort_ns | 1164.755 ms | 4.38% |
| cpu_insert_beam_update_ns | 464.396 ms | 1.75% |
| cpu_insert_select_ns | 335.892 ms | 1.26% |
| cpu_insert_finalize_ns | 229.408 ms | 0.86% |
| cpu_insert_overflow_prepare_ns | 124.157 ms | 0.47% |
| cpu_insert_pruned_neighbor_collect_ns | 37.662 ms | 0.14% |
| cpu_insert_neighbor_collect_ns | 32.614 ms | 0.12% |
| cpu_insert_prune_prepare_ns | 19.264 ms | 0.07% |
| cpu_insert_init_ns | 13.794 ms | 0.05% |
| cpu_insert_preprune_sort_ns | 10.759 ms | 0.04% |
| cpu_insert_candidate_collect_ns | 4.280 ms | 0.02% |
| cpu_cache_lookup_ns | 1.085 ms | 0.00% |
| cpu_insert_neighbor_prepare_ns | 0.306 ms | 0.00% |
| cpu_insert_quantize_prepare_ns | 0.088 ms | 0.00% |

- cpu_ns 内部热点：占比最高的几项是 `cpu_insert_stage_candidates_ns`（66.67%）、`cpu_insert_runtime_overhead_ns`（18.61%）、`cpu_insert_filter_ns`（5.55%）。

#### gpu_ns

| 部分 | 时间 | 占比 |
|---|---|---|
| gpu_insert_overflow_prune_ns | 248172.772 ms | 77.70% |
| gpu_insert_distance_ns | 38516.525 ms | 12.06% |
| gpu_insert_prune_ns | 24369.013 ms | 7.63% |
| gpu_insert_overflow_distance_ns | 7819.642 ms | 2.45% |
| gpu_insert_quantize_ns | 524.266 ms | 0.16% |

- gpu_ns 内部热点：占比最高的几项是 `gpu_insert_overflow_prune_ns`（77.70%）、`gpu_insert_distance_ns`（12.06%）、`gpu_insert_prune_ns`（7.63%）。

#### rdma_ns

| 部分 | 时间 | 占比 |
|---|---|---|
| rdma_vector_fetch_ns | 45390.037 ms | 38.59% |
| rdma_neighbor_fetch_ns | 39717.915 ms | 33.76% |
| rdma_overflow_vec_fetch_ns | 13047.958 ms | 11.09% |
| rdma_pruned_neighbor_write_ns | 4390.594 ms | 3.73% |
| rdma_neighbor_lock_ns | 4066.679 ms | 3.46% |
| rdma_neighbor_node_read_ns | 3276.654 ms | 2.79% |
| rdma_neighbor_list_read_ns | 3198.611 ms | 2.72% |
| rdma_neighbor_unlock_ns | 2834.434 ms | 2.41% |
| rdma_candidate_fetch_ns | 1203.338 ms | 1.02% |
| rdma_neighbor_list_write_ns | 276.296 ms | 0.23% |
| rdma_alloc_ns | 88.230 ms | 0.08% |
| rdma_medoid_ptr_ns | 82.298 ms | 0.07% |
| rdma_new_node_write_ns | 62.072 ms | 0.05% |

- rdma_ns 内部热点：占比最高的几项是 `rdma_vector_fetch_ns`（38.59%）、`rdma_neighbor_fetch_ns`（33.76%）、`rdma_overflow_vec_fetch_ns`（11.09%）。

#### transfer_ns

| 部分 | 时间 | 占比 |
|---|---|---|
| transfer_distance_d2h_ns | 7190.806 ms | 49.94% |
| transfer_candidate_h2d_ns | 2687.678 ms | 18.67% |
| transfer_overflow_prune_d2h_ns | 1431.522 ms | 9.94% |
| transfer_overflow_dist_d2h_ns | 1155.373 ms | 8.02% |
| transfer_overflow_prune_inputs_h2d_ns | 768.536 ms | 5.34% |
| transfer_overflow_query_h2d_ns | 489.778 ms | 3.40% |
| transfer_overflow_candidate_h2d_ns | 451.860 ms | 3.14% |
| transfer_quantize_d2h_ns | 164.715 ms | 1.14% |
| transfer_prune_d2h_ns | 29.205 ms | 0.20% |
| transfer_prune_h2d_ns | 19.267 ms | 0.13% |
| transfer_insert_query_h2d_ns | 10.263 ms | 0.07% |

- transfer_ns 内部热点：占比最高的几项是 `transfer_distance_d2h_ns`（49.94%）、`transfer_candidate_h2d_ns`（18.67%）、`transfer_overflow_prune_d2h_ns`（9.94%）。

### Latency

| 延迟字段 | 值 |
|---|---|
| end_to_end_ns | 478096.407 ms |
| mean_end_to_end_ns | 308.449 ms |
| mean_queue_wait_ns | 0.046 ms |
| mean_service_ns | 308.404 ms |
| p50_end_to_end_ns | 319.872 ms |
| p50_service_ns | 319.780 ms |
| p95_end_to_end_ns | 422.927 ms |
| p95_service_ns | 422.921 ms |
| p99_end_to_end_ns | 474.657 ms |
| p99_service_ns | 474.651 ms |
| queue_wait_ns | 70.865 ms |
| service_ns | 478025.542 ms |

### Counters

| 字段 | 值 |
|---|---|
| rdma_read_bytes | 249,881,017,337 |
| h2d_bytes | 249,529,317,096 |
| vector_rdma_bytes | 248,131,247,424 |
| neighbor_rdma_bytes | 1,749,720,393 |
| d2h_bytes | 318,304,096 |
| rdma_write_bytes | 206,027,825 |
| l2_kernels | 2,340,527 |
| prune_kernels | 318,371 |
| overflow_prunes | 312,134 |
| lock_attempts | 179,492 |
| cas_failures | 95,146 |
| lock_retries | 95,146 |
| remote_allocations | 6,197 |
| cache_hits | 6,188 |
| cache_misses | 0 |
| exact_reranks | 0 |
| rabitq_kernels | 0 |
| rabitq_rdma_bytes | 0 |
| visited_neighborlists | 0 |
| visited_nodes | 0 |

## QUERY 分析

- 操作数：**13,171**
- 平均端到端延迟：**36.287 ms**
- P50 端到端延迟：**34.062 ms**
- P95 端到端延迟：**59.062 ms**
- P99 端到端延迟：**72.925 ms**

### 一级 Breakdown 占比

| 部分 | 时间 | 占比 |
|---|---|---|
| rdma_ns | 275750.370 ms | 57.91% |
| gpu_ns | 131624.397 ms | 27.64% |
| cpu_ns | 53876.557 ms | 11.31% |
| transfer_ns | 14915.662 ms | 3.13% |

- query 一级热点：占比最高的几项是 `rdma_ns`（57.91%）、`gpu_ns`（27.64%）、`cpu_ns`（11.31%）。

### Sub Breakdown 细分占比

#### cpu_ns

| 部分 | 时间 | 占比 |
|---|---|---|
| cpu_query_stage_candidates_ns | 38099.255 ms | 70.50% |
| cpu_query_filter_ns | 5049.719 ms | 9.34% |
| cpu_query_finalize_ns | 3533.565 ms | 6.54% |
| cpu_query_result_ids_ns | 3531.821 ms | 6.54% |
| cpu_cache_lookup_ns | 3186.510 ms | 5.90% |
| cpu_query_beam_update_ns | 537.068 ms | 0.99% |
| cpu_query_select_ns | 96.536 ms | 0.18% |
| cpu_query_beam_sort_ns | 7.507 ms | 0.01% |

- cpu_ns 内部热点：占比最高的几项是 `cpu_query_stage_candidates_ns`（70.50%）、`cpu_query_filter_ns`（9.34%）、`cpu_query_finalize_ns`（6.54%）。

#### gpu_ns

| 部分 | 时间 | 占比 |
|---|---|---|
| gpu_query_distance_ns | 131624.397 ms | 100.00% |

- gpu_ns 内部热点：占比最高的几项是 `gpu_query_distance_ns`（100.00%）。

#### rdma_ns

| 部分 | 时间 | 占比 |
|---|---|---|
| rdma_vector_fetch_ns | 160385.410 ms | 58.16% |
| rdma_neighbor_fetch_ns | 113368.687 ms | 41.11% |
| rdma_medoid_ptr_ns | 1996.273 ms | 0.72% |

- rdma_ns 内部热点：占比最高的几项是 `rdma_vector_fetch_ns`（58.16%）、`rdma_neighbor_fetch_ns`（41.11%）、`rdma_medoid_ptr_ns`（0.72%）。

#### transfer_ns

| 部分 | 时间 | 占比 |
|---|---|---|
| transfer_distance_d2h_ns | 10553.952 ms | 70.76% |
| transfer_candidate_h2d_ns | 4272.976 ms | 28.65% |
| transfer_query_h2d_ns | 88.734 ms | 0.59% |

- transfer_ns 内部热点：占比最高的几项是 `transfer_distance_d2h_ns`（70.76%）、`transfer_candidate_h2d_ns`（28.65%）、`transfer_query_h2d_ns`（0.59%）。

### Latency

| 延迟字段 | 值 |
|---|---|
| end_to_end_ns | 477941.872 ms |
| mean_end_to_end_ns | 36.287 ms |
| mean_queue_wait_ns | 0.135 ms |
| mean_service_ns | 36.153 ms |
| p50_end_to_end_ns | 34.062 ms |
| p50_service_ns | 33.926 ms |
| p95_end_to_end_ns | 59.062 ms |
| p95_service_ns | 58.907 ms |
| p99_end_to_end_ns | 72.925 ms |
| p99_service_ns | 72.724 ms |
| queue_wait_ns | 1774.886 ms |
| service_ns | 476166.986 ms |

### Counters

| 字段 | 值 |
|---|---|
| rdma_read_bytes | 501,080,789,193 |
| h2d_bytes | 499,844,710,400 |
| vector_rdma_bytes | 499,620,569,568 |
| neighbor_rdma_bytes | 1,459,801,521 |
| d2h_bytes | 488,015,296 |
| visited_neighborlists | 2,843,409 |
| cache_hits | 498,816 |
| cache_misses | 76,855 |
| cas_failures | 0 |
| exact_reranks | 0 |
| l2_kernels | 0 |
| lock_attempts | 0 |
| lock_retries | 0 |
| overflow_prunes | 0 |
| prune_kernels | 0 |
| rabitq_kernels | 0 |
| rabitq_rdma_bytes | 0 |
| rdma_write_bytes | 0 |
| remote_allocations | 0 |
| visited_nodes | 0 |

## System Counters

| 字段 | 值 |
|---|---|
| rdma_read_bytes | 750,961,806,530 |
| h2d_bytes | 749,374,027,496 |
| d2h_bytes | 806,319,392 |
| rdma_write_bytes | 206,027,825 |

## Insert / Query 对比

| 类别 | Insert 占比 | Query 占比 |
|---|---|---|
| cpu_ns | 5.56% | 11.31% |
| gpu_ns | 66.82% | 27.64% |
| rdma_ns | 24.61% | 57.91% |
| transfer_ns | 3.01% | 3.13% |

- Insert 最大部分是 **gpu_ns**，占 **66.82%**。
- Query 最大部分是 **rdma_ns**，占 **57.91%**。
- Insert 更偏向 GPU 计算密集。
- Query 更偏向 RDMA / 远端访问受限。
