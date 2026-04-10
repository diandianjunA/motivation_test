# Breakdown 分析报告

## 实验元信息

- **client_threads**: 2
- **coroutines**: 4
- **dim**: 1,024
- **measure_mixed**
  - **completed_reads**: 5,757
  - **completed_writes**: 339
  - **issued_reads**: 5,757
  - **issued_writes**: 339
- **measure_ops**: 1,000
- **measure_seconds**: 60
- **operation_granularity**: single_vector
- **read_ratio**: 0.5
- **run_mode**: time
- **search_mode**: rabitq_gpu
- **synthetic_query_vectors**: 8,192
- **threads**: 4
- **time_completion_policy**: drain
- **time_issue_policy**: bounded_by_observed_call_latency
- **warmup_mixed**
  - **completed_reads**: 2,527
  - **completed_writes**: 403
  - **issued_reads**: 2,527
  - **issued_writes**: 403
- **warmup_ops**: 100
- **warmup_seconds**: 30
- **workload**: mixed

## Bottleneck Summary

### insert

```text
insert breakdown
  count: 339
  latency_ms: mean=176.877 p50=203.901 p95=228.636 p99=239.795
  top_categories:
    gpu_ns: 52292.1 ms (87.2117%)
    rdma_ns: 3550.11 ms (5.92081%)
    cpu_ns: 2083.67 ms (3.4751%)
    transfer_ns: 2034.05 ms (3.39236%)
```

### query

```text
query breakdown
  count: 5757
  latency_ms: mean=10.4033 p50=10.01 p95=16.7454 p99=20.588
  top_categories:
    gpu_ns: 29022.4 ms (48.4756%)
    rdma_ns: 15811.7 ms (26.41%)
    transfer_ns: 9314.97 ms (15.5587%)
    cpu_ns: 5721.02 ms (9.55573%)
```

## INSERT 分析

- 操作数：**339**
- 平均端到端延迟：**176.877 ms**
- P50 端到端延迟：**203.901 ms**
- P95 端到端延迟：**228.636 ms**
- P99 端到端延迟：**239.795 ms**

### 一级 Breakdown 占比

| 部分 | 时间 | 占比 |
|---|---|---|
| gpu_ns | 52292.109 ms | 87.21% |
| rdma_ns | 3550.115 ms | 5.92% |
| cpu_ns | 2083.667 ms | 3.48% |
| transfer_ns | 2034.054 ms | 3.39% |

- insert 一级热点：占比最高的几项是 `gpu_ns`（87.21%）、`rdma_ns`（5.92%）、`cpu_ns`（3.48%）。

### Sub Breakdown 细分占比

#### cpu_ns

| 部分 | 时间 | 占比 |
|---|---|---|
| cpu_insert_stage_candidates_ns | 1296.309 ms | 62.21% |
| cpu_insert_filter_ns | 250.975 ms | 12.04% |
| cpu_insert_runtime_overhead_ns | 212.590 ms | 10.20% |
| cpu_insert_beam_update_ns | 94.597 ms | 4.54% |
| cpu_insert_select_ns | 75.984 ms | 3.65% |
| cpu_insert_candidate_sort_ns | 75.912 ms | 3.64% |
| cpu_insert_finalize_ns | 36.834 ms | 1.77% |
| cpu_insert_overflow_prepare_ns | 20.453 ms | 0.98% |
| cpu_insert_pruned_neighbor_collect_ns | 6.109 ms | 0.29% |
| cpu_insert_neighbor_collect_ns | 4.968 ms | 0.24% |
| cpu_insert_prune_prepare_ns | 2.797 ms | 0.13% |
| cpu_insert_init_ns | 2.741 ms | 0.13% |
| cpu_insert_preprune_sort_ns | 2.380 ms | 0.11% |
| cpu_insert_candidate_collect_ns | 0.676 ms | 0.03% |
| cpu_cache_lookup_ns | 0.232 ms | 0.01% |
| cpu_insert_neighbor_prepare_ns | 0.095 ms | 0.00% |
| cpu_insert_quantize_prepare_ns | 0.015 ms | 0.00% |

- cpu_ns 内部热点：占比最高的几项是 `cpu_insert_stage_candidates_ns`（62.21%）、`cpu_insert_filter_ns`（12.04%）、`cpu_insert_runtime_overhead_ns`（10.20%）。

#### gpu_ns

| 部分 | 时间 | 占比 |
|---|---|---|
| gpu_insert_overflow_prune_ns | 43548.927 ms | 83.28% |
| gpu_insert_prune_ns | 5015.912 ms | 9.59% |
| gpu_insert_distance_ns | 2851.308 ms | 5.45% |
| gpu_insert_overflow_distance_ns | 787.554 ms | 1.51% |
| gpu_insert_quantize_ns | 88.408 ms | 0.17% |

- gpu_ns 内部热点：占比最高的几项是 `gpu_insert_overflow_prune_ns`（83.28%）、`gpu_insert_prune_ns`（9.59%）、`gpu_insert_distance_ns`（5.45%）。

#### rdma_ns

| 部分 | 时间 | 占比 |
|---|---|---|
| rdma_vector_fetch_ns | 1767.391 ms | 49.78% |
| rdma_neighbor_fetch_ns | 831.484 ms | 23.42% |
| rdma_overflow_vec_fetch_ns | 503.548 ms | 14.18% |
| rdma_pruned_neighbor_write_ns | 98.776 ms | 2.78% |
| rdma_neighbor_list_read_ns | 85.491 ms | 2.41% |
| rdma_neighbor_node_read_ns | 81.418 ms | 2.29% |
| rdma_candidate_fetch_ns | 62.558 ms | 1.76% |
| rdma_neighbor_lock_ns | 57.514 ms | 1.62% |
| rdma_neighbor_unlock_ns | 37.957 ms | 1.07% |
| rdma_neighbor_list_write_ns | 17.898 ms | 0.50% |
| rdma_new_node_write_ns | 2.507 ms | 0.07% |
| rdma_alloc_ns | 2.013 ms | 0.06% |
| rdma_medoid_ptr_ns | 1.559 ms | 0.04% |

- rdma_ns 内部热点：占比最高的几项是 `rdma_vector_fetch_ns`（49.78%）、`rdma_neighbor_fetch_ns`（23.42%）、`rdma_overflow_vec_fetch_ns`（14.18%）。

#### transfer_ns

| 部分 | 时间 | 占比 |
|---|---|---|
| transfer_distance_d2h_ns | 1070.821 ms | 52.64% |
| transfer_candidate_h2d_ns | 353.888 ms | 17.40% |
| transfer_overflow_prune_d2h_ns | 197.229 ms | 9.70% |
| transfer_overflow_dist_d2h_ns | 158.037 ms | 7.77% |
| transfer_overflow_prune_inputs_h2d_ns | 95.731 ms | 4.71% |
| transfer_overflow_query_h2d_ns | 62.034 ms | 3.05% |
| transfer_overflow_candidate_h2d_ns | 52.405 ms | 2.58% |
| transfer_quantize_d2h_ns | 33.834 ms | 1.66% |
| transfer_prune_d2h_ns | 4.976 ms | 0.24% |
| transfer_prune_h2d_ns | 2.799 ms | 0.14% |
| transfer_insert_query_h2d_ns | 2.300 ms | 0.11% |

- transfer_ns 内部热点：占比最高的几项是 `transfer_distance_d2h_ns`（52.64%）、`transfer_candidate_h2d_ns`（17.40%）、`transfer_overflow_prune_d2h_ns`（9.70%）。

### Latency

| 延迟字段 | 值 |
|---|---|
| end_to_end_ns | 59961.307 ms |
| mean_end_to_end_ns | 176.877 ms |
| mean_queue_wait_ns | 0.004 ms |
| mean_service_ns | 176.873 ms |
| p50_end_to_end_ns | 203.901 ms |
| p50_service_ns | 203.898 ms |
| p95_end_to_end_ns | 228.636 ms |
| p95_service_ns | 228.633 ms |
| p99_end_to_end_ns | 239.795 ms |
| p99_service_ns | 239.791 ms |
| queue_wait_ns | 1.362 ms |
| service_ns | 59959.945 ms |

### Counters

| 字段 | 值 |
|---|---|
| rdma_read_bytes | 13,932,792,980 |
| h2d_bytes | 13,892,676,912 |
| vector_rdma_bytes | 13,836,331,904 |
| neighbor_rdma_bytes | 96,458,364 |
| d2h_bytes | 17,035,272 |
| rdma_write_bytes | 10,971,232 |
| l2_kernels | 116,553 |
| lock_attempts | 17,848 |
| prune_kernels | 15,009 |
| overflow_prunes | 14,670 |
| cache_hits | 339 |
| remote_allocations | 339 |
| cache_misses | 0 |
| cas_failures | 0 |
| exact_reranks | 0 |
| lock_retries | 0 |
| rabitq_kernels | 0 |
| rabitq_rdma_bytes | 0 |
| visited_neighborlists | 0 |
| visited_nodes | 0 |

## QUERY 分析

- 操作数：**5,757**
- 平均端到端延迟：**10.403 ms**
- P50 端到端延迟：**10.010 ms**
- P95 端到端延迟：**16.745 ms**
- P99 端到端延迟：**20.588 ms**

### 一级 Breakdown 占比

| 部分 | 时间 | 占比 |
|---|---|---|
| gpu_ns | 29022.377 ms | 48.48% |
| rdma_ns | 15811.660 ms | 26.41% |
| transfer_ns | 9314.974 ms | 15.56% |
| cpu_ns | 5721.018 ms | 9.56% |

- query 一级热点：占比最高的几项是 `gpu_ns`（48.48%）、`rdma_ns`（26.41%）、`transfer_ns`（15.56%）。

### Sub Breakdown 细分占比

#### cpu_ns

| 部分 | 时间 | 占比 |
|---|---|---|
| cpu_query_stage_candidates_ns | 2076.578 ms | 36.30% |
| cpu_query_filter_ns | 1888.865 ms | 33.02% |
| cpu_query_beam_update_ns | 521.088 ms | 9.11% |
| cpu_query_runtime_overhead_ns | 513.817 ms | 8.98% |
| cpu_query_rerank_prepare_ns | 240.050 ms | 4.20% |
| cpu_query_select_ns | 170.501 ms | 2.98% |
| cpu_query_finalize_ns | 102.117 ms | 1.78% |
| cpu_query_result_ids_ns | 101.142 ms | 1.77% |
| cpu_cache_lookup_ns | 81.869 ms | 1.43% |
| cpu_query_beam_sort_ns | 21.988 ms | 0.38% |
| cpu_query_rerank_collect_ns | 2.455 ms | 0.04% |
| cpu_query_rerank_update_ns | 0.548 ms | 0.01% |

- cpu_ns 内部热点：占比最高的几项是 `cpu_query_stage_candidates_ns`（36.30%）、`cpu_query_filter_ns`（33.02%）、`cpu_query_beam_update_ns`（9.11%）。

#### gpu_ns

| 部分 | 时间 | 占比 |
|---|---|---|
| gpu_query_distance_ns | 28513.793 ms | 98.25% |
| gpu_query_rerank_ns | 280.052 ms | 0.96% |
| gpu_query_prepare_ns | 228.532 ms | 0.79% |

- gpu_ns 内部热点：占比最高的几项是 `gpu_query_distance_ns`（98.25%）、`gpu_query_rerank_ns`（0.96%）、`gpu_query_prepare_ns`（0.79%）。

#### rdma_ns

| 部分 | 时间 | 占比 |
|---|---|---|
| rdma_rabitq_fetch_ns | 11891.781 ms | 75.21% |
| rdma_neighbor_fetch_ns | 3655.070 ms | 23.12% |
| rdma_rerank_fetch_ns | 242.053 ms | 1.53% |
| rdma_medoid_ptr_ns | 22.757 ms | 0.14% |

- rdma_ns 内部热点：占比最高的几项是 `rdma_rabitq_fetch_ns`（75.21%）、`rdma_neighbor_fetch_ns`（23.12%）、`rdma_rerank_fetch_ns`（1.53%）。

#### transfer_ns

| 部分 | 时间 | 占比 |
|---|---|---|
| transfer_distance_d2h_ns | 6784.608 ms | 72.84% |
| transfer_rabitq_h2d_ns | 2421.909 ms | 26.00% |
| transfer_rerank_d2h_ns | 58.562 ms | 0.63% |
| transfer_query_h2d_ns | 28.711 ms | 0.31% |
| transfer_rerank_h2d_ns | 21.184 ms | 0.23% |

- transfer_ns 内部热点：占比最高的几项是 `transfer_distance_d2h_ns`（72.84%）、`transfer_rabitq_h2d_ns`（26.00%）、`transfer_rerank_d2h_ns`（0.63%）。

### Latency

| 延迟字段 | 值 |
|---|---|
| end_to_end_ns | 59891.549 ms |
| mean_end_to_end_ns | 10.403 ms |
| mean_queue_wait_ns | 0.004 ms |
| mean_service_ns | 10.400 ms |
| p50_end_to_end_ns | 10.010 ms |
| p50_service_ns | 10.007 ms |
| p95_end_to_end_ns | 16.745 ms |
| p95_service_ns | 16.742 ms |
| p99_end_to_end_ns | 20.588 ms |
| p99_service_ns | 20.584 ms |
| queue_wait_ns | 21.520 ms |
| service_ns | 59870.029 ms |

### Counters

| 字段 | 值 |
|---|---|
| rdma_read_bytes | 14,090,848,035 |
| h2d_bytes | 13,674,805,704 |
| rabitq_rdma_bytes | 11,387,480,520 |
| vector_rdma_bytes | 2,295,423,360 |
| neighbor_rdma_bytes | 407,898,099 |
| d2h_bytes | 89,806,692 |
| visited_neighborlists | 795,123 |
| rabitq_kernels | 686,909 |
| cache_hits | 55,623 |
| cache_misses | 7,704 |
| exact_reranks | 5,757 |
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
| rdma_read_bytes | 28,023,641,015 |
| h2d_bytes | 27,567,482,616 |
| d2h_bytes | 106,841,964 |
| rdma_write_bytes | 10,971,232 |

## Insert / Query 对比

| 类别 | Insert 占比 | Query 占比 |
|---|---|---|
| cpu_ns | 3.48% | 9.56% |
| gpu_ns | 87.21% | 48.48% |
| rdma_ns | 5.92% | 26.41% |
| transfer_ns | 3.39% | 15.56% |

- Insert 最大部分是 **gpu_ns**，占 **87.21%**。
- Query 最大部分是 **gpu_ns**，占 **48.48%**。
- Insert 更偏向 GPU 计算密集。
- Query 更偏向 RDMA / 远端访问受限。
