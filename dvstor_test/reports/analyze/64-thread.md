# Breakdown 分析报告

## 实验元信息

- **client_threads**: 64
- **coroutines**: 4
- **dim**: 1,024
- **measure_mixed**
  - **completed_reads**: 8,935
  - **completed_writes**: 1,629
  - **issued_reads**: 8,935
  - **issued_writes**: 1,629
- **measure_ops**: 1,000
- **measure_seconds**: 60
- **operation_granularity**: single_vector
- **read_ratio**: 0.5
- **run_mode**: time
- **search_mode**: rabitq_gpu
- **synthetic_query_vectors**: 262,144
- **threads**: 4
- **time_completion_policy**: drain
- **time_issue_policy**: bounded_by_observed_call_latency
- **warmup_mixed**
  - **completed_reads**: 3,818
  - **completed_writes**: 843
  - **issued_reads**: 3,818
  - **issued_writes**: 843
- **warmup_ops**: 100
- **warmup_seconds**: 30
- **workload**: mixed

## Bottleneck Summary

### insert

```text
insert breakdown
  count: 1629
  latency_ms: mean=1182.28 p50=345.534 p95=493.428 p99=60193.8
  top_categories:
    gpu_ns: 339846 ms (70.1205%)
    rdma_ns: 105251 ms (21.7165%)
    cpu_ns: 24556.1 ms (5.06667%)
    transfer_ns: 15006.9 ms (3.09638%)
```

### query

```text
query breakdown
  count: 8935
  latency_ms: mean=213.89 p50=57.907 p95=101.805 p99=293.919
  top_categories:
    rdma_ns: 285143 ms (59.318%)
    gpu_ns: 137728 ms (28.6515%)
    cpu_ns: 33570.9 ms (6.98371%)
    transfer_ns: 24260.2 ms (5.04683%)
```

## INSERT 分析

- 操作数：**1,629**
- 平均端到端延迟：**1182.283 ms**
- P50 端到端延迟：**345.534 ms**
- P95 端到端延迟：**493.428 ms**
- P99 端到端延迟：**60193.822 ms**

### 一级 Breakdown 占比

| 部分 | 时间 | 占比 |
|---|---|---|
| gpu_ns | 339846.058 ms | 70.12% |
| rdma_ns | 105251.135 ms | 21.72% |
| cpu_ns | 24556.131 ms | 5.07% |
| transfer_ns | 15006.938 ms | 3.10% |

- insert 一级热点：占比最高的几项是 `gpu_ns`（70.12%）、`rdma_ns`（21.72%）、`cpu_ns`（5.07%）。

### Sub Breakdown 细分占比

#### cpu_ns

| 部分 | 时间 | 占比 |
|---|---|---|
| cpu_insert_stage_candidates_ns | 16550.526 ms | 67.40% |
| cpu_insert_runtime_overhead_ns | 4119.029 ms | 16.77% |
| cpu_insert_filter_ns | 1511.049 ms | 6.15% |
| cpu_insert_candidate_sort_ns | 1062.922 ms | 4.33% |
| cpu_insert_beam_update_ns | 492.383 ms | 2.01% |
| cpu_insert_select_ns | 353.417 ms | 1.44% |
| cpu_insert_finalize_ns | 212.059 ms | 0.86% |
| cpu_insert_overflow_prepare_ns | 131.387 ms | 0.54% |
| cpu_insert_pruned_neighbor_collect_ns | 38.473 ms | 0.16% |
| cpu_insert_neighbor_collect_ns | 34.529 ms | 0.14% |
| cpu_insert_prune_prepare_ns | 18.876 ms | 0.08% |
| cpu_insert_init_ns | 14.608 ms | 0.06% |
| cpu_insert_preprune_sort_ns | 11.149 ms | 0.05% |
| cpu_insert_candidate_collect_ns | 4.302 ms | 0.02% |
| cpu_cache_lookup_ns | 1.131 ms | 0.00% |
| cpu_insert_neighbor_prepare_ns | 0.218 ms | 0.00% |
| cpu_insert_quantize_prepare_ns | 0.071 ms | 0.00% |

- cpu_ns 内部热点：占比最高的几项是 `cpu_insert_stage_candidates_ns`（67.40%）、`cpu_insert_runtime_overhead_ns`（16.77%）、`cpu_insert_filter_ns`（6.15%）。

#### gpu_ns

| 部分 | 时间 | 占比 |
|---|---|---|
| gpu_insert_overflow_prune_ns | 269453.915 ms | 79.29% |
| gpu_insert_distance_ns | 36630.198 ms | 10.78% |
| gpu_insert_prune_ns | 25730.167 ms | 7.57% |
| gpu_insert_overflow_distance_ns | 7525.791 ms | 2.21% |
| gpu_insert_quantize_ns | 505.987 ms | 0.15% |

- gpu_ns 内部热点：占比最高的几项是 `gpu_insert_overflow_prune_ns`（79.29%）、`gpu_insert_distance_ns`（10.78%）、`gpu_insert_prune_ns`（7.57%）。

#### rdma_ns

| 部分 | 时间 | 占比 |
|---|---|---|
| rdma_vector_fetch_ns | 40076.417 ms | 38.08% |
| rdma_neighbor_fetch_ns | 36284.262 ms | 34.47% |
| rdma_overflow_vec_fetch_ns | 11102.804 ms | 10.55% |
| rdma_pruned_neighbor_write_ns | 4014.065 ms | 3.81% |
| rdma_neighbor_lock_ns | 3929.591 ms | 3.73% |
| rdma_neighbor_node_read_ns | 2998.286 ms | 2.85% |
| rdma_neighbor_list_read_ns | 2927.870 ms | 2.78% |
| rdma_neighbor_unlock_ns | 2573.256 ms | 2.44% |
| rdma_candidate_fetch_ns | 956.330 ms | 0.91% |
| rdma_neighbor_list_write_ns | 179.665 ms | 0.17% |
| rdma_medoid_ptr_ns | 75.276 ms | 0.07% |
| rdma_alloc_ns | 74.524 ms | 0.07% |
| rdma_new_node_write_ns | 58.789 ms | 0.06% |

- rdma_ns 内部热点：占比最高的几项是 `rdma_vector_fetch_ns`（38.08%）、`rdma_neighbor_fetch_ns`（34.47%）、`rdma_overflow_vec_fetch_ns`（10.55%）。

#### transfer_ns

| 部分 | 时间 | 占比 |
|---|---|---|
| transfer_distance_d2h_ns | 7530.371 ms | 50.18% |
| transfer_candidate_h2d_ns | 2774.927 ms | 18.49% |
| transfer_overflow_prune_d2h_ns | 1501.048 ms | 10.00% |
| transfer_overflow_dist_d2h_ns | 1189.622 ms | 7.93% |
| transfer_overflow_prune_inputs_h2d_ns | 805.510 ms | 5.37% |
| transfer_overflow_query_h2d_ns | 523.525 ms | 3.49% |
| transfer_overflow_candidate_h2d_ns | 454.880 ms | 3.03% |
| transfer_quantize_d2h_ns | 166.997 ms | 1.11% |
| transfer_prune_d2h_ns | 30.188 ms | 0.20% |
| transfer_prune_h2d_ns | 18.877 ms | 0.13% |
| transfer_insert_query_h2d_ns | 10.994 ms | 0.07% |

- transfer_ns 内部热点：占比最高的几项是 `transfer_distance_d2h_ns`（50.18%）、`transfer_candidate_h2d_ns`（18.49%）、`transfer_overflow_prune_d2h_ns`（10.00%）。

### Latency

| 延迟字段 | 值 |
|---|---|
| end_to_end_ns | 1925939.038 ms |
| mean_end_to_end_ns | 1182.283 ms |
| mean_queue_wait_ns | 884.763 ms |
| mean_service_ns | 297.520 ms |
| p50_end_to_end_ns | 345.534 ms |
| p50_service_ns | 310.990 ms |
| p95_end_to_end_ns | 493.428 ms |
| p95_service_ns | 405.406 ms |
| p99_end_to_end_ns | 60193.822 ms |
| p99_service_ns | 453.040 ms |
| queue_wait_ns | 1441278.775 ms |
| service_ns | 484660.263 ms |

### Counters

| 字段 | 值 |
|---|---|
| rdma_read_bytes | 266,920,767,102 |
| h2d_bytes | 266,584,693,632 |
| vector_rdma_bytes | 265,084,734,224 |
| neighbor_rdma_bytes | 1,835,980,830 |
| d2h_bytes | 340,794,316 |
| rdma_write_bytes | 217,695,987 |
| l2_kernels | 2,538,200 |
| prune_kernels | 342,482 |
| overflow_prunes | 335,968 |
| lock_attempts | 184,580 |
| cas_failures | 95,412 |
| lock_retries | 95,412 |
| remote_allocations | 6,514 |
| cache_hits | 6,507 |
| cache_misses | 0 |
| exact_reranks | 0 |
| rabitq_kernels | 0 |
| rabitq_rdma_bytes | 0 |
| visited_neighborlists | 0 |
| visited_nodes | 0 |

## QUERY 分析

- 操作数：**8,935**
- 平均端到端延迟：**213.890 ms**
- P50 端到端延迟：**57.907 ms**
- P95 端到端延迟：**101.805 ms**
- P99 端到端延迟：**293.919 ms**

### 一级 Breakdown 占比

| 部分 | 时间 | 占比 |
|---|---|---|
| rdma_ns | 285142.919 ms | 59.32% |
| gpu_ns | 137728.310 ms | 28.65% |
| cpu_ns | 33570.858 ms | 6.98% |
| transfer_ns | 24260.238 ms | 5.05% |

- query 一级热点：占比最高的几项是 `rdma_ns`（59.32%）、`gpu_ns`（28.65%）、`cpu_ns`（6.98%）。

### Sub Breakdown 细分占比

#### cpu_ns

| 部分 | 时间 | 占比 |
|---|---|---|
| cpu_query_stage_candidates_ns | 20255.311 ms | 60.34% |
| cpu_query_filter_ns | 4960.625 ms | 14.78% |
| cpu_query_runtime_overhead_ns | 2471.492 ms | 7.36% |
| cpu_query_finalize_ns | 1270.341 ms | 3.78% |
| cpu_query_result_ids_ns | 1268.553 ms | 3.78% |
| cpu_cache_lookup_ns | 1108.857 ms | 3.30% |
| cpu_query_rerank_prepare_ns | 1027.017 ms | 3.06% |
| cpu_query_beam_update_ns | 890.162 ms | 2.65% |
| cpu_query_select_ns | 276.864 ms | 0.82% |
| cpu_query_beam_sort_ns | 35.690 ms | 0.11% |
| cpu_query_rerank_collect_ns | 4.730 ms | 0.01% |
| cpu_query_rerank_update_ns | 1.217 ms | 0.00% |

- cpu_ns 内部热点：占比最高的几项是 `cpu_query_stage_candidates_ns`（60.34%）、`cpu_query_filter_ns`（14.78%）、`cpu_query_runtime_overhead_ns`（7.36%）。

#### gpu_ns

| 部分 | 时间 | 占比 |
|---|---|---|
| gpu_query_distance_ns | 135464.485 ms | 98.36% |
| gpu_query_rerank_ns | 1156.197 ms | 0.84% |
| gpu_query_prepare_ns | 1107.628 ms | 0.80% |

- gpu_ns 内部热点：占比最高的几项是 `gpu_query_distance_ns`（98.36%）、`gpu_query_rerank_ns`（0.84%）、`gpu_query_prepare_ns`（0.80%）。

#### rdma_ns

| 部分 | 时间 | 占比 |
|---|---|---|
| rdma_rabitq_fetch_ns | 161464.950 ms | 56.63% |
| rdma_neighbor_fetch_ns | 120897.314 ms | 42.40% |
| rdma_rerank_fetch_ns | 1797.766 ms | 0.63% |
| rdma_medoid_ptr_ns | 982.888 ms | 0.34% |

- rdma_ns 内部热点：占比最高的几项是 `rdma_rabitq_fetch_ns`（56.63%）、`rdma_neighbor_fetch_ns`（42.40%）、`rdma_rerank_fetch_ns`（0.63%）。

#### transfer_ns

| 部分 | 时间 | 占比 |
|---|---|---|
| transfer_distance_d2h_ns | 16768.416 ms | 69.12% |
| transfer_rabitq_h2d_ns | 7256.595 ms | 29.91% |
| transfer_rerank_d2h_ns | 125.209 ms | 0.52% |
| transfer_query_h2d_ns | 55.917 ms | 0.23% |
| transfer_rerank_h2d_ns | 54.101 ms | 0.22% |

- transfer_ns 内部热点：占比最高的几项是 `transfer_distance_d2h_ns`（69.12%）、`transfer_rabitq_h2d_ns`（29.91%）、`transfer_rerank_d2h_ns`（0.52%）。

### Latency

| 延迟字段 | 值 |
|---|---|
| end_to_end_ns | 1911109.911 ms |
| mean_end_to_end_ns | 213.890 ms |
| mean_queue_wait_ns | 160.090 ms |
| mean_service_ns | 53.800 ms |
| p50_end_to_end_ns | 57.907 ms |
| p50_service_ns | 50.266 ms |
| p95_end_to_end_ns | 101.805 ms |
| p95_service_ns | 86.737 ms |
| p99_end_to_end_ns | 293.919 ms |
| p99_service_ns | 107.421 ms |
| queue_wait_ns | 1430407.586 ms |
| service_ns | 480702.325 ms |

### Counters

| 字段 | 值 |
|---|---|
| rdma_read_bytes | 93,086,965,110 |
| h2d_bytes | 90,519,620,048 |
| rabitq_rdma_bytes | 76,324,424,800 |
| vector_rdma_bytes | 14,220,090,160 |
| neighbor_rdma_bytes | 2,542,164,318 |
| d2h_bytes | 600,831,428 |
| visited_neighborlists | 4,955,483 |
| rabitq_kernels | 4,893,167 |
| cache_hits | 351,359 |
| cache_misses | 41,651 |
| exact_reranks | 35,728 |
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
| rdma_read_bytes | 360,007,732,212 |
| h2d_bytes | 357,104,313,680 |
| d2h_bytes | 941,625,744 |
| rdma_write_bytes | 217,695,987 |

## Insert / Query 对比

| 类别 | Insert 占比 | Query 占比 |
|---|---|---|
| cpu_ns | 5.07% | 6.98% |
| gpu_ns | 70.12% | 28.65% |
| rdma_ns | 21.72% | 59.32% |
| transfer_ns | 3.10% | 5.05% |

- Insert 最大部分是 **gpu_ns**，占 **70.12%**。
- Query 最大部分是 **rdma_ns**，占 **59.32%**。
- Insert 更偏向 GPU 计算密集。
- Query 更偏向 RDMA / 远端访问受限。
