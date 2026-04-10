# Breakdown 分析报告

## 实验元信息

- **client_threads**: 8
- **coroutines**: 4
- **dim**: 1,024
- **measure_mixed**
  - **completed_reads**: 9,782
  - **completed_writes**: 1,029
  - **issued_reads**: 9,782
  - **issued_writes**: 1,029
- **measure_ops**: 1,000
- **measure_seconds**: 60
- **operation_granularity**: single_vector
- **read_ratio**: 0.5
- **run_mode**: time
- **search_mode**: rabitq_gpu
- **synthetic_query_vectors**: 32,768
- **threads**: 4
- **time_completion_policy**: drain
- **time_issue_policy**: bounded_by_observed_call_latency
- **warmup_mixed**
  - **completed_reads**: 4,005
  - **completed_writes**: 650
  - **issued_reads**: 4,005
  - **issued_writes**: 650
- **warmup_ops**: 100
- **warmup_seconds**: 30
- **workload**: mixed

## Bottleneck Summary

### insert

```text
insert breakdown
  count: 1029
  latency_ms: mean=232.706 p50=254.995 p95=306.116 p99=330.482
  top_categories:
    gpu_ns: 191814 ms (80.1066%)
    rdma_ns: 26835.7 ms (11.2073%)
    cpu_ns: 11440.5 ms (4.77787%)
    transfer_ns: 9358.1 ms (3.9082%)
```

### query

```text
query breakdown
  count: 9782
  latency_ms: mean=24.5085 p50=23.1322 p95=39.8137 p99=49.8072
  top_categories:
    rdma_ns: 107359 ms (44.8035%)
    gpu_ns: 81419.9 ms (33.9785%)
    transfer_ns: 27557.8 ms (11.5006%)
    cpu_ns: 23284.9 ms (9.71738%)
```

## INSERT 分析

- 操作数：**1,029**
- 平均端到端延迟：**232.706 ms**
- P50 端到端延迟：**254.995 ms**
- P95 端到端延迟：**306.116 ms**
- P99 端到端延迟：**330.482 ms**

### 一级 Breakdown 占比

| 部分 | 时间 | 占比 |
|---|---|---|
| gpu_ns | 191813.575 ms | 80.11% |
| rdma_ns | 26835.706 ms | 11.21% |
| cpu_ns | 11440.499 ms | 4.78% |
| transfer_ns | 9358.095 ms | 3.91% |

- insert 一级热点：占比最高的几项是 `gpu_ns`（80.11%）、`rdma_ns`（11.21%）、`cpu_ns`（4.78%）。

### Sub Breakdown 细分占比

#### cpu_ns

| 部分 | 时间 | 占比 |
|---|---|---|
| cpu_insert_stage_candidates_ns | 7559.248 ms | 66.07% |
| cpu_insert_runtime_overhead_ns | 1752.356 ms | 15.32% |
| cpu_insert_filter_ns | 819.770 ms | 7.17% |
| cpu_insert_candidate_sort_ns | 522.036 ms | 4.56% |
| cpu_insert_beam_update_ns | 294.596 ms | 2.58% |
| cpu_insert_select_ns | 224.134 ms | 1.96% |
| cpu_insert_finalize_ns | 112.956 ms | 0.99% |
| cpu_insert_overflow_prepare_ns | 76.627 ms | 0.67% |
| cpu_insert_pruned_neighbor_collect_ns | 21.106 ms | 0.18% |
| cpu_insert_neighbor_collect_ns | 19.917 ms | 0.17% |
| cpu_insert_prune_prepare_ns | 13.221 ms | 0.12% |
| cpu_insert_init_ns | 8.844 ms | 0.08% |
| cpu_insert_candidate_collect_ns | 7.502 ms | 0.07% |
| cpu_insert_preprune_sort_ns | 7.158 ms | 0.06% |
| cpu_cache_lookup_ns | 0.785 ms | 0.01% |
| cpu_insert_neighbor_prepare_ns | 0.199 ms | 0.00% |
| cpu_insert_quantize_prepare_ns | 0.046 ms | 0.00% |

- cpu_ns 内部热点：占比最高的几项是 `cpu_insert_stage_candidates_ns`（66.07%）、`cpu_insert_runtime_overhead_ns`（15.32%）、`cpu_insert_filter_ns`（7.17%）。

#### gpu_ns

| 部分 | 时间 | 占比 |
|---|---|---|
| gpu_insert_overflow_prune_ns | 160194.168 ms | 83.52% |
| gpu_insert_prune_ns | 15948.400 ms | 8.31% |
| gpu_insert_distance_ns | 11916.741 ms | 6.21% |
| gpu_insert_overflow_distance_ns | 3464.931 ms | 1.81% |
| gpu_insert_quantize_ns | 289.335 ms | 0.15% |

- gpu_ns 内部热点：占比最高的几项是 `gpu_insert_overflow_prune_ns`（83.52%）、`gpu_insert_prune_ns`（8.31%）、`gpu_insert_distance_ns`（6.21%）。

#### rdma_ns

| 部分 | 时间 | 占比 |
|---|---|---|
| rdma_vector_fetch_ns | 11018.709 ms | 41.06% |
| rdma_neighbor_fetch_ns | 6877.569 ms | 25.63% |
| rdma_overflow_vec_fetch_ns | 4248.117 ms | 15.83% |
| rdma_neighbor_lock_ns | 1043.517 ms | 3.89% |
| rdma_pruned_neighbor_write_ns | 1031.782 ms | 3.84% |
| rdma_neighbor_list_read_ns | 798.285 ms | 2.97% |
| rdma_neighbor_node_read_ns | 777.668 ms | 2.90% |
| rdma_neighbor_unlock_ns | 504.730 ms | 1.88% |
| rdma_candidate_fetch_ns | 420.235 ms | 1.57% |
| rdma_neighbor_list_write_ns | 71.205 ms | 0.27% |
| rdma_alloc_ns | 17.484 ms | 0.07% |
| rdma_new_node_write_ns | 15.366 ms | 0.06% |
| rdma_medoid_ptr_ns | 11.038 ms | 0.04% |

- rdma_ns 内部热点：占比最高的几项是 `rdma_vector_fetch_ns`（41.06%）、`rdma_neighbor_fetch_ns`（25.63%）、`rdma_overflow_vec_fetch_ns`（15.83%）。

#### transfer_ns

| 部分 | 时间 | 占比 |
|---|---|---|
| transfer_distance_d2h_ns | 4629.555 ms | 49.47% |
| transfer_candidate_h2d_ns | 1646.926 ms | 17.60% |
| transfer_overflow_prune_d2h_ns | 992.743 ms | 10.61% |
| transfer_overflow_dist_d2h_ns | 790.718 ms | 8.45% |
| transfer_overflow_prune_inputs_h2d_ns | 540.926 ms | 5.78% |
| transfer_overflow_query_h2d_ns | 306.635 ms | 3.28% |
| transfer_overflow_candidate_h2d_ns | 301.007 ms | 3.22% |
| transfer_quantize_d2h_ns | 106.742 ms | 1.14% |
| transfer_prune_d2h_ns | 21.730 ms | 0.23% |
| transfer_prune_h2d_ns | 13.222 ms | 0.14% |
| transfer_insert_query_h2d_ns | 7.891 ms | 0.08% |

- transfer_ns 内部热点：占比最高的几项是 `transfer_distance_d2h_ns`（49.47%）、`transfer_candidate_h2d_ns`（17.60%）、`transfer_overflow_prune_d2h_ns`（10.61%）。

### Latency

| 延迟字段 | 值 |
|---|---|
| end_to_end_ns | 239454.049 ms |
| mean_end_to_end_ns | 232.706 ms |
| mean_queue_wait_ns | 0.006 ms |
| mean_service_ns | 232.700 ms |
| p50_end_to_end_ns | 254.995 ms |
| p50_service_ns | 254.991 ms |
| p95_end_to_end_ns | 306.116 ms |
| p95_service_ns | 306.110 ms |
| p99_end_to_end_ns | 330.482 ms |
| p99_service_ns | 330.475 ms |
| queue_wait_ns | 6.174 ms |
| service_ns | 239447.875 ms |

### Counters

| 字段 | 值 |
|---|---|
| rdma_read_bytes | 88,651,908,022 |
| h2d_bytes | 88,503,845,032 |
| vector_rdma_bytes | 88,020,765,296 |
| neighbor_rdma_bytes | 631,124,406 |
| d2h_bytes | 113,189,912 |
| rdma_write_bytes | 74,676,730 |
| l2_kernels | 777,955 |
| lock_attempts | 128,062 |
| prune_kernels | 114,009 |
| overflow_prunes | 111,786 |
| cas_failures | 72,721 |
| lock_retries | 72,721 |
| cache_hits | 2,292 |
| remote_allocations | 2,218 |
| cache_misses | 0 |
| exact_reranks | 0 |
| rabitq_kernels | 0 |
| rabitq_rdma_bytes | 0 |
| visited_neighborlists | 0 |
| visited_nodes | 0 |

## QUERY 分析

- 操作数：**9,782**
- 平均端到端延迟：**24.509 ms**
- P50 端到端延迟：**23.132 ms**
- P95 端到端延迟：**39.814 ms**
- P99 端到端延迟：**49.807 ms**

### 一级 Breakdown 占比

| 部分 | 时间 | 占比 |
|---|---|---|
| rdma_ns | 107358.874 ms | 44.80% |
| gpu_ns | 81419.923 ms | 33.98% |
| transfer_ns | 27557.822 ms | 11.50% |
| cpu_ns | 23284.935 ms | 9.72% |

- query 一级热点：占比最高的几项是 `rdma_ns`（44.80%）、`gpu_ns`（33.98%）、`transfer_ns`（11.50%）。

### Sub Breakdown 细分占比

#### cpu_ns

| 部分 | 时间 | 占比 |
|---|---|---|
| cpu_query_stage_candidates_ns | 13903.878 ms | 59.71% |
| cpu_query_filter_ns | 3670.766 ms | 15.76% |
| cpu_query_runtime_overhead_ns | 2021.134 ms | 8.68% |
| cpu_query_beam_update_ns | 942.844 ms | 4.05% |
| cpu_query_rerank_prepare_ns | 773.198 ms | 3.32% |
| cpu_query_finalize_ns | 610.736 ms | 2.62% |
| cpu_query_result_ids_ns | 608.823 ms | 2.61% |
| cpu_cache_lookup_ns | 405.701 ms | 1.74% |
| cpu_query_select_ns | 303.055 ms | 1.30% |
| cpu_query_beam_sort_ns | 38.966 ms | 0.17% |
| cpu_query_rerank_collect_ns | 4.594 ms | 0.02% |
| cpu_query_rerank_update_ns | 1.239 ms | 0.01% |

- cpu_ns 内部热点：占比最高的几项是 `cpu_query_stage_candidates_ns`（59.71%）、`cpu_query_filter_ns`（15.76%）、`cpu_query_runtime_overhead_ns`（8.68%）。

#### gpu_ns

| 部分 | 时间 | 占比 |
|---|---|---|
| gpu_query_distance_ns | 80084.291 ms | 98.36% |
| gpu_query_rerank_ns | 706.210 ms | 0.87% |
| gpu_query_prepare_ns | 629.422 ms | 0.77% |

- gpu_ns 内部热点：占比最高的几项是 `gpu_query_distance_ns`（98.36%）、`gpu_query_rerank_ns`（0.87%）、`gpu_query_prepare_ns`（0.77%）。

#### rdma_ns

| 部分 | 时间 | 占比 |
|---|---|---|
| rdma_rabitq_fetch_ns | 64616.904 ms | 60.19% |
| rdma_neighbor_fetch_ns | 41537.147 ms | 38.69% |
| rdma_rerank_fetch_ns | 987.606 ms | 0.92% |
| rdma_medoid_ptr_ns | 217.217 ms | 0.20% |

- rdma_ns 内部热点：占比最高的几项是 `rdma_rabitq_fetch_ns`（60.19%）、`rdma_neighbor_fetch_ns`（38.69%）、`rdma_rerank_fetch_ns`（0.92%）。

#### transfer_ns

| 部分 | 时间 | 占比 |
|---|---|---|
| transfer_distance_d2h_ns | 19470.600 ms | 70.65% |
| transfer_rabitq_h2d_ns | 7798.836 ms | 28.30% |
| transfer_rerank_d2h_ns | 149.482 ms | 0.54% |
| transfer_query_h2d_ns | 74.060 ms | 0.27% |
| transfer_rerank_h2d_ns | 64.843 ms | 0.24% |

- transfer_ns 内部热点：占比最高的几项是 `transfer_distance_d2h_ns`（70.65%）、`transfer_rabitq_h2d_ns`（28.30%）、`transfer_rerank_d2h_ns`（0.54%）。

### Latency

| 延迟字段 | 值 |
|---|---|
| end_to_end_ns | 239742.515 ms |
| mean_end_to_end_ns | 24.509 ms |
| mean_queue_wait_ns | 0.012 ms |
| mean_service_ns | 24.496 ms |
| p50_end_to_end_ns | 23.132 ms |
| p50_service_ns | 23.122 ms |
| p95_end_to_end_ns | 39.814 ms |
| p95_service_ns | 39.810 ms |
| p99_end_to_end_ns | 49.807 ms |
| p99_service_ns | 49.777 ms |
| queue_wait_ns | 120.962 ms |
| service_ns | 239621.554 ms |

### Counters

| 字段 | 值 |
|---|---|
| rdma_read_bytes | 51,082,729,355 |
| h2d_bytes | 49,616,009,912 |
| rabitq_rdma_bytes | 41,464,365,280 |
| vector_rdma_bytes | 8,162,354,144 |
| neighbor_rdma_bytes | 1,455,846,291 |
| d2h_bytes | 327,190,016 |
| visited_neighborlists | 2,834,521 |
| rabitq_kernels | 2,704,324 |
| cache_hits | 201,660 |
| cache_misses | 23,602 |
| exact_reranks | 20,487 |
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
| rdma_read_bytes | 139,734,637,377 |
| h2d_bytes | 138,119,854,944 |
| d2h_bytes | 440,379,928 |
| rdma_write_bytes | 74,676,730 |

## Insert / Query 对比

| 类别 | Insert 占比 | Query 占比 |
|---|---|---|
| cpu_ns | 4.78% | 9.72% |
| gpu_ns | 80.11% | 33.98% |
| rdma_ns | 11.21% | 44.80% |
| transfer_ns | 3.91% | 11.50% |

- Insert 最大部分是 **gpu_ns**，占 **80.11%**。
- Query 最大部分是 **rdma_ns**，占 **44.80%**。
- Insert 更偏向 GPU 计算密集。
- Query 更偏向 RDMA / 远端访问受限。
