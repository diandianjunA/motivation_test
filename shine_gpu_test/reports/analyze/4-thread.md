# Breakdown 分析报告

## 实验元信息

- **client_threads**: 4
- **coroutines**: 4
- **dim**: 1,024
- **measure_mixed**
  - **completed_reads**: 8,515
  - **completed_writes**: 597
  - **issued_reads**: 8,515
  - **issued_writes**: 597
- **measure_ops**: 1,000
- **measure_seconds**: 60
- **operation_granularity**: single_vector
- **read_ratio**: 0.5
- **run_mode**: time
- **search_mode**: rabitq_gpu
- **synthetic_query_vectors**: 16,384
- **threads**: 4
- **time_completion_policy**: drain
- **time_issue_policy**: bounded_by_observed_call_latency
- **warmup_mixed**
  - **completed_reads**: 3,701
  - **completed_writes**: 513
  - **issued_reads**: 3,701
  - **issued_writes**: 513
- **warmup_ops**: 100
- **warmup_seconds**: 30
- **workload**: mixed

## Bottleneck Summary

### insert

```text
insert breakdown
  count: 597
  latency_ms: mean=200.539 p50=229.404 p95=266.58 p99=288.997
  top_categories:
    gpu_ns: 101522 ms (84.8002%)
    rdma_ns: 8354.93 ms (6.97875%)
    cpu_ns: 4976.68 ms (4.15695%)
    transfer_ns: 4865.55 ms (4.06412%)
```

### query

```text
query breakdown
  count: 8515
  latency_ms: mean=14.0706 p50=13.3298 p95=22.5096 p99=27.3029
  top_categories:
    gpu_ns: 50792.7 ms (42.4047%)
    rdma_ns: 33633.7 ms (28.0794%)
    transfer_ns: 21685.9 ms (18.1046%)
    cpu_ns: 13668.5 ms (11.4113%)
```

## INSERT 分析

- 操作数：**597**
- 平均端到端延迟：**200.539 ms**
- P50 端到端延迟：**229.404 ms**
- P95 端到端延迟：**266.580 ms**
- P99 端到端延迟：**288.997 ms**

### 一级 Breakdown 占比

| 部分 | 时间 | 占比 |
|---|---|---|
| gpu_ns | 101522.431 ms | 84.80% |
| rdma_ns | 8354.930 ms | 6.98% |
| cpu_ns | 4976.683 ms | 4.16% |
| transfer_ns | 4865.550 ms | 4.06% |

- insert 一级热点：占比最高的几项是 `gpu_ns`（84.80%）、`rdma_ns`（6.98%）、`cpu_ns`（4.16%）。

### Sub Breakdown 细分占比

#### cpu_ns

| 部分 | 时间 | 占比 |
|---|---|---|
| cpu_insert_stage_candidates_ns | 3245.703 ms | 65.22% |
| cpu_insert_runtime_overhead_ns | 635.914 ms | 12.78% |
| cpu_insert_filter_ns | 451.406 ms | 9.07% |
| cpu_insert_candidate_sort_ns | 206.097 ms | 4.14% |
| cpu_insert_beam_update_ns | 165.628 ms | 3.33% |
| cpu_insert_select_ns | 129.146 ms | 2.60% |
| cpu_insert_finalize_ns | 64.116 ms | 1.29% |
| cpu_insert_overflow_prepare_ns | 38.639 ms | 0.78% |
| cpu_insert_pruned_neighbor_collect_ns | 11.426 ms | 0.23% |
| cpu_insert_neighbor_collect_ns | 9.374 ms | 0.19% |
| cpu_insert_prune_prepare_ns | 7.288 ms | 0.15% |
| cpu_insert_init_ns | 5.192 ms | 0.10% |
| cpu_insert_preprune_sort_ns | 4.199 ms | 0.08% |
| cpu_insert_candidate_collect_ns | 1.954 ms | 0.04% |
| cpu_cache_lookup_ns | 0.444 ms | 0.01% |
| cpu_insert_neighbor_prepare_ns | 0.128 ms | 0.00% |
| cpu_insert_quantize_prepare_ns | 0.028 ms | 0.00% |

- cpu_ns 内部热点：占比最高的几项是 `cpu_insert_stage_candidates_ns`（65.22%）、`cpu_insert_runtime_overhead_ns`（12.78%）、`cpu_insert_filter_ns`（9.07%）。

#### gpu_ns

| 部分 | 时间 | 占比 |
|---|---|---|
| gpu_insert_overflow_prune_ns | 85492.969 ms | 84.21% |
| gpu_insert_prune_ns | 8882.534 ms | 8.75% |
| gpu_insert_distance_ns | 5406.557 ms | 5.33% |
| gpu_insert_overflow_distance_ns | 1577.253 ms | 1.55% |
| gpu_insert_quantize_ns | 163.118 ms | 0.16% |

- gpu_ns 内部热点：占比最高的几项是 `gpu_insert_overflow_prune_ns`（84.21%）、`gpu_insert_prune_ns`（8.75%）、`gpu_insert_distance_ns`（5.33%）。

#### rdma_ns

| 部分 | 时间 | 占比 |
|---|---|---|
| rdma_vector_fetch_ns | 4001.566 ms | 47.89% |
| rdma_neighbor_fetch_ns | 1824.382 ms | 21.84% |
| rdma_overflow_vec_fetch_ns | 1426.480 ms | 17.07% |
| rdma_pruned_neighbor_write_ns | 240.838 ms | 2.88% |
| rdma_neighbor_lock_ns | 236.864 ms | 2.84% |
| rdma_neighbor_list_read_ns | 181.178 ms | 2.17% |
| rdma_neighbor_node_read_ns | 171.952 ms | 2.06% |
| rdma_candidate_fetch_ns | 157.296 ms | 1.88% |
| rdma_neighbor_unlock_ns | 77.440 ms | 0.93% |
| rdma_neighbor_list_write_ns | 24.825 ms | 0.30% |
| rdma_new_node_write_ns | 5.016 ms | 0.06% |
| rdma_alloc_ns | 4.297 ms | 0.05% |
| rdma_medoid_ptr_ns | 2.798 ms | 0.03% |

- rdma_ns 内部热点：占比最高的几项是 `rdma_vector_fetch_ns`（47.89%）、`rdma_neighbor_fetch_ns`（21.84%）、`rdma_overflow_vec_fetch_ns`（17.07%）。

#### transfer_ns

| 部分 | 时间 | 占比 |
|---|---|---|
| transfer_distance_d2h_ns | 2384.452 ms | 49.01% |
| transfer_candidate_h2d_ns | 899.412 ms | 18.49% |
| transfer_overflow_prune_d2h_ns | 504.545 ms | 10.37% |
| transfer_overflow_dist_d2h_ns | 389.620 ms | 8.01% |
| transfer_overflow_prune_inputs_h2d_ns | 281.513 ms | 5.79% |
| transfer_overflow_query_h2d_ns | 166.436 ms | 3.42% |
| transfer_overflow_candidate_h2d_ns | 154.219 ms | 3.17% |
| transfer_quantize_d2h_ns | 61.824 ms | 1.27% |
| transfer_prune_d2h_ns | 11.808 ms | 0.24% |
| transfer_prune_h2d_ns | 7.289 ms | 0.15% |
| transfer_insert_query_h2d_ns | 4.432 ms | 0.09% |

- transfer_ns 内部热点：占比最高的几项是 `transfer_distance_d2h_ns`（49.01%）、`transfer_candidate_h2d_ns`（18.49%）、`transfer_overflow_prune_d2h_ns`（10.37%）。

### Latency

| 延迟字段 | 值 |
|---|---|
| end_to_end_ns | 119721.970 ms |
| mean_end_to_end_ns | 200.539 ms |
| mean_queue_wait_ns | 0.004 ms |
| mean_service_ns | 200.535 ms |
| p50_end_to_end_ns | 229.404 ms |
| p50_service_ns | 229.398 ms |
| p95_end_to_end_ns | 266.580 ms |
| p95_service_ns | 266.577 ms |
| p99_end_to_end_ns | 288.997 ms |
| p99_service_ns | 288.993 ms |
| queue_wait_ns | 2.376 ms |
| service_ns | 119719.594 ms |

### Counters

| 字段 | 值 |
|---|---|
| rdma_read_bytes | 27,568,728,744 |
| h2d_bytes | 27,506,272,344 |
| vector_rdma_bytes | 27,370,910,304 |
| neighbor_rdma_bytes | 197,812,800 |
| d2h_bytes | 34,485,796 |
| rdma_write_bytes | 22,262,044 |
| l2_kernels | 232,778 |
| lock_attempts | 73,180 |
| cas_failures | 42,210 |
| lock_retries | 42,210 |
| prune_kernels | 32,807 |
| overflow_prunes | 32,107 |
| cache_hits | 705 |
| remote_allocations | 695 |
| cache_misses | 0 |
| exact_reranks | 0 |
| rabitq_kernels | 0 |
| rabitq_rdma_bytes | 0 |
| visited_neighborlists | 0 |
| visited_nodes | 0 |

## QUERY 分析

- 操作数：**8,515**
- 平均端到端延迟：**14.071 ms**
- P50 端到端延迟：**13.330 ms**
- P95 端到端延迟：**22.510 ms**
- P99 端到端延迟：**27.303 ms**

### 一级 Breakdown 占比

| 部分 | 时间 | 占比 |
|---|---|---|
| gpu_ns | 50792.676 ms | 42.40% |
| rdma_ns | 33633.704 ms | 28.08% |
| transfer_ns | 21685.867 ms | 18.10% |
| cpu_ns | 13668.523 ms | 11.41% |

- query 一级热点：占比最高的几项是 `gpu_ns`（42.40%）、`rdma_ns`（28.08%）、`transfer_ns`（18.10%）。

### Sub Breakdown 细分占比

#### cpu_ns

| 部分 | 时间 | 占比 |
|---|---|---|
| cpu_query_stage_candidates_ns | 7317.093 ms | 53.53% |
| cpu_query_filter_ns | 2959.148 ms | 21.65% |
| cpu_query_runtime_overhead_ns | 1331.251 ms | 9.74% |
| cpu_query_beam_update_ns | 794.909 ms | 5.82% |
| cpu_query_rerank_prepare_ns | 517.777 ms | 3.79% |
| cpu_query_select_ns | 277.603 ms | 2.03% |
| cpu_query_finalize_ns | 154.767 ms | 1.13% |
| cpu_query_result_ids_ns | 152.823 ms | 1.12% |
| cpu_cache_lookup_ns | 125.054 ms | 0.91% |
| cpu_query_beam_sort_ns | 33.190 ms | 0.24% |
| cpu_query_rerank_collect_ns | 3.899 ms | 0.03% |
| cpu_query_rerank_update_ns | 1.009 ms | 0.01% |

- cpu_ns 内部热点：占比最高的几项是 `cpu_query_stage_candidates_ns`（53.53%）、`cpu_query_filter_ns`（21.65%）、`cpu_query_runtime_overhead_ns`（9.74%）。

#### gpu_ns

| 部分 | 时间 | 占比 |
|---|---|---|
| gpu_query_distance_ns | 49952.901 ms | 98.35% |
| gpu_query_rerank_ns | 444.679 ms | 0.88% |
| gpu_query_prepare_ns | 395.095 ms | 0.78% |

- gpu_ns 内部热点：占比最高的几项是 `gpu_query_distance_ns`（98.35%）、`gpu_query_rerank_ns`（0.88%）、`gpu_query_prepare_ns`（0.78%）。

#### rdma_ns

| 部分 | 时间 | 占比 |
|---|---|---|
| rdma_rabitq_fetch_ns | 25815.358 ms | 76.75% |
| rdma_neighbor_fetch_ns | 7294.537 ms | 21.69% |
| rdma_rerank_fetch_ns | 484.389 ms | 1.44% |
| rdma_medoid_ptr_ns | 39.420 ms | 0.12% |

- rdma_ns 内部热点：占比最高的几项是 `rdma_rabitq_fetch_ns`（76.75%）、`rdma_neighbor_fetch_ns`（21.69%）、`rdma_rerank_fetch_ns`（1.44%）。

#### transfer_ns

| 部分 | 时间 | 占比 |
|---|---|---|
| transfer_distance_d2h_ns | 15092.783 ms | 69.60% |
| transfer_rabitq_h2d_ns | 6359.291 ms | 29.32% |
| transfer_rerank_d2h_ns | 121.424 ms | 0.56% |
| transfer_query_h2d_ns | 59.712 ms | 0.28% |
| transfer_rerank_h2d_ns | 52.656 ms | 0.24% |

- transfer_ns 内部热点：占比最高的几项是 `transfer_distance_d2h_ns`（69.60%）、`transfer_rabitq_h2d_ns`（29.32%）、`transfer_rerank_d2h_ns`（0.56%）。

### Latency

| 延迟字段 | 值 |
|---|---|
| end_to_end_ns | 119811.120 ms |
| mean_end_to_end_ns | 14.071 ms |
| mean_queue_wait_ns | 0.004 ms |
| mean_service_ns | 14.067 ms |
| p50_end_to_end_ns | 13.330 ms |
| p50_service_ns | 13.326 ms |
| p95_end_to_end_ns | 22.510 ms |
| p95_service_ns | 22.506 ms |
| p99_end_to_end_ns | 27.303 ms |
| p99_service_ns | 27.300 ms |
| queue_wait_ns | 30.350 ms |
| service_ns | 119780.770 ms |

### Counters

| 字段 | 值 |
|---|---|
| rdma_read_bytes | 23,173,329,733 |
| h2d_bytes | 22,507,478,008 |
| rabitq_rdma_bytes | 18,809,020,880 |
| vector_rdma_bytes | 3,703,422,208 |
| neighbor_rdma_bytes | 660,812,229 |
| d2h_bytes | 148,292,524 |
| visited_neighborlists | 1,287,826 |
| rabitq_kernels | 1,174,660 |
| cache_hits | 91,122 |
| cache_misses | 11,216 |
| exact_reranks | 9,304 |
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
| rdma_read_bytes | 50,742,058,477 |
| h2d_bytes | 50,013,750,352 |
| d2h_bytes | 182,778,320 |
| rdma_write_bytes | 22,262,044 |

## Insert / Query 对比

| 类别 | Insert 占比 | Query 占比 |
|---|---|---|
| cpu_ns | 4.16% | 11.41% |
| gpu_ns | 84.80% | 42.40% |
| rdma_ns | 6.98% | 28.08% |
| transfer_ns | 4.06% | 18.10% |

- Insert 最大部分是 **gpu_ns**，占 **84.80%**。
- Query 最大部分是 **gpu_ns**，占 **42.40%**。
- Insert 更偏向 GPU 计算密集。
- Query 更偏向 RDMA / 远端访问受限。
