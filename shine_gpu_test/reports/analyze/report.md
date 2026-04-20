# Breakdown 分析报告

## 实验元信息

- **client_threads**: 16
- **coroutines**: 4
- **dim**: 1,024
- **measure_mixed**
  - **completed_reads**: 1,680
  - **completed_writes**: 387
  - **issued_reads**: 1,680
  - **issued_writes**: 387
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
  - **completed_reads**: 876
  - **completed_writes**: 221
  - **issued_reads**: 876
  - **issued_writes**: 221
- **warmup_ops**: 100
- **warmup_seconds**: 30
- **workload**: mixed

## Bottleneck Summary

### insert

```text
insert breakdown
  count: 387
  latency_ms: mean=1223.62 p50=1151.16 p95=1940.61 p99=2140.47
  top_categories:
    gpu_ns: 406384 ms (85.8204%)
    rdma_ns: 54428.3 ms (11.4942%)
    cpu_ns: 8249.75 ms (1.74219%)
    transfer_ns: 4466.25 ms (0.943185%)
```

### query

```text
query breakdown
  count: 1680
  latency_ms: mean=285.47 p50=275.975 p95=479.782 p99=595.323
  top_categories:
    gpu_ns: 394440 ms (82.2708%)
    rdma_ns: 75088.7 ms (15.6617%)
    cpu_ns: 5531.64 ms (1.15377%)
    transfer_ns: 4380.47 ms (0.913664%)
```

## INSERT 分析

- 操作数：**387**
- 平均端到端延迟：**1223.623 ms**
- P50 端到端延迟：**1151.159 ms**
- P95 端到端延迟：**1940.608 ms**
- P99 端到端延迟：**2140.472 ms**

### 一级 Breakdown 占比

| 部分 | 时间 | 占比 |
|---|---|---|
| gpu_ns | 406383.738 ms | 85.82% |
| rdma_ns | 54428.342 ms | 11.49% |
| cpu_ns | 8249.750 ms | 1.74% |
| transfer_ns | 4466.245 ms | 0.94% |

- insert 一级热点：占比最高的几项是 `gpu_ns`（85.82%）、`rdma_ns`（11.49%）、`cpu_ns`（1.74%）。

### Sub Breakdown 细分占比

#### cpu_ns

| 部分 | 时间 | 占比 |
|---|---|---|
| cpu_insert_stage_candidates_ns | 6583.995 ms | 79.81% |
| cpu_insert_filter_ns | 636.994 ms | 7.72% |
| cpu_insert_runtime_overhead_ns | 518.278 ms | 6.28% |
| cpu_insert_beam_update_ns | 142.314 ms | 1.73% |
| cpu_insert_candidate_sort_ns | 121.166 ms | 1.47% |
| cpu_insert_select_ns | 103.845 ms | 1.26% |
| cpu_insert_finalize_ns | 102.036 ms | 1.24% |
| cpu_insert_overflow_prepare_ns | 18.432 ms | 0.22% |
| cpu_insert_prune_prepare_ns | 5.557 ms | 0.07% |
| cpu_insert_pruned_neighbor_collect_ns | 4.936 ms | 0.06% |
| cpu_insert_neighbor_collect_ns | 3.809 ms | 0.05% |
| cpu_insert_init_ns | 3.545 ms | 0.04% |
| cpu_insert_preprune_sort_ns | 2.866 ms | 0.03% |
| cpu_insert_candidate_collect_ns | 1.407 ms | 0.02% |
| cpu_cache_lookup_ns | 0.309 ms | 0.00% |
| cpu_insert_neighbor_prepare_ns | 0.240 ms | 0.00% |
| cpu_insert_quantize_prepare_ns | 0.019 ms | 0.00% |

- cpu_ns 内部热点：占比最高的几项是 `cpu_insert_stage_candidates_ns`（79.81%）、`cpu_insert_filter_ns`（7.72%）、`cpu_insert_runtime_overhead_ns`（6.28%）。

#### gpu_ns

| 部分 | 时间 | 占比 |
|---|---|---|
| gpu_insert_distance_ns | 306279.223 ms | 75.37% |
| gpu_insert_overflow_prune_ns | 63950.594 ms | 15.74% |
| gpu_insert_overflow_distance_ns | 27954.686 ms | 6.88% |
| gpu_insert_prune_ns | 7457.837 ms | 1.84% |
| gpu_insert_quantize_ns | 741.399 ms | 0.18% |

- gpu_ns 内部热点：占比最高的几项是 `gpu_insert_distance_ns`（75.37%）、`gpu_insert_overflow_prune_ns`（15.74%）、`gpu_insert_overflow_distance_ns`（6.88%）。

#### rdma_ns

| 部分 | 时间 | 占比 |
|---|---|---|
| rdma_vector_fetch_ns | 30970.255 ms | 56.90% |
| rdma_neighbor_fetch_ns | 19415.428 ms | 35.67% |
| rdma_neighbor_lock_ns | 1056.549 ms | 1.94% |
| rdma_overflow_vec_fetch_ns | 606.975 ms | 1.12% |
| rdma_neighbor_list_read_ns | 526.997 ms | 0.97% |
| rdma_neighbor_node_read_ns | 503.169 ms | 0.92% |
| rdma_neighbor_unlock_ns | 490.340 ms | 0.90% |
| rdma_pruned_neighbor_write_ns | 302.404 ms | 0.56% |
| rdma_candidate_fetch_ns | 236.792 ms | 0.44% |
| rdma_neighbor_list_write_ns | 224.331 ms | 0.41% |
| rdma_alloc_ns | 48.522 ms | 0.09% |
| rdma_new_node_write_ns | 28.075 ms | 0.05% |
| rdma_medoid_ptr_ns | 18.504 ms | 0.03% |

- rdma_ns 内部热点：占比最高的几项是 `rdma_vector_fetch_ns`（56.90%）、`rdma_neighbor_fetch_ns`（35.67%）、`rdma_neighbor_lock_ns`（1.94%）。

#### transfer_ns

| 部分 | 时间 | 占比 |
|---|---|---|
| transfer_distance_d2h_ns | 2912.200 ms | 65.20% |
| transfer_candidate_h2d_ns | 882.801 ms | 19.77% |
| transfer_overflow_prune_d2h_ns | 197.169 ms | 4.41% |
| transfer_overflow_dist_d2h_ns | 166.437 ms | 3.73% |
| transfer_overflow_prune_inputs_h2d_ns | 101.398 ms | 2.27% |
| transfer_overflow_query_h2d_ns | 70.244 ms | 1.57% |
| transfer_overflow_candidate_h2d_ns | 65.441 ms | 1.47% |
| transfer_quantize_d2h_ns | 53.486 ms | 1.20% |
| transfer_prune_d2h_ns | 8.395 ms | 0.19% |
| transfer_prune_h2d_ns | 5.558 ms | 0.12% |
| transfer_insert_query_h2d_ns | 3.116 ms | 0.07% |

- transfer_ns 内部热点：占比最高的几项是 `transfer_distance_d2h_ns`（65.20%）、`transfer_candidate_h2d_ns`（19.77%）、`transfer_overflow_prune_d2h_ns`（4.41%）。

### Latency

| 延迟字段 | 值 |
|---|---|
| end_to_end_ns | 473541.929 ms |
| mean_end_to_end_ns | 1223.623 ms |
| mean_queue_wait_ns | 0.036 ms |
| mean_service_ns | 1223.587 ms |
| p50_end_to_end_ns | 1151.159 ms |
| p50_service_ns | 1151.103 ms |
| p95_end_to_end_ns | 1940.608 ms |
| p95_service_ns | 1940.594 ms |
| p99_end_to_end_ns | 2140.472 ms |
| p99_service_ns | 2140.446 ms |
| queue_wait_ns | 13.853 ms |
| service_ns | 473528.075 ms |

### Counters

| 字段 | 值 |
|---|---|
| rdma_read_bytes | 77,139,705,031 |
| h2d_bytes | 76,790,091,176 |
| vector_rdma_bytes | 76,703,594,000 |
| neighbor_rdma_bytes | 436,098,735 |
| d2h_bytes | 83,066,768 |
| rdma_write_bytes | 39,880,551 |
| l2_kernels | 677,846 |
| prune_kernels | 38,148 |
| overflow_prunes | 36,602 |
| lock_attempts | 35,570 |
| cas_failures | 20,023 |
| lock_retries | 20,023 |
| remote_allocations | 1,543 |
| cache_hits | 1,540 |
| cache_misses | 0 |
| exact_reranks | 0 |
| rabitq_kernels | 0 |
| rabitq_rdma_bytes | 0 |
| visited_neighborlists | 0 |
| visited_nodes | 0 |

## QUERY 分析

- 操作数：**1,680**
- 平均端到端延迟：**285.470 ms**
- P50 端到端延迟：**275.975 ms**
- P95 端到端延迟：**479.782 ms**
- P99 端到端延迟：**595.323 ms**

### 一级 Breakdown 占比

| 部分 | 时间 | 占比 |
|---|---|---|
| gpu_ns | 394439.566 ms | 82.27% |
| rdma_ns | 75088.710 ms | 15.66% |
| cpu_ns | 5531.638 ms | 1.15% |
| transfer_ns | 4380.475 ms | 0.91% |

- query 一级热点：占比最高的几项是 `gpu_ns`（82.27%）、`rdma_ns`（15.66%）、`cpu_ns`（1.15%）。

### Sub Breakdown 细分占比

#### cpu_ns

| 部分 | 时间 | 占比 |
|---|---|---|
| cpu_query_stage_candidates_ns | 3278.116 ms | 59.26% |
| cpu_query_filter_ns | 829.242 ms | 14.99% |
| cpu_query_runtime_overhead_ns | 330.803 ms | 5.98% |
| cpu_query_finalize_ns | 228.616 ms | 4.13% |
| cpu_query_result_ids_ns | 228.162 ms | 4.12% |
| cpu_cache_lookup_ns | 219.982 ms | 3.98% |
| cpu_query_rerank_prepare_ns | 200.980 ms | 3.63% |
| cpu_query_beam_update_ns | 155.455 ms | 2.81% |
| cpu_query_select_ns | 52.299 ms | 0.95% |
| cpu_query_beam_sort_ns | 6.768 ms | 0.12% |
| cpu_query_rerank_collect_ns | 0.947 ms | 0.02% |
| cpu_query_rerank_update_ns | 0.269 ms | 0.00% |

- cpu_ns 内部热点：占比最高的几项是 `cpu_query_stage_candidates_ns`（59.26%）、`cpu_query_filter_ns`（14.99%）、`cpu_query_runtime_overhead_ns`（5.98%）。

#### gpu_ns

| 部分 | 时间 | 占比 |
|---|---|---|
| gpu_query_distance_ns | 388272.611 ms | 98.44% |
| gpu_query_rerank_ns | 3183.874 ms | 0.81% |
| gpu_query_prepare_ns | 2983.081 ms | 0.76% |

- gpu_ns 内部热点：占比最高的几项是 `gpu_query_distance_ns`（98.44%）、`gpu_query_rerank_ns`（0.81%）、`gpu_query_prepare_ns`（0.76%）。

#### rdma_ns

| 部分 | 时间 | 占比 |
|---|---|---|
| rdma_rabitq_fetch_ns | 42527.953 ms | 56.64% |
| rdma_neighbor_fetch_ns | 32051.632 ms | 42.69% |
| rdma_rerank_fetch_ns | 457.554 ms | 0.61% |
| rdma_medoid_ptr_ns | 51.572 ms | 0.07% |

- rdma_ns 内部热点：占比最高的几项是 `rdma_rabitq_fetch_ns`（56.64%）、`rdma_neighbor_fetch_ns`（42.69%）、`rdma_rerank_fetch_ns`（0.61%）。

#### transfer_ns

| 部分 | 时间 | 占比 |
|---|---|---|
| transfer_distance_d2h_ns | 2994.990 ms | 68.37% |
| transfer_rabitq_h2d_ns | 1330.176 ms | 30.37% |
| transfer_rerank_d2h_ns | 29.711 ms | 0.68% |
| transfer_query_h2d_ns | 13.567 ms | 0.31% |
| transfer_rerank_h2d_ns | 12.031 ms | 0.27% |

- transfer_ns 内部热点：占比最高的几项是 `transfer_distance_d2h_ns`（68.37%）、`transfer_rabitq_h2d_ns`（30.37%）、`transfer_rerank_d2h_ns`（0.68%）。

### Latency

| 延迟字段 | 值 |
|---|---|
| end_to_end_ns | 479588.837 ms |
| mean_end_to_end_ns | 285.470 ms |
| mean_queue_wait_ns | 0.088 ms |
| mean_service_ns | 285.381 ms |
| p50_end_to_end_ns | 275.975 ms |
| p50_service_ns | 275.971 ms |
| p95_end_to_end_ns | 479.782 ms |
| p95_service_ns | 479.585 ms |
| p99_end_to_end_ns | 595.323 ms |
| p99_service_ns | 595.316 ms |
| queue_wait_ns | 148.448 ms |
| service_ns | 479440.389 ms |

### Counters

| 字段 | 值 |
|---|---|
| rdma_read_bytes | 17,016,324,101 |
| h2d_bytes | 16,546,253,920 |
| rabitq_rdma_bytes | 13,864,765,720 |
| vector_rdma_bytes | 2,684,261,824 |
| neighbor_rdma_bytes | 467,242,965 |
| d2h_bytes | 109,531,628 |
| visited_neighborlists | 908,627 |
| rabitq_kernels | 802,904 |
| cache_hits | 62,132 |
| cache_misses | 11,610 |
| exact_reranks | 6,696 |
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
| rdma_read_bytes | 94,156,029,132 |
| h2d_bytes | 93,336,345,096 |
| d2h_bytes | 192,598,396 |
| rdma_write_bytes | 39,880,551 |

## Insert / Query 对比

| 类别 | Insert 占比 | Query 占比 |
|---|---|---|
| cpu_ns | 1.74% | 1.15% |
| gpu_ns | 85.82% | 82.27% |
| rdma_ns | 11.49% | 15.66% |
| transfer_ns | 0.94% | 0.91% |

- Insert 最大部分是 **gpu_ns**，占 **85.82%**。
- Query 最大部分是 **gpu_ns**，占 **82.27%**。
- Insert 更偏向 GPU 计算密集。
- Query 更偏向 RDMA / 远端访问受限。
