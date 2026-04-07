#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import sys
from pathlib import Path


def ns_to_ms(ns: float) -> float:
    return ns / 1e6


def format_pct(x: float) -> str:
    return f"{x:.2f}%"


def safe_div(a: float, b: float) -> float:
    if b == 0:
        return 0.0
    return a / b


def analyze_breakdown(name: str, data: dict) -> None:
    count = data.get("count", 0)
    latency = data.get("latency", {})
    phases = data.get("phases", {})
    counters = data.get("counters", {})

    service_ns = latency.get("service_ns", 0)
    end_to_end_ns = latency.get("end_to_end_ns", 0)
    queue_wait_ns = latency.get("queue_wait_ns", 0)

    mean_service_ns = latency.get("mean_service_ns", 0)
    mean_end_to_end_ns = latency.get("mean_end_to_end_ns", 0)
    mean_queue_wait_ns = latency.get("mean_queue_wait_ns", 0)

    total_phase_ns = sum(phases.values())

    print("=" * 90)
    print(f"{name.upper()} BREAKDOWN")
    print("=" * 90)
    print(f"count                : {count}")
    print(f"mean_end_to_end_ms   : {ns_to_ms(mean_end_to_end_ns):.3f}")
    print(f"mean_queue_wait_ms   : {ns_to_ms(mean_queue_wait_ns):.3f}")
    print(f"mean_service_ms      : {ns_to_ms(mean_service_ns):.3f}")
    print(f"end_to_end_ms(total) : {ns_to_ms(end_to_end_ns):.3f}")
    print(f"queue_wait_ms(total) : {ns_to_ms(queue_wait_ns):.3f}")
    print(f"service_ms(total)    : {ns_to_ms(service_ns):.3f}")
    print(f"phase_sum_ms(total)  : {ns_to_ms(total_phase_ns):.3f}")

    if service_ns and total_phase_ns:
        diff_ratio = abs(total_phase_ns - service_ns) / service_ns * 100
        print(f"phase_sum vs service : diff={diff_ratio:.2f}%")

    print("\n[Phase Breakdown]")
    print(
        f"{'phase':30} "
        f"{'total_ms':>12} "
        f"{'avg_ms/op':>12} "
        f"{'%service':>10} "
        f"{'%e2e':>10} "
        f"{'%phase_sum':>12}"
    )
    print("-" * 90)

    sorted_phases = sorted(phases.items(), key=lambda x: x[1], reverse=True)

    for phase, ns in sorted_phases:
        total_ms = ns_to_ms(ns)
        avg_ms = ns_to_ms(ns / count) if count else 0.0
        pct_service = safe_div(ns, service_ns) * 100
        pct_e2e = safe_div(ns, end_to_end_ns) * 100
        pct_phase_sum = safe_div(ns, total_phase_ns) * 100

        print(
            f"{phase:30} "
            f"{total_ms:12.3f} "
            f"{avg_ms:12.6f} "
            f"{format_pct(pct_service):>10} "
            f"{format_pct(pct_e2e):>10} "
            f"{format_pct(pct_phase_sum):>12}"
        )

    print("\n[Top 3 phases by %service]")
    for phase, ns in sorted_phases[:3]:
        pct_service = safe_div(ns, service_ns) * 100
        print(f"  {phase}: {ns_to_ms(ns):.3f} ms total ({pct_service:.2f}% of service)")

    if counters:
        print("\n[Selected Counters]")
        interesting_keys = [
            "rdma_read_bytes",
            "rdma_write_bytes",
            "neighbor_rdma_bytes",
            "vector_rdma_bytes",
            "rabitq_rdma_bytes",
            "h2d_bytes",
            "d2h_bytes",
            "lock_attempts",
            "lock_retries",
            "cas_failures",
            "rabitq_kernels",
            "l2_kernels",
            "prune_kernels",
            "exact_reranks",
            "visited_neighborlists",
            "visited_nodes",
        ]
        for key in interesting_keys:
            if key in counters:
                value = counters[key]
                per_op = value / count if count else 0
                print(f"  {key:24}: total={value}  per_op={per_op:.3f}")


def main():
    if len(sys.argv) != 2:
        print("Usage: python breakdown_ratio.py <input.json>")
        sys.exit(1)

    input_path = Path(sys.argv[1])
    if not input_path.exists():
        print(f"Error: file not found: {input_path}")
        sys.exit(1)

    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    meta = data.get("meta", {})
    system_counters = data.get("system_counters", {})

    print("=" * 90)
    print("META")
    print("=" * 90)
    for k, v in meta.items():
        print(f"{k:20}: {v}")

    print("\n" + "=" * 90)
    print("SYSTEM COUNTERS")
    print("=" * 90)
    for k, v in system_counters.items():
        print(f"{k:20}: {v}")

    query_breakdown = data.get("query_breakdown")
    if query_breakdown:
        print()
        analyze_breakdown("query", query_breakdown)

    insert_breakdown = data.get("insert_breakdown")
    if insert_breakdown:
        print()
        analyze_breakdown("insert", insert_breakdown)


if __name__ == "__main__":
    main()