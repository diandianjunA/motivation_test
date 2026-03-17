#!/usr/bin/env python3
"""
Integration test script for SHINE HNSW HTTP service.

Tests: status, insert, search, recall accuracy.
Generates random vectors — no external dataset needed.
Safe to run multiple times against the same server.

Usage:
  python3 test_http.py [--host localhost] [--port 8080] [--dim 128]
"""

import argparse
import json
import random
import sys
import time

try:
    import requests
except ImportError:
    print("Please install requests: pip install requests")
    sys.exit(1)


def log(msg, ok=True):
    tag = "\033[32mPASS\033[0m" if ok else "\033[31mFAIL\033[0m"
    print(f"  [{tag}] {msg}")


def log_info(msg):
    print(f"  [INFO] {msg}")


def random_vector(dim):
    return [random.gauss(0, 1) for _ in range(dim)]


def l2_distance(a, b):
    return sum((x - y) ** 2 for x, y in zip(a, b))


def brute_force_knn(query, vectors, k):
    """Return k nearest neighbor IDs by L2 distance."""
    dists = [(vid, l2_distance(query, vec)) for vid, vec in vectors.items()]
    dists.sort(key=lambda x: x[1])
    return [vid for vid, _ in dists[:k]]


class TestRunner:
    def __init__(self, host, port, dim):
        self.url = f"http://{host}:{port}"
        self.dim = dim
        self.session = requests.Session()
        self.passed = 0
        self.failed = 0
        self.vectors = {}  # id -> vector, for recall verification
        self.initial_count = 0  # vectors already on server before this run
        # use a large random base ID to avoid collisions with previous runs
        self.id_base = random.randint(10_000_000, 90_000_000)

    def check(self, condition, msg):
        if condition:
            log(msg, ok=True)
            self.passed += 1
        else:
            log(msg, ok=False)
            self.failed += 1

    def next_id(self, offset):
        return self.id_base + offset

    def test_status(self):
        print("\n=== Test: GET /status ===")
        resp = self.session.get(f"{self.url}/status")
        self.check(resp.status_code == 200, f"status code = {resp.status_code}")

        data = resp.json()
        self.check("status" in data, "response has 'status' field")
        self.check(data.get("status") == "running", f"status = {data.get('status')}")
        self.check(data.get("dimension") == self.dim, f"dimension = {data.get('dimension')}")

        self.initial_count = data.get("vectors_inserted", 0)
        log_info(f"vectors already on server: {self.initial_count}")
        log_info(f"ID base for this run: {self.id_base}")

    def test_insert_single(self):
        print("\n=== Test: POST /vectors (single) ===")
        vid = self.next_id(0)
        vec = random_vector(self.dim)
        payload = {"vectors": [{"id": vid, "values": vec}]}

        resp = self.session.post(f"{self.url}/vectors", json=payload)
        self.check(resp.status_code == 200, f"status code = {resp.status_code}")

        data = resp.json()
        self.check(data.get("inserted") == 1, f"inserted = {data.get('inserted')}")
        self.check(data.get("status") == "ok", f"status = {data.get('status')}")
        self.vectors[vid] = vec

    def test_insert_batch(self, count=99):
        print(f"\n=== Test: POST /vectors (batch, {count} vectors) ===")
        batch_size = 50
        total_inserted = 0

        for offset in range(0, count, batch_size):
            batch = []
            for i in range(min(batch_size, count - offset)):
                vid = self.next_id(1 + offset + i)
                vec = random_vector(self.dim)
                batch.append({"id": vid, "values": vec})
                self.vectors[vid] = vec

            resp = self.session.post(f"{self.url}/vectors", json={"vectors": batch})
            data = resp.json()
            total_inserted += data.get("inserted", 0)

        self.check(total_inserted == count, f"total inserted = {total_inserted} (expected {count})")

    def test_insert_wrong_dim(self):
        print("\n=== Test: POST /vectors (wrong dimension) ===")
        vec = random_vector(self.dim + 5)
        payload = {"vectors": [{"id": self.next_id(99999), "values": vec}]}

        resp = self.session.post(f"{self.url}/vectors", json=payload)
        data = resp.json()
        self.check(data.get("inserted") == 0, f"rejected wrong-dim vector (inserted={data.get('inserted')})")

    def test_insert_empty(self):
        print("\n=== Test: POST /vectors (empty batch) ===")
        payload = {"vectors": []}
        resp = self.session.post(f"{self.url}/vectors", json=payload)
        data = resp.json()
        self.check(data.get("inserted") == 0, f"empty batch inserted = {data.get('inserted')}")

    def test_search_basic(self):
        print("\n=== Test: POST /search (basic) ===")
        if not self.vectors:
            log("skipped — no vectors inserted", ok=False)
            self.failed += 1
            return

        # query with a known inserted vector — should find itself
        query_id = self.next_id(0)
        query_vec = self.vectors[query_id]

        payload = {"vector": query_vec, "k": 10}
        resp = self.session.post(f"{self.url}/search", json=payload)
        self.check(resp.status_code == 200, f"status code = {resp.status_code}")

        data = resp.json()
        results = data.get("results", [])
        self.check(len(results) > 0, f"got {len(results)} results")
        self.check(query_id in results, f"self-query found in top-10 (id={query_id} in results={results})")

    def test_search_recall(self, k=10, num_queries=20):
        """Recall computed only against this run's vectors."""
        print(f"\n=== Test: POST /search (recall@{k}, {num_queries} queries) ===")
        if len(self.vectors) < k:
            log(f"skipped — need at least {k} vectors", ok=False)
            self.failed += 1
            return

        total_hits = 0
        total_possible = 0
        our_ids = set(self.vectors.keys())

        for _ in range(num_queries):
            # pick a random vector from our set as query — guarantees ground truth is meaningful
            query_id = random.choice(list(self.vectors.keys()))
            query_vec = self.vectors[query_id]
            gt = set(brute_force_knn(query_vec, self.vectors, k))

            payload = {"vector": query_vec, "k": k}
            resp = self.session.post(f"{self.url}/search", json=payload)
            data = resp.json()
            results = data.get("results", [])

            # only count hits among our own vectors
            result_set = set(results) & our_ids
            hits = len(result_set & gt)
            total_hits += hits
            total_possible += k

        recall = total_hits / total_possible if total_possible > 0 else 0
        log_info(f"recall@{k} = {recall:.4f} ({total_hits}/{total_possible})")
        # threshold is lenient: server has other vectors too, so our vectors compete for top-k slots
        self.check(recall > 0.1, f"recall@{k} = {recall:.4f} (threshold > 0.1)")

    def test_search_missing_vector(self):
        print("\n=== Test: POST /search (missing vector field) ===")
        resp = self.session.post(f"{self.url}/search", json={"k": 5})
        self.check(resp.status_code == 400, f"status code = {resp.status_code} (expected 400)")

    def test_search_wrong_dim(self):
        print("\n=== Test: POST /search (wrong dimension) ===")
        vec = random_vector(self.dim + 3)
        resp = self.session.post(f"{self.url}/search", json={"vector": vec, "k": 5})
        self.check(resp.status_code == 400, f"status code = {resp.status_code} (expected 400)")

    def test_search_custom_k(self):
        print("\n=== Test: POST /search (custom k) ===")
        if not self.vectors:
            log("skipped — no vectors", ok=False)
            self.failed += 1
            return

        query_vec = list(self.vectors.values())[0]
        for k_val in [1, 3, 5]:
            payload = {"vector": query_vec, "k": k_val}
            resp = self.session.post(f"{self.url}/search", json=payload)
            data = resp.json()
            results = data.get("results", [])
            self.check(len(results) <= k_val, f"k={k_val}: got {len(results)} results (<= {k_val})")

    def test_insert_large_batch(self, count=500):
        print(f"\n=== Test: POST /vectors (large batch, {count} vectors) ===")
        batch_size = 100
        total_inserted = 0
        id_offset = 10000  # separate range from earlier inserts
        t0 = time.perf_counter()

        for offset in range(0, count, batch_size):
            batch = []
            for i in range(min(batch_size, count - offset)):
                vid = self.next_id(id_offset + offset + i)
                vec = random_vector(self.dim)
                batch.append({"id": vid, "values": vec})
                self.vectors[vid] = vec

            resp = self.session.post(f"{self.url}/vectors", json={"vectors": batch})
            data = resp.json()
            total_inserted += data.get("inserted", 0)

        elapsed = time.perf_counter() - t0
        throughput = total_inserted / elapsed if elapsed > 0 else 0
        self.check(total_inserted == count, f"inserted {total_inserted}/{count}")
        log_info(f"throughput: {throughput:.0f} vec/s ({elapsed:.2f}s)")

    def test_search_throughput(self, num_queries=100):
        print(f"\n=== Test: POST /search (throughput, {num_queries} queries) ===")
        if not self.vectors:
            log("skipped — no vectors", ok=False)
            self.failed += 1
            return

        latencies = []
        t0 = time.perf_counter()

        for _ in range(num_queries):
            query_vec = random_vector(self.dim)
            t_start = time.perf_counter()
            resp = self.session.post(f"{self.url}/search", json={"vector": query_vec, "k": 10})
            latencies.append(time.perf_counter() - t_start)
            assert resp.status_code == 200

        elapsed = time.perf_counter() - t0
        qps = num_queries / elapsed if elapsed > 0 else 0
        latencies.sort()
        p50 = latencies[len(latencies) // 2] * 1000
        p99 = latencies[int(len(latencies) * 0.99)] * 1000

        self.check(qps > 0, f"QPS = {qps:.1f}")
        log_info(f"latency p50={p50:.2f}ms, p99={p99:.2f}ms")

    def test_status_after_inserts(self):
        print("\n=== Test: GET /status (after inserts) ===")
        resp = self.session.get(f"{self.url}/status")
        data = resp.json()
        count = data.get("vectors_inserted", 0)
        expected = self.initial_count + len(self.vectors)
        self.check(count == expected, f"vectors_inserted = {count} (expected {expected} = {self.initial_count} + {len(self.vectors)})")

    def test_index_load_store(self):
        print("\n=== Test: POST /index/store & /index/load ===")

        # store the current index
        resp = self.session.post(f"{self.url}/index/store", json={"path": "/tmp/shine_test_index"})
        self.check(resp.status_code == 200, f"/index/store status = {resp.status_code}")
        data = resp.json()
        self.check(data.get("status") == "ok", f"/index/store result = {data.get('status')}")

        # load it back
        resp = self.session.post(f"{self.url}/index/load", json={"path": "/tmp/shine_test_index"})
        self.check(resp.status_code == 200, f"/index/load status = {resp.status_code}")
        data = resp.json()
        self.check(data.get("status") == "ok", f"/index/load result = {data.get('status')}")

        # verify search still works after load
        if self.vectors:
            query_id = self.next_id(0)
            query_vec = self.vectors[query_id]
            payload = {"vector": query_vec, "k": 10}
            resp = self.session.post(f"{self.url}/search", json=payload)
            data = resp.json()
            results = data.get("results", [])
            self.check(query_id in results, f"search works after load/store (id={query_id} in results)")

        # test missing path field
        resp = self.session.post(f"{self.url}/index/store", json={})
        self.check(resp.status_code == 400, f"/index/store without path returns 400 (got {resp.status_code})")

        resp = self.session.post(f"{self.url}/index/load", json={})
        self.check(resp.status_code == 400, f"/index/load without path returns 400 (got {resp.status_code})")

    def run_all(self):
        print(f"SHINE HNSW HTTP Service Test Suite")
        print(f"Target: {self.url}, dim={self.dim}")

        # connectivity
        try:
            self.session.get(f"{self.url}/status", timeout=5)
        except Exception as e:
            print(f"\n\033[31mCannot connect to {self.url}: {e}\033[0m")
            sys.exit(1)

        self.test_status()
        self.test_insert_single()
        self.test_insert_batch(count=99)
        self.test_insert_wrong_dim()
        self.test_insert_empty()
        self.test_search_basic()
        self.test_search_recall(k=10, num_queries=20)
        self.test_search_missing_vector()
        self.test_search_wrong_dim()
        self.test_search_custom_k()
        self.test_index_load_store()
        self.test_insert_large_batch(count=500)
        self.test_status_after_inserts()
        self.test_search_throughput(num_queries=100)

        # summary
        total = self.passed + self.failed
        print(f"\n{'=' * 50}")
        if self.failed == 0:
            print(f"\033[32mAll {total} tests passed.\033[0m")
        else:
            print(f"\033[31m{self.failed}/{total} tests failed.\033[0m")

        return 0 if self.failed == 0 else 1


def main():
    parser = argparse.ArgumentParser(description="SHINE HNSW HTTP Service Tests")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--dim", type=int, default=128)
    args = parser.parse_args()

    runner = TestRunner(args.host, args.port, args.dim)
    sys.exit(runner.run_all())


if __name__ == "__main__":
    main()
