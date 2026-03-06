import os
import time
import threading
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
from enum import Enum

from .vector_index import VectorIndex
from .config import read_config
from .util import read_fbin, read_bin, rand_vec
from .component import Timer, Stat, OperationType, MemoryMonitor


class TestType(Enum):
    STORAGE = "storage"
    DYNAMIC = "dynamic"
    RECALL = "recall"


class VectorTest:
    """
    Main test framework for vector index performance evaluation.
    
    Supports four test modes:
    - build: Build index from data file
    - storage: Test storage/memory usage
    - dynamic: Dynamic read/write performance test
    - recall: Recall rate test
    """
    
    def __init__(self, config_path: str, index: VectorIndex) -> None:
        """
        Initialize VectorTest with configuration and index.
        
        Args:
            config_path: path to configuration file
            index: VectorIndex instance to test
        """
        self.index = index
        self.config: Dict[str, str] = read_config(config_path)
        
        log_path = self.config.get('log_path', './logs')
        os.makedirs(log_path, exist_ok=True)
        
        log_name = f"{log_path}/{datetime.now().strftime('%Y%m%d-%H-%M.log')}"
        
        print(f"[INFO] Log file: {log_name}")
        for key, value in self.config.items():
            print(f"[INFO] Config: {key} = {value}")
        
        option = self.config.get('option', '')
        if option == 'build':
            self.build()
        elif option == 'storage':
            self.storage_test()
        elif option == 'dynamic':
            self.dynamic_test()
        elif option == 'recall':
            self.recall_test()
        else:
            raise ValueError(f"Unknown option: {option}")
    
    def build(self) -> None:
        """Build index from data file."""
        print(f"[INFO] Building index: {self.index.get_index_type()}")
        
        dataset_path = self.config.get('data_path', '')
        index_path = self.config.get('index_path', '')
        
        if not dataset_path:
            raise ValueError("Config file missing data_path")
        if not index_path:
            raise ValueError("Config file missing index_path")
        
        timer = Timer()
        timer.start()
        
        self.index.build_from_file(dataset_path)
        
        timer.stop()
        print(f"[INFO] Build time: {timer.elapsed():.2f} s")
        
        self.index.save(index_path)
        print(f"[INFO] Index saved to: {index_path}")
    
    def storage_test(self) -> None:
        """Test storage/memory usage."""
        print(f"[INFO] Storage test: {self.index.get_index_type()}")
        
        index_path = self.config.get('index_path', '')
        if not index_path:
            raise ValueError("Config file missing index_path")
        
        mem_monitor = MemoryMonitor()
        mem_monitor.start()
        
        self.index.load(index_path)
        
        dim = int(self.config.get('dim', '0'))
        if dim == 0:
            raise ValueError("Config file missing dim")
        
        topk = int(self.config.get('topk', '10'))
        
        query_count = int(self.config.get('test_scale', '100'))
        
        if 'query_data' in self.config:
            query_data, query_info = read_fbin(self.config['query_data'])
            query_count = query_info[0]
        else:
            query_data = rand_vec(dim, query_count)
        
        ids_res = np.zeros((query_count, topk), dtype=np.uint32)
        distances_res = np.zeros((query_count, topk), dtype=np.float32)
        
        for i in range(query_count):
            ids, distances = self.index.search(query_data[i], topk)
            ids_res[i] = ids
            distances_res[i] = distances
        
        time.sleep(10)
        
        mem_monitor.stop()
        print(f"[INFO] Storage test memory usage: {mem_monitor.get_average_memory_usage():.2f}%")
    
    def dynamic_test(self) -> None:
        print(f"[INFO] Dynamic test: {self.index.get_index_type()}")
        
        index_path = self.config.get('index_path', '')
        if not index_path:
            raise ValueError("Config file missing index_path")
        
        self.index.load(index_path)
        
        dim = int(self.config.get('dim', '0'))
        if dim == 0:
            raise ValueError("Config file missing dim")
        
        topk = int(self.config.get('topk', '10'))
        thread_count = int(self.config.get('threads', '4'))
        test_count = int(self.config.get('test_scale', '1000'))
        
        print(f"[INFO] Using {thread_count} threads for testing")
        
        if 'vector_data' in self.config:
            vector_data, vector_info = read_fbin(self.config['vector_data'])
            test_count = vector_info[0]
        else:
            vector_data = rand_vec(dim, test_count)
        
        read_ratio = float(self.config.get('read_ratio', '0.5'))
        total_test = self.config.get('total_test', 'false').lower() == 'true'
        
        if total_test:
            for ratio in [i * 0.1 for i in range(11)]:
                print(f"[INFO] Dynamic test read ratio {ratio * 100:.0f}%")
                self._run_dynamic_core(vector_data, dim, topk, thread_count, test_count, ratio)
        else:
            print(f"[INFO] Dynamic test read ratio {read_ratio * 100:.0f}%")
            self._run_dynamic_core(vector_data, dim, topk, thread_count, test_count, read_ratio)
    
    def _run_dynamic_core(self, vector_data: np.ndarray, dim: int, topk: int,
                          thread_count: int, test_count: int, read_ratio: float) -> None:
        """Core dynamic test implementation."""
        stat = Stat(2)
        stat.set_operation_name(OperationType.WRITE.value, "Write")
        stat.set_operation_name(OperationType.READ.value, "Read")
        
        completed_ops = [0]
        completed_lock = threading.Lock()
        
        def worker(start: int, end: int) -> None:
            for i in range(start, end):
                is_read = np.random.random() < read_ratio
                
                if is_read:
                    timer = Timer()
                    timer.start()
                    self.index.search(vector_data[i], topk)
                    timer.stop()
                    stat.add_operation(OperationType.READ.value, timer.elapsed())
                else:
                    timer = Timer()
                    timer.start()
                    self.index.insert(vector_data[i], np.array([i + 1], dtype=np.uint32))
                    timer.stop()
                    stat.add_operation(OperationType.WRITE.value, timer.elapsed())
                
                with completed_lock:
                    completed_ops[0] += 1
        
        start_time = time.time()
        
        threads = []
        ops_per_thread = test_count // thread_count
        remainder = test_count % thread_count
        start_idx = 0
        
        for i in range(thread_count):
            end_idx = start_idx + ops_per_thread + (1 if i < remainder else 0)
            t = threading.Thread(target=worker, args=(start_idx, end_idx))
            threads.append(t)
            t.start()
            start_idx = end_idx
        
        progress_width = 50
        while completed_ops[0] < test_count:
            progress = completed_ops[0] / test_count
            bar_len = int(progress * progress_width)
            elapsed = time.time() - start_time
            print(f"\r[{'=' * bar_len}{' ' * (progress_width - bar_len)}] "
                  f"{progress * 100:.0f}% ({completed_ops[0]}/{test_count}) "
                  f"Time: {elapsed:.1f}s", end='')
            time.sleep(0.1)
        
        for t in threads:
            t.join()
        
        total_time = (time.time() - start_time) * 1000  # ms
        
        write_calls = stat.get_call_count(OperationType.WRITE.value)
        read_calls = stat.get_call_count(OperationType.READ.value)
        write_total_time = stat.get_total_time(OperationType.WRITE.value)
        read_total_time = stat.get_total_time(OperationType.READ.value)
        
        write_avg_latency = write_total_time / write_calls if write_calls > 0 else 0
        read_avg_latency = read_total_time / read_calls if read_calls > 0 else 0
        
        write_throughput = write_calls / total_time * 1000 if total_time > 0 else 0
        read_throughput = read_calls / total_time * 1000 if total_time > 0 else 0
        total_throughput = (write_calls + read_calls) / total_time * 1000 if total_time > 0 else 0
        
        print(f"\n[INFO] Dynamic test completed, total time: {total_time:.2f} ms")
        print(f"[INFO] Write operations: {write_calls}, Read operations: {read_calls}")
        print(f"[INFO] Write avg latency: {write_avg_latency:.6f} s")
        print(f"[INFO] Read avg latency: {read_avg_latency:.6f} s")
        print(f"[INFO] Write throughput: {write_throughput:.2f} ops/s")
        print(f"[INFO] Read throughput: {read_throughput:.2f} ops/s")
        print(f"[INFO] Total throughput: {total_throughput:.2f} ops/s")
    
    def recall_test(self) -> None:
        """Recall rate test."""
        print(f"[INFO] Recall test: {self.index.get_index_type()}")
        
        index_path = self.config.get('index_path', '')
        if not index_path:
            raise ValueError("Config file missing index_path")
        
        self.index.load(index_path)
        
        if 'query_data' not in self.config:
            raise ValueError("Config file missing query_data")
        if 'groundtruth' not in self.config:
            raise ValueError("Config file missing groundtruth")
        
        query_data, query_info = read_fbin(self.config['query_data'])
        dim = query_info[1]
        topk = int(self.config.get('topk', '10'))
        
        ids_res = []
        distances_res = []
        
        for i in range(query_info[0]):
            ids, distances = self.index.search(query_data[i], topk)
            ids_res.append(ids)
            distances_res.append(distances)
        
        groundtruth, gt_info = read_bin(self.config['groundtruth'])
        
        if gt_info[0] != query_info[0]:
            raise ValueError("Groundtruth number does not match query number")
        if gt_info[1] != topk:
            raise ValueError("Groundtruth topk does not match test topk")
        
        recall = 0.0
        for i in range(query_info[0]):
            recall_per_query = 0.0
            for j in range(topk):
                if groundtruth[i, j] in ids_res[i]:
                    recall_per_query += 1.0
            recall += recall_per_query / topk
        
        recall /= query_info[0]
        print(f"[INFO] Recall: {recall:.4f}")
