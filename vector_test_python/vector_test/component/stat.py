import threading
from typing import List, Optional
from enum import Enum


class OperationType(Enum):
    WRITE = 0
    READ = 1


class OperationStat:
    """Thread-safe statistics for a single operation type."""
    
    def __init__(self) -> None:
        self._mutex = threading.Lock()
        self._total_time: float = 0.0
        self._call_count: int = 0
    
    def add(self, duration: float) -> None:
        """Add a timing measurement."""
        with self._mutex:
            self._total_time += duration
            self._call_count += 1
    
    def reset(self) -> None:
        """Reset statistics."""
        with self._mutex:
            self._total_time = 0.0
            self._call_count = 0
    
    def get_total_time(self) -> float:
        """Get total time for all operations."""
        with self._mutex:
            return self._total_time
    
    def get_call_count(self) -> int:
        """Get number of operations."""
        with self._mutex:
            return self._call_count
    
    def get_avg_time(self) -> float:
        """Get average time per operation."""
        with self._mutex:
            if self._call_count == 0:
                return 0.0
            return self._total_time / self._call_count


class Stat:
    """
    Thread-safe statistics collector for multiple operation types.
    """
    
    def __init__(self, operation_count: int) -> None:
        self._stats: List[OperationStat] = [OperationStat() for _ in range(operation_count)]
        self._operation_names: List[str] = ["" for _ in range(operation_count)]
    
    def set_operation_name(self, index: int, name: str) -> None:
        """Set the name for an operation type."""
        if 0 <= index < len(self._operation_names):
            self._operation_names[index] = name
    
    def add_operation(self, operation_index: int, duration: float) -> None:
        """Add a timing measurement for an operation."""
        if 0 <= operation_index < len(self._stats):
            self._stats[operation_index].add(duration)
    
    def reset(self, operation_index: int) -> None:
        """Reset statistics for an operation type."""
        if 0 <= operation_index < len(self._stats):
            self._stats[operation_index].reset()
    
    def get_total_time(self, operation_index: int) -> float:
        """Get total time for an operation type."""
        if 0 <= operation_index < len(self._stats):
            return self._stats[operation_index].get_total_time()
        return 0.0
    
    def get_call_count(self, operation_index: int) -> int:
        """Get call count for an operation type."""
        if 0 <= operation_index < len(self._stats):
            return self._stats[operation_index].get_call_count()
        return 0
    
    def get_avg_time(self, operation_index: int) -> float:
        """Get average time for an operation type."""
        if 0 <= operation_index < len(self._stats):
            return self._stats[operation_index].get_avg_time()
        return 0.0
    
    def print_all(self) -> None:
        """Print all statistics."""
        for i, stat in enumerate(self._stats):
            name = self._operation_names[i] if self._operation_names[i] else f"Operation {i}"
            print(f"{name}:")
            print(f"  Total time: {stat.get_total_time():.6f} s")
            print(f"  Call count: {stat.get_call_count()}")
            print(f"  Avg time: {stat.get_avg_time():.6f} s")
