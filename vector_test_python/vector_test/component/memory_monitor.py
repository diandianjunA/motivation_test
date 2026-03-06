import os
import time
import threading
from typing import Optional, List


class MemoryInfo:
    """Memory usage information for a process."""
    
    def __init__(self, pid: int, rss: int, vsz: int, 
                 shared: int, text: int, data: int, 
                 rss_percentage: float) -> None:
        self.pid = pid
        self.rss = rss  # Resident Set Size in KB
        self.vsz = vsz  # Virtual Memory Size in KB
        self.shared = shared
        self.text = text
        self.data = data
        self.rss_percentage = rss_percentage


class MemoryMonitor:
    """
    Monitor process memory usage.
    """
    
    def __init__(self, pid: Optional[int] = None) -> None:
        self._target_pid = pid if pid is not None else os.getpid()
        self._running = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._samples: List[float] = []
        self._lock = threading.Lock()
    
    def _get_system_total_memory(self) -> int:
        """Get total system memory in KB."""
        try:
            with open('/proc/meminfo', 'r') as f:
                for line in f:
                    if line.startswith('MemTotal:'):
                        return int(line.split()[1])  # KB
        except Exception:
            pass
        return 0
    
    def _parse_mem_info(self) -> MemoryInfo:
        """Parse memory info from /proc/[pid]/status."""
        try:
            total_mem = self._get_system_total_memory()
            
            with open(f'/proc/{self._target_pid}/status', 'r') as f:
                content = f.read()
            
            def get_value(key: str) -> int:
                for line in content.split('\n'):
                    if line.startswith(key):
                        return int(line.split()[1])
                return 0
            
            rss = get_value('VmRSS:')
            vsz = get_value('VmSize:')
            shared = get_value('RssShmem:')
            text = get_value('Text:') if 'Text:' in content else 0
            data = get_value('Data:') if 'Data:' in content else 0
            
            rss_percentage = (rss / total_mem * 100) if total_mem > 0 else 0.0
            
            return MemoryInfo(
                pid=self._target_pid,
                rss=rss,
                vsz=vsz,
                shared=shared,
                text=text,
                data=data,
                rss_percentage=rss_percentage
            )
        except Exception as e:
            return MemoryInfo(self._target_pid, 0, 0, 0, 0, 0, 0.0)
    
    def _monitor_loop(self) -> None:
        """Background monitoring loop."""
        while self._running:
            mem_info = self._parse_mem_info()
            with self._lock:
                self._samples.append(mem_info.rss_percentage)
            time.sleep(0.1)
    
    def start(self) -> None:
        """Start monitoring memory usage."""
        if not self._running:
            self._running = True
            self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self._monitor_thread.start()
    
    def stop(self) -> None:
        """Stop monitoring memory usage."""
        if self._running:
            self._running = False
            if self._monitor_thread is not None:
                self._monitor_thread.join(timeout=1.0)
    
    def get_current_memory_usage(self) -> MemoryInfo:
        """Get current memory usage."""
        return self._parse_mem_info()
    
    def get_average_memory_usage(self) -> float:
        """Get average memory usage percentage."""
        with self._lock:
            if not self._samples:
                return 0.0
            return sum(self._samples) / len(self._samples)
    
    def get_target_pid(self) -> int:
        """Get the target process ID."""
        return self._target_pid
    
    def set_target_pid(self, pid: int) -> None:
        """Set the target process ID to monitor."""
        self._target_pid = pid
    
    def __del__(self) -> None:
        """Cleanup on deletion."""
        self.stop()
