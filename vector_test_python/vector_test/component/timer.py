import time
from typing import Optional


class Timer:
    """
    High-precision timer for measuring elapsed time.
    """
    
    def __init__(self) -> None:
        self._start_time: Optional[float] = None
        self._end_time: Optional[float] = None
        self._elapsed: float = 0.0
    
    def start(self) -> None:
        """Start the timer."""
        self._start_time = time.perf_counter()
        self._end_time = None
    
    def pause(self) -> None:
        """Pause the timer (accumulate elapsed time so far)."""
        if self._start_time is not None:
            self._elapsed += time.perf_counter() - self._start_time
            self._start_time = None
    
    def resume(self) -> None:
        """Resume the timer from where it was paused."""
        if self._start_time is None:
            self._start_time = time.perf_counter()
    
    def stop(self) -> None:
        """Stop the timer and record final elapsed time."""
        if self._start_time is not None:
            self._elapsed += time.perf_counter() - self._start_time
            self._end_time = time.perf_counter()
            self._start_time = None
    
    def elapsed(self) -> float:
        """
        Get elapsed time in seconds.
        
        Returns:
            Elapsed time in seconds
        """
        if self._start_time is not None:
            return self._elapsed + time.perf_counter() - self._start_time
        return self._elapsed
