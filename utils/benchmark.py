"""
Benchmarking utilities for performance analysis and optimization.
"""

import os
import json
import time
import threading
import psutil
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict, field
from datetime import datetime
import gc
import queue
from utils.logger import get_logger

# Use existing logger
logger = get_logger(__name__, console_output=False)

# Cache for system metrics to avoid frequent calls
_last_cpu_check = 0
_last_cpu_value = 0
_CPU_CHECK_INTERVAL = 1.0  # Only check CPU usage every second

def get_cpu_usage():
    """Get CPU usage with caching to reduce system calls."""
    global _last_cpu_check, _last_cpu_value
    now = time.time()
    if now - _last_cpu_check > _CPU_CHECK_INTERVAL:
        _last_cpu_value = psutil.cpu_percent()
        _last_cpu_check = now
    return _last_cpu_value

@dataclass
class BenchmarkMetrics:
    """Core metrics for benchmarking."""
    id: str
    timestamp: float
    operation: str
    duration_ms: float
    data_size: int
    memory_mb: float
    throughput: float
    cpu_usage: float
    gpu_utilization: Optional[float] = None
    peak_memory_mb: Optional[float] = None
    queue_size: Optional[int] = None
    context: Dict[str, Any] = field(default_factory=dict)
    
    # Simplified validation rules
    VALIDATION_RULES = {
        "finalize": lambda m: m.data_size >= 8000 and m.duration_ms > 0,
        "inference": lambda m: m.duration_ms > 0,
        "process": lambda m: m.duration_ms > 0 and m.data_size >= 500
    }
    
    def validate(self) -> bool:
        """Simplified validation with basic checks only."""
        if self.duration_ms < 0 or self.memory_mb < 0:
            return False
        rule = self.VALIDATION_RULES.get(self.operation)
        if rule and not rule(self):
            return False
        return True

class ComponentBenchmark:
    """Handles benchmarking for a specific component."""
    
    def __init__(self, component_name: str, benchmark_dir: str = "benchmarks", batch_size: int = 100):
        self.component_name = component_name
        self.benchmark_dir = Path(benchmark_dir)
        self._batch_size = batch_size
        self._lock = threading.Lock()
        self._operation_counts = {}
        self._is_running = True
        self._buffer = []
        self._entry_queue = queue.Queue(maxsize=1000)  # Limit queue size
        
        # Create directories and file
        self.benchmark_dir.mkdir(parents=True, exist_ok=True)
        self.component_dir = self.benchmark_dir / component_name
        self.component_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.benchmark_file = self.component_dir / f"benchmark_{timestamp}.jsonl"
        
        self._write_header()
        self._writer_thread = threading.Thread(target=self._writer_loop, daemon=True)
        self._writer_thread.start()
    
    def _write_header(self):
        """Write benchmark file header."""
        import platform
        header = {
            "timestamp": time.time(),
            "type": "header",
            "system_info": {
                "platform": platform.platform(),
                "processor": platform.processor(),
                "cpu_count": psutil.cpu_count(),
                "memory_total": psutil.virtual_memory().total,
                "component": self.component_name,
                "batch_size": self._batch_size
            }
        }
        with open(self.benchmark_file, 'a', encoding='utf-8') as f:
            json.dump(header, f)
            f.write('\n')
    
    def _writer_loop(self):
        """Background thread for batch writing entries."""
        while self._is_running or not self._entry_queue.empty():
            try:
                # Get next entry with timeout
                entry = self._entry_queue.get(timeout=1.0)
                self._buffer.append(entry)
                
                # Flush if buffer is full
                if len(self._buffer) >= self._batch_size:
                    self._flush_buffer()
                    
            except queue.Empty:
                if self._buffer:
                    self._flush_buffer()
                continue
            except Exception as e:
                logger.error(f"Error in writer thread: {e}")
    
    def _flush_buffer(self):
        """Write buffered entries to file."""
        if not self._buffer:
            return
        try:
            with open(self.benchmark_file, 'a', encoding='utf-8') as f:
                for entry in self._buffer:
                    json.dump(entry, f)
                    f.write('\n')
            self._buffer.clear()
        except Exception as e:
            logger.error(f"Error flushing buffer: {e}")
    
    def _write_entry(self, entry: Dict[str, Any]):
        """Queue an entry for writing."""
        try:
            self._entry_queue.put_nowait(entry)
        except queue.Full:
            pass  # Drop metrics if queue is full
    
    def record_metric(self, metrics: BenchmarkMetrics):
        """Record a benchmark metric."""
        if not metrics.validate():
            return
        with self._lock:
            self._operation_counts[metrics.operation] = self._operation_counts.get(metrics.operation, 0) + 1
        entry = {
            "timestamp": metrics.timestamp,
            "type": "metric",
            "data": asdict(metrics)
        }
        self._write_entry(entry)
    
    def record_event(self, event_type: str, details: Dict[str, Any]):
        """Record a significant event."""
        entry = {
            "timestamp": time.time(),
            "type": "event",
            "event_type": event_type,
            "details": details
        }
        self._write_entry(entry)
    
    def get_stats(self) -> Dict[str, int]:
        """Get operation counts."""
        with self._lock:
            return self._operation_counts.copy()
    
    def close(self):
        """Shut down the writer thread."""
        self._is_running = False
        if self._writer_thread.is_alive():
            self._writer_thread.join(timeout=5.0)
        self._flush_buffer()

def get_memory_usage() -> float:
    """Get current memory usage in MB."""
    import psutil
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    return max(0, memory_info.rss / (1024 * 1024))  # Return RSS in MB, ensure non-negative

class BenchmarkContext:
    """Context manager for benchmarking."""
    
    def __init__(self, benchmark, operation: str, data_size: int = 0, 
                 context: Optional[Dict[str, Any]] = None):
        self.benchmark = benchmark
        self.operation = operation
        self.data_size = data_size
        self.context = context or {}
        self.start_time = None
        self.start_memory = None
    
    def __enter__(self):
        if self.benchmark is None:
            return self
        self.start_time = time.perf_counter()
        self.start_memory = get_memory_usage()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.benchmark is None:
            return True
        if exc_type is not None:
            self.benchmark.record_event("error", {
                "operation": self.operation,
                "error": str(exc_val)
            })
            return False
            
        # Quick time and memory calculations
        duration = (time.perf_counter() - self.start_time) * 1000
        memory_delta = max(0, get_memory_usage() - self.start_memory)
        
        # Simple throughput calculation
        throughput = (self.data_size / duration) * 1000 if duration > 0.1 and self.data_size > 0 else 0
        
        # Create metrics with minimal system calls
        metrics = BenchmarkMetrics(
            id=str(time.time()),
            timestamp=time.time(),
            operation=self.operation,
            duration_ms=duration,
            data_size=self.data_size,
            memory_mb=memory_delta,
            throughput=throughput,
            cpu_usage=get_cpu_usage(),  # Use cached CPU value
            context=self.context
        )
        
        self.benchmark.record_metric(metrics)
        return True 