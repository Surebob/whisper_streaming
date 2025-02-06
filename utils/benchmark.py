"""
Benchmarking utilities for performance analysis and optimization.
"""

import os
import json
import time
import threading
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import gc

@dataclass
class BenchmarkMetrics:
    """Core metrics for benchmarking."""
    timestamp: float
    operation: str
    duration_ms: float
    data_size: int  # Size of data processed (e.g., samples, bytes)
    memory_mb: float  # Memory used for operation
    throughput: float  # Data processed per second
    context: Dict[str, Any]  # Additional context-specific metrics
    
    def validate(self) -> bool:
        """Validate metrics before recording."""
        # Basic validation
        if self.duration_ms < 0:
            return False
        if self.memory_mb < 0:
            return False
            
        # Operation-specific validation
        if self.operation == "finalize":
            # Finalize should have at least 0.5 seconds of audio
            if self.data_size < 8000:  # 16kHz * 0.5s
                return False
            # Context should include text length
            if "final_text_length" not in self.context:
                return False
        elif self.operation == "inference":
            # Check component through context
            if "voice_probability" in self.context:
                # VAD inference - allow multiples of 512 up to 4096
                if self.data_size < 500 or self.data_size > 4200:
                    return False
                if self.data_size % 512 > 50:  # Allow some flexibility
                    return False
            else:
                # Whisper inference expects ~1 second chunks
                if not (15000 <= self.data_size <= 17000):
                    return False
        elif self.operation == "process":
            # VAD process uses 512-sample chunks or 4096 for system audio
            # Audio process uses 1024-sample chunks
            # Whisper process uses 16000-sample chunks
            if self.data_size < 500:
                return False
            
            # Check for standard chunk sizes with 10% tolerance
            standard_chunks = [512, 1024, 4096, 16000]
            chunk_valid = False
            for chunk_size in standard_chunks:
                if abs(self.data_size - chunk_size) <= chunk_size * 0.1:
                    chunk_valid = True
                    break
            if not chunk_valid and "chunk_size" not in self.context:
                return False
            
        return True

class ComponentBenchmark:
    """Handles benchmarking for a specific component."""
    
    def __init__(self, component_name: str, benchmark_dir: str = "benchmarks"):
        self.component_name = component_name
        self.benchmark_dir = Path(benchmark_dir)
        self.benchmark_dir.mkdir(parents=True, exist_ok=True)
        
        # Create component-specific directory
        self.component_dir = self.benchmark_dir / component_name
        self.component_dir.mkdir(exist_ok=True)
        
        # Create new benchmark file for this session
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.benchmark_file = self.component_dir / f"benchmark_{timestamp}.jsonl"
        
        self._lock = threading.Lock()
        self._operation_counts = {}  # Track operation counts
        
        # Write header with system info
        self._write_header()
    
    def _write_header(self):
        """Write benchmark file header with system information."""
        import platform
        import psutil
        
        header = {
            "timestamp": time.time(),
            "type": "header",
            "system_info": {
                "platform": platform.platform(),
                "processor": platform.processor(),
                "cpu_count": psutil.cpu_count(),
                "memory_total": psutil.virtual_memory().total,
                "component": self.component_name
            }
        }
        
        self._write_entry(header)
    
    def _write_entry(self, entry: Dict[str, Any]):
        """Thread-safe write of a benchmark entry."""
        with self._lock:
            with open(self.benchmark_file, 'a', encoding='utf-8') as f:
                json.dump(entry, f)
                f.write('\n')
    
    def record_metric(self, metrics: BenchmarkMetrics):
        """Record a benchmark metric with validation."""
        # Validate metrics
        if not metrics.validate():
            self.record_event("invalid_metric", {
                "operation": metrics.operation,
                "duration_ms": metrics.duration_ms,
                "data_size": metrics.data_size,
                "context": metrics.context
            })
            return
            
        # Update operation count
        with self._lock:
            self._operation_counts[metrics.operation] = self._operation_counts.get(metrics.operation, 0) + 1
        
        entry = {
            "timestamp": metrics.timestamp,
            "type": "metric",
            "data": asdict(metrics)
        }
        self._write_entry(entry)
    
    def record_event(self, event_type: str, details: Dict[str, Any]):
        """Record a significant event or error."""
        entry = {
            "timestamp": time.time(),
            "type": "event",
            "event_type": event_type,
            "details": details
        }
        self._write_entry(entry)
    
    def get_stats(self) -> Dict[str, int]:
        """Get current operation counts."""
        with self._lock:
            return self._operation_counts.copy()

def get_memory_usage() -> float:
    """Get current memory usage in MB."""
    import psutil
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    return max(0, memory_info.rss / (1024 * 1024))  # Return RSS in MB, ensure non-negative

class BenchmarkContext:
    """Context manager for easy benchmarking of code blocks."""
    
    def __init__(self, benchmark: ComponentBenchmark, operation: str, data_size: int = 0, 
                 context: Optional[Dict[str, Any]] = None):
        self.benchmark = benchmark
        self.operation = operation
        self.data_size = data_size
        self.context = context or {}
        self.start_time = None
        self.start_memory = None
    
    def __enter__(self):
        self.start_time = time.perf_counter()
        self.start_memory = get_memory_usage()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self.benchmark.record_event("error", {
                "operation": self.operation,
                "error": str(exc_val),
                "type": exc_type.__name__
            })
            return False
        
        duration = (time.perf_counter() - self.start_time) * 1000  # Convert to ms
        end_memory = get_memory_usage()
        memory_delta = max(0, end_memory - self.start_memory)  # Ensure non-negative delta
        
        # Calculate throughput (data processed per second)
        # Only calculate if duration is significant and data_size is non-zero
        if duration > 0.1 and self.data_size > 0:  # Minimum 0.1ms duration
            throughput = (self.data_size / duration) * 1000
        else:
            throughput = 0
        
        # Add operation count to context
        stats = self.benchmark.get_stats()
        self.context["operation_count"] = stats.get(self.operation, 0)
        
        metrics = BenchmarkMetrics(
            timestamp=time.time(),
            operation=self.operation,
            duration_ms=duration,
            data_size=self.data_size,
            memory_mb=memory_delta,
            throughput=throughput,
            context=self.context
        )
        
        self.benchmark.record_metric(metrics)
        return True 