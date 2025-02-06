"""
Performance monitoring and statistics tracking for the whisper streaming pipeline.
"""

import threading
import time
from collections import deque
from typing import Dict, Optional, Any
from dataclasses import dataclass
from statistics import mean, median

@dataclass
class LatencyMetrics:
    """Holds latency metrics for a component."""
    avg: float
    median: float
    p95: float
    min: float
    max: float
    last: float
    count: int

class ModelStatsMonitor:
    """Tracks inference statistics for different models."""
    def __init__(self, window_size=100):
        self._lock = threading.Lock()
        self.window_size = window_size
        self._stats = {
            'vad': deque(maxlen=window_size),
            'whisper': deque(maxlen=window_size),
            'diarizer': deque(maxlen=window_size),
            'pipeline': deque(maxlen=window_size)
        }
    
    def add_stat(self, model: str, latency: float):
        """Add a latency measurement for a model."""
        with self._lock:
            if model in self._stats:
                self._stats[model].append(latency)
    
    def get_stats(self):
        """Get current statistics for all models."""
        with self._lock:
            stats = {}
            for model, measurements in self._stats.items():
                if measurements:
                    stats[model] = {
                        'avg': sum(measurements) / len(measurements),
                        'last': measurements[-1] if measurements else 0,
                        'min': min(measurements),
                        'max': max(measurements)
                    }
                else:
                    stats[model] = None
            return stats

class PipelineStats:
    """Enhanced statistics tracking for the streaming pipeline."""
    
    def __init__(self, window_size: int = 1000):
        self._lock = threading.Lock()
        self.window_size = window_size
        
        # Component latencies
        self._latencies = {
            'vad': deque(maxlen=window_size),
            'whisper': deque(maxlen=window_size),
            'diarizer': deque(maxlen=window_size),
            'pipeline': deque(maxlen=window_size),
            'audio_capture': deque(maxlen=window_size)
        }
        
        # Throughput tracking
        self._throughput = {
            'audio_samples': deque(maxlen=window_size),
            'vad_chunks': deque(maxlen=window_size),
            'whisper_chunks': deque(maxlen=window_size)
        }
        
        # Queue monitoring
        self._queue_sizes = {
            'audio': deque(maxlen=window_size),
            'vad': deque(maxlen=window_size),
            'transcription': deque(maxlen=window_size),
            'diarization': deque(maxlen=window_size)
        }
        
        # Error tracking
        self._errors = {
            'vad': deque(maxlen=window_size),
            'whisper': deque(maxlen=window_size),
            'diarizer': deque(maxlen=window_size),
            'audio': deque(maxlen=window_size)
        }
        
        # Component state tracking
        self._state_changes = {
            'vad': deque(maxlen=window_size),
            'pipeline': deque(maxlen=window_size)
        }
        
        # Timestamps for rate calculations
        self._last_update = time.time()
        self._start_time = time.time()
    
    def _calculate_metrics(self, measurements: deque) -> Optional[LatencyMetrics]:
        """Calculate comprehensive metrics for a set of measurements."""
        if not measurements:
            return None
            
        sorted_vals = sorted(measurements)
        p95_idx = int(len(sorted_vals) * 0.95)
        
        return LatencyMetrics(
            avg=mean(measurements),
            median=median(measurements),
            p95=sorted_vals[p95_idx],
            min=min(measurements),
            max=max(measurements),
            last=measurements[-1],
            count=len(measurements)
        )
    
    def add_latency(self, component: str, latency: float):
        """Record a latency measurement for a component."""
        with self._lock:
            if component in self._latencies:
                self._latencies[component].append(latency)
    
    def add_throughput(self, component: str, samples: int):
        """Record throughput for a component."""
        with self._lock:
            if component in self._throughput:
                self._throughput[component].append(samples)
    
    def add_queue_size(self, queue_name: str, size: int):
        """Record queue size."""
        with self._lock:
            if queue_name in self._queue_sizes:
                self._queue_sizes[queue_name].append(size)
    
    def add_error(self, component: str, error: str):
        """Record an error for a component."""
        with self._lock:
            if component in self._errors:
                self._errors[component].append((time.time(), error))
    
    def add_state_change(self, component: str, state: str):
        """Record a state change for a component."""
        with self._lock:
            if component in self._state_changes:
                self._state_changes[component].append((time.time(), state))
    
    def get_component_stats(self, component: str) -> Dict[str, Any]:
        """Get comprehensive stats for a specific component."""
        with self._lock:
            stats = {}
            
            # Latency metrics
            if component in self._latencies:
                stats['latency'] = self._calculate_metrics(self._latencies[component])
            
            # Throughput
            if component in self._throughput:
                measurements = self._throughput[component]
                if measurements:
                    current_time = time.time()
                    time_window = current_time - self._last_update
                    if time_window > 0:
                        stats['throughput'] = sum(measurements) / time_window
            
            # Recent errors
            if component in self._errors:
                stats['recent_errors'] = list(self._errors[component])[-5:]
            
            # State changes
            if component in self._state_changes:
                stats['state_history'] = list(self._state_changes[component])[-5:]
            
            return stats
    
    def get_pipeline_health(self) -> Dict[str, Any]:
        """Get overall pipeline health metrics."""
        with self._lock:
            health = {
                'uptime': time.time() - self._start_time,
                'queue_stats': {},
                'error_counts': {},
                'component_status': {}
            }
            
            # Queue health
            for queue_name, sizes in self._queue_sizes.items():
                if sizes:
                    health['queue_stats'][queue_name] = {
                        'current': sizes[-1],
                        'avg': mean(sizes),
                        'max': max(sizes)
                    }
            
            # Error counts
            for component, errors in self._errors.items():
                health['error_counts'][component] = len(errors)
            
            # Component status
            for component, states in self._state_changes.items():
                if states:
                    health['component_status'][component] = states[-1][1]
            
            return health 