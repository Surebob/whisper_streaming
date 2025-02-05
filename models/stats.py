import threading
from collections import deque

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