"""
Audio loading utilities with performance monitoring and benchmarking
"""

import numpy as np
import librosa
import time
import os
import sounddevice as sd
import pyaudiowpatch as pyaudio
from functools import lru_cache
from typing import Optional, Tuple, Callable, Union
from models.stats import PipelineStats
from utils.benchmark import ComponentBenchmark, BenchmarkContext
from utils.logger import get_logger

# Initialize logger
logger = get_logger(__name__, console_output=False)

# Initialize component benchmark
_benchmark = ComponentBenchmark("audio")

# Constants
SAMPLING_RATE = 16000
CAPTURE_CHUNK_SIZE = 1024  # Stable audio capture size (64ms)

def create_audio_stream(callback: Callable, use_system_audio: bool = False) -> Union[sd.InputStream, pyaudio.Stream]:
    """Create an audio input stream for either microphone or system audio."""
    try:
        if use_system_audio:
            p = pyaudio.PyAudio()
            
            try:
                # Get WASAPI info
                wasapi_info = p.get_host_api_info_by_type(pyaudio.paWASAPI)
            except OSError:
                raise RuntimeError("WASAPI is not available on the system")
            
            # Get default speakers
            default_speakers = p.get_device_info_by_index(wasapi_info["defaultOutputDevice"])
            logger.info(f"Default Output Device: {default_speakers['name']}")
            
            # Find loopback device
            loopback_device = None
            for device_index in range(p.get_device_count()):
                device_info = p.get_device_info_by_index(device_index)
                if device_info.get('isLoopbackDevice', False):
                    logger.info(f"Found loopback device: {device_info['name']}")
                    loopback_device = device_info
                    break
            
            if not loopback_device:
                raise RuntimeError("No loopback device found")
            
            logger.info(f"Recording from: {loopback_device['name']}")
            logger.info(f"Sample Rate: {int(loopback_device['defaultSampleRate'])}Hz")
            logger.info(f"Channels: {loopback_device['maxInputChannels']}")
            
            def wrapped_callback(in_data, frame_count, time_info, status):
                """Wrapper callback to handle audio format conversion."""
                try:
                    # Convert to numpy array
                    audio_data = np.frombuffer(in_data, dtype=np.int16)
                    
                    # Reshape if multi-channel
                    if loopback_device['maxInputChannels'] > 1:
                        audio_data = audio_data.reshape(-1, loopback_device['maxInputChannels'])
                        # Convert to mono by averaging channels
                        audio_data = audio_data.mean(axis=1)
                    
                    # Convert to float32 in range [-1, 1]
                    audio_data = audio_data.astype(np.float32) / 32768.0
                    
                    # Resample to target rate if needed
                    if int(loopback_device['defaultSampleRate']) != SAMPLING_RATE:
                        from scipy import signal
                        samples_needed = int(len(audio_data) * SAMPLING_RATE / loopback_device['defaultSampleRate'])
                        audio_data = signal.resample(audio_data, samples_needed)
                    
                    # Call the original callback
                    if callback(audio_data, frame_count, time_info, status) == (None, pyaudio.paComplete):
                        return (None, pyaudio.paComplete)
                    
                    return (None, pyaudio.paContinue)
                    
                except Exception as e:
                    logger.error(f"Error in audio callback: {e}")
                    return (None, pyaudio.paComplete)
            
            # Create stream for system audio
            chunk_size = 1024 * 4  # Larger chunk size for stability
            stream = p.open(
                format=pyaudio.paInt16,
                channels=loopback_device['maxInputChannels'],
                rate=int(loopback_device['defaultSampleRate']),
                frames_per_buffer=chunk_size,
                input=True,
                input_device_index=loopback_device['index'],
                stream_callback=wrapped_callback
            )
            
            # Start the stream immediately
            stream.start_stream()
            return stream
            
        else:
            # Use default input device for microphone
            default_input = sd.query_devices(kind='input')
            device_index = default_input['index']
            device_name = default_input['name']
            logger.info(f"Using microphone device: {device_name}")
            
            # Create stream for microphone
            stream = sd.InputStream(
                device=device_index,
                channels=1,
                samplerate=SAMPLING_RATE,
                blocksize=CAPTURE_CHUNK_SIZE,
                dtype=np.float32,
                callback=callback,
                latency='low'
            )
            
            return stream
        
    except Exception as e:
        _benchmark.record_event("error", {
            "operation": "create_stream",
            "device_type": "system" if use_system_audio else "microphone",
            "error": str(e)
        })
        raise

def load_audio(fname: str, stats_monitor: Optional[PipelineStats] = None) -> np.ndarray:
    """Load an audio file into a numpy array with performance monitoring."""
    try:
        file_size = os.path.getsize(fname)
        context = {
            "file_name": fname,
            "file_size_mb": file_size / (1024 * 1024),
            "cache_hit": fname in _load_audio_cached.cache_info().hits
        }
        
        with BenchmarkContext(_benchmark, "load_audio", file_size, context) as bench:
            start = time.perf_counter()
            audio = _load_audio_cached(fname)
            
            if stats_monitor:
                load_time = time.perf_counter() - start
                stats_monitor.add_latency('audio_load', load_time)
                stats_monitor.add_throughput('audio_samples', len(audio))
            
            # Record audio properties in benchmark
            bench.context.update({
                "sample_count": len(audio),
                "duration_seconds": len(audio) / 16000,
                "max_amplitude": float(np.max(np.abs(audio))),
                "rms_level": float(np.sqrt(np.mean(audio**2)))
            })
            
            return audio
            
    except Exception as e:
        if stats_monitor:
            stats_monitor.add_error('audio', f"Error loading {fname}: {str(e)}")
        _benchmark.record_event("error", {
            "operation": "load_audio",
            "file": fname,
            "error": str(e)
        })
        raise

@lru_cache(10**6)
def _load_audio_cached(fname: str) -> np.ndarray:
    """Cached audio loading function."""
    a, _ = librosa.load(fname, sr=16000, dtype=np.float32)
    return a

def load_audio_chunk(fname: str, beg: float, end: float, 
                    stats_monitor: Optional[PipelineStats] = None) -> np.ndarray:
    """Load a chunk of an audio file with performance monitoring."""
    try:
        context = {
            "file_name": fname,
            "chunk_start": beg,
            "chunk_end": end,
            "chunk_duration": end - beg
        }
        
        with BenchmarkContext(_benchmark, "load_audio_chunk", 0, context) as bench:
            # Load full audio (will use cache if available)
            audio = load_audio(fname, stats_monitor)
            
            # Extract chunk
            beg_s = int(beg * 16000)
            end_s = int(end * 16000)
            chunk = audio[beg_s:end_s]
            
            # Update benchmark context with actual chunk info
            bench.data_size = len(chunk)
            bench.context.update({
                "chunk_samples": len(chunk),
                "max_amplitude": float(np.max(np.abs(chunk))),
                "rms_level": float(np.sqrt(np.mean(chunk**2))),
                "cache_hit": fname in _load_audio_cached.cache_info().hits
            })
            
            if stats_monitor:
                chunk_time = time.perf_counter() - bench.start_time
                stats_monitor.add_latency('audio_chunk', chunk_time)
                stats_monitor.add_throughput('audio_chunks', len(chunk))
            
            return chunk
            
    except Exception as e:
        if stats_monitor:
            stats_monitor.add_error('audio', f"Error loading chunk {beg}-{end}s from {fname}: {str(e)}")
        _benchmark.record_event("error", {
            "operation": "load_audio_chunk",
            "file": fname,
            "chunk": f"{beg}-{end}s",
            "error": str(e)
        })
        raise

def validate_audio(audio: np.ndarray, stats_monitor: Optional[PipelineStats] = None) -> Tuple[bool, str]:
    """Validate audio data and format."""
    try:
        context = {
            "array_shape": audio.shape,
            "dtype": str(audio.dtype),
            "memory_size_mb": audio.nbytes / (1024 * 1024)
        }
        
        with BenchmarkContext(_benchmark, "validate_audio", audio.nbytes, context) as bench:
            if not isinstance(audio, np.ndarray):
                return False, "Audio must be a numpy array"
            
            if audio.dtype != np.float32:
                return False, f"Audio must be float32, got {audio.dtype}"
            
            if len(audio.shape) != 1:
                return False, f"Audio must be mono (1D), got shape {audio.shape}"
            
            has_nan = np.isnan(audio).any()
            has_inf = np.isinf(audio).any()
            
            bench.context.update({
                "has_nan": has_nan,
                "has_inf": has_inf,
                "min_value": float(np.min(audio)),
                "max_value": float(np.max(audio)),
                "mean_value": float(np.mean(audio)),
                "std_value": float(np.std(audio))
            })
            
            if has_nan:
                return False, "Audio contains NaN values"
            
            if has_inf:
                return False, "Audio contains Inf values"
            
            if stats_monitor:
                stats_monitor.add_state_change('audio', 'valid')
            
            return True, "Audio validation passed"
            
    except Exception as e:
        if stats_monitor:
            stats_monitor.add_error('audio', f"Validation error: {str(e)}")
            stats_monitor.add_state_change('audio', 'invalid')
        _benchmark.record_event("error", {
            "operation": "validate_audio",
            "error": str(e)
        })
        return False, str(e) 