"""
Whisper ASR model implementation using faster-whisper backend
"""

import sys
import time
import logging
from collections import deque
from statistics import mean, median
from faster_whisper import WhisperModel, BatchedInferencePipeline

logger = logging.getLogger(__name__)

class LatencyStats:
    def __init__(self, window_size=100):
        self.audio_latencies = deque(maxlen=window_size)
        self.processing_latencies = deque(maxlen=window_size)
        self.total_latencies = deque(maxlen=window_size)
        self.last_audio_timestamp = None
        self.last_process_start = None
        
    def start_audio_capture(self):
        self.last_audio_timestamp = time.perf_counter()
        
    def audio_received(self):
        if self.last_audio_timestamp:
            audio_latency = time.perf_counter() - self.last_audio_timestamp
            self.audio_latencies.append(audio_latency)
        self.last_process_start = time.perf_counter()
        
    def transcription_complete(self, text):
        now = time.perf_counter()
        if self.last_process_start:
            process_latency = now - self.last_process_start
            self.processing_latencies.append(process_latency)
        if self.last_audio_timestamp:
            total_latency = now - self.last_audio_timestamp
            self.total_latencies.append(total_latency)
            
    def get_stats(self):
        stats = {
            "Audio Latency (ms)": {
                "mean": mean(self.audio_latencies) * 1000 if self.audio_latencies else 0,
                "median": median(self.audio_latencies) * 1000 if self.audio_latencies else 0
            },
            "Processing Latency (ms)": {
                "mean": mean(self.processing_latencies) * 1000 if self.processing_latencies else 0,
                "median": median(self.processing_latencies) * 1000 if self.processing_latencies else 0
            },
            "Total Latency (ms)": {
                "mean": mean(self.total_latencies) * 1000 if self.total_latencies else 0,
                "median": median(self.total_latencies) * 1000 if self.total_latencies else 0
            }
        }
        return stats

class ASRBase:
    sep = " "   # join transcribe words with this character

    def __init__(self, lan, modelsize=None, cache_dir=None, model_dir=None, logfile=sys.stderr):
        self.logfile = logfile
        self.transcribe_kargs = {}
        if lan == "auto":
            self.original_language = None
        else:
            self.original_language = lan
        self.model = self.load_model(modelsize, cache_dir, model_dir)

    def load_model(self, modelsize, cache_dir):
        raise NotImplemented("must be implemented in the child class")

    def transcribe(self, audio, init_prompt=""):
        raise NotImplemented("must be implemented in the child class")

    def use_vad(self):
        raise NotImplemented("must be implemented in the child class")

class FasterWhisperASR(ASRBase):
    """Uses faster-whisper library as the backend. Works much faster, appx 4-times (in offline mode).
    For GPU, it requires installation with a specific CUDNN version.
    """
    sep = ""

    def __init__(self, *args, batch_size=8, **kwargs):
        self.batch_size = batch_size
        super().__init__(*args, **kwargs)
        self.latency_stats = LatencyStats()

    def load_model(self, modelsize=None, cache_dir=None, model_dir=None):
        if model_dir is not None:
            logger.debug(f"Loading whisper model from model_dir {model_dir}. modelsize and cache_dir parameters are not used.")
            model_size_or_path = model_dir
        elif modelsize is not None:
            model_size_or_path = modelsize
        else:
            raise ValueError("modelsize or model_dir parameter must be set")

        # Initialize base model with int8 quantization
        base_model = WhisperModel(
            model_size_or_path, 
            device="cuda", 
            compute_type="int8_float16", 
            download_root=cache_dir,
            num_workers=8,  # Match performance cores count
            cpu_threads=16  # Use half of logical processors to avoid oversubscription
        )
        # Wrap with BatchedInferencePipeline for batch processing
        self.batch_model = BatchedInferencePipeline(model=base_model)
        return base_model

    def transcribe(self, audio, init_prompt=""):
        self.latency_stats.start_audio_capture()
        self.latency_stats.audio_received()
        
        # Use batch_model for transcription when possible
        if hasattr(self, 'batch_model'):
            segments, info = self.batch_model.transcribe(
                audio, 
                language=self.original_language, 
                initial_prompt=init_prompt, 
                beam_size=5, 
                word_timestamps=True, 
                condition_on_previous_text=True,
                batch_size=self.batch_size,
                **self.transcribe_kargs
            )
        else:
            # Fallback to regular model if batch_model isn't available
            segments, info = self.model.transcribe(
                audio, 
                language=self.original_language, 
                initial_prompt=init_prompt, 
                beam_size=5, 
                word_timestamps=True, 
                condition_on_previous_text=True,
                **self.transcribe_kargs
            )
        
        segments_list = list(segments)
        if segments_list:
            self.latency_stats.transcription_complete(segments_list[-1].text if segments_list else "")
            
            # Print latency stats every 10 segments
            if len(segments_list) % 10 == 0:
                stats = self.latency_stats.get_stats()
                print("\nLatency Statistics:")
                for metric, values in stats.items():
                    print(f"{metric}:")
                    print(f"  Mean: {values['mean']:.2f} ms")
                    print(f"  Median: {values['median']:.2f} ms")
                print()
                
        return segments_list

    def ts_words(self, segments):
        o = []
        for segment in segments:
            for word in segment.words:
                if segment.no_speech_prob > 0.9:
                    continue
                # not stripping the spaces -- should not be merged with them!
                w = word.word
                t = (word.start, word.end, w)
                o.append(t)
        return o

    def segments_end_ts(self, res):
        return [s.end for s in res]

    def use_vad(self):
        self.transcribe_kargs["vad_filter"] = True

    def set_translate_task(self):
        self.transcribe_kargs["task"] = "translate" 