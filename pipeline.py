#!/usr/bin/env python3
import os
import sys
import time
import logging
import argparse
import warnings
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from collections import deque
import torch
import sounddevice as sd
from colorama import init, Fore, Style
from faster_whisper import WhisperModel, BatchedInferencePipeline
import threading
import queue
from utils.logger import configure_logging, get_logger
from utils.cli import parse_args
from utils.benchmark import ComponentBenchmark, BenchmarkContext
from utils.audio import create_audio_stream
import pyaudio

# Import from models
from models.vad import VADModel
from models.diarization import DiarizationModel
from models.stats import ModelStatsMonitor

# Initialize logging
configure_logging(log_file="main.log", console_output=True)
logger = get_logger(__name__)

# Suppress warnings but log them to file
warnings.filterwarnings('ignore')
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

# Configure module-specific logging
for module in ["faster_whisper", "models.diarization", "speechbrain", "pyannote"]:
    module_logger = logging.getLogger(module)
    module_logger.setLevel(logging.INFO)
    module_logger.handlers = []  # Remove existing handlers
    module_logger.addHandler(logging.FileHandler("main.log"))

# Import whisper components
from models.whisper.asr import FasterWhisperASR
from models.whisper.processor import OnlineASRProcessor, HypothesisBuffer
from utils.audio import load_audio_chunk, load_audio

# Initialize colorama
init(autoreset=True)

# Constants
SAMPLING_RATE = 16000
VAD_WINDOW_SIZE = 512  # 32ms chunks for VAD (optimal for Silero VAD)
CAPTURE_CHUNK_SIZE = 1024  # Stable audio capture size (64ms)
WHISPER_CHUNK_SIZE = int(SAMPLING_RATE * 1.0)  # 1 second chunks for Whisper

class StreamingPipeline:
    def __init__(self, model_name="large-v3", language="en", compute_type="int8_float16", min_chunk_size=1.0, batch_size=8, enable_benchmark=False):
        # Initialize benchmark if enabled
        if enable_benchmark:
            self.audio_benchmark = ComponentBenchmark("audio")
            self.vad_benchmark = ComponentBenchmark("vad")
            self.whisper_benchmark = ComponentBenchmark("whisper")
        else:
            self.audio_benchmark = None
            self.vad_benchmark = None
            self.whisper_benchmark = None
        
        # Initialize state tracking
        self.running = True
        self.is_speaking = False
        self.silence_start = None
        self.silence_threshold = 2.0  # seconds
        self.current_speaker = None
        self.live_transcript = ""  # For live display
        self.current_segment = ""  # Buffer for current speech segment
        self.audio_buffer = []  # Only for final segment processing
        self.whisper_buffer = []  # Separate buffer for whisper
        self.transcript_history = []
        self.last_processed_text = ""  # Track last processed text to avoid duplicates
        self.last_whisper_time = 0  # Track last whisper processing time
        self.min_whisper_interval = 0.1  # Minimum time between whisper processing
        self.stream = None  # Store microphone stream
        
        # Voice activity tracking
        self.voice_prob = 0.0  # Current voice probability
        self.voice_prob_history = deque(maxlen=10)  # Keep last 10 probabilities for smoothing
        
        # Buffer management
        self.max_buffer_size = int(SAMPLING_RATE * 30)  # Maximum 30 seconds of audio
        
        # Thread-safe queues for pipeline communication
        self.audio_queue = queue.Queue(maxsize=100)  # Raw audio chunks
        self.vad_queue = queue.Queue(maxsize=100)    # VAD results
        self.transcription_queue = queue.Queue()  # Text to transcribe
        self.diarization_queue = queue.Queue()   # Audio to diarize
        
        # Thread control
        self.threads = []
        self.thread_running = True
        
        # Thread-safe state variables
        self._state_lock = threading.Lock()
        self._vad_state = {
            'is_speaking': False,
            'silence_start': None,
            'last_update_time': time.time()
        }
        
        # Initialize model stats monitor
        self.model_stats = ModelStatsMonitor()
        
        try:
            # Initialize models and parameters
            self.model_name = model_name
            self.language = language
            self.compute_type = compute_type
            self.min_chunk_size = min_chunk_size
            self.batch_size = batch_size
            
            # Disable logging to console for certain modules
            logging.getLogger("faster_whisper").setLevel(logging.ERROR)
            logging.getLogger("models.diarization").setLevel(logging.ERROR)
            
            # Initialize components
            self._init_models()
            
            logger.info(f"{Fore.GREEN}Pipeline initialized successfully{Style.RESET_ALL}")
        except Exception as e:
            logger.error(f"Error initializing pipeline: {str(e)}", exc_info=True)
            raise
    
    def _init_models(self):
        """Initialize all models."""
        logger.info(f"{Fore.CYAN}Initializing models...{Style.RESET_ALL}")
        
        # Initialize VAD
        self.vad_model = VADModel(
            sampling_rate=SAMPLING_RATE,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        
        # Initialize Whisper exactly like in stream_whisper.py
        print(f"{Fore.GREEN}Initializing model {self.model_name}...{Style.RESET_ALL}")
        print(f"{Fore.GREEN}Language set to: {self.language}{Style.RESET_ALL}")
        
        self.asr = FasterWhisperASR(
            lan=self.language,
            modelsize=self.model_name,
            batch_size=self.batch_size
        )
        
        # Verify language setting
        print(f"{Fore.YELLOW}Verifying language setting: {self.asr.original_language}{Style.RESET_ALL}")
        
        # Override the default model initialization with our compute type
        base_model = WhisperModel(
            self.model_name,
            device="cuda" if torch.cuda.is_available() else "cpu",
            compute_type=self.compute_type,
            download_root=None,
            num_workers=2,
            cpu_threads=4
        )
        self.asr.model = base_model
        self.asr.batch_model = BatchedInferencePipeline(model=base_model)
        
        # Initialize streaming processor
        self.processor = OnlineASRProcessor(
            self.asr,
            buffer_trimming=("segment", 15.0)
        )
        
        # Warm up the model with dummy audio
        print(f"{Fore.YELLOW}Warming up model...{Style.RESET_ALL}")
        try:
            # Try to load warmup file if it exists
            warmup_file = "warmup.wav"
            if os.path.exists(warmup_file):
                warmup_audio = load_audio_chunk(warmup_file, 0, 5)  # Load first 5 seconds
                print(f"{Fore.GREEN}Using warmup file: {warmup_file}{Style.RESET_ALL}")
            else:
                # Generate synthetic audio for warmup
                print(f"{Fore.YELLOW}Using synthetic audio for warmup{Style.RESET_ALL}")
                warmup_audio = np.zeros(16000 * 5, dtype=np.float32)  # 5 seconds of silence
                # Add some noise to make it more realistic
                warmup_audio += np.random.normal(0, 0.01, warmup_audio.shape)
            
            self.processor.insert_audio_chunk(warmup_audio)
            self.processor.process_iter()  # Process it silently
            
            # Reset processor after warmup with same settings
            self.processor = OnlineASRProcessor(
                self.asr,
                buffer_trimming=("segment", 15.0)
            )
            print(f"{Fore.GREEN}Warmup complete!{Style.RESET_ALL}")
            
        except Exception as e:
            logger.warning(f"Error during warmup: {str(e)}, continuing without warmup")
            # No need to raise the error, we can continue without warmup
        
        # Initialize Diarization
        self.diarization = DiarizationModel(device="cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info(f"{Fore.GREEN}All models loaded successfully{Style.RESET_ALL}")
    
    def start_processing_threads(self):
        """Start all processing threads."""
        # VAD processing thread
        vad_thread = threading.Thread(target=self._vad_processing_loop, daemon=True)
        self.threads.append(vad_thread)
        
        # Transcription processing thread
        transcription_thread = threading.Thread(target=self._transcription_processing_loop, daemon=True)
        self.threads.append(transcription_thread)
        
        # Diarization processing thread
        diarization_thread = threading.Thread(target=self._diarization_processing_loop, daemon=True)
        self.threads.append(diarization_thread)
        
        # Start all threads
        for thread in self.threads:
            thread.start()
    
    def _vad_processing_loop(self):
        """Continuous VAD processing loop."""
        vad_buffer = []  # Local buffer for VAD processing
        whisper_buffer = []  # Separate buffer for Whisper
        device = self.vad_model.device  # Get VAD model device
        
        while self.thread_running:
            try:
                # Time pipeline processing
                pipeline_start = time.perf_counter()
                
                # Non-blocking get with benchmarking
                with BenchmarkContext(self.vad_benchmark, "process", 0) as bench:
                    try:
                        audio_chunk = self.audio_queue.get_nowait()
                        vad_buffer.extend(audio_chunk)  # Extend instead of append for flat array
                        whisper_buffer.append(audio_chunk)
                        bench.data_size = len(audio_chunk)
                    except queue.Empty:
                        if len(vad_buffer) == 0:  # Check length directly
                            time.sleep(0.001)  # Tiny sleep if no data
                            continue
                    
                    # Process VAD when we have enough data (512 samples)
                    while len(vad_buffer) >= VAD_WINDOW_SIZE:
                        # Get 512 samples for VAD
                        vad_audio = vad_buffer[:VAD_WINDOW_SIZE]
                        vad_buffer = vad_buffer[VAD_WINDOW_SIZE:]  # Keep remainder
                        
                        # Process VAD with 512-sample window
                        audio_tensor = torch.from_numpy(np.array(vad_audio)).float().to(device)
                        if audio_tensor.max() > 1.0 or audio_tensor.min() < -1.0:
                            audio_tensor = audio_tensor / max(abs(audio_tensor.max()), abs(audio_tensor.min()))
                        
                        # Time VAD inference with benchmark
                        with BenchmarkContext(self.vad_benchmark, "inference", len(vad_audio)) as vad_bench:
                            prob = self.vad_model.model(
                                audio_tensor.unsqueeze(0),  # Add batch dimension
                                self.vad_model.sampling_rate
                            ).item()
                            vad_bench.context.update({
                                "voice_probability": prob,
                                "is_speech": prob > 0.5
                            })
                        
                        # Update voice probability tracking
                        self.voice_prob = prob
                        self.voice_prob_history.append(prob)
                        
                        # Record VAD timing
                        if hasattr(self, 'model_stats'):
                            self.model_stats.add_stat('vad', time.perf_counter() - pipeline_start)
                        
                        is_speech = prob > 0.5
                        current_time = time.time()
                        
                        # Update state with minimal locking
                        with self._state_lock:
                            state = self._vad_state
                            if is_speech:
                                # Always reset silence counter when speech is detected
                                state['silence_start'] = None
                                self.silence_start = None
                                
                                if not state['is_speaking']:
                                    state['is_speaking'] = True
                                    self.is_speaking = True
                                    logger.info("Speech detected")  # Debug log
                                state['last_update_time'] = current_time
                            else:
                                if state['is_speaking']:
                                    if state['silence_start'] is None:
                                        state['silence_start'] = current_time
                                        self.silence_start = current_time
                                        logger.info("Silence started")  # Debug log
                                    elif current_time - state['silence_start'] > self.silence_threshold:
                                        state['is_speaking'] = False
                                        self.is_speaking = False
                                        logger.info("Speech ended")  # Debug log
                
                # Process Whisper when we have enough data (1 second)
                if whisper_buffer:  # Only process if we have data
                    whisper_data = np.concatenate(whisper_buffer)
                    whisper_samples = len(whisper_data)
                    
                    if whisper_samples >= WHISPER_CHUNK_SIZE:
                        # Keep only the remainder after 1 second
                        if whisper_samples > WHISPER_CHUNK_SIZE:
                            whisper_buffer = [whisper_data[WHISPER_CHUNK_SIZE:]]
                        else:
                            whisper_buffer = []
                        
                        # Send 1-second chunk to transcription
                        try:
                            self.vad_queue.put_nowait((self.is_speaking, whisper_data[:WHISPER_CHUNK_SIZE]))
                        except queue.Full:
                            # Drop frame if queue is full to prevent lag
                            pass
                
                # Record pipeline timing
                pipeline_time = time.perf_counter() - pipeline_start
                self.model_stats.add_stat('pipeline', pipeline_time)
                
            except Exception as e:
                logger.error(f"Error in VAD processing: {str(e)}")
                vad_buffer = []  # Reset buffer on error
                whisper_buffer = []
    
    def _transcription_processing_loop(self):
        """Continuous transcription processing loop."""
        while self.thread_running:
            try:
                # Get audio from queue with timeout
                is_speech, audio_chunk = self.vad_queue.get(timeout=0.1)
                
                # Process transcription if speaking
                with self._state_lock:
                    if self.is_speaking:
                        # Time Whisper processing with benchmark
                        with BenchmarkContext(self.whisper_benchmark, "process", len(audio_chunk), {
                            "is_speech": is_speech,
                            "chunk_size": len(audio_chunk)
                        }) as bench:
                            # Check buffer size before appending
                            total_samples = sum(len(chunk) for chunk in self.audio_buffer) + len(audio_chunk)
                            if total_samples <= self.max_buffer_size:
                                # Accumulate audio
                                self.audio_buffer.append(audio_chunk)
                            else:
                                logger.warning("Audio buffer full, dropping new audio chunk")
                            
                            # Process transcription with benchmark
                            with BenchmarkContext(self.whisper_benchmark, "inference", len(audio_chunk)) as whisper_bench:
                                self.processor.insert_audio_chunk(audio_chunk)
                                result = self.processor.process_iter()
                                
                                if result[0] is not None:
                                    _, _, text = result
                                    whisper_bench.context.update({
                                        "text_length": len(text) if text else 0,
                                        "has_result": result[0] is not None
                                    })
                            
                            if result[0] is not None:
                                _, _, text = result
                                if text and isinstance(text, str):
                                    new_text = text.strip()
                                    if new_text and not self.live_transcript.endswith(new_text):
                                        self.live_transcript += " " + new_text
                                        self.live_transcript = self.live_transcript.strip()
                                        self.current_segment = self.live_transcript
                                        bench.context["transcribed_text"] = new_text
                        
                    # Check if speech just ended
                    elif len(self.audio_buffer) > 0:
                        with BenchmarkContext(self.whisper_benchmark, "finalize", sum(len(c) for c in self.audio_buffer), {
                            "final_text_length": len(self.current_segment),
                            "total_audio_samples": sum(len(c) for c in self.audio_buffer),
                            "silence_duration": time.time() - self.silence_start if self.silence_start else 0,
                            "is_valid_segment": len(self.current_segment.strip()) > 0
                        }) as bench:
                            segment_audio = np.concatenate(self.audio_buffer)
                            segment_text = self.current_segment.strip()
                            
                            # Only process if we have valid text
                            if segment_text and segment_text != self.last_processed_text:
                                self.last_processed_text = segment_text
                                self.diarization_queue.put((segment_audio, segment_text))
                                bench.context.update({
                                    "final_text_length": len(segment_text),
                                    "total_audio_samples": len(segment_audio),
                                    "words_per_second": len(segment_text.split()) / (len(segment_audio) / SAMPLING_RATE)
                                })
                            
                            # Clear buffers
                            self.audio_buffer = []
                            self.current_segment = ""
                            self.live_transcript = ""
            
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in transcription processing: {str(e)}")
                self.whisper_benchmark.record_event("error", {
                    "operation": "transcription",
                    "error": str(e)
                })
    
    def _diarization_processing_loop(self):
        """Continuous diarization processing loop."""
        while self.thread_running:
            try:
                # Get segment from queue with timeout
                segment_audio, segment_text = self.diarization_queue.get(timeout=0.1)
                
                # Time diarization processing
                diar_start = time.perf_counter()
                
                # Process diarization
                diarization_results = self.diarization.process_audio(segment_audio, samplerate=SAMPLING_RATE)
                
                diar_time = time.perf_counter() - diar_start
                if hasattr(self, 'model_stats'):
                    self.model_stats.add_stat('diarizer', diar_time)
                
                if diarization_results:
                    diar_segment = diarization_results[0]
                    self.transcript_history.append({
                        'text': segment_text,
                        'speaker': diar_segment['speaker'],
                        'start': diar_segment['start'],
                        'end': diar_segment['end'],
                        'timestamp': time.time()
                    })
                    
                    # Keep only last 100 segments
                    if len(self.transcript_history) > 100:
                        self.transcript_history = self.transcript_history[-100:]
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in diarization processing: {str(e)}")
    
    def process_microphone(self):
        """Process audio from microphone input."""
        try:
            logger.info("Starting microphone processing...")
            
            # Start processing threads
            self.start_processing_threads()
            
            # Create microphone stream
            self.stream = create_audio_stream(self._audio_callback, use_system_audio=False)
            
            self.stream.start()
            print(f"\n{Fore.GREEN}Ready! Listening to microphone input... Press Ctrl+C to stop{Style.RESET_ALL}")
            
            while self.running:
                time.sleep(0.1)  # Main thread sleep
                
        except Exception as e:
            logger.error(f"Error in microphone processing: {str(e)}", exc_info=True)
            raise
        finally:
            self.cleanup()

    def process_system_audio(self):
        """Process system audio output by recording from default output device."""
        try:
            logger.info("Starting system audio capture...")
            
            # Start processing threads
            self.start_processing_threads()
            
            # Create system audio stream (starts automatically)
            self.stream = create_audio_stream(self._audio_callback, use_system_audio=True)
            
            print(f"\n{Fore.GREEN}Ready! Recording system audio... Press Ctrl+C to stop{Style.RESET_ALL}")
            
            while self.running:
                time.sleep(0.1)  # Main thread sleep
                
        except Exception as e:
            logger.error(f"Error in system audio processing: {str(e)}", exc_info=True)
            raise
        finally:
            self.cleanup()

    def _audio_callback(self, indata, frames, time_info, status):
        """Common audio callback for both microphone and system audio."""
        try:
            # For sounddevice (microphone)
            if isinstance(status, sd.CallbackFlags):
                if status and not status.input_overflow:
                    return
                audio_data = indata[:, 0].astype(np.float32)
            # For pyaudiowpatch (system audio)
            else:
                # Audio data is already processed in the wrapped callback
                audio_data = indata
            
            # Benchmark audio capture
            with BenchmarkContext(self.audio_benchmark, "capture", len(audio_data), {
                "frames": frames,
                "status": str(status) if status else None
            }) as bench:
                bench.context["max_amplitude"] = float(np.max(np.abs(audio_data)))
                bench.context["rms_level"] = float(np.sqrt(np.mean(audio_data**2)))
                
                # Put audio chunk in queue without blocking
                try:
                    self.audio_queue.put_nowait(audio_data)
                except queue.Full:
                    # Drop frame if queue is full
                    self.audio_benchmark.record_event("drop", {"reason": "queue_full"})
                    pass
                
                # For pyaudiowpatch, we need to return the callback continuation flag
                if not isinstance(status, sd.CallbackFlags):
                    return (None, pyaudio.paContinue)
        
        except Exception as e:
            logger.error(f"Error in audio callback: {str(e)}")
            if not isinstance(status, sd.CallbackFlags):
                return (None, pyaudio.paComplete)

    def cleanup(self):
        """Cleanup resources."""
        self.running = False
        self.thread_running = False
        
        # Wait for threads to finish
        for thread in self.threads:
            thread.join(timeout=1.0)
        
        # Close audio stream properly
        if self.stream is not None:
            try:
                # For sounddevice stream
                if hasattr(self.stream, 'active'):
                    if self.stream.active:
                        self.stream.stop()
                        self.stream.close()
                # For PyAudio stream
                else:
                    if self.stream.is_active():
                        self.stream.stop_stream()
                    self.stream.close()
            except Exception as e:
                logger.error(f"Error closing stream: {e}")
            self.stream = None
        
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)
        
        # Cleanup models
        if hasattr(self, 'vad_model'):
            self.vad_model.reset()
        if hasattr(self, 'diarization'):
            del self.diarization
        
        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser()
    args = parse_args("pipeline", parser)
    
    # Initialize pipeline
    pipeline = StreamingPipeline(
        model_name=args.model,
        language=args.language,
        compute_type=args.compute_type,
        min_chunk_size=args.min_chunk_size,
        batch_size=args.batch_size,
        enable_benchmark=args.benchmark
    )
    
    try:
        # Process audio based on input source
        if args.use_mic:
            pipeline.process_microphone()
        elif args.use_system_audio:
            pipeline.process_system_audio()
        else:
            parser.error("Must specify either --use-mic or --use-system-audio")
            
    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW}Interrupted by user{Style.RESET_ALL}")
    finally:
        pipeline.cleanup()

if __name__ == "__main__":
    main() 