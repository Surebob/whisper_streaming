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

# Import from models
from models.vad import VADModel
from models.diarization import DiarizationModel
from models.stats import ModelStatsMonitor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("main.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

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
    def __init__(self, model_name="large-v3", language="en", compute_type="int8_float16", min_chunk_size=1.0, batch_size=8):
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
        
        # Warm up the model with real audio
        print(f"{Fore.YELLOW}Warming up model with real audio...{Style.RESET_ALL}")
        try:
            warmup_audio = load_audio_chunk("lex.wav", 0, 5)  # Load first 5 seconds
            self.processor.insert_audio_chunk(warmup_audio)
            self.processor.process_iter()  # Process it silently
            
            # Reset processor after warmup with same settings
            self.processor = OnlineASRProcessor(
                self.asr,
                buffer_trimming=("segment", 15.0)
            )
            print(f"{Fore.GREEN}Warmup complete!{Style.RESET_ALL}")
        except Exception as e:
            print(f"{Fore.YELLOW}Warmup file not found, using dummy audio...{Style.RESET_ALL}")
            dummy = np.zeros(16000 * 5, dtype=np.float32)  # 5 seconds of silence
            self.asr.transcribe(dummy)
        
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
                
                # Non-blocking get
                try:
                    audio_chunk = self.audio_queue.get_nowait()
                    vad_buffer.extend(audio_chunk)  # Extend instead of append for flat array
                    whisper_buffer.append(audio_chunk)
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
                    
                    # Time VAD inference
                    vad_start = time.perf_counter()
                    prob = self.vad_model.model(
                        audio_tensor.unsqueeze(0),  # Add batch dimension
                        self.vad_model.sampling_rate
                    ).item()
                    vad_time = time.perf_counter() - vad_start
                    
                    # Update voice probability tracking
                    self.voice_prob = prob
                    self.voice_prob_history.append(prob)
                    
                    # Record VAD timing
                    if hasattr(self, 'model_stats'):
                        self.model_stats.add_stat('vad', vad_time)
                    
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
                        # Time Whisper processing
                        whisper_start = time.perf_counter()
                        
                        # Check buffer size before appending
                        total_samples = sum(len(chunk) for chunk in self.audio_buffer) + len(audio_chunk)
                        if total_samples <= self.max_buffer_size:
                            # Accumulate audio
                            self.audio_buffer.append(audio_chunk)
                        else:
                            logger.warning("Audio buffer full, dropping new audio chunk")
                        
                        # Process transcription
                        self.processor.insert_audio_chunk(audio_chunk)
                        result = self.processor.process_iter()
                        
                        whisper_time = time.perf_counter() - whisper_start
                        if hasattr(self, 'model_stats'):
                            self.model_stats.add_stat('whisper', whisper_time)
                        
                        if result[0] is not None:
                            _, _, text = result
                            if text and isinstance(text, str):
                                new_text = text.strip()
                                if new_text and not self.live_transcript.endswith(new_text):
                                    self.live_transcript += " " + new_text
                                    self.live_transcript = self.live_transcript.strip()
                                    self.current_segment = self.live_transcript
                    
                    # Check if speech just ended
                    elif len(self.audio_buffer) > 0:
                        segment_audio = np.concatenate(self.audio_buffer)
                        segment_text = self.current_segment.strip()
                        if segment_text and segment_text != self.last_processed_text:
                            self.last_processed_text = segment_text
                            self.diarization_queue.put((segment_audio, segment_text))
                        
                        # Clear buffers
                        self.audio_buffer = []
                        self.current_segment = ""
                        self.live_transcript = ""
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in transcription processing: {str(e)}")
    
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
            logger.info(f"{Fore.CYAN}Starting microphone processing...{Style.RESET_ALL}")
            
            # Start processing threads
            self.start_processing_threads()
            
            def audio_callback(indata, frames, time, status):
                if status and not status.input_overflow:
                    return
                
                # Put audio chunk in queue without blocking
                try:
                    self.audio_queue.put_nowait(indata[:, 0].astype(np.float32))
                except queue.Full:
                    # Drop frame if queue is full
                    pass
            
            # Create stream with stable chunk size
            self.stream = sd.InputStream(
                channels=1,
                samplerate=SAMPLING_RATE,
                blocksize=CAPTURE_CHUNK_SIZE,  # Use larger chunks for stable capture
                dtype=np.float32,
                callback=audio_callback,
                latency='low'
            )
            
            self.stream.start()
            print(f"{Fore.GREEN}Listening... Press Ctrl+C to stop{Style.RESET_ALL}")
            
            while self.running:
                time.sleep(0.1)  # Main thread sleep
                
        except Exception as e:
            logger.error(f"Error in microphone processing: {str(e)}", exc_info=True)
            raise
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Cleanup resources."""
        self.running = False
        self.thread_running = False
        
        # Wait for threads to finish
        for thread in self.threads:
            thread.join(timeout=1.0)
        
        # Close microphone stream properly
        if self.stream is not None and self.stream.active:
            self.stream.stop()
            self.stream.close()
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
    parser = argparse.ArgumentParser(description="Streaming transcription with diarization")
    
    # Input arguments
    parser.add_argument("--use-mic", action="store_true", help="Use microphone input")
    
    # Model configuration
    parser.add_argument("--model", default="large-v3-turbo", 
                      choices=['tiny.en','tiny','base.en','base','small.en','small',
                              'medium.en','medium','large-v1','large-v2','large-v3','large',
                              'large-v3-turbo','distil-large-v3'],
                      help="Name of the Whisper model to use")
    parser.add_argument("--language", default="en", help="Language code")
    parser.add_argument("--compute-type", choices=['int8_float16', 'float16', 'float32'],
                      default='int8_float16', help="Model computation type")
    parser.add_argument("--min-chunk-size", type=float, default=1.0,
                      help="Minimum chunk size in seconds")
    parser.add_argument("--batch-size", type=int, default=8,
                      help="Batch size for model inference")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = StreamingPipeline(
        model_name=args.model,
        language=args.language,
        compute_type=args.compute_type,
        min_chunk_size=args.min_chunk_size,
        batch_size=args.batch_size
    )
    
    try:
        # Process audio
        if args.use_mic:
            pipeline.process_microphone()
        else:
            parser.error("This implementation only supports microphone input")
            
    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW}Interrupted by user{Style.RESET_ALL}")
    finally:
        pipeline.cleanup()

if __name__ == "__main__":
    main() 