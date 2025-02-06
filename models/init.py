"""
Centralized model initialization and setup module.
"""

import os
import torch
import logging
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any
from colorama import Fore, Style
from dotenv import load_dotenv
from faster_whisper import WhisperModel, BatchedInferencePipeline

from models.vad import VADModel
from models.diarization import DiarizationModel
from models.whisper.asr import FasterWhisperASR
from models.whisper.processor import OnlineASRProcessor
from utils.audio import load_audio_chunk
from utils.logger import get_logger, configure_logging

# Configure logging first
configure_logging(log_file="main.log", console_output=True)

# Initialize logger
logger = get_logger(__name__, console_output=True)

# Load environment variables
load_dotenv()

class ModelManager:
    def __init__(self, 
                 model_name: str = "large-v3",
                 language: str = "en",
                 compute_type: str = "int8_float16",
                 batch_size: int = 8,
                 device: Optional[str] = None):
        """Initialize the model manager.
        
        Args:
            model_name: Name/size of the Whisper model
            language: Language code for transcription
            compute_type: Model computation type
            batch_size: Batch size for inference
            device: Device to use (cuda/cpu), if None will auto-detect
        """
        self.model_name = model_name
        self.language = language
        self.compute_type = compute_type
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Set up model cache directories
        self.model_cache_root = Path("model_cache")
        self.model_cache_root.mkdir(parents=True, exist_ok=True)
        
        # Initialize models
        self.vad_model = None
        self.whisper_model = None
        self.diarization_model = None
        self.asr_processor = None
    
    def setup_vad(self) -> VADModel:
        """Initialize and set up VAD model."""
        logger.info("Initializing VAD model...")
        self.vad_model = VADModel(
            sampling_rate=16000,
            device=self.device
        )
        return self.vad_model
    
    def setup_whisper(self) -> tuple[FasterWhisperASR, OnlineASRProcessor]:
        """Initialize and set up Whisper model and processor."""
        logger.info(f"Initializing Whisper model {self.model_name}...")
        logger.info(f"Language set to: {self.language}")
        
        # Initialize ASR
        asr = FasterWhisperASR(
            lan=self.language,
            modelsize=self.model_name,
            batch_size=self.batch_size
        )
        
        # Override with specific compute type
        base_model = WhisperModel(
            self.model_name,
            device=self.device,
            compute_type=self.compute_type,
            download_root=str(self.model_cache_root / "whisper"),
            num_workers=2,
            cpu_threads=4
        )
        asr.model = base_model
        asr.batch_model = BatchedInferencePipeline(model=base_model)
        
        # Initialize processor
        processor = OnlineASRProcessor(
            asr,
            buffer_trimming=("segment", 15.0)
        )
        
        # Warm up the model
        self._warmup_whisper(processor)
        
        self.whisper_model = asr
        self.asr_processor = processor
        return asr, processor
    
    def _warmup_whisper(self, processor: OnlineASRProcessor):
        """Warm up the Whisper model with dummy audio."""
        logger.info("Warming up Whisper model...")
        try:
            # Try to load warmup file if it exists
            warmup_file = "warmup.wav"
            if os.path.exists(warmup_file):
                warmup_audio = load_audio_chunk(warmup_file, 0, 5)
                logger.info(f"Using warmup file: {warmup_file}")
            else:
                # Generate synthetic audio for warmup
                logger.info("Using synthetic audio for warmup")
                warmup_audio = np.zeros(16000 * 5, dtype=np.float32)
                warmup_audio += np.random.normal(0, 0.01, warmup_audio.shape)
            
            processor.insert_audio_chunk(warmup_audio)
            processor.process_iter()
            
            # Reset processor after warmup
            processor = OnlineASRProcessor(
                processor.asr,
                buffer_trimming=("segment", 15.0)
            )
            logger.info("Warmup complete!")
            
        except Exception as e:
            logger.warning(f"Error during warmup: {str(e)}, continuing without warmup")
    
    def setup_diarization(self) -> DiarizationModel:
        """Initialize and set up diarization model."""
        logger.info("Initializing diarization model...")
        self.diarization_model = DiarizationModel(device=self.device)
        return self.diarization_model
    
    def setup_all(self) -> Dict[str, Any]:
        """Initialize all models and return them in a dictionary."""
        try:
            # Configure logging levels
            for module in ["faster_whisper", "models.diarization", "speechbrain", "pyannote"]:
                module_logger = logging.getLogger(module)
                module_logger.setLevel(logging.INFO)
            
            # Main initialization header
            print(f"\n{Fore.CYAN}╔═════════════════════════════════════════════╗")
            print(f"║      {Fore.GREEN}Initializing Whisper Components{Fore.CYAN}        ║")
            print(f"╚═════════════════════════════════════════════╝{Style.RESET_ALL}\n")
            
            # Initialize VAD with box drawing
            print(f"{Fore.CYAN}┌─ {Fore.GREEN}Voice Activity Detection{Fore.CYAN} ─┐")
            vad = self.setup_vad()
            print(f"{Fore.CYAN}└──────────────────────────┘{Style.RESET_ALL}\n")
            
            # Initialize Whisper with box drawing
            print(f"{Fore.CYAN}┌─ {Fore.GREEN}Whisper ASR{Fore.CYAN} ─┐")
            asr, processor = self.setup_whisper()
            print(f"{Fore.CYAN}└───────────────┘{Style.RESET_ALL}\n")
            
            # Initialize Diarization with box drawing
            print(f"{Fore.CYAN}┌─ {Fore.GREEN}Speaker Diarization{Fore.CYAN} ─┐")
            diarization = self.setup_diarization()
            print(f"{Fore.CYAN}└─────────────────────┘{Style.RESET_ALL}\n")
            
            # Success message with double-line box
            print(f"{Fore.CYAN}╔═════════════════════════════════════════════╗")
            print(f"║  {Fore.GREEN}✓ All Components Initialized Successfully{Fore.CYAN}  ║")
            print(f"╚═════════════════════════════════════════════╝{Style.RESET_ALL}\n")
            
            return {
                "vad": vad,
                "asr": asr,
                "processor": processor,
                "diarization": diarization
            }
            
        except Exception as e:
            logger.error(f"Error initializing models: {str(e)}", exc_info=True)
            raise
    
    def cleanup(self):
        """Clean up all models and free resources."""
        try:
            logger.info("Starting model cleanup...")
            
            # Clean up VAD
            if self.vad_model:
                try:
                    self.vad_model.reset()
                    del self.vad_model
                except:
                    pass
                self.vad_model = None
            
            # Clean up Whisper
            if self.whisper_model:
                try:
                    del self.whisper_model
                except:
                    pass
                self.whisper_model = None
            
            if self.asr_processor:
                try:
                    del self.asr_processor
                except:
                    pass
                self.asr_processor = None
            
            # Clean up Diarization
            if self.diarization_model:
                try:
                    del self.diarization_model
                except:
                    pass
                self.diarization_model = None
            
            # Force CUDA cache cleanup
            if torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                except:
                    pass
            
            # Force garbage collection
            import gc
            gc.collect()
            
            logger.info("Model cleanup completed successfully")
            
        except Exception as e:
            logger.error(f"Error during model cleanup: {str(e)}")
        finally:
            # Ensure all models are set to None even if cleanup fails
            self.vad_model = None
            self.whisper_model = None
            self.asr_processor = None
            self.diarization_model = None 

def initialize_models(args):
    # Configure logging first
    configure_logging(log_file="main.log", console_output=True)

    print(f"\n{Fore.CYAN}╔═════════════════════════════════════════════╗")
    print(f"║      {Fore.GREEN}Initializing Whisper Components{Fore.CYAN}        ║")
    print(f"╚═════════════════════════════════════════════╝{Style.RESET_ALL}\n")

    # Initialize VAD
    print(f"{Fore.CYAN}┌─ {Fore.GREEN}Voice Activity Detection{Fore.CYAN} ─┐")
    vad_model = VADModel()
    print(f"{Fore.CYAN}└──────────────────────────┘{Style.RESET_ALL}\n")

    # Initialize Whisper
    print(f"{Fore.CYAN}┌─ {Fore.GREEN}Whisper ASR{Fore.CYAN} ─┐")
    asr_model = WhisperModel(
        model_size_or_path=args.model,
        device="cuda" if torch.cuda.is_available() else "cpu",
        compute_type=args.compute_type,
        download_root=os.path.join(os.getcwd(), "model_cache")
    )
    print(f"{Fore.CYAN}└───────────────┘{Style.RESET_ALL}\n")

    # Initialize Speaker Diarization
    print(f"{Fore.CYAN}┌─ {Fore.GREEN}Speaker Diarization{Fore.CYAN} ─┐")
    diarization = DiarizationModel()
    print(f"{Fore.CYAN}└─────────────────────┘{Style.RESET_ALL}\n")

    print(f"{Fore.CYAN}╔═════════════════════════════════════════════╗")
    print(f"║  {Fore.GREEN}✓ All Components Initialized Successfully{Fore.CYAN}  ║")
    print(f"╚═════════════════════════════════════════════╝{Style.RESET_ALL}\n")

    return vad_model, asr_model, diarization 