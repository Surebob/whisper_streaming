import torch
import logging
from typing import List, Dict, Optional
import numpy as np
import os

logger = logging.getLogger(__name__)

class VADModel:
    def __init__(self, sampling_rate: int = 16000, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """Initialize Silero VAD model."""
        self.sampling_rate = sampling_rate
        self.device = torch.device(device)
        self.model = None
        self.vad_iterator = None
        self.window_size_samples = 512 if sampling_rate == 16000 else 256  # Optimal size based on sample rate
        self.max_speech_duration = 10.0  # Maximum duration to accumulate before processing
        
        # Use root model_cache directory for torch hub
        self.model_cache_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model_cache")
        os.makedirs(self.model_cache_dir, exist_ok=True)
        torch.hub.set_dir(self.model_cache_dir)
        
        self._initialize_model()
    
    def _initialize_model(self):
        """Load and initialize the Silero VAD model."""
        try:
            logger.info("Loading Silero VAD model...")
            torch.set_num_threads(1)  # Better for real-time processing
            
            # Load model and utils with correct opset version
            self.model, utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False,
                onnx=False,
                trust_repo=True
            )
            self.model.to(self.device)
            
            # Get utility functions
            (self.get_speech_timestamps,
             _,  # save_audio
             _,  # read_audio
             VADIterator,
             self.collect_chunks) = utils
            
            # Initialize VAD iterator for streaming
            self.vad_iterator = VADIterator(self.model, sampling_rate=self.sampling_rate)
            
            logger.info(f"Silero VAD model loaded successfully on {self.device}")
        except Exception as e:
            logger.error(f"Error loading Silero VAD model: {e}")
            raise
    
    def process_chunk(self, audio_chunk: np.ndarray, threshold: float = 0.5) -> Dict:
        """
        Process a single chunk of audio.
        Returns speech detection result with timestamp information.
        """
        try:
            # Skip processing if chunk is too small
            if len(audio_chunk) < self.window_size_samples:
                logger.debug("Chunk too small for processing")
                return {}
                
            # Ensure chunk size matches window_size_samples
            if len(audio_chunk) != self.window_size_samples:
                # Pad or truncate to match window size
                if len(audio_chunk) < self.window_size_samples:
                    audio_chunk = np.pad(audio_chunk, (0, self.window_size_samples - len(audio_chunk)))
                else:
                    audio_chunk = audio_chunk[:self.window_size_samples]
            
            # Convert numpy array to torch tensor and move to correct device
            audio_tensor = torch.from_numpy(audio_chunk).float().to(self.device)
            
            # Normalize audio if needed
            if audio_tensor.max() > 1.0 or audio_tensor.min() < -1.0:
                audio_tensor = audio_tensor / max(abs(audio_tensor.max()), abs(audio_tensor.min()))
            
            # Get speech probability
            speech_prob = self.model(audio_tensor, self.sampling_rate).item()
            
            if speech_prob > threshold:
                logger.debug(f"Speech detected with probability: {speech_prob:.3f}")
                return {'start': 0, 'end': len(audio_chunk) / self.sampling_rate, 'probability': speech_prob}
            
            return {}
            
        except Exception as e:
            logger.error(f"Error processing audio chunk in VAD: {e}")
            self.reset()  # Reset states on error
            raise
    
    def process_audio(self, audio_data: np.ndarray) -> List[Dict]:
        """
        Process complete audio file or larger segment.
        Returns list of speech segments with timestamps.
        """
        try:
            logger.debug(f"Processing audio segment of length {len(audio_data)} samples")
            
            # Convert numpy array to torch tensor and move to correct device
            audio_tensor = torch.from_numpy(audio_data).float().to(self.device)
            
            # Normalize audio if needed
            if audio_tensor.max() > 1.0 or audio_tensor.min() < -1.0:
                audio_tensor = audio_tensor / max(abs(audio_tensor.max()), abs(audio_tensor.min()))
            
            # First pass - get initial segments with more sensitive settings
            initial_timestamps = self.get_speech_timestamps(
                audio_tensor,
                self.model,
                sampling_rate=self.sampling_rate,
                return_seconds=True,
                threshold=0.2,  # Very sensitive to catch all potential speech
                min_speech_duration_ms=500,  # Shorter duration to catch everything
                min_silence_duration_ms=400,
                window_size_samples=self.window_size_samples,
                speech_pad_ms=1000  # Large padding to help with merging
            )
            
            # First merge pass - combine very close segments
            if initial_timestamps:
                merged = []
                current = initial_timestamps[0].copy()
                
                for next_seg in initial_timestamps[1:]:
                    gap = next_seg['start'] - current['end']
                    # Merge if gap is less than 1 second
                    if gap < 1.0:
                        current['end'] = next_seg['end']
                    else:
                        merged.append(current)
                        current = next_seg.copy()
                merged.append(current)
                
                # Second pass - filter and clean up segments
                final_segments = []
                for segment in merged:
                    duration = segment['end'] - segment['start']
                    
                    # Skip segments that are too short
                    if duration < 1.0:
                        logger.debug(f"Skipping short segment: {duration:.2f}s")
                        continue
                    
                    # For longer segments, ensure they're not too long
                    if duration > 20.0:  # Max segment length of 20 seconds
                        # Split into smaller segments
                        num_splits = int(duration / 10.0) + 1
                        segment_duration = duration / num_splits
                        for i in range(num_splits):
                            start = segment['start'] + (i * segment_duration)
                            end = start + segment_duration
                            final_segments.append({
                                'start': start,
                                'end': end
                            })
                    else:
                        final_segments.append(segment)
                
                timestamps = final_segments
            else:
                timestamps = []
            
            if timestamps:
                logger.debug(f"Found {len(timestamps)} speech segments")
                for i, ts in enumerate(timestamps):
                    duration = ts['end'] - ts['start']
                    logger.debug(f"Speech segment {i+1}: {ts['start']:.2f}s to {ts['end']:.2f}s (duration: {duration:.2f}s)")
            
            return timestamps
            
        except Exception as e:
            logger.error(f"Error processing audio in VAD: {e}")
            raise
        finally:
            self.reset()
    
    def reset(self):
        """Reset the VAD iterator states."""
        if hasattr(self, 'vad_iterator') and self.vad_iterator:
            self.vad_iterator.reset_states()
        if hasattr(self, 'model') and hasattr(self.model, 'reset_states'):
            self.model.reset_states()
    
    def __del__(self):
        """Cleanup when object is deleted."""
        try:
            self.reset()
        except:
            pass  # Ignore cleanup errors during deletion 