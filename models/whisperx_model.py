import whisperx
import torch
import logging
import numpy as np
from typing import Dict, List, Optional
import os
from pyannote.audio.pipelines.speaker_diarization import SpeakerDiarization

logger = logging.getLogger(__name__)

class WhisperXModel:
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self._setup_torch_config()
        self._initialize_model()
        
    def _setup_torch_config(self):
        torch.set_grad_enabled(False)
        if self.device == "cuda":
            torch.cuda.is_available()
            torch.backends.cudnn.benchmark = True
        self._setup_torch_config_old()
        
    def _setup_torch_config_old(self):
        """Configure PyTorch settings for reproducibility."""
        if self.device == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            logger.info('TF32 has been disabled for reproducibility.')
    
    def _initialize_model(self):
        """Initialize the WhisperX model."""
        compute_type = "float16" if self.device == "cuda" else "float32"
        
        logger.info("Loading WhisperX model...")
        self.model = whisperx.load_model("large-v2", self.device, compute_type=compute_type)
        logger.info("WhisperX model loaded successfully")

    def process_audio(self, audio_data: np.ndarray, sample_rate: int = 16000, diarization_segments=None) -> Dict:
        """Process audio with WhisperX for transcription only, using traditional diarization results.
        
        Args:
            audio_data: Audio data as numpy
            sample_rate: Sample rate of audio
            diarization_segments: Pre-computed diarization segments from traditional model (required)
        """
        try:
            # Transcribe with word-level timestamps
            logger.info('Running WhisperX transcription...')
            result = self.model.transcribe(audio_data, batch_size=16)
            
            if not diarization_segments:
                logger.warning('No diarization segments provided, transcription will not have speaker labels')
                return {
                    'segments': result['segments'],
                    'word_segments': result.get('word_segments', []),
                    'speaker_segments': []
                }
            
            # Add speaker labels to the transcription segments using traditional diarization
            for segment in result['segments']:
                segment_mid = (segment['start'] + segment['end']) / 2
                # Find the speaker active at this time
                for spk in diarization_segments:
                    if spk['start'] <= segment_mid <= spk['end']:
                        segment['speaker'] = spk['speaker']
                        break
            
            logger.info('WhisperX transcription completed successfully')
            return {
                'segments': result['segments'],
                'word_segments': result.get('word_segments', []),
                'speaker_segments': diarization_segments
            }
            
        except Exception as e:
            logger.error(f'Error in WhisperX processing: {e}')
            return {'segments': [], 'word_segments': [], 'speaker_segments': []}
            
    def process_segments(self, segments: List[Dict], sample_rate: int = 16000) -> List[Dict]:
        """Process audio segments with WhisperX."""
        if not segments:
            return []
            
        # Process each segment individually
        results = []
        for segment in segments:
            if 'audio' in segment:
                try:
                    result = self.process_audio(segment['audio'], sample_rate)
                    # Adjust timestamps to account for segment start time
                    for seg in result['segments']:
                        seg['start'] += segment.get('start', 0)
                        seg['end'] += segment.get('start', 0)
                    for seg in result['speaker_segments']:
                        seg['start'] += segment.get('start', 0)
                        seg['end'] += segment.get('start', 0)
                    
                    results.append({
                        'start': segment.get('start', 0),
                        'end': segment.get('end', 0),
                        'result': result
                    })
                except Exception as e:
                    logger.error(f'Error processing segment: {e}')
                    
        return results 