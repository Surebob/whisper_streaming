import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple
import time

@dataclass
class PauseEvent:
    """Represents a pause in speech."""
    start_time: float
    end_time: Optional[float] = None
    
    @property
    def duration(self) -> Optional[float]:
        """Get the duration of the pause if it has ended."""
        if self.end_time is not None:
            return self.end_time - self.start_time
        return None

@dataclass
class SpeechSegment:
    """Represents a segment of speech."""
    start_time: float
    end_time: Optional[float] = None
    
    @property
    def duration(self) -> Optional[float]:
        """Get the duration of the speech segment if it has ended."""
        if self.end_time is not None:
            return self.end_time - self.start_time
        return None

class CircularAudioBuffer:
    """Manages a circular buffer for audio data with fixed maximum duration."""
    
    def __init__(self, max_duration: float, sample_rate: int):
        """Initialize the circular buffer.
        
        Args:
            max_duration: Maximum duration in seconds to store
            sample_rate: Audio sample rate in Hz
        """
        self.max_duration = max_duration
        self.sample_rate = sample_rate
        self.max_samples = int(max_duration * sample_rate)
        self.buffer = np.zeros(self.max_samples, dtype=np.float32)
        self.write_pos = 0
        self.total_samples = 0
    
    def add_chunk(self, chunk: np.ndarray) -> None:
        """Add a new chunk of audio data to the buffer.
        
        Args:
            chunk: Audio data chunk as numpy array
        """
        chunk_size = len(chunk)
        
        # If chunk is larger than buffer, only keep the last max_samples
        if chunk_size > self.max_samples:
            chunk = chunk[-self.max_samples:]
            chunk_size = len(chunk)
        
        # Calculate positions for writing
        space_until_end = self.max_samples - self.write_pos
        
        if chunk_size <= space_until_end:
            # Simple case: chunk fits before end of buffer
            self.buffer[self.write_pos:self.write_pos + chunk_size] = chunk
            self.write_pos = (self.write_pos + chunk_size) % self.max_samples
        else:
            # Chunk wraps around end of buffer
            first_part = chunk[:space_until_end]
            second_part = chunk[space_until_end:]
            
            self.buffer[self.write_pos:] = first_part
            self.buffer[:len(second_part)] = second_part
            self.write_pos = len(second_part)
        
        self.total_samples = min(self.total_samples + chunk_size, self.max_samples)
    
    def get_last_n_seconds(self, duration: float) -> np.ndarray:
        """Get the last N seconds of audio from the buffer.
        
        Args:
            duration: Duration in seconds to retrieve
            
        Returns:
            numpy array containing the requested audio data
        """
        n_samples = min(int(duration * self.sample_rate), self.total_samples)
        if n_samples == 0:
            return np.array([], dtype=np.float32)
        
        if self.write_pos >= n_samples:
            # Simple case: data is contiguous
            return self.buffer[self.write_pos - n_samples:self.write_pos]
        else:
            # Data wraps around buffer end
            first_part = self.buffer[self.max_samples - (n_samples - self.write_pos):]
            second_part = self.buffer[:self.write_pos]
            return np.concatenate([first_part, second_part])
    
    def clear(self) -> None:
        """Clear the buffer."""
        self.buffer.fill(0)
        self.write_pos = 0
        self.total_samples = 0

class SpeechTracker:
    """Tracks speech segments and pauses in audio stream."""
    
    def __init__(self, silence_threshold: float = 1.5):
        """Initialize the speech tracker.
        
        Args:
            silence_threshold: Duration in seconds to consider as a pause
        """
        self.silence_threshold = silence_threshold
        self.is_speaking = False
        self.speech_segments: List[SpeechSegment] = []
        self.pauses: List[PauseEvent] = []
        self.current_segment: Optional[SpeechSegment] = None
        self.current_pause: Optional[PauseEvent] = None
        self.last_update_time = None
    
    def update(self, is_speech: bool, current_time: float) -> Tuple[bool, Optional[float]]:
        """Update speech tracking state.
        
        Args:
            is_speech: Whether speech is currently detected
            current_time: Current timestamp
            
        Returns:
            Tuple of (should_process, remaining_pause_time)
        """
        if self.last_update_time is None:
            self.last_update_time = current_time
        
        should_process = False
        remaining_pause = None

        if is_speech:
            # Handle start of speech
            if not self.is_speaking:
                self.is_speaking = True
                if self.current_pause:
                    self.current_pause.end_time = current_time
                    self.pauses.append(self.current_pause)
                    self.current_pause = None
                self.current_segment = SpeechSegment(start_time=current_time)
        else:
            # Handle end of speech
            if self.is_speaking:
                self.is_speaking = False
                if self.current_segment:
                    self.current_segment.end_time = current_time
                    self.speech_segments.append(self.current_segment)
                    self.current_segment = None
                self.current_pause = PauseEvent(start_time=current_time)
            
            # Update pause duration
            if self.current_pause:
                pause_duration = current_time - self.current_pause.start_time
                remaining_pause = max(0.0, self.silence_threshold - pause_duration)
                
                # Check if we should process
                if pause_duration >= self.silence_threshold:
                    should_process = True
        
        self.last_update_time = current_time
        return should_process, remaining_pause
    
    def clear(self) -> None:
        """Clear all tracking state."""
        self.is_speaking = False
        self.speech_segments.clear()
        self.pauses.clear()
        self.current_segment = None
        self.current_pause = None
        self.last_update_time = None 