"""
Audio loading utilities
"""

import numpy as np
import librosa
from functools import lru_cache

@lru_cache(10**6)
def load_audio(fname):
    """Load an audio file into a numpy array."""
    a, _ = librosa.load(fname, sr=16000, dtype=np.float32)
    return a

def load_audio_chunk(fname, beg, end):
    """Load a chunk of an audio file into a numpy array."""
    audio = load_audio(fname)
    beg_s = int(beg*16000)
    end_s = int(end*16000)
    return audio[beg_s:end_s] 