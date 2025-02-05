import torch
import logging
from pyannote.audio import Pipeline
import warnings
from contextlib import contextmanager
import numpy as np
import faiss
import os
from speechbrain.inference.speaker import EncoderClassifier
import torchaudio
import torch.nn as nn
from speechbrain.lobes.features import Fbank
from speechbrain.processing.features import InputNormalization
from speechbrain.nnet.CNN import Conv1d
from speechbrain.nnet.pooling import StatisticsPooling
from speechbrain.nnet.linear import Linear
from speechbrain.nnet.normalization import BatchNorm1d
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
HF_TOKEN = os.getenv("HUGGING_FACE_HUB_TOKEN")

logger = logging.getLogger(__name__)

@contextmanager
def suppress_reproducibility_warning():
    """Context manager to suppress various warnings."""
    with warnings.catch_warnings():
        # PyAnnote TF32 warning
        warnings.filterwarnings(
            "ignore",
            message="TensorFloat-32 \\(TF32\\) has been disabled",
            category=UserWarning,
            module="pyannote.audio.utils.reproducibility"
        )
        
        # Standard deviation warning
        warnings.filterwarnings(
            "ignore",
            message="std\\(\\): degrees of freedom is <= 0",
            category=UserWarning
        )
        
        yield

class Xvector(torch.nn.Module):
    """TDNN-based x-vector architecture for speaker embedding extraction."""
    def __init__(
        self,
        in_channels=23,
        activation=torch.nn.LeakyReLU,
        tdnn_blocks=5,
        tdnn_channels=[512, 512, 512, 512, 1500],
        tdnn_kernel_sizes=[5, 3, 3, 1, 1],
        tdnn_dilations=[1, 2, 3, 1, 1],
        lin_neurons=512,
    ):
        super().__init__()
        self.blocks = nn.ModuleList()

        # TDNN layers
        for block_index in range(tdnn_blocks):
            out_channels = tdnn_channels[block_index]
            self.blocks.extend([
                Conv1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=tdnn_kernel_sizes[block_index],
                    dilation=tdnn_dilations[block_index],
                ),
                activation(),
                BatchNorm1d(input_size=out_channels),
            ])
            in_channels = tdnn_channels[block_index]

        # Statistical pooling
        self.blocks.append(StatisticsPooling())

        # Final linear transformation
        self.blocks.append(
            Linear(
                input_size=out_channels * 2,  # mean + std
                n_neurons=lin_neurons,
                bias=True,
                combine_dims=False,
            )
        )

    def forward(self, x, lengths=None):
        """Returns the x-vectors."""
        for layer in self.blocks:
            try:
                x = layer(x, lengths=lengths)
            except TypeError:
                x = layer(x)
        return x

class DiarizationModel:
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.pipeline = None
        
        # Set up model cache directories
        model_cache_root = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model_cache")
        self.diarization_cache_dir = os.path.join(model_cache_root, "pyannote_diarization")
        self.speechbrain_cache_dir = os.path.join(model_cache_root, "speechbrain_ecapa")
        self.embeddings_dir = os.path.join(model_cache_root, "speaker_embeddings")
        
        # Create cache directories
        os.makedirs(self.diarization_cache_dir, exist_ok=True)
        os.makedirs(self.speechbrain_cache_dir, exist_ok=True)
        os.makedirs(self.embeddings_dir, exist_ok=True)
        
        self._setup_torch_config()
        self._initialize_model()
        
        # Initialize FAISS index for speaker embeddings
        self.embedding_dim = 192  # ECAPA-TDNN embedding size
        try:
            # Use cosine similarity for ECAPA-TDNN
            self.speaker_index = faiss.IndexFlatIP(self.embedding_dim)
            # L2 normalize input vectors to get cosine similarity
            self.speaker_index = faiss.IndexIDMap(self.speaker_index)
            logger.info("FAISS index initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing FAISS index: {e}")
            raise
        
        # Initialize feature extraction and normalization
        self.feature_extractor = Fbank(n_mels=23).to(self.device)
        self.normalizer = InputNormalization().to(self.device)
        
        self.speaker_ids = []
        self.next_speaker_id = 0
        self.embedding_store = {}
        
        # Try to load existing embeddings
        embedding_path = os.path.join(self.embeddings_dir, "speaker_embeddings.pt")
        if os.path.exists(embedding_path):
            try:
                self.load_speaker_embeddings(embedding_path)
                logger.info(f"Loaded speaker embeddings from {embedding_path}")
            except Exception as e:
                logger.error(f"Error loading speaker embeddings: {e}")
    
    def _setup_torch_config(self):
        """Configure PyTorch settings for reproducibility."""
        if self.device == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            logger.info('TF32 has been disabled for reproducibility.')
    
    def _initialize_model(self):
        """Initialize the diarization pipeline and embedding model."""
        try:
            logger.debug('Loading models...')
            
            with suppress_reproducibility_warning():
                self.pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1",
                    use_auth_token=HF_TOKEN,
                    cache_dir=self.diarization_cache_dir
                ).to(torch.device(self.device))
                
                # Initialize SpeechBrain embedding model with new cache directory
                self.embedding_model = EncoderClassifier.from_hparams(
                    source="speechbrain/spkrec-ecapa-voxceleb",
                    savedir=self.speechbrain_cache_dir,
                    run_opts={"device": self.device}
                )
                
            logger.info(f'Models loaded successfully on {self.device}.')
        except Exception as e:
            logger.error(f'Error loading models: {e}')
            raise

    def _extract_embedding(self, waveform, start_sample, end_sample):
        """Extract embedding using SpeechBrain's ECAPA-TDNN model."""
        try:
            # Ensure minimum segment length (1s at 16kHz = 16000 samples)
            MIN_SAMPLES = 16000
            if end_sample - start_sample < MIN_SAMPLES:
                logger.warning("Segment too short for reliable embedding extraction")
                return None
            
            # Extract segment
            segment_waveform = waveform[:, start_sample:end_sample]
            
            # Extract embedding using SpeechBrain
            with torch.no_grad():
                try:
                    # ECAPA-TDNN expects normalized input
                    segment_length = segment_waveform.shape[1]
                    segment_waveform = segment_waveform / (torch.max(torch.abs(segment_waveform)) + 1e-8)
                    
                    # Get embedding
                    embeddings = self.embedding_model.encode_batch(segment_waveform)
                    embedding = embeddings.squeeze().cpu().numpy()
                    
                    # Convert to float32 and normalize for cosine similarity
                    embedding = embedding.astype(np.float32)
                    embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
                    
                    return embedding
                    
                except Exception as e:
                    logger.error(f"Error in embedding extraction: {e}", exc_info=True)
                    return None
                    
        except Exception as e:
            logger.error(f"Error extracting embedding: {e}", exc_info=True)
            return None

    def _match_speaker(self, embedding, threshold=0.75):
        """Match speaker embedding using cosine similarity."""
        try:
            if embedding is None:
                return None
            
            # Reshape and ensure normalized
            embedding = embedding.reshape(1, -1)
            
            if len(self.speaker_ids) == 0:
                # First speaker
                speaker_id = f"SPEAKER_{self.next_speaker_id:02d}"
                self.speaker_ids.append(speaker_id)
                self.speaker_index.add_with_ids(embedding, np.array([0]))  # Always use 0 for first speaker
                self.embedding_store[speaker_id] = [embedding[0].copy()]
                self.next_speaker_id += 1
                return speaker_id
            
            # Search for nearest speakers
            k = min(len(self.speaker_ids), 3)
            similarities, indices = self.speaker_index.search(embedding, k)
            
            if len(similarities) == 0 or len(indices) == 0:
                logger.warning("No similarities found in search")
                return f"SPEAKER_{self.next_speaker_id:02d}"
            
            similarities = similarities[0]
            indices = indices[0]
            
            best_similarity = similarities[0]
            best_index = indices[0]
            
            if best_index >= len(self.speaker_ids):
                logger.error(f"Index {best_index} out of range for speaker_ids (len={len(self.speaker_ids)})")
                return f"SPEAKER_{self.next_speaker_id:02d}"

            # Debug info
            logger.debug(f"Best similarity: {best_similarity:.3f}")
            if len(similarities) > 1:
                logger.debug(f"Second best similarity: {similarities[1]:.3f}")
            
            # More nuanced speaker detection
            if best_similarity < threshold:
                if self.next_speaker_id < 4:  # Limit to 4 speakers
                    # Check if this might be a short interjection
                    if len(embedding[0]) < 16000:  # Short segment
                        # For short segments, try to match with existing speakers
                        if best_similarity > 0.65:  # Still strict for short segments
                            speaker_id = self.speaker_ids[int(best_index)]
                            logger.info(f"Matched short segment with existing speaker {speaker_id} (similarity: {best_similarity:.3f})")
                            return speaker_id
                        elif len(similarities) > 1 and similarities[1] > 0.6:  # Check second best
                            speaker_id = self.speaker_ids[int(indices[1])]
                            logger.info(f"Matched short segment with second best speaker {speaker_id} (similarity: {similarities[1]:.3f})")
                            return speaker_id
                    
                    # For longer segments, be more conservative about creating new speakers
                    if best_similarity < 0.5:  # Only create new speaker if very different
                        speaker_id = f"SPEAKER_{self.next_speaker_id:02d}"
                        self.speaker_ids.append(speaker_id)
                        self.speaker_index.add_with_ids(embedding, np.array([self.next_speaker_id]))
                        self.embedding_store[speaker_id] = [embedding[0].copy()]
                        self.next_speaker_id += 1
                        logger.info(f"New speaker detected: {speaker_id} (similarity was {best_similarity:.3f})")
                        return speaker_id
                    else:
                        # Try to match with existing speaker
                        speaker_id = self.speaker_ids[int(best_index)]
                        logger.info(f"Matched with existing speaker {speaker_id} (similarity: {best_similarity:.3f})")
                        return speaker_id
                
                # If we already have max speakers, be more careful about matching
                if len(similarities) > 1 and similarities[1] > threshold - 0.05:  # Small threshold reduction
                    speaker_id = self.speaker_ids[int(indices[1])]
                    logger.info(f"Matched with second best speaker {speaker_id} (similarity: {similarities[1]:.3f})")
                    return speaker_id
                speaker_id = self.speaker_ids[int(best_index)]
                logger.info(f"Forced match with existing speaker {speaker_id} (similarity: {best_similarity:.3f})")
                return speaker_id
            
            # Match with existing speaker
            speaker_id = self.speaker_ids[int(best_index)]
            logger.info(f"Matched with existing speaker {speaker_id} (similarity: {best_similarity:.3f})")
            
            # Update embedding store with sliding window
            self.embedding_store[speaker_id].append(embedding[0].copy())
            if len(self.embedding_store[speaker_id]) > 8:  # Keep more history (8 instead of 5)
                self.embedding_store[speaker_id] = self.embedding_store[speaker_id][-8:]
            
            # Update index with weighted average
            try:
                self.speaker_index.remove_ids(np.array([best_index]))
                # More balanced weighting for better stability
                weights = np.array([0.85 ** i for i in range(len(self.embedding_store[speaker_id]))], dtype=np.float32)
                weights = weights / weights.sum()
                avg_embedding = np.average(self.embedding_store[speaker_id], axis=0, weights=weights).astype(np.float32)
                avg_embedding = avg_embedding / (np.linalg.norm(avg_embedding) + 1e-8)
                avg_embedding = avg_embedding.reshape(1, -1)
                self.speaker_index.add_with_ids(avg_embedding, np.array([best_index]))
                
            except Exception as e:
                logger.error(f"Error updating FAISS index: {e}", exc_info=True)
            
            return speaker_id
                
        except Exception as e:
            logger.error(f"Error in speaker matching: {e}", exc_info=True)
            return None

    def process_audio(self, audio_data, samplerate):
        """Process audio data and return diarization results."""
        try:
            logger.info('Running diarization on the audio...')
            # Ensure audio data is on the correct device
            waveform = torch.from_numpy(audio_data).unsqueeze(0).to(self.device)
            
            with suppress_reproducibility_warning():
                diarization = self.pipeline({
                    "waveform": waveform,
                    "sample_rate": samplerate
                })
            
            # Convert diarization result to a more usable format
            results = []
            
            # Auto-save embeddings periodically
            should_save = False
            
            # Group very short segments
            MIN_SEGMENT_DURATION = 0.5  # seconds
            current_speaker = None
            current_start = None
            current_end = None

            for turn, _, speaker in diarization.itertracks(yield_label=True):
                try:
                    duration = turn.end - turn.start
                    if duration < MIN_SEGMENT_DURATION:
                        logger.debug(f"Skipping short segment: {duration:.2f}s")
                        continue
                        
                    # Get embedding for this segment
                    segment_start_sample = int(turn.start * samplerate)
                    segment_end_sample = int(turn.end * samplerate)
                    segment_embedding = self._extract_embedding(waveform, segment_start_sample, segment_end_sample)
                    
                    # Match with existing speakers or create new one
                    matched_speaker = self._match_speaker(segment_embedding)
                    if matched_speaker is None:
                        matched_speaker = speaker  # Fallback to original label
                    
                    # Try to merge with previous segment if same speaker and close in time
                    if current_speaker == matched_speaker and turn.start - current_end < 0.5:
                        current_end = turn.end
                    else:
                        # Save previous segment if exists
                        if current_speaker is not None:
                            results.append({
                                'start': current_start,
                                'end': current_end,
                                'speaker': current_speaker,
                                'duration': current_end - current_start
                            })
                        current_speaker = matched_speaker
                        current_start = turn.start
                        current_end = turn.end
                    
                    logger.info(f'start={turn.start:.1f}s stop={turn.end:.1f}s {matched_speaker} (duration: {duration:.1f}s)')
                    should_save = True
                    
                except Exception as e:
                    logger.error(f'Error processing segment: {e}')
                    continue
            
            # Add final segment
            if current_speaker is not None:
                results.append({
                    'start': current_start,
                    'end': current_end,
                    'speaker': current_speaker,
                    'duration': current_end - current_start
                })
            
            # Auto-save embeddings if we processed any segments
            if should_save:
                self.save_speaker_embeddings(os.path.join(self.embeddings_dir, "speaker_embeddings.pt"))
            
            return results
            
        except Exception as e:
            logger.error(f'Error during diarization: {e}')
            raise

    def save_speaker_embeddings(self, path):
        """Save speaker embeddings to disk."""
        try:
            # Ensure all stored embeddings are normalized and float32
            normalized_store = {}
            for speaker_id, embeddings in self.embedding_store.items():
                normalized_store[speaker_id] = [
                    self._normalize_embedding(emb.astype(np.float32))[0] for emb in embeddings
                ]
            
            data = {
                'embeddings': normalized_store,
                'next_speaker_id': self.next_speaker_id
            }
            torch.save(data, path)
            logger.info(f"Saved speaker embeddings to {path}")
        except Exception as e:
            logger.error(f"Error saving speaker embeddings: {e}")
            
    def load_speaker_embeddings(self, path):
        """Load speaker embeddings from disk."""
        try:
            data = torch.load(path)
            self.embedding_store = data['embeddings']
            self.next_speaker_id = data['next_speaker_id']
            
            # Rebuild FAISS index and speaker_ids list
            self.speaker_index = faiss.IndexFlatIP(self.embedding_dim)
            self.speaker_index = faiss.IndexIDMap(self.speaker_index)
            
            # Clear and rebuild speaker_ids list
            self.speaker_ids = []
            
            # Add embeddings in order
            for speaker_id in sorted(self.embedding_store.keys()):
                embeddings = self.embedding_store[speaker_id]
                embeddings_float32 = [emb.astype(np.float32) for emb in embeddings]
                weights = np.array([0.8 ** i for i in range(len(embeddings_float32))], dtype=np.float32)
                weights = weights / weights.sum()
                avg_embedding = np.average(embeddings_float32, axis=0, weights=weights).astype(np.float32)
                
                # Ensure normalization
                avg_embedding = self._normalize_embedding(avg_embedding)
                if avg_embedding is not None:
                    index = len(self.speaker_ids)  # Use current length as index
                    self.speaker_index.add_with_ids(avg_embedding, np.array([index]))
                    self.speaker_ids.append(speaker_id)
            
            logger.info(f"Loaded {len(self.speaker_ids)} speakers from {path}")
        except Exception as e:
            logger.error(f"Error loading speaker embeddings: {e}")
            # Reset everything on error
            self.speaker_ids = []
            self.next_speaker_id = 0
            self.embedding_store = {}
            self.speaker_index = faiss.IndexFlatIP(self.embedding_dim)
            self.speaker_index = faiss.IndexIDMap(self.speaker_index)

    def _normalize_embedding(self, embedding):
        """Normalize embedding using L2 normalization."""
        try:
            if embedding is None:
                return None
            
            # Ensure float32 type and correct shape
            embedding = embedding.astype(np.float32)
            if len(embedding.shape) == 1:
                embedding = embedding.reshape(1, -1)
            
            # Normalize
            faiss.normalize_L2(embedding)
            return embedding
            
        except Exception as e:
            logger.error(f"Error normalizing embedding: {e}")
            return None 