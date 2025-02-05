# Whisper Streaming Transcription

A real-time streaming transcription system with voice activity detection (VAD), speaker diarization, and a rich terminal UI. The system leverages Whisper models through the Faster-Whisper backend for efficient transcription.

## Features

- **Real-time Transcription**: Low-latency streaming transcription using Faster-Whisper
- **Voice Activity Detection**: Silero VAD for accurate speech detection
- **Speaker Diarization**: Pyannote-based speaker diarization with ECAPA-TDNN embeddings
- **Rich Terminal UI**: Interactive display with:
  - Live transcription view
  - Transcript history with speaker labels
  - System performance metrics
  - Voice activity visualization
  - Model status monitoring
- **Performance Optimization**:
  - Efficient audio buffering and processing
  - Multi-threaded pipeline architecture
  - GPU acceleration support
  - Latency monitoring and statistics

## Project Structure

```
└── whisper_streaming/
    ├── pipeline.py           # Core streaming pipeline implementation
    ├── pipeline_ui.py        # Terminal UI and visualization
    ├── silero_vad_iterator.py # VAD utilities
    ├── whisper_env.yml       # Conda environment specification
    ├── models/
    │   ├── buffer_manager.py # Audio buffer management
    │   ├── diarization.py    # Speaker diarization
    │   ├── stats.py         # Performance monitoring
    │   ├── vad.py           # Voice activity detection
    │   ├── visualizer.py    # UI visualization components
    │   └── whisper/         # Whisper ASR components
    │       ├── asr.py       # ASR model implementation
    │       └── processor.py # Online ASR processing
    └── utils/
        └── audio.py         # Audio loading utilities
```

## Installation

1. Create a Conda environment using the provided environment file:
   ```bash
   conda env create -f whisper_env.yml
   conda activate whisper_env
   ```

2. Set up environment variables:
   Create a `.env` file with your Hugging Face token for diarization model access:
   ```
   HUGGING_FACE_HUB_TOKEN=your_token_here
   ```

## Usage

### Terminal UI (pipeline_ui.py)

The UI provides a rich interactive display with real-time transcription, speaker tracking, and system monitoring.

```bash
python pipeline_ui.py --use-mic [options]
```

Options:
- `--model`: Whisper model size (default: large-v3-turbo)
  - Available: tiny.en, tiny, base.en, base, small.en, small, medium.en, medium, large-v1/v2/v3, large-v3-turbo, distil-large-v3
- `--compute-type`: Model computation type (default: int8_float16)
  - Options: int8_float16, float16, float32, int8
- `--min-chunk-size`: Minimum audio chunk size in seconds (default: 1.0)
- `--batch-size`: Batch size for inference (default: 8)

### Core Pipeline (pipeline.py)

For headless operation or integration into other applications:

```bash
python pipeline.py --use-mic [options]
```

Additional options:
- `--language`: Target language code (default: en)

## Components

### Voice Activity Detection (VAD)

- Uses Silero VAD for efficient speech detection
- Configurable thresholds and window sizes
- Optimized for 16kHz audio input
- Provides speech probability scores

### Speaker Diarization

- Pyannote.audio 3.1 for diarization
- ECAPA-TDNN speaker embeddings
- FAISS-based speaker indexing
- Adaptive speaker tracking
- Supports up to 4 concurrent speakers

### ASR Processing

- Faster-Whisper backend
- Streaming-optimized processing
- Configurable batch processing
- Word-level timestamps
- Context management for improved accuracy

### Performance Monitoring

- Real-time latency tracking
- GPU memory and utilization monitoring
- CPU usage statistics
- Model-specific performance metrics

## UI Layout

The terminal UI is divided into several panels:

1. **Live Transcription** (Top)
   - Real-time speech-to-text output
   - Punctuation highlighting

2. **Transcript History** (Middle-Left)
   - Speaker-labeled transcripts
   - Timestamp and duration info
   - Color-coded speaker identification

3. **System Status** (Bottom)
   - Model latency statistics
   - CPU/GPU utilization
   - Memory usage tracking

4. **Voice Activity** (Right)
   - Speech probability visualization
   - Buffer usage indicators
   - Silence detection progress

## Performance Considerations

- **GPU Memory**: Models are optimized for efficient GPU memory usage
- **CPU Usage**: Multi-threaded design with configurable thread allocation
- **Latency**: Monitored and optimized for real-time performance
- **Buffer Management**: Efficient circular buffer implementation

## Troubleshooting

1. **GPU Issues**:
   - Ensure CUDA and cuDNN are properly installed
   - Monitor GPU memory usage in the UI
   - Try different compute_type settings

2. **Audio Input**:
   - Check microphone configuration
   - Verify audio sample rate (should be 16kHz)
   - Monitor VAD visualization for input detection

3. **Performance**:
   - Adjust batch_size for better throughput
   - Monitor latency statistics in the UI
   - Consider using a smaller model for faster processing

4. **Diarization**:
   - Ensure valid Hugging Face token
   - Check speaker detection thresholds
   - Monitor diarization latency metrics

## Logging

- Main log file: `main.log`
- Transcription log: `transcription.log`
- Detailed model performance metrics in logs
- Error tracking and debugging information

## License

This project is licensed under the MIT License.

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests. 