# Whisper Streaming Transcription

## Overview

This repository implements a real-time streaming transcription application using various ASR backends, enhanced with features like voice activity detection (VAD), diarization, and a rich terminal UI. The system leverages Whisper, Faster-Whisper, and other models to provide robust, low-latency transcription with detailed system monitoring.

## Project Structure

- **pipeline_ui.py**: Real-time transcription UI that displays live transcription, transcript history, assistant responses, model status, system stats, and VAD information.
- **pipeline.py**: Streaming transcription pipeline with diarization support; processes audio from the microphone and outputs results to the console.
- **whisper_online.py**: Implements various ASR backends (Faster-Whisper, MLX-Whisper, WhisperTimestamped, OpenAI API) and is used by the pipeline modules.
- **models/**: Contains modules for VAD (`vad.py`), diarization (`diarization.py`), and other audio processing utilities.
- **README.md**: This file.

## Installation

Ensure you have Python 3.7+ installed. Install required dependencies. It is recommended to use a virtual environment.

```bash
pip install -r requirements.txt
```

If a `requirements.txt` file is not provided, make sure to install the following packages (and their dependencies):

- torch
- rich
- psutil
- numpy
- sounddevice
- librosa
- faster-whisper (or whisper model libraries)
- pynvml
- pyannote.audio
- speechbrain
- faiss
- and other dependencies as required by the code.

## Usage

### Running the UI (pipeline_ui.py)

The `pipeline_ui.py` script launches an interactive user interface that displays live transcription along with transcript history, assistant responses, active model status, system stats, and VAD information.

#### Command-Line Arguments (pipeline_ui.py):

- `--use-mic`: Use microphone input.
- `--model`: Choose the ASR model. Available options: `tiny.en`, `tiny`, `base.en`, `base`, `small.en`, `small`, `medium.en`, `medium`, `large-v1`, `large-v2`, `large-v3`, `large`, `large-v3-turbo`, `distil-large-v3`. (Default: `large-v3-turbo`)
- `--compute-type`: Set the model's computation type. Options: `int8_float16`, `float16`, `float32`, `int8`. (Default: `int8_float16`)
- `--min-chunk-size`: Minimum audio chunk size in seconds. (Default: `1.0`)
- `--batch-size`: Batch size for model inference. (Default: `8`)

#### Examples (pipeline_ui.py):

- **Basic Usage with Microphone**

  ```bash
  python pipeline_ui.py --use-mic
  ```

- **Using a Different Model and Compute Type**

  ```bash
  python pipeline_ui.py --use-mic --model base --compute-type float16
  ```

- **Custom Chunk Size and Batch Size**

  ```bash
  python pipeline_ui.py --use-mic --min-chunk-size 0.5 --batch-size 4
  ```

- **Full Command Example**

  ```bash
  python pipeline_ui.py --use-mic --model small.en --compute-type int8_float16 --min-chunk-size 1.0 --batch-size 8
  ```

### Running the Pipeline (pipeline.py)

The `pipeline.py` script runs the complete streaming transcription pipeline with diarization capabilities. It captures audio from the microphone and outputs transcription results to the console.

#### Command-Line Arguments (pipeline.py):

- `--use-mic`: Use microphone input.
- `--model`: Choose the ASR model. (Same options as above; Default: `large-v3-turbo`)
- `--language`: Language code (e.g., `en`, `de`). (Default: `en`)
- `--compute-type`: Set the model's computation type. Options: `int8_float16`, `float16`, `float32`. (Default: `int8_float16`)
- `--min-chunk-size`: Minimum audio chunk size in seconds. (Default: `1.0`)
- `--batch-size`: Batch size for model inference. (Default: `8`)

#### Examples (pipeline.py):

- **Running with Microphone Input**

  ```bash
  python pipeline.py --use-mic
  ```

- **Specifying a Model and Language**

  ```bash
  python pipeline.py --use-mic --model medium --language en
  ```

- **Custom Compute Type**

  ```bash
  python pipeline.py --use-mic --compute-type float32
  ```

- **Full Command Example**

  ```bash
  python pipeline.py --use-mic --model large-v3-turbo --language en --compute-type int8_float16 --min-chunk-size 1.0 --batch-size 8
  ```

## Additional Notes

- **Rich UI**: The terminal UI is built using the Rich library, offering visually appealing progress bars and layout management.
- **Real-Time Processing**: The system uses multi-threading for real-time VAD, transcription, and diarization. Monitor logs for performance details.
- **Logging**: Detailed logs are written to `transcription.log` (for UI) and `main.log` (for the pipeline). Use these for troubleshooting.
- **Resource Monitoring**: System and model performance metrics (like CPU usage, GPU load, and latency statistics) are displayed in the UI.
- **UI-specific Configurations**: Latency thresholds and duration color-coding remain in `pipeline_ui.py` to maintain focus on UI concerns.

## Troubleshooting

- **NVML/GPU Issues**: Ensure your GPU drivers and NVML library are up-to-date.
- **Audio Input Issues**: Verify that your microphone is properly configured and working.
- **Performance**: Adjust the `--min-chunk-size` and `--compute-type` if you experience high latency or CPU load.
- **Logs**: Refer to `transcription.log` and `main.log` for detailed error messages and performance stats.

## Contributing

Contributions, bug reports, and feature requests are welcome. Please submit a pull request or open an issue on GitHub.

## License

This project is licensed under the MIT License. 