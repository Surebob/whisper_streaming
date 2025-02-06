"""
Centralized command-line argument parsing for the whisper streaming application.
"""

import argparse
from typing import Optional

def create_base_parser() -> argparse.ArgumentParser:
    """
    Create the base argument parser with common arguments.
    
    Returns:
        Base argument parser with common arguments
    """
    parser = argparse.ArgumentParser(add_help=False)
    
    # Input configuration
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument(
        "--use-mic",
        action="store_true",
        help="Use microphone input"
    )
    input_group.add_argument(
        "--use-system-audio",
        action="store_true",
        help="Capture system audio output"
    )
    
    # Model configuration
    parser.add_argument(
        "--model",
        type=str,
        default="large-v3-turbo",
        choices=[
            'tiny.en', 'tiny',
            'base.en', 'base',
            'small.en', 'small',
            'medium.en', 'medium',
            'large-v1', 'large-v2', 'large-v3', 'large',
            'large-v3-turbo', 'distil-large-v3'
        ],
        help="Name of the Whisper model to use"
    )
    
    parser.add_argument(
        "--compute-type",
        type=str,
        default="int8_float16",
        choices=['int8_float16', 'float16', 'float32', 'int8'],
        help="Model computation type"
    )
    
    parser.add_argument(
        "--min-chunk-size",
        type=float,
        default=1.0,
        help="Minimum chunk size in seconds"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for model inference"
    )

    # Add benchmark flag
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Enable performance benchmarking"
    )
    
    return parser

def create_pipeline_parser() -> argparse.ArgumentParser:
    """
    Create argument parser for the main pipeline script.
    
    Returns:
        Argument parser for pipeline.py
    """
    parent_parser = create_base_parser()
    parser = argparse.ArgumentParser(
        description="Streaming transcription with diarization",
        parents=[parent_parser],
        conflict_handler='resolve'
    )
    
    # Add pipeline-specific arguments
    parser.add_argument(
        "--language",
        type=str,
        default="en",
        help="Language code"
    )
    
    return parser

def create_ui_parser() -> argparse.ArgumentParser:
    """
    Create argument parser for the UI script.
    
    Returns:
        Argument parser for pipeline_ui.py
    """
    parent_parser = create_base_parser()
    parser = argparse.ArgumentParser(
        description="Whisper Streaming Transcription UI",
        parents=[parent_parser],
        conflict_handler='resolve'
    )
    
    return parser

def parse_args(parser_type: str = "ui", parser: Optional[argparse.ArgumentParser] = None) -> argparse.Namespace:
    """
    Parse command-line arguments based on the specified parser type.
    
    Args:
        parser_type: Type of parser to use ("ui" or "pipeline")
        parser: Optional existing ArgumentParser to use
    
    Returns:
        Parsed command-line arguments
    """
    if parser_type == "ui":
        if parser is None:
            parser = create_ui_parser()
    elif parser_type == "pipeline":
        if parser is None:
            parser = create_pipeline_parser()
    else:
        raise ValueError(f"Unknown parser type: {parser_type}")
    
    return parser.parse_args() 