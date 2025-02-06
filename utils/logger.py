"""
Centralized logging configuration for the whisper streaming application.
"""

import logging
import sys
import warnings
import os

def setup_logger(name: str, log_file: str = "main.log", console_output: bool = True) -> logging.Logger:
    """
    Set up a logger with the specified configuration.
    
    Args:
        name: Name of the logger
        log_file: Path to the log file
        console_output: Whether to also log to console
    
    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Remove any existing handlers
    logger.handlers = []
    
    # Create formatters
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Add file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Add console handler if requested
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    return logger

def configure_logging(log_file: str = "main.log", console_output: bool = True):
    """
    Configure global logging settings and module-specific loggers.
    
    Args:
        log_file: Path to the log file
        console_output: Whether to also log to console
    """
    # Basic configuration
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            *([] if not console_output else [logging.StreamHandler(sys.stdout)])
        ]
    )
    
    # Suppress warnings but log them to file
    warnings.filterwarnings('ignore')
    os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
    
    # Configure module-specific logging
    noisy_modules = [
        "faster_whisper",
        "models.diarization",
        "speechbrain",
        "pyannote",
        "pipeline"
    ]
    
    for module in noisy_modules:
        module_logger = logging.getLogger(module)
        module_logger.setLevel(logging.ERROR)  # Set to ERROR to reduce console noise
        module_logger.handlers = []  # Remove existing handlers
        
        # Add file handler for debugging
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        module_logger.addHandler(file_handler)

def get_logger(name: str, log_file: str = "main.log", console_output: bool = True) -> logging.Logger:
    """
    Get or create a logger with the specified configuration.
    
    Args:
        name: Name of the logger
        log_file: Path to the log file
        console_output: Whether to also log to console
    
    Returns:
        Configured logger instance
    """
    return setup_logger(name, log_file, console_output) 