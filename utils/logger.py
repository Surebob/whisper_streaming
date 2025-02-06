"""
Centralized logging configuration for the whisper streaming application.
"""

import logging
import sys
import warnings
import os
from colorama import Fore, Style

class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for different log levels."""
    
    def format(self, record):
        # Skip SpeechBrain initialization messages
        if record.name == 'speechbrain.utils.quirks' or record.name == 'speechbrain.utils.autocast':
            return ""
            
        # Special formatting for initialization messages
        if record.name == 'speechbrain.utils.fetching':
            if 'Fetch' in record.msg:
                file = record.msg.split(': ')[0].replace('Fetch ', '')
                return f"{Fore.BLUE}║  Loading {file}...{Style.RESET_ALL}"
            return ""
            
        # Special formatting for parameter transfer
        if record.name == 'speechbrain.utils.parameter_transfer':
            return f"{Fore.BLUE}║  Loading pretrained models...{Style.RESET_ALL}"
            
        # Special formatting for faster whisper
        if record.name == 'faster_whisper':
            return f"{Fore.BLUE}║  {record.msg}{Style.RESET_ALL}"
            
        # Special formatting for diarization
        if record.name == 'models.diarization':
            msg = record.msg
            if 'TF32' in msg:
                return ""
            if 'loaded successfully' in msg:
                return f"{Fore.BLUE}║  Models loaded on {Fore.GREEN}cuda{Style.RESET_ALL}"
            if 'FAISS index' in msg:
                return f"{Fore.BLUE}║  Speaker index initialized{Style.RESET_ALL}"
            if 'Loaded' in msg and 'speaker' in msg:
                count = msg.split()[1]
                return f"{Fore.BLUE}║  Loaded {Fore.GREEN}{count}{Fore.BLUE} speakers from cache{Style.RESET_ALL}"
            return f"{Fore.BLUE}║  {msg}{Style.RESET_ALL}"
            
        # Format other messages
        if record.levelname == 'INFO':
            if 'cache found' in record.msg:
                return f"{Fore.BLUE}║  Using cached model{Style.RESET_ALL}"
            return f"{Fore.CYAN}► {record.message}{Style.RESET_ALL}"
        elif record.levelname == 'WARNING':
            return f"{Fore.YELLOW}► Warning: {record.message}{Style.RESET_ALL}"
        elif record.levelname == 'ERROR':
            return f"{Fore.RED}► Error: {record.message}{Style.RESET_ALL}"
        
        return ""

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
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_formatter = ColoredFormatter()
    
    # Add file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # Add console handler if requested
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
    
    # Prevent propagation to root logger
    logger.propagate = False
    
    return logger

def configure_logging(log_file: str = "main.log", console_output: bool = True):
    """
    Configure global logging settings and module-specific loggers.
    """
    # Suppress all warnings first
    warnings.filterwarnings('ignore')
    os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
    
    # Specifically suppress the custom_fwd warning
    warnings.filterwarnings(
        'ignore',
        category=FutureWarning,
        message='.*torch.cuda.amp.custom_fwd.*'
    )
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.WARNING)
    
    # Create file handler for all logs
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    
    # Configure module-specific logging
    noisy_modules = {
        # Module name: (log level, propagate)
        "faster_whisper": (logging.INFO, False),
        "models.diarization": (logging.INFO, False),
        "speechbrain": (logging.ERROR, False),  # Suppress all SpeechBrain base logs
        "speechbrain.utils.quirks": (logging.ERROR, False),
        "speechbrain.utils.autocast": (logging.ERROR, False),
        "pyannote": (logging.INFO, False),
        "pipeline": (logging.INFO, False),
    }
    
    for module, (level, propagate) in noisy_modules.items():
        module_logger = logging.getLogger(module)
        module_logger.setLevel(level)
        module_logger.handlers = []
        module_logger.addHandler(file_handler)
        module_logger.propagate = propagate

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