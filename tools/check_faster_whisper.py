import os
import hashlib
import logging
import subprocess
import tempfile
import shutil
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_file_hash(file_path):
    """Calculate SHA256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def get_pip_package_path():
    """Get the installed faster-whisper package path."""
    try:
        result = subprocess.run(
            ["pip", "show", "faster-whisper"],
            capture_output=True,
            text=True
        )
        for line in result.stdout.split('\n'):
            if line.startswith('Location:'):
                return os.path.join(line.split(':', 1)[1].strip(), 'faster_whisper')
    except Exception as e:
        logger.error(f"Error finding pip package: {e}")
    return None

def compare_directories():
    """Compare local faster-whisper with pip package."""
    local_dir = os.path.join(os.getcwd(), 'faster-whisper', 'faster_whisper')
    pip_dir = get_pip_package_path()
    
    if not os.path.exists(local_dir):
        logger.error("Local faster-whisper directory not found")
        return
    
    if not pip_dir or not os.path.exists(pip_dir):
        logger.error("Pip package faster-whisper not found. Please install it first.")
        return
    
    logger.info("Comparing faster-whisper directories:")
    logger.info(f"Local dir: {local_dir}")
    logger.info(f"Pip dir: {pip_dir}")
    
    # Track differences
    modified_files = []
    added_files = []
    removed_files = []
    
    # Get all Python files from both directories
    local_files = {f.relative_to(local_dir) for f in Path(local_dir).rglob('*.py')}
    pip_files = {f.relative_to(pip_dir) for f in Path(pip_dir).rglob('*.py')}
    
    # Find added and removed files
    added_files = sorted(local_files - pip_files)
    removed_files = sorted(pip_files - local_files)
    
    # Compare common files
    common_files = local_files & pip_files
    for rel_path in sorted(common_files):
        local_file = os.path.join(local_dir, rel_path)
        pip_file = os.path.join(pip_dir, rel_path)
        
        local_hash = get_file_hash(local_file)
        pip_hash = get_file_hash(pip_file)
        
        if local_hash != pip_hash:
            modified_files.append(rel_path)
    
    # Print results
    logger.info("\nAnalysis Results:")
    
    if not any([modified_files, added_files, removed_files]):
        logger.info("\n✅ Local faster-whisper is identical to pip package!")
        logger.info("You can safely delete the local copy and use the pip package instead.")
        return
    
    if modified_files:
        logger.info("\n🔄 Modified files:")
        for f in modified_files:
            logger.info(f"  {f}")
    
    if added_files:
        logger.info("\n➕ Added files:")
        for f in added_files:
            logger.info(f"  {f}")
    
    if removed_files:
        logger.info("\n➖ Removed files:")
        for f in removed_files:
            logger.info(f"  {f}")
    
    logger.info("\n⚠️ Local faster-whisper has modifications!")
    logger.info("Review the changes before deciding to remove it.")

if __name__ == "__main__":
    compare_directories() 