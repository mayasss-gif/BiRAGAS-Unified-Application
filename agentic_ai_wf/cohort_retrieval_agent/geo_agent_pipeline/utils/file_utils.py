"""
File utilities for the Cohort Retrieval Agent system.
"""

import os
import gzip
import shutil
from pathlib import Path
from typing import Optional, Union


def ensure_directory(path: Union[str, Path]) -> Path:
    """
    Ensure directory exists, create if it doesn't.
    
    Args:
        path: Directory path
        
    Returns:
        Path object
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_file_size(path: Union[str, Path], unit: str = "bytes") -> float:
    """
    Get file size in specified unit.
    
    Args:
        path: File path
        unit: Unit (bytes, kb, mb, gb)
        
    Returns:
        File size in specified unit
    """
    path = Path(path)
    if not path.exists():
        return 0.0
    
    size_bytes = path.stat().st_size
    
    if unit.lower() == "bytes":
        return float(size_bytes)
    elif unit.lower() == "kb":
        return size_bytes / 1024
    elif unit.lower() == "mb":
        return size_bytes / (1024 * 1024)
    elif unit.lower() == "gb":
        return size_bytes / (1024 * 1024 * 1024)
    else:
        raise ValueError(f"Unsupported unit: {unit}")


def compress_file(source_path: Union[str, Path], 
                 dest_path: Optional[Union[str, Path]] = None,
                 compression: str = "gzip") -> Path:
    """
    Compress a file.
    
    Args:
        source_path: Source file path
        dest_path: Destination path (optional)
        compression: Compression type (gzip, zip)
        
    Returns:
        Path to compressed file
    """
    source_path = Path(source_path)
    
    if dest_path is None:
        if compression == "gzip":
            dest_path = source_path.with_suffix(source_path.suffix + ".gz")
        else:
            dest_path = source_path.with_suffix(source_path.suffix + f".{compression}")
    else:
        dest_path = Path(dest_path)
    
    if compression == "gzip":
        with open(source_path, 'rb') as f_in:
            with gzip.open(dest_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
    else:
        raise ValueError(f"Unsupported compression: {compression}")
    
    return dest_path


def safe_remove(path: Union[str, Path]) -> bool:
    """
    Safely remove a file or directory.
    
    Args:
        path: Path to remove
        
    Returns:
        True if removed, False if not found or error
    """
    try:
        path = Path(path)
        if path.is_file():
            path.unlink()
        elif path.is_dir():
            shutil.rmtree(path)
        return True
    except Exception:
        return False 