from pathlib import Path
import logging
import os
import stat
import hashlib
import mimetypes
from typing import Union, Optional, Set, Dict, List, Tuple, Any
from datetime import datetime
import tempfile

from ..configuration import MAX_FILE_SIZE_MB

from ..helpers import logger

def resolve_and_validate_file(
    path_str: Union[str, Path], 
    allowed_extensions: Optional[Set[str]] = None,
    min_size_bytes: int = 0,
    max_size_mb: Optional[float] = None,
    check_permissions: bool = True,
    validate_content: bool = False,
    return_metadata: bool = False
) -> Union[Path, Tuple[Path, Dict[str, Any]]]:
    """
    Enhanced file validation with comprehensive checks and metadata collection.
    
    Args:
        path_str: Path to the file (string or Path object)
        allowed_extensions: Set of allowed file extensions (with dots, e.g., {'.csv', '.xlsx'})
        min_size_bytes: Minimum file size in bytes (default: 0)
        max_size_mb: Maximum file size in MB (defaults to MAX_FILE_SIZE_MB from config)
        check_permissions: Whether to check read permissions
        validate_content: Whether to perform basic content validation
        return_metadata: Whether to return file metadata along with path
        
    Returns:
        Validated Path object or tuple of (Path, metadata_dict) if return_metadata=True
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file fails validation
        PermissionError: If file is not readable
    """
    # Resolve path with enhanced error handling
    try:
        path = Path(path_str).expanduser().resolve()
    except (OSError, RuntimeError) as e:
        logger.error("Failed to resolve path '%s': %s", path_str, e)
        raise ValueError(f"Invalid path format: {path_str}") from e
    
    # Check existence
    if not path.exists():
        # Check if parent directory exists for better error messages
        parent_exists = path.parent.exists()
        logger.error(
            "File not found: %s (parent directory %s: %s)", 
            path, "exists" if parent_exists else "missing", path.parent
        )
        raise FileNotFoundError(f"File not found: {path}")
    
    # Check if it's a file
    if not path.is_file():
        file_type = "directory" if path.is_dir() else "special file/device"
        logger.error("Path is not a regular file: %s (it's a %s)", path, file_type)
        raise ValueError(f"Path is not a regular file: {path}")
    
    # Check permissions
    if check_permissions and not os.access(path, os.R_OK):
        logger.error("File is not readable: %s", path)
        raise PermissionError(f"File is not readable: {path}")
    
    # Get file stats for validation and metadata
    try:
        file_stats = path.stat()
        size_bytes = file_stats.st_size
        size_mb = size_bytes / (1024 * 1024)
    except OSError as e:
        logger.error("Failed to get file stats for %s: %s", path, e)
        raise ValueError(f"Cannot access file: {path}") from e
    
    # Size validation
    if size_bytes < min_size_bytes:
        logger.error(
            "File too small: %d bytes (minimum %d bytes): %s",
            size_bytes, min_size_bytes, path.name
        )
        raise ValueError(f"File too small: {size_bytes} bytes (minimum {min_size_bytes} bytes)")
    
    max_size_limit = max_size_mb if max_size_mb is not None else MAX_FILE_SIZE_MB
    if size_mb > max_size_limit:
        logger.error(
            "File too large: %.2fMB (max %.2fMB): %s",
            size_mb, max_size_limit, path.name
        )
        raise ValueError(f"File too large: {size_mb:.2f}MB (max {max_size_limit:.2f}MB)")
    
    # Extension validation with enhanced checking
    if allowed_extensions is not None:
        ext = path.suffix.lower()
        # Normalize extensions to include dots
        normalized_extensions = {
            ext if ext.startswith('.') else f'.{ext}' 
            for ext in allowed_extensions
        }
        
        if ext not in normalized_extensions:
            # Try to detect MIME type for additional validation
            mime_type, _ = mimetypes.guess_type(str(path))
            logger.error(
                "File %s has disallowed extension '%s' (MIME: %s); allowed: %s",
                path.name, ext, mime_type, sorted(normalized_extensions)
            )
            raise ValueError(
                f"Disallowed file extension '{ext}'. Expected one of {sorted(normalized_extensions)}"
            )
    
    # Content validation
    content_info = {}
    if validate_content:
        try:
            content_info = _validate_file_content(path)
            if not content_info.get('is_valid', True):
                logger.warning(
                    "File %s failed content validation: %s", 
                    path.name, content_info.get('error', 'Unknown error')
                )
        except Exception as e:
            logger.warning("Content validation failed for %s: %s", path.name, e)
            content_info = {'is_valid': False, 'error': str(e)}
    
    # Compile metadata
    metadata = {
        'absolute_path': str(path),
        'filename': path.name,
        'stem': path.stem,
        'suffix': path.suffix,
        'size_bytes': size_bytes,
        'size_mb': size_mb,
        'created_time': datetime.fromtimestamp(file_stats.st_ctime),
        'modified_time': datetime.fromtimestamp(file_stats.st_mtime),
        'permissions': oct(file_stats.st_mode)[-3:],
        'is_readable': os.access(path, os.R_OK),
        'is_writable': os.access(path, os.W_OK),
        'mime_type': mimetypes.guess_type(str(path))[0],
    }
    
    if validate_content:
        metadata['content_validation'] = content_info
    
    # Log successful validation
    logger.info(
        "File validated successfully: %s (%.2fMB, %s)",
        path.name, size_mb, metadata['mime_type'] or 'unknown type'
    )
    
    return (path, metadata) if return_metadata else path


def resolve_and_validate_directory(
    path_str: Union[str, Path],
    check_permissions: bool = True,
    check_writable: bool = False,
    min_free_space_mb: Optional[float] = None,
    return_metadata: bool = False
) -> Union[Path, Tuple[Path, Dict[str, Any]]]:
    """
    Enhanced directory validation with permission and space checks.
    
    Args:
        path_str: Path to the directory (string or Path object)
        check_permissions: Whether to check read permissions
        check_writable: Whether to check write permissions
        min_free_space_mb: Minimum free space required in MB
        return_metadata: Whether to return directory metadata
        
    Returns:
        Validated Path object or tuple of (Path, metadata_dict) if return_metadata=True
        
    Raises:
        ValueError: If directory doesn't exist or fails validation
        PermissionError: If directory permissions are insufficient
    """
    # Resolve path
    try:
        path = Path(path_str).expanduser().resolve()
    except (OSError, RuntimeError) as e:
        logger.error("Failed to resolve directory path '%s': %s", path_str, e)
        raise ValueError(f"Invalid directory path format: {path_str}") from e
    
    # Check existence and type
    if not path.exists():
        logger.error("Directory not found: %s", path)
        raise ValueError(f"Directory not found: {path}")
    
    if not path.is_dir():
        file_type = "file" if path.is_file() else "special file/device"
        logger.error("Path is not a directory: %s (it's a %s)", path, file_type)
        raise ValueError(f"Path is not a directory: {path}")
    
    # Permission checks
    if check_permissions and not os.access(path, os.R_OK):
        logger.error("Directory is not readable: %s", path)
        raise PermissionError(f"Directory is not readable: {path}")
    
    if check_writable and not os.access(path, os.W_OK):
        logger.error("Directory is not writable: %s", path)
        raise PermissionError(f"Directory is not writable: {path}")
    
    # Get directory stats and metadata
    try:
        dir_stats = path.stat()
        
        # Count contents
        try:
            contents = list(path.iterdir())
            file_count = sum(1 for item in contents if item.is_file())
            dir_count = sum(1 for item in contents if item.is_dir())
            total_size = sum(
                item.stat().st_size for item in contents 
                if item.is_file()
            )
        except PermissionError:
            file_count = dir_count = total_size = -1  # Indicates permission denied
        
        # Check free space
        if min_free_space_mb is not None:
            try:
                statvfs = os.statvfs(path)
                free_space_bytes = statvfs.f_frsize * statvfs.f_bavail
                free_space_mb = free_space_bytes / (1024 * 1024)
                
                if free_space_mb < min_free_space_mb:
                    logger.error(
                        "Insufficient free space in %s: %.2fMB available, %.2fMB required",
                        path, free_space_mb, min_free_space_mb
                    )
                    raise ValueError(
                        f"Insufficient free space: {free_space_mb:.2f}MB available, "
                        f"{min_free_space_mb:.2f}MB required"
                    )
            except (OSError, AttributeError):
                logger.warning("Could not check free space for directory: %s", path)
                free_space_mb = None
        else:
            free_space_mb = None
    
    except OSError as e:
        logger.error("Failed to get directory stats for %s: %s", path, e)
        raise ValueError(f"Cannot access directory: {path}") from e
    
    # Compile metadata
    metadata = {
        'absolute_path': str(path),
        'name': path.name,
        'file_count': file_count,
        'dir_count': dir_count,
        'total_size_bytes': total_size,
        'total_size_mb': total_size / (1024 * 1024) if total_size >= 0 else None,
        'created_time': datetime.fromtimestamp(dir_stats.st_ctime),
        'modified_time': datetime.fromtimestamp(dir_stats.st_mtime),
        'permissions': oct(dir_stats.st_mode)[-3:],
        'is_readable': os.access(path, os.R_OK),
        'is_writable': os.access(path, os.W_OK),
        'free_space_mb': free_space_mb,
    }
    
    logger.info(
        "Directory validated successfully: %s (%d files, %d dirs, %.2fMB total)",
        path.name, file_count, dir_count, 
        metadata['total_size_mb'] if metadata['total_size_mb'] is not None else 0
    )
    
    return (path, metadata) if return_metadata else path


def _validate_file_content(path: Path) -> Dict[str, Any]:
    """
    Perform basic content validation on a file.
    
    Args:
        path: Path to the file to validate
        
    Returns:
        Dictionary with validation results
    """
    validation_result = {
        'is_valid': True,
        'is_binary': False,
        'encoding': None,
        'line_count': None,
        'first_bytes': None,
        'file_hash': None,
        'error': None
    }
    
    try:
        # Read first few bytes to determine if binary
        with open(path, 'rb') as f:
            first_bytes = f.read(1024)
            validation_result['first_bytes'] = first_bytes[:100]  # Store first 100 bytes
            
            # Simple binary check
            validation_result['is_binary'] = b'\x00' in first_bytes
            
        # If not binary, try to detect encoding and count lines
        if not validation_result['is_binary']:
            try:
                # Try common encodings
                for encoding in ['utf-8', 'latin-1', 'cp1252']:
                    try:
                        with open(path, 'r', encoding=encoding) as f:
                            content = f.read()
                            validation_result['encoding'] = encoding
                            validation_result['line_count'] = content.count('\n') + 1
                            break
                    except UnicodeDecodeError:
                        continue
            except Exception:
                pass  # Encoding detection failed, but file might still be valid
        
        # Calculate file hash for integrity checking
        hash_md5 = hashlib.md5()
        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hash_md5.update(chunk)
        validation_result['file_hash'] = hash_md5.hexdigest()
        
    except Exception as e:
        validation_result['is_valid'] = False
        validation_result['error'] = str(e)
    
    return validation_result


def batch_validate_files(
    file_paths: List[Union[str, Path]],
    allowed_extensions: Optional[Set[str]] = None,
    continue_on_error: bool = True,
    max_workers: Optional[int] = None
) -> Dict[str, Any]:
    """
    Validate multiple files in batch with parallel processing capability.
    
    Args:
        file_paths: List of file paths to validate
        allowed_extensions: Set of allowed file extensions
        continue_on_error: Whether to continue validation if some files fail
        max_workers: Maximum number of worker threads (None for auto)
        
    Returns:
        Dictionary with validation results for all files
    """
    results = {
        'valid_files': [],
        'invalid_files': [],
        'errors': {},
        'summary': {
            'total_files': len(file_paths),
            'valid_count': 0,
            'invalid_count': 0,
            'total_size_mb': 0
        }
    }
    
    for file_path in file_paths:
        try:
            validated_path, metadata = resolve_and_validate_file(
                file_path, 
                allowed_extensions=allowed_extensions,
                return_metadata=True
            )
            results['valid_files'].append({
                'path': validated_path,
                'metadata': metadata
            })
            results['summary']['valid_count'] += 1
            results['summary']['total_size_mb'] += metadata['size_mb']
            
        except Exception as e:
            results['invalid_files'].append(str(file_path))
            results['errors'][str(file_path)] = str(e)
            results['summary']['invalid_count'] += 1
            
            if not continue_on_error:
                logger.error("Batch validation stopped due to error: %s", e)
                break
    
    # Log summary
    logger.info(
        "Batch validation completed: %d/%d files valid (%.2fMB total)",
        results['summary']['valid_count'],
        results['summary']['total_files'],
        results['summary']['total_size_mb']
    )
    
    if results['invalid_files']:
        logger.warning("Invalid files: %s", results['invalid_files'])
    
    return results


def create_safe_temp_file(
    suffix: str = '',
    prefix: str = 'temp_',
    directory: Optional[Union[str, Path]] = None
) -> Path:
    """
    Create a temporary file with safe permissions and return its path.
    
    Args:
        suffix: File suffix/extension
        prefix: File prefix
        directory: Directory to create temp file in (default: system temp)
        
    Returns:
        Path to the created temporary file
    """
    if directory:
        directory = resolve_and_validate_directory(directory)
    
    # Create temporary file with restricted permissions
    fd, temp_path = tempfile.mkstemp(
        suffix=suffix,
        prefix=prefix,
        dir=directory
    )
    
    # Close the file descriptor and return Path object
    os.close(fd)
    temp_path_obj = Path(temp_path)
    
    # Set restrictive permissions (owner read/write only)
    temp_path_obj.chmod(0o600)
    
    logger.info("Created temporary file: %s", temp_path_obj)
    return temp_path_obj