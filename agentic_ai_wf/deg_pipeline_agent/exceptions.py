"""
Custom exceptions for the DEG Pipeline Agent.
"""

from typing import Optional, Dict, Any


class DEGPipelineError(Exception):
    """Base exception for DEG pipeline errors."""
    
    def __init__(self, message: str, error_code: Optional[str] = None, context: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.error_code = error_code
        self.context = context or {}
        self.message = message
    
    def __str__(self) -> str:
        if self.error_code:
            return f"[{self.error_code}] {self.message}"
        return self.message


class DataLoadError(DEGPipelineError):
    """Raised when data loading fails."""
    pass


class MetadataError(DEGPipelineError):
    """Raised when metadata extraction or processing fails."""
    pass


class AnalysisError(DEGPipelineError):
    """Raised when DESeq2 analysis fails."""
    pass


class ValidationError(DEGPipelineError):
    """Raised when data validation fails."""
    pass


class FileSystemError(DEGPipelineError):
    """Raised when file system operations fail."""
    pass


class GeneMapperError(DEGPipelineError):
    """Raised when gene ID mapping fails."""
    pass


class RecoverableError(DEGPipelineError):
    """Raised for errors that can be automatically fixed."""
    
    def __init__(self, message: str, fix_suggestion: str, **kwargs):
        super().__init__(message, **kwargs)
        self.fix_suggestion = fix_suggestion


class NonRecoverableError(DEGPipelineError):
    """Raised for errors that require manual intervention."""
    pass 