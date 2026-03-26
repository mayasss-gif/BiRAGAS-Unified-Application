"""
Custom exceptions for the Cohort Retrieval Agent system.

This module defines a hierarchy of exceptions that provide specific error handling
for different failure modes in the cohort retrieval pipeline.
"""

from typing import Optional, Any, Dict


class CohortRetrievalError(Exception):
    """Base exception for all cohort retrieval errors."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}
    
    def __str__(self):
        if self.details:
            return f"{self.message}. Details: {self.details}"
        return self.message


class QueryError(CohortRetrievalError):
    """Raised when dataset querying fails."""
    
    def __init__(self, message: str, source: str = "unknown", query: str = "", details: Optional[Dict[str, Any]] = None):
        super().__init__(message, details)
        self.source = source
        self.query = query


class FilterError(CohortRetrievalError):
    """Raised when dataset filtering fails."""
    
    def __init__(self, message: str, dataset_id: str = "", filter_type: str = "", details: Optional[Dict[str, Any]] = None):
        super().__init__(message, details)
        self.dataset_id = dataset_id
        self.filter_type = filter_type


class DownloadError(CohortRetrievalError):
    """Raised when file download fails."""
    
    def __init__(self, message: str, url: str = "", file_path: str = "", details: Optional[Dict[str, Any]] = None):
        super().__init__(message, details)
        self.url = url
        self.file_path = file_path


class MetadataError(CohortRetrievalError):
    """Raised when metadata extraction fails."""
    
    def __init__(self, message: str, dataset_id: str = "", sample_id: str = "", details: Optional[Dict[str, Any]] = None):
        super().__init__(message, details)
        self.dataset_id = dataset_id
        self.sample_id = sample_id


class ValidationError(CohortRetrievalError):
    """Raised when data validation fails."""
    
    def __init__(self, message: str, file_path: str = None, validation_type: str = None):
        super().__init__(message)
        self.file_path = file_path
        self.validation_type = validation_type


class ClassificationError(CohortRetrievalError):
    """Raised when sample classification fails."""
    
    def __init__(self, message: str, sample_id: str = None, classification_type: str = None):
        super().__init__(message)
        self.sample_id = sample_id
        self.classification_type = classification_type


class AgentError(CohortRetrievalError):
    """Raised when an agent fails to execute."""
    
    def __init__(self, message: str, agent_name: str = "", operation: str = "", details: Optional[Dict[str, Any]] = None):
        super().__init__(message, details)
        self.agent_name = agent_name
        self.operation = operation


class ConfigurationError(CohortRetrievalError):
    """Raised when configuration is invalid."""
    
    def __init__(self, message: str, config_key: str = "", details: Optional[Dict[str, Any]] = None):
        super().__init__(message, details)
        self.config_key = config_key


class NetworkError(CohortRetrievalError):
    """Raised when network operations fail."""
    
    def __init__(self, message: str, endpoint: str = "", status_code: Optional[int] = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, details)
        self.endpoint = endpoint
        self.status_code = status_code


class TimeoutError(CohortRetrievalError):
    """Raised when operations timeout."""
    
    def __init__(self, message: str, operation: str = "", timeout_seconds: Optional[int] = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, details)
        self.operation = operation
        self.timeout_seconds = timeout_seconds 