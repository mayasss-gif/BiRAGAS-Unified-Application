"""
Tools for the Cohort Retrieval Agent system.
"""

from .query_tool import QueryTool
from .filter_tool import FilterTool, FilterCriteria
from .download_tool import DownloadTool, DownloadInfo
from .metadata_tool import MetadataTool, SampleMetadata
from .validation_tool import ValidationTool, ValidationResult
from .classification_tool import GPTClassificationTool, SampleInfo, ClassificationResult
from .ftp_download_tool import FTPDownloadTool, FTPFileInfo, FTPDownloadResult
from .supplementary_validation_tool import (
    SupplementaryValidationTool, 
    ValidationSeverity, 
    ValidationIssue, 
    FileValidationResult, 
    DatasetValidationResult
)
from .evaluator import LLMEvaluator, EvaluationTool, EvaluationResult  # Import EvaluationTool and EvaluationResult

__all__ = [
    "QueryTool",
    "FilterTool",
    "FilterCriteria", 
    "DownloadTool",
    "DownloadInfo",
    "MetadataTool",
    "SampleMetadata",
    "ValidationTool",
    "ValidationResult",
    "GPTClassificationTool",
    "SampleInfo",
    "ClassificationResult",
    "FTPDownloadTool",
    "FTPFileInfo", 
    "FTPDownloadResult",
    "SupplementaryValidationTool",
    "ValidationSeverity",
    "ValidationIssue",
    "FileValidationResult",
    "DatasetValidationResult",
    "LLMEvaluator",  # Make sure LLMEvaluator is still included
    "EvaluationTool",  # Add EvaluationTool
    "EvaluationResult"  # Add EvaluationResult
]
