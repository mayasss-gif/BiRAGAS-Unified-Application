"""
DEG Pipeline Agent Package

A robust, self-healing agent system for Differential Expression Gene analysis.
"""

from .agent import DEGPipelineAgent
from .exceptions import DEGPipelineError, DataLoadError, MetadataError, AnalysisError
from .config import DEGPipelineConfig

__version__ = "1.0.0"
__all__ = [
    "DEGPipelineAgent",
    "DEGPipelineError", 
    "DataLoadError", 
    "MetadataError", 
    "AnalysisError",
    "DEGPipelineConfig"
]