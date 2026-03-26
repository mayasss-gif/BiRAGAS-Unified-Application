"""
Base classes for the Cohort Retrieval Agent system.

This module provides abstract base classes that define common interfaces
and functionality for all agents and tools in the system.
"""

from .base_agent import BaseRetrievalAgent
from .base_tool import BaseTool

__all__ = ["BaseRetrievalAgent", "BaseTool"] 