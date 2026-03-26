# mdp_engine/exceptions.py
from __future__ import annotations


class MDPError(Exception):
    """Base exception for mdp_engine."""


class ValidationError(MDPError):
    """Raised when inputs are invalid or inconsistent."""


class DependencyError(MDPError):
    """Raised when an optional dependency is required but missing."""


class DataError(MDPError):
    """Raised when data is missing, empty, or malformed."""


class NetworkError(MDPError):
    """Raised for network/HTTP failures."""
