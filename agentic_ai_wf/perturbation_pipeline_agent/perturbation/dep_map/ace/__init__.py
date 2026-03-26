"""
ACE (Average Causal Effect) Causality Analysis Module

This module provides ACE computation, therapeutic alignment analysis,
and causality visualization for DepMap CRISPR screening data.
"""

from .causality_analysis import compute_ace_analysis
from .causality_figures import generate_causality_figures
from .causality_graph import generate_ace_graph
from .config import ACEConfig

__all__ = [
    "compute_ace_analysis",
    "generate_causality_figures",
    "generate_ace_graph",
    "ACEConfig",
]
