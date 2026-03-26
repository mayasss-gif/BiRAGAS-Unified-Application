"""
BiRAGAS CRISPR Multi-Knockout Predicting Engine
==================================================
Ayass Bioscience LLC | Version 1.0.0

Genome-scale CRISPR perturbation prediction system covering
177,000+ knockout possibilities and their N×N combinations.

Architecture:
    MultiKnockoutEngine     — 177K single-gene knockout predictions
    CombinationPredictor    — N×N combination outcome prediction
    ACESuperiorScorer       — Enhanced ACE scoring with 12 evidence streams
    BrunelloIntegrator      — Full Brunello library (77,441 guides, 19,091 genes)
    CausalCRISPROrchestrator — Integrates with all 7 BiRAGAS phases

Mathematical Foundations:
    - Pearl's Do-Calculus with multi-target graph surgery
    - Bayesian Network propagation with epistasis modeling
    - Bliss Independence for combination synergy
    - Highest Single Agent (HSA) model for additivity
    - Loewe Additivity for dose equivalence
    - Random Forest ensemble for effect prediction
    - Graph Neural Network-inspired propagation
"""

__version__ = "1.0.0"

from .multi_knockout_engine import MultiKnockoutEngine, KnockoutResult
from .combination_predictor import CombinationPredictor, CombinationResult
from .ace_superior_scorer import ACESuperiorScorer
from .brunello_integrator import BrunelloIntegrator

__all__ = [
    "MultiKnockoutEngine",
    "KnockoutResult",
    "CombinationPredictor",
    "CombinationResult",
    "ACESuperiorScorer",
    "BrunelloIntegrator",
]
