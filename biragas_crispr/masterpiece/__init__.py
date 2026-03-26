"""
BiRAGAS CRISPR MASTERPIECE
============================
Ayass Bioscience LLC | Version 1.0.0

The ULTIMATE unified CRISPR Editor + Analyzer system integrating:

EDITING CAPABILITIES:
    - Guide RNA design (GenET/rs3 scoring)
    - On-target activity prediction
    - Off-target search and scoring
    - Editing outcome prediction (indels, HDR, base editing)
    - BiRAGAS EditingEngine amplicon analysis integration

ANALYSIS CAPABILITIES:
    - MAGeCK RRA/MLE screening analysis
    - BAGEL2 essentiality classification
    - Perturb-seq single-cell analysis (pertpy)
    - Drug sensitivity screening (DrugZ)

PREDICTION CAPABILITIES:
    - 177,000 knockout configurations (Brunello library)
    - 31.3 BILLION pairwise combination predictions
    - 7-method ensemble knockout prediction
    - 6-model synergy combination prediction
    - 15-stream Superior ACE scoring
    - Sparse matrix O(1) per-knockout prediction

CAUSALITY INTEGRATION:
    - Full BiRAGAS 7-phase pipeline integration (28 phase modules)
    - Causal DAG construction from 9 multi-modal data streams
    - MR validation, hallucination detection, confounding check
    - Drug target ranking (9D), safety profiling, combination therapy
    - Patient stratification, clinical report generation

Architecture:
    MasterpieceOrchestrator (master agent)
    ├── EditingEngine — guide design + outcome prediction
    ├── ScreeningEngine — MAGeCK + BAGEL2 + DrugZ analysis
    ├── PerturbSeqEngine — single-cell CRISPR analysis
    ├── AmpliconAnalyzer — BiRAGAS EditingEngine editing verification
    ├── KnockoutPredictor — 177K × 31.3B prediction engine
    ├── CausalityIntegrator — BiRAGAS 7-phase bridge
    └── ReportGenerator — unified clinical report
"""

__version__ = "1.0.0"

from .masterpiece_orchestrator import MasterpieceOrchestrator

__all__ = ["MasterpieceOrchestrator"]
