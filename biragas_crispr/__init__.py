"""
BiRAGAS CRISPR Complete — Unified DNA + RNA CRISPR Analysis Platform v3.0
==========================================================================
Ayass Bioscience LLC — Proprietary

The first platform to unify DNA editing (Cas9/Cas12a) and RNA targeting
(Cas13/dCas13/CRISPRi/CRISPRa) with causal inference for drug discovery.

DNA Engines:
    EditingEngine       — SpCas9/SaCas9/Cas12a guide design + amplicon analysis
    ScreeningEngine     — MAGeCK RRA/MLE, BAGEL2, DrugZ
    KnockoutEngine      — 7-method ensemble, 210,859 configurations
    MegaScaleEngine     — Sparse matrix O(1), 22.2B combinations
    CombinationEngine   — 6-model synergy + true 3-way epistasis
    ACEScoringEngine    — 15-stream Superior ACE

RNA Engines:
    RNAGuideEngine      — Cas13a/b/d crRNA design + PFS scoring
    RNAKnockdownEngine  — RNA degradation + collateral modeling
    RNABaseEditEngine    — dCas13 A-to-I (ADAR2) + C-to-U (APOBEC) editing
    TranscriptomeEngine — CRISPRi/CRISPRa/Perturb-seq/CROP-seq analysis
    NonCodingEngine     — lncRNA/miRNA/siRNA targeting + ncRNA networks
    SpatialRNAEngine    — CRISPR-TO spatial transcriptomics

Unified:
    UnifiedOrchestrator — DNA + RNA combined pipeline
    BiRAGASBridge       — 7-phase causality integration
    SelfCorrector       — Autonomous DAG validation + repair
    PipelineDebugger    — Error diagnosis + retry + fallback
"""

__version__ = "3.0.0"
__author__ = "Ayass Bioscience LLC"

from .pipeline.unified_orchestrator import UnifiedOrchestrator

__all__ = ["UnifiedOrchestrator"]
