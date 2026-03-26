"""
BiRAGAS Orchestrator — Agentic AI Ecosystem for Causal Inference
================================================================
Ayass Bioscience LLC | Version 2.0.0

Multi-agent orchestration system for the 7-phase, 23-module BiRAGAS
Causality Inference Framework. Supports autonomous execution,
self-correction, self-learning, and stress test validation.
"""

__version__ = "2.0.0"

from .master_orchestrator import MasterOrchestrator
from .phase_agents import (
    PhaseAgent,
    Phase1Agent,
    Phase2Agent,
    Phase3Agent,
    Phase4Agent,
    Phase5Agent,
    Phase6Agent,
    Phase7Agent,
)
from .stress_test_agent import StressTestAgent
from .self_corrector import SelfCorrector
from .learning_engine import LearningEngine
from .config import OrchestratorConfig, StressTestConfig

__all__ = [
    "MasterOrchestrator",
    "PhaseAgent",
    "Phase1Agent", "Phase2Agent", "Phase3Agent", "Phase4Agent",
    "Phase5Agent", "Phase6Agent", "Phase7Agent",
    "StressTestAgent",
    "SelfCorrector",
    "LearningEngine",
    "OrchestratorConfig",
    "StressTestConfig",
]
