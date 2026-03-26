"""
BiRAGAS Merged Causality System
==================================
Ayass Bioscience LLC | Version 3.0.0

COMPLETE MERGER of:
- Causality_agent (LLM intelligence, intent, literature, eligibility)
- 4-Layer BiRAGAS (23 implemented science modules, orchestrator, universal agent)

This is the PRODUCTION system with all capabilities:
- Natural language query understanding (7 intents)
- Literature search (PubMed, EuropePMC, Semantic Scholar)
- Pre-flight data quality gates (9 eligibility checks)
- Interactive clarification before running
- 23 implemented causal science modules
- Self-correction and learning
- 60+ disease taxonomy with unlimited scenario generation
- 17-scenario stress test validation
"""

__version__ = "3.0.0"

from .agent import SupervisorAgent, FileInspector, EligibilityChecker, WorkflowRouter
from .cli import save_result

__all__ = ["SupervisorAgent", "FileInspector", "EligibilityChecker", "WorkflowRouter", "save_result"]
