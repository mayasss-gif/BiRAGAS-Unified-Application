"""
BiRAGAS Universal Agent System
================================
Ayass Bioscience LLC | Version 2.0.0

Autonomous disease intelligence engine that can analyze ANY medical disease.
Fetches data from public biomedical databases, generates negative controls,
creates unlimited stress test scenarios, and runs them through the
7-phase, 23-module BiRAGAS Causality Framework.

Components:
    DiseaseKnowledgeAgent  — Resolves diseases, fetches molecular data from APIs
    DataAcquisitionAgent   — Converts API data into BiRAGAS input file formats
    ScenarioEngine         — Generates unlimited differential diagnosis scenarios
    UniversalRunner        — Orchestrates everything end-to-end
"""

__version__ = "2.0.0"

from .disease_knowledge_agent import (
    DiseaseKnowledgeAgent,
    DiseaseResolver,
    NegativeControlGenerator,
    APICache,
    RobustHTTPClient,
)
from .data_acquisition_agent import DataAcquisitionAgent
from .scenario_engine import ScenarioEngine
from .universal_runner import UniversalRunner

__all__ = [
    "DiseaseKnowledgeAgent",
    "DiseaseResolver",
    "NegativeControlGenerator",
    "DataAcquisitionAgent",
    "ScenarioEngine",
    "UniversalRunner",
]
