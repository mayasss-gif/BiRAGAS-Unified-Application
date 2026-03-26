from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any

from agentic_ai_wf.cohort_retrieval_agent.geo_agent_pipeline.config import (
    CohortRetrievalConfig,
    DirectoryPathsConfig,
    AgentName
)


@dataclass
class GlobalAgentConfig:
    """
    Global orchestrator-level configuration for all modules (Cohort, DEG, Pathway, Reports, etc.).
    Ensures every step has consistent access to directories, retry configs, and network settings.
    """

    cohort_config: CohortRetrievalConfig = field(default_factory=CohortRetrievalConfig)

    def __post_init__(self):
        self.paths: DirectoryPathsConfig = self.cohort_config.directory_paths
        self._ensure_base_structure()

        # Flatten common defaults
        self.defaults: Dict[str, Any] = {
            "max_retries": self.cohort_config.retry_config.max_retries,
            "timeout": self.cohort_config.network_config.timeout,
            "user_agent": self.cohort_config.network_config.user_agent,
            "entrez_email": self.cohort_config.geo_config.entrez_email,
        }

    # ------------------------------------------------------------------
    # GLOBAL DIRECTORY MANAGEMENT
    # ------------------------------------------------------------------
    def _ensure_base_structure(self):
        """Ensure global directories exist for all agents and shared data."""
        base = self.paths.get_base_cohort_path()
        base.mkdir(parents=True, exist_ok=True)

        for agent in AgentName:
            self.paths.get_agent_base_path(agent.value).mkdir(parents=True, exist_ok=True)

        # Shared directories
        for sub in ["analysis", "reports", "logs", "cache"]:
            Path(self.paths.base_project_dir, self.paths.shared_data_dir, sub).mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # AGENT/DATASET PATHS
    # ------------------------------------------------------------------
    def ensure_agent_dirs(self, agent: AgentName, disease_name: str) -> Dict[str, Path]:
        """
        Create all directories for an agent/disease combination (idempotent).
        Returns dict of paths.
        """
        return self.paths.create_all_directories(agent.value, disease_name)

    def get_agent_disease_dir(self, agent: AgentName, disease_name: str) -> Path:
        """Return the main base directory for a given agent and disease."""
        return self.paths.get_disease_path(agent.value, disease_name)

    def get_analysis_dir(self, disease_name: str, create: bool = True) -> Path:
        path = self.paths.get_analysis_path(disease_name)
        if create:
            path.mkdir(parents=True, exist_ok=True)
        return path

    def get_reports_root(self, create: bool = True) -> Path:
        path = Path(self.paths.base_project_dir, self.paths.shared_data_dir, "reports")
        if create:
            path.mkdir(parents=True, exist_ok=True)
        return path

    def get_report_dir(self, report_type: str, create: bool = True) -> Path:
        """Return directory for a specific report type (e.g. clinical_report, pharma_report)."""
        base = self.get_reports_root(create=create)
        path = base / report_type
        if create:
            path.mkdir(parents=True, exist_ok=True)
        return path

    # ------------------------------------------------------------------
    # SERIALIZATION FOR WORKFLOW STATE
    # ------------------------------------------------------------------
    def as_state_dict(self) -> Dict[str, Any]:
        """Convert to plain dict for LangGraph injection."""
        return {
            "base_dir": str(Path(self.paths.base_project_dir).resolve()),
            "shared_dir": str(Path(self.paths.shared_data_dir).resolve()),
            "cohort_data_dir": str(self.paths.get_base_cohort_path().resolve()),
            "reports_dir": str(self.get_reports_root().resolve()),
            "analysis_dir": str(self.get_analysis_dir('example').parent.resolve()),
            "defaults": self.defaults,
        }


# Singleton factory
_GLOBAL_CONFIG: GlobalAgentConfig | None = None

def get_global_config() -> GlobalAgentConfig:
    global _GLOBAL_CONFIG
    if _GLOBAL_CONFIG is None:
        _GLOBAL_CONFIG = GlobalAgentConfig()
    return _GLOBAL_CONFIG
