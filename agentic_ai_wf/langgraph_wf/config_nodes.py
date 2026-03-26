"""
Centralized node output path configuration.

All node output directories and file paths flow through this module.
Extensible via env vars and consistent with GlobalAgentConfig/DirectoryPathsConfig.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class NodePathsConfig:
    """
    Node output path configuration.
    Uses AGENTIC_AI_BASE_DIR and SHARED_DATA_DIR for consistency with DirectoryPathsConfig.
    """

    base_dir: str = field(
        default_factory=lambda: os.getenv("AGENTIC_AI_BASE_DIR", "./agentic_ai_wf")
    )
    shared_dir: str = field(
        default_factory=lambda: os.getenv("SHARED_DATA_DIR", "shared")
    )

    # Subdirs relative to base_dir/shared_dir (add new nodes here)
    subdirs: dict = field(
        default_factory=lambda: {
            "cohort_bulk": "cohort_data/bulk",
            "cohort_single_cell": "cohort_data/single_cell",
            "cohort_fastq": "cohort_data/fastq_data",
            "deconvolution": "deconv_data",
            "temporal": "temporal_data",
            "perturbation": "perturbation_data",
            "harmonization": "harmonization_data",
            "multiomics": "multiomics_data",
            "mdp": "mdp_data",
            "ipaa": "ipaa_data",
            "single_cell": "single_cell_data",
            "fastq": "fastq_data",
            "crispr": "crispr_data",
            "crispr_targeted": "crispr_data/targeted",
            "crispr_screening": "crispr_data/screening",
            "drug_discovery": "drugs_discovery",
            "pharma_reports": "reports/pharma_reports",
            "gwas_mr": "gwas_mr_data",
        },
        repr=False,
    )

    def _shared_root(self) -> Path:
        return Path(self.base_dir) / self.shared_dir

    def get_dir(
        self,
        node_key: str,
        analysis_id: str,
        *,
        subdir_suffix: Optional[str] = None,
        disease_or_id: Optional[str] = None,
        create: bool = True,
    ) -> Path:
        """
        Get output directory for a node.

        Args:
            node_key: Key from subdirs (e.g. "deconvolution", "cohort_bulk")
            analysis_id: Analysis identifier
            subdir_suffix: Optional extra subdir (e.g. technique name)
            disease_or_id: For cohort nodes: use disease name or analysis_id when unnamed
            create: Create directory if missing
        """
        subdir = self.subdirs.get(node_key)
        if not subdir:
            raise KeyError(f"Unknown node_key: {node_key}. Known: {list(self.subdirs.keys())}")

        if disease_or_id is not None:
            leaf = disease_or_id
        else:
            leaf = analysis_id

        path = self._shared_root() / subdir / leaf
        if subdir_suffix:
            path = path / subdir_suffix

        if create:
            path.mkdir(parents=True, exist_ok=True)
        return path.resolve()

    def get_file_path(
        self,
        node_key: str,
        analysis_id: str,
        filename: str,
        *,
        create_parent: bool = True,
    ) -> Path:
        """Get file path under a node output dir."""
        parent = self.get_dir(node_key, analysis_id, create=create_parent)
        return (parent / filename).resolve()

    def get_base_dir(self, node_key: str, create: bool = True) -> Path:
        """Get base dir for node (no analysis_id); e.g. cohort_fastq input discovery."""
        subdir = self.subdirs.get(node_key)
        if not subdir:
            raise KeyError(f"Unknown node_key: {node_key}")
        path = self._shared_root() / subdir
        if create:
            path.mkdir(parents=True, exist_ok=True)
        return path.resolve()


# Singleton; nodes import get_node_dir / get_node_file for convenience
_config: Optional[NodePathsConfig] = None


def get_node_paths_config() -> NodePathsConfig:
    global _config
    if _config is None:
        _config = NodePathsConfig()
    return _config


def get_node_dir(
    node_key: str,
    analysis_id: str,
    *,
    subdir_suffix: Optional[str] = None,
    disease_or_id: Optional[str] = None,
    create: bool = True,
) -> Path:
    """Convenience: get output dir for a node."""
    return get_node_paths_config().get_dir(
        node_key, analysis_id,
        subdir_suffix=subdir_suffix,
        disease_or_id=disease_or_id,
        create=create,
    )


def get_node_file(
    node_key: str,
    analysis_id: str,
    filename: str,
    *,
    create_parent: bool = True,
) -> Path:
    """Convenience: get file path under node output dir."""
    return get_node_paths_config().get_file_path(
        node_key, analysis_id, filename, create_parent=create_parent
    )


def get_node_base_dir(node_key: str, create: bool = True) -> Path:
    """Convenience: get base dir for node (no analysis_id)."""
    return get_node_paths_config().get_base_dir(node_key, create=create)
