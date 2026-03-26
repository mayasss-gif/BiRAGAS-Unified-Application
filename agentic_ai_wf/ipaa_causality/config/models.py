"""
Pydantic config models for IPAA pipeline.
Replaces argparse-based CLI with structured config objects.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field


# -----------------------------------------------------------------------------
# Item spec (cohort definition)
# -----------------------------------------------------------------------------


class ItemSpec(BaseModel):
    """Single cohort item: name, input path, optional metadata and tissue."""
    name: str
    input: str
    meta: Optional[str] = None
    tissue: Optional[str] = None


# -----------------------------------------------------------------------------
# Phase-specific configs
# -----------------------------------------------------------------------------


class IPAAPhaseConfig(BaseModel):
    """Config for IPAA M6 preprocessing and enrichment phase."""
    outdir: str
    items: List[Union[ItemSpec, dict]]
    spec: Optional[str] = None
    counts_default: bool = False
    verbose: bool = True
    refresh_omnipath: bool = False
    workers: int = 0
    threads_per_cohort: int = 0
    report_top: int = 20
    threads: int = 6
    gsea_permutations: int = 200
    sig_fdr: float = 0.05
    sig_top_n: int = 200
    msigdb_dbver: str = "2024.1.Hs"
    run_baseline: bool = True
    baseline_dir: str = ""
    auto_select_tissue: bool = True
    tissue_top_k: int = 3
    skip_engine1: bool = False
    engine1_strict: bool = False
    engine1_license_mode: str = "academic"
    engine1_tf_method: str = "viper"
    engine1_tmin: int = 5
    engine1_no_overwrite: bool = False
    engine1_no_regulators_evidence: bool = False


class CausalityPhaseConfig(BaseModel):
    """Config for causality engines (Engine0–3) phase."""
    out_root: str
    diseases: Optional[List[str]] = None
    run_all: bool = True
    strict: bool = False
    log_level: str = "INFO"
    engine23_script: Optional[str] = None
    # Engine toggles
    no_engine0: bool = False
    no_engine1: bool = False
    no_engine2: bool = False
    no_engine3: bool = False
    no_omnipath_layer: bool = False
    no_refresh_omnipath_cache: bool = False
    no_build_pkn: bool = False
    no_refresh_pkn: bool = False
    no_force_engine1: bool = False
    no_force_engine0: bool = False
    signor_edges: Optional[str] = None
    ptm_min_substrates: int = 5
    ptm_n_perm: int = 200
    engine1_method: str = "mean"
    engine1_min_size: int = 10
    engine1_max_pathways: int = 1500
    engine1_max_tfs: int = 300
    corr_method: str = "spearman"
    corr_flag_threshold: float = 0.40
    min_markers: int = 5
    pkn_edges: Optional[str] = None
    max_steps: int = 3
    top_tfs: int = 30
    confound_penalty_threshold: float = 0.40


class AggregationPhaseConfig(BaseModel):
    """Config for pathway summary aggregation phase."""
    out_root: str
    diseases: Optional[List[str]] = None
    out_subdir: Optional[str] = None
    filter_fdr: float = 0.05


class ReportPhaseConfig(BaseModel):
    """Config for HTML report generation phase."""
    outdir: str
    topn: int = 20
    fdr_cutoff: float = 0.10
    candidate_pool: int = 30
    llm_selector: str = "auto"  # "off" | "on" | "auto"
    llm_narrative: str = "auto"
    llm_model: str = "gpt-4o-mini"
    api_key_env: str = "OPENAI_API_KEY"
    generate_pdf: bool = False


# -----------------------------------------------------------------------------
# Unified IPAA config (all phases)
# -----------------------------------------------------------------------------


class IPAAConfig(BaseModel):
    """Unified config for full IPAA pipeline."""
    outdir: str
    items: List[Union[ItemSpec, dict]] = Field(default_factory=list)
    spec: Optional[str] = None
    counts_default: bool = False
    verbose: bool = True
    refresh_omnipath: bool = False
    workers: int = 0
    threads_per_cohort: int = 0
    report_top: int = 20
    threads: int = 6
    gsea_permutations: int = 200
    sig_fdr: float = 0.05
    sig_top_n: int = 200
    msigdb_dbver: str = "2024.1.Hs"
    run_baseline: bool = True
    baseline_dir: str = ""
    auto_select_tissue: bool = True
    tissue_top_k: int = 3
    skip_engine1: bool = False
    engine1_strict: bool = False
    engine1_license_mode: str = "academic"
    engine1_tf_method: str = "viper"
    engine1_tmin: int = 5
    engine1_no_overwrite: bool = False
    engine1_no_regulators_evidence: bool = False
    # Stage toggles
    skip_ipaa: bool = False
    skip_causality: bool = False
    skip_aggregator: bool = False
    skip_html_report: bool = False
    # Causality overrides
    no_engine0: bool = False
    no_engine1: bool = False
    no_engine2: bool = False
    no_engine3: bool = False
    no_omnipath_layer: bool = False
    strict: bool = False
    diseases: Optional[List[str]] = None
    run_all: bool = True
    aggregator_diseases: Optional[str] = None
    aggregator_out_subdir: Optional[str] = None
    # Report overrides
    report_topn: int = 20
    report_fdr_cutoff: float = 0.10
    report_llm_selector: str = "auto"
    report_llm_narrative: str = "auto"
    report_llm_model: str = "gpt-4o-mini"
    report_api_key_env: str = "OPENAI_API_KEY"
    generate_pdf_report: bool = False
    # Causality engine23
    engine23_script: Optional[str] = None
    no_refresh_omnipath_cache: bool = False
    no_build_pkn: bool = False
    no_refresh_pkn: bool = False
    no_force_engine1: bool = False
    no_force_engine0: bool = False
    signor_edges: Optional[str] = None
    ptm_min_substrates: int = 5
    ptm_n_perm: int = 200
    engine1_method: str = "mean"
    engine1_min_size: int = 10
    engine1_max_pathways: int = 1500
    engine1_max_tfs: int = 300
    corr_method: str = "spearman"
    corr_flag_threshold: float = 0.40
    min_markers: int = 5
    pkn_edges: Optional[str] = None
    max_steps: int = 3
    top_tfs: int = 30
    confound_penalty_threshold: float = 0.40
    log_level: str = "INFO"


# -----------------------------------------------------------------------------
# Result types (dataclasses for mutable results)
# -----------------------------------------------------------------------------


@dataclass
class IPAAPhaseResult:
    success: bool = True
    out_root: str = ""
    cohort_runs: List[Dict[str, str]] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "out_root": self.out_root,
            "cohort_runs": self.cohort_runs,
            "errors": self.errors,
            "metadata": self.metadata,
        }


@dataclass
class CausalityPhaseResult:
    success: bool = True
    out_root: str = ""
    diseases_processed: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "out_root": self.out_root,
            "diseases_processed": self.diseases_processed,
            "errors": self.errors,
            "metadata": self.metadata,
        }


@dataclass
class AggregationPhaseResult:
    success: bool = True
    out_root: str = ""
    diseases_aggregated: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "out_root": self.out_root,
            "diseases_aggregated": self.diseases_aggregated,
            "errors": self.errors,
            "metadata": self.metadata,
        }


@dataclass
class ReportPhaseResult:
    success: bool = True
    outdir: str = ""
    reports_generated: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "outdir": self.outdir,
            "reports_generated": self.reports_generated,
            "errors": self.errors,
            "metadata": self.metadata,
        }


@dataclass
class IPAAResult:
    """Full pipeline result."""
    out_root: str = ""
    status: str = "ok"  # "ok" | "partial" | "failed"
    ipaa: Optional[IPAAPhaseResult] = None
    causality: Optional[CausalityPhaseResult] = None
    aggregation: Optional[AggregationPhaseResult] = None
    reports: Optional[ReportPhaseResult] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "out_root": self.out_root,
            "status": self.status,
            "ipaa": self.ipaa.to_dict() if self.ipaa else None,
            "causality": self.causality.to_dict() if self.causality else None,
            "aggregation": self.aggregation.to_dict() if self.aggregation else None,
            "reports": self.reports.to_dict() if self.reports else None,
            "metadata": self.metadata,
            "errors": self.errors,
        }
