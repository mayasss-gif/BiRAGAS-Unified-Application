from dataclasses import dataclass, field
from typing import Dict


@dataclass
class PipelineConfig:
    """Configuration for the DAG Builder pipeline.

    Required:
        disease_name:  Human-readable disease name (used for trait node labelling and outputs).

    All file paths must be absolute. Disease-specific filenames (empty-string
    defaults) must be supplied by the caller.
    """

    disease_name: str

    # --- Observational & Transcriptomic ---
    RAW_COUNTS: str = ""
    DEGS_FULL: str = ""
    METADATA: str = ""
    DEGS_PRIORITY: str = ""

    # --- Functional & Cellular ---
    PATHWAYS: str = ""
    DECONVOLUTION: str = ""

    # --- Temporal ---
    TEMPORAL_FITS: str = ""
    GRANGER_EDGES: str = ""

    # --- Perturbation ---
    PERTURBATION: str = ""
    ESSENTIALITY: str = ""
    DRUG_LINKS: str = ""
    CAUSAL_DRIVERS: str = ""

    # --- Priors & Genetics (GWAS/MR) ---
    GWAS_ASSOC: str = ""
    SIGNOR_EDGES: str = ""
    GENETIC_PRIORS: str = ""
    GENE_EVIDENCE: str = ""
    VARIANT_EVIDENCE: str = ""
    MR_RESULTS: str = ""

    # --- Analysis Thresholds & Weights ---
    CONSENSUS_THRESHOLD: float = 0.20
    CORRELATION_THRESHOLD: float = 0.4

    WEIGHTS: Dict[str, float] = field(default_factory=lambda: {
        'GWAS': 0.90,
        'MR': 0.95,
        'CRISPR': 0.85,
        'SIGNOR': 0.90,
        'TEMPORAL': 0.65,
        'STATISTICAL': 0.35,
    })
