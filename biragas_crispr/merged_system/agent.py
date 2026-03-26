"""
agent.py — consolidated agent module for the molecular causal discovery pipeline.

Sections:
    1.  Imports & constants
    2.  Enums  (IntentID, GateStatus, FilePhase)
    3.  Dataclasses  (GateResult, FileAuditResult, LiteraturePaper, LitClaim,
                      LitBrief, ParsedIntent, FinalResult, ClarificationResult)
    4.  PLATFORM_TOOLS list
    5.  CAUSAL_MODULE_CHAINS dict
    6.  FileInspector class
    7.  EligibilityChecker class
    8.  WorkflowRouter class
    9.  SupervisorAgent class

intelligence.py is imported lazily inside SupervisorAgent.__init__ to avoid
circular imports (intelligence.py imports agent types under TYPE_CHECKING only).
"""
from __future__ import annotations

# ── Section 1: Imports & constants ────────────────────────────────────────────
import hashlib
import io
import json
import logging
import os
import re
import sys
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import pandas as pd

# Ensure stdout handles Unicode on Windows
if sys.stdout and hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

log = logging.getLogger("causal_platform")

ANTHROPIC_MODEL      = "claude-sonnet-4-20250514"
MAX_TOKENS_INTENT    = 800
MAX_TOKENS_LIT       = 1000
MAX_TOKENS_NARRATE   = 400
MAX_TOKENS_RESULT    = 1200
MAX_TOKENS_CLAIMS    = 1000
MAX_TOKENS_BRIEF     = 1000

PUBMED_BASE  = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
EPMC_BASE    = "https://www.ebi.ac.uk/europepmc/webservices/rest"
S2_BASE      = "https://api.semanticscholar.org/graph/v1"

LIT_MAX_PAPERS      = 30
LIT_TOP_K           = 15
LIT_TIMEOUT         = 12.0
LIT_CACHE_TTL_DAYS  = 7

GATE_MIN_SAMPLES        = 30
GATE_WARN_SAMPLES       = 60
GATE_MIN_MR_INSTRUMENTS = 3
GATE_EDGE_CONFIDENCE    = 0.70

EVIDENCE_WEIGHTS = {
    "genetic":      0.30,
    "perturbation": 0.25,
    "temporal":     0.20,
    "network":      0.15,
    "expression":   0.05,
    "immuno":       0.05,
}


# ── Section 2: Enums ──────────────────────────────────────────────────────────

class IntentID(str, Enum):
    I_01 = "I_01"   # Causal Drivers Discovery
    I_02 = "I_02"   # Directed Causality X->Y
    I_03 = "I_03"   # Intervention / Actionability
    I_04 = "I_04"   # Comparative Causality
    I_05 = "I_05"   # Counterfactual / What-If
    I_06 = "I_06"   # Evidence Inspection / Explain
    I_07 = "I_07"   # Standard Association Analysis


class GateStatus(str, Enum):
    PASS  = "pass"
    WARN  = "warn"
    BLOCK = "block"
    SKIP  = "skip"


class FilePhase(str, Enum):
    BIO          = "bio_prep"
    INTERVENTION = "intervention_prep"
    DAG          = "dag_build"
    CAUSAL       = "causal_core"
    ACQUISITION  = "acquisition"
    UNKNOWN      = "unknown"


# ── Section 3: Dataclasses ────────────────────────────────────────────────────

@dataclass
class GateResult:
    gate_id: str
    name:    str
    status:  GateStatus
    message: str
    value:   Any = None


@dataclass
class FileAuditResult:
    file_path:     str
    file_name:     str
    type_id:       str
    type_label:    str
    phase:         FilePhase
    platform_tool: str
    errors:        list = field(default_factory=list)
    warnings:      list = field(default_factory=list)
    infos:         list = field(default_factory=list)
    gates:         list = field(default_factory=list)
    n_rows:        int  = 0
    n_cols:        int  = 0
    columns:       list = field(default_factory=list)

    @property
    def has_errors(self) -> bool:
        return bool(self.errors)

    @property
    def has_warnings(self) -> bool:
        return bool(self.warnings)


@dataclass
class LiteraturePaper:
    pmid:            Optional[str] = None
    doi:             Optional[str] = None
    title:           str = ""
    abstract:        str = ""
    authors:         str = ""
    year:            Optional[int] = None
    journal:         str = ""
    source:          str = ""
    relevance_score: float = 0.0


@dataclass
class LitClaim:
    entity_x:      str
    relation:      str
    entity_y:      str
    direction:     str
    evidence_type: str
    strength:      str
    pmid:          Optional[str] = None
    quote:         str = ""
    confidence:    float = 0.0


@dataclass
class LitBrief:
    inferred_context:       str  = ""
    key_entities:           list = field(default_factory=list)
    search_queries_used:    list = field(default_factory=list)
    papers_found:           int  = 0
    papers_processed:       int  = 0
    claims:                 list = field(default_factory=list)
    high_confidence_edges:  list = field(default_factory=list)
    conflicts:              list = field(default_factory=list)
    causal_vs_associative:  str  = "mixed"
    prior_evidence_summary: str  = ""
    recommended_modules:    list = field(default_factory=list)
    data_gaps:              list = field(default_factory=list)
    supervisor_brief:       str  = ""
    conflict_rate:          float = 0.0
    error:                  Optional[str] = None


@dataclass
class ParsedIntent:
    intent_id:              IntentID
    intent_name:            str
    confidence:             float
    needs_clarification:    bool
    clarifying_question:    Optional[str]
    context:                dict = field(default_factory=dict)
    entities:               dict = field(default_factory=dict)
    requires:               dict = field(default_factory=dict)
    needs_literature_first: bool = True
    requires_existing_dag:  bool = False
    routing_summary:        str  = ""
    module_chain:           list = field(default_factory=list)
    parallel_blocks:        list = field(default_factory=list)
    fallback:               str  = ""


@dataclass
class FinalResult:
    headline:            str  = ""
    analyzed_context:    str  = ""
    top_findings:        list = field(default_factory=list)
    tier1_candidates:    list = field(default_factory=list)
    tier2_candidates:    list = field(default_factory=list)
    actionable_targets:  list = field(default_factory=list)
    evidence_quality:    dict = field(default_factory=dict)
    caveats:             list = field(default_factory=list)
    next_experiments:    list = field(default_factory=list)
    missing_data_impact: list = field(default_factory=list)
    modules_run:         list = field(default_factory=list)
    artifacts_produced:  list = field(default_factory=list)


# ClarificationResult is defined in intelligence.py but re-exported here
# for callers that do `from agent import ClarificationResult`.
# We import it lazily at first use to avoid circular imports.
def _get_clarification_result_class():
    from intelligence import ClarificationResult
    return ClarificationResult


# ── Section 4: PLATFORM_TOOLS ─────────────────────────────────────────────────

PLATFORM_TOOLS: list[dict] = [
    {"id": "T_00", "label": "Cohort Data Retrieval", "phase": "acquisition",
     "tool": "cohort_data_retrieval()", "outputs": ["raw_counts.csv", "metadata.csv"],
     "desc": "Retrieves a matching cohort from GEO/SRA using disease name from query.",
     "auto_fetch": True, "auto_fetch_source": "GEO / SRA by disease name",
     "condition": lambda p: p["no_data"]},

    {"id": "T_04b", "label": "Single-Cell RNA-seq Pipeline", "phase": "bio_prep",
     "tool": "sc_pipeline()", "outputs": ["sc_norm.h5ad", "cell_annotations.csv", "pseudobulk.csv"],
     "desc": "QC, normalisation, clustering, and cell-type annotation for scRNA-seq data.",
     "auto_fetch": False, "auto_fetch_source": "",
     "condition": lambda p: p["has_sc"]},

    {"id": "T_01", "label": "Expression Normalization", "phase": "bio_prep",
     "tool": "normalize_expression()", "outputs": ["expr_norm.parquet", "qc_report.json"],
     "desc": "Log2-CPM normalisation, QC filtering, and variance-based gene selection.",
     "auto_fetch": False, "auto_fetch_source": "",
     "condition": lambda p: p["has_expression"] and not p["has_deg_output"]},

    {"id": "T_02", "label": "Differential Expression (DESeq2)", "phase": "bio_prep",
     "tool": "run_deseq2()", "outputs": ["DEGs_prioritized.csv"],
     "desc": "Case vs control DEG analysis using phenotype labels from metadata.",
     "auto_fetch": False, "auto_fetch_source": "",
     "condition": lambda p: p["has_metadata"] and not p["has_deg_output"]},

    {"id": "T_03", "label": "Pathway Enrichment", "phase": "bio_prep",
     "tool": "pathway_enrichment()", "outputs": ["Pathways_Enrichment.csv"],
     "desc": "ORA + GSEA across GO / KEGG / Reactome / MSigDB using DEG results.",
     "auto_fetch": False, "auto_fetch_source": "",
     "condition": lambda p: not p["has_pathway_output"]},

    {"id": "T_04", "label": "Cell-Type Deconvolution", "phase": "bio_prep",
     "tool": "deconvolution()", "outputs": ["cell_fractions.csv", "deconv_confidence.json"],
     "desc": "BisQue (SC reference), CIBERSORT (uploaded matrix), or LM22 (built-in) deconvolution.",
     "auto_fetch": True, "auto_fetch_source": "built-in LM22 immune markers (or BisQue if SC uploaded)",
     "condition": lambda p: True},

    {"id": "T_05", "label": "Temporal / Pseudotime Pipeline", "phase": "bio_prep",
     "tool": "temporal_pipeline()", "outputs": ["temporal_gene_fits.tsv", "granger_edges_raw.csv"],
     "desc": "Impulse model + Granger causality; infers pseudotime if no temporal files uploaded.",
     "auto_fetch": True, "auto_fetch_source": "pseudotime inference from expression (no real timepoints needed)",
     "condition": lambda p: not p["has_temporal"]},

    {"id": "T_06", "label": "CRISPR / Perturbation Pipeline", "phase": "intervention_prep",
     "tool": "perturbation_pipeline()", "outputs": ["CausalDrivers_Ranked.csv", "GeneEssentiality_ByMedian.csv"],
     "desc": "ACE scoring; uses uploaded CRISPR files or auto-fetches DepMap Avana + LINCS L1000.",
     "auto_fetch": True, "auto_fetch_source": "DepMap Avana CRISPR screen + LINCS L1000 by disease name",
     "condition": lambda p: True},

    {"id": "T_07", "label": "Prior Knowledge Network (SIGNOR)", "phase": "intervention_prep",
     "tool": "prior_knowledge_pipeline()", "outputs": ["SIGNOR_Subnetwork_Edges.tsv", "kg_annotations.json"],
     "desc": "Uses uploaded SIGNOR edges; else auto-fetches STRING v12 + KEGG + SIGNOR API.",
     "auto_fetch": True, "auto_fetch_source": "STRING v12 + KEGG + SIGNOR by disease gene set",
     "condition": lambda p: True},

    {"id": "T_08", "label": "GWAS / eQTL / MR Preprocessing", "phase": "intervention_prep",
     "tool": "gwas_eqtl_mr_pipeline()",
     "outputs": ["genetic-evidence.xlsx", "GeneLevel_GeneticEvidence.tsv", "MR_MAIN_RESULTS_ALL_GENES.csv"],
     "desc": "Uses uploaded GWAS/eQTL/MR files; else auto-fetches GWAS Catalog + eQTL Catalogue.",
     "auto_fetch": True, "auto_fetch_source": "GWAS Catalog + eQTL Catalogue + OpenGWAS by disease name",
     "condition": lambda p: True},
]


# ── Section 5: CAUSAL_MODULE_CHAINS ──────────────────────────────────────────

CAUSAL_MODULE_CHAINS: dict = {
    "I_01": [
        {"id": "M12",  "label": "DAGBuilder",           "phase": "dag_build",
         "algo": "PC/FCI/GES consensus + multi-modal evidence fusion + MR edge constraints + 1000x bootstrap",
         "outputs": ["consensus_causal_dag.json"]},
        {"id": "M13",  "label": "DAGValidator",          "phase": "causal_core",
         "algo": "Edge stability + PPI + pathway coherence validation",
         "outputs": ["validated_dag.json", "edge_stability.csv"]},
        {"id": "M14",  "label": "CentralityCalculator",  "phase": "causal_core",
         "algo": "kME + betweenness + PageRank -> causal_importance_score -> Tier 1/2/3",
         "outputs": ["causal_importance_scores.csv", "ranked_targets.csv", "tier_assignments.csv"]},
        {"id": "M15",  "label": "EvidenceAggregator",    "phase": "causal_core",
         "algo": "6-stream weighted fusion: Gen(0.30)+Pert(0.25)+Temp(0.20)+Net(0.15)+Expr(0.05)+Imm(0.05)",
         "outputs": ["evidence_matrix.parquet", "conflict_flags.csv"]},
        {"id": "M_DC", "label": "DoCalculusEngine",      "phase": "causal_core",
         "algo": "Backdoor Criterion + do-calculus: directionality test + confounder removal",
         "outputs": ["do_calculus_results.json", "backdoor_sets.csv", "causal_effects.csv"]},
    ],
    "I_02": [
        {"id": "M12",  "label": "DAGBuilder (targeted)", "phase": "dag_build",
         "algo": "Targeted feature set for X->Y test + MR constraints + 1000x bootstrap",
         "outputs": ["consensus_causal_dag.json"]},
        {"id": "M_DC", "label": "DoCalculusEngine",      "phase": "causal_core",
         "algo": "Backdoor Criterion: tests X->Y directionality and removes confounding",
         "outputs": ["do_calculus_results.json", "causal_effects.csv"]},
    ],
    "I_03": [
        {"id": "M_IS", "label": "InSilicoSimulator",        "phase": "causal_core",
         "algo": "do(X=0) graph propagation: dose-response + compensation + resistance discovery",
         "outputs": ["predicted_changes.csv", "compensation_pathways.json", "resistance_mechanisms.csv"]},
        {"id": "M_PI", "label": "PharmaInterventionEngine", "phase": "causal_core",
         "algo": "DrugBank + ChEMBL + DepMap + LINCS: therapeutic efficacy vs systemic safety -> Target Product Profile",
         "outputs": ["target_product_profile.json", "prioritized_targets.csv", "drug_candidates.csv"]},
    ],
    "I_04": [
        {"id": "M12",   "label": "DAGBuilder (group A)", "phase": "dag_build",
         "algo": "Group A DAG: PC/FCI + multi-modal + MR constraints + bootstrap",
         "outputs": ["consensus_dag_a.json"]},
        {"id": "M12b",  "label": "DAGBuilder (group B)", "phase": "dag_build",
         "algo": "Group B DAG: PC/FCI + multi-modal + MR constraints + bootstrap (parallel)",
         "outputs": ["consensus_dag_b.json"]},
        {"id": "M13",   "label": "DAGValidator",          "phase": "causal_core",
         "algo": "Bootstrap + PPI + pathway coherence per group",
         "outputs": ["validated_dags.json", "edge_stability_per_group.csv"]},
        {"id": "M14",   "label": "CentralityCalculator",  "phase": "causal_core",
         "algo": "kME + PageRank per group -> causal_importance_score per group",
         "outputs": ["causal_importance_per_group.csv", "ranked_targets_per_group.csv"]},
        {"id": "M15",   "label": "EvidenceAggregator",    "phase": "causal_core",
         "algo": "6-stream weighted evidence integration per group",
         "outputs": ["evidence_matrix_per_group.parquet"]},
        {"id": "M_DC",  "label": "DoCalculusEngine",      "phase": "causal_core",
         "algo": "Backdoor Criterion per group + delta directionality comparison",
         "outputs": ["do_calculus_per_group.json", "direction_delta.json"]},
        {"id": "DELTA", "label": "Delta Analysis",        "phase": "causal_core",
         "algo": "Shared/group-specific/lost edges + delta causal_importance_scores",
         "outputs": ["delta_graph.csv", "conserved_edges.csv", "group_specific_drivers.csv"]},
    ],
    "I_05": [
        {"id": "M_IS", "label": "InSilicoSimulator",        "phase": "causal_core",
         "algo": "do(X=0) counterfactual propagation: reveal resistance + compensation pathways",
         "outputs": ["predicted_changes.csv", "compensation_pathways.json", "resistance_mechanisms.csv"],
         "optional": False},
        {"id": "M_DC", "label": "DoCalculusEngine",          "phase": "causal_core",
         "algo": "Validates intervention target is causal node + residual confounding check",
         "outputs": ["do_calculus_results.json"],
         "optional": True},
        {"id": "M_PI", "label": "PharmaInterventionEngine",  "phase": "causal_core",
         "algo": "Drug prioritisation for simulated intervention",
         "outputs": ["target_product_profile.json"],
         "optional": True},
    ],
    "I_06": [
        {"id": "M15", "label": "EvidenceAggregator (read-only)", "phase": "causal_core",
         "algo": "Retrieve evidence matrix and conflict flags from prior run",
         "outputs": ["evidence_breakdown.json"]},
        {"id": "M14", "label": "CentralityCalculator (read-only)", "phase": "causal_core",
         "algo": "Retrieve causal_importance_scores and Tier assignments from prior run",
         "outputs": ["gap_analysis.json", "citations.csv"]},
    ],
    "I_07": [],
}


# ── Section 6: FileInspector ──────────────────────────────────────────────────

class FileInspector:
    """
    Inspects an uploaded data file and returns a FileAuditResult containing:
      - Detected file type (type_id, type_label, phase, platform_tool)
      - Quality audit messages (errors, warnings, infos)
      - Per-file eligibility gate results (G_01, G_02, G_06, G_07)

    Detection uses filename patterns first, then column signatures.
    """

    BINARY_EXTS = {".h5ad", ".pkl", ".rds", ".h5", ".hdf5", ".bam", ".bai", ".loom"}

    # (type_id, label, phase, tool, filename_pattern)
    _FILENAME_RULES: list[tuple] = [
        ("expression",      "Expression / count matrix",
         FilePhase.BIO, "T_01",
         r"_raw_count\.tsv$|_counts\.csv$|rnaseq"),
        ("metadata",        "Sample metadata with outcome labels",
         FilePhase.ACQUISITION, "T_01",
         r"prep_meta\.csv$|-METADATA\.xlsx$|_meta\.csv$|metadata"),
        ("deg_output",      "DEG output (pre-computed)",
         FilePhase.BIO, "T_02",
         r"_DEGs_prioritized\.csv$|DEG"),
        ("pathway_output",  "Pathway enrichment output (pre-computed)",
         FilePhase.BIO, "T_03",
         r"_Pathways_Enrichment\.csv$|pathway.*enrichment"),
        ("signature_matrix","Cell-type signature / deconvolution reference matrix",
         FilePhase.BIO, "T_04",
         r"signature_matrix\.tsv$|signature_matrix"),
        ("gwas_mr",         "GWAS / MR results",
         FilePhase.INTERVENTION, "T_08",
         r"__genetic-evidence\.xlsx$|genetic.*evidence"),
        ("eqtl_gene",       "eQTL gene-level genetic evidence",
         FilePhase.INTERVENTION, "T_08",
         r"_GeneLevel_GeneticEvidence\.tsv$"),
        ("eqtl_variant",    "eQTL variant-level genetic evidence",
         FilePhase.INTERVENTION, "T_08",
         r"_VariantLevel_GeneticEvidence\.tsv$"),
        ("mr_results",      "Mendelian Randomization results",
         FilePhase.INTERVENTION, "T_08",
         r"MR_MAIN_RESULTS_ALL_GENES\.csv$|MR_MAIN"),
        ("crispr_guide",    "CRISPR guide-level screen",
         FilePhase.INTERVENTION, "T_06",
         r"CRISPR_GuideLevel_Avana|CRISPR.*GuideLevel"),
        ("crispr_essentiality", "Gene essentiality scores",
         FilePhase.INTERVENTION, "T_06",
         r"GeneEssentiality_ByMedian\.csv$|GeneEssentiality"),
        ("crispr_ranked",   "Causal drivers ranked",
         FilePhase.INTERVENTION, "T_06",
         r"CausalDrivers_Ranked\.csv$|CausalDrivers"),
        ("temporal_fits",   "Temporal / impulse model fits",
         FilePhase.BIO, "T_05",
         r"temporal_gene_fits\.tsv$|temporal.*gene.*fits"),
        ("granger_edges",   "Granger causality edges",
         FilePhase.BIO, "T_05",
         r"granger_edges_raw\.csv$|granger_edges"),
        ("prior_network",   "Prior knowledge network (SIGNOR / STRING / KEGG)",
         FilePhase.INTERVENTION, "T_07",
         r"SIGNOR_Subnetwork_Edges\.tsv$|SIGNOR"),
    ]

    def __init__(self):
        self._col_signatures = self._build_col_signatures()

    def _build_col_signatures(self) -> list[tuple]:
        def _cols(df, *names) -> bool:
            c = set(df.columns.str.lower())
            return all(n in c for n in names)

        def _has_sample_col(df) -> bool:
            return any("sample" in c.lower() for c in df.columns)

        def _has_outcome_col(df) -> bool:
            kw = ["disease_status", "outcome", "phenotype", "label",
                  "group", "response", "status", "trait", "condition",
                  "diagnosis", "class", "category"]
            return any(any(k in c.lower() for k in kw) for c in df.columns)

        def _is_expression(df) -> bool:
            gene_kw = {"gene", "gene_id", "geneid", "ens-id", "ensembl",
                       "symbol", "feature_id", "transcript_id", "description"}
            col_lower = [c.lower() for c in df.columns]
            has_gene  = any(any(g in c for g in gene_kw) for c in col_lower)
            if not has_gene:
                return False
            non_gene = [c for c in df.columns
                        if not any(g in c.lower() for g in gene_kw)]
            if not non_gene:
                return False
            sample_like = sum(
                1 for c in non_gene
                if re.match(r"(GSM|SRR|ERR|DRR|SAMN|SRS|S\d+|sample[_\s]?\d+|\d+)", c, re.I)
            )
            return sample_like / len(non_gene) >= 0.5

        return [
            ("temporal_fits",   "Temporal / impulse model fits",
             FilePhase.BIO, "T_05",
             lambda df: _cols(df, "gene_id", "pattern", "r2_impulse")),
            ("granger_edges",   "Granger causality edges",
             FilePhase.BIO, "T_05",
             lambda df: _cols(df, "source", "target", "effect_f", "q_value")),
            ("crispr_ranked",   "CRISPR / ACE perturbation drivers",
             FilePhase.INTERVENTION, "T_06",
             lambda df: _cols(df, "gene", "ace", "rank")),
            ("gwas_mr",         "GWAS / MR results",
             FilePhase.INTERVENTION, "T_08",
             lambda df: _cols(df, "gene", "or", "b") and
                        any(c in df.columns.str.lower() for c in ["pval", "p_value", "pvalue"])),
            ("eqtl_gene",       "eQTL data",
             FilePhase.INTERVENTION, "T_08",
             lambda df: _cols(df, "gene", "snp", "slope")),
            ("prior_network",   "Prior knowledge network (SIGNOR / STRING / KEGG)",
             FilePhase.INTERVENTION, "T_07",
             lambda df: _cols(df, "source", "target", "mechanism")),
            ("gwas_mr",         "GWAS summary statistics",
             FilePhase.INTERVENTION, "T_08",
             lambda df: any(c in df.columns.str.lower()
                            for c in ["snp_id", "rsid", "snp", "variant_id"])),
            ("expression",      "Expression / count matrix",
             FilePhase.BIO, "T_01",
             lambda df: _is_expression(df)),
            ("metadata",        "Sample metadata with outcome labels",
             FilePhase.ACQUISITION, "T_01",
             lambda df: _has_sample_col(df) and _has_outcome_col(df)),
            ("metadata_partial","Sample metadata (no outcome column)",
             FilePhase.ACQUISITION, "T_01",
             lambda df: _has_sample_col(df)),
            ("signature_matrix","Cell-type signature / deconvolution reference matrix",
             FilePhase.BIO, "T_04",
             lambda df: any(c.lower() in ("gene", "gene_id", "geneid") for c in df.columns)
                        and len(df.columns) >= 4
                        and not any(c in df.columns.str.lower()
                                    for c in ["ace", "pattern", "mechanism", "source"])),
            # Fallback
            ("expression",      "Expression / count matrix",
             FilePhase.BIO, "T_01",
             lambda df: True),
        ]

    def inspect(self, file_path) -> FileAuditResult:
        path  = Path(file_path)
        fname = path.name

        # Binary / special formats
        if path.suffix.lower() in self.BINARY_EXTS:
            return FileAuditResult(
                file_path=str(path), file_name=fname,
                type_id="sc_data", type_label="Single-cell data (.h5ad / binary)",
                phase=FilePhase.BIO, platform_tool="T_04b",
                infos=["Single-cell binary detected -> T_04b (sc pipeline) runs before T_04 (deconvolution)."],
            )

        # Filename pattern matching (checked first)
        nl = fname.lower()
        for (tid, tlabel, tphase, ttool, pattern) in self._FILENAME_RULES:
            if re.search(pattern, nl, re.I):
                # Load file for quality audit
                try:
                    df = self._load_tabular(path, fname)
                except Exception as exc:
                    return FileAuditResult(
                        file_path=str(path), file_name=fname,
                        type_id=tid, type_label=tlabel,
                        phase=tphase, platform_tool=ttool,
                        errors=[f"Parse error: {exc}"],
                    )
                audit = FileAuditResult(
                    file_path=str(path), file_name=fname,
                    type_id=tid, type_label=tlabel,
                    phase=tphase, platform_tool=ttool,
                    n_rows=len(df), n_cols=len(df.columns),
                    columns=list(df.columns),
                )
                self._audit(audit, df)
                return audit

        # Column signature fallback
        try:
            df = self._load_tabular(path, fname)
        except Exception as exc:
            return FileAuditResult(
                file_path=str(path), file_name=fname,
                type_id="unknown", type_label="Could not parse",
                phase=FilePhase.UNKNOWN, platform_tool="?",
                errors=[f"Parse error: {exc}"],
            )

        df.attrs["filename"] = fname.lower()
        type_id, label, phase, tool = "unknown", "Unknown", FilePhase.UNKNOWN, "?"
        for (tid, tlabel, tphase, ttool, test) in self._col_signatures:
            try:
                if test(df):
                    type_id, label, phase, tool = tid, tlabel, tphase, ttool
                    break
            except Exception:
                continue

        audit = FileAuditResult(
            file_path=str(path), file_name=fname,
            type_id=type_id, type_label=label,
            phase=phase, platform_tool=tool,
            n_rows=len(df), n_cols=len(df.columns),
            columns=list(df.columns),
        )
        self._audit(audit, df)
        return audit

    @staticmethod
    def _load_tabular(path: Path, fname: str) -> pd.DataFrame:
        if fname.endswith((".xlsx", ".xls")):
            return pd.read_excel(path, nrows=300)
        sep = "\t" if fname.endswith((".tsv", ".txt")) else ","
        return pd.read_csv(path, sep=sep, nrows=300)

    def _audit(self, audit: FileAuditResult, df: pd.DataFrame):
        tid = audit.type_id
        n   = len(df)
        if tid in ("metadata",):
            self._audit_metadata(audit, df, n)
        elif tid == "metadata_partial":
            audit.errors.append(
                "G_01 BLOCK: No outcome/disease_status column found. "
                "Add a column named 'outcome', 'disease_status', 'phenotype', or 'group'."
            )
            audit.gates.append(GateResult("G_01", "Phenotype label completeness", GateStatus.BLOCK, "Missing"))
        elif tid in ("expression",):
            self._audit_expression(audit, df, n)
        elif tid == "deg_output":
            sig_col = next((c for c in df.columns if "padj" in c.lower() or "fdr" in c.lower()), None)
            fc_col  = next((c for c in df.columns if "log2" in c.lower() and "fold" in c.lower()), None)
            sig = 0
            if sig_col:
                sig = int((pd.to_numeric(df[sig_col], errors="coerce") < 0.05).sum())
            audit.infos.append(f"{n} genes in DEG output; {sig} significant (padj<0.05).")
        elif tid in ("pathway_output",):
            audit.infos.append(f"{n} pathways in enrichment output.")
        elif tid in ("gwas_mr", "gwas_raw", "mr_results", "eqtl_gene", "eqtl_variant"):
            self._audit_gwas(audit, df, n)
        elif tid in ("crispr_guide", "crispr_essentiality", "crispr_ranked"):
            self._audit_crispr(audit, df, n)
        elif tid == "temporal_fits":
            self._audit_temporal_fits(audit, df, n)
        elif tid == "granger_edges":
            self._audit_granger_edges(audit, df, n)
        elif tid == "prior_network":
            self._audit_prior_network(audit, df, n)
        elif tid == "signature_matrix":
            cell_types = [c for c in df.columns if c.lower() not in ("gene", "gene_id", "geneid")]
            audit.infos.append(f"{n:,} genes x {len(cell_types)} reference cell types")
            if len(cell_types) < 3:
                audit.warnings.append("Fewer than 3 cell types — deconvolution resolution will be limited.")
            if n < 1000:
                audit.warnings.append("Fewer than 1,000 reference genes — consider a more comprehensive signature matrix.")

    def _audit_metadata(self, audit: FileAuditResult, df: pd.DataFrame, n: int):
        outcome_cols = [c for c in df.columns
                        if re.search(r"outcome|phenotype|status|label|group|response|trait|disease", c, re.I)]
        audit.infos.append(f"{n} samples · outcome columns: {', '.join(outcome_cols[:4]) or 'none found'}")
        audit.gates.append(GateResult(
            "G_01", "Phenotype label completeness",
            GateStatus.PASS if outcome_cols else GateStatus.BLOCK,
            "Outcome column(s) present" if outcome_cols
            else "No outcome/phenotype column found — causal analysis requires outcome labels.",
        ))
        if n < GATE_MIN_SAMPLES:
            audit.errors.append(f"G_02 BLOCK: {n} samples < {GATE_MIN_SAMPLES} minimum for causal discovery.")
            audit.gates.append(GateResult("G_02", "Cohort sample size", GateStatus.BLOCK, f"{n} samples"))
        elif n < GATE_WARN_SAMPLES:
            audit.warnings.append(
                f"G_02 WARN: {n} samples — potentially underpowered. Targeted X->Y test (I_02) recommended."
            )
            audit.gates.append(GateResult("G_02", "Cohort sample size", GateStatus.WARN, f"{n} samples"))
        else:
            audit.infos.append(f"G_02 pass: {n} samples >= {GATE_MIN_SAMPLES}.")
            audit.gates.append(GateResult("G_02", "Cohort sample size", GateStatus.PASS, f"{n} samples"))

    def _audit_expression(self, audit: FileAuditResult, df: pd.DataFrame, n: int):
        n_samples = len(df.columns) - 1
        if n < 100:
            audit.warnings.append(f"Only {n} feature rows — may be pre-filtered.")
        else:
            audit.infos.append(f"{n:,} features x {n_samples} samples.")
        if n_samples < GATE_MIN_SAMPLES:
            audit.errors.append(f"G_02 BLOCK: Only {n_samples} samples in expression matrix (<{GATE_MIN_SAMPLES}).")

    def _audit_temporal_fits(self, audit: FileAuditResult, df: pd.DataFrame, n: int):
        good_r2 = int(pd.to_numeric(df.get("r2_impulse", pd.Series(dtype=float)), errors="coerce").ge(0.5).sum())
        sig     = int(pd.to_numeric(df.get("p_adj", pd.Series(dtype=float)), errors="coerce").lt(0.05).sum())
        audit.infos.append(f"{n:,} genes · {sig} significant (p_adj<0.05) · {good_r2} well-fitted (R2>=0.5)")
        if n > 0 and good_r2 / n < 0.15:
            audit.warnings.append(f"Only {round(good_r2/n*100)}% of genes have R2>=0.5.")

    def _audit_granger_edges(self, audit: FileAuditResult, df: pd.DataFrame, n: int):
        q_col = next((c for c in df.columns if c.lower() in ("q_value", "qvalue", "q")), None)
        sig_q = int(pd.to_numeric(df[q_col], errors="coerce").lt(0.05).sum()) if q_col else 0
        audit.infos.append(f"{n:,} Granger edges · {sig_q} q<0.05")

    def _audit_gwas(self, audit: FileAuditResult, df: pd.DataFrame, n: int):
        audit.infos.append(f"{n} rows in genetic evidence file.")
        audit.gates.append(GateResult(
            "G_06", "MR instrument sufficiency",
            GateStatus.PASS if n >= GATE_MIN_MR_INSTRUMENTS else GateStatus.WARN,
            f"{n} rows (need >={GATE_MIN_MR_INSTRUMENTS})",
        ))

    def _audit_crispr(self, audit: FileAuditResult, df: pd.DataFrame, n: int):
        ace_col = next((c for c in df.columns if c.lower() == "ace"), None)
        if ace_col:
            ace     = pd.to_numeric(df[ace_col], errors="coerce")
            drivers = int((ace > 0.1).sum())
            audit.infos.append(f"{n} genes · {drivers} causal drivers (ACE>0.1)")
            audit.gates.append(GateResult(
                "G_07", "IV strength / ACE coverage",
                GateStatus.PASS if drivers > 0 else GateStatus.WARN,
                f"{drivers} driver genes with ACE>0.1",
            ))
        else:
            audit.infos.append(f"{n} rows in CRISPR file.")

    def _audit_prior_network(self, audit: FileAuditResult, df: pd.DataFrame, n: int):
        mech_col = next((c for c in df.columns if "mechanism" in c.lower()), None)
        mechs    = list(df[mech_col].dropna().unique()[:5]) if mech_col else []
        audit.infos.append(f"{n} prior knowledge edges · mechanisms: {', '.join(str(m) for m in mechs)}")


# ── Section 7: EligibilityChecker ────────────────────────────────────────────

class EligibilityChecker:
    """
    Aggregates per-file gate results and fills in cross-file gates
    G_03, G_04, G_05, G_08, G_09.
    """

    def evaluate(self, audits: list) -> list:
        gates:    list[GateResult] = []
        type_map: dict = {a.type_id: a for a in audits}
        seen_ids: set  = set()

        for audit in audits:
            for g in audit.gates:
                if g.gate_id not in seen_ids:
                    gates.append(g)
                    seen_ids.add(g.gate_id)

        if "G_03" not in seen_ids:
            has_expr = any(a.type_id in ("expression", "sc_data") for a in audits)
            gates.append(GateResult(
                "G_03", "Expression matrix availability",
                GateStatus.PASS if has_expr else GateStatus.WARN,
                "Expression matrix detected" if has_expr
                else "No expression matrix — normalization step will need raw counts or pre-normalized data",
            ))
            seen_ids.add("G_03")

        meta = type_map.get("metadata")
        if meta and "G_04" not in seen_ids:
            batch_cols = [c for c in meta.columns if re.search(r"batch|run|plate|lane|center|site", c, re.I)]
            status     = GateStatus.WARN if batch_cols else GateStatus.PASS
            gates.append(GateResult(
                "G_04", "Batch confounding risk", status,
                f"Batch column(s) detected: {', '.join(batch_cols)}" if batch_cols else "No batch columns found",
            ))
            seen_ids.add("G_04")

        if "G_05" not in seen_ids:
            gates.append(GateResult("G_05", "Graph stability (convergence)", GateStatus.SKIP,
                                    "Evaluated during DAG discovery (M12/M13)"))
            seen_ids.add("G_05")

        if "G_08" not in seen_ids:
            has_intervention = any(
                a.type_id in ("crispr_guide", "crispr_essentiality", "crispr_ranked")
                for a in audits
            )
            has_meta = any(a.type_id in ("metadata", "metadata_partial") for a in audits)
            needs_warn = has_intervention and not has_meta
            gates.append(GateResult(
                "G_08", "Intervention context mismatch",
                GateStatus.WARN if needs_warn else GateStatus.PASS,
                "Intervention data present without matched cohort metadata — context penalties may apply"
                if needs_warn else "Context match OK",
            ))
            seen_ids.add("G_08")

        if "G_09" not in seen_ids:
            gates.append(GateResult(
                "G_09", f"Validated graph edge confidence >={GATE_EDGE_CONFIDENCE}",
                GateStatus.SKIP,
                f"Edges below {GATE_EDGE_CONFIDENCE} excluded from validated graph; kept as exploratory",
            ))
            seen_ids.add("G_09")

        return sorted(gates, key=lambda g: g.gate_id)

    @staticmethod
    def is_blocked(gates: list) -> bool:
        return any(g.status == GateStatus.BLOCK for g in gates)


# ── Section 8: WorkflowRouter ─────────────────────────────────────────────────

# Upload flags: which presence flag means a tool already has its file
_UPLOAD_FLAGS: dict[str, list] = {
    "T_04": ["has_sig_matrix", "has_sc"],
    "T_05": ["has_temporal"],
    "T_06": ["has_crispr"],
    "T_07": ["has_prior_network"],
    "T_08": ["has_gwas", "has_eqtl", "has_mr"],
}

# Hard skip reasons (tool truly cannot run without specific file)
_HARD_SKIP: dict[str, str] = {
    "T_00":  "Files uploaded — cohort retrieval not needed",
    "T_04b": "No .h5ad / single-cell file uploaded",
    "T_01":  "DEG output already uploaded — normalization step skipped",
    "T_02":  "No metadata or DEG output already uploaded — DESeq2 skipped",
    "T_03":  "Pathway enrichment output already uploaded — enrichment step skipped",
    "T_05":  "Temporal fits / Granger edges already uploaded — temporal step skipped",
}


class WorkflowRouter:
    """
    Builds an ordered execution plan as a list of step dicts.
    Each step has: id, label, phase, tool, outputs, desc, skip, required,
                   data_source, auto_fetch, auto_fetch_source.
    """

    def build_steps(self, intent: ParsedIntent, audits: list, clarification_context: dict = {}) -> list:
        tid_set = {a.type_id for a in audits}
        p = self._presence_flags(tid_set, audits, clarification_context)

        iid = intent.intent_id.value if hasattr(intent.intent_id, "value") else str(intent.intent_id)

        platform = self._select_platform_tools(iid)
        steps: list[dict] = []

        for tool in platform:
            condition = tool.get("condition")
            runs = condition(p) if callable(condition) else True

            if not runs:
                skip = _HARD_SKIP.get(tool["id"], "Condition not met")
                data_source = None
            else:
                skip = None
                data_source = self._data_source(tool, p)

            steps.append({
                "id":                tool["id"],
                "label":             tool["label"],
                "phase":             tool["phase"],
                "tool":              tool["tool"],
                "outputs":           tool["outputs"],
                "desc":              tool["desc"],
                "skip":              skip,
                "required":          not tool.get("optional", False),
                "data_source":       data_source,
                "auto_fetch":        tool.get("auto_fetch", False) and not self._has_uploaded(tool["id"], p),
                "auto_fetch_source": tool.get("auto_fetch_source", ""),
            })

        for mod in CAUSAL_MODULE_CHAINS.get(iid, []):
            is_optional = mod.get("optional", False)
            skip = "Optional module — skipped (no blocking dependency)" if is_optional else None
            steps.append({
                "id":                mod["id"],
                "label":             mod["label"],
                "phase":             mod["phase"],
                "tool":              mod["algo"],
                "outputs":           mod["outputs"],
                "desc":              mod.get("algo", ""),
                "skip":              skip,
                "required":          not is_optional,
                "data_source":       "computed from pipeline outputs",
                "auto_fetch":        False,
                "auto_fetch_source": "",
            })

        return steps

    def _data_source(self, tool: dict, p: dict) -> str:
        tid = tool["id"]
        if tid == "T_00":
            return f"auto-fetch: {tool.get('auto_fetch_source', 'GEO/SRA')}"
        if tid in ("T_01", "T_02", "T_03"):
            return "uploaded file"
        if tid == "T_04b":
            return "uploaded .h5ad file"
        if tool.get("auto_fetch"):
            if self._has_uploaded(tid, p):
                return "uploaded file"
            return f"auto-fetch: {tool.get('auto_fetch_source', 'public database')}"
        return "uploaded file"

    @staticmethod
    def _has_uploaded(tool_id: str, p: dict) -> bool:
        flags = _UPLOAD_FLAGS.get(tool_id, [])
        return any(p.get(f) for f in flags)

    @staticmethod
    def _presence_flags(tid_set: set, audits: list, ctx: dict) -> dict:
        p = {
            "no_data":         len(audits) == 0,
            "has_sc":          "sc_data" in tid_set,
            "has_expression":  "expression" in tid_set,
            "has_metadata":    any(t in tid_set for t in ("metadata", "phenotype", "metadata_partial")),
            "has_deg_output":  "deg_output" in tid_set,
            "has_pathway_output": "pathway_output" in tid_set,
            "has_sig_matrix":  "signature_matrix" in tid_set,
            "has_temporal":    any(t in tid_set for t in ("temporal_fits", "granger_edges")),
            "has_gwas":        any(t in tid_set for t in ("gwas_mr", "gwas_raw")),
            "has_eqtl":        any(t in tid_set for t in ("eqtl_gene", "eqtl_variant")),
            "has_mr":          "mr_results" in tid_set,
            "has_crispr":      any(t in tid_set for t in ("crispr_guide", "crispr_essentiality", "crispr_ranked")),
            "has_prior_network": "prior_network" in tid_set,
        }
        p["has_genetic"]         = p["has_gwas"] or p["has_eqtl"] or p["has_mr"]
        p["has_expression_or_sc"] = p["has_expression"] or p["has_sc"]

        # Clarification context can promote flags
        data_ans = str(ctx.get("data_availability", "")).lower()
        if "crispr" in data_ans or "perturbation" in data_ans:
            p["has_crispr"] = True
        if "temporal" in data_ans or "longitudinal" in data_ans:
            p["has_temporal"] = True
        if "gwas" in data_ans or "genetic" in data_ans:
            p["has_gwas"] = True
        if "eqtl" in data_ans:
            p["has_eqtl"] = True
        return p

    @staticmethod
    def _select_platform_tools(iid: str) -> list:
        if iid == "I_06":
            return []
        if iid == "I_07":
            return [t for t in PLATFORM_TOOLS if t["id"] in ("T_01", "T_02", "T_03")]
        if iid in ("I_03", "I_05"):
            return []   # reads existing DAG — platform tools skipped
        return PLATFORM_TOOLS


# ── Section 9: SupervisorAgent ────────────────────────────────────────────────

_PHASE_ORDER  = ["acquisition", "bio_prep", "intervention_prep", "dag_build", "causal_core"]
_PHASE_LABELS = {
    "acquisition":       "PHASE 1 — Data Acquisition",
    "bio_prep":          "PHASE 2 — Biological Preparation",
    "intervention_prep": "PHASE 3 — Intervention Preparation",
    "dag_build":         "PHASE 4 — DAG Construction (M12)",
    "causal_core":       "PHASE 5 — Causal Core Analysis",
}
_NEXT_PHASE_LABEL = {
    "acquisition":       "Biological Preparation",
    "bio_prep":          "Intervention Preparation",
    "intervention_prep": "DAG Construction",
    "dag_build":         "Causal Core Analysis",
}


class SupervisorAgent:
    """
    Orchestrates the complete pipeline from file inspection to final result.

    Usage:
        agent = SupervisorAgent()          # reads ANTHROPIC_API_KEY from env
        result = agent.run(query, file_paths)
    """

    LIT_CACHE_TTL_DAYS = LIT_CACHE_TTL_DAYS

    def __init__(self, api_key: Optional[str] = None):
        # Lazy import to avoid circular imports (intelligence.py uses TYPE_CHECKING for agent types)
        from intelligence import (
            ClaudeClient, ClarificationEngine, IntentClassifier,
            LiteraturePipeline, ResultSynthesiser, StepNarrator,
        )
        self._claude        = ClaudeClient(api_key)
        self._inspector     = FileInspector()
        self._eligibility   = EligibilityChecker()
        self._router        = WorkflowRouter()
        self._clarification = ClarificationEngine(self._claude)
        self._intent_clf    = IntentClassifier(self._claude)
        self._lit           = LiteraturePipeline(self._claude)
        self._narrator      = StepNarrator(self._claude)
        self._synthesiser   = ResultSynthesiser(self._claude)

    # ── Pre-run clarification ──────────────────────────────────────────────────

    def pre_clarify(self, query: str, file_paths: Optional[list] = None) -> list:
        file_paths = file_paths or []
        try:
            intent = self._intent_clf.classify(query)
            intent_name = intent.intent_name
        except Exception:
            intent_name = ""
        file_types: list = []
        for fp in file_paths:
            try:
                audit = self._inspector.inspect(fp)
                file_types.append(audit.type_id)
            except Exception:
                pass
        return self._clarification.generate_questions(query, file_types, intent_name)

    def build_clarification(self, query: str, questions: list, answers: list, skipped: bool = False):
        return self._clarification.build_result(query, questions, answers, skipped)

    # ── Main run entry point ───────────────────────────────────────────────────

    def run(
        self,
        query:                str,
        file_paths:           Optional[list] = None,
        clarification_result                 = None,
        skip_literature:      bool           = False,
        verbose:              bool           = True,
        output_dir:           str            = "pipeline_outputs",
    ) -> dict:
        from dataclasses import asdict

        run_id = f"RUN_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        run_output_dir = f"{output_dir}/{run_id}"

        clarif_ctx: dict = {}
        if clarification_result and not clarification_result.skipped:
            query      = clarification_result.enriched_query
            clarif_ctx = clarification_result.enriched_context
        elif clarification_result is None:
            pass  # no clarification

        if verbose:
            self._print_header(run_id, query)

        file_paths = file_paths or []

        # ── 1. File inspection ────────────────────────────────────────────────
        if verbose:
            print("\n[1/7] Inspecting files...")
        audits: list[FileAuditResult] = []
        for fp in file_paths:
            audit = self._inspector.inspect(fp)
            audits.append(audit)
            if verbose:
                icon = "X" if audit.has_errors else ("!" if audit.has_warnings else "OK")
                print(f"      [{icon}] {audit.file_name} -> {audit.type_label} ({audit.platform_tool})")
                for e in audit.errors[:1]:   print(f"         ERR: {e}")
                for w in audit.warnings[:1]: print(f"         WARN: {w}")
                for i in audit.infos[:1]:    print(f"         INFO: {i}")

        # ── 2. Eligibility gates ──────────────────────────────────────────────
        if verbose:
            print("\n[2/7] Running eligibility gates...")
        gates = self._eligibility.evaluate(audits)
        if verbose:
            _icons = {"pass": "OK", "warn": "WARN", "block": "BLOCK", "skip": "SKIP"}
            for g in gates:
                icon = _icons.get(g.status.value if hasattr(g.status, "value") else str(g.status), "?")
                print(f"      [{icon:5s}] {g.gate_id}: {g.name} -- {g.message}")

        if self._eligibility.is_blocked(gates):
            blocking = [g for g in gates if g.status == GateStatus.BLOCK]
            msg = "Pipeline blocked:\n" + "\n".join(
                f"  BLOCK {g.gate_id}: {g.message}" for g in blocking
            )
            if verbose:
                print(f"\n{msg}")
            return {
                "run_id":           run_id,
                "status":           "blocked",
                "gate_results":     [asdict(g) for g in gates],
                "file_audits":      [asdict(a) for a in audits],
                "blocking_message": msg,
            }

        # ── 3. Intent classification ──────────────────────────────────────────
        if verbose:
            print("\n[3/7] Classifying intent...")
        intent = self._intent_clf.classify(query)

        already_clarified = bool(clarification_result)
        if intent.needs_clarification and not already_clarified:
            if verbose:
                print(f"\n  Clarification needed:")
                print(f"     {intent.clarifying_question}")
            return {
                "run_id":              run_id,
                "status":              "needs_clarification",
                "clarifying_question": intent.clarifying_question,
                "intent_partial":      asdict(intent),
                "file_audits":         [asdict(a) for a in audits],
                "gate_results":        [asdict(g) for g in gates],
            }

        if verbose:
            print(f"      {intent.intent_id.value}: {intent.intent_name} ({round(intent.confidence*100)}% confidence)")
            print(f"      Context: {intent.context}")
            print(f"      Module chain: {' -> '.join(intent.module_chain)}")
            if intent.parallel_blocks:
                print(f"      Parallel: {', '.join(intent.parallel_blocks)}")

        # ── 4. Literature pipeline ────────────────────────────────────────────
        lit_brief: Optional[LitBrief] = None
        should_run_lit = (
            not skip_literature
            and intent.needs_literature_first
            and str(intent.intent_id) not in ("I_06", "I_07")
            and intent.intent_id not in (IntentID.I_06, IntentID.I_07)
        )
        if should_run_lit:
            cache_hit, lit_brief = self._load_lit_cache(query, output_dir)
            if cache_hit and verbose:
                print("\n[4/7] Literature pipeline -- loaded from cache")
                print(f"      Papers: {lit_brief.papers_processed} | Claims: {len(lit_brief.claims)}")
            else:
                if verbose:
                    print("\n[4/7] Running literature pipeline (LIT_01-LIT_08)...")
                try:
                    lit_brief = self._lit.run(
                        query       = query,
                        intent_name = intent.intent_name,
                        entities    = intent.entities or {},
                    )
                    self._save_lit_cache(query, lit_brief, output_dir)
                    if verbose:
                        print(f"      Papers found: {lit_brief.papers_found} -> processed: {lit_brief.papers_processed}")
                        print(f"      Claims: {len(lit_brief.claims)} | High-conf edges: {len(lit_brief.high_confidence_edges)}")
                        print(f"      Conflicts: {len(lit_brief.conflicts)} | Rate: {lit_brief.conflict_rate:.1%}")
                        if lit_brief.supervisor_brief:
                            print(f"      Brief: {lit_brief.supervisor_brief[:200]}...")
                except Exception as exc:
                    log.warning("Literature pipeline failed: %s", exc)
                    lit_brief = LitBrief(error=str(exc))
        else:
            if verbose:
                print("\n[4/7] Literature pipeline -- skipped")

        # ── 5. Build execution plan ───────────────────────────────────────────
        if verbose:
            print("\n[5/7] Building execution plan...")
        steps   = self._router.build_steps(intent, audits, clarif_ctx)
        active  = [s for s in steps if not s.get("skip")]
        skipped_steps = [s for s in steps if s.get("skip")]
        if verbose:
            print(f"      {len(active)} steps active · {len(skipped_steps)} skipped")

        # ── 5b. Confirm plan with user ────────────────────────────────────────
        if verbose:
            steps = self._confirm_plan(steps, intent, audits)

        # ── 6. Execute steps phase by phase ──────────────────────────────────
        if verbose:
            print("\n[6/7] Executing pipeline steps (phase-by-phase with checkpoints)...")

        executed_steps: list[dict] = []
        artifact_store: dict       = {}

        phase_groups: dict = {ph: [] for ph in _PHASE_ORDER}
        for step in steps:
            ph = step.get("phase", "causal_core")
            if ph not in phase_groups:
                phase_groups[ph] = []
            phase_groups[ph].append(step)

        from tool_registry import TOOL_REGISTRY

        def _run_one_step(step: dict) -> dict:
            if step.get("skip"):
                if verbose:
                    print(f"      SKIP [{step['id']:8s}] {step['label'][:38]:38s} -- {str(step['skip'])[:40]}")
                return {**step, "status": "skipped"}
            if verbose:
                print(f"      RUN  [{step['id']:8s}] {step['label'][:38]:38s}", end="", flush=True)

            step_id = step["id"]
            fn = TOOL_REGISTRY.get(step_id)
            exec_status = "pending"
            new_arts: dict = {}

            if fn is None:
                log.debug("[%s] not yet implemented (None in registry)", step_id)
                exec_status = "pending"
            else:
                os.makedirs(run_output_dir, exist_ok=True)
                try:
                    result = fn(artifact_store, audits, intent, run_output_dir)
                    new_arts    = result or {}
                    exec_status = "done"
                    log.info("[%s] done -- artifacts: %s", step_id, list(new_arts.keys()))
                except NotImplementedError as exc:
                    log.debug("[%s] pending: %s", step_id, exc)
                    exec_status = "pending"
                except RuntimeError as exc:
                    log.warning("[%s] pipeline error: %s", step_id, exc)
                    exec_status = "error"
                except Exception as exc:
                    log.warning("[%s] unexpected error: %s", step_id, exc, exc_info=True)
                    exec_status = "error"

            artifact_store.update(new_arts)
            narration = self._narrator.narrate(step, intent)

            if verbose:
                tag = {"done": "done", "pending": "pending", "error": "ERROR"}.get(exec_status, exec_status)
                print(f" [{tag}]")
                preview = narration[:110]
                suffix  = "..." if len(narration) > 110 else ""
                print(f"            {preview}{suffix}")

            return {
                **step,
                "status":        exec_status,
                "narration":     narration,
                "artifacts_out": list(new_arts.keys()),
            }

        pipeline_aborted = False
        for phase_name in _PHASE_ORDER:
            p_steps = phase_groups.get(phase_name, [])
            if not p_steps:
                continue

            active_in_phase = [s for s in p_steps if not s.get("skip")]
            if not active_in_phase:
                for s in p_steps:
                    executed_steps.append({**s, "status": "skipped"})
                continue

            if verbose:
                sep2 = "-" * 62
                print(f"\n  {sep2}")
                print(f"  {_PHASE_LABELS.get(phase_name, phase_name.upper())}")
                print(f"  {sep2}")

            phase_results: list[dict] = []
            for step in p_steps:
                sr = _run_one_step(step)
                phase_results.append(sr)
            executed_steps.extend(phase_results)

            # Checkpoint after every phase except causal_core
            if verbose and phase_name != "causal_core":
                keep_going = self._phase_checkpoint(
                    phase_name, phase_results, artifact_store,
                    audits, intent, run_output_dir,
                    _run_one_step, executed_steps,
                )
                if not keep_going:
                    pipeline_aborted = True
                    break

        if pipeline_aborted:
            if verbose:
                print("\n  Pipeline stopped by user.")
            return {
                "run_id":       run_id,
                "status":       "aborted",
                "query":        query,
                "intent":       asdict(intent),
                "gate_results": [asdict(g) for g in gates],
                "file_audits":  [asdict(a) for a in audits],
                "steps":        executed_steps,
            }

        # ── 7. Synthesise result ──────────────────────────────────────────────
        if verbose:
            print("\n[7/7] Synthesising results...")
        result = self._synthesiser.synthesise(
            query          = query,
            intent         = intent,
            steps_run      = executed_steps,
            lit_brief      = lit_brief,
            audits         = audits,
            artifact_store = artifact_store,
        )

        if verbose:
            self._print_result(result)

        pipeline_result = {
            "run_id":       run_id,
            "status":       "complete",
            "query":        query,
            "intent":       asdict(intent),
            "context":      intent.context,
            "gate_results": [asdict(g) for g in gates],
            "file_audits":  [asdict(a) for a in audits],
            "lit_brief":    asdict(lit_brief) if lit_brief else None,
            "steps":        executed_steps,
            "result":       asdict(result),
        }

        self._write_next_steps(pipeline_result, output_dir, verbose)
        return pipeline_result

    # ── Literature cache ───────────────────────────────────────────────────────

    @staticmethod
    def _lit_cache_path(query: str, output_dir: str) -> Path:
        key = hashlib.md5(query.strip().lower().encode()).hexdigest()[:12]
        return Path(output_dir) / "lit_cache" / f"{key}.json"

    def _load_lit_cache(self, query: str, output_dir: str):
        path = self._lit_cache_path(query, output_dir)
        if not path.exists():
            return False, None
        try:
            data      = json.loads(path.read_text(encoding="utf-8"))
            cached_at = datetime.fromisoformat(data.get("cached_at", "2000-01-01"))
            age_days  = (datetime.now() - cached_at).days
            if age_days > self.LIT_CACHE_TTL_DAYS:
                log.info("Lit cache expired (%d days old) — re-running", age_days)
                return False, None
            bd = data["lit_brief"]
            brief = LitBrief(
                inferred_context       = bd.get("inferred_context", ""),
                key_entities           = bd.get("key_entities", []),
                search_queries_used    = bd.get("search_queries_used", []),
                papers_found           = bd.get("papers_found", 0),
                papers_processed       = bd.get("papers_processed", 0),
                claims                 = [LitClaim(**c) for c in bd.get("claims", [])],
                high_confidence_edges  = bd.get("high_confidence_edges", []),
                conflicts              = bd.get("conflicts", []),
                causal_vs_associative  = bd.get("causal_vs_associative", "mixed"),
                prior_evidence_summary = bd.get("prior_evidence_summary", ""),
                recommended_modules    = bd.get("recommended_modules", []),
                data_gaps              = bd.get("data_gaps", []),
                supervisor_brief       = bd.get("supervisor_brief", ""),
                conflict_rate          = bd.get("conflict_rate", 0.0),
            )
            return True, brief
        except Exception as exc:
            log.debug("Lit cache load failed: %s", exc)
            return False, None

    def _save_lit_cache(self, query: str, brief: LitBrief, output_dir: str) -> None:
        from dataclasses import asdict
        path = self._lit_cache_path(query, output_dir)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            path.write_text(
                json.dumps({"cached_at": datetime.now().isoformat(), "lit_brief": asdict(brief)},
                           indent=2, default=str),
                encoding="utf-8",
            )
        except Exception as exc:
            log.debug("Lit cache save failed: %s", exc)

    # ── Plan confirmation ──────────────────────────────────────────────────────

    def _confirm_plan(self, steps: list, intent: ParsedIntent, audits: list) -> list:
        sep = "-" * 70
        print(f"\n{sep}")
        print("  EXECUTION PLAN -- review data sources before running")
        print(f"{sep}")
        print(f"  {'ID':<7} {'DATA SOURCE':<42} TOOL")
        print(f"  {'-'*7} {'-'*42} {'-'*28}")

        for s in steps:
            if s.get("skip"):
                print(f"  {'SKIP':<7} {'-- ' + str(s['skip'])[:40]:<42} {s['label']}")
            else:
                src = s.get("data_source", "")
                print(f"  {s['id']:<7} {str(src):<42} {s['label']}")

        print(f"{sep}")

        auto_fetch_steps = [s for s in steps if not s.get("skip") and s.get("auto_fetch")]
        if auto_fetch_steps:
            print("\n  Auto-fetch steps -- you can supply your own file for any of these.")
            print("  Format:  T_06 C:\\path\\to\\crispr.csv")
            print("  (Press Enter with nothing to accept all auto-fetch sources)\n")
            overrides: dict = {}
            while True:
                try:
                    ans = input("  Override (or Enter to proceed): ").strip()
                except (EOFError, KeyboardInterrupt):
                    break
                if not ans:
                    break
                parts = ans.split(None, 1)
                if len(parts) == 2:
                    step_id, fpath = parts[0].upper(), parts[1].strip()
                    overrides[step_id] = fpath
                    print(f"  Registered: {step_id} -> {fpath}")
                else:
                    print("  Format: <STEP_ID> <file_path>  e.g.  T_06 crispr.csv")

            if overrides:
                inspector = FileInspector()
                for s in steps:
                    if s["id"] in overrides:
                        fpath = overrides[s["id"]]
                        try:
                            audit = inspector.inspect(fpath)
                            audits.append(audit)
                            s["data_source"]   = f"uploaded file: {audit.file_name}"
                            s["auto_fetch"]    = False
                            s["override_file"] = fpath
                            print(f"  {s['id']}: will use {audit.file_name} (detected as {audit.type_label})")
                        except Exception as exc:
                            print(f"  Warning: could not read {fpath}: {exc}")

        try:
            go = input("\n  Proceed with this plan? [Y/n]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            go = "y"

        if go in ("n", "no"):
            print("  Run aborted by user.")
            sys.exit(0)

        print(f"{sep}\n")
        return steps

    # ── Phase checkpoint ───────────────────────────────────────────────────────

    def _phase_checkpoint(
        self,
        phase_name:     str,
        phase_steps:    list,
        artifact_store: dict,
        audits:         list,
        intent:         ParsedIntent,
        run_output_dir: str,
        run_step_fn,
        all_executed:   list,
    ) -> bool:
        sep = "-" * 62
        print(f"\n  {sep}")

        done  = [s for s in phase_steps if s.get("status") == "done"]
        pend  = [s for s in phase_steps if s.get("status") == "pending"]
        errs  = [s for s in phase_steps if s.get("status") == "error"]
        skips = [s for s in phase_steps if s.get("status") == "skipped"]

        next_label = _NEXT_PHASE_LABEL.get(phase_name, "next phase")

        print(f"  PHASE COMPLETE  |  {len(done)} done  |  {len(pend)} pending  "
              f"|  {len(errs)} errors  |  {len(skips)} skipped")

        # Key artifact highlights
        highlights: list[str] = []

        # DEG count
        for key in artifact_store:
            if "deg" in key.lower():
                try:
                    import pandas as _pd
                    df = _pd.read_csv(artifact_store[key])
                    sig_col = next((c for c in df.columns if "padj" in c.lower()), None)
                    fc_col  = next((c for c in df.columns if "log2" in c.lower() and "fold" in c.lower()), None)
                    if sig_col and fc_col:
                        sig = int(((df[sig_col] < 0.05) & (df[fc_col].abs() > 1)).sum())
                        highlights.append(f"DEGs: {sig} significant (padj<0.05, |log2FC|>1)")
                    break
                except Exception:
                    pass

        # Pathway count
        for key in artifact_store:
            if "pathway" in key.lower():
                try:
                    import pandas as _pd
                    df = _pd.read_csv(artifact_store[key])
                    highlights.append(f"Enriched pathways: {len(df)}")
                    break
                except Exception:
                    pass

        # Cell types
        for key in artifact_store:
            if "cell_frac" in key.lower():
                try:
                    import pandas as _pd
                    df = _pd.read_csv(artifact_store[key], index_col=0)
                    highlights.append(f"Cell types deconvolved: {len(df.columns)}")
                    break
                except Exception:
                    pass

        # Network edges
        for key in artifact_store:
            if "edge" in key.lower() or "network" in key.lower():
                try:
                    with open(artifact_store[key]) as f:
                        data = json.load(f)
                    highlights.append(f"Prior network edges: {len(data)}")
                    break
                except Exception:
                    pass

        # GWAS genes
        for key in artifact_store:
            if "gwas" in key.lower():
                try:
                    import pandas as _pd
                    df = _pd.read_csv(artifact_store[key]) if artifact_store[key].endswith(".csv") else None
                    if df is not None:
                        highlights.append(f"GWAS gene scores: {len(df)}")
                    break
                except Exception:
                    pass

        # DAG nodes/edges
        for key in artifact_store:
            if "dag" in key.lower() or "consensus" in key.lower():
                try:
                    with open(artifact_store[key]) as f:
                        dag_data = json.load(f)
                    nodes = len(dag_data.get("nodes", dag_data.get("vertices", [])))
                    edges = len(dag_data.get("edges", dag_data.get("links", [])))
                    highlights.append(f"Consensus DAG: {nodes} nodes, {edges} edges")
                    break
                except Exception:
                    pass

        if highlights:
            for h in highlights:
                print(f"    -> {h}")

        if pend:
            print(f"\n  Note: {len(pend)} step(s) are pending implementation:")
            for s in pend:
                print(f"    - {s['id']}: {s['label']}")

        print(f"\n  Ready to proceed to: {next_label}")
        print(f"  {sep}")
        print("  Options:")
        print("    [Enter]          -- proceed to next phase")
        print("    rerun <STEP_ID>  -- re-run a step (e.g.  rerun T_02)")
        print("    rerun <STEP_ID> <file_path>  -- re-run with a different input file")
        print("    stop             -- stop the pipeline here")
        print()

        while True:
            try:
                ans = input("  Your choice: ").strip()
            except (EOFError, KeyboardInterrupt):
                ans = ""

            if not ans:
                print(f"  Proceeding to {next_label}...\n")
                return True

            if ans.lower() in ("stop", "quit", "exit", "q"):
                print("  Pipeline stopped at user request.")
                return False

            if ans.lower().startswith("rerun"):
                parts = ans.split(None, 2)
                if len(parts) < 2:
                    print("  Usage:  rerun <STEP_ID>   or   rerun <STEP_ID> <file_path>")
                    continue
                target_id  = parts[1].upper()
                new_file   = parts[2].strip() if len(parts) > 2 else None
                target_step = next((s for s in all_executed if s.get("id") == target_id), None)
                if target_step is None:
                    print(f"  Step '{target_id}' not found. Available: "
                          f"{', '.join(s['id'] for s in phase_steps)}")
                    continue
                if new_file:
                    try:
                        new_audit = self._inspector.inspect(new_file)
                        audits.append(new_audit)
                        target_step = dict(target_step)
                        target_step["override_file"] = new_file
                        target_step["data_source"]   = f"uploaded file: {new_audit.file_name}"
                        target_step["auto_fetch"]     = False
                        target_step.pop("skip", None)
                        print(f"  Using {new_audit.file_name} (detected as {new_audit.type_label})")
                    except Exception as exc:
                        print(f"  Could not read file: {exc}")
                        continue
                print(f"  Re-running {target_id}...")
                new_result = run_step_fn(target_step)
                for i, s in enumerate(all_executed):
                    if s.get("id") == target_id:
                        all_executed[i] = new_result
                        break
                for i, s in enumerate(phase_steps):
                    if s.get("id") == target_id:
                        phase_steps[i] = new_result
                        break
                print(f"  Re-run complete. Status: {new_result.get('status', '?')}")
                print()
                continue

            print("  Unrecognised input. Press Enter to proceed, 'stop' to abort, "
                  "or 'rerun <STEP_ID>' to re-run a step.")

    # ── Next-steps file ────────────────────────────────────────────────────────

    def _write_next_steps(self, result: dict, output_dir: str, verbose: bool) -> None:
        intent_d   = result.get("intent", {})
        steps      = result.get("steps", [])
        audits_raw = result.get("file_audits", [])
        gates      = result.get("gate_results", [])
        present    = {a["type_id"] for a in audits_raw}

        DATA_REQUIREMENTS = [
            {
                "type_id":      "expression",
                "label":        "Expression matrix (raw counts or normalised)",
                "example_file": "raw_counts.csv  OR  expression.h5ad",
                "format":       "Rows = genes/features, Columns = samples. CSV/TSV or h5ad.",
                "unlocks":      ["T_01 Data Normalization", "M12 DAGBuilder"],
                "priority":     "CRITICAL",
            },
            {
                "type_id":      "metadata",
                "label":        "Sample metadata / phenotype labels",
                "example_file": "metadata.csv  OR  prep_meta.csv",
                "format":       "Rows = samples, must include outcome/group column.",
                "unlocks":      ["T_02 DESeq2 Differential Expression", "T_03 Pathway Enrichment"],
                "priority":     "CRITICAL",
            },
            {
                "type_id":      "gwas_mr",
                "label":        "GWAS / Mendelian Randomisation results",
                "example_file": "*__genetic-evidence.xlsx  OR  MR_MAIN_RESULTS_ALL_GENES.csv",
                "format":       "Gene-level genetic association scores + MR beta/p-value columns.",
                "unlocks":      ["T_08 GWAS/eQTL Preprocessing", "M12 MR edge-constraints (causal backbone)"],
                "priority":     "HIGH",
            },
            {
                "type_id":      "eqtl_gene",
                "label":        "eQTL instrument data",
                "example_file": "*_GeneLevel_GeneticEvidence.tsv  OR  *_VariantLevel_GeneticEvidence.tsv",
                "format":       "Variant-to-gene mapping with F-stat >= 10.",
                "unlocks":      ["T_08 GWAS/eQTL Preprocessing", "M12 MR edge-constraints"],
                "priority":     "HIGH",
            },
            {
                "type_id":      "crispr_guide",
                "label":        "CRISPR / perturbation data",
                "example_file": "CRISPR_GuideLevel_Avana_SelectedModels_long.csv  OR  GeneEssentiality_ByMedian.csv",
                "format":       "Guide-level or gene-level essentiality/effect scores.",
                "unlocks":      ["T_06 Perturbation Pipeline", "M12 perturbation priors",
                                 "M15 Perturbation evidence stream (weight 0.25)"],
                "priority":     "HIGH",
            },
            {
                "type_id":      "prior_network",
                "label":        "Prior knowledge network (SIGNOR / STRING / KEGG)",
                "example_file": "SIGNOR_Subnetwork_Edges.tsv",
                "format":       "Edge list: source, target, effect, confidence.",
                "unlocks":      ["T_07 Prior Knowledge Network", "M12 edge priors",
                                 "M15 Network evidence stream (weight 0.15)"],
                "priority":     "MEDIUM",
            },
            {
                "type_id":      "temporal_fits",
                "label":        "Temporal / longitudinal gene fits",
                "example_file": "temporal_gene_fits.tsv  OR  granger_edges_raw.csv",
                "format":       "Time-series expression fits per gene across >=4 timepoints.",
                "unlocks":      ["T_05 Temporal Pipeline", "M15 Temporal evidence stream (weight 0.20)"],
                "priority":     "MEDIUM",
            },
            {
                "type_id":      "sc_data",
                "label":        "Single-cell RNA-seq data",
                "example_file": "single_cell.h5ad",
                "format":       "AnnData h5ad with raw counts in .X or .layers['counts'].",
                "unlocks":      ["T_04b Single-Cell Pipeline", "T_04 Cell-Type Deconvolution"],
                "priority":     "OPTIONAL",
            },
            {
                "type_id":      "deg_output",
                "label":        "Pre-computed DEG output",
                "example_file": "*_DEGs_prioritized.csv",
                "format":       "gene, log2FoldChange, padj columns — skips T_01+T_02.",
                "unlocks":      ["Skips T_01 normalization and T_02 DESeq2"],
                "priority":     "OPTIONAL",
            },
            {
                "type_id":      "pathway_output",
                "label":        "Pre-computed pathway enrichment output",
                "example_file": "*_Pathways_Enrichment.csv",
                "format":       "pathway, pvalue, padj, genes columns — skips T_03.",
                "unlocks":      ["Skips T_03 pathway enrichment"],
                "priority":     "OPTIONAL",
            },
            {
                "type_id":      "signature_matrix",
                "label":        "Cell-type signature matrix for deconvolution",
                "example_file": "signature_matrix.tsv",
                "format":       "Genes x cell types — enables CIBERSORT mode in T_04.",
                "unlocks":      ["T_04 CIBERSORT deconvolution mode"],
                "priority":     "OPTIONAL",
            },
        ]

        missing_data = [
            {k: v for k, v in req.items() if k != "type_id"}
            for req in DATA_REQUIREMENTS
            if req["type_id"] not in present
        ]
        present_data = [
            {"label": req["label"], "file": next(
                (a["file_name"] for a in audits_raw if a["type_id"] == req["type_id"]), "?"
            )}
            for req in DATA_REQUIREMENTS
            if req["type_id"] in present
        ]

        IMPLEMENTATION_NOTES = {
            "T_00":  "tool_registry.py -> replace None with cohort retrieval wrapper",
            "T_01":  "tool_registry.py -> replace None with normalize_expression() wrapper",
            "T_02":  "tool_registry.py -> replace None with run_deseq2() wrapper",
            "T_03":  "tool_registry.py -> replace None with pathway_enrichment() wrapper",
            "T_04":  "tool_registry.py -> replace None with deconvolution() wrapper",
            "T_04b": "tool_registry.py -> replace None with sc_pipeline() wrapper",
            "T_05":  "tool_registry.py -> replace None with temporal_pipeline() wrapper",
            "T_06":  "tool_registry.py -> replace None with perturbation_pipeline() wrapper",
            "T_07":  "tool_registry.py -> replace None with prior_knowledge_pipeline() wrapper",
            "T_08":  "tool_registry.py -> replace None with gwas_eqtl_pipeline() wrapper",
            "M12":   "tool_registry.py -> M12: import dag_engine; call dag_engine_run(artifact_store, output_dir)",
            "M12b":  "tool_registry.py -> M12b: same as M12 but filter to group B",
            "M13":   "tool_registry.py -> M13: implement 1000x bootstrap + PPI + pathway coherence",
            "M14":   "tool_registry.py -> M14: import centrality_engine; call centrality_engine_run(dag_path, output_dir)",
            "M15":   "tool_registry.py -> M15: implement 6-stream weighted fusion",
            "M_DC":  "tool_registry.py -> M_DC: implement Backdoor Criterion / Do-Calculus engine",
            "M_IS":  "tool_registry.py -> M_IS: implement do(X=0) graph propagation",
            "M_PI":  "tool_registry.py -> M_PI: integrate DrugBank + ChEMBL + DepMap + LINCS",
            "DELTA": "tool_registry.py -> DELTA: implement cross-group DAG delta engine",
        }

        pending_impl = [
            {
                "module_id": s["id"],
                "label":     s["label"],
                "phase":     s.get("phase", ""),
                "how_to_connect": IMPLEMENTATION_NOTES.get(s["id"], "See tool_registry.py"),
            }
            for s in steps
            if not s.get("skip") and s.get("status") == "pending"
        ]

        warnings = [
            {"gate": g["gate_id"], "name": g["name"], "message": g["message"]}
            for g in gates if g["status"] in ("warn", "block")
        ]

        critical_missing = [m for m in missing_data if m.get("priority") == "CRITICAL"]
        if critical_missing:
            next_action = (
                f"Upload '{critical_missing[0]['example_file']}' — "
                f"this is the most critical missing file and will unlock: "
                f"{', '.join(critical_missing[0]['unlocks'][:3])}"
            )
        elif pending_impl:
            top = pending_impl[0]
            next_action = f"Implement {top['module_id']} ({top['label']}): {top['how_to_connect']}"
        else:
            next_action = "All data present and modules implemented — pipeline is fully operational."

        doc = {
            "_description": (
                "This file is regenerated after every run. "
                "It tells you exactly what data to upload and what code to implement "
                "to advance the pipeline. Address CRITICAL items first."
            ),
            "run_id":                        result["run_id"],
            "query":                         result.get("query", ""),
            "intent":                        f"{intent_d.get('intent_id')} -- {intent_d.get('intent_name')}",
            "immediate_next_action":         next_action,
            "data_present":                  present_data,
            "data_missing":                  missing_data,
            "modules_pending_implementation": pending_impl,
            "eligibility_warnings":          warnings,
            "api_cost_estimate": {
                "model":                  ANTHROPIC_MODEL,
                "cost_per_run_no_lit":    "~$0.10",
                "cost_per_run_with_lit":  "~$0.16",
                "literature_cache_ttl":   f"{self.LIT_CACHE_TTL_DAYS} days",
                "note": "Literature results are cached by query hash. Repeat runs are free.",
            },
        }

        out_path = Path(output_dir) / "next_steps.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(doc, indent=2, default=str), encoding="utf-8")

        if verbose:
            print(f"\n  Next steps written to: {out_path}")
            print(f"  Immediate action: {next_action[:100]}")

    # ── Pretty printing ────────────────────────────────────────────────────────

    @staticmethod
    def _print_header(run_id: str, query: str):
        line = "-" * 68
        print(f"\n{line}")
        print("  MOLECULAR CAUSAL DISCOVERY PLATFORM")
        print(f"  Run: {run_id}")
        print(f"  Query: {query[:80]}")
        print(f"{line}")

    @staticmethod
    def _print_result(result: FinalResult):
        line = "=" * 68
        print(f"\n{line}")
        print("  RESULTS")
        print(f"{line}")
        print(f"\n  {result.headline}\n")
        if result.analyzed_context:
            print(f"  Context: {result.analyzed_context}")
        if result.tier1_candidates:
            print(f"\n  Tier-1 master regulators: {', '.join(result.tier1_candidates)}")
        if result.tier2_candidates:
            print(f"  Tier-2 secondary drivers: {', '.join(result.tier2_candidates)}")
        if result.top_findings:
            print("\n  Key findings:")
            for f in result.top_findings:
                print(f"    -> {f}")
        if result.actionable_targets:
            print("\n  Actionable targets:")
            for t in result.actionable_targets:
                drug = f" [{t.get('existing_drug')}]" if t.get("existing_drug") else ""
                print(f"    {t.get('action','?').upper():12s} {t.get('entity','?')}{drug}")
                print(f"               {t.get('rationale','')[:80]}")
        if result.evidence_quality:
            eq = result.evidence_quality
            print(f"\n  Evidence quality: {eq.get('overall_confidence','?')} -- {eq.get('note','')[:80]}")
            if eq.get("streams_present"):
                print(f"    Present: {', '.join(eq['streams_present'])}")
            if eq.get("streams_missing"):
                print(f"    Missing: {', '.join(eq['streams_missing'])}")
        if result.caveats:
            print("\n  Caveats:")
            for c in result.caveats[:2]:
                print(f"    ! {c[:100]}")
        if result.next_experiments:
            print("\n  Recommended next experiments:")
            for e in result.next_experiments[:3]:
                print(f"    -> {e[:100]}")
        print(f"\n  Modules run: {', '.join(result.modules_run)}")
        print(f"  Artifacts:   {', '.join(result.artifacts_produced[:8])}")
        print(f"\n{line}\n")
