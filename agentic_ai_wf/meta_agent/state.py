"""Meta agent state schema and node registry."""

from typing import Any, Dict, List, Optional

from ..langgraph_wf import (
    NODE_DEPENDENCIES,
    TranscriptomeAnalysisState,
    cohort_retrieval_node,
    clinical_report_node,
    crispr_analysis_node,
    crispr_screening_analysis_node,
    crispr_targeted_analysis_node,
    deg_analysis_node,
    deconvolution_node,
    drug_discovery_node,
    error_handler_node,
    fastq_analysis_node,
    finalization_node,
    gene_prioritization_node,
    gwas_mr_analysis_node,
    harmonization_node,
    ipaa_analysis_node,
    mdp_analysis_node,
    multiomics_node,
    pathway_enrichment_node,
    perturbation_analysis_node,
    pharma_report_node,
    single_cell_node,
    temporal_analysis_node,
)


class MetaState(TranscriptomeAnalysisState, total=False):
    """Extended state for meta orchestration loop."""

    user_query: str
    plan: List[str]
    plan_history: List[List[str]]
    reflection: Optional[str]
    attempt: int
    done: bool
    memory: List[Dict[str, Any]]
    is_causal: bool


AVAILABLE_NODES: Dict[str, Any] = {
    "cohort_retrieval": cohort_retrieval_node,
    "deg_analysis": deg_analysis_node,
    "gene_prioritization": gene_prioritization_node,
    "pathway_enrichment": pathway_enrichment_node,
    "drug_discovery": drug_discovery_node,
    "deconvolution": deconvolution_node,
    "temporal_analysis": temporal_analysis_node,
    "perturbation_analysis": perturbation_analysis_node,
    "gwas_mr_analysis": gwas_mr_analysis_node,
    "harmonization": harmonization_node,
    "multiomics": multiomics_node,
    "mdp_analysis": mdp_analysis_node,
    "ipaa_analysis": ipaa_analysis_node,
    "single_cell": single_cell_node,
    "fastq_analysis": fastq_analysis_node,
    "crispr_analysis": crispr_analysis_node,
    "crispr_targeted_analysis": crispr_targeted_analysis_node,
    "crispr_screening_analysis": crispr_screening_analysis_node,
    "clinical_report": clinical_report_node,
    "pharma_report": pharma_report_node,
    "finalization": finalization_node,
    "error_handler": error_handler_node,
}

OUTPUT_TO_NODE = {
    out: node
    for node, meta in NODE_DEPENDENCIES.items()
    for out in meta["produces"]
}
