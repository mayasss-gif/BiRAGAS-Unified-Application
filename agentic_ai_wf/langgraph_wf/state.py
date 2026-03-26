"""State schema and node dependencies for transcriptome analysis workflow."""

from typing import TypedDict, Optional, List, Dict, Any

from agentic_ai_wf.global_config import GlobalAgentConfig


class TranscriptomeAnalysisState(TypedDict, total=False):
    """Complete state schema for transcriptome analysis workflow."""

    analysis_id: str
    disease_name: str
    patient_id: str
    patient_name: str
    user_id: str
    analysis_transcriptome_dir: str
    gse_ids: Optional[List[str]]
    is_causal: Optional[bool]

    user_query: Optional[str]
    tissue_filter: Optional[str]
    experiment_filter: Optional[str]

    workflow_id: str
    run_id: str
    start_time: float
    current_step: int
    total_steps: int

    cohort_output_dir: Optional[str]
    cohort_summary_text: Optional[str]
    deg_analysis_result: Optional[Dict[str, Any]]
    deg_base_dir: Optional[str]
    prioritized_genes_path: Optional[str]
    pathway_consolidation_path: Optional[str]
    drug_discovery_path: Optional[str]
    deconvolution_output_dir: Optional[str]
    deconvolution_technique: Optional[str]
    temporal_analysis_output_dir: Optional[str]
    perturbation_analysis_output_dir: Optional[str]
    multiomics_output_dir: Optional[str]
    single_cell_output_dir: Optional[str]
    ipaa_output_dir: Optional[str]
    fastq_analysis_output_dir: Optional[str]
    crispr_output_dir: Optional[str]
    crispr_targeted_output_dir: Optional[str]
    crispr_screening_output_dir: Optional[str]
    gwas_mr_output_dir: Optional[str]
    biosample_type: Optional[str]

    crispr_targeted_project_id: Optional[str]
    crispr_targeted_target_gene: Optional[str]
    crispr_targeted_protospacer: Optional[str]
    crispr_targeted_region: Optional[str]
    crispr_targeted_reference_seq: Optional[str]
    crispr_targeted_extract_metadata: Optional[bool]
    crispr_targeted_download_fastq: Optional[bool]
    crispr_screening_modes: Optional[List[int]]
    crispr_screening_input_dir: Optional[str]
    crispr_screening_generate_report: Optional[bool]

    clinical_report_path: Optional[str]
    pharma_report_path: Optional[str]

    errors: List[Dict[str, Any]]
    failed_steps: List[str]
    retry_count: int
    step_start_times: Dict[str, float]
    step_durations: Dict[str, float]

    enable_cleanup: bool
    enable_progress_tracking: bool
    max_retries: int
    workflow_failed: bool
    failure_reason: Optional[str]

    global_config: GlobalAgentConfig


NODE_DEPENDENCIES = {
    "cohort_retrieval": {"requires": [], "produces": ["cohort_output_dir"]},
    "deg_analysis": {"requires": [], "produces": ["deg_base_dir"]},
    "gene_prioritization": {"requires": ["deg_base_dir"], "produces": ["prioritized_genes_path"]},
    "pathway_enrichment": {"requires": ["prioritized_genes_path"], "produces": ["pathway_consolidation_path"]},
    "drug_discovery": {
        "requires": ["pathway_consolidation_path", "prioritized_genes_path"],
        "produces": ["drug_discovery_path"],
    },
    "deconvolution": {"requires": [], "produces": ["deconvolution_output_dir"]},
    "temporal_analysis": {"requires": [], "produces": ["temporal_analysis_output_dir"]},
    "perturbation_analysis": {
        "requires": ["prioritized_genes_path", "pathway_consolidation_path"],
        "produces": ["perturbation_analysis_output_dir"],
    },
    "harmonization": {"requires": [], "produces": ["harmonization_output_dir"]},
    "multiomics": {"requires": [], "produces": ["multiomics_output_dir"]},
    "mdp_analysis": {"requires": [], "produces": ["mdp_output_dir"]},
    "ipaa_analysis": {"requires": [], "produces": ["ipaa_output_dir"]},
    "single_cell": {"requires": [], "produces": ["single_cell_output_dir"]},
    "fastq_analysis": {"requires": [], "produces": ["fastq_analysis_output_dir"]},
    "crispr_analysis": {"requires": [], "produces": ["crispr_output_dir"]},
    "crispr_targeted_analysis": {"requires": [], "produces": ["crispr_targeted_output_dir"]},
    "crispr_screening_analysis": {"requires": [], "produces": ["crispr_screening_output_dir"]},
    "gwas_mr_analysis": {"requires": [], "produces": ["gwas_mr_output_dir"]},
    "clinical_report": {
        "requires": ["prioritized_genes_path", "pathway_consolidation_path", "drug_discovery_path"],
        "produces": ["clinical_report_path"],
    },
    "pharma_report": {
        "requires": [
            "cohort_output_dir",
            "prioritized_genes_path",
            "pathway_consolidation_path",
            "drug_discovery_path",
        ],
        "produces": ["pharma_report_path"],
    },
    "finalization": {
        "requires": ["clinical_report_path", "pharma_report_path"],
        "produces": ["workflow_completed"],
    },
}
