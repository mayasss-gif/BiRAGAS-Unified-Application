"""Clinical and pharma report nodes."""

import logging
import time
from pathlib import Path

from agentic_ai_wf.clinical_report.clinical_report_generator import generate_clinical_report
from agentic_ai_wf.pharma_report.report import pharmareport

from ..config_nodes import get_node_file
from ..wf_common import ensure_global_config
from ..node_logger import create_logged_node
from ..state import TranscriptomeAnalysisState

logger = logging.getLogger(__name__)


@create_logged_node("clinical_report", step_number=10)
async def clinical_report_node(state: TranscriptomeAnalysisState) -> TranscriptomeAnalysisState:
    """Generate clinical report (runs in parallel with pharma report)."""
    gcfg = ensure_global_config(state)
    report_dir = gcfg.get_report_dir("clinical_report")
    output_pdf = report_dir / f"{state['analysis_id']}_report.pdf"
    output_pdf.parent.mkdir(parents=True, exist_ok=True)

    patient_profile = {"name": state["patient_name"], "diagnosis": state["disease_name"]}

    report_path = generate_clinical_report(
        deg_csv=state.get("prioritized_genes_path", ""),
        path_csv=state.get("pathway_consolidation_path", ""),
        drug_csv=state.get("drug_discovery_path", ""),
        output_pdf=str(output_pdf),
        disease_name=state["disease_name"],
        patient_profile=patient_profile,
        patient_prefix=state["analysis_id"],
    )

    return {"clinical_report_path": str(Path(report_path).resolve()), "current_step": 9}


@create_logged_node("pharma_report", step_number=10)
async def pharma_report_node(state: TranscriptomeAnalysisState) -> TranscriptomeAnalysisState:
    """Generate pharma report (runs in parallel with clinical report)."""
    import glob

    output_html = get_node_file(
        "pharma_reports", state["analysis_id"], f"{state['analysis_id']}_pharma_report.html"
    )

    prioritized_genes_dir = Path(state["prioritized_genes_path"]).parent
    stats_files = glob.glob(str(prioritized_genes_dir / f"stats_filtered_{state['analysis_id']}*.txt"))
    if not stats_files:
        raise FileNotFoundError("Stats filtered file not found")
    stats_filtered_path = stats_files[0]

    cohort_dir = Path(state["cohort_output_dir"])
    cohort_json_files = list(cohort_dir.glob("*_cohort_details.json"))
    if not cohort_json_files:
        raise FileNotFoundError("Cohort details JSON not found")
    cohort_json = cohort_json_files[0]

    pharma_input = {
        "Cohort": cohort_json,
        "Harmonization": [stats_filtered_path, state["prioritized_genes_path"]],
        "DEG": [state["prioritized_genes_path"], state["disease_name"]],
        "Gene": state["prioritized_genes_path"],
        "Pathway": state.get("pathway_consolidation_path"),
        "Drug": state.get("drug_discovery_path"),
    }

    report_path = pharmareport(
        path=pharma_input,
        output_html_path=output_html,
        patient_prefix=state["analysis_id"],
    )

    return {"pharma_report_path": str(Path(report_path).resolve()), "current_step": 10}


@create_logged_node("finalization", step_number=11)
async def finalization_node(state: TranscriptomeAnalysisState) -> TranscriptomeAnalysisState:
    """Finalize workflow and validate outputs."""
    clinical_ok = bool(state.get("clinical_report_path"))
    pharma_ok = bool(state.get("pharma_report_path"))

    if not (clinical_ok or pharma_ok):
        raise Exception("No reports were generated")

    execution_time = time.time() - state["start_time"]
    logger.info(f"Workflow completed successfully in {execution_time:.2f}s")

    return {"current_step": 10, "workflow_completed": True}
