"""GWAS + Mendelian Randomization analysis node."""

import logging
from pathlib import Path

from agentic_ai_wf.gwas_mr_pipeline_agent.gwas_mr import run_full_pipeline

from ..config_nodes import get_node_dir
from ..node_logger import create_logged_node
from ..state import TranscriptomeAnalysisState
from ...global_config import get_global_config

logger = logging.getLogger(__name__)

DISEASE_TO_BIOSAMPLE = {
    "lupus": "Whole Blood",
    "sle": "Whole Blood",
    "systemic lupus erythematosus": "Whole Blood",
    "breast cancer": "Breast",
    "diabetes": "Pancreas",
    "type 2 diabetes": "Pancreas",
    "t2d": "Pancreas",
    "colorectal cancer": "Colon",
    "alzheimer": "Brain",
    "alzheimer's": "Brain",
    "ad": "Brain",
}


def _infer_biosample(disease_name: str) -> str:
    d = (disease_name or "").strip().lower()
    return DISEASE_TO_BIOSAMPLE.get(d, "Whole Blood")


@create_logged_node("gwas_mr_analysis", step_number=21)
async def gwas_mr_analysis_node(state: TranscriptomeAnalysisState) -> TranscriptomeAnalysisState:
    """GWAS retrieval + Mendelian Randomization pipeline."""
    disease_name = state.get("disease_name", "")
    if not disease_name:
        raise ValueError("gwas_mr_analysis requires disease_name.")

    biosample_type = state.get("biosample_type") or _infer_biosample(disease_name)
    cfg = get_global_config()
    shared_root = Path(cfg.paths.base_project_dir) / cfg.paths.shared_data_dir

    output_dir = get_node_dir("gwas_mr", state["analysis_id"], disease_or_id=disease_name)
    gwas_data_dir = str(shared_root / "gwas_data")

    output_dir.mkdir(parents=True, exist_ok=True)
    Path(gwas_data_dir).mkdir(parents=True, exist_ok=True)

    existing_report = None
    for trait_dir in output_dir.iterdir():
        if trait_dir.is_dir():
            report_path = trait_dir / "MR_PIPELINE_REPORT.html"
            if report_path.exists():
                existing_report = report_path
                break
    if existing_report:
        logger.info(
            f"GWAS-MR results already exist for {disease_name} at {output_dir}, skipping re-run"
        )
        return {
            "gwas_mr_output_dir": str(output_dir),
            "current_step": 21,
        }

    try:
        datasets = run_full_pipeline(
            disease_name=disease_name,
            biosample_type=biosample_type,
            output_dir=str(output_dir),
            gwas_data_dir=gwas_data_dir,
            use_llm=True,
            run_mr_analysis=True,
            skip_preflight=False,
        )
        if not datasets:
            logger.warning("GWAS-MR pipeline retrieved no datasets.")
        return {
            "gwas_mr_output_dir": str(output_dir),
            "current_step": 21,
        }
    except Exception as e:
        logger.exception(f"GWAS-MR analysis failed: {e}")
        raise
