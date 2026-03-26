"""Cohort retrieval node - GEO/ArrayExpress cohort data."""

import asyncio
import logging
from pathlib import Path

from agentic_ai_wf.cohort_retrieval_agent.cohortagent import run_pipeline
from agentic_ai_wf.cohort_retrieval_agent.geo_agent_pipeline import CohortResult
from agentic_ai_wf.cohort_retrieval_agent.geo_singlecell.runner import run_geo_sc_pipeline

from ..config_nodes import get_node_dir
from ..detection import detect_single_cell_query
from ..node_logger import create_logged_node
from ..state import TranscriptomeAnalysisState
from ..utils import normalize_disease_name

logger = logging.getLogger(__name__)


@create_logged_node("cohort_retrieval", step_number=1)
async def cohort_retrieval_node(state: TranscriptomeAnalysisState) -> TranscriptomeAnalysisState:
    """Retrieve cohort data from GEO/ArrayExpress. Routes to scRNA-seq if query indicates."""
    disease_name = state["disease_name"]
    analysis_id = state["analysis_id"]
    user_query = state.get("user_query")
    if not user_query:
        tissue = state.get("tissue_filter", "any tissue")
        experiment = state.get("experiment_filter", "any")
        user_query = f"Find {experiment} datasets for {disease_name} in {tissue}"

    is_single_cell = await detect_single_cell_query(user_query, disease_name)
    normalized_disease_name = normalize_disease_name(disease_name)

    if is_single_cell:
        logger.info("Single-cell query detected, routing to scRNA-seq pipeline")
        disease_or_id = analysis_id if normalized_disease_name == "unnamed" else normalized_disease_name
        outdir = str(get_node_dir("cohort_single_cell", analysis_id, disease_or_id=disease_or_id)) + "/"
        try:
            sc_results = await asyncio.to_thread(
                run_geo_sc_pipeline,
                disease=disease_name,
                outdir=outdir,
                retmax=2000,
                max_gses=10,
                search_only=False,
                gse=state.get("gse_ids"),
            )
            if not sc_results:
                raise Exception("Single-cell pipeline returned no results")
            output_directory = outdir
            if sc_results and isinstance(sc_results[0], dict):
                first_result = sc_results[0]
                if "output_dir" in first_result:
                    output_directory = first_result["output_dir"]
            cohort_result = CohortResult(
                success=True,
                disease_name=disease_name,
                total_datasets_found=len(sc_results),
                total_datasets_downloaded=len(sc_results),
                output_directory=str(output_directory),
            )
            summary_text = f"Retrieved {len(sc_results)} single-cell RNA-seq datasets for {disease_name}"
        except Exception as e:
            logger.error(f"Single-cell pipeline failed: {e}")
            raise Exception(f"Single-cell cohort retrieval failed: {str(e)}")
    else:
        disease_or_id = analysis_id if normalized_disease_name == "unnamed" else normalized_disease_name
        outdir = str(get_node_dir("cohort_bulk", analysis_id, disease_or_id=disease_or_id)) + "/"
        pipeline_state = await run_pipeline(query=user_query, output_dir=outdir)
        cohort_result = pipeline_state.get("result")
        if not cohort_result:
            raise Exception(
                "Cohort retrieval failed: No result returned from pipeline. "
                f"Summary: {pipeline_state.get('summary_text', 'N/A')}"
            )
        summary_text = pipeline_state.get("summary_text", "")

    if not getattr(cohort_result, "success", False):
        error_msg = getattr(cohort_result, "error", "Unknown error")
        raise Exception(f"Cohort retrieval failed: {error_msg}")

    output_directory = getattr(cohort_result, "output_directory", None)
    if not output_directory:
        raise Exception("Cohort retrieval succeeded but no output_directory was set")

    state["cohort_output_dir"] = str(output_directory)
    state["cohort_summary_text"] = summary_text

    return {
        "cohort_output_dir": str(output_directory),
        "success": True,
        "current_step": 1,
    }
