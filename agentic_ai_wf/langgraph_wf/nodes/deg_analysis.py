"""DEG analysis node - differential expression."""

import asyncio
import logging
from pathlib import Path
from typing import Any, Dict, Optional

from agentic_ai_wf.deg_pipeline_agent import DEGPipelineAgent, DEGPipelineConfig

from ..wf_common import ensure_global_config, sanitize_for_state
from ..node_logger import create_logged_node
from ..state import TranscriptomeAnalysisState
from agentic_ai_wf.workflow_context import get_workflow_logger as _get_workflow_logger

logger = logging.getLogger(__name__)


@create_logged_node("deg_analysis", step_number=2)
async def deg_analysis_node(
    state: TranscriptomeAnalysisState,
    config: Optional[Dict[str, Any]] = None,
) -> TranscriptomeAnalysisState:
    """DEG analysis. Priority: cohort_output_dir > analysis_transcriptome_dir."""
    ensure_global_config(state)
    disease_name = state["disease_name"]
    cohort_output_dir = state.get("cohort_output_dir")
    analysis_transcriptome_dir = state.get("analysis_transcriptome_dir")

    if cohort_output_dir:
        geo_dir = str(cohort_output_dir)
        analysis_transcriptome_dir_for_deg = None
        logger.info(f"DEG Analysis: Using cohort retrieval output directory: {cohort_output_dir}")
    elif analysis_transcriptome_dir:
        geo_dir = None
        analysis_transcriptome_dir_for_deg = analysis_transcriptome_dir
        logger.info(f"DEG Analysis: Using analysis transcriptome directory: {analysis_transcriptome_dir}")
    else:
        raise Exception(
            "DEG Analysis requires either cohort_output_dir (from cohort retrieval) "
            "or analysis_transcriptome_dir (uploaded files)"
        )

    deg_config = DEGPipelineConfig(
        geo_dir=geo_dir,
        analysis_transcriptome_dir=analysis_transcriptome_dir_for_deg,
        disease_name=disease_name,
        analysis_id=state["analysis_id"],
        max_retries=ensure_global_config(state).defaults["max_retries"],
        enable_auto_fix=True,
        user_query=state.get("user_query"),
    )

    deg_agent = DEGPipelineAgent(deg_config)
    workflow_logger = (config or {}).get("configurable", {}).get("workflow_logger") if config else None
    if not workflow_logger:
        workflow_logger = _get_workflow_logger()
    loop = asyncio.get_running_loop()

    result = await asyncio.to_thread(
        deg_agent.run_pipeline,
        geo_dir=geo_dir,
        analysis_transcriptome_dir=analysis_transcriptome_dir_for_deg,
        disease_name=disease_name,
        workflow_logger=workflow_logger,
        event_loop=loop,
    )

    if result["status"] != "success":
        raise Exception(f"DEG pipeline failed: {result.get('error')}")

    sanitized_result = sanitize_for_state(result)

    return {
        "deg_analysis_result": sanitized_result,
        "deg_base_dir": str(deg_config.get_disease_output_dir()),
        "current_step": 2,
    }
