"""Temporal bulk RNA-seq analysis node."""

import asyncio
import logging
from pathlib import Path
from typing import Any, Dict, Optional

from agentic_ai_wf.temporal_pipeline_agent.main import (
    TemporalAnalysisArgs,
    run_temporal_agent_with_args_sync,
)
from agentic_ai_wf.workflow_context import get_workflow_logger as _get_workflow_logger

from ..config_nodes import get_node_dir
from ..node_logger import create_logged_node
from ..state import TranscriptomeAnalysisState
from ..utils import find_counts_file, find_metadata_file

logger = logging.getLogger(__name__)


@create_logged_node("temporal_analysis", step_number=7)
async def temporal_analysis_node(
    state: TranscriptomeAnalysisState,
    config: Optional[Dict[str, Any]] = None,
) -> TranscriptomeAnalysisState:
    """Temporal bulk RNA-seq analysis."""
    workflow_logger = (config or {}).get("configurable", {}).get("workflow_logger") if config else None
    if not workflow_logger:
        workflow_logger = _get_workflow_logger()
    loop = asyncio.get_running_loop()
    agent_name = "Temporal Analysis Agent"

    transcriptome_dir = Path(state["analysis_transcriptome_dir"])
    counts_file_path = find_counts_file(transcriptome_dir, file_type="counts")
    counts_file = str(counts_file_path) if counts_file_path else None
    metadata_file_path = find_metadata_file(transcriptome_dir)
    metadata_file = str(metadata_file_path) if metadata_file_path else None

    if not counts_file:
        raise FileNotFoundError(
            f"No counts (expression) file found in {transcriptome_dir}. "
            "Supported formats: CSV, TSV, XLSX, XLS."
        )

    output_dir = get_node_dir("temporal", state["analysis_id"])

    treatment_level = state.get("temporal_treatment_level", "")
    genes_list = state.get("temporal_genes_list", "")
    deconv_csv = state.get("temporal_deconv_csv", "")

    temporal_args = TemporalAnalysisArgs(
        output_dir=str(output_dir),
        counts=counts_file,
        metadata=metadata_file if metadata_file else "",
        input_dir=str(transcriptome_dir),
        treatment_level=treatment_level if treatment_level else "",
        genes_list=genes_list if genes_list else None,
        deconv_csv=deconv_csv if deconv_csv else None,
    )

    result = await asyncio.to_thread(
        run_temporal_agent_with_args_sync,
        args=temporal_args,
        max_attempts=3,
        workflow_logger=workflow_logger,
        event_loop=loop,
    )

    if isinstance(result, dict):
        status = result.get("status", "unknown")
        if status == "failed":
            errors = result.get("errors", [])
            error_msg = "\n".join([str(e) for e in errors]) if errors else "Unknown error"
            raise Exception(f"Temporal analysis failed: {error_msg}")

    if not output_dir.exists():
        raise Exception(f"Temporal analysis output directory not found: {output_dir}")

    return {"temporal_analysis_output_dir": str(output_dir), "current_step": 7}
