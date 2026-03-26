"""Cell type deconvolution node (xcell, cibersortx, bisque)."""

import asyncio
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from agentic_ai_wf.deconv_pipeline_agent.single_cell_deconv.run_analysis import (
    run_pipeline as run_deconv_pipeline,
)
from agentic_ai_wf.workflow_context import get_workflow_logger as _get_workflow_logger

from ..config_nodes import get_node_dir
from ..wf_common import ensure_global_config
from ..node_logger import create_logged_node
from ..state import TranscriptomeAnalysisState
from ..utils import find_counts_file, find_metadata_file

logger = logging.getLogger(__name__)


@create_logged_node("deconvolution", step_number=6)
async def deconvolution_node(
    state: TranscriptomeAnalysisState,
    config: Optional[Dict[str, Any]] = None,
) -> TranscriptomeAnalysisState:
    """Cell type deconvolution with xcell, cibersortx, or bisque."""
    workflow_logger = (config or {}).get("configurable", {}).get("workflow_logger") if config else None
    if not workflow_logger:
        workflow_logger = _get_workflow_logger()
    loop = asyncio.get_running_loop()
    agent_name = "Deconvolution Agent"

    technique = state.get("deconvolution_technique")
    if technique is not None and technique not in ["xcell", "cibersortx", "bisque"]:
        raise ValueError(
            f"Invalid deconvolution technique: {technique}. Must be 'xcell', 'cibersortx', 'bisque', or None"
        )

    if workflow_logger:
        msg = f"Using technique: {technique}" if technique else "Auto-selecting technique"
        await workflow_logger.info(agent_name=agent_name, message=msg, step="deconvolution")

    transcriptome_dir = Path(state["analysis_transcriptome_dir"])
    bulk_file_path = find_counts_file(transcriptome_dir, file_type="counts")
    if not bulk_file_path:
        raise FileNotFoundError(
            f"No counts/expression file found in {transcriptome_dir}. "
            "Supported formats: CSV, TSV, XLSX, XLS. "
            "Supported naming patterns: *_counts_data.*, *_counts.*, *_count.*, etc."
        )

    bulk_file = str(bulk_file_path)
    logger.info(f"Found counts file: {bulk_file}")
    metadata_file_path = find_metadata_file(transcriptome_dir)
    metadata_file = str(metadata_file_path) if metadata_file_path else None

    output_dir = get_node_dir("deconvolution", state["analysis_id"])

    disease_name = state.get("disease_name")
    if not disease_name or (isinstance(disease_name, str) and not disease_name.strip()):
        if metadata_file and Path(metadata_file).exists():
            try:
                metadata_df = pd.read_csv(metadata_file, nrows=5)
                disease_cols = [
                    col
                    for col in metadata_df.columns
                    if "disease" in col.lower() or "condition" in col.lower()
                ]
                if disease_cols:
                    disease_name = str(metadata_df[disease_cols[0]].iloc[0]).strip()
            except Exception as e:
                logger.warning(f"Failed to extract disease_name from metadata: {e}")
        if not disease_name or not str(disease_name).strip():
            dir_parts = Path(state["analysis_transcriptome_dir"]).parts
            for part in reversed(dir_parts):
                if part and part not in ["shared", "deg_data", "analysis", "data"]:
                    disease_name = part.replace("_", " ").strip()
                    break

    if not disease_name or (isinstance(disease_name, str) and not disease_name.strip()):
        raise ValueError("disease_name is required for deconvolution.")

    disease_name = str(disease_name).strip()

    available_techniques = ["cibersortx", "xcell", "bisque"]
    sc_base_dir_path = Path("./scRNA_reference_data")
    bisque_available = sc_base_dir_path.exists() and any(sc_base_dir_path.rglob("*.h5ad"))
    feasible_techniques = [t for t in available_techniques if t != "bisque" or bisque_available]

    if technique:
        technique_order = (
            [technique] + [t for t in feasible_techniques if t != technique]
            if technique in feasible_techniques
            else feasible_techniques
        )
    else:
        technique_order = feasible_techniques

    attempted_techniques = []
    last_error = None

    for attempt_technique in technique_order:
        if attempt_technique in attempted_techniques:
            continue
        attempted_techniques.append(attempt_technique)
        logger.info(f"Attempting deconvolution with technique: {attempt_technique}")

        try:
            await asyncio.to_thread(
                run_deconv_pipeline,
                bulk_file=bulk_file,
                metadata=metadata_file,
                output_dir=str(output_dir),
                technique=attempt_technique,
                disease_name=disease_name,
                sc_base_dir="./scRNA_reference_data",
                sample_type="blood",
                workflow_logger=workflow_logger,
                event_loop=loop,
            )
            technique = attempt_technique
            break
        except (RuntimeError, ValueError, Exception) as e:
            last_error = e
            error_msg = str(e)
            if "SIGKILL" in error_msg or "OOM" in error_msg or "killed" in error_msg.lower():
                if len(attempted_techniques) >= len(technique_order):
                    raise
                continue
            if len(attempted_techniques) < len(technique_order):
                continue
            raise RuntimeError(
                f"Deconvolution failed with all techniques. Last error ({attempt_technique}): {error_msg}"
            ) from e

    if technique:
        technique_output = output_dir / technique
    else:
        technique_output = None
        for tech in ["xcell", "cibersortx", "bisque"]:
            candidate = output_dir / tech
            if candidate.exists():
                technique_output = candidate
                technique = tech
                break
        if not technique_output:
            technique_output = output_dir

    if not technique_output.exists():
        raise Exception(f"Deconvolution output not found: {technique_output}")

    return {
        "deconvolution_output_dir": str(technique_output),
        "deconvolution_technique": technique,
        "current_step": 6,
    }
