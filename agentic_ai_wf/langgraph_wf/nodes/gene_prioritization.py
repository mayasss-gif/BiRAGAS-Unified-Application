"""Gene prioritization node."""

import asyncio
import glob
import logging
from pathlib import Path
from typing import Any, Dict, Optional

from agentic_ai_wf import gene_prioritization as gp
from agentic_ai_wf.workflow_context import get_workflow_logger as _get_workflow_logger

from ..node_logger import create_logged_node
from ..state import TranscriptomeAnalysisState

logger = logging.getLogger(__name__)


@create_logged_node("gene_prioritization", step_number=3)
async def gene_prioritization_node(
    state: TranscriptomeAnalysisState,
    config: Optional[Dict[str, Any]] = None,
) -> TranscriptomeAnalysisState:
    """Gene prioritization from DEG outputs."""
    deg_base_dir = Path(state["deg_base_dir"])
    analysis_id = state["analysis_id"]

    workflow_logger = (config or {}).get("configurable", {}).get("workflow_logger") if config else None
    if not workflow_logger:
        workflow_logger = _get_workflow_logger()
    loop = asyncio.get_running_loop()

    nested_dir = deg_base_dir / analysis_id
    if nested_dir.exists() and nested_dir.is_dir():
        actual_deg_dir = nested_dir
        logger.info(f"Found nested DEG directory: {actual_deg_dir}")
    else:
        actual_deg_dir = deg_base_dir

    output_dir = deg_base_dir / "prioritized_genes"
    output_dir.mkdir(parents=True, exist_ok=True)

    deg_files = list(actual_deg_dir.glob("*.csv"))
    if not deg_files:
        deg_files = list(deg_base_dir.rglob("*_DEGs.csv"))
        if deg_files:
            actual_deg_dir = deg_files[0].parent
            logger.info(f"Found DEG files via recursive search in: {actual_deg_dir}")
        else:
            raise FileNotFoundError(
                f"No DEG CSV files found in {deg_base_dir} or subdirectories. "
                "DEG analysis may have failed or produced no results."
            )

    logger.info(f"Found {len(deg_files)} DEG files in {actual_deg_dir}")

    try:
        prioritized_genes_path = await asyncio.to_thread(
            gp.run_deg_filtering,
            deg_base_dir=actual_deg_dir,
            disease_name=state["disease_name"],
            analysis_id=state["analysis_id"],
            output_dir=output_dir,
            patient_prefix=state["analysis_id"],
            causal=bool(state.get("is_causal", False)),
            workflow_logger=workflow_logger,
            event_loop=loop,
        )
        if prioritized_genes_path is None:
            raise ValueError(
                "run_deg_filtering returned None. Check logs for filtering/merging failures."
            )
        prioritized_path = Path(prioritized_genes_path)
        if not prioritized_path.exists():
            raise FileNotFoundError(f"Prioritized genes file not created: {prioritized_path}")
        logger.info(f"Gene prioritization complete: {prioritized_path}")
        return {"prioritized_genes_path": str(prioritized_path.resolve()), "current_step": 3}
    except Exception as e:
        filtered_files = glob.glob(str(output_dir / "filtered_*.csv"))
        logger.error(
            f"Gene prioritization failed: {e}\n"
            f"DEG base dir: {deg_base_dir}\n"
            f"Actual DEG dir: {actual_deg_dir}\n"
            f"DEG files found: {len(deg_files)}\n"
            f"Filtered files created: {len(filtered_files)}\n"
            f"Output dir: {output_dir}"
        )
        if filtered_files:
            logger.info(f"Filtered files found: {[Path(f).name for f in filtered_files[:5]]}")
        raise
