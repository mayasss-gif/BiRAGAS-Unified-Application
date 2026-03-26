"""Pathway enrichment node."""

import asyncio
import logging
from pathlib import Path
from typing import Any, Dict, Optional

from agentic_ai_wf.pathway_agent.agent_runner import run_autonomous_analysis as pathway_agent_runner
from agentic_ai_wf.workflow_context import get_workflow_logger as _get_workflow_logger

from ..node_logger import create_logged_node
from ..state import TranscriptomeAnalysisState

logger = logging.getLogger(__name__)


@create_logged_node("pathway_enrichment", step_number=4)
async def pathway_enrichment_node(
    state: TranscriptomeAnalysisState,
    config: Optional[Dict[str, Any]] = None,
) -> TranscriptomeAnalysisState:
    """Pathway enrichment analysis."""
    workflow_logger = (config or {}).get("configurable", {}).get("workflow_logger") if config else None
    if not workflow_logger:
        workflow_logger = _get_workflow_logger()
    loop = asyncio.get_running_loop()

    prioritized_genes_path_str = state.get("prioritized_genes_path")
    if not prioritized_genes_path_str:
        raise ValueError(
            "prioritized_genes_path is missing from state. Gene prioritization step may have failed."
        )

    deg_base_dir = Path(state["deg_base_dir"])
    prioritized_genes_path = Path(prioritized_genes_path_str)
    if not prioritized_genes_path.exists():
        raise FileNotFoundError(f"Prioritized genes file not found: {prioritized_genes_path}")

    if workflow_logger:
        await workflow_logger.info(
            agent_name="Pathway Enrichment Agent",
            message=f"Loading prioritized genes from {prioritized_genes_path.name} — starting 5-stage pipeline",
            step="pathway_enrichment",
        )

    output_dir = deg_base_dir / "pathway_enrichment"

    result = await pathway_agent_runner(
        user_query=f"Perform enrichment analysis for {state['disease_name']} disease using the prioritized DEGs file",
        deg_file_path=Path(prioritized_genes_path),
        disease_name=state["disease_name"],
        patient_prefix=state["analysis_id"],
        output_dir=Path(output_dir),
        causal=bool(state.get("is_causal", False)),
        workflow_logger=workflow_logger,
        event_loop=loop,
    )

    output_path = result.get("output_file") or result.get("enrichment_output")
    if not output_path:
        raise ValueError("Pathway enrichment produced no output file")

    completed = result.get("completed_stages", [])
    if workflow_logger and completed:
        await workflow_logger.info(
            agent_name="Pathway Enrichment Agent",
            message=f"Pathway pipeline complete: {len(completed)} stages ({', '.join(completed)}) — output saved",
            step="pathway_enrichment",
        )

    return {
        "pathway_consolidation_path": str(Path(output_path).resolve()),
        "analysis_summary": result["analysis_summary"],
        "current_step": 4,
    }
