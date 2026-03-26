"""IPAA causality pipeline node."""

import asyncio
import logging
from pathlib import Path

from agentic_ai_wf.ipaa_causality.agent import IPAAAgent
from agentic_ai_wf.ipaa_causality.config.models import IPAAConfig, ItemSpec

from ..config_nodes import get_node_dir
from ..node_logger import create_logged_node
from ..state import TranscriptomeAnalysisState
from ..utils import find_counts_file, find_metadata_file

logger = logging.getLogger(__name__)


@create_logged_node("ipaa_analysis", step_number=20)
async def ipaa_analysis_node(state: TranscriptomeAnalysisState) -> TranscriptomeAnalysisState:
    """IPAA causality pipeline (Engine0–3, pathway summary, HTML reports)."""
    analysis_id = state["analysis_id"]
    disease_name = state.get("disease_name", "unknown")

    base_dir = (
        state.get("cohort_output_dir")
        or state.get("deg_base_dir")
        or state.get("analysis_transcriptome_dir")
    )
    if not base_dir:
        raise ValueError("IPAA requires cohort_output_dir, deg_base_dir, or analysis_transcriptome_dir")

    transcriptome_dir = Path(base_dir)
    if not transcriptome_dir.exists():
        raise FileNotFoundError(f"IPAA input directory not found: {transcriptome_dir}")

    counts_file_path = find_counts_file(transcriptome_dir, file_type="counts")
    metadata_file_path = find_metadata_file(transcriptome_dir)
    counts_file = str(counts_file_path) if counts_file_path else None
    metadata_file = str(metadata_file_path) if metadata_file_path else None
    input_path = counts_file or str(transcriptome_dir)

    out_root = get_node_dir("ipaa", analysis_id)

    config = IPAAConfig(
        outdir=str(out_root),
        items=[ItemSpec(name=disease_name, input=input_path, meta=metadata_file)],
        skip_html_report=False,
    )
    agent = IPAAAgent(config)
    result = await asyncio.to_thread(agent.run_full)

    if result.status == "failed" and result.errors:
        raise Exception(f"IPAA failed: {result.errors}")

    return {"ipaa_output_dir": str(out_root.resolve()), "current_step": 20}
