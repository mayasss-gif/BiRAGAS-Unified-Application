"""Dataset harmonization node."""

import logging
from pathlib import Path

from agentic_ai_wf.harmonization_pipeline_agent.main import (
    HarmonizationArgs,
    run_harmonization_agent_with_args,
)

from ..config_nodes import get_node_dir
from ..node_logger import create_logged_node
from ..state import TranscriptomeAnalysisState
from ..utils import find_counts_file, find_metadata_file

logger = logging.getLogger(__name__)


@create_logged_node("harmonization", step_number=9)
async def harmonization_node(state: TranscriptomeAnalysisState) -> TranscriptomeAnalysisState:
    """Dataset harmonization - normalizes and harmonizes RNA-seq datasets."""
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

    output_dir = get_node_dir("harmonization", state["analysis_id"])

    harmonization_mode = state.get("harmonization_mode", "single")
    combine = state.get("harmonization_combine", True)
    out_mode = state.get("harmonization_out_mode", "default")
    create_zip = state.get("harmonization_create_zip", False)

    if harmonization_mode == "local":
        harmonization_args = HarmonizationArgs(
            mode="local",
            data_root=str(transcriptome_dir),
            output_dir=str(output_dir),
            combine=combine,
        )
    else:
        harmonization_args = HarmonizationArgs(
            mode="single",
            counts_path=counts_file,
            meta_path=metadata_file if metadata_file else "",
            output_dir=str(output_dir),
            out_mode=out_mode,
            create_zip=create_zip,
        )

    result = await run_harmonization_agent_with_args(args=harmonization_args, max_attempts=3)

    if isinstance(result, dict):
        status = result.get("status", "unknown")
        if status == "failed":
            errors = result.get("errors", [])
            error_msg = "\n".join([str(e) for e in errors]) if errors else "Unknown error"
            raise Exception(f"Harmonization failed: {error_msg}")
        result_payload = result.get("result", {})
        if isinstance(result_payload, dict):
            outputs = result_payload.get("outputs", [])
            if outputs and outputs[0]:
                actual_output_dir = Path(outputs[0])
                if actual_output_dir.exists():
                    output_dir = actual_output_dir

    if not output_dir.exists():
        raise Exception("Harmonization output directory not found")

    return {"harmonization_output_dir": str(output_dir), "current_step": 8}
