"""FASTQ pipeline analysis node."""

import asyncio
import logging
from pathlib import Path

from agentic_ai_wf.fastq_pipeline_agent.fastq.main_module import run_pipeline as run_fastq_pipeline

from ..config_nodes import get_node_base_dir, get_node_dir
from ..wf_common import ensure_global_config
from ..node_logger import create_logged_node
from ..state import TranscriptomeAnalysisState

logger = logging.getLogger(__name__)


@create_logged_node("fastq_analysis", step_number=10)
async def fastq_analysis_node(state: TranscriptomeAnalysisState) -> TranscriptomeAnalysisState:
    """FASTQ pipeline - processes FASTQ files."""
    ensure_global_config(state)
    analysis_id = state["analysis_id"]
    disease_name = state.get("disease_name", "unknown_disease")

    output_dir = get_node_dir("fastq", analysis_id)

    analysis_transcriptome_dir = state.get("analysis_transcriptome_dir")
    input_path = None

    if analysis_transcriptome_dir:
        transcriptome_path = Path(analysis_transcriptome_dir)
        if transcriptome_path.exists():
            fastq_files = list(transcriptome_path.glob("*.fastq*")) + list(transcriptome_path.glob("*.fq*"))
            fastq_dirs = [d for d in transcriptome_path.iterdir() if d.is_dir() and any(d.glob("*.fastq*"))]
            if fastq_files or fastq_dirs:
                input_path = transcriptome_path

    if not input_path:
        fastq_base = get_node_base_dir("cohort_fastq", create=False)
        if fastq_base.exists():
            srp_dirs = [d for d in fastq_base.iterdir() if d.is_dir() and d.name.startswith("SRP")]
            if srp_dirs:
                input_path = srp_dirs[0]

    if not input_path or not input_path.exists():
        raise FileNotFoundError(
            f"No FASTQ input directory found. Provide FASTQ files in {analysis_transcriptome_dir} "
            "or ensure fastq_data exists in cohort_data."
        )

    combine_after = state.get("fastq_combine_after", True)
    model = state.get("fastq_model", "gpt-4o")
    max_turns = state.get("fastq_max_turns", 120)

    result_output_dir = await asyncio.to_thread(
        run_fastq_pipeline,
        input_path=str(input_path),
        results_root=str(output_dir),
        disease_name=disease_name,
        combine_after=combine_after,
        model=model,
        max_turns=max_turns,
    )

    if isinstance(result_output_dir, Path):
        output_dir = result_output_dir
    elif isinstance(result_output_dir, str):
        output_dir = Path(result_output_dir)
    else:
        output_dir = output_dir / f"{disease_name.replace(' ', '_')}_fastq"

    if not output_dir.exists():
        raise Exception("FASTQ pipeline output directory not found")

    return {"fastq_analysis_output_dir": str(output_dir.resolve()), "current_step": 10}
