"""CRISPR Perturb-seq and screening nodes."""

import asyncio
import logging
from pathlib import Path

from agentic_ai_wf.crispr_pipeline_agent.crispr.run_pipeline import (
    get_available_samples as crispr_get_available_samples,
    run_pipeline as run_crispr_pipeline,
)
from agentic_ai_wf.crispr_pipeline_agent.screening_crispr import run_screening
from agentic_ai_wf.crispr_pipeline_agent.targeted import run_targeted_pipeline

from ..config_nodes import get_node_dir
from ..node_logger import create_logged_node
from ..state import TranscriptomeAnalysisState

logger = logging.getLogger(__name__)


@create_logged_node("crispr_analysis", step_number=11)
async def crispr_analysis_node(state: TranscriptomeAnalysisState) -> TranscriptomeAnalysisState:
    """CRISPR Perturb-seq pipeline - GSE RAW 10x-like data."""
    analysis_id = state["analysis_id"]
    disease_name = state.get("disease_name", "unknown_disease")

    output_dir = get_node_dir("crispr", analysis_id)

    analysis_transcriptome_dir = state.get("analysis_transcriptome_dir")
    cohort_output_dir = state.get("cohort_output_dir")
    gse_dir = None

    if analysis_transcriptome_dir:
        transcriptome_path = Path(analysis_transcriptome_dir)
        if transcriptome_path.exists():
            gse_dir = transcriptome_path
    if not gse_dir and cohort_output_dir:
        cohort_path = Path(cohort_output_dir)
        if cohort_path.exists():
            gse_dirs = [d for d in cohort_path.iterdir() if d.is_dir() and d.name.upper().startswith("GSE")]
            if gse_dirs:
                gse_dir = gse_dirs[0]

    if not gse_dir or not gse_dir.exists():
        default_gse = Path("agentic_ai_wf/crispr_pipeline_agent/input_data/GSE90546_RAW")
        if default_gse.exists():
            gse_dir = default_gse
        else:
            raise FileNotFoundError(
                "No CRISPR input directory found. Provide GSE RAW directory with "
                "GSM*_barcodes.tsv, GSM*_matrix.mtx, GSM*_genes.tsv."
            )

    samples = state.get("crispr_samples")
    if not samples:
        try:
            samples = crispr_get_available_samples(gse_dir)
        except Exception as e:
            logger.warning(f"Sample discovery failed: {e}")
            samples = []
    if not samples:
        raise ValueError(f"No CRISPR samples found in {gse_dir}")
    if not isinstance(samples, list):
        samples = [samples] if isinstance(samples, str) else list(samples)

    generate_report = state.get("crispr_generate_report", True)

    await asyncio.to_thread(
        run_crispr_pipeline,
        input_gse_dirs=gse_dir,
        samples=samples,
        output_dir=output_dir,
        generate_report=generate_report,
    )

    result_dir = output_dir
    for sub in output_dir.iterdir():
        if sub.is_dir() and not sub.name.startswith("."):
            result_dir = sub
            break

    return {"crispr_output_dir": str(result_dir.resolve()), "current_step": 11}


@create_logged_node("crispr_targeted_analysis", step_number=12)
async def crispr_targeted_analysis_node(state: TranscriptomeAnalysisState) -> TranscriptomeAnalysisState:
    """Targeted CRISPR-seq: SRA project + target gene/region + protospacer."""
    analysis_id = state["analysis_id"]
    base_dir = get_node_dir("crispr_targeted", analysis_id)
    input_dir = base_dir / "input"
    output_dir = base_dir / "output"
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    project_id = state.get("crispr_targeted_project_id") or ""
    target_gene = state.get("crispr_targeted_target_gene") or ""
    protospacer = state.get("crispr_targeted_protospacer") or "GGTGGATCCTATTCTAAACG"
    region = state.get("crispr_targeted_region") or ""
    reference_seq = state.get("crispr_targeted_reference_seq") or ""
    extract_metadata = state.get("crispr_targeted_extract_metadata", True)
    download_fastq = state.get("crispr_targeted_download_fastq", True)

    if not project_id:
        raise ValueError("targeted CRISPR requires crispr_targeted_project_id")
    mode = "region" if region else ("reference_seq" if reference_seq else "gene")
    if mode == "gene" and not target_gene:
        raise ValueError("targeted CRISPR in gene mode requires crispr_targeted_target_gene")
    if mode == "region" and not region:
        raise ValueError("targeted CRISPR in region mode requires crispr_targeted_region")
    if mode == "reference_seq" and not reference_seq:
        raise ValueError("targeted CRISPR in reference_seq mode requires crispr_targeted_reference_seq")

    kwargs = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "project_id": project_id,
        "protospacer": protospacer,
        "extract_metadata": extract_metadata,
        "download_fastq": download_fastq,
    }
    if target_gene:
        kwargs["target_gene"] = target_gene
    if region:
        kwargs["region"] = region
    if reference_seq:
        kwargs["reference_seq"] = reference_seq

    await asyncio.to_thread(run_targeted_pipeline, **kwargs)

    return {"crispr_targeted_output_dir": str(output_dir.resolve()), "current_step": 12}


@create_logged_node("crispr_screening_analysis", step_number=13)
async def crispr_screening_analysis_node(state: TranscriptomeAnalysisState) -> TranscriptomeAnalysisState:
    """CRISPR genetic screening: MAGeCK RRA/MLE, BAGEL2."""
    analysis_id = state["analysis_id"]
    base_dir = get_node_dir("crispr_screening", analysis_id)
    output_dir = base_dir
    base_dir.mkdir(parents=True, exist_ok=True)

    modes = state.get("crispr_screening_modes") or [3]
    input_dir_str = state.get("crispr_screening_input_dir") or ""
    generate_report = state.get("crispr_screening_generate_report", True)

    default_input = Path("./agentic_ai_wf/crispr_pipeline_agent/input_data/screening_data")
    if not default_input.is_absolute():
        default_input = Path.cwd() / default_input
    input_dir = Path(input_dir_str).resolve() if input_dir_str and Path(input_dir_str).exists() else default_input

    if not input_dir.exists() or not input_dir.is_dir():
        raise ValueError(f"CRISPR screening input directory not found: {input_dir}")

    result = await asyncio.to_thread(
        run_screening,
        input_dir=str(input_dir),
        output_dir=str(output_dir),
        modes=modes,
        generate_report=generate_report,
    )

    if not result.success:
        failed = [r for r in result.mode_results if not r.success]
        err_msg = "; ".join(f"Mode {r.mode}: exit {r.return_code}" for r in failed)
        raise RuntimeError(f"CRISPR screening failed: {err_msg}")

    return {"crispr_screening_output_dir": str(output_dir.resolve()), "current_step": 13}
