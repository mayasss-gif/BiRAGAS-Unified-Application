"""Single-cell 10x pipeline node."""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from agentic_ai_wf.single_cell_pipeline_agent.main import (
    SingleCellPipelineArgs,
    run_single_cell_agent_with_args,
)

from ..config_nodes import get_node_dir
from ..node_logger import create_logged_node
from ..state import TranscriptomeAnalysisState

logger = logging.getLogger(__name__)


def _discover_cohort_sample_directories(cohort_output_dir: Path) -> List[Dict[str, Any]]:
    """Discover and rank 10x sample directories from cohort output."""
    candidates = []
    feature_barcode_dirs = list(cohort_output_dir.glob("10_Feature_barcode_matrix_*"))
    if not feature_barcode_dirs:
        return candidates

    matrix_patterns = ["matrix.mtx", "matrix.mtx.gz", "*.matrix.mtx", "*.mtx", "*.mtx.gz"]
    barcodes_patterns = ["barcodes.tsv", "barcodes.tsv.gz", "*barcodes.tsv", "barcode.tsv"]
    features_patterns = ["features.tsv", "features.tsv.gz", "genes.tsv", "*genes.tsv"]

    def _check_files(sample_dir: Path, patterns: List[str]) -> bool:
        for pat in patterns:
            if "*" in pat:
                if list(sample_dir.glob(pat)):
                    return True
            elif (sample_dir / pat).exists():
                return True
        return False

    for parent_dir in feature_barcode_dirs:
        samples_dir = parent_dir / "samples"
        if not samples_dir.exists() or not samples_dir.is_dir():
            continue
        for sample_subdir in samples_dir.iterdir():
            if not sample_subdir.is_dir():
                continue
            has_matrix = _check_files(sample_subdir, matrix_patterns)
            has_barcodes = _check_files(sample_subdir, barcodes_patterns)
            has_features = _check_files(sample_subdir, features_patterns)
            has_all_files = has_matrix and has_barcodes and has_features
            score = 100 if has_all_files else 0
            if has_matrix:
                score += 10
            if has_barcodes:
                score += 10
            if has_features:
                score += 10
            gse_match = None
            if has_all_files:
                try:
                    gse_match = int("".join(filter(str.isdigit, parent_dir.name.split("_")[-1])))
                except Exception:
                    pass
            candidates.append({
                "sample_dir": sample_subdir,
                "parent_dir": parent_dir,
                "score": score,
                "has_all_files": has_all_files,
                "gse_number": gse_match or 0,
            })

    candidates.sort(key=lambda x: (x["score"], x["gse_number"]), reverse=True)
    return candidates


def _find_geo_json_path(sample_dir: Path, parent_dir: Path) -> Optional[str]:
    """Find GEO JSON metadata file."""
    for d in [sample_dir, parent_dir]:
        jsons = list(d.glob("*_metadata.json"))
        if jsons:
            return str(jsons[0])
    return None


@create_logged_node("single_cell", step_number=12)
async def single_cell_node(state: TranscriptomeAnalysisState) -> TranscriptomeAnalysisState:
    """Single-cell 10x pipeline - processes 10x Genomics data."""
    analysis_id = state["analysis_id"]

    if state.get("single_cell_output_dir"):
        output_dir = Path(state["single_cell_output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = get_node_dir("single_cell", analysis_id)

    analysis_transcriptome_dir = state.get("analysis_transcriptome_dir")
    cohort_output_dir = state.get("cohort_output_dir")
    is_cohort_mode = False
    sample_candidates = []

    if analysis_transcriptome_dir and analysis_id in str(analysis_transcriptome_dir):
        single_10x_dir = Path(analysis_transcriptome_dir)
    elif cohort_output_dir:
        cohort_path = Path(cohort_output_dir)
        sample_candidates = _discover_cohort_sample_directories(cohort_path)
        if not sample_candidates:
            raise FileNotFoundError(
                f"No valid sample directories found in {cohort_path}. "
                "Expected: cohort_dir/10_Feature_barcode_matrix_*/samples/*/"
            )
        complete = [c for c in sample_candidates if c["has_all_files"]]
        if not complete:
            raise FileNotFoundError("No sample directories with all required 10x files found.")
        sample_candidates = complete
        is_cohort_mode = True
        single_10x_dir = sample_candidates[0]["sample_dir"]
    else:
        raise ValueError(
            "Provide analysis_transcriptome_dir or cohort_output_dir "
            "(path to 10x data: matrix.mtx, barcodes.tsv, features.tsv)"
        )

    if not single_10x_dir.exists():
        raise FileNotFoundError(f"Single-cell input directory not found: {single_10x_dir}")

    geo_json_path = None
    if is_cohort_mode:
        cand = next((c for c in sample_candidates if c["sample_dir"] == single_10x_dir), None)
        if cand:
            geo_json_path = _find_geo_json_path(cand["sample_dir"], cand["parent_dir"])
    else:
        jsons = list(single_10x_dir.glob("*_metadata.json"))
        if jsons:
            geo_json_path = str(jsons[0])

    max_attempts = len(sample_candidates) if is_cohort_mode else 1

    for attempt_idx in range(max_attempts):
        if is_cohort_mode and attempt_idx > 0:
            single_10x_dir = sample_candidates[attempt_idx]["sample_dir"]
            cand = sample_candidates[attempt_idx]
            geo_json_path = _find_geo_json_path(cand["sample_dir"], cand["parent_dir"])

        single_cell_args = SingleCellPipelineArgs(
            single_10x_dir=str(single_10x_dir),
            output_dir=str(output_dir),
            geo_json_path=geo_json_path,
        )

        result = await run_single_cell_agent_with_args(args=single_cell_args, max_attempts=3)

        if isinstance(result, str):
            try:
                result = json.loads(result)
            except json.JSONDecodeError as e:
                if is_cohort_mode and attempt_idx < max_attempts - 1:
                    continue
                raise Exception(f"Single-cell pipeline returned invalid JSON: {e}")

        if not isinstance(result, dict):
            if is_cohort_mode and attempt_idx < max_attempts - 1:
                continue
            raise Exception(f"Single-cell returned unexpected type: {type(result)}")

        wrapper_status = result.get("status", "unknown")
        if wrapper_status == "ok" and "result" in result:
            result_payload = result.get("result", {})
            status = result_payload.get("status", "unknown")
            result_output_dir = result_payload.get("output_dir")
        elif wrapper_status == "failed":
            status = "failed"
            result_output_dir = None
        else:
            result_payload = result
            status = result.get("status", "unknown")
            result_output_dir = result.get("output_dir")

        if status == "completed":
            if result_output_dir:
                rp = Path(result_output_dir)
                if rp.exists():
                    output_dir = rp
            break

        if status == "failed":
            error_list = result.get("errors", [])
            error_msg = error_list[-1] if error_list else result_payload.get("error", "Pipeline failed")
            if is_cohort_mode and attempt_idx < max_attempts - 1:
                continue
            raise Exception(f"Single-cell pipeline failed: {error_msg}")

        if is_cohort_mode and attempt_idx < max_attempts - 1:
            continue
        raise Exception(f"Single-cell returned unknown status: {status}")

    if not output_dir.exists():
        raise Exception("Single-cell output directory not found")

    return {"single_cell_output_dir": str(output_dir.resolve()), "current_step": 12}
