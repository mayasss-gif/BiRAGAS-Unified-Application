"""Multi-Disease Pathways (MDP) analysis node."""

import logging
from pathlib import Path
from typing import List, Optional

from agentic_ai_wf.mdp_pipeline_agent.main import (
    MDPPipelineArgs,
    extract_mdp_diseases_and_files,
    run_mdp_agent_with_args,
)

from ..config_nodes import get_node_dir
from ..wf_common import ensure_global_config
from ..node_logger import create_logged_node
from ..state import TranscriptomeAnalysisState

logger = logging.getLogger(__name__)


def _resolve_mdp_item_paths(
    items: List[str],
    analysis_transcriptome_dir: Optional[str] = None,
    known_file_paths: Optional[List[str]] = None,
) -> List[str]:
    """Resolve and verify MDP item input paths."""
    resolved = []
    upload_dir = Path(analysis_transcriptome_dir) if analysis_transcriptome_dir else None
    if upload_dir and not upload_dir.is_absolute():
        upload_dir = Path.cwd() / upload_dir
    path_by_name = {Path(p).name: p for p in (known_file_paths or [])}

    for item in items:
        item = (item or "").strip()
        if not item or "=" not in item or "input" not in item:
            resolved.append(item)
            continue
        parts = {}
        for kv in item.split(","):
            if "=" in kv:
                k, v = kv.split("=", 1)
                parts[k.strip()] = v.strip()
        disease = parts.get("name", "").strip()
        inp = parts.get("input", "").strip()
        tissue = parts.get("tissue", "").strip()
        if not disease or not inp:
            resolved.append(item)
            continue
        p = Path(inp).expanduser().resolve()
        if not p.exists():
            fn = Path(inp).name
            if fn in path_by_name:
                p = Path(path_by_name[fn]).resolve()
            elif upload_dir and upload_dir.exists():
                fallback = upload_dir / fn
                if fallback.exists():
                    p = fallback.resolve()
        if p.exists():
            new_item = f"name={disease},input={str(p)}"
            if tissue:
                new_item += f",tissue={tissue}"
            resolved.append(new_item)
        else:
            resolved.append(f"name={disease}")
    return resolved


@create_logged_node("mdp_analysis", step_number=11)
async def mdp_analysis_node(state: TranscriptomeAnalysisState) -> TranscriptomeAnalysisState:
    """MDP analysis - pathway analysis pipeline for single or multi-disease."""
    ensure_global_config(state)
    analysis_id = state["analysis_id"]
    disease_name = state.get("disease_name", "unknown_disease")
    user_query = state.get("user_query", "")

    output_dir = get_node_dir("mdp", analysis_id)
    mdp_items = state.get("mdp_items")
    mdp_uploaded_files = state.get("mdp_uploaded_files")

    if not mdp_items and user_query:
        analysis_transcriptome_dir = state.get("analysis_transcriptome_dir")
        uploaded_file_paths = []
        if analysis_transcriptome_dir:
            upload_dir = Path(analysis_transcriptome_dir)
            if upload_dir.exists() and upload_dir.is_dir():
                for ext in ["*.csv", "*.tsv", "*.txt", "*.json"]:
                    uploaded_file_paths.extend(upload_dir.glob(ext))
                    uploaded_file_paths.extend(upload_dir.glob(ext.upper()))
        try:
            extraction_result = await extract_mdp_diseases_and_files(
                user_query=user_query,
                available_files=[str(f) for f in uploaded_file_paths],
            )
            extracted_items = extraction_result.get("items", [])
            extracted_diseases = extraction_result.get("diseases", [])
            if extracted_items:
                mdp_items = _resolve_mdp_item_paths(
                    extracted_items,
                    analysis_transcriptome_dir=state.get("analysis_transcriptome_dir"),
                    known_file_paths=[str(f) for f in uploaded_file_paths],
                )
            elif extracted_diseases and uploaded_file_paths:
                mdp_items = [f"name={extracted_diseases[0]},input={uploaded_file_paths[0]}"]
        except Exception as e:
            logger.exception(f"MDP agent extraction failed: {e}")
            mdp_items = None

    if mdp_items:
        upload_paths = []
        if state.get("analysis_transcriptome_dir"):
            ud = Path(state["analysis_transcriptome_dir"])
            if ud.exists() and ud.is_dir():
                for ext in ["*.csv", "*.tsv", "*.txt", "*.json"]:
                    upload_paths.extend(ud.glob(ext))
                    upload_paths.extend(ud.glob(ext.upper()))
        mdp_items = _resolve_mdp_item_paths(
            mdp_items,
            analysis_transcriptome_dir=state.get("analysis_transcriptome_dir"),
            known_file_paths=[str(p) for p in upload_paths],
        )
        input_path = None
    else:
        input_path = None
        analysis_transcriptome_dir = state.get("analysis_transcriptome_dir")
        cohort_output_dir = state.get("cohort_output_dir")
        deg_analysis_result = state.get("deg_analysis_result")

        if state.get("mdp_input_path"):
            input_path = Path(state["mdp_input_path"])
        elif analysis_transcriptome_dir:
            upload_dir = Path(analysis_transcriptome_dir)
            if upload_dir.exists():
                data_files = []
                for ext in ["*.csv", "*.tsv", "*.txt", "*.json"]:
                    data_files.extend(upload_dir.glob(ext))
                    data_files.extend(upload_dir.glob(ext.upper()))
                input_path = data_files[0] if data_files else upload_dir
        if not input_path and cohort_output_dir:
            input_path = Path(cohort_output_dir)
        if not input_path and deg_analysis_result and isinstance(deg_analysis_result, dict):
            deg_output = deg_analysis_result.get("output_dir")
            if deg_output:
                input_path = Path(deg_output)

        if not input_path or not input_path.exists():
            raise FileNotFoundError(
                "MDP input path not found. Provide via mdp_input_path, "
                "analysis_transcriptome_dir, cohort_output_dir, or mdp_items."
            )

    mode = "auto" if mdp_items else state.get("mdp_mode", "counts")
    if mode not in ["counts", "degs", "gl", "gc"]:
        mode = "counts"

    mdp_args = MDPPipelineArgs(
        mode=mode,
        input_path=str(input_path) if input_path and not mdp_items else None,
        output_dir=str(output_dir),
        disease_name=disease_name,
        items=mdp_items,
        uploaded_files=mdp_uploaded_files,
        cohort=state.get("mdp_cohort"),
        tissue=state.get("mdp_tissue", "") or "",
        degs_extra=state.get("mdp_degs_extra", []) or [],
        run_enzymes=state.get("mdp_run_enzymes", True),
        run_analyses=state.get("mdp_run_analyses", True),
        analysis=state.get("mdp_analysis", ["client_dashboard", "category_landscape", "enzyme_signaling", "pathway_entity", "tf_activity"]),
        sig=state.get("mdp_sig", 0.10),
        cap=state.get("mdp_cap", 300),
        pathways=state.get("mdp_pathways", []) or [],
        direction_mode=state.get("mdp_direction_mode", "both"),
        pc_out=state.get("mdp_pc_out", "PC_out"),
        skip_report=state.get("mdp_skip_report"),
        report_no_llm=state.get("mdp_report_no_llm"),
        report_q_cutoff=state.get("mdp_report_q_cutoff"),
    )

    result = await run_mdp_agent_with_args(args=mdp_args, max_attempts=3)

    if isinstance(result, dict):
        if result.get("status") == "failed":
            errors = result.get("errors", [])
            raise Exception(f"MDP failed: {'; '.join(str(e) for e in errors)}")
        result_payload = result.get("result", {})
        if isinstance(result_payload, dict) and result_payload.get("output_dir"):
            od = Path(result_payload["output_dir"])
            if od.exists():
                output_dir = od

    if not output_dir.exists():
        raise Exception("MDP output directory not found")

    return {"mdp_output_dir": str(output_dir), "current_step": 11}
