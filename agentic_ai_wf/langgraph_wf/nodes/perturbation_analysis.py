"""Perturbation analysis node (DEPMAP + L1000)."""

import logging
import time
from pathlib import Path

import psutil

from agentic_ai_wf.perturbation_pipeline_agent.perturbation.run_full_pipeline_optimized import (
    run_full_pipeline as perturbation_pipeline_optimized,
)

from ..config_nodes import get_node_dir
from ..node_logger import create_logged_node
from ..state import TranscriptomeAnalysisState

logger = logging.getLogger(__name__)


def _cleanup_chrome_processes():
    """Kill orphaned Chrome/Kaleido processes."""
    try:
        current_process = psutil.Process()
        for child in current_process.children(recursive=True):
            try:
                cmdline = " ".join(child.cmdline())
                if any(term in cmdline.lower() for term in ["chrome", "chromium", "kaleido"]):
                    logger.warning(f"Killing orphaned process: PID={child.pid}")
                    child.kill()
                    child.wait(timeout=3)
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.TimeoutExpired):
                pass
    except Exception as e:
        logger.error(f"Failed to cleanup Chrome processes: {e}")


def _log_resource_usage():
    try:
        process = psutil.Process()
        logger.info(f"Resource usage: FDs={process.num_fds()}, Memory={process.memory_info().rss / 1024 / 1024:.1f}MB")
    except Exception:
        pass


@create_logged_node("perturbation_analysis", step_number=8)
async def perturbation_analysis_node(state: TranscriptomeAnalysisState) -> TranscriptomeAnalysisState:
    """Perturbation analysis with optimized direct pipeline execution."""
    prioritized_genes_path = state.get("prioritized_genes_path")
    pathway_consolidation_path = state.get("pathway_consolidation_path")
    disease_name = state.get("disease_name", "")

    if not prioritized_genes_path:
        raise ValueError("perturbation_analysis requires prioritized_genes_path.")
    if not pathway_consolidation_path:
        raise ValueError("perturbation_analysis requires pathway_consolidation_path.")

    output_dir = get_node_dir("perturbation", state["analysis_id"])
    deg_path = Path(prioritized_genes_path)
    pathway_path = Path(pathway_consolidation_path)

    if not deg_path.exists():
        raise FileNotFoundError(f"DEGs prioritized file not found: {deg_path}")
    if not pathway_path.exists():
        raise FileNotFoundError(f"Pathways consolidated file not found: {pathway_path}")

    try:
        _log_resource_usage()
        start_time = time.time()
        direct_result = perturbation_pipeline_optimized(
            raw_deg_path=deg_path,
            pathway_path=pathway_path,
            output_dir=output_dir,
            disease=disease_name,
            dep_map_addons={"mode_model": None, "genes_selection": "all", "top_up": None, "top_down": None},
            l1000_addons={"tissue": None, "drug": None, "time_points": None, "cell_lines": None},
            max_sigs=400000,
            parallel=True,
        )
        _log_resource_usage()

        status = direct_result.get("status", "unknown")
        if status == "error":
            error_msg = direct_result.get("message", "Unknown error")
            raise Exception(f"Perturbation analysis failed: {error_msg}")

        if not output_dir.exists():
            raise Exception("Output directory not found")
        return {"perturbation_analysis_output_dir": str(output_dir), "current_step": 8}
    except Exception as e:
        logger.exception(f"Perturbation analysis failed: {e}")
        raise
    finally:
        _cleanup_chrome_processes()
        _log_resource_usage()
