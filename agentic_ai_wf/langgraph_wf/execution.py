"""Workflow execution functions."""

import logging
import time
import uuid
from typing import Any, Dict, Optional

from agentic_ai_wf.global_config import get_global_config

from .graph import build_transcriptome_analysis_graph
from .state import TranscriptomeAnalysisState

logger = logging.getLogger(__name__)


async def run_transcriptome_analysis(
    analysis_id: str,
    disease_name: str,
    patient_id: str,
    patient_name: str,
    analysis_transcriptome_dir: str,
    user_id: str,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Main entry point for LangGraph-based workflow."""
    from agentic_ai_wf.helpers.logging_utils import AsyncWorkflowLogger

    app = build_transcriptome_analysis_graph()
    global_cfg = get_global_config()
    workflow_id = f"WF-{uuid.uuid4().hex[:8]}"
    workflow_logger = AsyncWorkflowLogger(
        user_id=user_id,
        analysis_id=analysis_id,
        workflow_name="Agentic AI Transcriptome Analysis (LangGraph)",
        total_steps=9,
        task_id=kwargs.get("task_id", ""),
        correlation_id=workflow_id,
    )

    await workflow_logger.start_workflow()

    initial_state: TranscriptomeAnalysisState = {
        "global_config": global_cfg,
        "analysis_id": analysis_id,
        "disease_name": disease_name,
        "patient_id": patient_id,
        "patient_name": patient_name or "Test Patient",
        "user_id": user_id,
        "analysis_transcriptome_dir": analysis_transcriptome_dir,
        "workflow_id": workflow_id,
        "run_id": str(uuid.uuid4()),
        "start_time": time.time(),
        "current_step": 0,
        "total_steps": 9,
        "user_query": kwargs.get("user_query"),
        "tissue_filter": kwargs.get("tissue_filter"),
        "experiment_filter": kwargs.get("experiment_filter"),
        "errors": [],
        "failed_steps": [],
        "retry_count": 0,
        "step_start_times": {},
        "step_durations": {},
        "enable_cleanup": kwargs.get("enable_cleanup", True),
        "enable_progress_tracking": kwargs.get("enable_progress_tracking", True),
        "max_retries": kwargs.get("max_retries", 0),
        "workflow_failed": False,
        "failure_reason": None,
    }

    config = {
        "configurable": {
            "thread_id": analysis_id,
            "workflow_logger": workflow_logger,
        }
    }

    logger.info(f"Starting workflow {workflow_id} for analysis {analysis_id}")

    try:
        final_state = await app.ainvoke(initial_state, config=config)

        if final_state.get("workflow_failed"):
            await workflow_logger.fail_workflow(
                error_message=final_state.get("failure_reason", "Workflow failed"),
                error_code="WORKFLOW_FAILED",
            )
            raise Exception(final_state.get("failure_reason", "Workflow failed"))

        execution_time_ms = int((time.time() - final_state["start_time"]) * 1000)
        await workflow_logger.complete_workflow(elapsed_ms=execution_time_ms)

        return {
            "success": True,
            "reports": {
                "clinical_report": final_state.get("clinical_report_path"),
                "pharma_report": final_state.get("pharma_report_path"),
            },
            "execution_time_ms": execution_time_ms,
            "steps_completed": final_state.get("current_step", 0),
            "total_steps": final_state.get("total_steps", 9),
            "errors": final_state.get("errors", []),
            "failed_steps": final_state.get("failed_steps", []),
            "step_durations": final_state.get("step_durations", {}),
        }
    except Exception as e:
        await workflow_logger.fail_workflow(
            error_message=str(e),
            error_code="UNEXPECTED_ERROR",
            exception=e,
        )
        raise


async def resume_workflow(analysis_id: str) -> Dict[str, Any]:
    """Resume a failed/interrupted workflow from its last checkpoint."""
    app = build_transcriptome_analysis_graph()
    config = {"configurable": {"thread_id": analysis_id}}
    checkpoints = list(app.checkpointer.list(config))
    if not checkpoints:
        raise ValueError(f"No checkpoint found for analysis_id: {analysis_id}")
    logger.info(f"Resuming workflow from checkpoint for analysis {analysis_id}")
    final_state = await app.ainvoke(None, config=config)
    return {
        "success": True,
        "reports": {
            "clinical_report": final_state.get("clinical_report_path"),
            "pharma_report": final_state.get("pharma_report_path"),
        },
        "resumed": True,
        "checkpoint_id": checkpoints[0].checkpoint_id,
    }


def inspect_workflow_state(analysis_id: str) -> TranscriptomeAnalysisState:
    """Inspect the current state of a workflow."""
    app = build_transcriptome_analysis_graph()
    config = {"configurable": {"thread_id": analysis_id}}
    checkpoints = list(app.checkpointer.list(config))
    if not checkpoints:
        raise ValueError(f"No checkpoints found for {analysis_id}")
    return checkpoints[0].checkpoint["channel_values"]


def visualize_workflow() -> str:
    """Generate a Mermaid diagram of the workflow."""
    app = build_transcriptome_analysis_graph()
    return app.get_graph().draw_mermaid()


async def main(
    analysis_id: str,
    disease_name: str,
    patient_id: str,
    patient_name: str,
    analysis_transcriptome_dir: str,
    user_id: Optional[str] = None,
    **kwargs: Any,
) -> Dict[str, str]:
    """Backward-compatible entry point matching original API."""
    result = await run_transcriptome_analysis(
        analysis_id=analysis_id,
        disease_name=disease_name,
        patient_id=patient_id,
        patient_name=patient_name,
        analysis_transcriptome_dir=analysis_transcriptome_dir,
        user_id=user_id or "system",
        **kwargs,
    )
    if not result["success"]:
        raise Exception(result.get("failure_reason", "Workflow failed"))
    return result["reports"]
