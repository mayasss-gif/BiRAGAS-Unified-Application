"""Workflow error handler node."""

from .node_logger import create_logged_node
from .state import TranscriptomeAnalysisState


@create_logged_node("error_handler", step_number=0)
async def error_handler_node(state: TranscriptomeAnalysisState) -> TranscriptomeAnalysisState:
    """Handle workflow abortion due to critical errors."""
    errors = state.get("errors", [])
    error_summary = "\n".join([f"- {e['step']}: {e['error']}" for e in errors])
    return {
        "workflow_failed": True,
        "failure_reason": f"Critical errors occurred:\n{error_summary}",
    }
