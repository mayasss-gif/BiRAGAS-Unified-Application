"""
Pathway Enrichment Agent - WebSocket/UI log streaming support.

Provides _emit_pathway_ui_log for emitting real-time logs from sync/async nodes
when running inside the main workflow (e.g. from langgraph_wf pathway_enrichment_node).
"""

import asyncio
from typing import Optional, Any

# Run-scoped context: thread_id -> {workflow_logger, event_loop}
# Avoids putting non-serializable objects in checkpointed state.
_PATHWAY_UI_CTX: dict[str, dict[str, Any]] = {}

AGENT_NAME = "Pathway Enrichment Agent"
STEP = "pathway_enrichment"


def set_pathway_ui_context(key: str, workflow_logger: Any, event_loop: Optional[asyncio.AbstractEventLoop]) -> None:
    """Store workflow_logger and event_loop for pathway nodes (called from agent_runner)."""
    if key:
        _PATHWAY_UI_CTX[key] = {"workflow_logger": workflow_logger, "event_loop": event_loop}


def clear_pathway_ui_context(key: str) -> None:
    """Clear context after pathway workflow completes."""
    _PATHWAY_UI_CTX.pop(key, None)


def get_pathway_ui_context(key: Optional[str]) -> tuple[Any, Optional[asyncio.AbstractEventLoop]]:
    """Retrieve workflow_logger and event_loop for a run. Returns (None, None) if not set."""
    if not key:
        return None, None
    ctx = _PATHWAY_UI_CTX.get(key, {})
    return ctx.get("workflow_logger"), ctx.get("event_loop")


def _emit_pathway_ui_log(
    workflow_logger: Any,
    event_loop: Optional[asyncio.AbstractEventLoop],
    level: str,
    message: str,
    **kwargs: Any,
) -> None:
    """Emit log to UI via workflow_logger (safe from thread pool)."""
    if not workflow_logger or not event_loop:
        return
    try:
        async def _do_log() -> None:
            try:
                if level == "info":
                    await workflow_logger.info(
                        agent_name=AGENT_NAME, message=message, step=STEP, **kwargs
                    )
                elif level == "warning":
                    await workflow_logger.warning(
                        agent_name=AGENT_NAME, message=message, step=STEP, **kwargs
                    )
                elif level == "error":
                    await workflow_logger.error(
                        agent_name=AGENT_NAME, message=message, step=STEP, **kwargs
                    )
            except Exception:
                pass

        # Use run_coroutine_threadsafe for reliable scheduling from thread pool
        # (pathway sync nodes run in executor; gene_prioritization uses same pattern)
        asyncio.run_coroutine_threadsafe(_do_log(), event_loop)
    except Exception:
        pass


def emit_from_state(state: dict, level: str, message: str, **kwargs: Any) -> None:
    """
    Emit UI log using workflow_logger from run-scoped context.
    Call from pathway nodes; safe in sync (thread pool) or async context.
    """
    key = state.get("_pathway_ui_ctx_key")
    workflow_logger, event_loop = get_pathway_ui_context(key)
    _emit_pathway_ui_log(workflow_logger, event_loop, level, message, **kwargs)


async def emit_from_state_async(state: dict, level: str, message: str, **kwargs: Any) -> None:
    """
    Emit UI log directly when in async context (no call_soon_threadsafe).
    Use in async nodes (literature, consolidation) for immediate emission.
    """
    key = state.get("_pathway_ui_ctx_key")
    workflow_logger, _ = get_pathway_ui_context(key)
    if not workflow_logger:
        return
    try:
        if level == "info":
            await workflow_logger.info(
                agent_name=AGENT_NAME, message=message, step=STEP, **kwargs
            )
        elif level == "warning":
            await workflow_logger.warning(
                agent_name=AGENT_NAME, message=message, step=STEP, **kwargs
            )
        elif level == "error":
            await workflow_logger.error(
                agent_name=AGENT_NAME, message=message, step=STEP, **kwargs
            )
    except Exception:
        pass
