"""
Workflow-scoped context for passing non-serializable objects through LangGraph.

LangGraph checkpointers serialize state; workflow_logger cannot be in state.
Config passthrough can fail with compiled subgraphs. Use contextvars for reliability.
"""
from contextvars import ContextVar
from typing import Any, Optional

_workflow_logger: ContextVar[Optional[Any]] = ContextVar(
    "workflow_logger", default=None
)


def get_workflow_logger() -> Optional[Any]:
    """Get current workflow logger from context (for DEG and other sub-nodes)."""
    return _workflow_logger.get()


def set_workflow_logger(logger: Optional[Any]) -> None:
    """Set workflow logger in context. Call before app.ainvoke()."""
    _workflow_logger.set(logger)


def reset_workflow_logger() -> None:
    """Clear workflow logger from context. Call after app.ainvoke() in finally."""
    try:
        _workflow_logger.set(None)
    except LookupError:
        pass
