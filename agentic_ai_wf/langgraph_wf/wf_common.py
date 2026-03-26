"""Shared workflow utilities: state sanitization, global config, error routing."""

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from .state import TranscriptomeAnalysisState


def sanitize_for_state(obj: Any) -> Any:
    """Recursively sanitize objects for state storage."""
    if obj is None:
        return None

    if isinstance(obj, pd.DataFrame):
        if len(obj) > 1000:
            return {"_type": "dataframe", "shape": list(obj.shape), "columns": list(obj.columns), "note": "dataframe_too_large"}
        return {"_type": "dataframe", "shape": list(obj.shape), "columns": list(obj.columns), "data": obj.to_dict("records")}

    if isinstance(obj, pd.Series):
        return {"_type": "series", "data": obj.to_dict()}

    if isinstance(obj, np.ndarray):
        return {
            "_type": "ndarray",
            "shape": list(obj.shape),
            "dtype": str(obj.dtype),
            "data": obj.tolist() if obj.size < 1000 else "too_large",
        }

    if isinstance(obj, Path):
        return str(obj)

    if isinstance(obj, dict):
        return {k: sanitize_for_state(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        return [sanitize_for_state(item) for item in obj]

    if hasattr(obj, "__dict__") and not isinstance(obj, type):
        try:
            return sanitize_for_state(obj.__dict__)
        except Exception:
            return str(obj)

    try:
        json.dumps(obj)
        return obj
    except (TypeError, ValueError):
        return str(obj)


def ensure_global_config(state: Dict[str, Any]) -> "GlobalAgentConfig":
    """Ensure global_config exists in state; reload if missing."""
    from agentic_ai_wf.global_config import GlobalAgentConfig, get_global_config

    if "global_config" not in state or state["global_config"] is None:
        state["global_config"] = get_global_config()
    return state["global_config"]


def should_continue_after_error(state: TranscriptomeAnalysisState) -> str:
    """Router for conditional error handling."""
    failed_steps = state.get("failed_steps", [])
    critical_steps = {"cohort_retrieval", "deg_analysis"}
    if any(s in critical_steps for s in failed_steps):
        return "abort"
    if len(failed_steps) > 3:
        return "abort"
    return "continue"
