# src/bisque_deconv/utils.py
from __future__ import annotations
import os
from pathlib import Path
from datetime import datetime
from typing import Optional, List

def abspath_any(p: Optional[str]) -> Optional[str]:
    """Expand ~ and return absolute path for files/dirs; None if blank."""
    if not p:
        return None
    q = Path(str(p)).expanduser()
    try:
        q = q if q.is_absolute() else q.resolve(strict=False)
    except Exception:
        q = Path(os.path.abspath(str(q)))
    return str(q)

def timestamped_run_root(root_name: str = "bisque_runs") -> str:
    """~/bisque_runs/2025-10-27_153012"""
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    root = Path.home() / root_name / stamp
    root.mkdir(parents=True, exist_ok=True)
    return str(root)

__all__ = ["abspath_any", "timestamped_run_root"]
