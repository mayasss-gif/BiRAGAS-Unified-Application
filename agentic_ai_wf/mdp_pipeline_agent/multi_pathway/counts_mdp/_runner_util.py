# counts_mdp/_runner_util.py
from __future__ import annotations
import sys, subprocess
from pathlib import Path

def run_script(relpath: str) -> int:
    """
    Run a repo-top-level Python script by relative path from this file.
    Passes through all CLI args unchanged.
    """
    here = Path(__file__).resolve()
    repo_root = here.parents[1]  # .../mdp_pipeline_3/
    script = (repo_root / relpath).resolve()
    if not script.exists():
        print(f"[runner] Script not found: {script}")
        return 2
    cmd = [sys.executable, str(script), *sys.argv[1:]]
    return subprocess.call(cmd)
