# GC_enricher/pairwise_cli.py
from __future__ import annotations
import sys
from . import GC_2 as mod

def main() -> None:
    # Prefer calling a function if present; else run script.
    try:
        fn = getattr(mod, "main", None)
        if callable(fn):
            fn()
            return
    except Exception as e:
        print(f"[mdp-gc2] falling back to script exec: {e}")
    from counts_mdp._runner_util import run_script
    sys.exit(run_script("GC_enricher/GC_2.py"))
