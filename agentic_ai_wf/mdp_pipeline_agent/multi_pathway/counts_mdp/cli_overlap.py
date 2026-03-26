# counts_mdp/cli_overlap.py
from __future__ import annotations
import sys
from . import mdp_overlap as mod
from ._runner_util import run_script


def main() -> None:
    # Try to call a function if it exists; else just execute the module as a script.
    try:
        fn = getattr(mod, "main", None) or getattr(mod, "build_overlap_jsons", None)
        if callable(fn):
            fn()  # relies on mdp_overlap to parse its own args
            return
    except Exception as e:
        print(f"[mdp-overlap] falling back to script exec: {e}")
    sys.exit(run_script("counts_mdp/mdp_overlap.py"))
