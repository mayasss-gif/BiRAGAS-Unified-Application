# counts_mdp/pipeline_cli.py
from __future__ import annotations
import sys
from ._runner_util import run_script

def main() -> None:
    sys.exit(run_script("pipeline_run.py"))
