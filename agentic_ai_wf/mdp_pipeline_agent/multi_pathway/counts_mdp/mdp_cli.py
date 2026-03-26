# mdp_cli.py
from __future__ import annotations

from pathlib import Path
import sys, subprocess  # NEW

from .mdp_orchestrator import run_all
from .mdp_overlap import build_overlap_jsons
from .mdp_logging import info, warn  # warn added
from .mdp_config import out_root

def _run_json_comparison_counts() -> None:  # NEW
    """
    Run the robust JSON comparison on the counts pipeline outputs.
    Input JSONs are the per-cohort files mirrored into:
      <OUT_ROOT>/results/all_jsons/*.json
    Results go to:
      <OUT_ROOT>/json_comparison/
    """
    try:
        # Figure OUT_ROOT the same way other modules do
        
        root = out_root()
        in_dir = root / "results" / "all_jsons"
        out_dir = root / "json_comparison"
        out_dir.mkdir(parents=True, exist_ok=True)

        if not in_dir.exists():
            warn(f"[json-compare] Input JSON folder not found: {in_dir}")
            return

        # Call the standalone script via the same Python interpreter
        script = Path(__file__).parent.parent / "json_comparison" / "json_compare.py"
        cmd = [
            sys.executable, str(script),
            "--input", str(in_dir),
            "--outdir", str(out_dir),
            "--prefix", "counts_pipeline_comparison",
            "-v"
        ]
        info(f"[json-compare] Running: {' '.join(cmd)}")
        subprocess.run(cmd, check=False)
    except Exception as e:
        warn(f"[json-compare] Skipped due to error: {e}")

def main() -> None:
    # 1) Run the full DE→GSEA→VIPER→ULM→consensus pipeline
    run_all()

    # 2) Build pathway–entity overlap JSONs per cohort
    info("[CLI] Starting overlap JSON construction...")
    build_overlap_jsons()
    info("[CLI] Overlap JSONs done.")

    # 3) Compare JSONs across cohorts (NEW)
    _run_json_comparison_counts()
    info("[CLI] JSON comparison done.")

if __name__ == "__main__":
    main()
