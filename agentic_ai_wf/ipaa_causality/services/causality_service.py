"""
Causality engines (Engine0–3) service.
No argparse, no subprocess. Calls run_causality_all.run_causality_from_config.
"""
from __future__ import annotations

from pathlib import Path

from ..config.models import CausalityPhaseConfig, CausalityPhaseResult

# from ..run_causality_all import run_causality_from_config


def run_causality(config: CausalityPhaseConfig | dict) -> CausalityPhaseResult:
    """
    Execute causality workflow (Engine0–3).
    No argparse, no sys.argv, no subprocess.
    """
    if isinstance(config, dict):
        config = CausalityPhaseConfig.model_validate(config)
    out_root = str(Path(config.out_root).resolve())

    from ..run_causality_all import run_causality_from_config

    errors: list[str] = []
    try:
        run_causality_from_config(config)
        success = True
        diseases_processed = config.diseases or []
        if not diseases_processed:
            summary_path = Path(out_root) / "engines" / "CAUSALITY_ALL_RUN_SUMMARY.tsv"
            if summary_path.exists():
                import pandas as pd
                df = pd.read_csv(summary_path, sep="\t")
                diseases_processed = df["disease"].astype(str).tolist() if "disease" in df.columns else []
    except Exception as e:
        success = False
        errors.append(str(e))
        diseases_processed = []

    return CausalityPhaseResult(
        success=success,
        out_root=out_root,
        diseases_processed=diseases_processed,
        errors=errors,
        metadata={},
    )
