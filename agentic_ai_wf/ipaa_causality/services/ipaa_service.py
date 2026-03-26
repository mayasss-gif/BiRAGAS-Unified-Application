"""
IPAA M6 preprocessing and enrichment service.
No argparse, no subprocess. Calls main_ipaa.run_ipaa_from_config.
"""
from __future__ import annotations

from pathlib import Path

from ..config.models import IPAAPhaseConfig, IPAAPhaseResult

# Import from parent to avoid circular imports - use late import in run_ipaa
# from ..main_ipaa import run_ipaa_from_config


def run_ipaa(config: IPAAPhaseConfig | dict) -> IPAAPhaseResult:
    """
    Execute IPAA M6 preprocessing and enrichment.
    No argparse, no sys.argv, no subprocess.
    """
    if isinstance(config, dict):
        config = IPAAPhaseConfig.model_validate(config)
    out_root = str(Path(config.outdir).resolve())

    from ..main_ipaa import run_ipaa_from_config

    exit_code = run_ipaa_from_config(config)
    success = exit_code == 0

    return IPAAPhaseResult(
        success=success,
        out_root=out_root,
        cohort_runs=[],
        errors=[] if success else [f"IPAA exited with code {exit_code}"],
        metadata={"exit_code": exit_code},
    )
