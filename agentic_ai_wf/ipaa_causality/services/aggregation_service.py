"""
Pathway summary aggregation service.
No argparse, no subprocess. Calls pathway_summary_aggregator.run_aggregation_from_config.
"""
from __future__ import annotations

from pathlib import Path

from ..config.models import AggregationPhaseConfig, AggregationPhaseResult

# from ..pathway_summary_aggregator import run_aggregation_from_config


def run_aggregation(config: AggregationPhaseConfig | dict) -> AggregationPhaseResult:
    """
    Aggregate pathway summaries for all diseases.
    No argparse, no sys.argv, no subprocess.
    """
    if isinstance(config, dict):
        config = AggregationPhaseConfig.model_validate(config)
    out_root = str(Path(config.out_root).resolve())

    from ..pathway_summary_aggregator import run_aggregation_from_config

    exit_code = run_aggregation_from_config(config)
    success = exit_code == 0

    diseases_aggregated: list[str] = []
    if success:
        out_base = Path(out_root) / (config.out_subdir or "engines/pathway_summary")
        if out_base.exists():
            manifest = out_base / "PATHWAY_SUMMARY_ALL_MANIFEST.json"
            if manifest.exists():
                import json
                with open(manifest) as f:
                    data = json.load(f)
                diseases_aggregated = [d.get("disease", "") for d in data.get("diseases", []) if d.get("disease")]

    return AggregationPhaseResult(
        success=success,
        out_root=out_root,
        diseases_aggregated=diseases_aggregated,
        errors=[] if success else [f"Aggregator exited with code {exit_code}"],
        metadata={"exit_code": exit_code},
    )
