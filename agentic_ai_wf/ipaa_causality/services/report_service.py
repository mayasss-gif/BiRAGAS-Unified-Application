"""
HTML report generation service.
No argparse, no subprocess. Calls generate_ipaa_html_reports.run_reports_from_config.
"""
from __future__ import annotations

from pathlib import Path

from ..config.models import ReportPhaseConfig, ReportPhaseResult

# from ..generate_ipaa_html_reports import run_reports_from_config


def run_reports(config: ReportPhaseConfig | dict) -> ReportPhaseResult:
    """
    Generate HTML reports per disease.
    No argparse, no sys.argv, no subprocess.
    """
    if isinstance(config, dict):
        config = ReportPhaseConfig.model_validate(config)
    outdir = str(Path(config.outdir).resolve())

    from ..generate_ipaa_html_reports import run_reports_from_config

    exit_code = run_reports_from_config(config)
    success = exit_code == 0

    reports_generated: list[str] = []
    if success:
        base = Path(outdir) / "engines" / "pathway_summary"
        if base.exists():
            reports_generated = [d.name for d in base.iterdir() if d.is_dir() and (d / "report.html").exists()]

    return ReportPhaseResult(
        success=success,
        outdir=outdir,
        reports_generated=reports_generated,
        errors=[] if success else [f"Report generation exited with code {exit_code}"],
        metadata={"exit_code": exit_code},
    )
