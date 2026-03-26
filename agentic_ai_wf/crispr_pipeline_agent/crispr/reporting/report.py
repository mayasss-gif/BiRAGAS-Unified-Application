#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CRISPRModel Full Report Builder — Dynamic Orchestrator

Generates a complete HTML report for a pipeline run by invoking each
stage builder **in-process** (no subprocesses).  All paths are derived
dynamically from the sample output directory.

Usage (programmatic):
    from crispr.reporting import generate_report
    generate_report(sample_dir=Path("crispr_output/GSE90546/GSM2406675_10X001"))

Usage (CLI):
    python -m crispr.reporting.report /path/to/sample_output_dir
"""

import logging
import time
from pathlib import Path
from typing import Optional

from .report_common import (
    get_openai_client,
    ensure_base_report,
    write_report,
)
from . import build_header_dataset_stage0
from . import build_stage1
from . import build_stage2
from . import build_stage3
from . import build_stage4
from . import build_stage5
from . import build_stage6
from . import build_stage7
from . import build_stage8
from . import build_stage9
from . import build_stage10_11
from . import build_stage12

logger = logging.getLogger(__name__)

STAGE_BUILDERS = [
    ("Header + Dataset + Stage0", build_header_dataset_stage0),
    ("Stage 1", build_stage1),
    ("Stage 2", build_stage2),
    ("Stage 3", build_stage3),
    ("Stage 4", build_stage4),
    ("Stage 5", build_stage5),
    ("Stage 6", build_stage6),
    ("Stage 7", build_stage7),
    ("Stage 8", build_stage8),
    ("Stage 9", build_stage9),
    ("Stage 10-11", build_stage10_11),
    ("Stage 12", build_stage12),
]


def generate_report(
    sample_dir: Path,
    output_path: Optional[Path] = None,
) -> Path:
    """
    Generate a full HTML report for a single pipeline sample run.

    Parameters
    ----------
    sample_dir : Path
        The sample-level output directory produced by ``run_pipeline``.
        For example: ``crispr_output/GSE90546/GSM2406675_10X001``
    output_path : Path, optional
        Where to write the final HTML report.  Defaults to
        ``<sample_dir>/CRISPRModel_Report_<sample_name>.html``.

    Returns
    -------
    Path
        The path to the generated HTML report.
    """
    sample_dir = Path(sample_dir).resolve()
    if output_path is None:
        sample_name = sample_dir.name
        output_path = sample_dir / f"CRISPRModel_Report_{sample_name}.html"
    else:
        output_path = Path(output_path).resolve()

    client = get_openai_client()

    print(f"\n{'=' * 70}")
    print("CRISPR Perturb-seq Report Generation")
    print(f"{'=' * 70}")
    print(f"  Sample directory : {sample_dir}")
    print(f"  Output report    : {output_path}")
    print(f"  LLM enabled      : {'Yes' if client else 'No (set OPENAI_API_KEY in .env)'}")
    print(f"{'=' * 70}\n")

    html = ensure_base_report(str(output_path))

    for stage_name, builder in STAGE_BUILDERS:
        t0 = time.time()
        print(f"  Building {stage_name} ... ", end="", flush=True)

        try:
            if builder is build_header_dataset_stage0:
                html = builder.build(sample_dir, client=client, report_path=str(output_path))
            else:
                html = builder.build(sample_dir, client=client, html=html)
            dt = time.time() - t0
            print(f"done ({dt:.1f}s)")
        except Exception as exc:
            dt = time.time() - t0
            print(f"FAILED ({dt:.1f}s) — {exc}")
            logger.exception("Error building %s", stage_name)

    write_report(str(output_path), html)

    print(f"\n{'=' * 70}")
    print(f"Report saved: {output_path}")
    print(f"{'=' * 70}\n")

    return output_path


def main():
    """CLI entry point."""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m crispr.reporting.report <sample_dir> [output.html]")
        sys.exit(1)

    sample_dir = Path(sys.argv[1])
    output_path = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    generate_report(sample_dir, output_path)


if __name__ == "__main__":
    main()
