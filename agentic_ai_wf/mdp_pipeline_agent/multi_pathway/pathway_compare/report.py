from __future__ import annotations
import logging
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

def write_single_report(
    out_dir: Path,
    pathway: str,
    coverage: pd.DataFrame,
    shared_counts: pd.DataFrame,
    jaccard: pd.DataFrame,
    pvals: pd.DataFrame,
    figures: List[Path],
) -> None:
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        md = out_dir / f"{safe_name(pathway)}_report.md"
        lines: List[str] = []
        lines.append(f"# Pathway: {pathway}\n")
        lines.append("## Summary\n")
        total_cov = coverage["n_genes"].sum()
        lines.append(f"- Diseases covered: **{coverage.shape[0]}**, total genes across diseases: **{int(total_cov)}**\n")

        lines.append("## Coverage per Disease\n")
        lines.append(coverage.to_markdown(index=False))
        lines.append("")

        lines.append("## Pairwise Shared (counts)\n")
        lines.append(shared_counts.to_markdown(index=False))
        lines.append("")

        lines.append("## Pairwise Jaccard\n")
        lines.append(jaccard.to_markdown(index=False))
        lines.append("")

        lines.append("## Pairwise One-sided Overlap p-values\n")
        lines.append(pvals.to_markdown(index=False))
        lines.append("")

        if figures:
            lines.append("## Figures\n")
            for f in figures:
                rel = f.name
                lines.append(f"![{pathway}]({rel})")

        md.write_text("\n".join(lines), encoding="utf-8")
        logging.info(f"Wrote single pathway report: {md}")
    except Exception as e:
        logging.error(f"write_single_report failed: {e}")

def safe_name(s: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-","_"," ") else "_" for ch in s).strip().replace(" ","_")
