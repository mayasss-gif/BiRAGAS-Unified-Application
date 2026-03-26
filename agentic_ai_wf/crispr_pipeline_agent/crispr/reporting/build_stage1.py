#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Report builder: Stage 1 — Guide Assignment and Perturbation Classification.

All paths are discovered dynamically from the sample output directory.
"""

import json
from pathlib import Path

import pandas as pd

from .report_common import (
    ensure_base_report, write_report, insert_or_replace_block,
    llm_interpret_table, first_match,
)


def format_top_guides(raw_string: str) -> str:
    try:
        guide_dict = json.loads(raw_string.replace("'", '"'))
    except Exception:
        try:
            guide_dict = eval(raw_string)
        except Exception:
            return str(raw_string)

    html = "<ol style='margin:0; padding-left:18px;'>"
    for g, c in guide_dict.items():
        html += f"<li><b>{g}</b>: {c}</li>"
    html += "</ol>"
    return html


def build_stage1_block(df: pd.DataFrame, interpretation: str) -> str:
    row = df.iloc[0]
    total = int(row["total_cells"])
    multiplets = int(row["multiplets"])
    multiplet_rate = (multiplets / total) * 100 if total > 0 else 0
    confidence = float(row["mean_perturbation_confidence"])
    guides_html = format_top_guides(row["top10_guides"])

    return f"""
<!-- STAGE:STAGE1 -->
<div class="stage-container">

    <div class="stage-header">
        Stage 1: Guide Assignment and Perturbation Classification
    </div>

    <div class="stage-content">

        <p class="stage-description">
        This stage normalizes guide RNA identities, assigns cells into <b>control</b>, <b>perturbed</b>, or <b>unknown</b> classes,
        and flags likely multiplets. A composite <i>perturbation confidence</i> score summarizes guide evidence quality prior to
        responder modeling and downstream inference.
        </p>

        <div class="table-scroll table-small" style="max-height:220px;">
            <table class="styled-table sticky">
                <thead>
                    <tr>
                        <th>GSM ID</th>
                        <th>Total</th>
                        <th>Control</th>
                        <th>Perturbed</th>
                        <th>Unknown</th>
                        <th>Multiplets</th>
                        <th>Confidence</th>
                        <th>Top Guides</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td>{row["gsm_id"]}</td>
                        <td class="num">{total:,}</td>
                        <td class="num">{int(row["control_cells"]):,}</td>
                        <td class="num">{int(row["perturbed_cells"]):,}</td>
                        <td class="num">{int(row["unknown_cells"]):,}</td>
                        <td class="num multiplet">{multiplets:,} ({multiplet_rate:.1f}%)</td>
                        <td class="num confidence">{confidence:.4f}</td>
                        <td>{guides_html}</td>
                    </tr>
                </tbody>
            </table>
        </div>

        <div class="caption">
            Table 1. Perturbation assignment statistics and guide-level distribution.
        </div>

        <div class="interpretation">{interpretation}</div>

    </div>
</div>
<!-- END:STAGE1 -->
""".strip()


def build(sample_dir: Path, client=None, html: str = "") -> str:
    """Build Stage 1 block from pipeline outputs in *sample_dir*."""
    table_path = first_match(sample_dir, "tables/sample_control_perturbation_summary.tsv")
    if table_path is None:
        return html

    df = pd.read_csv(table_path, sep="\t")
    interp = llm_interpret_table(
        client, "Stage 1 Perturbation Assignment Summary", df,
        "Highlight perturbation coverage, multiplet burden, confidence interpretation, and guide dominance."
    )
    block = build_stage1_block(df, interp)
    return insert_or_replace_block(html, "STAGE1", block)


if __name__ == "__main__":
    import sys
    d = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    out = d / "CRISPRModel_Simulator_Report.html"
    result = build(d, html=ensure_base_report(str(out)))
    write_report(out, result)
    print("Done: Stage 1")
