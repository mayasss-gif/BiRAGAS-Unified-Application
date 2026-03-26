#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Report builder: Stage 4 — Differential Expression & Signature Extraction.

Dynamically discovers DEG tables and ranked perturbation files.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from .report_common import (
    insert_or_replace_block, ensure_base_report, write_report,
    llm_interpret_table, df_to_scrollable_table,
    first_match,
)


def select_top5(ranked_df):
    ranked_df = ranked_df.copy()
    ranked_df["responder_rate"] = pd.to_numeric(ranked_df["responder_rate"], errors="coerce")
    ranked_df = ranked_df.sort_values(["responder_rate", "n_cells"], ascending=[False, False])
    return ranked_df["perturbation_id"].head(5).tolist()


def build_combined_volcano(deg_df, top_perts):
    try:
        import plotly.express as px
        import plotly.io as pio
    except ImportError:
        return "<p><em>Plotly not installed — volcano plot skipped.</em></p>"

    df = deg_df[deg_df["perturbation_id"].isin(top_perts)].copy()
    df["logfoldchange"] = pd.to_numeric(df["logfoldchange"], errors="coerce")
    df["pval_adj"] = pd.to_numeric(df["pval_adj"], errors="coerce")
    df = df.dropna(subset=["logfoldchange", "pval_adj"])
    df["neg_log10_padj"] = -np.log10(df["pval_adj"] + 1e-300)

    df["significance"] = "NS"
    df.loc[
        (df["pval_adj"] < 0.05) & (df["logfoldchange"].abs() > 0.5),
        "significance",
    ] = "Significant"

    fig = px.scatter(
        df, x="logfoldchange", y="neg_log10_padj",
        color="perturbation_id", symbol="significance",
        hover_data=["gene", "pval_adj"],
        title="Combined Volcano Plot: Top 5 Perturbations",
    )
    fig.update_layout(template="plotly_white", height=650, legend_title="Perturbation")
    return pio.to_html(fig, include_plotlyjs="cdn", full_html=False)


def build_stage4_block(deg_df, volcano_html, interp_deg):
    deg_table_html = df_to_scrollable_table(
        deg_df.head(50), table_id="stage4_deg",
        numeric_cols=["score", "logfoldchange", "pval_adj"],
        max_height_px=260,
    )

    return f"""
<!-- STAGE:STAGE4 -->
<div class="stage-container">

  <div class="stage-header" style="background: linear-gradient(90deg,#14532d,#22c55e);">
      Stage 4: Differential Expression & Signature Extraction
  </div>

  <div class="stage-content">

    <p class="stage-description">
    Stage 4 performs <b>Differential Gene Expression (DEG)</b> analysis
    comparing each perturbation against control using Wilcoxon testing.
    Significant transcriptional shifts define perturbation-specific
    molecular signatures.
    </p>

    <div class="subsection-title">
        Top Differential Markers (Top 50)
    </div>

    {deg_table_html}

    <div class="interpretation">{interp_deg}</div>

    <div class="subsection-title">
        Combined Volcano Plot (Top 5 Perturbations)
    </div>

    <div class="volcano-card">
        {volcano_html}
    </div>

  </div>
</div>
<!-- END:STAGE4 -->
""".strip()


def build(sample_dir: Path, client=None, html: str = "") -> str:
    """Build Stage 4 block from pipeline outputs in *sample_dir*."""
    deg_path = first_match(sample_dir, "**/stage4_deg_top_markers.tsv")
    ranked_path = first_match(sample_dir, "**/stage3_perturbation_ranked.tsv")

    if deg_path is None:
        return html

    deg_df = pd.read_csv(deg_path, sep="\t")

    top_perts = []
    if ranked_path is not None:
        ranked_df = pd.read_csv(ranked_path, sep="\t")
        top_perts = select_top5(ranked_df)

    volcano_html = build_combined_volcano(deg_df, top_perts) if top_perts else ""

    interp_deg = llm_interpret_table(
        client, "Stage 4 DEG Table", deg_df.head(50),
        "Explain dominant transcriptional shifts and biological relevance."
    )

    block = build_stage4_block(deg_df, volcano_html, interp_deg)
    return insert_or_replace_block(html, "STAGE4", block)


if __name__ == "__main__":
    import sys
    d = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    out = d / "CRISPRModel_Simulator_Report.html"
    result = build(d, html=ensure_base_report(str(out)))
    write_report(out, result)
    print("Done: Stage 4")
