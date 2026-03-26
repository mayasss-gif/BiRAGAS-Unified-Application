#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Report builder: Stage 8 — Quality Control and Optimization.

Dynamically discovers QC tables and figures from processed_stage8/.
"""

from pathlib import Path

import pandas as pd

from .report_common import (
    insert_or_replace_block, ensure_base_report, write_report,
    llm_interpret_table, df_to_scrollable_table,
    encode_image_to_base64, first_match, find_figures,
)


def build_stage8_block(table1_html, interp1, table2_html, interp2,
                       fig_umap_b64, fig_bar_b64):
    fig_section = ""
    if fig_umap_b64:
        fig_section += f"""
        <div class="fig-card">
            <img class="figure-img" src="data:image/png;base64,{fig_umap_b64}">
            <div class="caption">
                Figure 8A. UMAP embedding colored by prediction confidence.
            </div>
        </div>
        """

    if fig_bar_b64:
        fig_section += f"""
        <div class="fig-card">
            <img class="figure-img" src="data:image/png;base64,{fig_bar_b64}">
            <div class="caption">
                Figure 8B. Top predicted perturbations after QC filtering.
            </div>
        </div>
        """

    return f"""
<!-- STAGE:STAGE8 -->
<div class="stage-container">

    <div class="stage-header" style="background: linear-gradient(90deg,#1e3a8a,#2563eb);">
        Stage 8: Quality Control and Optimization
    </div>

    <div class="stage-content">

        <p class="stage-description">
        Stage 8 applies confidence-based filtering to refine predicted perturbation labels.
        Cells are stratified into high- and low-confidence groups, enabling threshold
        optimization and robustness evaluation of model predictions.
        </p>

        <div class="subsection-title">
            Perturbation-Level Confidence Summary (Top 50)
        </div>

        <div class="panel-card">
            {table1_html}
            <div class="caption">
                Table 8A. Predicted perturbations ranked by mean confidence and fraction of high-confidence cells.
            </div>
            <div class="interpretation">{interp1}</div>
        </div>

        <div class="subsection-title">
            High-Confidence Cells (Top 50)
        </div>

        <div class="panel-card">
            {table2_html}
            <div class="caption">
                Table 8B. Example high-confidence cell-level predictions.
            </div>
            <div class="interpretation">{interp2}</div>
        </div>

        <div class="subsection-title">
            Confidence Diagnostics
        </div>

        <div class="fig-grid">
            {fig_section}
        </div>

    </div>
</div>
<!-- END:STAGE8 -->
"""


def build(sample_dir: Path, client=None, html: str = "") -> str:
    """Build Stage 8 block from pipeline outputs in *sample_dir*."""
    s8 = sample_dir / "processed_stage8"

    pert_qc = first_match(s8, "**/stage8_predicted_perturbation_qc.tsv")
    if pert_qc is None:
        pert_qc = first_match(sample_dir, "**/stage8_predicted_perturbation_qc.tsv")

    if pert_qc is not None:
        df1 = pd.read_csv(pert_qc, sep="\t")
        df1 = df1.sort_values(
            ["mean_conf", "frac_high_conf"], ascending=[False, False]
        ).head(50)
        table1_html = df_to_scrollable_table(
            df1, table_id="stage8_pert_qc",
            numeric_cols=["n_cells", "mean_conf", "median_conf", "frac_high_conf"],
            percent_cols=["frac_high_conf"],
            max_height_px=280,
        )
        interp1 = llm_interpret_table(
            client, "Stage 8 Perturbation Confidence QC", df1,
            "Explain confidence stability, robustness across perturbations, and threshold implications."
        )
    else:
        table1_html = "<p><em>No data available.</em></p>"
        interp1 = ""

    highconf_path = first_match(s8, "**/stage8_cells_highconf*.tsv")
    if highconf_path is None:
        highconf_path = first_match(sample_dir, "**/stage8_cells_highconf*.tsv")

    if highconf_path is not None:
        df2 = pd.read_csv(highconf_path, sep="\t")
        df2 = df2.sort_values(["pred_confidence"], ascending=False).head(50)
        table2_html = df_to_scrollable_table(
            df2, table_id="stage8_highconf_cells",
            numeric_cols=["pred_confidence"],
            max_height_px=280,
        )
        interp2 = llm_interpret_table(
            client, "Stage 8 High Confidence Cells", df2.head(50),
            "Interpret high-confidence distribution and implications for label refinement."
        )
    else:
        table2_html = "<p><em>No data available.</em></p>"
        interp2 = ""

    umap_fig = first_match(s8, "**/stage8_umap_pred_confidence.png")
    if umap_fig is None:
        umap_fig = first_match(sample_dir, "**/stage8_umap_pred_confidence.png")

    bar_fig = first_match(s8, "**/stage8_top_predicted_perts.png")
    if bar_fig is None:
        bar_fig = first_match(sample_dir, "**/stage8_top_predicted_perts.png")

    fig_umap_b64 = encode_image_to_base64(umap_fig) if umap_fig else None
    fig_bar_b64 = encode_image_to_base64(bar_fig) if bar_fig else None

    block = build_stage8_block(table1_html, interp1, table2_html, interp2,
                               fig_umap_b64, fig_bar_b64)
    return insert_or_replace_block(html, "STAGE8", block)


if __name__ == "__main__":
    import sys
    d = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    out = d / "CRISPRModel_Simulator_Report.html"
    result = build(d, html=ensure_base_report(str(out)))
    write_report(out, result)
    print("Done: Stage 8")
