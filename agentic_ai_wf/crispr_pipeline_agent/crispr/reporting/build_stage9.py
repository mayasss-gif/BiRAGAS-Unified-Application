#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Report builder: Stage 9 — Causal Inference (IV + 2SLS).

Dynamically discovers IV results and diagnostics from processed_stage9/.
"""

from pathlib import Path

import pandas as pd

from .report_common import (
    insert_or_replace_block, ensure_base_report, write_report,
    llm_interpret_table, df_to_scrollable_table,
    encode_image_to_base64, first_match,
)


def build_stage9_block(iv_df, iv_interp, diag_df, diag_interp, fig_b64):
    iv_table_html = df_to_scrollable_table(
        iv_df, table_id="stage9_iv",
        max_rows=50,
        numeric_cols=["beta_iv", "se_robust", "p_value"],
        max_height_px=220,
    )

    diag_table_html = df_to_scrollable_table(
        diag_df, table_id="stage9_diag",
        max_height_px=180,
    )

    fig_html = ""
    if fig_b64:
        fig_html = f"""
        <div class="fig-card">
          <img class="figure-img"
               src="data:image/png;base64,{fig_b64}">
          <div class="caption">
            Figure 9A. Top first-stage instrumental variable coefficients (absolute magnitude).
          </div>
        </div>
        """

    return f"""
<!-- STAGE:STAGE9 -->
<div class="stage-container">

  <div class="stage-header" style="background: linear-gradient(90deg,#1e3a8a,#2563eb);">
      Stage 9: Causal Inference (IV + 2SLS)
  </div>

  <div class="stage-content">

    <p class="stage-description">
    Stage 9 estimates causal relationships between predicted perturbation labels
    and downstream transcriptional variation using
    <b>Instrumental Variables (IV)</b> and
    <b>Two-Stage Least Squares (2SLS)</b>.
    First-stage models assess instrument strength, while second-stage models
    quantify causal effect estimates under exogeneity assumptions.
    </p>

    <div class="two-col">

      <div class="panel-card">
        <div class="panel-title">IV Effect Estimates (Top rows)</div>
        {iv_table_html}
        <div class="caption">
            Table 9A. Two-stage least squares (2SLS) causal effect estimates.
        </div>
        <div class="interpretation">{iv_interp}</div>
      </div>

      <div>
        {fig_html}
      </div>

    </div>

    <div style="margin-top:35px;">
      <div class="panel-card">
        <div class="panel-title">Model Diagnostics</div>
        {diag_table_html}
        <div class="caption">
            Table 9B. First- and second-stage R\u00b2 metrics and instrument count.
        </div>
        <div class="interpretation">{diag_interp}</div>
      </div>
    </div>

  </div>
</div>
<!-- END:STAGE9 -->
"""


def build(sample_dir: Path, client=None, html: str = "") -> str:
    """Build Stage 9 block from pipeline outputs in *sample_dir*."""
    s9 = sample_dir / "processed_stage9"

    iv_path = first_match(s9, "**/stage9_iv_results.tsv")
    if iv_path is None:
        iv_path = first_match(sample_dir, "**/stage9_iv_results.tsv")

    diag_path = first_match(s9, "**/stage9_diagnostics.tsv")
    if diag_path is None:
        diag_path = first_match(sample_dir, "**/stage9_diagnostics.tsv")

    if iv_path is None and diag_path is None:
        return html

    iv_df = pd.read_csv(iv_path, sep="\t") if iv_path else pd.DataFrame()
    diag_df = pd.read_csv(diag_path, sep="\t") if diag_path else pd.DataFrame()

    iv_interp = llm_interpret_table(
        client, "Stage 9 IV Results", iv_df,
        "Interpret the magnitude, direction, and statistical significance of causal effect estimates."
    )
    diag_interp = llm_interpret_table(
        client, "Stage 9 Diagnostics", diag_df,
        "Explain instrument strength (first-stage R2), explanatory power (second-stage R2), and adequacy of instrument count."
    )

    fig_path = first_match(s9, "**/stage9_first_stage_top_instruments.png")
    if fig_path is None:
        fig_path = first_match(sample_dir, "**/stage9_first_stage_top_instruments.png")
    fig_b64 = encode_image_to_base64(fig_path) if fig_path else None

    block = build_stage9_block(iv_df, iv_interp, diag_df, diag_interp, fig_b64)
    return insert_or_replace_block(html, "STAGE9", block)


if __name__ == "__main__":
    import sys
    d = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    out = d / "CRISPRModel_Simulator_Report.html"
    result = build(d, html=ensure_base_report(str(out)))
    write_report(out, result)
    print("Done: Stage 9")
