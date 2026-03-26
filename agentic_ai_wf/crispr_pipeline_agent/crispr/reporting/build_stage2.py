#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Report builder: Stage 2 — Mixscape Responder Deconvolution.

Paths are discovered dynamically from the sample output directory.
"""

from pathlib import Path

import pandas as pd

from .report_common import (
    insert_or_replace_block, ensure_base_report, write_report,
    llm_interpret_table, df_to_scrollable_table,
    encode_image_to_base64, first_match, find_figures,
)


def _find_representative_figure(sample_dir: Path, prefix: str):
    """Find the first figure matching a mixscape pattern."""
    hits = find_figures(sample_dir, f"figures/{prefix}*.png")
    return hits[0] if hits else None


def build_stage2_block(t2a, i2a, t2b, i2b, t2c, i2c, fig1_b64, fig2_b64) -> str:
    tableA = df_to_scrollable_table(t2a, "stage2A", numeric_cols=["n_cells"], max_height_px=240)
    tableB = df_to_scrollable_table(t2b, "stage2B", numeric_cols=["n_cells"], max_height_px=260)
    tableC = df_to_scrollable_table(
        t2c, "stage2C",
        drop_cols=["mean_mixscape_score"],
        numeric_cols=["n_cells", "responder_rate"],
        percent_cols=["responder_rate"],
        max_height_px=260,
    )

    fig_html = ""
    if fig1_b64 or fig2_b64:
        f1 = f"""
        <div class="fig-card">
          <img class="figure-img" src="data:image/png;base64,{fig1_b64}">
          <div class="caption">Figure 2A. Mixscape perturbation score distribution for a representative perturbation.</div>
        </div>
        """.strip() if fig1_b64 else ""

        f2 = f"""
        <div class="fig-card">
          <img class="figure-img" src="data:image/png;base64,{fig2_b64}">
          <div class="caption">Figure 2B. Mixscape class separation (KO vs NP vs control) for a representative perturbation.</div>
        </div>
        """.strip() if fig2_b64 else ""

        fig_html = f"""
        <div class="subsection-title">Representative Mixscape Visualizations</div>
        <div class="fig-grid">{f1}{f2}</div>
        """.strip()

    return f"""
<!-- STAGE:STAGE2 -->
<div class="stage-container">

  <div class="stage-header" style="background: linear-gradient(90deg,#0b7285,#22b8cf);">
      Stage 2: Mixscape Responder Deconvolution
  </div>

  <div class="stage-content">

    <p class="stage-description">
    Stage 2 applies <b>Pertpy Mixscape</b> to separate <b>responders (KO-like)</b> from <b>non-responders (NP)</b> within each
    perturbation group, providing responder-aware labels for downstream consolidation and inference.
    </p>

    <div class="panel-grid">

      <div class="panel-card">
        <div class="panel-title">Condition Class Distribution</div>
        {tableA}
        <div class="caption">Table 2A. Global condition class composition.</div>
        <div class="interpretation">{i2a}</div>
      </div>

      <div class="panel-card">
        <div class="panel-title">KO / NP Counts (Top 50 rows)</div>
        {tableB}
        <div class="caption">Table 2B. Responder and non-responder counts per perturbation (Top 50 rows).</div>
        <div class="interpretation">{i2b}</div>
      </div>

      <div class="panel-card">
        <div class="panel-title">Responder Summary (Top 50)</div>
        {tableC}
        <div class="caption">Table 2C. Responder rate per perturbation (Top 50 rows).</div>
        <div class="interpretation">{i2c}</div>
      </div>

    </div>

    {fig_html}

  </div>
</div>
<!-- END:STAGE2 -->
""".strip()


def build(sample_dir: Path, client=None, html: str = "") -> str:
    """Build Stage 2 block from pipeline outputs in *sample_dir*."""
    t_cond = first_match(sample_dir, "tables/stage2_condition_class_summary.tsv")
    t_counts = first_match(sample_dir, "tables/stage2_mixscape_class_counts.tsv")
    t_summary = first_match(sample_dir, "tables/stage2_mixscape_summary.tsv")

    if t_cond is None and t_counts is None and t_summary is None:
        return html

    t2a = pd.read_csv(t_cond, sep="\t") if t_cond else pd.DataFrame()
    t2b = (pd.read_csv(t_counts, sep="\t").head(50)) if t_counts else pd.DataFrame()
    t2c = (pd.read_csv(t_summary, sep="\t").head(50)) if t_summary else pd.DataFrame()

    if "mean_mixscape_score" in t2c.columns:
        t2c = t2c.drop(columns=["mean_mixscape_score"], errors="ignore")

    i2a = llm_interpret_table(client, "Stage 2 Condition Class Distribution", t2a,
                              "Explain how class composition impacts responder inference.")
    i2b = llm_interpret_table(client, "Stage 2 KO/NP Counts (Top 50)", t2b,
                              "Interpret KO vs NP counts as evidence of responder separation across perturbations.")
    i2c = llm_interpret_table(client, "Stage 2 Responder Summary (Top 50)", t2c,
                              "Focus on responder_rate patterns and implications for perturbation efficacy.")

    fig_perturb = _find_representative_figure(sample_dir, "mixscape_perturbscore")
    fig_violin = _find_representative_figure(sample_dir, "mixscape_violin")

    fig1 = encode_image_to_base64(fig_perturb) if fig_perturb else None
    fig2 = encode_image_to_base64(fig_violin) if fig_violin else None

    block = build_stage2_block(t2a, i2a, t2b, i2b, t2c, i2c, fig1, fig2)
    return insert_or_replace_block(html, "STAGE2", block)


if __name__ == "__main__":
    import sys
    d = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    out = d / "CRISPRModel_Simulator_Report.html"
    result = build(d, html=ensure_base_report(str(out)))
    write_report(out, result)
    print("Done: Stage 2")
