#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Report builder: Stage 5 — Machine Learning Data Export.

Dynamically discovers the stage5 metadata table from the sample output directory.
"""

from pathlib import Path

import pandas as pd

from .report_common import (
    insert_or_replace_block, ensure_base_report, write_report,
    llm_interpret_table, df_to_scrollable_table,
    first_match,
)


def build_stage5_block(meta_df: pd.DataFrame, interp: str) -> str:
    meta_top = meta_df.head(50).copy()

    table_html = df_to_scrollable_table(
        meta_top, table_id="stage5_cells_meta",
        max_rows=50, max_height_px=320,
    )

    class_counts = (
        meta_df.groupby("condition_class", dropna=False)
        .size().reset_index(name="n_cells")
        .sort_values("n_cells", ascending=False)
    )
    class_table = df_to_scrollable_table(
        class_counts, table_id="stage5_class_counts",
        numeric_cols=["n_cells"], max_height_px=180,
    )

    pert_counts = (
        meta_df.groupby("perturbation_id", dropna=False)
        .size().reset_index(name="n_cells")
        .sort_values("n_cells", ascending=False)
        .head(25)
    )
    pert_table = df_to_scrollable_table(
        pert_counts, table_id="stage5_pert_counts",
        numeric_cols=["n_cells"], max_height_px=220,
    )

    return f"""
<!-- STAGE:STAGE5 -->
<div class="stage-container">

  <div class="stage-header" style="background: linear-gradient(90deg,#1f2937,#4b5563);">
      Stage 5: Machine Learning Data Export
  </div>

  <div class="stage-content">

    <p class="stage-description">
    This stage converts the consolidated single-cell dataset into standardized, model-ready tables.
    Each cell is paired with its <b>perturbation label</b>, <b>condition class</b>, and a numeric
    <b>training label</b> to support downstream model training and evaluation.
    </p>

    <div class="panel-grid">

      <div class="panel-card">
        <div class="panel-title">Condition Class Composition</div>
        {class_table}
        <div class="caption">Table 5A. Distribution of exported cells by condition class.</div>
      </div>

      <div class="panel-card">
        <div class="panel-title">Most Frequent Perturbations (Top 25)</div>
        {pert_table}
        <div class="caption">Table 5B. Largest perturbation groups retained for model training.</div>
      </div>

      <div class="panel-card">
        <div class="panel-title">Example Exported Cell Metadata (Top 50 rows)</div>
        {table_html}
        <div class="caption">Table 5C. Cell-level metadata export (subset view).</div>
      </div>

    </div>

    <div class="interpretation">{interp}</div>

  </div>
</div>
<!-- END:STAGE5 -->
""".strip()


def build(sample_dir: Path, client=None, html: str = "") -> str:
    """Build Stage 5 block from pipeline outputs in *sample_dir*."""
    meta_path = first_match(sample_dir, "stage5_cells_metadata.tsv")
    if meta_path is None:
        meta_path = first_match(sample_dir, "**/stage5_cells_metadata.tsv")
    if meta_path is None:
        return html

    meta = pd.read_csv(meta_path, sep="\t")

    mini = (
        meta.groupby(["condition_class"], dropna=False)
        .size().reset_index(name="n_cells")
        .sort_values("n_cells", ascending=False)
    )
    interp = llm_interpret_table(
        client, "Stage 5 ML Export Summary", mini,
        "Explain export readiness for modeling, class composition, and likely effects of class imbalance."
    )

    block = build_stage5_block(meta, interp)
    return insert_or_replace_block(html, "STAGE5", block)


if __name__ == "__main__":
    import sys
    d = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    out = d / "CRISPRModel_Simulator_Report.html"
    result = build(d, html=ensure_base_report(str(out)))
    write_report(out, result)
    print("Done: Stage 5")
