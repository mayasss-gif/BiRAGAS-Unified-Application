#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Report builder: Stage 6 — Model Training.

Dynamically discovers metrics tables and figures from the sample output directory.
"""

from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .report_common import (
    insert_or_replace_block, ensure_base_report, write_report,
    llm_interpret_table, df_to_scrollable_table,
    encode_image_to_base64, first_match,
)


def make_metrics_plot(val_df: pd.DataFrame, test_df: pd.DataFrame, out_png: Path):
    """Simple bar plot comparing models on key metrics (val vs test)."""
    metrics = ["accuracy", "balanced_accuracy", "f1_macro", "f1_weighted", "top5_acc"]

    v = val_df.set_index("model")
    t = test_df.set_index("model")

    common_models = [m for m in v.index if m in t.index]
    if not common_models:
        return

    labels = []
    heights = []
    for m in common_models:
        for split_name, df in [("val", v), ("test", t)]:
            for met in metrics:
                if met in df.columns:
                    labels.append(f"{m}_{split_name}_{met}")
                    heights.append(float(df.loc[m, met]))

    plt.figure(figsize=(14, 4))
    plt.bar(range(len(heights)), heights)
    plt.xticks(range(len(heights)), labels, rotation=90)
    plt.ylabel("Score")
    plt.title("Stage 6 Model Performance (Validation/Test)")
    plt.tight_layout()
    plt.savefig(str(out_png), dpi=200, bbox_inches="tight")
    plt.close()


def build_stage6_block(val_df, test_df, per_class_df,
                       interp_val, interp_test, interp_class,
                       metrics_plot_b64, prf_b64):
    val_table = df_to_scrollable_table(
        val_df, table_id="stage6_val",
        numeric_cols=["n_cells", "accuracy", "balanced_accuracy", "f1_macro", "f1_weighted", "top5_acc"],
        percent_cols=["accuracy", "balanced_accuracy", "f1_macro", "f1_weighted", "top5_acc"],
        max_height_px=220,
    )

    test_table = df_to_scrollable_table(
        test_df, table_id="stage6_test",
        numeric_cols=["n_cells", "accuracy", "balanced_accuracy", "f1_macro", "f1_weighted", "top5_acc"],
        percent_cols=["accuracy", "balanced_accuracy", "f1_macro", "f1_weighted", "top5_acc"],
        max_height_px=220,
    )

    per_class_top = per_class_df.head(50).copy() if per_class_df is not None else pd.DataFrame()
    class_table = df_to_scrollable_table(
        per_class_top, table_id="stage6_per_class",
        numeric_cols=["precision", "recall", "f1", "support"],
        percent_cols=["precision", "recall", "f1"],
        max_height_px=300,
    )

    figs = ""
    if metrics_plot_b64:
        figs += f"""
        <div class="fig-card">
          <img class="figure-img" src="data:image/png;base64,{metrics_plot_b64}">
          <div class="caption">Figure 6A. Validation vs test performance across models and key metrics.</div>
        </div>
        """.strip()

    if prf_b64:
        figs += f"""
        <div class="fig-card">
          <img class="figure-img" src="data:image/png;base64,{prf_b64}">
          <div class="caption">Figure 6B. Precision/Recall/F1 barplot (test set) summarizing overall model behavior.</div>
        </div>
        """.strip()

    return f"""
<!-- STAGE:STAGE6 -->
<div class="stage-container">

  <div class="stage-header" style="background: linear-gradient(90deg,#0f766e,#14b8a6);">
      Stage 6: Model Training
  </div>

  <div class="stage-content">

    <p class="stage-description">
    Stage 6 trains classification models to predict perturbation identities from learned expression signatures.
    Performance is evaluated on validation and test splits using accuracy, balanced accuracy, macro/weighted F1,
    and top-5 accuracy to quantify both overall correctness and robustness under class imbalance.
    </p>

    <div class="panel-grid">

      <div class="panel-card">
        <div class="panel-title">Validation Metrics</div>
        {val_table}
        <div class="caption">Table 6A. Model performance on the validation split.</div>
        <div class="interpretation">{interp_val}</div>
      </div>

      <div class="panel-card">
        <div class="panel-title">Test Metrics</div>
        {test_table}
        <div class="caption">Table 6B. Model performance on the held-out test split.</div>
        <div class="interpretation">{interp_test}</div>
      </div>

      <div class="panel-card">
        <div class="panel-title">Per-Class Report (Top 50 rows)</div>
        {class_table}
        <div class="caption">Table 6C. Per-class precision/recall/F1 for top entries (test set view).</div>
        <div class="interpretation">{interp_class}</div>
      </div>

    </div>

    <div class="subsection-title">Model Diagnostics</div>
    <div class="fig-grid">
      {figs}
    </div>

  </div>
</div>
<!-- END:STAGE6 -->
""".strip()


def build(sample_dir: Path, client=None, html: str = "") -> str:
    """Build Stage 6 block from pipeline outputs in *sample_dir*."""
    val_path = first_match(sample_dir, "stage6_metrics_val.tsv")
    test_path = first_match(sample_dir, "stage6_metrics_test.tsv")

    if val_path is None and test_path is None:
        return html

    val_df = pd.read_csv(val_path, sep="\t") if val_path else pd.DataFrame()
    test_df = pd.read_csv(test_path, sep="\t") if test_path else pd.DataFrame()

    per_class_path = first_match(sample_dir, "stage6_per_class_report_test.tsv")
    per_class_df = pd.read_csv(per_class_path, sep="\t") if per_class_path else None

    auto_png = sample_dir / "stage6_metrics_comparison.png"
    if not val_df.empty and not test_df.empty:
        try:
            make_metrics_plot(val_df, test_df, auto_png)
        except Exception:
            pass

    metrics_plot_b64 = encode_image_to_base64(auto_png)
    prf_path = first_match(sample_dir, "stage6_prf_barplot_test.png")
    prf_b64 = encode_image_to_base64(prf_path) if prf_path else None

    interp_val = llm_interpret_table(
        client, "Stage 6 Validation Metrics", val_df,
        "Compare models, explain imbalance implications, and interpret top-5 accuracy vs top-1 accuracy."
    )
    interp_test = llm_interpret_table(
        client, "Stage 6 Test Metrics", test_df,
        "Summarize generalization and whether performance shifts indicate overfitting or stability."
    )
    interp_class = llm_interpret_table(
        client, "Stage 6 Per-Class Performance (Top)",
        per_class_df.head(30) if per_class_df is not None else pd.DataFrame(),
        "Explain which perturbations are easiest/hardest to classify and what precision/recall patterns imply."
    )

    block = build_stage6_block(
        val_df, test_df, per_class_df,
        interp_val, interp_test, interp_class,
        metrics_plot_b64, prf_b64,
    )
    return insert_or_replace_block(html, "STAGE6", block)


if __name__ == "__main__":
    import sys
    d = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    out = d / "CRISPRModel_Simulator_Report.html"
    result = build(d, html=ensure_base_report(str(out)))
    write_report(out, result)
    print("Done: Stage 6")
