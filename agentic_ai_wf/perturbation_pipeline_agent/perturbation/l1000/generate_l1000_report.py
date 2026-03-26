#!/usr/bin/env python
import os
import argparse
import base64
import mimetypes
from textwrap import dedent

import pandas as pd


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def image_to_data_uri(path: str) -> str:
    """
    Convert an image file to a base64 data URI.
    Returns empty string if file does not exist.
    """
    if not path:
        return ""
    if not os.path.exists(path):
        return ""
    mime, _ = mimetypes.guess_type(path)
    if mime is None:
        mime = "image/png"
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode("ascii")
    return f"data:{mime};base64,{data}"


def html_escape(text: str) -> str:
    if text is None:
        return ""
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def load_latest_run(l1000_root: str) -> str:
    """
    Pick the most recent L1000 run directory under l1000_root.
    Expects folders like L1000_Run_YYYYMMDD_HHMMSS
    """
    if l1000_root is None:
        raise ValueError("l1000_root must be provided.")

    if not os.path.isdir(l1000_root):
        raise FileNotFoundError(f"l1000_root not found: {l1000_root}")

    candidates = [
        os.path.join(l1000_root, d)
        for d in os.listdir(l1000_root)
        if d.startswith("L1000_Run_") and os.path.isdir(os.path.join(l1000_root, d))
    ]
    if not candidates:
        raise RuntimeError(f"No L1000 run directories found under {l1000_root}")

    candidates.sort(key=lambda p: os.path.getmtime(p))
    return candidates[-1]


def read_disease_description(path: str) -> str:
    """
    Read disease / primary site description from userinput.txt (preferred).

    Behavior:
    - If file contains a line starting with 'Disease:' (case-insensitive), use that value.
    - Else if contains 'Primary site:', use that value.
    - Else fallback: whole file content (legacy).
    """
    default_msg = "Disease context not specified"

    if not path or not os.path.exists(path):
        return default_msg

    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f.readlines()]

    for ln in lines:
        if ln.lower().startswith("disease:"):
            value = ln.split(":", 1)[1].strip() if ":" in ln else ""
            return value or default_msg

    for ln in lines:
        if ln.lower().startswith("primary site:"):
            value = ln.split(":", 1)[1].strip() if ":" in ln else ""
            return value or default_msg

    content = "\n".join(lines).strip()
    return content if content else default_msg


def df_to_html_table(
    df: pd.DataFrame,
    table_id: str,
    max_rows: int = None,
    show_note: bool = False,
    note_text: str = ""
) -> str:
    """
    Convert a DataFrame to a scrollable HTML table with a search box.
    """
    if max_rows is not None and len(df) > max_rows:
        df_display = df.head(max_rows)
        note = note_text or f"Showing first {max_rows} of {len(df):,} rows."
    else:
        df_display = df
        note = ""

    cols = list(df_display.columns)
    header_html = "".join(f"<th>{html_escape(c)}</th>" for c in cols)

    rows_html_parts = []
    for _, row in df_display.iterrows():
        cells = "".join(f"<td>{html_escape(row[c])}</td>" for c in cols)
        rows_html_parts.append(f"<tr>{cells}</tr>")
    rows_html = "\n".join(rows_html_parts)

    note_html = (
        f'<div class="table-note">{html_escape(note)}</div>'
        if note and show_note
        else ""
    )

    return f"""
    <div class="table-wrapper">
      <div class="table-search-row">
        <input type="text" class="table-search"
               placeholder="Search this table..."
               data-table-target="{table_id}">
      </div>
      <div class="table-container">
        <div class="table-scroll">
          <table id="{table_id}">
            <thead>
              <tr>{header_html}</tr>
            </thead>
            <tbody>
              {rows_html}
            </tbody>
          </table>
        </div>
      </div>
      {note_html}
    </div>
    """


def make_figure_block(img_path: str, title: str, subtitle: str = "", wide: bool = False) -> str:
    """
    Build a figure block; embed image as base64 if present,
    otherwise show a 'file not found' message.
    """
    data_uri = image_to_data_uri(img_path)
    figure_class = "figure-block figure-block-wide" if wide else "figure-block"

    if data_uri:
        img_html = f"""
        <a href="{data_uri}" target="_blank" class="figure-img-link">
          <img src="{data_uri}" alt="{html_escape(subtitle or title)}" class="figure-img-full">
        </a>
        """
        missing_html = ""
    else:
        img_html = ""
        missing_html = f'<div class="figure-missing">Image file not found: {html_escape(img_path)}</div>'

    subtitle_html = f'<div class="figure-subtitle">{html_escape(subtitle)}</div>' if subtitle else ""

    return f"""
    <div class="{figure_class}">
      <div class="figure-title">{html_escape(title)}</div>
      {subtitle_html}
      {img_html}
      {missing_html}
    </div>
    """


def make_panel_grid_from_files(
    file_paths,
    title: str = "",
    cols: int = 5,
    caption_mode: str = "stem",
    max_items: int = None,
) -> str:
    """
    Make a 5x5-style panel grid (cols=5) using base64-embedded images.
    """
    paths = list(file_paths)
    if max_items is not None:
        paths = paths[:max_items]

    if not paths:
        return ""

    def caption_for(p: str) -> str:
        base = os.path.splitext(os.path.basename(p))[0]
        if caption_mode == "gene_drug" and "__" in base:
            g, d = base.split("__", 1)
            return f"{g} • {d}"
        return base

    tiles = []
    for p in paths:
        data_uri = image_to_data_uri(p)
        if not data_uri:
            continue
        cap = caption_for(p)
        tiles.append(f"""
          <figure class="panel-tile">
            <a href="{data_uri}" target="_blank" class="figure-img-link">
              <img src="{data_uri}" alt="{html_escape(cap)}" class="panel-img">
            </a>
            <figcaption class="panel-caption">{html_escape(cap)}</figcaption>
          </figure>
        """)

    if not tiles:
        return ""

    title_html = f'<h4 class="subsection-title">{html_escape(title)}</h4>' if title else ""
    return f"""
      {title_html}
      <div class="panel-grid" style="grid-template-columns: repeat({cols}, minmax(0, 1fr));">
        {''.join(tiles)}
      </div>
    """


def collect_dose_curves_for_table(df: pd.DataFrame, dose_curve_dir: str, top_n: int = 25):
    """
    Match dose curve files for the top rows in a table.
    Expected filenames: GENE__DRUG.png (or other image extension)
    """
    if df is None or df.empty:
        return []
    if not os.path.isdir(dose_curve_dir):
        return []

    existing = {}
    for fn in os.listdir(dose_curve_dir):
        if not fn.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
            continue
        stem = os.path.splitext(fn)[0]
        existing[stem] = os.path.join(dose_curve_dir, fn)

    out = []
    for _, r in df.head(top_n).iterrows():
        gene = str(r.get("Gene", "")).strip()
        drug = str(r.get("Drug", "")).strip()
        if not gene or not drug:
            continue
        stem = f"{gene}__{drug}"
        if stem in existing:
            out.append(existing[stem])
    return out


def parse_latest_gene_presence_from_logs(run_dir: str) -> str:
    """
    Look in {run_dir}/logs for the latest line like:
    "Genes requested: 1328 | present: 60 | missing: 1268"
    Returns a short string for display, or "" if not found.
    """
    logs_dir = os.path.join(run_dir, "logs")
    if not os.path.isdir(logs_dir):
        return ""

    candidates = []
    for fn in os.listdir(logs_dir):
        p = os.path.join(logs_dir, fn)
        if os.path.isfile(p):
            candidates.append(p)
    if not candidates:
        return ""

    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)

    # Scan newest-first and keep the last match within each file; return first file that contains it.
    for path in candidates:
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                matches = []
                for ln in f:
                    if "Genes requested:" in ln and "| present:" in ln and "| missing:" in ln:
                        matches.append(ln.strip())
                if matches:
                    # Use the last occurrence within the newest relevant log file
                    return matches[-1]
        except Exception:
            continue

    return ""


# ---------------------------------------------------------------------
# Main report builder
# ---------------------------------------------------------------------

def build_report(
    run_dir: str,
    disease_path: str,
    out_html: str,
    logo_left_path: str = None,
    logo_right_path: str = None,
    style: str = "full",
):
    disease_desc = read_disease_description(disease_path)

    left_logo_data = image_to_data_uri(logo_left_path) if logo_left_path else ""
    right_logo_data = image_to_data_uri(logo_right_path) if logo_right_path else ""

    left_logo_html = f'<img src="{left_logo_data}" alt="Ayass Bioscience" class="logo-img">' if left_logo_data else ""
    right_logo_html = f'<img src="{right_logo_data}" alt="Ayass Bioscience" class="logo-img">' if right_logo_data else ""

    causal_path = os.path.join(run_dir, "causal_link_table_with_relevance.csv")
    qc_path = os.path.join(run_dir, "causal_link_table_qc_with_relevance.csv")
    cos_cluster_path = os.path.join(run_dir, "drug_clusters_cosine.csv")
    pear_cluster_path = os.path.join(run_dir, "drug_clusters_pearson.csv")
    spear_cluster_path = os.path.join(run_dir, "drug_clusters_spearman.csv")
    kend_cluster_path = os.path.join(run_dir, "drug_clusters_kendall.csv")

    if not os.path.exists(causal_path):
        raise FileNotFoundError(f"Missing causal link table: {causal_path}")
    if not os.path.exists(qc_path):
        raise FileNotFoundError(f"Missing QC table: {qc_path}")

    causal_df = pd.read_csv(causal_path)
    qc_df = pd.read_csv(qc_path)

    # -----------------------------
    # Dynamic overview metrics (from QC file)
    # -----------------------------
    overview_df = qc_df.copy()

    n_links = int(len(overview_df))
    n_genes = int(overview_df["Gene"].nunique()) if "Gene" in overview_df.columns else 0
    n_drugs = int(overview_df["Drug"].nunique()) if "Drug" in overview_df.columns else 0

    rev = aggr = 0
    if "Therapeutic_Relevance" in overview_df.columns:
        tr = overview_df["Therapeutic_Relevance"].astype(str).str.strip().str.lower()
        rev = int((tr == "reversal").sum())
        aggr = int((tr == "aggravating").sum())

    gene_presence_logline = parse_latest_gene_presence_from_logs(run_dir)

    if {"Gene_Main_Class", "Gene_Sub_Class", "Gene_DEG_Trend", "Sensitivity Index"}.issubset(causal_df.columns):
        causal_filtered = causal_df[
            (causal_df["Gene_Main_Class"] == "Unknown")
            & (causal_df["Gene_Sub_Class"] == "Unknown")
            & (causal_df["Gene_DEG_Trend"] == "DOWN")
        ].copy()
        causal_filtered = causal_filtered.sort_values("Sensitivity Index", ascending=False)
    else:
        causal_filtered = causal_df.copy()

    # Table 1: remove only these columns
    drop_t1_cols = ["Gene_Main_Class", "Gene_Sub_Class", "Gene_Pathway", "Gene_Regulation"]
    causal_filtered_t1 = causal_filtered.drop(columns=[c for c in drop_t1_cols if c in causal_filtered.columns], errors="ignore")

    if "SI_clamped" in qc_df.columns:
        qc_sorted = qc_df.sort_values("SI_clamped", ascending=False).copy()
    else:
        qc_sorted = qc_df.copy()

    # Tables
    causal_table_html = df_to_html_table(
        causal_filtered_t1,
        table_id="causal_table",
        max_rows=120,
        show_note=False,   # remove the "Showing first ..." line for Table 1
        note_text="",
    )
    qc_table_html = df_to_html_table(
        qc_sorted,
        table_id="qc_table",
        max_rows=120,
        show_note=True,
        note_text=f"Showing first 120 of {len(qc_sorted):,} QC-adjusted gene–drug sensitivity entries.",
    )

    # Dose panels (full width BELOW both tables)
    fig_dir = os.path.join(run_dir, "figures")
    dose_curve_dir = os.path.join(fig_dir, "dose_curves")

    dose_panel_t1_paths = collect_dose_curves_for_table(causal_filtered, dose_curve_dir, top_n=25)
    dose_panel_t2_paths = collect_dose_curves_for_table(qc_sorted, dose_curve_dir, top_n=25)

    dose_panel_t1_html = make_panel_grid_from_files(
        dose_panel_t1_paths,
        title="Drug dose–response panels (Table 1)",
        cols=5,
        caption_mode="gene_drug",
        max_items=25,
    )
    dose_panel_t2_html = make_panel_grid_from_files(
        dose_panel_t2_paths,
        title="Drug dose–response panels (Table 2)",
        cols=5,
        caption_mode="gene_drug",
        max_items=25,
    )

    # Cluster tables
    def safe_read_csv(path):
        return pd.read_csv(path) if os.path.exists(path) else None

    cos_df = safe_read_csv(cos_cluster_path)
    pear_df = safe_read_csv(pear_cluster_path)
    spear_df = safe_read_csv(spear_cluster_path)
    kend_df = safe_read_csv(kend_cluster_path)

    cluster_tables_html = ""
    if cos_df is not None:
        cluster_tables_html += f"""
        <h4 class="subsection-title">Drug clusters (Cosine)</h4>
        {df_to_html_table(cos_df, "cluster_cosine", max_rows=None, show_note=False)}
        """
    if pear_df is not None:
        cluster_tables_html += f"""
        <h4 class="subsection-title">Drug clusters (Pearson)</h4>
        {df_to_html_table(pear_df, "cluster_pearson", max_rows=None, show_note=False)}
        """
    if spear_df is not None:
        cluster_tables_html += f"""
        <h4 class="subsection-title">Drug clusters (Spearman)</h4>
        {df_to_html_table(spear_df, "cluster_spearman", max_rows=None, show_note=False)}
        """
    if kend_df is not None:
        cluster_tables_html += f"""
        <h4 class="subsection-title">Drug clusters (Kendall Tau)</h4>
        {df_to_html_table(kend_df, "cluster_kendall", max_rows=None, show_note=False)}
        """

    # Figures
    volcano_fig = make_figure_block(os.path.join(fig_dir, "top_hits_ATE_volcano.png"), "FIGURE T1",
                                   "Therapeutic targets – volcano plot (ATE vs FDR)", wide=True)
    top_hits_bar_fig = make_figure_block(os.path.join(fig_dir, "top_hits_barplot.png"), "FIGURE T2",
                                         "Top-ranked therapeutic targets by effect size", wide=False)
    ec50_si_fig = make_figure_block(os.path.join(fig_dir, "top_hits_ec50_vs_si.png"), "FIGURE T3",
                                    "EC50 versus sensitivity index for key causal gene–drug links", wide=False)
    qc_corr_fig = make_figure_block(os.path.join(fig_dir, "N1_causal_readouts_correlation.png"), "FIGURE Q1",
                                    "Causal model concordance across N1 readouts", wide=True)
    def latest_png_with_prefix(prefix: str):
        if not os.path.isdir(fig_dir):
            return None
        candidates = []
        for fn in os.listdir(fig_dir):
            if not fn.lower().endswith(".png"):
                continue
            if not fn.startswith(prefix):
                continue
            candidates.append(os.path.join(fig_dir, fn))
        if not candidates:
            return None
        candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        return candidates[0]

    itime_path = latest_png_with_prefix("36_fig_pert_itime_distribution_")
    itime_fig = make_figure_block(itime_path or "", "FIGURE Q2",
                                  "Distribution of perturbation incubation times", wide=False)

    cluster_bar_cosine = make_figure_block(os.path.join(fig_dir, "drug_cluster_bar_cosine.png"), "FIGURE C1",
                                          "Drug cluster sizes – Cosine similarity", wide=False)
    cluster_bar_pearson = make_figure_block(os.path.join(fig_dir, "drug_cluster_bar_pearson.png"), "FIGURE P1",
                                           "Drug cluster sizes – Pearson similarity", wide=False)
    cluster_bar_spearman = make_figure_block(os.path.join(fig_dir, "drug_cluster_bar_spearman.png"), "FIGURE S1",
                                            "Drug cluster sizes – Spearman similarity", wide=False)
    cluster_bar_kendall = make_figure_block(os.path.join(fig_dir, "drug_cluster_bar_kendall.png"), "FIGURE K1",
                                           "Drug cluster sizes – Kendall Tau similarity", wide=False)

    sim_heat_cosine = make_figure_block(os.path.join(fig_dir, "drug_similarity_heatmap_cosine.png"), "FIGURE C2",
                                        "Drug–drug similarity heatmap – Cosine", wide=True)
    sim_heat_pearson = make_figure_block(os.path.join(fig_dir, "drug_similarity_heatmap_pearson.png"), "FIGURE P2",
                                         "Drug–drug similarity heatmap – Pearson", wide=True)
    sim_heat_spearman = make_figure_block(os.path.join(fig_dir, "drug_similarity_heatmap_spearman.png"), "FIGURE S2",
                                          "Drug–drug similarity heatmap – Spearman", wide=True)
    sim_heat_kendall = make_figure_block(os.path.join(fig_dir, "drug_similarity_heatmap_kendall.png"), "FIGURE K2",
                                         "Drug–drug similarity heatmap – Kendall Tau", wide=True)

    def first_png_in(subfolder_name):
        subdir = os.path.join(fig_dir, subfolder_name)
        if not os.path.isdir(subdir):
            return None
        pngs = [f for f in os.listdir(subdir) if f.lower().endswith(".png")]
        if not pngs:
            return None
        pngs.sort()
        return os.path.join(subdir, pngs[0])

    reversal_heatmap_png = first_png_in("gene_drug_heatmaps")
    reversal_bar_png = first_png_in("reversal_barplots_from_summary")
    reversal_multi_png = first_png_in("reversal_multi_drug_from_summary")

    reversal_heatmap_fig = make_figure_block(reversal_heatmap_png or "", "FIGURE R1",
                                             "Gene–drug reversal heatmap (sensitivity index)", wide=True)
    reversal_bar_fig = make_figure_block(reversal_bar_png or "", "FIGURE R2",
                                         "Top reversal interactions per gene", wide=False)
    reversal_multi_fig = make_figure_block(reversal_multi_png or "", "FIGURE R3",
                                           "Multi-drug reversal combinations", wide=True)

    reversal_barplots_dir = os.path.join(fig_dir, "reversal_barplots_from_summary")
    reversal_multi_dir = os.path.join(fig_dir, "reversal_multi_drug_from_summary")

    reversal_barplot_files = []
    if os.path.isdir(reversal_barplots_dir):
        reversal_barplot_files = sorted(
            os.path.join(reversal_barplots_dir, f)
            for f in os.listdir(reversal_barplots_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))
        )

    reversal_multi_files = []
    if os.path.isdir(reversal_multi_dir):
        reversal_multi_files = sorted(
            os.path.join(reversal_multi_dir, f)
            for f in os.listdir(reversal_multi_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))
        )

    reversal_barplots_panel_html = make_panel_grid_from_files(
        reversal_barplot_files,
        title="Reversal barplots (per gene)",
        cols=5,
        caption_mode="stem",
        max_items=None,
    )
    reversal_multi_panel_html = make_panel_grid_from_files(
        reversal_multi_files,
        title="Multi-drug reversal combinations (per gene)",
        cols=5,
        caption_mode="stem",
        max_items=None,
    )

    top_hits_figures_html = f"""
    <div class="figure-grid-2">
      {volcano_fig}
    </div>
    <div class="figure-grid-2">
      {top_hits_bar_fig}
      {ec50_si_fig}
    </div>
    <div class="figure-grid-2">
      {qc_corr_fig}
      {itime_fig}
    </div>
    """

    cluster_figures_html = f"""
    <div class="figure-grid-2">
      {cluster_bar_cosine}
      {cluster_bar_pearson}
    </div>
    <div class="figure-grid-2">
      {cluster_bar_spearman}
      {cluster_bar_kendall}
    </div>
    <div class="figure-grid-1">
      {sim_heat_cosine}
      {sim_heat_pearson}
      {sim_heat_spearman}
      {sim_heat_kendall}
    </div>
    """

    reversal_figures_html = f"""
    <div class="figure-grid-1">
      {reversal_heatmap_fig}
    </div>
    <div class="figure-grid-2">
      {reversal_bar_fig}
      {reversal_multi_fig}
    </div>
    """

    # CSS + JS
    css = dedent("""
    <style>
      :root {
        --bg-main: #f5f7fb;
        --card-bg: #ffffff;
        --accent: #0f766e;
        --accent-soft: #14b8a6;
        --text-main: #111827;
        --text-muted: #6b7280;
        --border-subtle: #e5e7eb;
        --table-stripe: #f9fafb;
      }
      * { box-sizing: border-box; }
      body {
        margin: 0; padding: 0;
        font-family: system-ui, -apple-system, BlinkMacSystemFont, "SF Pro Text","Segoe UI", sans-serif;
        background: radial-gradient(circle at top left, #e0f2fe 0, #f5f7fb 40%, #f5f7fb 100%);
        color: var(--text-main);
      }
      .page { max-width: 1200px; margin: 24px auto 48px auto; padding: 0 16px 32px 16px; }
      .report-card {
        background: var(--card-bg);
        border-radius: 20px;
        box-shadow: 0 18px 40px rgba(15, 23, 42, 0.12), 0 0 0 1px rgba(148, 163, 184, 0.25);
        overflow: hidden;
      }
      .report-header {
        background: linear-gradient(120deg, #0f766e, #14b8a6, #22c55e);
        color: #ecfeff;
        padding: 20px 24px;
        display: flex;
        align-items: center;
        justify-content: space-between;
      }
      .header-left { display: flex; align-items: center; gap: 16px; }
      .logo-img { height: 44px; width: auto; border-radius: 10px; background: rgba(15, 23, 42, 0.2); padding: 4px 8px; }
      .report-title-block { display: flex; flex-direction: column; gap: 4px; }
      .report-kicker { font-size: 11px; letter-spacing: 0.16em; text-transform: uppercase; opacity: 0.85; }
      .report-title { font-size: 22px; font-weight: 650; }
      .report-subtitle { font-size: 13px; opacity: 0.9; }
      .header-tags { display: flex; flex-direction: column; align-items: flex-end; gap: 6px; }
      .chip { display: inline-flex; align-items: center; gap: 6px; padding: 4px 10px; border-radius: 999px; background: rgba(15, 23, 42, 0.25); font-size: 11px; }
      .chip-label { font-weight: 500; }
      .chip-value { opacity: 0.9; }

      .report-body { padding: 20px 24px 24px 24px; }

      .section { margin-bottom: 28px; padding-bottom: 20px; border-bottom: 1px solid var(--border-subtle); }
      .section:last-of-type { border-bottom: none; padding-bottom: 0; }
      .section-header { display: flex; flex-direction: column; gap: 4px; margin-bottom: 8px; }
      .section-title { font-size: 16px; font-weight: 650; letter-spacing: 0.06em; text-transform: uppercase; color: #0f172a; }
      .section-kicker { font-size: 13px; color: var(--text-muted); }
      .section-body p { font-size: 13px; line-height: 1.6; color: var(--text-main); margin: 6px 0; }

      .split-row { display: flex; flex-wrap: wrap; gap: 16px; margin-top: 12px; }
      .split-col { flex: 1 1 0; min-width: 260px; }

      .subsection-title { margin: 18px 0 4px 0; font-size: 13px; font-weight: 600; color: var(--text-main); }

      .table-wrapper { margin-top: 6px; }
      .table-search-row { margin-bottom: 4px; }
      .table-search { width: 100%; padding: 6px 8px; font-size: 12px; border-radius: 999px; border: 1px solid var(--border-subtle); outline: none; }
      .table-container { border-radius: 12px; border: 1px solid var(--border-subtle); background: #fff; overflow: hidden; }
      .table-scroll { max-height: 340px; overflow-y: auto; overflow-x: auto; }
      table { border-collapse: collapse; width: 100%; min-width: 720px; font-size: 11px; }
      thead { position: sticky; top: 0; background: #e5f3ff; z-index: 1; }
      th, td { padding: 6px 8px; border-bottom: 1px solid #e5e7eb; text-align: left; white-space: nowrap; }
      th { font-weight: 600; color: #0f172a; border-bottom: 1px solid #cbd5e1; }
      tbody tr:nth-child(odd) { background-color: var(--table-stripe); }
      tbody tr:hover { background-color: #e0f2fe; }
      .table-note { margin-top: 4px; font-size: 11px; color: var(--text-muted); }

      .summary-grid { display: flex; flex-wrap: wrap; gap: 10px; margin-top: 8px; }
      .summary-card { flex: 1 1 0; min-width: 160px; padding: 8px 10px; border-radius: 12px; background: #f9fafb; border: 1px solid #e5e7eb; }
      .summary-label { font-size: 11px; color: var(--text-muted); margin-bottom: 2px; }
      .summary-value { font-size: 15px; font-weight: 600; color: #0f172a; }

      .figure-grid-1, .figure-grid-2 { display: grid; gap: 14px; margin-top: 10px; }
      .figure-grid-1 { grid-template-columns: minmax(0, 1fr); }
      .figure-grid-2 { grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); }

      .figure-block { background: #f9fafb; border-radius: 14px; border: 1px solid #e5e7eb; padding: 10px 12px 12px 12px; }
      .figure-block-wide { grid-column: 1 / -1; }
      .figure-title { font-size: 12px; font-weight: 600; color: var(--accent); margin-bottom: 2px; }
      .figure-subtitle { font-size: 12px; font-weight: 500; color: #111827; margin-bottom: 4px; }
      .figure-img-full { display: block; max-width: 100%; height: auto; border-radius: 12px; border: 1px solid #d1d5db; margin-top: 4px; }
      .figure-img-link { text-decoration: none; }
      .figure-missing { font-size: 11px; color: #b91c1c; margin-top: 4px; }

      /* Panel grids (FULL WIDTH) */
      .panel-grid {
        width: 100%;
        display: grid;
        gap: 10px;
        margin-top: 10px;
        align-items: start;
      }
      .panel-tile {
        margin: 0;
        padding: 8px;
        border-radius: 12px;
        border: 1px solid #e5e7eb;
        background: #ffffff;
      }
      .panel-img {
        display: block;
        width: 100%;
        height: auto;
        border-radius: 10px;
        border: 1px solid #d1d5db;
      }
      .panel-caption {
        margin-top: 6px;
        font-size: 11px;
        color: var(--text-muted);
        text-align: center;
        word-break: break-word;
      }

      .end-note {
        margin-top: 14px;
        padding: 10px 12px;
        border-radius: 12px;
        background: #f9fafb;
        border: 1px solid #e5e7eb;
        font-size: 12px;
        color: #111827;
      }

      @media (max-width: 768px) {
        .report-header { flex-direction: column; align-items: flex-start; gap: 10px; }
        .header-tags { align-items: flex-start; }
        .panel-grid { grid-template-columns: repeat(2, minmax(0, 1fr)) !important; }
      }
    </style>
    """)

    js = dedent("""
    <script>
      document.addEventListener("DOMContentLoaded", function () {
        const inputs = document.querySelectorAll(".table-search");
        inputs.forEach(function (inp) {
          inp.addEventListener("input", function () {
            const targetId = this.getAttribute("data-table-target");
            const table = document.getElementById(targetId);
            if (!table) return;
            const filter = this.value.toLowerCase();
            const rows = table.querySelectorAll("tbody tr");
            rows.forEach(function (row) {
              const text = row.textContent.toLowerCase();
              row.style.display = text.indexOf(filter) > -1 ? "" : "none";
            });
          });
        });
      });
    </script>
    """)

    references_html = dedent("""
    <ul class="reference-list">
      <li>
        Subramanian A, Narayan R, Corsello SM, <em>et al.</em>
        A Next Generation Connectivity Map: L1000 Platform and the First 1,000,000 Profiles.
        <strong>Cell</strong>. 2017;171(6):1437-1452.e17.
      </li>
      <li>
        Demichev V, Messner CB, Vernardis SI, <em>et al.</em>
        An integrative proteomics and transcriptomics analysis of breast cancer.
        <strong>Nat Commun</strong>. 2023;14: (see
        <a href="https://pmc.ncbi.nlm.nih.gov/articles/PMC10663987/" target="_blank" rel="noopener noreferrer">
        PMC10663987</a>).
      </li>
    </ul>
    """)

    gene_presence_note_html = ""
    if gene_presence_logline:
        gene_presence_note_html = f"""
        <div class="end-note">
          {html_escape(gene_presence_logline)}
        </div>
        """

    html = f"""
    <!DOCTYPE html>
    <html lang="en">
      <head>
        <meta charset="utf-8">
        <title>Ayass Bioscience · L1000 Functional Transcriptomics Report</title>
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        {css}
      </head>
      <body>
        <div class="page">
          <div class="report-card">
            <header class="report-header">
              <div class="header-left">
                {left_logo_html}
                <div class="report-title-block">
                  <div class="report-kicker">AYASS BIOSCIENCE · FUNCTIONAL TRANSCRIPTOMICS</div>
                  <div class="report-title">Transcriptional responses to drugs and perturbations</div>
                  <div class="report-subtitle">L1000 drug perturbation signatures contextualised for patient-specific gene programs</div>
                </div>
              </div>
              <div class="header-tags">
                <div class="chip">
                  <span class="chip-label">Disease context</span>
                  <span class="chip-value">{html_escape(disease_desc)}</span>
                </div>
                <div class="chip">
                  <span class="chip-label">Platform</span>
                  <span class="chip-value">Drug Perturbation transcriptional profiling</span>
                </div>
                {right_logo_html}
              </div>
            </header>

            <div class="report-body">
              <div class="section">
                <div class="section-header">
                  <div class="section-title">L1000 PERTURBATION OVERVIEW</div>
                  <div class="section-kicker">Causal gene–drug links &amp; similarity modules</div>
                </div>
                <div class="section-body">
                  <p>
                    Causal transcriptional modelling links prioritised genes to dose–response profiles across the small-molecule
                    panel, quantifying EC50, Hill slopes, maximal response and a sensitivity index for each gene–drug pair.
                    Reversal and aggravating interactions highlight compounds that counteract or reinforce disease-associated
                    expression programs.
                  </p>

                  <div class="summary-grid">
                    <div class="summary-card">
                      <div class="summary-label">Causal gene–drug links</div>
                      <div class="summary-value">{n_links}</div>
                    </div>
                    <div class="summary-card">
                      <div class="summary-label">Modelled genes</div>
                      <div class="summary-value">{n_genes}</div>
                    </div>
                    <div class="summary-card">
                      <div class="summary-label">Drugs / perturbagens</div>
                      <div class="summary-value">{n_drugs}</div>
                    </div>
                    <div class="summary-card">
                      <div class="summary-label">Reversal-like links</div>
                      <div class="summary-value">{rev}</div>
                    </div>
                    <div class="summary-card">
                      <div class="summary-label">Aggravating links</div>
                      <div class="summary-value">{aggr}</div>
                    </div>
                  </div>

                  {gene_presence_note_html}
                </div>
              </div>

              <div class="section">
                <div class="section-header">
                  <div class="section-title">CAUSAL GENE–DRUG MODELLING</div>
                  <div class="section-kicker">Dose–response, sensitivity and therapeutic relevance</div>
                </div>
                <div class="section-body">
                  <p>
                    Nonlinear dose–response curves quantify EC50, Hill slope, baseline expression and maximal perturbation (Emax)
                    per gene–drug pair. The sensitivity index summarises potency and amplitude, distinguishing activation-like from
                    repression-like responses and flagging reversal versus aggravating patterns relative to disease-associated
                    expression changes.
                  </p>

                  <!-- KEEP TABLES SPLIT -->
                  <div class="split-row">
                    <div class="split-col">
                      <h4 class="subsection-title">
                        Table 1. Causal gene–drug links prioritising genes (sorted by Sensitivity Index).
                      </h4>
                      {causal_table_html}
                    </div>

                    <div class="split-col">
                      <h4 class="subsection-title">
                        Table 2. QC-adjusted gene–drug sensitivity metrics with clamped amplitudes and tiered sensitivity calls.
                      </h4>
                      {qc_table_html}
                    </div>
                  </div>

                  <!-- DOSE PANELS FULL WIDTH BELOW BOTH TABLES -->
                  {dose_panel_t1_html}
                  {dose_panel_t2_html}

                  <h4 class="subsection-title">Key causal and target-prioritisation figures</h4>
                  {top_hits_figures_html}
                </div>
              </div>

              <div class="section">
                <div class="section-header">
                  <div class="section-title">DRUG SIMILARITY &amp; CLUSTERING</div>
                  <div class="section-kicker">Cosine, Pearson, Spearman and Kendall metrics</div>
                </div>
                <div class="section-body">
                  <p>
                    Drug–drug similarity matrices are computed from stacked gene-level responses, and unsupervised clustering
                    groups compounds into modules with shared transcriptional fingerprints. Cosine, Pearson, Spearman and
                    Kendall metrics provide complementary views of similarity, capturing both linear and rank-based concordance.
                  </p>

                  <h4 class="subsection-title">Cluster size summaries</h4>
                  {cluster_figures_html}

                  <h4 class="subsection-title">Cluster assignments per compound</h4>
                  {cluster_tables_html}
                </div>
              </div>

              <div class="section">
                <div class="section-header">
                  <div class="section-title">REVERSAL-FOCUSED GENE–DRUG PROGRAMS</div>
                  <div class="section-kicker">Gene–drug heatmaps and reversal combinations</div>
                </div>
                <div class="section-body">
                  <p>
                    Reversal summaries aggregate gene–drug pairs where the perturbation counteracts the disease-direction of
                    expression change. Gene–drug heatmaps, per-gene barplots and multi-drug summaries highlight specific
                    compounds and combinations that most effectively restore a healthier transcriptional state <em>in silico</em>.
                  </p>

                  {reversal_figures_html}

                  {reversal_barplots_panel_html}
                  {reversal_multi_panel_html}
                </div>
              </div>

              <div class="section">
                <div class="section-header">
                  <div class="section-title">REFERENCES</div>
                  <div class="section-kicker">L1000 platform and disease-contextualised interpretation</div>
                </div>
                <div class="section-body">
                  {references_html}

                  <div class="end-note">
                    Please review the main downloadable files for more information please!
                  </div>
                </div>
              </div>

            </div>
          </div>
        </div>
        {js}
      </body>
    </html>
    """

    with open(out_html, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"[INFO] L1000 run used: {run_dir}")
    print(f"[INFO] Wrote report to: {out_html}")
    print(f"[INFO] Style: {style}")


# from pathlib import Path

# if __name__ == "__main__":
    
#     run_dir = Path("L1000_Output")
#     out_html = run_dir / "l1000_report.html"
    
#     build_report(
#         run_dir=run_dir,
#         out_html=out_html,
#         disease_path= run_dir / "userinput.txt",
#         logo_left_path=Path(r"C:\Ayass Bio Work\Agentic_AI_ABS\perturbation_pipeline\perturbation\logos\Ayass_logo_left.png"),
#         logo_right_path=Path(r"C:\Ayass Bio Work\Agentic_AI_ABS\perturbation_pipeline\perturbation\logos\Ayass_logo_right.png"),
#     )
