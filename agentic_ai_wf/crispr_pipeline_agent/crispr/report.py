#!/usr/bin/env python3
"""
generate_crispr_html_report.py

Self-contained HTML report generator for CRISPR Perturb-seq pipeline outputs.

Key features:
- Option A: One report per sample (recommended)
- Option B: Multi-sample combined report
- Standalone HTML: embeds PNG as base64 and embeds HTML figures via iframe srcdoc
- Summary table is Table 1
- Each step section: definition + split view (interactive table + figure viewer)
- Correct global numbering across the report
- Skips missing content gracefully
- Uses full names (no acronyms in headings)

Usage:
  Option A (per-sample):
    python generate_crispr_html_report.py --processed_root processed --out_dir CRISPR_Folder/reports

  Option B (combined):
    python generate_crispr_html_report.py --processed_root processed --out_html CRISPR_Folder/combined_report.html

If you're running on a single sample dir directly:
    python generate_crispr_html_report.py --sample_dir processed/GSM2406681_10X010 --out_html CRISPR_Folder/GSM2406681_10X010_report.html
"""

import argparse
import base64
import datetime as dt
import html
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


# -------------------------
# User prompt
# -------------------------
def ask_mode_interactively() -> str:
    print("\nSelect report mode:")
    print("Option A: One report per sample (default, simpler, safest)")
    print("Option B: Multi-sample combined report (more complex but doable)")
    choice = input("Choose A or B [A]: ").strip().upper()
    return "multi" if choice == "B" else "single"


# -------------------------
# IO helpers
# -------------------------
def read_table_any(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists() or not path.is_file():
        return None
    try:
        if path.suffix.lower() == ".tsv":
            return pd.read_csv(path, sep="\t")
        if path.suffix.lower() == ".csv":
            return pd.read_csv(path)
    except Exception:
        return None
    return None


def encode_file_to_data_uri(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    suffix = path.suffix.lower()
    mime = "application/octet-stream"
    if suffix == ".png":
        mime = "image/png"
    elif suffix in [".jpg", ".jpeg"]:
        mime = "image/jpeg"
    elif suffix == ".svg":
        mime = "image/svg+xml"
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{data}"


def escape_html_for_srcdoc(path: Path) -> Optional[str]:
    """Read an HTML file and escape it so it can be embedded as iframe srcdoc."""
    if not path.exists():
        return None
    try:
        raw = path.read_text(encoding="utf-8", errors="replace")
        return html.escape(raw, quote=True)
    except Exception:
        return None


def safe_title(name: str) -> str:
    name = name.replace("_", " ").strip()
    name = re.sub(r"\s+", " ", name)
    return name[:1].upper() + name[1:]


def list_sample_dirs(processed_root: Path) -> List[Path]:
    if not processed_root.exists():
        return []
    # sample dirs: GSM*
    samples = [p for p in processed_root.iterdir() if p.is_dir() and p.name.startswith("GSM")]
    return sorted(samples, key=lambda p: p.name)


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


# -------------------------
# Stage definitions (full names + 3-line definitions)
# -------------------------
STAGES = [
    ("processed_stage3", "Post Mixscape Analysis",
     "This step consolidates corrected perturbation labels and summarizes post-correction quality metrics.\n"
     "It is important because it ensures perturbation assignments are reliable before downstream statistics.\n"
     "Outputs typically include merged annotated datasets, summary tables, and diagnostic plots."),

    ("processed_stage4", "Differential Gene Expression",
     "This step identifies genes whose expression changes significantly after each perturbation.\n"
     "It is important because it quantifies the biological response to each perturbation at gene level.\n"
     "Outputs typically include differential expression tables and associated volcano/heatmap figures."),

    ("processed_stage5", "Machine Learning Dataset Export",
     "This step converts processed results into standardized feature tables for model training.\n"
     "It is important because it guarantees consistent inputs and metadata for predictive modeling.\n"
     "Outputs typically include training tables, feature matrices, and export manifests."),

    ("processed_stage6", "Model Training",
     "This step trains classification models to predict perturbation effects from expression signatures.\n"
     "It is important because it enables automated inference and ranking of perturbations.\n"
     "Outputs typically include model artifacts, training metrics, and performance plots."),

    ("processed_stage7", "Prediction and Ranking",
     "This step generates per-cell predictions and confidence scores for perturbation labels.\n"
     "It is important because it provides downstream labels used for causal analysis and reporting.\n"
     "Outputs typically include prediction tables and confidence summary visualizations."),

    ("processed_stage8", "Quality Control and Optimization",
     "This step applies confidence filters and optimizes thresholds to improve label quality.\n"
     "It is important because it removes ambiguous cells and stabilizes downstream inference.\n"
     "Outputs typically include optimized parameters, filtered datasets, and quality figures."),

    ("processed_stage9", "Causal Inference using Instrumental Variables",
     "This step estimates causal perturbation effects while controlling for confounding.\n"
     "It is important because it supports more robust effect estimates than correlation alone.\n"
     "Outputs typically include causal effect tables and diagnostic plots."),

    ("processed_stage10", "Bayesian Network Structure Learning",
     "This step learns a directed dependency graph linking perturbations to latent transcriptomic features.\n"
     "It is important because it suggests mechanistic structure and potential causal pathways.\n"
     "Outputs typically include network edge tables and a directed graph figure."),

    ("processed_stage11", "Latent Representation and User Interface Cards",
     "This step builds latent clusters and generates user-friendly summaries for exploration.\n"
     "It is important because it compresses high-dimensional data into interpretable modules.\n"
     "Outputs typically include latent cluster tables, top genes, enrichment tables, and UMAP figures."),

    ("processed_stage12", "Final Handoff Package",
     "This step consolidates key outputs into a deliverable package for downstream review.\n"
     "It is important because it ensures results are easy to share, reproduce, and interpret.\n"
     "Outputs typically include final summary tables, links, and export bundles."),
]


# -------------------------
# Discover assets per stage
# -------------------------
def discover_stage_assets(sample_dir: Path, stage_folder: str) -> Dict[str, List[Path]]:
    stage_path = sample_dir / stage_folder
    assets = {"tables": [], "fig_png": [], "fig_html": []}
    if not stage_path.exists():
        return assets

    # tables in any tables/ folder
    for ext in ("*.tsv", "*.csv"):
        assets["tables"].extend(sorted(stage_path.rglob(ext), key=lambda p: p.name.lower()))

    # figures in any figures/ folder (PNG) + interactive html
    assets["fig_png"].extend(sorted(stage_path.rglob("*.png"), key=lambda p: p.name.lower()))
    assets["fig_html"].extend(sorted(stage_path.rglob("*.html"), key=lambda p: p.name.lower()))

    return assets


# -------------------------
# HTML renderers
# -------------------------
def render_table_block(table_id: str, df: pd.DataFrame, caption: str, table_number: int, max_rows: int = 50) -> str:
    df2 = df.copy()
    truncated = False
    if len(df2) > max_rows:
        df2 = df2.head(max_rows)
        truncated = True

    cols = [str(c) for c in df2.columns]
    head = "".join(f'<th onclick="sortTable(\'{table_id}\',{i})" title="Click to sort">{html.escape(c)}</th>'
                   for i, c in enumerate(cols))

    body_rows = []
    for _, r in df2.iterrows():
        tds = "".join(f"<td>{html.escape(str(x))}</td>" for x in r.values.tolist())
        body_rows.append(f"<tr>{tds}</tr>")
    body = "\n".join(body_rows) if body_rows else f'<tr><td colspan="{len(cols)}">No rows</td></tr>'

    note = f"<div class='note'>Showing first {max_rows} rows.</div>" if truncated else ""

    return f"""
    <div class="table-card">
      <div class="table-header">
        <div class="table-title">Table {table_number}. {html.escape(caption)}</div>
        <input class="table-search" type="text" placeholder="Search…" oninput="filterTable('{table_id}', this.value)">
      </div>
      <div class="table-wrapper">
        <table id="{table_id}">
          <thead><tr>{head}</tr></thead>
          <tbody>{body}</tbody>
        </table>
      </div>
      {note}
    </div>
    """


def render_figure_block(fig_number: int, title: str, caption: str, png_path: Optional[Path], html_path: Optional[Path]) -> str:
    title = html.escape(title)
    caption = html.escape(caption)

    png_html = ""
    if png_path and png_path.exists():
        uri = encode_file_to_data_uri(png_path)
        if uri:
            png_html = f'<img class="figure-img" src="{uri}" alt="{title}">'

    html_embed = ""
    if html_path and html_path.exists():
        escaped = escape_html_for_srcdoc(html_path)
        if escaped:
            html_embed = f"""
            <div class="interactive">
              <div class="interactive-label">Interactive visualization</div>
              <iframe class="iframe" srcdoc="{escaped}"></iframe>
            </div>
            """

    if not png_html and not html_embed:
        return ""

    return f"""
    <div class="figure-card">
      <div class="figure-title">Figure {fig_number}. {title}</div>
      <div class="figure-caption">{caption}</div>
      {png_html}
      {html_embed}
    </div>
    """


def render_step_section(
    step_id: str,
    step_title: str,
    definition_3lines: str,
    tables: List[Tuple[str, pd.DataFrame, str, int]],
    figures: List[Tuple[str, str, Optional[Path], Optional[Path], int]],
) -> str:
    # tables: (table_key, df, caption, table_number)
    # figures: (fig_key, title, png_path, html_path, fig_number)
    # Build dropdown-driven viewers
    table_options = []
    table_panels = []
    for idx, (key, df, caption, tnum) in enumerate(tables):
        tid = f"{step_id}_table_{idx}"
        table_options.append(f'<option value="{tid}">Table {tnum}: {html.escape(caption[:80])}</option>')
        table_panels.append(f'<div class="panel" id="{tid}" style="display:none;">'
                            f'{render_table_block(tid + "_inner", df, caption, tnum)}</div>')

    fig_options = []
    fig_panels = []
    for idx, (key, title, png_path, html_path, fnum) in enumerate(figures):
        fid = f"{step_id}_fig_{idx}"
        fig_options.append(f'<option value="{fid}">Figure {fnum}: {html.escape(title[:80])}</option>')
        fig_panels.append(f'<div class="panel" id="{fid}" style="display:none;">'
                          f'{render_figure_block(fnum, title, "Automatically generated from pipeline outputs.", png_path, html_path)}</div>')

    left = ""
    if table_panels:
        left = f"""
        <div class="viewer">
          <div class="viewer-head">
            <div class="viewer-label">Tables</div>
            <select onchange="switchPanel(this.value, '{step_id}_table_group')">
              {''.join(table_options)}
            </select>
          </div>
          <div id="{step_id}_table_group">
            {''.join(table_panels)}
          </div>
        </div>
        """
    else:
        left = '<div class="empty">No tables found for this step.</div>'

    right = ""
    if fig_panels:
        right = f"""
        <div class="viewer">
          <div class="viewer-head">
            <div class="viewer-label">Figures</div>
            <select onchange="switchPanel(this.value, '{step_id}_fig_group')">
              {''.join(fig_options)}
            </select>
          </div>
          <div id="{step_id}_fig_group">
            {''.join(fig_panels)}
          </div>
        </div>
        """
    else:
        right = '<div class="empty">No figures found for this step.</div>'

    definition_html = "<br>".join(html.escape(x.strip()) for x in definition_3lines.splitlines() if x.strip())

    return f"""
    <section class="section" id="{step_id}">
      <div class="section-eyebrow">Pipeline step</div>
      <h2 class="section-title">{html.escape(step_title)}</h2>
      <div class="section-def">{definition_html}</div>

      <div class="split">
        <div class="left">{left}</div>
        <div class="right">{right}</div>
      </div>
    </section>
    """
HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Ayass Bioscience · CRISPR Perturb-seq Analysis Report</title>
<style>
  :root { --bg:#f5f7fb; --card:#fff; --border:#e4e9f6; --muted:#5b6b82; --strong:#0f2438; --accent:#0ea5a5; }
  * { box-sizing: border-box; }
  body { margin:0; font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial; background:var(--bg); color:#111; }
  header { background: linear-gradient(90deg, #0099cc, #00b894); color:#fff; padding:18px 22px; }
  .hero { display:flex; align-items:center; gap:16px; justify-content:space-between; flex-wrap:wrap; }
  .logo { height:54px; object-fit:contain; background:rgba(255,255,255,0.15); padding:6px 10px; border-radius:14px; }
  .title { font-size:22px; font-weight:750; margin:0; }
  .subtitle { margin:6px 0 0 0; font-size:13px; opacity:0.95; }
  .meta { font-size:12px; opacity:0.95; white-space:nowrap; }
  .wrap { max-width:1200px; margin: 18px auto 40px auto; padding: 0 18px; }
  .section { background:var(--card); border:1px solid var(--border); border-radius:18px; padding:18px; margin-top:16px; }
  .section-eyebrow { font-size:11px; letter-spacing:0.14em; text-transform:uppercase; color:#d9ffff; display:none; }
  .section-title { margin:0 0 6px 0; font-size:18px; color:var(--strong); }
  .section-def { color:var(--muted); font-size:13px; line-height:1.45; margin-bottom:14px; }
  .split { display:flex; gap:14px; }
  .left, .right { flex:1; min-width: 280px; }
  .viewer { border:1px solid var(--border); border-radius:16px; background:#fbfdff; padding:12px; }
  .viewer-head { display:flex; justify-content:space-between; align-items:center; gap:10px; flex-wrap:wrap; margin-bottom:10px; }
  .viewer-label { font-size:12px; font-weight:700; text-transform:uppercase; letter-spacing:0.08em; color:#224; }
  select { border:1px solid #d5dbea; border-radius:999px; padding:6px 10px; background:#fff; font-size:12px; }
  .panel { }
  .empty { color:var(--muted); font-size:13px; padding:10px; }

  .table-card { background:#fff; border:1px solid var(--border); border-radius:14px; padding:10px; }
  .table-header { display:flex; justify-content:space-between; gap:10px; flex-wrap:wrap; align-items:center; margin-bottom:8px; }
  .table-title { font-size:12px; font-weight:750; color:#143; }
  .table-search { border:1px solid #d5dbea; border-radius:999px; padding:6px 10px; font-size:12px; min-width:180px; }
  .table-wrapper { max-height:420px; overflow:auto; border:1px solid #e8edf8; border-radius:12px; }
  table { border-collapse: collapse; width:100%; font-size:12px; }
  thead tr { background:#eef2ff; position:sticky; top:0; z-index:1; }
  th, td { border-bottom:1px solid #eef2ff; padding:6px 10px; white-space:nowrap; text-align:left; }
  th { cursor:pointer; user-select:none; }
  tbody tr:nth-child(even) td { background:#fafcff; }
  tbody tr:hover td { background:#eef6ff; }
  .note { font-size:11px; color:var(--muted); margin-top:6px; }

  .figure-card { background:#fff; border:1px solid var(--border); border-radius:14px; padding:10px; }
  .figure-title { font-size:12px; font-weight:750; color:#143; margin-bottom:4px; }
  .figure-caption { font-size:12px; color:var(--muted); margin-bottom:10px; }
  .figure-img { width:100%; border:1px solid #d5dbea; border-radius:12px; display:block; }

  .interactive { margin-top:10px; border:1px dashed #cbd5ff; border-radius:12px; padding:10px; background:#f8faff; }
  .interactive-label { font-size:11px; color:var(--muted); margin-bottom:6px; }
  .iframe { width:100%; min-height:420px; border:1px solid #d5dbea; border-radius:10px; background:#fff; }

  .footer { margin-top:22px; font-size:12px; color:var(--muted); padding: 8px 2px; }
  .refs a { color:#1d4ed8; text-decoration:none; }
  .refs a:hover { text-decoration:underline; }

  @media (max-width: 980px) {
    .split { flex-direction: column; }
  }
</style>

<script>
  function filterTable(tableId, query) {
    const filter = (query || "").toLowerCase();
    const table = document.getElementById(tableId);
    if (!table) return;
    const rows = table.getElementsByTagName("tr");
    for (let i = 1; i < rows.length; i++) {
      const cells = rows[i].getElementsByTagName("td");
      let show = false;
      for (let j = 0; j < cells.length; j++) {
        if ((cells[j].innerText || "").toLowerCase().indexOf(filter) !== -1) { show = true; break; }
      }
      rows[i].style.display = show ? "" : "none";
    }
  }

  const sortState = {};
  function parseCellValue(text) {
    const t = (text || "").trim();
    const num = parseFloat(t.replace(/[^0-9eE\\-\\.]/g, ""));
    if (!isNaN(num) && /[0-9]/.test(t)) return num;
    return t.toLowerCase();
  }
  function sortTable(tableId, colIndex) {
    const table = document.getElementById(tableId);
    if (!table) return;
    const tbody = table.tBodies[0];
    const rows = Array.from(tbody.rows);
    const key = tableId + ":" + colIndex;
    const current = sortState[key] || "desc";
    const next = current === "asc" ? "desc" : "asc";
    sortState[key] = next;
    rows.sort((a, b) => {
      const va = parseCellValue(a.cells[colIndex].innerText);
      const vb = parseCellValue(b.cells[colIndex].innerText);
      if (va < vb) return next === "asc" ? -1 : 1;
      if (va > vb) return next === "asc" ? 1 : -1;
      return 0;
    });
    for (const r of rows) tbody.appendChild(r);
  }

  function switchPanel(panelId, groupId) {
    const group = document.getElementById(groupId);
    if (!group) return;
    const panels = group.querySelectorAll(".panel");
    panels.forEach(p => p.style.display = "none");
    const active = document.getElementById(panelId);
    if (active) active.style.display = "block";
  }

  function initDefaultPanels() {
    document.querySelectorAll("[id$='_table_group']").forEach(group => {
      const first = group.querySelector(".panel");
      if (first) first.style.display = "block";
    });
    document.querySelectorAll("[id$='_fig_group']").forEach(group => {
      const first = group.querySelector(".panel");
      if (first) first.style.display = "block";
    });
  }
  window.addEventListener("load", initDefaultPanels);
</script>

</head>
<body>
<header>
  <div class="hero">
    <div style="display:flex; align-items:center; gap:14px;">
      {logo_html}
      <div>
        <div class="title">CRISPR Perturb-seq Analysis Report</div>
        <div class="subtitle">All figures and tables embedded as a standalone HTML report.</div>
      </div>
    </div>
    <div class="meta">Generated: {generated_at}</div>
  </div>
</header>

<div class="wrap">
  {summary_section}
  {content_sections}

  <div class="footer">
    <div><b>References</b></div>
    <div class="refs">
      <a href="https://ayassbioscience.com/biragas/" target="_blank" rel="noopener">https://ayassbioscience.com/biragas/</a><br>
      <a href="https://pertpy.readthedocs.io/en/stable/tutorials/notebooks/mixscape.html" target="_blank" rel="noopener">
        https://pertpy.readthedocs.io/en/stable/tutorials/notebooks/mixscape.html
      </a>
    </div>
  </div>
</div>
</body>
</html>
"""


def build_summary_table(sample_name: str, stage_asset_map: Dict[str, Dict[str, List[Path]]]) -> pd.DataFrame:
    rows = []
    for stage_folder, stage_title, _def in STAGES:
        a = stage_asset_map.get(stage_folder, {"tables": [], "fig_png": [], "fig_html": []})
        rows.append({
            "Sample": sample_name,
            "Step": stage_title,
            "Tables (CSV/TSV)": len(a["tables"]),
            "Figures (PNG)": len(a["fig_png"]),
            "Interactive figures (HTML)": len(a["fig_html"]),
        })
    return pd.DataFrame(rows)


def build_report_for_sample(sample_dir: Path, logo_path: Optional[Path]) -> str:
    sample_name = sample_dir.name

    # Logo embed
    logo_html = ""
    if logo_path and logo_path.exists():
        uri = encode_file_to_data_uri(logo_path)
        if uri:
            logo_html = f'<img class="logo" src="{uri}" alt="Ayass Bioscience logo">'
    else:
        logo_html = ""

    # Discover assets per stage
    stage_assets: Dict[str, Dict[str, List[Path]]] = {}
    for stage_folder, _title, _def in STAGES:
        stage_assets[stage_folder] = discover_stage_assets(sample_dir, stage_folder)

    # ---- Global numbering
    table_no = 1
    fig_no = 1

    # ---- Summary section is Table 1
    summary_df = build_summary_table(sample_name, stage_assets)
    summary_html = render_table_block(
        table_id=f"{sample_name}_summary_table",
        df=summary_df,
        caption="Summary of available outputs by pipeline step.",
        table_number=table_no,
        max_rows=50
    )
    summary_section = f"""
    <section class="section">
      <h2 class="section-title">Report Summary</h2>
      <div class="section-def">
        This summary lists the number of tables and figures produced at each pipeline step for the selected sample.<br>
        It is important because it provides a fast quality check for completeness of the pipeline outputs.<br>
        Missing steps are skipped in later sections to keep the report accurate.
      </div>
      {summary_html}
    </section>
    """
    table_no += 1

    # ---- Build each stage section
    content_sections = []
    for stage_folder, stage_title, stage_def in STAGES:
        a = stage_assets.get(stage_folder, {})
        table_paths = a.get("tables", [])
        png_paths = a.get("fig_png", [])
        html_paths = a.get("fig_html", [])

        # If nothing exists, skip entirely
        if not table_paths and not png_paths and not html_paths:
            continue

        # prepare tables (limit render to first N tables, but user can still switch)
        tables_for_section = []
        for p in table_paths:
            df = read_table_any(p)
            if df is None or df.empty:
                continue
            caption = safe_title(p.stem)
            tables_for_section.append((p.name, df, caption, table_no))
            table_no += 1

        # prepare figures
        figures_for_section = []
        # Pairing heuristic: group by stem prefix; fallback: just list
        used_html = set()
        for png in png_paths:
            # try match html by same stem
            match = next((h for h in html_paths if h.stem == png.stem), None)
            if match:
                used_html.add(match)
            figures_for_section.append((png.name, safe_title(png.stem), png, match, fig_no))
            fig_no += 1

        # add remaining html (interactive) figures without png
        for h in html_paths:
            if h in used_html:
                continue
            figures_for_section.append((h.name, safe_title(h.stem), None, h, fig_no))
            fig_no += 1

        step_id = f"{sample_name}_{stage_folder}"

        content_sections.append(
            render_step_section(
                step_id=step_id,
                step_title=stage_title,
                definition_3lines=stage_def,
                tables=tables_for_section,
                figures=figures_for_section,
            )
        )

    generated_at = dt.datetime.now().strftime("%Y-%m-%d %H:%M")

    return HTML_TEMPLATE.format(
        logo_html=logo_html,
        generated_at=html.escape(generated_at),
        summary_section=summary_section,
        content_sections="\n".join(content_sections),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed_root", help="Path to processed/ directory containing GSM* sample folders", default=None)
    ap.add_argument("--sample_dir", help="Path to a single sample directory processed/GSM...", default=None)

    ap.add_argument("--out_html", help="Output HTML file (used for single combined report)", default=None)
    ap.add_argument("--out_dir", help="Output directory (used for per-sample reports)", default=None)

    ap.add_argument("--logo", default="Ayass-BioScience-Logo-FC.png", help="Logo file to embed (PNG)")
    ap.add_argument("--no_prompt", action="store_true", help="Skip interactive prompt and infer mode from args")
    ap.add_argument("--mode", choices=["single", "multi"], default=None, help="Override prompt: single or multi")

    args = ap.parse_args()

    logo_path = Path(args.logo) if args.logo else None

    # Determine inputs
    if args.sample_dir:
        sample_dirs = [Path(args.sample_dir)]
        mode = "single"
    else:
        if not args.processed_root:
            raise SystemExit("ERROR: Provide --processed_root processed  OR  --sample_dir processed/GSMxxxx")
        processed_root = Path(args.processed_root)
        sample_dirs = list_sample_dirs(processed_root)

        if not sample_dirs:
            raise SystemExit(f"ERROR: No sample directories found under: {processed_root}")

        if args.no_prompt:
            mode = args.mode or ("single" if args.out_dir else "multi")
        else:
            mode = args.mode or ask_mode_interactively()

    if mode == "single":
        # one report per sample
        out_dir = Path(args.out_dir) if args.out_dir else Path("CRISPR_Folder/reports")
        ensure_dir(out_dir)

        for sd in sample_dirs:
            html_text = build_report_for_sample(sd, logo_path)
            out_path = out_dir / f"{sd.name}_CRISPR_PerturbSeq_Report.html"
            out_path.write_text(html_text, encoding="utf-8")
            print(f"[OK] Wrote per-sample report: {out_path}")

    else:
        # combined report
        if not args.out_html:
            raise SystemExit("ERROR: For multi-sample combined report, provide --out_html <file.html>")

        combined_sections = []
        for sd in sample_dirs:
            combined_sections.append(build_report_for_sample(sd, logo_path))

        # For combined mode: wrap the per-sample full HTML bodies into one mega-page:
        # simplest safe approach: keep ONE header and insert only sample content sections.
        # We'll extract summary + content sections from each generated report using markers.
        # If extraction fails, fallback to concatenating the whole HTML strings (still works but nested).
        def extract_between(txt: str, key: str) -> str:
            # naive extraction: find `{summary_section}` block by known headers
            return txt

        # Safer: build a combined page by regenerating sections directly (not parsing HTML)
        generated_at = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
        logo_html = ""
        if logo_path and logo_path.exists():
            uri = encode_file_to_data_uri(logo_path)
            if uri:
                logo_html = f'<img class="logo" src="{uri}" alt="Ayass Bioscience logo">'

        # Build combined content: each sample gets its own "Report Summary" + steps
        combined_content = []
        table_no_global = 1
        fig_no_global = 1

        # In combined mode, we keep numbering correct across ALL samples.
        # We'll reuse the same logic but with shared counters by rebuilding sections here.
        processed_root = Path(args.processed_root) if args.processed_root else None

        def discover_all_assets(sd: Path) -> Dict[str, Dict[str, List[Path]]]:
            m = {}
            for stage_folder, _t, _d in STAGES:
                m[stage_folder] = discover_stage_assets(sd, stage_folder)
            return m

        # Build combined summary table as Table 1 (all samples × steps)
        all_rows = []
        for sd in sample_dirs:
            stage_assets = discover_all_assets(sd)
            df = build_summary_table(sd.name, stage_assets)
            all_rows.append(df)
        combined_summary_df = pd.concat(all_rows, axis=0, ignore_index=True)

        combined_summary_html = render_table_block(
            table_id="combined_summary",
            df=combined_summary_df,
            caption="Summary of available outputs by pipeline step across all samples.",
            table_number=table_no_global,
            max_rows=50
        )
        table_no_global += 1

        combined_content.append(f"""
        <section class="section">
          <h2 class="section-title">Report Summary (All Samples)</h2>
          <div class="section-def">
            This summary aggregates outputs across all selected samples and pipeline steps.<br>
            It is important because it quickly highlights missing steps and incomplete outputs across the run.<br>
            Steps without outputs are skipped in the detailed sections below.
          </div>
          {combined_summary_html}
        </section>
        """)

        # Build per-sample detailed sections with global numbering
        for sd in sample_dirs:
            sample_name = sd.name
            stage_assets = discover_all_assets(sd)

            combined_content.append(f"""
            <section class="section">
              <h2 class="section-title">Sample: {html.escape(sample_name)}</h2>
              <div class="section-def">
                Detailed outputs for this sample are shown below, organized by pipeline step.<br>
                Tables and figures are embedded directly into the HTML for portability.<br>
                Missing items are omitted automatically.
              </div>
            </section>
            """)

            for stage_folder, stage_title, stage_def in STAGES:
                a = stage_assets.get(stage_folder, {})
                table_paths = a.get("tables", [])
                png_paths = a.get("fig_png", [])
                html_paths = a.get("fig_html", [])

                if not table_paths and not png_paths and not html_paths:
                    continue

                tables_for_section = []
                for p in table_paths:
                    df = read_table_any(p)
                    if df is None or df.empty:
                        continue
                    caption = safe_title(p.stem)
                    tables_for_section.append((p.name, df, caption, table_no_global))
                    table_no_global += 1

                figures_for_section = []
                used_html = set()
                for png in png_paths:
                    match = next((h for h in html_paths if h.stem == png.stem), None)
                    if match:
                        used_html.add(match)
                    figures_for_section.append((png.name, safe_title(png.stem), png, match, fig_no_global))
                    fig_no_global += 1
                for h in html_paths:
                    if h in used_html:
                        continue
                    figures_for_section.append((h.name, safe_title(h.stem), None, h, fig_no_global))
                    fig_no_global += 1

                step_id = f"{sample_name}_{stage_folder}"
                combined_content.append(
                    render_step_section(
                        step_id=step_id,
                        step_title=stage_title,
                        definition_3lines=stage_def,
                        tables=tables_for_section,
                        figures=figures_for_section,
                    )
                )

        combined_html = HTML_TEMPLATE.format(
            logo_html=logo_html,
            generated_at=html.escape(generated_at),
            summary_section="",  # already included in combined_content
            content_sections="\n".join(combined_content),
        )

        out_path = Path(args.out_html)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(combined_html, encoding="utf-8")
        print(f"[OK] Wrote combined report: {out_path}")


if __name__ == "__main__":
    main()

