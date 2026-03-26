#!/usr/bin/env python3
"""
Stage-12: LLM & Handoff + Delivery (Option D: report.md + report.pdf + ui_payload.json + bundle)

STEP-BY-STEP CAPTIONS
1) Output structure
   Caption: Create delivery folder structure (tables/figures/bundle)
2) Load Stage-11 (Latent AI & UI)
   Caption: Summarize latent clusters + top genes + perturbation enrichment + UMAP figures
3) Load Stage-10 (Bayesian Network)
   Caption: Summarize BN edges + key relationships around UPR_score
4) Load Stage-9 (Causal IV)
   Caption: Summarize IV effect estimates + diagnostics + instrument ranking plot
5) Load Stage-8 (QC + Optimization)
   Caption: Use high-confidence predicted perturbations as reliable candidates
6) Build Markdown report
   Caption: Narrative report with captions and links to copied artifacts
7) Export UI JSON payload
   Caption: UI-ready cards + tables + figure references for downstream app/hand-off
8) Render PDF (optional)
   Caption: Convert report.md -> report.pdf (ReportLab if available)

Notes:
- This stage is "LLM & handoff" in your roadmap.
- It does NOT re-run ML or biology computations; it packages, summarizes, and delivers.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


# -----------------------------
# Utilities
# -----------------------------
def log(msg: str) -> None:
    print(f"[INFO] {msg}")


def warn(msg: str) -> None:
    print(f"[WARN] {msg}")


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_tsv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t")


def safe_copy(src: Path, dst: Path) -> Optional[Path]:
    if not src.exists():
        warn(f"Missing file (skip copy): {src}")
        return None
    ensure_dir(dst.parent)
    shutil.copy2(src, dst)
    return dst


def top_rows(df: pd.DataFrame, n: int) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    return df.head(n).copy()


def md_table(df: pd.DataFrame, max_rows: int = 12) -> str:
    if df is None or df.empty:
        return "_(none)_"
    view = df.head(max_rows).copy()
    return view.to_markdown(index=False)


def try_import_reportlab():
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.units import inch
        from reportlab.pdfgen import canvas
        return letter, inch, canvas
    except Exception:
        return None


def md_to_simple_pdf(md_text: str, out_pdf: Path, title: str = "Stage-12 Report") -> bool:
    """
    Minimal Markdown->PDF renderer using ReportLab (text-only).
    Keeps headings/bullets in a readable way.
    """
    rl = try_import_reportlab()
    if rl is None:
        warn("ReportLab not available; skipping PDF export.")
        return False

    letter, inch, canvas = rl
    c = canvas.Canvas(str(out_pdf), pagesize=letter)
    width, height = letter

    x = 0.9 * inch
    y = height - 0.9 * inch
    line_h = 12

    def draw_line(s: str, bold: bool = False):
        nonlocal y
        if y < 0.9 * inch:
            c.showPage()
            y = height - 0.9 * inch
        if bold:
            c.setFont("Helvetica-Bold", 11)
        else:
            c.setFont("Helvetica", 10)
        c.drawString(x, y, s[:120])  # prevent overly long lines
        y -= line_h

    # Title
    c.setFont("Helvetica-Bold", 14)
    c.drawString(x, y, title)
    y -= 18

    for raw in md_text.splitlines():
        line = raw.rstrip("\n")
        if not line.strip():
            y -= line_h * 0.5
            continue

        if line.startswith("# "):
            draw_line(line[2:].strip(), bold=True)
            y -= 4
        elif line.startswith("## "):
            draw_line(line[3:].strip(), bold=True)
        elif line.startswith("### "):
            draw_line("• " + line[4:].strip(), bold=True)
        elif line.startswith("- "):
            draw_line("  - " + line[2:].strip(), bold=False)
        else:
            draw_line(line, bold=False)

    c.save()
    return True


# -----------------------------
# Data Structures
# -----------------------------
@dataclass
class Stage12Artifacts:
    out_dir: Path
    tables_dir: Path
    figures_dir: Path
    bundle_dir: Path


@dataclass
class Stage12Inputs:
    stage8_dir: Path
    stage9_dir: Path
    stage10_dir: Path
    stage11_dir: Path


# -----------------------------
# Loaders
# -----------------------------
def load_stage8(stage8_dir: Path) -> Dict[str, Optional[pd.DataFrame]]:
    """
    Stage-8 key tables used:
      - tables/stage8_highconf_perts_ge_50.tsv
      - tables/stage8_cells_highconf_ge_0.40.tsv
      - tables/stage8_predicted_perturbation_qc.tsv
    """
    tables = stage8_dir / "tables"
    out: Dict[str, Optional[pd.DataFrame]] = {"highconf_perts": None, "pert_qc": None, "cells_highconf": None}

    p1 = tables / "stage8_highconf_perts_ge_50.tsv"
    p2 = tables / "stage8_predicted_perturbation_qc.tsv"
    p3 = tables / "stage8_cells_highconf_ge_0.40.tsv"

    if p1.exists():
        out["highconf_perts"] = read_tsv(p1)
    else:
        warn(f"Stage-8 highconf perts not found: {p1}")

    if p2.exists():
        out["pert_qc"] = read_tsv(p2)
    else:
        warn(f"Stage-8 per-pert QC not found: {p2}")

    if p3.exists():
        out["cells_highconf"] = read_tsv(p3)
    else:
        warn(f"Stage-8 highconf cells not found: {p3}")

    return out


def load_stage9(stage9_dir: Path) -> Dict[str, Optional[pd.DataFrame]]:
    """
    Stage-9 key tables:
      - tables/stage9_iv_results.tsv
      - tables/stage9_diagnostics.tsv
    """
    tables = stage9_dir / "tables"
    out: Dict[str, Optional[pd.DataFrame]] = {"iv_results": None, "diagnostics": None}

    p1 = tables / "stage9_iv_results.tsv"
    p2 = tables / "stage9_diagnostics.tsv"

    if p1.exists():
        out["iv_results"] = read_tsv(p1)
    else:
        warn(f"Stage-9 IV results not found: {p1}")

    if p2.exists():
        out["diagnostics"] = read_tsv(p2)
    else:
        warn(f"Stage-9 diagnostics not found: {p2}")

    return out


def load_stage10(stage10_dir: Path) -> Dict[str, Optional[pd.DataFrame]]:
    """
    Stage-10 key tables:
      - tables/stage10_edges.tsv
      - tables/stage10_nodes.tsv
      - tables/stage10_data_used.tsv
    """
    tables = stage10_dir / "tables"
    out: Dict[str, Optional[pd.DataFrame]] = {"edges": None, "nodes": None, "data_used": None}

    for key, fname in [("edges", "stage10_edges.tsv"), ("nodes", "stage10_nodes.tsv"), ("data_used", "stage10_data_used.tsv")]:
        p = tables / fname
        if p.exists():
            out[key] = read_tsv(p)
        else:
            warn(f"Stage-10 missing: {p}")

    return out


def load_stage11(stage11_dir: Path) -> Dict[str, Optional[pd.DataFrame]]:
    """
    Stage-11 key tables:
      - tables/stage11_cluster_summary.tsv
      - tables/stage11_cluster_top_genes.tsv
      - tables/stage11_perturbation_cluster_enrichment.tsv
      - tables/stage11_ui_cards.tsv
    """
    tables = stage11_dir / "tables"
    out: Dict[str, Optional[pd.DataFrame]] = {
        "cluster_summary": None,
        "cluster_top_genes": None,
        "cluster_enrichment": None,
        "ui_cards": None,
    }

    for key, fname in [
        ("cluster_summary", "stage11_cluster_summary.tsv"),
        ("cluster_top_genes", "stage11_cluster_top_genes.tsv"),
        ("cluster_enrichment", "stage11_perturbation_cluster_enrichment.tsv"),
        ("ui_cards", "stage11_ui_cards.tsv"),
    ]:
        p = tables / fname
        if p.exists():
            out[key] = read_tsv(p)
        else:
            warn(f"Stage-11 missing: {p}")

    return out


# -----------------------------
# Report builder
# -----------------------------
def build_report_md(
    inputs: Stage12Inputs,
    artifacts: Stage12Artifacts,
    s8: Dict[str, Optional[pd.DataFrame]],
    s9: Dict[str, Optional[pd.DataFrame]],
    s10: Dict[str, Optional[pd.DataFrame]],
    s11: Dict[str, Optional[pd.DataFrame]],
    topn_perts: int,
    topn_clusters: int,
) -> str:
    """
    Markdown report with captions (explicit).
    """
    lines: List[str] = []
    lines.append("# Stage-12 Handoff Report (LLM + Delivery)")
    lines.append("")
    lines.append("## What this stage delivers")
    lines.append("- report.md (this file)")
    lines.append("- report.pdf (text-only conversion if ReportLab is installed)")
    lines.append("- ui_payload.json (UI-ready structured output)")
    lines.append("- bundle/ (copied tables + figures for sharing)")
    lines.append("")
    lines.append("## Step-by-step summary (with captions)")
    lines.append("1) **Output structure** — *Caption:* Create delivery folder structure (tables/figures/bundle).")
    lines.append("2) **Load latent + UI (Stage-11)** — *Caption:* Summarize clusters, top genes, enrichment, and UMAPs.")
    lines.append("3) **Load Bayesian network (Stage-10)** — *Caption:* Extract learned edges and highlight UPR relationships.")
    lines.append("4) **Load causal IV (Stage-9)** — *Caption:* Summarize IV estimates + diagnostics and first-stage plot.")
    lines.append("5) **Load QC (Stage-8)** — *Caption:* Use high-confidence predicted perturbations as reliable candidates.")
    lines.append("6) **Narrative report** — *Caption:* Human-readable summary with figure captions.")
    lines.append("7) **UI payload** — *Caption:* Export cards + tables + figure refs.")
    lines.append("8) **PDF** — *Caption:* Export PDF for handoff (if available).")
    lines.append("")

    # ---- Stage 8 ----
    lines.append("## QC (Stage-8) — high-confidence predicted perturbations")
    lines.append("**Caption:** QC-filtered predicted perturbations (conf >= threshold, min cells per pert) as reliable candidates.")
    df_hcp = s8.get("highconf_perts")
    if df_hcp is not None and not df_hcp.empty:
        lines.append("")
        lines.append(f"Top {min(topn_perts, len(df_hcp))} high-confidence perturbations:")
        lines.append("")
        lines.append(md_table(df_hcp, max_rows=topn_perts))
    else:
        lines.append("")
        lines.append("_(No Stage-8 high-confidence perturbation table found.)_")

    # ---- Stage 9 ----
    lines.append("")
    lines.append("## Causal inference (Stage-9) — IV results")
    lines.append("**Caption:** 2SLS/IV estimation summary and diagnostics (first stage + second stage).")
    df_iv = s9.get("iv_results")
    df_diag = s9.get("diagnostics")
    if df_iv is not None and not df_iv.empty:
        lines.append("")
        lines.append("IV results (head):")
        lines.append("")
        lines.append(md_table(df_iv, max_rows=12))
    else:
        lines.append("")
        lines.append("_(No IV results table found.)_")

    if df_diag is not None and not df_diag.empty:
        lines.append("")
        lines.append("Diagnostics (head):")
        lines.append("")
        lines.append(md_table(df_diag, max_rows=12))

    # ---- Stage 10 ----
    lines.append("")
    lines.append("## Bayesian network (Stage-10) — structure learning")
    lines.append("**Caption:** Learned BN edges using HillClimb + BIC; highlights potential causal/associational links.")
    df_edges = s10.get("edges")
    if df_edges is not None and not df_edges.empty:
        lines.append("")
        lines.append("Top BN edges (head):")
        lines.append("")
        lines.append(md_table(df_edges, max_rows=20))
    else:
        lines.append("")
        lines.append("_(No BN edges table found.)_")

    # ---- Stage 11 ----
    lines.append("")
    lines.append("## Latent AI & UI (Stage-11) — clusters, markers, enrichment")
    lines.append("**Caption:** Leiden clusters in latent space; cluster markers and perturbation enrichment.")
    df_cs = s11.get("cluster_summary")
    df_ctg = s11.get("cluster_top_genes")
    df_enr = s11.get("cluster_enrichment")

    if df_cs is not None and not df_cs.empty:
        lines.append("")
        lines.append(f"Cluster summary (top {min(topn_clusters, len(df_cs))}):")
        lines.append("")
        lines.append(md_table(df_cs, max_rows=topn_clusters))
    else:
        lines.append("")
        lines.append("_(No cluster summary found.)_")

    if df_ctg is not None and not df_ctg.empty:
        lines.append("")
        lines.append("Top genes per cluster (head):")
        lines.append("")
        lines.append(md_table(df_ctg, max_rows=20))

    if df_enr is not None and not df_enr.empty:
        lines.append("")
        lines.append("Perturbation ↔ cluster enrichment (head):")
        lines.append("")
        lines.append(md_table(df_enr, max_rows=20))

    # ---- Figures ----
    lines.append("")
    lines.append("## Key figures")
    lines.append("**Caption:** Copied figures from Stage-8/9/10/11 into this handoff package for sharing.")
    fig_list = sorted([p.name for p in artifacts.figures_dir.glob("*") if p.is_file()])
    if fig_list:
        lines.append("")
        for f in fig_list:
            lines.append(f"- `{f}`")
    else:
        lines.append("")
        lines.append("_(No figures were copied into Stage-12 figures folder.)_")

    # ---- Next steps ----
    lines.append("")
    lines.append("## Where you are in the roadmap")
    lines.append("- Optimization & Cell Type: **Stage-8 ✅**")
    lines.append("- Mixscape Analysis: **Stage-2 → Stage-4 ✅**")
    lines.append("- Causal Inference (IV): **Stage-9 ✅**")
    lines.append("- Bayesian Networks: **Stage-10 ✅**")
    lines.append("- Latent AI & UI: **Stage-11 ✅**")
    lines.append("- LLM & Handoff and delivery: **Stage-12 ✅ (this stage)**")
    lines.append("")
    # lines.append("## Handoff package contents")
    # lines.append(f"- Output directory: `{artifacts.out_dir}`")
    # lines.append(f"- Tables: `{artifacts.tables_dir}`")
    # lines.append(f"- Figures: `{artifacts.figures_dir}`")
    # lines.append(f"- Bundle: `{artifacts.bundle_dir}`")
    # lines.append("")

    return "\n".join(lines) + "\n"


# -----------------------------
# UI payload builder
# -----------------------------
def build_ui_payload(
    artifacts: Stage12Artifacts,
    s8: Dict[str, Optional[pd.DataFrame]],
    s9: Dict[str, Optional[pd.DataFrame]],
    s10: Dict[str, Optional[pd.DataFrame]],
    s11: Dict[str, Optional[pd.DataFrame]],
    topn_perts: int,
    topn_clusters: int,
) -> Dict:
    """
    UI payload intended for downstream UI / LLM / dashboard.

    Includes:
      - top perturbations (high conf)
      - cluster cards
      - IV headline results
      - BN edges
      - figure index
      - table index
    """
    df_hcp = s8.get("highconf_perts")
    df_cs = s11.get("cluster_summary")
    df_cards = s11.get("ui_cards")
    df_iv = s9.get("iv_results")
    df_diag = s9.get("diagnostics")
    df_edges = s10.get("edges")

    def df_to_records(df: Optional[pd.DataFrame], n: int = 50) -> List[Dict]:
        if df is None or df.empty:
            return []
        return df.head(n).to_dict(orient="records")

    figures = [p.name for p in sorted(artifacts.figures_dir.glob("*")) if p.is_file()]
    tables = [p.name for p in sorted(artifacts.tables_dir.glob("*.tsv")) if p.is_file()]

    payload = {
        "stage": 12,
        "title": "LLM & Handoff Delivery",
        "summary": {
            "topn_perts": topn_perts,
            "topn_clusters": topn_clusters,
            "figures_count": len(figures),
            "tables_count": len(tables),
        },
        "cards": {
            "top_highconf_perts": df_to_records(df_hcp, n=topn_perts),
            "cluster_summary": df_to_records(df_cs, n=topn_clusters),
            "ui_cards": df_to_records(df_cards, n=200),
            "iv_results_head": df_to_records(df_iv, n=25),
            "iv_diagnostics_head": df_to_records(df_diag, n=25),
            "bn_edges_head": df_to_records(df_edges, n=50),
        },
        "artifacts": {
            "report_md": "stage12_report.md",
            "report_pdf": "stage12_report.pdf",
            "tables_dir": "tables",
            "figures_dir": "figures",
            "bundle_dir": "bundle",
            "figures": figures,
            "tables": tables,
        },
    }
    return payload


# -----------------------------
# Main
# -----------------------------
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Stage-12: LLM & Handoff + Delivery (all artifacts)")
    ap.add_argument("--stage8_dir", required=True, help="processed_stage8 directory")
    ap.add_argument("--stage9_dir", required=True, help="processed_stage9 directory")
    ap.add_argument("--stage10_dir", required=True, help="processed_stage10 directory")
    ap.add_argument("--stage11_dir", required=True, help="processed_stage11 directory")
    ap.add_argument("--out_dir", required=True, help="Output directory for stage12")

    ap.add_argument("--topn_perts", type=int, default=20, help="Top N perturbations to show in report/UI")
    ap.add_argument("--topn_clusters", type=int, default=15, help="Top N clusters to show in report/UI")
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    inputs = Stage12Inputs(
        stage8_dir=Path(args.stage8_dir),
        stage9_dir=Path(args.stage9_dir),
        stage10_dir=Path(args.stage10_dir),
        stage11_dir=Path(args.stage11_dir),
    )

    out_dir = ensure_dir(Path(args.out_dir))
    artifacts = Stage12Artifacts(
        out_dir=out_dir,
        tables_dir=ensure_dir(out_dir / "tables"),
        figures_dir=ensure_dir(out_dir / "figures"),
        bundle_dir=ensure_dir(out_dir / "bundle"),
    )

    log("STEP 1/8: Create output structure")
    log("Caption: Create delivery folder structure (tables/figures/bundle).")

    log("STEP 2/8: Load Stage-11 outputs")
    log("Caption: Summarize clusters, top genes, enrichment, and UMAPs.")
    s11 = load_stage11(inputs.stage11_dir)

    log("STEP 3/8: Load Stage-10 outputs")
    log("Caption: Extract BN edges and highlight UPR relationships.")
    s10 = load_stage10(inputs.stage10_dir)

    log("STEP 4/8: Load Stage-9 outputs")
    log("Caption: Summarize IV estimates + diagnostics and first-stage plot.")
    s9 = load_stage9(inputs.stage9_dir)

    log("STEP 5/8: Load Stage-8 outputs")
    log("Caption: Use high-confidence predicted perturbations as reliable candidates.")
    s8 = load_stage8(inputs.stage8_dir)

    # Copy important tables into Stage-12 tables
    log("Copying key tables into Stage-12 tables/ (for handoff)")
    copy_map = [
        (inputs.stage8_dir / "tables" / "stage8_cells_qc.tsv", artifacts.tables_dir / "stage8_cells_qc.tsv"),
        (inputs.stage8_dir / "tables" / "stage8_predicted_perturbation_qc.tsv", artifacts.tables_dir / "stage8_predicted_perturbation_qc.tsv"),
        (inputs.stage8_dir / "tables" / "stage8_highconf_perts_ge_50.tsv", artifacts.tables_dir / "stage8_highconf_perts_ge_50.tsv"),
        (inputs.stage9_dir / "tables" / "stage9_iv_results.tsv", artifacts.tables_dir / "stage9_iv_results.tsv"),
        (inputs.stage9_dir / "tables" / "stage9_diagnostics.tsv", artifacts.tables_dir / "stage9_diagnostics.tsv"),
        (inputs.stage10_dir / "tables" / "stage10_edges.tsv", artifacts.tables_dir / "stage10_edges.tsv"),
        (inputs.stage10_dir / "tables" / "stage10_nodes.tsv", artifacts.tables_dir / "stage10_nodes.tsv"),
        (inputs.stage11_dir / "tables" / "stage11_cluster_summary.tsv", artifacts.tables_dir / "stage11_cluster_summary.tsv"),
        (inputs.stage11_dir / "tables" / "stage11_cluster_top_genes.tsv", artifacts.tables_dir / "stage11_cluster_top_genes.tsv"),
        (inputs.stage11_dir / "tables" / "stage11_perturbation_cluster_enrichment.tsv", artifacts.tables_dir / "stage11_perturbation_cluster_enrichment.tsv"),
        (inputs.stage11_dir / "tables" / "stage11_ui_cards.tsv", artifacts.tables_dir / "stage11_ui_cards.tsv"),
    ]
    for src, dst in copy_map:
        safe_copy(src, dst)

    # Copy figures (best-effort)
    log("Copying figures into Stage-12 figures/ (for handoff)")
    fig_sources = []
    fig_sources += list((inputs.stage8_dir / "figures").glob("*.png"))
    fig_sources += list((inputs.stage9_dir / "figures").glob("*.png"))
    fig_sources += list((inputs.stage10_dir / "figures").glob("*.png"))
    fig_sources += list((inputs.stage11_dir / "figures").glob("*.png"))
    for src in fig_sources:
        safe_copy(src, artifacts.figures_dir / src.name)

    # Bundle = mirror tables + figures + final outputs
    log("STEP 6/8: Build Markdown report")
    log("Caption: Narrative report with captions and links to copied artifacts.")
    md_text = build_report_md(
        inputs=inputs,
        artifacts=artifacts,
        s8=s8,
        s9=s9,
        s10=s10,
        s11=s11,
        topn_perts=int(args.topn_perts),
        topn_clusters=int(args.topn_clusters),
    )
    report_md_path = artifacts.out_dir / "stage12_report.md"
    report_md_path.write_text(md_text, encoding="utf-8")
    log(f"[OK] report.md -> {report_md_path}")

    log("STEP 7/8: Export UI payload JSON")
    log("Caption: UI-ready cards + tables + figure references for downstream app/hand-off.")
    payload = build_ui_payload(
        artifacts=artifacts,
        s8=s8,
        s9=s9,
        s10=s10,
        s11=s11,
        topn_perts=int(args.topn_perts),
        topn_clusters=int(args.topn_clusters),
    )
    ui_path = artifacts.out_dir / "ui_payload.json"
    ui_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    log(f"[OK] ui_payload.json -> {ui_path}")

    log("STEP 8/8: Render PDF (optional)")
    log("Caption: Produce a PDF report for non-technical sharing (if ReportLab is available).")
    pdf_path = artifacts.out_dir / "stage12_report.pdf"
    ok_pdf = md_to_simple_pdf(md_text, pdf_path, title="Stage-12 Handoff Report")
    if ok_pdf:
        log(f"[OK] report.pdf -> {pdf_path}")
    else:
        warn("PDF export skipped (ReportLab not installed). report.md is still complete.")

    # Build bundle
    log("Building bundle/ (copy tables, figures, and reports)")
    safe_copy(report_md_path, artifacts.bundle_dir / report_md_path.name)
    if pdf_path.exists():
        safe_copy(pdf_path, artifacts.bundle_dir / pdf_path.name)
    safe_copy(ui_path, artifacts.bundle_dir / ui_path.name)

    # Copy whole tables and figures folders
    # (Keep it simple: copy file-by-file)
    ensure_dir(artifacts.bundle_dir / "tables")
    ensure_dir(artifacts.bundle_dir / "figures")
    for p in artifacts.tables_dir.glob("*.tsv"):
        safe_copy(p, artifacts.bundle_dir / "tables" / p.name)
    for p in artifacts.figures_dir.glob("*"):
        if p.is_file():
            safe_copy(p, artifacts.bundle_dir / "figures" / p.name)

    log(f"[DONE] Stage-12 complete -> {artifacts.out_dir}")
    log("Deliverables: stage12_report.md, stage12_report.pdf (optional), ui_payload.json, bundle/")


if __name__ == "__main__":
    main()

