#!/usr/bin/env python3
"""
ipaa_report.py

Generate an IPAA-style single-cohort report (HTML + Markdown) from an M4.py run directory.

It expects the M4 output layout (files/subfolders). Missing sections are skipped gracefully.

Usage:
  python IPAA/ipaa_report.py --run-dir /path/to/IPAA_Results --out /path/to/IPAA_Results/IPAA_REPORT.html

Common:
  python IPAA/ipaa_report.py --run-dir /mnt/d/temp/IPAA_Results2

Optional:
  python IPAA/ipaa_report.py --run-dir OUT --open

Notes:
- This is stand-alone (no MDP imports).
- It reuses the same visual/card style conventions as your existing report.py template (clean cards, compact tables).
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from jinja2 import Template
import matplotlib.pyplot as plt  # noqa
import webbrowser


def _now_utc() -> str:
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")


def _read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(str(path))
    suf = path.suffix.lower()
    sep = "\t" if suf in {".tsv", ".txt"} else ","
    return pd.read_csv(path, sep=sep)


def _read_table_indexed(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(str(path))
    suf = path.suffix.lower()
    sep = "\t" if suf in {".tsv", ".txt"} else ","
    return pd.read_csv(path, sep=sep, index_col=0)


def _try_read_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _safe_float(x) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def _df_head(df: pd.DataFrame, n: int = 15) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    return df.head(n).copy()


def _df_to_html_table(df: pd.DataFrame, max_rows: int = 20) -> str:
    if df is None or df.empty:
        return "<div class='muted'>No data</div>"
    d = df.copy()
    if len(d) > max_rows:
        d = d.head(max_rows)
    return d.to_html(index=False, escape=True, classes="tbl")


def _df_to_md_table(df: pd.DataFrame, max_rows: int = 20) -> str:
    if df is None or df.empty:
        return "_No data_"
    d = df.copy()
    if len(d) > max_rows:
        d = d.head(max_rows)
    return d.to_markdown(index=False)


def _fig_to_data_uri(fig) -> str:
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=170, bbox_inches="tight")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _volcano_data_uri(df: pd.DataFrame, effect_col: str, p_col: str, label_col: str, top_n: int = 12) -> Optional[str]:
    if df is None or df.empty or effect_col not in df.columns or p_col not in df.columns:
        return None

    d = df[[effect_col, p_col, label_col]].copy()
    d[effect_col] = pd.to_numeric(d[effect_col], errors="coerce")
    d[p_col] = pd.to_numeric(d[p_col], errors="coerce")
    d = d.dropna()
    if d.empty:
        return None

    d["mlog10p"] = -np.log10(np.clip(d[p_col].values, 1e-300, 1.0))

    # pick labels by smallest p among largest |effect|
    d2 = d.assign(abs_eff=d[effect_col].abs()).sort_values(["abs_eff", p_col], ascending=[False, True]).head(top_n)

    fig = plt.figure(figsize=(7.2, 4.6))
    ax = fig.add_subplot(111)
    ax.scatter(d[effect_col].values, d["mlog10p"].values, s=10, alpha=0.7)
    ax.set_xlabel(effect_col)
    ax.set_ylabel(f"-log10({p_col})")
    ax.axhline(-np.log10(0.05), linestyle="--", linewidth=1)

    for _, r in d2.iterrows():
        ax.text(r[effect_col], r["mlog10p"], str(r[label_col])[:40], fontsize=7)

    fig.tight_layout()
    uri = _fig_to_data_uri(fig)
    plt.close(fig)
    return uri


def _barh_data_uri(df: pd.DataFrame, label_col: str, value_col: str, top_n: int = 20, title: str = "") -> Optional[str]:
    if df is None or df.empty or label_col not in df.columns or value_col not in df.columns:
        return None


    d = df[[label_col, value_col]].copy()
    d[value_col] = pd.to_numeric(d[value_col], errors="coerce")
    d = d.dropna()
    if d.empty:
        return None
    d = d.sort_values(value_col, ascending=True).tail(top_n)

    fig = plt.figure(figsize=(7.2, 4.8))
    ax = fig.add_subplot(111)
    ax.barh(d[label_col].astype(str).values, d[value_col].values)
    if title:
        ax.set_title(title)
    ax.set_xlabel(value_col)
    fig.tight_layout()
    uri = _fig_to_data_uri(fig)
    plt.close(fig)
    return uri


def _heatmap_data_uri(mat: pd.DataFrame, top_rows: int = 35, top_cols: int = 60, title: str = "") -> Optional[str]:
    if mat is None or mat.empty:
        return None

    M = mat.copy()
    # auto-orient: expect samples x features. If more rows than cols by a lot, transpose for display.
    if M.shape[0] > M.shape[1] * 2:
        M = M.T

    # keep most variable rows
    var = M.var(axis=1, numeric_only=True).sort_values(ascending=False)
    keep_rows = var.index[: min(top_rows, len(var))]
    M = M.loc[keep_rows]

    # cap columns
    if M.shape[1] > top_cols:
        M = M.iloc[:, :top_cols]

    fig = plt.figure(figsize=(9.0, 6.0))
    ax = fig.add_subplot(111)
    im = ax.imshow(M.values, aspect="auto")
    ax.set_yticks(range(M.shape[0]))
    ax.set_yticklabels([str(x)[:50] for x in M.index], fontsize=7)
    ax.set_xticks(range(M.shape[1]))
    ax.set_xticklabels([str(x)[:30] for x in M.columns], fontsize=6, rotation=90)
    if title:
        ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    fig.tight_layout()
    uri = _fig_to_data_uri(fig)
    plt.close(fig)
    return uri


def _open_in_browser(path: Path) -> None:
    try:
        
        webbrowser.open(f"file://{path.resolve()}")
    except Exception:
        pass


CSS = r"""
:root{
  --bg:#0b0e14; --card:#111827; --muted:#94a3b8; --fg:#e5e7eb;
  --accent:#60a5fa; --accent2:#34d399; --warn:#fbbf24; --bad:#fb7185;
  --line:#1f2937;
  --mono: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
  --sans: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, "Apple Color Emoji","Segoe UI Emoji";
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font-family:var(--sans);line-height:1.4}
.container{max-width:1180px;margin:0 auto;padding:22px}
.header{display:flex;justify-content:space-between;align-items:flex-end;gap:14px;margin-bottom:18px}
.h-title{font-size:22px;font-weight:700}
.h-sub{color:var(--muted);font-family:var(--mono);font-size:12px;margin-top:6px}
.grid{display:grid;grid-template-columns:repeat(12,1fr);gap:12px}
.card{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:14px}
.card h2{font-size:14px;margin:0 0 10px 0}
.kpis{display:flex;gap:10px;flex-wrap:wrap}
.pill{display:inline-flex;align-items:center;gap:8px;border:1px solid var(--line);border-radius:999px;padding:6px 10px;color:var(--fg);background:rgba(255,255,255,0.03);font-family:var(--mono);font-size:12px}
.pill b{color:var(--accent)}
.muted{color:var(--muted)}
.tbl{width:100%;border-collapse:collapse;font-size:12px}
.tbl th,.tbl td{border-bottom:1px solid var(--line);padding:6px 8px;text-align:left;vertical-align:top}
.tbl th{color:var(--muted);font-weight:600}
.row{display:flex;gap:12px;flex-wrap:wrap}
img{max-width:100%;border-radius:12px;border:1px solid var(--line)}
hr{border:none;border-top:1px solid var(--line);margin:12px 0}
.small{font-size:12px;color:var(--muted)}
pre{background:#0a0d12;border:1px solid var(--line);border-radius:12px;padding:10px;overflow:auto;font-family:var(--mono);font-size:12px;color:var(--fg)}
"""

HTML_TEMPLATE = r"""
<!doctype html>
<html>
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>{{ title }}</title>
<style>{{ css }}</style>
</head>
<body>
<div class="container">
  <div class="header">
    <div>
      <div class="h-title">{{ title }}</div>
      <div class="h-sub">Generated {{ generated }} · Run dir: {{ run_dir }}</div>
    </div>
    <div class="kpis">
      {% for kpi in kpis %}
      <span class="pill"><b>{{ kpi.key }}</b> {{ kpi.val }}</span>
      {% endfor %}
    </div>
  </div>

  <div class="grid">

    <div class="card" style="grid-column: span 12;">
      <h2>Executive summary</h2>
      <div class="row">
        <div style="flex:2;min-width:340px">
          <ul>
            {% for b in bullets %}
            <li>{{ b }}</li>
            {% endfor %}
          </ul>
          <div class="small">Interpretation note: these activities are expression-derived proxies (no phosphoproteomics). Treat as hypothesis generators.</div>
        </div>
        <div style="flex:1;min-width:280px">
          {% if volcano_uri %}
            <img src="{{ volcano_uri }}" alt="volcano"/>
          {% else %}
            <div class="muted">No volcano plot available.</div>
          {% endif %}
        </div>
      </div>
    </div>

    <div class="card" style="grid-column: span 6;">
      <h2>Groups & inputs</h2>
      <div class="small">Input: <span class="muted">{{ input_path }}</span></div>
      <hr/>
      {{ group_table | safe }}
      <hr/>
      <div class="small">Key files detected:</div>
      <pre>{{ files_list }}</pre>
    </div>

    <div class="card" style="grid-column: span 6;">
      <h2>IPAA pathway activity heatmap</h2>
      {% if heatmap_uri %}
        <img src="{{ heatmap_uri }}" alt="heatmap"/>
      {% else %}
        <div class="muted">No pathway activity matrix found.</div>
      {% endif %}
    </div>

    <div class="card" style="grid-column: span 12;">
      <h2>Top differential pathways (IPAA)</h2>
      <div class="row">
        <div style="flex:1;min-width:360px">
          <div class="small muted">Up in case</div>
          {{ top_up_table | safe }}
        </div>
        <div style="flex:1;min-width:360px">
          <div class="small muted">Up in control</div>
          {{ top_dn_table | safe }}
        </div>
      </div>
    </div>

    <div class="card" style="grid-column: span 12;">
      <h2>GSEA prerank (MSigDB C2 CP)</h2>
      <div class="row">
        <div style="flex:1;min-width:360px">
          <div class="small muted">Top NES (positive)</div>
          {{ gsea_pos_table | safe }}
        </div>
        <div style="flex:1;min-width:360px">
          <div class="small muted">Top NES (negative)</div>
          {{ gsea_neg_table | safe }}
        </div>
      </div>
    </div>

    <div class="card" style="grid-column: span 12;">
      <h2>Regulator activity layers</h2>
      <div class="row">
        <div style="flex:1;min-width:360px">
          <div class="small muted">TF activity (ULM)</div>
          {% if tf_ulm_uri %}<img src="{{ tf_ulm_uri }}" alt="tf_ulm"/>{% endif %}
          {{ tf_ulm_table | safe }}
        </div>
        <div style="flex:1;min-width:360px">
          <div class="small muted">Kinase activity proxy</div>
          {% if kin_uri %}<img src="{{ kin_uri }}" alt="kin"/>{% endif %}
          {{ kin_table | safe }}
        </div>
      </div>
      <hr/>
      <div class="row">
        <div style="flex:1;min-width:360px">
          <div class="small muted">PROGENy signaling</div>
          {% if prog_uri %}<img src="{{ prog_uri }}" alt="progeny"/>{% endif %}
          {{ prog_table | safe }}
        </div>
        <div style="flex:1;min-width:360px">
          <div class="small muted">Intercell categories</div>
          {% if inter_uri %}<img src="{{ inter_uri }}" alt="intercell"/>{% endif %}
          {{ inter_table | safe }}
        </div>
      </div>
    </div>

    <div class="card" style="grid-column: span 12;">
      <h2>Run manifest</h2>
      <pre>{{ manifest_json }}</pre>
    </div>

  </div>
</div>
</body>
</html>
"""


@dataclass
class KPI:
    key: str
    val: str


def _collect_files(run_dir: Path) -> List[str]:
    out = []
    for p in sorted(run_dir.rglob("*")):
        if p.is_file() and p.stat().st_size > 0:
            rel = str(p.relative_to(run_dir))
            if len(rel) <= 120:
                out.append(rel)
    # cap for readability
    if len(out) > 80:
        out = out[:80] + [f"... ({len(out)-80} more)"]
    return out


def _load_group_table(run_dir: Path) -> Tuple[pd.DataFrame, str]:
    """
    Prefer pathway_activity_meta.tsv if present; otherwise infer from RUN_MANIFEST.json if it contains regex.
    """
    meta = run_dir / "pathway_activity_meta.tsv"
    if meta.exists():
        df = _read_table(meta)
        # expect columns: sample, group
        cols = {c.lower(): c for c in df.columns}
        if "group" in cols:
            gcol = cols["group"]
            df[gcol] = df[gcol].astype(str).str.lower()
            tab = df.groupby(gcol).size().reset_index(name="n").rename(columns={gcol: "group"})
            return tab, gcol

    # fallback: no meta — return empty
    return pd.DataFrame({"group": [], "n": []}), "group"


def _load_pathway_stats(run_dir: Path) -> pd.DataFrame:
    p = run_dir / "pathway_stats.tsv"
    if not p.exists():
        return pd.DataFrame()
    df = _read_table(p)
    # normalize column names
    ren = {c: c.strip() for c in df.columns}
    df = df.rename(columns=ren)
    # ensure expected
    if "pathway" not in df.columns:
        # maybe first col is pathway
        df = df.rename(columns={df.columns[0]: "pathway"})
    for c in ["delta_activity", "t", "p", "FDR"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _split_top(df: pd.DataFrame, label: str, n: int = 20) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if df is None or df.empty:
        return pd.DataFrame(), pd.DataFrame()
    d = df.copy()
    if "FDR" in d.columns:
        d = d.sort_values(["FDR", "p"], ascending=[True, True])
    elif "p" in d.columns:
        d = d.sort_values("p", ascending=True)
    if "t" in d.columns:
        up = d[d["t"] > 0].head(n)
        dn = d[d["t"] < 0].head(n)
    else:
        up = d.head(n)
        dn = pd.DataFrame()
    keep = ["pathway"] + [c for c in ["delta_activity", "t", "p", "FDR"] if c in d.columns]
    return up[keep].copy(), dn[keep].copy()


def _load_gsea(run_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # prefer gsea_c2cp/gsea_prerank_results.tsv
    p1 = run_dir / "gsea_c2cp" / "gsea_prerank_results.tsv"
    p2 = run_dir / "gsea_c2cp_top.tsv"
    if p1.exists():
        df = _read_table(p1)
        # if index col included as first unnamed
        if "Term" in df.columns:
            df = df.rename(columns={"Term": "pathway"})
        if "NES" in df.columns:
            df["NES"] = pd.to_numeric(df["NES"], errors="coerce")
        if "FDR q-val" in df.columns:
            df = df.rename(columns={"FDR q-val": "FDR"})
        if "FDR" in df.columns:
            df["FDR"] = pd.to_numeric(df["FDR"], errors="coerce")
    elif p2.exists():
        df = _read_table(p2)
        if "NES" in df.columns:
            df["NES"] = pd.to_numeric(df["NES"], errors="coerce")
        if "FDR" in df.columns:
            df["FDR"] = pd.to_numeric(df["FDR"], errors="coerce")
        if "pathway" not in df.columns and "Term" in df.columns:
            df = df.rename(columns={"Term": "pathway"})
    else:
        return pd.DataFrame(), pd.DataFrame()

    if df.empty or "NES" not in df.columns:
        return pd.DataFrame(), pd.DataFrame()

    pos = df.sort_values("NES", ascending=False).head(20)
    neg = df.sort_values("NES", ascending=True).head(20)

    keep = [c for c in ["pathway", "NES", "pval", "FDR"] if c in df.columns]
    return pos[keep].copy(), neg[keep].copy()


def _load_diff_table(run_dir: Path, rel: str, key_col: str) -> pd.DataFrame:
    p = run_dir / rel
    if not p.exists():
        return pd.DataFrame()
    df = _read_table(p)
    if key_col not in df.columns:
        df = df.rename(columns={df.columns[0]: key_col})
    for c in ["delta_activity", "t", "p", "FDR"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _top_diff(df: pd.DataFrame, key_col: str, n: int = 20) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    d = df.copy()
    sort_cols = [c for c in ["FDR", "p"] if c in d.columns]
    if sort_cols:
        d = d.sort_values(sort_cols, ascending=[True]*len(sort_cols))
    keep = [key_col] + [c for c in ["delta_activity", "t", "p", "FDR"] if c in d.columns]
    return d[keep].head(n).copy()


def build_ipaa_report(run_dir: Path, out_html: Optional[Path] = None, out_md: Optional[Path] = None) -> Tuple[Optional[Path], Optional[Path]]:
    run_dir = Path(run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(str(run_dir))

    title = "IPAA – Single-cohort Activity Report"
    generated = _now_utc()

    manifest = _try_read_json(run_dir / "RUN_MANIFEST.json") or {}
    input_path = str(manifest.get("expr_path", "")) or "N/A"

    # KPIs
    group_tab, _ = _load_group_table(run_dir)
    n_case = int(group_tab.loc[group_tab["group"].astype(str).str.lower().eq("case"), "n"].sum()) if not group_tab.empty else 0
    n_ctrl = int(group_tab.loc[group_tab["group"].astype(str).str.lower().eq("control"), "n"].sum()) if not group_tab.empty else 0

    # Determine genes/samples from expression_used if exists
    expr_used = run_dir / "expression_used.tsv"
    n_samples = n_genes = None
    if expr_used.exists():
        X = _read_table_indexed(expr_used)
        n_samples, n_genes = int(X.shape[0]), int(X.shape[1])

    pstats = _load_pathway_stats(run_dir)
    sig = int((pstats["FDR"] <= 0.05).sum()) if (not pstats.empty and "FDR" in pstats.columns) else 0

    kpis = [
        KPI("Samples", str(n_samples) if n_samples is not None else "—"),
        KPI("Genes", str(n_genes) if n_genes is not None else "—"),
        KPI("Case", str(n_case) if n_case else "—"),
        KPI("Control", str(n_ctrl) if n_ctrl else "—"),
        KPI("Sig pathways (FDR≤0.05)", str(sig) if sig else "0"),
    ]

    files_list = "\n".join(_collect_files(run_dir))

    # Executive bullets (simple, data-driven)
    bullets: List[str] = []
    if not pstats.empty and "t" in pstats.columns:
        up, dn = _split_top(pstats, "pathway", n=5)
        if not up.empty:
            bullets.append("Top case-associated pathways: " + ", ".join(up["pathway"].astype(str).tolist()))
        if not dn.empty:
            bullets.append("Top control-associated pathways: " + ", ".join(dn["pathway"].astype(str).tolist()))
    else:
        bullets.append("No pathway differential table found (pathway_stats.tsv).")

    # Load activity matrix for heatmap
    heatmap_uri = None
    p_activity = run_dir / "pathway_activity.tsv"
    if p_activity.exists():
        try:
            A = _read_table_indexed(p_activity)
            heatmap_uri = _heatmap_data_uri(A, title="Pathway activity (top variable)")
        except Exception:
            heatmap_uri = None

    volcano_uri = _volcano_data_uri(pstats, "delta_activity", "p", "pathway") if not pstats.empty else None

    top_up, top_dn = _split_top(pstats, "pathway", n=20)
    top_up_table = _df_to_html_table(top_up)
    top_dn_table = _df_to_html_table(top_dn)

    # GSEA
    gsea_pos, gsea_neg = _load_gsea(run_dir)
    gsea_pos_table = _df_to_html_table(gsea_pos)
    gsea_neg_table = _df_to_html_table(gsea_neg)

    # TF ULM
    tf_ulm = _load_diff_table(run_dir, "tf_activity/tf_ulm_diff.tsv", "regulator")
    tf_ulm_top = _top_diff(tf_ulm, "regulator", n=20)
    tf_ulm_uri = _barh_data_uri(tf_ulm_top.iloc[::-1], "regulator", "delta_activity", title="TF Δactivity (ULM)") if not tf_ulm_top.empty else None
    tf_ulm_table = _df_to_html_table(tf_ulm_top)

    # Kinase
    kin = _load_diff_table(run_dir, "kinase_activity/kinase_viper_diff.tsv", "regulator")
    kin_top = _top_diff(kin, "regulator", n=20)
    kin_uri = _barh_data_uri(kin_top.iloc[::-1], "regulator", "delta_activity", title="Kinase Δactivity") if not kin_top.empty else None
    kin_table = _df_to_html_table(kin_top)

    # PROGENy
    prog = _load_diff_table(run_dir, "signaling_progeny/progeny_ulm_diff.tsv", "regulator")
    prog_top = _top_diff(prog, "regulator", n=20)
    prog_uri = _barh_data_uri(prog_top.iloc[::-1], "regulator", "delta_activity", title="PROGENy Δactivity") if not prog_top.empty else None
    prog_table = _df_to_html_table(prog_top)

    # Intercell
    inter = _load_diff_table(run_dir, "intercell_categories/intercell_ulm_diff.tsv", "regulator")
    inter_top = _top_diff(inter, "regulator", n=20)
    inter_uri = _barh_data_uri(inter_top.iloc[::-1], "regulator", "delta_activity", title="Intercell Δactivity") if not inter_top.empty else None
    inter_table = _df_to_html_table(inter_top)

    group_table_html = _df_to_html_table(group_tab, max_rows=10)

    manifest_json = json.dumps(manifest, indent=2) if manifest else "{}"

    # Markdown companion (quick)
    md_lines = []
    md_lines.append(f"# {title}")
    md_lines.append(f"- Generated: {generated}")
    md_lines.append(f"- Run dir: `{run_dir}`")
    md_lines.append("")
    md_lines.append("## Summary")
    for b in bullets:
        md_lines.append(f"- {b}")
    md_lines.append("")
    md_lines.append("## Groups")
    md_lines.append(_df_to_md_table(group_tab, max_rows=20))
    md_lines.append("")
    md_lines.append("## Top differential pathways (IPAA)")
    md_lines.append("### Up in case")
    md_lines.append(_df_to_md_table(top_up, max_rows=20))
    md_lines.append("")
    md_lines.append("### Up in control")
    md_lines.append(_df_to_md_table(top_dn, max_rows=20))
    md_lines.append("")
    md_lines.append("## GSEA prerank (C2 CP)")
    md_lines.append("### Positive NES")
    md_lines.append(_df_to_md_table(gsea_pos, max_rows=20))
    md_lines.append("")
    md_lines.append("### Negative NES")
    md_lines.append(_df_to_md_table(gsea_neg, max_rows=20))
    md_lines.append("")
    md_lines.append("## TF activity (ULM)")
    md_lines.append(_df_to_md_table(tf_ulm_top, max_rows=20))
    md_lines.append("")
    md_lines.append("## Kinase activity proxy")
    md_lines.append(_df_to_md_table(kin_top, max_rows=20))
    md_lines.append("")
    md_lines.append("## PROGENy signaling")
    md_lines.append(_df_to_md_table(prog_top, max_rows=20))
    md_lines.append("")
    md_lines.append("## Intercell categories")
    md_lines.append(_df_to_md_table(inter_top, max_rows=20))
    md_lines.append("")
    md_lines.append("## Manifest")
    md_lines.append("```json")
    md_lines.append(manifest_json)
    md_lines.append("```")
    md_content = "\n".join(md_lines)

    if out_md is None:
        out_md = run_dir / "IPAA_REPORT.md"
    out_md.write_text(md_content, encoding="utf-8")

    # HTML
    if out_html is None:
        out_html = run_dir / "IPAA_REPORT.html"

    try:
        
        html = Template(HTML_TEMPLATE).render(
            title=title,
            css=CSS,
            generated=generated,
            run_dir=str(run_dir),
            kpis=kpis,
            bullets=bullets,
            volcano_uri=volcano_uri,
            input_path=input_path,
            group_table=group_table_html,
            files_list=files_list,
            heatmap_uri=heatmap_uri,
            top_up_table=top_up_table,
            top_dn_table=top_dn_table,
            gsea_pos_table=gsea_pos_table,
            gsea_neg_table=gsea_neg_table,
            tf_ulm_uri=tf_ulm_uri,
            tf_ulm_table=tf_ulm_table,
            kin_uri=kin_uri,
            kin_table=kin_table,
            prog_uri=prog_uri,
            prog_table=prog_table,
            inter_uri=inter_uri,
            inter_table=inter_table,
            manifest_json=manifest_json,
        )
        out_html.write_text(html, encoding="utf-8")
    except Exception:
        # If jinja2 missing, we still wrote markdown.
        out_html = None

    return out_html, out_md


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, help="M4.py output directory (the one that contains pathway_stats.tsv etc).")
    ap.add_argument("--out", default=None, help="Optional output HTML path (default: <run-dir>/IPAA_REPORT.html).")
    ap.add_argument("--out-md", default=None, help="Optional output Markdown path (default: <run-dir>/IPAA_REPORT.md).")
    ap.add_argument("--open", action="store_true", help="Open HTML in browser after creation.")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = Path(args.run_dir)
    out_html = Path(args.out) if args.out else None
    out_md = Path(args.out_md) if args.out_md else None
    html_path, md_path = build_ipaa_report(run_dir, out_html=out_html, out_md=out_md)
    print(f"[ipaa_report] Wrote markdown: {md_path}")
    if html_path:
        print(f"[ipaa_report] Wrote HTML: {html_path}")
        if args.open:
            _open_in_browser(html_path)
    else:
        print("[ipaa_report] HTML not written (missing jinja2). Install with: pip install -U jinja2")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
