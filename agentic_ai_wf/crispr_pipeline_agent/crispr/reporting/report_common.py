#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Shared utilities for the CRISPR Perturb-seq HTML report builder.

Provides:
- OpenAI client initialisation from environment variables (.env)
- Dynamic path discovery helpers for pipeline output artifacts
- HTML block insertion / base-report scaffolding
- Scrollable table renderer
- Graceful LLM interpretation (returns empty string when client is None)
"""

import os
import base64
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict

import pandas as pd

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
#  OpenAI client (reads OPENAI_API_KEY from .env / environment)
# ------------------------------------------------------------------ #

def get_openai_client():
    """
    Return an OpenAI client using the ``OPENAI_API_KEY`` env-var.

    Loads ``.env`` from the project root (two levels above this file)
    so that users only need to ``cp .env.example .env`` and fill in
    their key.  Returns *None* when the key is absent — callers must
    handle this gracefully (report is generated without LLM sections).
    """
    try:
        from dotenv import load_dotenv

        project_root = Path(__file__).resolve().parents[2]
        load_dotenv(project_root / ".env")
    except ImportError:
        pass  # python-dotenv not installed; fall back to env-var

    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key or api_key.startswith("sk-your"):
        logger.warning(
            "OPENAI_API_KEY not set or still placeholder — "
            "LLM interpretations will be skipped."
        )
        return None

    try:
        from openai import OpenAI
        return OpenAI(api_key=api_key)
    except Exception as exc:
        logger.warning("Failed to create OpenAI client: %s", exc)
        return None


def get_openai_model() -> str:
    """Return the model name to use for LLM calls (overridable via env)."""
    return os.environ.get("OPENAI_MODEL", "gpt-4o-mini")


# ------------------------------------------------------------------ #
#  Dynamic path discovery helpers
# ------------------------------------------------------------------ #

def find_tables(sample_dir: Path, pattern: str) -> List[Path]:
    """Glob for table files (TSV/CSV) matching *pattern* under *sample_dir*."""
    matches = sorted(sample_dir.rglob(pattern))
    return matches


def find_figures(sample_dir: Path, pattern: str) -> List[Path]:
    """Glob for figure files (PNG/HTML) matching *pattern* under *sample_dir*."""
    return sorted(sample_dir.rglob(pattern))


def first_match(sample_dir: Path, pattern: str) -> Optional[Path]:
    """Return the first file matching *pattern* under *sample_dir*, or None."""
    for p in sample_dir.rglob(pattern):
        return p
    return None


def discover_integration_method_dirs(stage3_dir: Path) -> Dict[str, Path]:
    """
    Auto-discover scRNA integration method directories under stage3_dir.

    The scRNA pipeline writes timestamped zip-extracted folders like:
        bbknn-20260212T045708Z-1-001/bbknn/figures/
        harmony-20260212T045821Z-1-001/harmony/figures/

    This function discovers them dynamically instead of hard-coding timestamps.
    """
    methods: Dict[str, Path] = {}
    method_names = ["bbknn", "harmony", "scvi", "none"]

    for method in method_names:
        for candidate in stage3_dir.glob(f"{method}*"):
            if candidate.is_dir():
                figures_path = candidate / method / "figures"
                if figures_path.is_dir():
                    label = method.upper() if method != "none" else "None"
                    if method == "scvi":
                        label = "scVI"
                    elif method == "harmony":
                        label = "Harmony"
                    elif method == "bbknn":
                        label = "BBKNN"
                    methods[label] = figures_path
                    break  # take the first match for each method

    return methods


# ------------------------------------------------------------------ #
#  Core IO helpers
# ------------------------------------------------------------------ #

def safe_read_text(path) -> str:
    p = Path(path)
    return p.read_text(encoding="utf-8").strip() if p.exists() else ""


def encode_image_to_base64(image_path) -> Optional[str]:
    p = Path(image_path)
    if not p.exists():
        return None
    return base64.b64encode(p.read_bytes()).decode("utf-8")


def safe_read_tsv(path, sep="\t") -> Optional[pd.DataFrame]:
    """Read a TSV/CSV file, returning None if the file does not exist."""
    p = Path(path)
    if not p.exists():
        return None
    try:
        return pd.read_csv(p, sep=sep)
    except Exception as exc:
        logger.warning("Failed to read %s: %s", path, exc)
        return None


def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M")


# ------------------------------------------------------------------ #
#  HTML block insertion
# ------------------------------------------------------------------ #

def insert_or_replace_block(html: str, block_id: str, block_html: str) -> str:
    start = f"<!-- STAGE:{block_id} -->"
    end = f"<!-- END:{block_id} -->"

    if start in html and end in html:
        before = html.split(start)[0]
        after = html.split(end)[1]
        return before + block_html + after

    return html.replace("</body>", block_html + "\n</body>")


def ensure_base_report(report_path) -> str:
    rp = Path(report_path)
    if not rp.exists():
        rp.parent.mkdir(parents=True, exist_ok=True)
        rp.write_text(create_base_report(), encoding="utf-8")
    return rp.read_text(encoding="utf-8")


def write_report(report_path, html: str):
    Path(report_path).write_text(html, encoding="utf-8")


# ------------------------------------------------------------------ #
#  LLM interpretation (graceful when client is None)
# ------------------------------------------------------------------ #

def llm_interpret_table(client, title: str, df: pd.DataFrame, guidance: str) -> str:
    """
    Ask the LLM for a short scientific interpretation of *df*.

    Returns an empty string when *client* is None so the report can
    still be generated without an API key.
    """
    if client is None:
        return ""

    prompt = f"""
You are writing a scientific HTML report for a CRISPR Perturb-seq simulator pipeline (transcriptomic CRISPR perturbation).

Write 3-4 concise scientific sentences interpreting the table titled: "{title}".
Rules:
- Do NOT mention LLM/AI.
- Be quantitative when possible.
- Keep it specific to CRISPR perturbation / responder modeling.

Guidance:
{guidance}

Table:
{df.to_string(index=False)}
""".strip()

    try:
        resp = client.chat.completions.create(
            model=get_openai_model(),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        return resp.choices[0].message.content.strip()
    except Exception as exc:
        logger.warning("LLM call failed for '%s': %s", title, exc)
        return ""


def llm_generate(client, prompt: str, model: Optional[str] = None,
                 temperature: float = 0.3) -> str:
    """General-purpose LLM generation. Returns empty string on failure."""
    if client is None:
        return ""
    try:
        resp = client.chat.completions.create(
            model=model or get_openai_model(),
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )
        return resp.choices[0].message.content.strip()
    except Exception as exc:
        logger.warning("LLM call failed: %s", exc)
        return ""


# ------------------------------------------------------------------ #
#  Scrollable table renderer
# ------------------------------------------------------------------ #

def df_to_scrollable_table(
    df: pd.DataFrame,
    table_id: str,
    max_rows: Optional[int] = None,
    drop_cols: Optional[list] = None,
    numeric_cols: Optional[list] = None,
    percent_cols: Optional[list] = None,
    sticky_header: bool = True,
    font_small: bool = True,
    max_height_px: int = 280,
) -> str:
    if df is None or df.empty:
        return "<p><em>No data available.</em></p>"

    out = df.copy()

    if drop_cols:
        out = out.drop(columns=[c for c in drop_cols if c in out.columns], errors="ignore")

    if max_rows is not None and len(out) > max_rows:
        out = out.head(max_rows)

    numeric_cols = numeric_cols or []
    percent_cols = percent_cols or []

    for c in set(numeric_cols + percent_cols):
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    ths = "".join([f"<th>{c}</th>" for c in out.columns])

    rows_html = []
    for _, r in out.iterrows():
        tds = []
        for c in out.columns:
            v = r[c]
            cls = ""
            if c in percent_cols and pd.notna(v):
                cls = "num"
                cell = f"{float(v)*100:.2f}%"
            elif c in numeric_cols and pd.notna(v):
                cls = "num"
                fv = float(v)
                cell = f"{int(fv):,}" if fv.is_integer() else f"{fv:.6f}".rstrip("0").rstrip(".")
            else:
                cell = "" if pd.isna(v) else str(v)
            tds.append(f"<td class='{cls}'>{cell}</td>")
        rows_html.append("<tr>" + "".join(tds) + "</tr>")

    sticky = "sticky" if sticky_header else ""
    small = "table-small" if font_small else ""

    return f"""
<div class="table-scroll {small}" style="max-height:{max_height_px}px;">
  <table class="styled-table {sticky}" id="{table_id}">
    <thead><tr>{ths}</tr></thead>
    <tbody>{''.join(rows_html)}</tbody>
  </table>
</div>
""".strip()


# ------------------------------------------------------------------ #
#  Base report template (all stage placeholders)
# ------------------------------------------------------------------ #

def create_base_report() -> str:
    return """<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>CRISPR Perturb-seq Simulator Report</title>

<style>
  * {
    box-sizing: border-box;
  }

  body {
      font-family: "Segoe UI", Arial, sans-serif;
      background-color: #f3f4f6;
      margin: 0;
  }

  .stage-container {
      margin: 56px auto;
      width: 88%;
      border-radius: 14px;
      background: white;
      box-shadow: 0 10px 25px rgba(0,0,0,0.06);
      overflow: hidden;
      clear: both;
  }

  .stage-container + .stage-container {
      margin-top: 72px;
  }

  .stage-header {
      background: linear-gradient(90deg, #0f766e, #14b8a6);
      color: white;
      padding: 18px 30px;
      font-size: 22px;
      font-weight: 650;
  }

  .stage-content {
      padding: 34px 46px;
  }

  .stage-description {
      font-size: 15px;
      line-height: 1.7;
      margin: 0 0 18px 0;
      color: #374151;
  }

  .dataset-summary {
      font-size: 15px;
      line-height: 1.8;
      color: #374151;
  }

  .panel-grid {
      display: grid;
      grid-template-columns: 1fr 1fr 1fr;
      gap: 14px;
      margin-top: 14px;
      align-items: start;
  }

  .two-col {
      display: grid;
      grid-template-columns: 1.25fr 1fr;
      gap: 14px;
      margin-top: 12px;
      align-items: start;
  }

  .panel-card {
      border: 1px solid #e5e7eb;
      border-radius: 12px;
      padding: 14px;
      background: #fff;
      overflow: hidden;
  }

  .panel-title {
      font-weight: 750;
      margin-bottom: 10px;
      color: #111827;
      font-size: 14px;
  }

  .subsection-title {
      margin-top: 18px;
      margin-bottom: 10px;
      font-size: 16px;
      font-weight: 800;
      color: #111827;
  }

  .table-scroll {
      overflow: auto;
      border-radius: 10px;
      border: 1px solid #eef2f7;
      width: 100%;
      background: #fff;
  }

  .table-small table {
      font-size: 12px;
  }

  .styled-table {
      width: 100%;
      border-collapse: collapse;
  }

  .styled-table th {
      text-align: left;
      padding: 10px 10px;
      background-color: #e6f4f1;
      font-size: 12px;
      position: sticky;
      top: 0;
      z-index: 2;
  }

  .styled-table td {
      padding: 8px 10px;
      border-bottom: 1px solid #e5e7eb;
      vertical-align: top;
      font-size: 12px;
      color: #111827;
  }

  .num {
      text-align: right;
      font-variant-numeric: tabular-nums;
  }

  .multiplet {
      font-style: italic;
      background-color: #f8f1e5;
  }

  .confidence {
      font-weight: 800;
  }

  .fig-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 14px;
      margin-top: 14px;
      align-items: start;
  }

  .fig-card {
      border: 1px solid #e5e7eb;
      border-radius: 12px;
      padding: 12px;
      background: #fff;
      overflow: hidden;
  }

  .figure-img {
      width: 100%;
      height: auto;
      border-radius: 10px;
      border: 1px solid #f3f4f6;
      background: #fff;
      display: block;
  }

  .caption {
      font-size: 12.5px;
      margin-top: 8px;
      font-style: italic;
      color: #6b7280;
      line-height: 1.5;
  }

  .interpretation {
      font-size: 13.5px;
      margin-top: 10px;
      font-style: italic;
      color: #374151;
      line-height: 1.6;
  }

  .method-card {
      border: 1px solid #e5e7eb;
      border-radius: 12px;
      padding: 14px;
      background: #fff;
      margin-top: 14px;
      overflow: hidden;
  }

  .method-title {
      font-weight: 850;
      font-size: 15px;
      color: #0f172a;
      margin-bottom: 10px;
  }

  @media (max-width: 1200px){
      .panel-grid { grid-template-columns: 1fr; }
      .two-col { grid-template-columns: 1fr; }
      .fig-grid { grid-template-columns: 1fr; }
  }
</style>
</head>

<body>

<!-- STAGE:HEADER --><!-- END:HEADER -->
<!-- STAGE:DATASET --><!-- END:DATASET -->
<!-- STAGE:STAGE0 --><!-- END:STAGE0 -->
<!-- STAGE:STAGE1 --><!-- END:STAGE1 -->
<!-- STAGE:STAGE2 --><!-- END:STAGE2 -->
<!-- STAGE:STAGE3 --><!-- END:STAGE3 -->
<!-- STAGE:STAGE4 --><!-- END:STAGE4 -->
<!-- STAGE:STAGE5 --><!-- END:STAGE5 -->
<!-- STAGE:STAGE6 --><!-- END:STAGE6 -->
<!-- STAGE:STAGE7 --><!-- END:STAGE7 -->
<!-- STAGE:STAGE8 --><!-- END:STAGE8 -->
<!-- STAGE:STAGE9 --><!-- END:STAGE9 -->
<!-- STAGE:STAGE10_11 --><!-- END:STAGE10_11 -->
<!-- STAGE:STAGE12 --><!-- END:STAGE12 -->

</body>
</html>
""".strip()
