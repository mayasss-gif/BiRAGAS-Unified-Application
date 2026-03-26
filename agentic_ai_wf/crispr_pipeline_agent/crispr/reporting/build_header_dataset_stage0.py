#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Report builder: Header, Dataset Overview, and Stage 0.

All paths are derived dynamically from the sample output directory.
Dataset metadata is fetched live from the NCBI GEO API based on the
GSM accession extracted from the sample directory name.
"""

from pathlib import Path

from .report_common import (
    ensure_base_report, write_report, insert_or_replace_block,
    encode_image_to_base64, safe_read_text, now_str,
    first_match, llm_generate,
)
from .geo_fetch import fetch_sample_info_cached


def build_header_block(sample_dir: Path) -> str:
    reporting_dir = Path(__file__).resolve().parent

    left = encode_image_to_base64(reporting_dir / "ABS_left.png")
    right = encode_image_to_base64(reporting_dir / "ABS_Right.png")
    today = now_str()

    left_img = (
        f'<img src="data:image/png;base64,{left}" style="height:70px;">'
        if left else ""
    )
    right_img = (
        f'<img src="data:image/png;base64,{right}" style="height:70px;">'
        if right else ""
    )

    return f"""
<!-- STAGE:HEADER -->
<div style="background: linear-gradient(90deg,#0f766e,#14b8a6);
            padding:40px 60px;
            color:white;
            display:flex;
            align-items:center;
            justify-content:space-between;">
    {left_img}
    <div style="text-align:center;">
        <div style="font-size:28px;font-weight:600;">
            CRISPR Perturb-seq Simulator Report
        </div>
        <div style="font-size:16px;opacity:0.9;margin-top:6px;">
            Transcriptomic CRISPR Perturbation Modeling Framework
        </div>
        <div style="font-size:13px;margin-top:8px;opacity:0.8;">
            Generated: {today} &mdash; {sample_dir.name}
        </div>
    </div>
    {right_img}
</div>
<!-- END:HEADER -->
""".strip()


def build_dataset_block(sample_dir: Path, client=None) -> str:
    """
    Build the Dataset Overview block.

    Resolution order:
      1. Use an existing cached LLM summary (``LLM_support/step-1_datasetsummary.txt``).
      2. Fetch sample metadata live from the NCBI GEO API (cached to disk
         as ``.geo_sample_info.txt`` so repeated builds don't re-hit the API).
         If an OpenAI client is available, an LLM summary is generated from
         the fetched metadata and cached for next time.
      3. Fall back to a placeholder message.
    """
    # 1) Check for a pre-existing LLM summary
    summary_path = first_match(sample_dir, "LLM_support/step-1_datasetsummary.txt")
    summary = safe_read_text(summary_path) if summary_path else ""

    # 2) No cached summary — fetch live from NCBI GEO and generate via LLM
    if not summary:
        sample_text = fetch_sample_info_cached(sample_dir)

        if sample_text and client is not None:
            summary = llm_generate(
                client,
                "You are writing documentation for a CRISPR computational "
                "analysis pipeline.\n\n"
                "From the following GEO sample information, generate ONE "
                "concise, formal paragraph that discusses ONLY the CRISPR "
                "perturbation design and its relevance for CRISPR analysis.\n\n"
                "Strict rules:\n"
                "- Focus exclusively on CRISPR screening / Perturb-seq aspects.\n"
                "- Emphasize guide RNA design, barcode identification, and "
                "perturbation assignment.\n"
                "- Mention single-cell CRISPR profiling only in the context of "
                "perturbation analysis.\n"
                "- Highlight how the dataset supports downstream CRISPR modeling, "
                "perturbation classification, and transcriptional response analysis.\n"
                "- Do NOT provide general RNA-seq explanations.\n"
                "- Do NOT include administrative or contact details.\n\n"
                f"Sample Information:\n{sample_text}",
                temperature=0.1,
            )

            # Cache the generated summary for future runs
            if summary:
                llm_dir = sample_dir / "LLM_support"
                llm_dir.mkdir(parents=True, exist_ok=True)
                (llm_dir / "step-1_datasetsummary.txt").write_text(
                    summary, encoding="utf-8",
                )

        elif sample_text:
            # No LLM client but we have raw GEO info — use it directly
            summary = (
                "<b>GEO Sample Metadata (raw):</b><br><pre style='font-size:12px;"
                "white-space:pre-wrap;'>" + sample_text[:2000] + "</pre>"
            )

    # 3) Fallback
    if not summary:
        summary = (
            "Dataset summary not available. Ensure the sample directory name "
            "contains a valid GSM accession (e.g. <code>GSM2406675_10X001</code>) "
            "so that metadata can be fetched from NCBI GEO. "
            "Set <code>OPENAI_API_KEY</code> in <code>.env</code> to enable "
            "LLM-generated scientific summaries."
        )

    return f"""
<!-- STAGE:DATASET -->
<div class="stage-container">
  <div class="stage-header" style="background: linear-gradient(90deg,#1e3a8a,#3b82f6);">
      Dataset Overview
  </div>
  <div class="stage-content">
      <div class="dataset-summary">{summary}</div>
  </div>
</div>
<!-- END:DATASET -->
""".strip()


def build_stage0_block(sample_dir: Path) -> str:
    return """
<!-- STAGE:STAGE0 -->
<div class="stage-container">
  <div class="stage-header" style="background: linear-gradient(90deg,#7c3aed,#a78bfa);">
      Stage 0: Data Ingestion and Preprocessing
  </div>
  <div class="stage-content">
      <p class="stage-description">
      Raw gene\u2013barcode matrices were imported into structured <b>AnnData</b> objects and linked to guide identity metadata.
      This step consolidates per-cell guide signals and coverage metrics, enabling consistent downstream
      <i>transcriptomic CRISPR perturbation</i> modeling across all subsequent stages.
      </p>
  </div>
</div>
<!-- END:STAGE0 -->
""".strip()


def build(sample_dir: Path, client=None, report_path: str = None) -> str:
    """Build header + dataset + stage 0 blocks; return updated HTML."""
    html = ensure_base_report(report_path)
    html = insert_or_replace_block(html, "HEADER", build_header_block(sample_dir))
    html = insert_or_replace_block(html, "DATASET", build_dataset_block(sample_dir, client))
    html = insert_or_replace_block(html, "STAGE0", build_stage0_block(sample_dir))
    return html


if __name__ == "__main__":
    import sys
    d = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    out = d / "CRISPRModel_Simulator_Report.html"
    html = build(d, report_path=str(out))
    write_report(out, html)
    print("Done: Header + Dataset + Stage0")
