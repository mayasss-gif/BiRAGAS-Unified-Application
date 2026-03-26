from __future__ import annotations
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate a self-contained HTML master report from nf-core/crisprseq
targeted pipeline results.

Can be used as:
  - A callable function:  generate_report(results_dir, output_dir)
  - A CLI script:         python make_crispr_master_report_final.py --results /path/to/results --output /path/to/output
"""

import argparse
import base64
import html as ihtml
import re
from pathlib import Path
from datetime import datetime

try:
    import pandas as pd
except ImportError:
    pd = None

_SCRIPT_DIR = Path(__file__).resolve().parent
_LOGOS_DIR = _SCRIPT_DIR / "logos"
DEFAULT_LOGO_LEFT = _LOGOS_DIR / "ARI-Logo.png"
DEFAULT_LOGO_RIGHT = _LOGOS_DIR / "ARI-Ayass-Research-Institute.png"

# ===================== BRAND / STYLE =====================

TITLE = "Targeted CRISPR Genome Editing Outcomes (AmpliConsequence™)"
SUBTITLE = (
    "This pipeline analyzes targeted sequencing data to quantify and characterize genome editing outcomes. "
    "It applies standardized quality control, consensus reconstruction, and reference-guided alignment "
    "to accurately identify insertions, deletions, and substitutions introduced by CRISPR editing."
)

BG = "#f7fafc"
CARD = "#ffffff"
TEXT = "#0f172a"
MUTED = "#475569"
ACCENT = "#14b8a6"
ACCENT2 = "#22c55e"

# ===================== IO HELPERS =====================

def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")

def _img_to_b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode()

def _read_table(path: Path) -> "pd.DataFrame":
    if not path.exists():
        raise FileNotFoundError(path)
    for sep in ["\t", ","]:
        try:
            df = pd.read_csv(path, sep=sep, engine="c", low_memory=False)
            if df.shape[1] >= 2:
                return df
        except Exception:
            pass
    return pd.read_csv(path, sep=None, engine="python")

def _read_table_safe(path: Path) -> "pd.DataFrame | None":
    try:
        return _read_table(path)
    except Exception:
        return None

def _norm_cols(df: "pd.DataFrame") -> "pd.DataFrame":
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df

def _find_col(df, candidates):
    cols = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols:
            return cols[cand.lower()]
    return None

def _find_col_contains(df, needle: str):
    needle = needle.lower()
    for c in df.columns:
        if needle in str(c).lower():
            return c
    return None

def _sample_from_filename(p: Path) -> str:
    m = re.match(r"^([A-Za-z0-9]+)_", p.name)
    return m.group(1) if m else p.stem.split("_")[0]

# ===================== HTML BUILDERS =====================

def _card(inner: str) -> str:
    return f"<div class='card'>{inner}</div>"

def _section(title: str, inner: str, desc: str = "") -> str:
    d = f"<p class='desc'>{desc}</p>" if desc else ""
    anchor = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
    return f"<section id='{anchor}'><h2>{title}</h2>{d}{inner}</section>"

def _caption(text: str) -> str:
    return f"<div class='caption'>{ihtml.escape(text)}</div>"

def _table_html(df, limit=200) -> str:
    if df is None or df.empty:
        return "<div class='empty'>No data detected for this section.</div>"
    if len(df) > limit:
        df = df.head(limit)
    return df.to_html(index=False, border=0, classes="datatable", escape=True)

def _embed_html_file(path: Path, height=560) -> str:
    if not path.exists():
        return "<div class='empty'>Missing HTML artifact.</div>"
    safe = ihtml.escape(_read_text(path), quote=True)
    return f"<iframe class='frame' style='height:{height}px' srcdoc=\"{safe}\"></iframe>"

def _embed_png(path: Path, cap: str = "") -> str:
    if not path.exists():
        return f"<div class='empty'>Missing: {path.name}</div>"
    b64 = _img_to_b64(path)
    out = f"<div class='imgcard'><img src='data:image/png;base64,{b64}'/>"
    if cap:
        out += f"<div class='caption'>{ihtml.escape(cap)}</div>"
    out += "</div>"
    return out

def _embed_png_logo(path: Path) -> str:
    if not path.is_file():
        return ""
    b64 = _img_to_b64(path)
    return f"<img class='logo' src='data:image/png;base64,{b64}'/>"

def _grid(items, cols=3) -> str:
    return f"<div class='grid cols{cols}'>" + "".join(items) + "</div>"

# ===================== REPORT GENERATOR =====================

def generate_report(
    results_dir: str,
    output_dir: str,
    max_samples_panels: int = 6,
    logo_left: str = "",
    logo_right: str = "",
) -> str:
    """Build the master HTML report from pipeline results.

    Parameters
    ----------
    results_dir : str
        Path to the Nextflow ``--outdir`` results directory.  Expected
        sub-directories: ``cigar/``, ``plots/``, ``multiqc/``,
        ``preprocessing/fastqc/``.
    output_dir : str
        Directory where the report HTML file will be written.
    max_samples_panels : int
        Maximum number of per-sample panels to embed.
    logo_left, logo_right : str
        Paths to brand logo PNGs for the header bar.  When not
        provided, the bundled logos from ``targeted/logos/`` are used.

    Returns
    -------
    str
        Absolute path to the generated report file.
    """
    if pd is None:
        raise ImportError(
            "pandas is required for report generation. "
            "Install it with: pip install pandas"
        )

    base = Path(results_dir)
    if not base.exists():
        raise SystemExit(f"[REPORT ERROR] Results directory not found: {base}")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    multiqc_dir = base / "multiqc" / "multiqc_data"
    cigar_dir = base / "cigar"
    plots_dir = base / "plots"
    fastqc_dir = base / "preprocessing" / "fastqc"

    report_path = out_dir / "crisprseq_master_report.html"
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    logo_l = Path(logo_left) if logo_left else DEFAULT_LOGO_LEFT
    logo_r = Path(logo_right) if logo_right else DEFAULT_LOGO_RIGHT

    # ----- MultiQC tables (all optional) -----
    general_stats = _safe_load(multiqc_dir / "multiqc_general_stats.txt")
    fastqc_tbl = _safe_load(multiqc_dir / "multiqc_fastqc.txt")
    edition_tbl = _safe_load(multiqc_dir / "multiqc_edition_bargraph.txt")
    indelqc_tbl = _safe_load(multiqc_dir / "multiqc_indelqc_bargraph.txt")
    read_processing = _safe_load(multiqc_dir / "multiqc_read_processing.txt")
    software = _safe_load(multiqc_dir / "multiqc_software_versions.txt")

    # ----- sample list -----
    samples = sorted({_sample_from_filename(p) for p in cigar_dir.glob("*_edition.html")}) if cigar_dir.exists() else []
    if not samples and general_stats is not None:
        samples = sorted(general_stats.iloc[:, 0].dropna().astype(str).unique().tolist())
    panel_samples = samples[:max_samples_panels]

    # ----- cigar panels helper -----
    def sample_panels(kind_glob, cap_prefix, height=520):
        panels = []
        for s in panel_samples:
            p = cigar_dir / kind_glob.format(sample=s)
            if p.exists():
                panels.append(f"<div class='panel'>{_embed_html_file(p, height)}{_caption(f'{s} — {cap_prefix}')}</div>")
            else:
                panels.append(f"<div class='panel'><div class='empty'>Missing: {s}</div></div>")
        return _grid(panels, cols=2)

    edition_panels = sample_panels("{sample}_edition.html", "CIGAR — Reads by edition type")
    qc_panels = sample_panels("{sample}_QC-indels.html", "CIGAR — QC of indels")
    reads_panels = sample_panels("{sample}_reads.html", "CIGAR — Read processing & clonality")

    # ----- indels top-5 per sample -----
    indels_top = _build_indels_top(cigar_dir, samples)

    # ----- FastQC panels -----
    fastqc_panels = []
    for s in panel_samples:
        p = fastqc_dir / f"{s}_fastqc.html"
        if p.exists():
            fastqc_panels.append(f"<div class='panel'>{_embed_html_file(p, 600)}{_caption(f'{s} — FastQC')}</div>")
        else:
            fastqc_panels.append(f"<div class='panel'><div class='empty'>Missing FastQC for {s}</div></div>")
    fastqc_html = _grid(fastqc_panels, cols=2)

    # ----- plot panels (HTML and/or PNG) -----
    acc_panels = []
    for s in samples[:3]:
        p = plots_dir / f"{s}_accumulative.html"
        if p.exists():
            acc_panels.append(f"<div class='panel'>{_embed_html_file(p, 560)}{_caption(f'{s} — Accumulative indels')}</div>")
        else:
            acc_panels.append(f"<div class='panel'><div class='empty'>Missing accumulative for {s}</div></div>")
    acc_html = _grid(acc_panels, cols=3)

    del_panels = []
    for s in samples[:3]:
        png = plots_dir / f"{s}_delAlleles_plot.png"
        html = plots_dir / f"{s}_Deletions.html"
        if png.exists() and png.stat().st_size > 100:
            del_panels.append(_embed_png(png, f"{s} — Deletion alleles"))
        elif html.exists():
            del_panels.append(f"<div class='panel'>{_embed_html_file(html, 520)}{_caption(f'{s} — Deletions')}</div>")
        else:
            del_panels.append(f"<div class='panel'><div class='empty'>Missing deletion plot for {s}</div></div>")
    del_html = _grid(del_panels, cols=3)

    logo_panels = []
    for s in panel_samples:
        p1 = plots_dir / f"{s}_subs-perc_plot_LOGO.png"
        p2 = plots_dir / f"{s}_top-alleles_LOGO.png"
        has_png = (p1.exists() and p1.stat().st_size > 100) or (p2.exists() and p2.stat().st_size > 100)
        if has_png:
            block = "<div class='imgpair'>"
            block += _embed_png(p1, f"{s} — Substitution LOGO")
            block += _embed_png(p2, f"{s} — Top alleles LOGO")
            block += "</div>"
            logo_panels.append(block)
        else:
            ins_html = plots_dir / f"{s}_Insertions.html"
            top_html = plots_dir / f"{s}_top.html"
            if ins_html.exists() or top_html.exists():
                block = "<div class='imgpair'>"
                if ins_html.exists():
                    block += _embed_html_file(ins_html, 400) + _caption(f"{s} — Insertions")
                if top_html.exists():
                    block += _embed_html_file(top_html, 400) + _caption(f"{s} — Top variants")
                block += "</div>"
                logo_panels.append(block)
    logo_html = _grid(logo_panels, cols=2) if logo_panels else "<div class='empty'>No LOGO / sequence-level plots available.</div>"

    # ----- summary stats -----
    n_samples = len(samples)
    mean_gc = float("nan")
    mean_len = float("nan")
    if general_stats is not None and general_stats.shape[1] > 3:
        try:
            mean_gc = float(general_stats.iloc[:, 2].mean())
            mean_len = float(general_stats.iloc[:, 3].mean())
        except Exception:
            pass

    stats_cards = _grid([
        _card(f"<h3>Total samples</h3><div class='big'>{n_samples}</div>"),
        _card(f"<h3>Mean GC%</h3><div class='big'>{mean_gc:.1f}</div>"),
        _card(f"<h3>Mean read length</h3><div class='big'>{mean_len:.0f} bp</div>"),
    ], cols=3)

    # ----- citations -----
    cite = """
    <ul class="refs">
      <li><a href="https://nf-co.re/crisprseq/" target="_blank">nf-core/crisprseq documentation</a></li>
      <li><a href="https://www.nature.com/articles/s41596-021-00626-x" target="_blank">Nature Protocols — CRISPR sequencing</a></li>
    </ul>"""

    # ----- assemble HTML -----
    html = _TEMPLATE.format(
        TITLE=TITLE,
        SUBTITLE=SUBTITLE,
        BG=BG, CARD=CARD, TEXT=TEXT, MUTED=MUTED, ACCENT=ACCENT, ACCENT2=ACCENT2,
        logo_left=_embed_png_logo(logo_l),
        logo_right=_embed_png_logo(logo_r),
        now=now,
        project=ihtml.escape(str(base)),
        stats_cards=stats_cards,
        general_stats=_section("General Statistics (MultiQC)",
            _card("<div class='tablewrap'>" + _table_html(general_stats) + "</div>"),
            "Sample-level summary from MultiQC general stats."),
        fastqc_table=_section("FastQC (MultiQC table)",
            _card("<div class='tablewrap'>" + _table_html(fastqc_tbl) + "</div>"),
            "FastQC status checks aggregated by MultiQC."),
        fastqc_panels=_section("FastQC (per-sample HTML)",
            fastqc_html,
            f"Embedded FastQC reports ({len(panel_samples)} samples)."),
        read_processing=_section("Read Processing Summary",
            _card("<div class='tablewrap'>" + _table_html(read_processing) + "</div>"),
            "Read processing overview aggregated by MultiQC."),
        edition_table=_section("Edition — Reads by Type",
            _card("<div class='tablewrap'>" + _table_html(edition_tbl) + "</div>"),
            "Edition-type counts per sample (WT, delins, insertions, deletions)."),
        edition_panels=_section("CIGAR — Edition (interactive)", edition_panels,
            "Interactive per-sample edition summaries."),
        indels_top=_section("CIGAR — Top Indels (top 5 / sample)",
            _card("<div class='tablewrap'>" + _table_html(indels_top, limit=9999) + "</div>"),
            "Top 5 indel events per sample ranked by frequency."),
        qc_table=_section("CIGAR — QC Indels (table)",
            _card("<div class='tablewrap'>" + _table_html(indelqc_tbl) + "</div>"),
            "QC of indels for all samples."),
        qc_panels=_section("CIGAR — QC Indels (interactive)", qc_panels,
            "Interactive QC of indels per sample."),
        reads_panels=_section("CIGAR — Reads / Clonality (interactive)", reads_panels,
            "Interactive read-level summaries per sample."),
        acc_html=_section("Accumulative Indel Distribution", acc_html,
            "Cumulative distribution of CRISPR-induced insertions and deletions."),
        del_html=_section("Deletion Alleles", del_html,
            "Deletion allele spectrum for representative samples."),
        logo_html=_section("Sequence-level Editing Outcomes (LOGO)", logo_html,
            "Substitution and top-allele LOGO plots, or insertion/top-variant HTML where PNGs are unavailable."),
        software=_section("Software Versions",
            _card("<div class='tablewrap'>" + _table_html(software, limit=9999) + "</div>"),
            "Software versions parsed by MultiQC."),
        citations=_section("Citations / Resources", _card(cite)),
    )

    report_path.write_text(html, encoding="utf-8")
    return str(report_path)


# ===================== INTERNAL HELPERS =====================

def _safe_load(path: Path) -> "pd.DataFrame | None":
    df = _read_table_safe(path)
    if df is not None:
        df = _norm_cols(df)
        df.rename(columns={df.columns[0]: "Sample"}, inplace=True)
    return df


def _build_indels_top(cigar_dir: Path, samples: list) -> "pd.DataFrame":
    rows = []
    if not cigar_dir.exists():
        return pd.DataFrame()
    for f in sorted(cigar_dir.glob("*_indels.csv")):
        s = _sample_from_filename(f)
        if s not in samples:
            continue
        df = _read_table_safe(f)
        if df is None or df.empty:
            continue
        df = _norm_cols(df)
        freq_col = _find_col(df, ["freq"]) or _find_col_contains(df, "freq")
        if not freq_col:
            freq_col = _find_col(df, ["Perc", "perc"]) or _find_col_contains(df, "perc")
        if not freq_col:
            continue
        pat_col = _find_col(df, ["patterns", "pattern"]) or _find_col_contains(df, "pattern")
        if pat_col:
            df = df[df[pat_col].notna()]
        for c in ["pre_ins_nt", "ins_nt", "post_ins_nt"]:
            if c in df.columns:
                df = df[df[c].notna()]
        df = df.sort_values(freq_col, ascending=False).head(5).copy()
        df.insert(0, "Sample", s)
        df.insert(1, "Rank", range(1, len(df) + 1))
        rows.append(df)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


# ===================== HTML TEMPLATE =====================

_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{TITLE}</title>
<style>
  body {{
    margin:0; padding:0;
    background:{BG};
    color:{TEXT};
    font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial;
  }}
  .wrap {{ padding: 28px 30px 60px; max-width: 1400px; margin: 0 auto; }}
  .topbar {{
    background: linear-gradient(90deg, {ACCENT}, {ACCENT2});
    border-radius: 18px; padding: 18px;
    color: white; box-shadow: 0 10px 30px rgba(2,6,23,0.18);
  }}
  .toprow {{ display:flex; align-items:center; justify-content:space-between; gap:14px; }}
  .logo {{
    height:54px; width:auto; object-fit:contain;
    background:rgba(255,255,255,.92); border-radius:12px; padding:8px 10px;
  }}
  .titleblock {{ text-align:center; flex:1; padding:6px 10px; }}
  h1 {{ margin:0; font-size:22px; letter-spacing:0.2px; }}
  .subtitle {{ margin:8px 0 0; opacity:.95; font-size:13px; line-height:1.45; }}
  .meta {{ margin:10px 0 0; opacity:.95; font-size:12px; }}
  section {{ margin-top:28px; }}
  h2 {{ font-size:18px; margin:0 0 12px; padding-left:6px; }}
  .desc {{ margin:0 0 12px; color:{MUTED}; line-height:1.55; max-width:1050px; padding-left:6px; }}
  .card {{
    background:{CARD}; border-radius:18px; padding:16px;
    box-shadow:0 10px 24px rgba(2,6,23,0.06); border:1px solid rgba(15,23,42,0.06);
  }}
  .grid {{ display:grid; gap:14px; }}
  .cols2 {{ grid-template-columns: repeat(2, minmax(0,1fr)); }}
  .cols3 {{ grid-template-columns: repeat(3, minmax(0,1fr)); }}
  @media (max-width:1100px) {{ .cols2,.cols3 {{ grid-template-columns:1fr; }} }}
  .big {{ font-size:26px; font-weight:800; margin-top:6px; }}
  .datatable {{ width:100%; border-collapse:collapse; }}
  .datatable th {{
    position:sticky; top:0; background:#f1f5f9; z-index:1;
    font-size:12px; color:#0f172a; padding:10px 8px; border-bottom:1px solid #e2e8f0;
  }}
  .datatable td {{
    font-size:12px; padding:8px; border-bottom:1px solid #eef2f7;
    text-align:center; white-space:nowrap;
  }}
  .tablewrap {{ max-height:520px; overflow:auto; border-radius:14px; border:1px solid #e2e8f0; }}
  .frame {{
    width:100%; border:0; border-radius:14px; background:white;
    box-shadow:inset 0 0 0 1px rgba(15,23,42,0.06);
  }}
  .panel {{
    background:white; border-radius:16px; padding:10px;
    box-shadow:0 10px 24px rgba(2,6,23,0.06); border:1px solid rgba(15,23,42,0.06);
  }}
  .caption {{ margin-top:8px; font-size:12px; color:{MUTED}; text-align:center; line-height:1.35; }}
  .imgcard img {{ width:100%; border-radius:12px; border:1px solid rgba(15,23,42,0.08); }}
  .imgpair {{
    background:white; border-radius:16px; padding:10px;
    border:1px solid rgba(15,23,42,0.06); box-shadow:0 10px 24px rgba(2,6,23,0.06);
  }}
  .empty {{
    padding:18px; border-radius:14px; background:#fff7ed;
    border:1px solid #fed7aa; color:#7c2d12; text-align:center; font-size:13px;
  }}
  .refs li {{ margin:8px 0; }}
  a {{ color:#0ea5e9; text-decoration:none; }}
  a:hover {{ text-decoration:underline; }}
</style>
</head>
<body>
<div class="wrap">
  <div class="topbar"><div class="toprow">
    {logo_left}
    <div class="titleblock">
      <h1>{TITLE}</h1>
      <div class="subtitle">{SUBTITLE}</div>
      <div class="meta"><b>Generated:</b> {now} </div>
    </div>
    {logo_right}
  </div></div>

  {stats_cards}
  {general_stats}
  {fastqc_table}
  {fastqc_panels}
  {read_processing}
  {edition_table}
  {edition_panels}
  {indels_top}
  {qc_table}
  {qc_panels}
  {reads_panels}
  {acc_html}
  {del_html}
  {logo_html}
  {software}
  {citations}
</div>
</body>
</html>"""


# ===================== CLI =====================

def main():
    ap = argparse.ArgumentParser(description="Generate crisprseq master report")
    ap.add_argument("--results", required=True, help="Pipeline results directory")
    ap.add_argument("--output", required=True, help="Output directory for the report")
    ap.add_argument("--max_samples_panels", type=int, default=6)
    ap.add_argument("--logo_left", default="")
    ap.add_argument("--logo_right", default="")
    args = ap.parse_args()

    path = generate_report(
        results_dir=args.results,
        output_dir=args.output,
        max_samples_panels=args.max_samples_panels,
        logo_left=args.logo_left,
        logo_right=args.logo_right,
    )
    print(f"[OK] Report written: {path}")


if __name__ == "__main__":
    main()
