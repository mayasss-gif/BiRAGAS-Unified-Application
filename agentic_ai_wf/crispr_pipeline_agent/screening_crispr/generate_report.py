#!/usr/bin/env python3
"""
CRISPR-Based Genetic Screening: DNA Sequencing Analysis
Complete single-file HTML report generator (portable, self-contained)

WHAT THIS SCRIPT DOES (A→Z)
- Generates ONE HTML report with embedded (base64) images (no external assets).
- Builds a consistent CRISPR screening report from a CrisprModel-like output folder.
- Sections (HEADINGS only; no “module” wording):
  1) OVERALL CRISPR SCREENING LANDSCAPE (true counts + explanations)
  2) ROBUST RANK AGGREGATION (COUNTS → RRA): sgRNA + gene tables + key plots + captions
  3) MODEL-BASED ESTIMATION (COUNTS → MLE / FLUTE): MLE table + QC + selection + enrichment + pathways + captions
  4) REFERENCES

KEY REQUIREMENTS IMPLEMENTED
- ✅ NO fake numbers: summary uses REAL counts derived from detected tables/images.
- ✅ sgRNA/gene counts are UNIQUE when possible (heuristic column detection); otherwise explicitly says “rows”.
- ✅ NO repetition: figures + tables are deduplicated by content fingerprint (hash).
- ✅ Captions are present for ALL tables and ALL figure blocks.
- ✅ Enrichment plots and enrichment tables remain separate (cleaner), but deduped.
- ✅ Pathways: shows TOP 5 pathway figures one-by-one, with extended names (no file extensions).
- ✅ No pathway table is added if it is not detected (and we don’t try to invent one).
- ✅ Pastel heading colors (different for each major heading) + headings in CAPITAL.
- ✅ Header “pills” have different pastel backgrounds for CRISPR / RRA / FLUTE.
- ✅ Removes “auto-detected” wording from UI text.
- ✅ Adds a final note: “Please also check downloadable files for more details.”

RUN
  python screening_generate_crispr_html_report_COMPLETE.py --base /path/to/CrisprModel --out CRISPR_Report.html

NOTES
- This script is intentionally robust to different nf-core/crisprseq + MAGeCK/Flute output layouts.
- It avoids heavy assumptions and explicitly labels uncertainties (e.g., if an sgRNA ID column cannot be found).
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import html
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
from jinja2 import Environment, BaseLoader


# -----------------------------
# Result type returned by generate_report()
# -----------------------------
@dataclass
class ReportResult:
    """Outcome of report generation."""
    success: bool
    message: str
    report_path: Optional[Path] = None


# -----------------------------
# Header copy
# -----------------------------
HEADER_TITLE = "CRISPR-Based Genetic Screening: DNA Sequencing Analysis"
HEADER_PARAGRAPH = (
    "This CRISPR-Seq screening pipeline takes bulk pooled CRISPR DNA-sequencing data "
    "(either raw FASTQs or precomputed sgRNA count tables) and processes it end-to-end to "
    "quantify sgRNA abundances, assess data quality, and infer gene-level essentiality and "
    "condition-specific effects. Depending on the chosen mode, it performs quality control, "
    "aligns and counts sgRNAs against a provided guide library, and applies established "
    "statistical models (MAGeCK RRA, MAGeCK MLE, and MAGeCK-Flute) to identify genes "
    "whose perturbation significantly affects cell fitness or response to treatments. "
    "The pipeline aggregates results across guides, prioritizes candidate hits, and produces "
    "standardized outputs and MultiQC-style summaries, supporting reproducible screening analysis."
)

# -----------------------------
# Captions and figure/table text
# -----------------------------
LANDSCAPE_DESC = (
    "Bulk pooled CRISPR–Cas9 screening data were analysed using MAGeCK RRA and MAGeCK MLE/Flute frameworks to quantify sgRNA- and gene-level "
    "selection effects, assess screen quality, and derive functional and pathway-level interpretations of treatment-associated dependencies."
)

RRA_DESC = (
    "This section aggregates sgRNA- and gene-level statistics from MAGeCK RRA outputs to rank perturbations by depletion and enrichment under treatment. "
    "Tables show the top rows for fast review; use search and column sorting for deeper inspection."
)

MLE_DESC = (
    "This section focuses on model-based estimation of gene effects and downstream diagnostics/interpretation. "
    "It includes gene-level MAGeCK MLE summaries, screen quality diagnostics, selection/ranking plots, functional enrichment, and pathway visualizations."
)

# Table titles and captions
TABLE_SGRNA_TITLE = "SGRNA-LEVEL DIFFERENTIAL ABUNDANCE RESULTS FROM CRISPR SCREENING"
TABLE_SGRNA_CAPTION = (
    "This table reports sgRNA-level statistics comparing treatment versus control conditions, including counts, "
    "log fold change (LFC), variance estimates, enrichment/depletion scores, and associated p-values with multiple-testing correction where present. "
    "sgRNAs are annotated with their target genes. Use these results to validate guide-level consistency supporting gene-level selection signals."
)

TABLE_GENE_RRA_TITLE = "GENE-LEVEL ESSENTIALITY ANALYSIS SUMMARY FROM MAGeCK (RRA)"
TABLE_GENE_RRA_CAPTION = (
    "This table summarizes gene-level results aggregated across sgRNAs using MAGeCK RRA, reporting depletion/enrichment scores, "
    "p-values, false discovery rates (FDR), ranks, supporting sgRNA counts, and estimated effect sizes where available. "
    "These metrics identify genes whose perturbation significantly affects cellular fitness or treatment response."
)

TABLE_MLE_TITLE = "GENE-LEVEL MAGeCK MLE RESULTS SUMMARY"
TABLE_MLE_CAPTION = (
    "This table reports gene-level statistics from MAGeCK maximum likelihood estimation (MLE). "
    "Depending on output format, columns may include gene effect estimates (beta), z-scores, Wald statistics, p-values, and FDR. "
    "MLE supports robust estimation of gene effects while accounting for sgRNA efficiency and variance."
)

# Figure captions
RRA_FIG_TITLE_1 = "RANKED GENE-LEVEL EFFECTS FROM CRISPR SCREENING"
RRA_FIG_CAP_1 = (
    "Genes are ordered by an estimated effect (often log2 fold change) between treatment and control. "
    "Negative values typically indicate depletion, while positive values indicate enrichment under treatment. "
    "Labeled genes highlight extreme effects across the ranked distribution."
)

RRA_FIG_TITLE_2 = "RELATIONSHIP BETWEEN GENE-LEVEL SCORE AND STATISTICAL SIGNIFICANCE"
RRA_FIG_CAP_2 = (
    "Scatter plots show gene-level scores versus a significance metric (often FDR or -log10 FDR). "
    "Each point represents a gene aggregated across multiple sgRNAs, illustrating effect-size distribution and confidence used to prioritize hits."
)

QC_TITLE = "QUALITY CONTROL"
QC_CAPTION = (
    "These plots evaluate screen quality by examining distribution, reproducibility, and consistency of effects across samples and controls. "
    "They help assess variance structure, detect technical bias, and confirm robustness prior to downstream interpretation."
)

SEL_TITLE = "GENE-LEVEL SELECTION AND RANKING"
SEL_CAPTION = (
    "These figures summarize gene-level prioritization from MAGeCK-Flute outputs, highlighting enrichment/depletion patterns, "
    "treatment-versus-control effect structure, and selection signatures used to shortlist candidate hits."
)

ENR_TITLE = "FUNCTIONAL ENRICHMENT"
ENR_CAPTION = (
    "Enrichment plots summarize overrepresented biological processes, pathways, or molecular complexes identified from gene groups "
    "derived from CRISPR screening results. These analyses provide functional context for observed selection patterns."
)

PATH_TITLE = "PATHWAY-LEVEL VISUALIZATION"
PATH_CAPTION = (
    "Pathway figures map gene-level effects onto reference pathways to visualize treatment-associated selection across networks. "
    "Top pathway figures are shown below; use the pipeline output directory to inspect the full pathway set."
)

FINAL_NOTE = "Please also check downloadable files for more details."

# References
REFS = [
    ("Ayass Bioscience – Laboratory Testing Services", "https://ayassbioscience.com/ayass-bioscience-laboratory-testing/"),
    ("nf-core/crisprseq pipeline (v2.1.1)", "https://nf-co.re/crisprseq/2.1.1/"),
]

IMAGE_EXTS = {".png", ".jpg", ".jpeg"}


# -----------------------------
# Utilities
# -----------------------------
def now_stamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M")


def slug(s: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "_", s.strip())
    return s.strip("_").lower()


def safe_read_text(path: Path, max_bytes: int = 2_000_000) -> str:
    try:
        data = path.read_bytes()
        if len(data) > max_bytes:
            data = data[:max_bytes]
        return data.decode("utf-8", errors="replace")
    except Exception:
        return ""


def detect_delimiter(sample: str) -> str:
    if "\t" in sample:
        return "\t"
    # many mageck tables are whitespace delimited
    return r"\s+"


def read_table_any(path: Path, max_rows: int = 1_000_000) -> pd.DataFrame:
    """
    Robust reader for *.txt, *.tsv, *.csv:
    - often tab-separated
    - sometimes whitespace
    - sometimes CSV
    """
    text = safe_read_text(path, max_bytes=1_000_000)
    if not text.strip():
        return pd.DataFrame()

    lines = [ln for ln in text.splitlines() if ln.strip()]
    sample = "\n".join(lines[:40])
    delim = detect_delimiter(sample)

    try:
        if path.suffix.lower() == ".csv":
            df = pd.read_csv(path, engine="python")
        elif delim == "\t":
            df = pd.read_csv(path, sep="\t", engine="python")
        else:
            df = pd.read_csv(path, sep=delim, engine="python")
    except Exception:
        try:
            df = pd.read_csv(path, engine="python")
        except Exception:
            return pd.DataFrame()

    if len(df) > max_rows:
        df = df.iloc[:max_rows].copy()
    return df


def best_guess_sort(df: pd.DataFrame) -> pd.DataFrame:
    """
    Auto-sort:
    Prefer significance ascending (FDR, qvalue, padj, pvalue) else effect descending by absolute magnitude.
    """
    if df is None or df.empty:
        return df

    cols = [str(c) for c in df.columns]
    low = {c.lower(): c for c in cols}

    sig_keys = [
        "fdr", "qvalue", "q-value", "q_value",
        "p.adjust", "padj", "adj_p", "adj.p", "p_adj",
        "pvalue", "p.value", "p-value", "pval",
        "neg_fdr", "pos_fdr",
    ]
    for k in sig_keys:
        if k in low:
            c = low[k]
            out = df.copy()
            out[c] = pd.to_numeric(out[c], errors="coerce")
            out = out.sort_values(by=c, ascending=True, na_position="last")
            return out

    eff_keys = ["beta", "lfc", "log2fc", "score", "z", "wald", "diff"]
    for k in eff_keys:
        if k in low:
            c = low[k]
            out = df.copy()
            out[c] = pd.to_numeric(out[c], errors="coerce")
            out["_abs_sort"] = out[c].abs()
            out = out.sort_values(by="_abs_sort", ascending=False, na_position="last").drop(columns=["_abs_sort"])
            return out

    return df


def top_n_df(df: pd.DataFrame, n: int = 100) -> Tuple[pd.DataFrame, bool]:
    if df is None or df.empty:
        return df, False
    truncated = len(df) > n
    return df.head(n).copy(), truncated


def df_to_records(df: pd.DataFrame) -> Tuple[List[str], List[List[str]]]:
    cols = [str(c) for c in df.columns.tolist()]
    rows: List[List[str]] = []
    for _, r in df.iterrows():
        rows.append([("" if pd.isna(v) else str(v)) for v in r.tolist()])
    return cols, rows


def b64_image_data_uri(path: Path) -> Optional[str]:
    try:
        ext = path.suffix.lower()
        if ext not in IMAGE_EXTS:
            return None
        mime = "image/png" if ext == ".png" else "image/jpeg"
        data = path.read_bytes()
        b64 = base64.b64encode(data).decode("ascii")
        return f"data:{mime};base64,{b64}"
    except Exception:
        return None


def find_first(base: Path, patterns: List[str]) -> Optional[Path]:
    for pat in patterns:
        hits = list(base.rglob(pat))
        if hits:
            hits.sort(key=lambda p: (len(str(p)), -p.stat().st_mtime))
            return hits[0]
    return None


def find_all(base: Path, patterns: List[str]) -> List[Path]:
    out: List[Path] = []
    for pat in patterns:
        out.extend(base.rglob(pat))
    uniq = {str(p): p for p in out}
    paths = list(uniq.values())
    paths.sort(key=lambda p: (len(str(p)), str(p).lower()))
    return paths


def _logos_dir() -> Path:
    """Return the ``logos/`` directory bundled inside the screening_crispr package."""
    return Path(__file__).resolve().parent / "logos"


def pick_logo_pair(base: Path) -> Tuple[Optional[Path], Optional[Path]]:
    logos = _logos_dir()
    left = find_first(logos, ["ARI-Ayass-Research-Institute.png", "*Ayass*Research*Institute*.png", "*ARI*.png"])
    right = find_first(logos, ["Laboratory-Testing-RF.jpg", "*Laboratory*Testing*.jpg", "*Laboratory*Testing*.png", "*RF*.jpg", "*RF*.png"])
    return left, right


def pretty_label_from_path(p: Path) -> str:
    # "full name of file not extension": stem, plus disambiguation if needed externally
    s = p.stem
    s = s.replace("__", "_").strip("_")
    s = s.replace("_", " ")
    return s


def normalize_pathway_display_name(stem: str) -> str:
    """
    Make an extended, human-friendly pathway title from typical Flute pathway filenames.
    Examples:
      'Group1_Group2_hsa00230.pathview.multi' -> 'Group 1 & Group 2 — hsa00230 pathway view'
    """
    s = stem
    s = s.replace(".pathview.multi", "")
    s = s.replace(".pathview", "")
    s = s.replace(".pathviewmulti", "")
    s = s.replace(".multi", "")

    # common patterns: Group1_Group2_hsaXXXX
    s = s.replace("__", "_").strip("_")
    parts = s.split("_")
    groups = [x for x in parts if re.match(r"(?i)^group\d+$", x)]
    rest = [x for x in parts if x not in groups]

    group_text = ""
    if groups:
        gnums = [re.sub(r"(?i)group", "", g) for g in groups]
        if len(gnums) == 1:
            group_text = f"Group {gnums[0]}"
        else:
            group_text = " & ".join([f"Group {x}" for x in gnums])

    rest_text = " ".join(rest).strip()
    rest_text = rest_text.replace("hsa", "hsa ").strip()
    rest_text = re.sub(r"\s+", " ", rest_text)

    if group_text and rest_text:
        return f"{group_text} — {rest_text}"
    if group_text:
        return group_text
    if rest_text:
        return rest_text
    return stem


# -----------------------------
# Deduplication fingerprints
# -----------------------------
def file_fingerprint_fast(path: Path) -> str:
    """
    Fast-ish content fingerprint: SHA1(size + first64k + last64k).
    Avoids hashing huge images fully while still catching duplicates.
    """
    try:
        size = path.stat().st_size
        with path.open("rb") as f:
            head = f.read(65536)
            if size > 65536:
                f.seek(max(0, size - 65536))
                tail = f.read(65536)
            else:
                tail = b""
        h = hashlib.sha1()
        h.update(str(size).encode("utf-8"))
        h.update(head)
        h.update(tail)
        return h.hexdigest()
    except Exception:
        return f"err:{path}"


def df_fingerprint(df: pd.DataFrame, max_rows: int = 2000) -> str:
    """
    Fingerprint a dataframe to dedupe identical tables.
    We hash: columns + first rows + last rows (as strings).
    """
    try:
        if df is None or df.empty:
            return "empty"
        cols = [str(c) for c in df.columns.tolist()]
        n = len(df)
        # sample head + tail for stability
        head = df.head(min(max_rows // 2, n)).astype(str)
        tail = df.tail(min(max_rows // 2, n)).astype(str) if n > 1 else head
        payload = {
            "cols": cols,
            "head": head.to_dict(orient="records"),
            "tail": tail.to_dict(orient="records"),
            "n": n,
        }
        s = str(payload).encode("utf-8", errors="ignore")
        return hashlib.sha1(s).hexdigest()
    except Exception:
        return "df_err"


# -----------------------------
# Count helpers (unique when possible)
# -----------------------------
def pick_id_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    if df is None or df.empty:
        return None
    cols = [str(c) for c in df.columns]
    low = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand.lower() in low:
            return low[cand.lower()]
    # fallback: contains match
    for c in cols:
        cl = c.lower()
        for cand in candidates:
            if cand.lower() in cl:
                return c
    return None


def unique_or_rows_count(df: pd.DataFrame, id_candidates: List[str]) -> Tuple[Optional[int], str]:
    """
    Returns (count, label), where label is 'unique' or 'rows' or '—'.
    """
    if df is None or df.empty:
        return None, "—"
    col = pick_id_column(df, id_candidates)
    if col:
        try:
            return int(df[col].astype(str).nunique(dropna=True)), "unique"
        except Exception:
            pass
    return int(len(df)), "rows"


# -----------------------------
# Discovery functions (QC / Selection / Enrichment / Pathways)
# -----------------------------
def discover_qc_images(base: Path) -> List[Path]:
    qc_dirs = [p for p in base.rglob("*") if p.is_dir() and p.name.lower() == "qc"]
    candidates: List[Path] = []
    for d in qc_dirs:
        candidates.extend([p for p in d.glob("*.png") if p.suffix.lower() in IMAGE_EXTS])

    if not candidates:
        candidates = find_all(base, ["*Consistency*.png", "*DensityView*.png", "*MAView*.png", "*ViolinView*.png"])

    pref = ["consistency", "densityview", "maview", "violinview"]
    def score(p: Path) -> Tuple[int, str]:
        name = p.name.lower()
        for i, key in enumerate(pref):
            if key in name:
                return (i, name)
        return (len(pref), name)

    uniq = {str(p): p for p in candidates}
    out = list(uniq.values())
    out.sort(key=score)
    return out


def discover_selection_images(base: Path) -> List[Path]:
    patterns = [
        "*Selection*/*ScatterView*.png",
        "*selection*/*ScatterView*.png",
        "*Selection*/*SquareView*.png",
        "*selection*/*SquareView*.png",
        "*Selection*/*Treat*Ctrl*.png",
        "*selection*/*Treat*Ctrl*.png",
        "*MAGeCKFlute*/*ScatterView*.png",
        "*MAGeCKFlute*/*SquareView*.png",
        "*MAGeCKFlute*/*Treat*Ctrl*.png",
    ]
    imgs = find_all(base, patterns)
    # exclude very long rank view if present (historical request)
    imgs = [p for p in imgs if "rankview" not in p.name.lower()]

    order_keys = ["treat", "ctrl", "negative", "positive", "scatterview", "squareview"]
    def score(p: Path) -> Tuple[int, int, str]:
        nm = p.name.lower()
        best = 99
        for i, k in enumerate(order_keys):
            if k in nm:
                best = min(best, i)
        return (best, len(nm), nm)

    imgs.sort(key=score)
    return imgs


def discover_enrichment_assets(base: Path) -> Tuple[Dict[str, List[Path]], Dict[str, List[Path]]]:
    """
    Return:
      (png_by_group, txt_by_group)
    Groups inferred from filenames starting with GroupXX_
    """
    enr_pngs = find_all(base, ["*Enrichment*/Group*.png", "*enrichment*/Group*.png"])
    enr_txts = find_all(base, ["*Enrichment*/Group*.txt", "*enrichment*/Group*.txt"])

    def group_key(p: Path) -> str:
        m = re.match(r"(Group\d+)", p.name, flags=re.IGNORECASE)
        return m.group(1).capitalize() if m else "Group"

    png_by_group: Dict[str, List[Path]] = {}
    txt_by_group: Dict[str, List[Path]] = {}

    for p in enr_pngs:
        png_by_group.setdefault(group_key(p), []).append(p)
    for p in enr_txts:
        txt_by_group.setdefault(group_key(p), []).append(p)

    for g in png_by_group:
        png_by_group[g].sort(key=lambda x: x.name.lower())
    for g in txt_by_group:
        txt_by_group[g].sort(key=lambda x: x.name.lower())

    return png_by_group, txt_by_group


def discover_pathway_images(base: Path) -> List[Path]:
    pats = [
        "*PathwayView*/*.png",
        "*pathwayview*/*.png",
        "*pathview*/*.png",
        "*Pathview*/*.png",
    ]
    imgs = find_all(base, pats)
    # sometimes these folders include non-pathway PNGs; keep those that look like pathview outputs
    scored = []
    for p in imgs:
        nm = p.name.lower()
        score = 0
        if "pathview" in nm or "pathway" in nm:
            score -= 5
        if "hsa" in nm or "mmu" in nm:
            score -= 3
        if "group" in nm:
            score -= 2
        scored.append((score, len(nm), nm, p))
    scored.sort(key=lambda t: (t[0], t[1], t[2]))
    return [t[3] for t in scored]


# -----------------------------
# Data structures
# -----------------------------
@dataclass
class TablePanel:
    panel_id: str
    title: str
    caption: str
    df: pd.DataFrame
    truncated: bool
    font_small: bool = False


@dataclass
class ImagePanel:
    title: str
    caption: str
    images: List[Tuple[str, str]]  # (label, data_uri)
    layout: str
    accent_class: str


# -----------------------------
# Landscape table builder
# -----------------------------
def build_landscape_table(
    rra_gene_df_full: pd.DataFrame,
    rra_sgrna_df_full: pd.DataFrame,
    mle_df_full: pd.DataFrame,
    qc_unique_count: int,
    sel_unique_count: int,
    enr_group_count: int,
    enr_table_count: int,
    pathway_unique_total: int,
    sgrna_count_label: str,
    gene_rra_label: str,
    gene_mle_label: str,
) -> pd.DataFrame:
    def fmt_count(x: Optional[int], label: str, unit: str) -> str:
        if x is None:
            return f"— {unit}"
        if label == "unique":
            return f"{x:,} {unit} (unique)"
        if label == "rows":
            return f"{x:,} {unit} (rows)"
        return f"{x:,} {unit}"

    gene_rra_n, _ = unique_or_rows_count(rra_gene_df_full, ["gene", "gene_id", "symbol", "Gene"])
    sgrna_n, _ = unique_or_rows_count(rra_sgrna_df_full, ["sgrna", "sgRNA", "guide", "sequence", "id", "sgrna_id"])
    gene_mle_n, _ = unique_or_rows_count(mle_df_full, ["gene", "gene_id", "symbol", "Gene"])

    rows = [
        {
            "Analysis layer": "Gene-level essentiality (RRA)",
            "Data scope": suggesting_or_dash(fmt_count(gene_rra_n, gene_rra_label, "genes")),
            "Key outputs": "MAGeCK RRA scores, effect sizes, FDR-ranked depletion/enrichment",
            "Biological interpretation": "Genes whose loss significantly decreases or increases cellular fitness under treatment",
        },
        {
            "Analysis layer": "sgRNA-level CRISPR effects",
            "Data scope": suggesting_or_dash(fmt_count(sgrna_n, sgrna_count_label, "sgRNAs")),
            "Key outputs": "Per-guide counts, LFC/effects, variance metrics, enrichment flags (when present)",
            "Biological interpretation": "Guide-specific activity and consistency supporting gene-level selection signals",
        },
        {
            "Analysis layer": "Model-based gene effects (MLE / Flute)",
            "Data scope": suggesting_or_dash(fmt_count(gene_mle_n, gene_mle_label, "genes")),
            "Key outputs": "Beta coefficients, Wald/z statistics, contrasts (format-dependent)",
            "Biological interpretation": "Robust estimation of gene effects accounting for sgRNA efficiency and variance",
        },
        {
            "Analysis layer": "Screen quality control",
            "Data scope": f"{qc_unique_count} diagnostic plots",
            "Key outputs": "Density, MA, violin, consistency views (as available)",
            "Biological interpretation": "Validation of reproducibility, variance structure, and model calibration",
        },
        {
            "Analysis layer": "Gene-level selection & ranking",
            "Data scope": "Ranked genome-wide gene set",
            "Key outputs": f"{sel_unique_count} selection figures (unique)",
            "Biological interpretation": "Prioritization of candidate sensitizers and resistance-associated genes",
        },
        {
            "Analysis layer": "Functional enrichment (complexes & GO)",
            "Data scope": f"{enr_group_count}+ gene groups",
            "Key outputs": f"{enr_table_count} enrichment tables (unique) + per-group enrichment plots",
            "Biological interpretation": "Functional programs overrepresented among selected gene groups",
        },
        {
            "Analysis layer": "Pathway-level visualization",
            "Data scope": "Top pathways shown",
            "Key outputs": f"Top 5 figures shown; {pathway_unique_total} available in outputs (unique)",
            "Biological interpretation": "Network-level mapping of treatment-associated perturbations",
        },
    ]
    return pd.DataFrame(rows)


def suggesting_or_dash(s: str) -> str:
    return s if s and s.strip() else "—"


# -----------------------------
# Main builder
# -----------------------------
def build_report(base: Path, out_html: Path) -> None:
    generated = now_stamp()

    # logos
    logo_left_path, logo_right_path = pick_logo_pair(base)
    logo_left = b64_image_data_uri(logo_left_path) if logo_left_path else None
    logo_right = b64_image_data_uri(logo_right_path) if logo_right_path else None

    # ------------------ RRA discovery ------------------
    rra_gene_path = find_first(base, ["*mode1*/*gene_summary*.txt", "*rra*/*gene_summary*.txt", "*gene_summary*.txt"])
    rra_sgrna_path = find_first(base, ["*mode1*/*sgrna_summary*.txt", "*rra*/*sgrna_summary*.txt", "*sgrna_summary*.txt"])

    # RRA figures
    rra_rank_img = find_first(base, ["*mode1*/*rank*.png", "*rra*/*rank*.png", "*control_vs_treatment_rank*.png", "*Rank*.png"])
    rra_scatter_img = find_first(base, ["*mode1*/*scatter*.png", "*rra*/*scatter*.png", "*Scatter*.png", "*scatterview*.png"])

    rra_gene_df_full = read_table_any(rra_gene_path) if rra_gene_path and rra_gene_path.exists() else pd.DataFrame()
    rra_sgrna_df_full = read_table_any(rra_sgrna_path) if rra_sgrna_path and rra_sgrna_path.exists() else pd.DataFrame()

    # Unique vs rows labels
    gene_rra_n, gene_rra_label = unique_or_rows_count(rra_gene_df_full, ["gene", "gene_id", "symbol", "Gene"])
    sgrna_n, sgrna_label = unique_or_rows_count(rra_sgrna_df_full, ["sgrna", "sgRNA", "guide", "sequence", "id", "sgrna_id"])

    # RRA tables (top 100)
    rra_tables: List[TablePanel] = []
    if not rra_sgrna_df_full.empty:
        df_sorted = best_guess_sort(rra_sgrna_df_full)
        df_top, truncated = top_n_df(df_sorted, 100)
        rra_tables.append(TablePanel(
            panel_id="rra_table_sgrna",
            title=TABLE_SGRNA_TITLE,
            caption=TABLE_SGRNA_CAPTION,
            df=df_top,
            truncated=truncated,
            font_small=True,
        ))

    if not rra_gene_df_full.empty:
        df_sorted = best_guess_sort(rra_gene_df_full)
        df_top, truncated = top_n_df(df_sorted, 100)
        rra_tables.append(TablePanel(
            panel_id="rra_table_gene",
            title=TABLE_GENE_RRA_TITLE,
            caption=TABLE_GENE_RRA_CAPTION,
            df=df_top,
            truncated=truncated,
            font_small=True,
        ))

    # RRA images (dedupe by file fingerprint)
    rra_img_panels: List[ImagePanel] = []
    two_imgs: List[Tuple[str, str]] = []
    seen_img_fp = set()

    def add_img(label: str, p: Optional[Path]) -> None:
        if not p or not p.exists():
            return
        fp = file_fingerprint_fast(p)
        if fp in seen_img_fp:
            return
        seen_img_fp.add(fp)
        uri = b64_image_data_uri(p)
        if uri:
            two_imgs.append((label, uri))

    add_img("Ranked effects", rra_rank_img)
    add_img("Score vs significance", rra_scatter_img)

    if two_imgs:
        caption = (
            f"<b>{html.escape(RRA_FIG_TITLE_1)}.</b> {html.escape(RRA_FIG_CAP_1)}"
        )
        if len(two_imgs) > 1:
            caption += f"<br><br><b>{html.escape(RRA_FIG_TITLE_2)}.</b> {html.escape(RRA_FIG_CAP_2)}"
        rra_img_panels.append(ImagePanel(
            title="GENE-LEVEL EFFECTS OVERVIEW",
            caption=caption,
            images=two_imgs[:2],
            layout="two",
            accent_class="accent-rra",
        ))

    # ------------------ MLE discovery ------------------
    mle_path = find_first(base, [
        "*mode2*/*design_matrix.gene_summary*.txt",
        "*mle*/*design_matrix.gene_summary*.txt",
        "*mode2*/*gene_summary*.txt",
        "*mle*/*gene_summary*.txt",
        "*design_matrix.gene_summary*.txt",
    ])
    mle_df_full = read_table_any(mle_path) if mle_path and mle_path.exists() else pd.DataFrame()
    gene_mle_n, gene_mle_label = unique_or_rows_count(mle_df_full, ["gene", "gene_id", "symbol", "Gene"])

    mle_tables: List[TablePanel] = []
    if not mle_df_full.empty:
        df_sorted = best_guess_sort(mle_df_full)
        df_top, truncated = top_n_df(df_sorted, 100)
        mle_tables.append(TablePanel(
            panel_id="mle_table_gene",
            title=TABLE_MLE_TITLE,
            caption=TABLE_MLE_CAPTION,
            df=df_top,
            truncated=truncated,
            font_small=True,
        ))

    # ------------------ QC images (dedupe) ------------------
    qc_paths = discover_qc_images(base)
    qc_seen = set()
    qc_images: List[Tuple[str, str]] = []
    label_counts: Dict[str, int] = {}
    for p in qc_paths:
        fp = file_fingerprint_fast(p)
        if fp in qc_seen:
            continue
        qc_seen.add(fp)

        uri = b64_image_data_uri(p)
        if not uri:
            continue

        lbl = pretty_label_from_path(p)
        # disambiguate labels if repeated
        label_counts.setdefault(lbl, 0)
        label_counts[lbl] += 1
        if label_counts[lbl] > 1:
            lbl = f"{lbl} ({p.parent.name})"
        qc_images.append((lbl, uri))

        if len(qc_images) >= 8:
            break

    qc_panel: Optional[ImagePanel] = None
    if qc_images:
        qc_panel = ImagePanel(
            title=QC_TITLE,
            caption=QC_CAPTION,
            images=qc_images,
            layout="grid_2x4",
            accent_class="accent-qc",
        )

    # ------------------ Selection images (dedupe) ------------------
    sel_paths = discover_selection_images(base)
    sel_seen = set()
    sel_images: List[Tuple[str, str]] = []
    label_counts_sel: Dict[str, int] = {}
    for p in sel_paths:
        fp = file_fingerprint_fast(p)
        if fp in sel_seen:
            continue
        sel_seen.add(fp)

        uri = b64_image_data_uri(p)
        if not uri:
            continue
        lbl = pretty_label_from_path(p)
        label_counts_sel.setdefault(lbl, 0)
        label_counts_sel[lbl] += 1
        if label_counts_sel[lbl] > 1:
            lbl = f"{lbl} ({p.parent.name})"
        sel_images.append((lbl, uri))
        if len(sel_images) >= 6:
            break

    sel_panel: Optional[ImagePanel] = None
    if sel_images:
        sel_panel = ImagePanel(
            title=SEL_TITLE,
            caption=SEL_CAPTION,
            images=sel_images[:4],  # 2x2 for clean look
            layout="grid_4",
            accent_class="accent-sel",
        )

    sel_unique_count = len(sel_seen)

    # ------------------ Enrichment (dedupe plots + tables) ------------------
    png_by_group, txt_by_group = discover_enrichment_assets(base)
    enr_groups = sorted(set(list(png_by_group.keys()) + list(txt_by_group.keys())))

    enrichment_plot_panels: List[ImagePanel] = []
    enrichment_table_panels: List[TablePanel] = []

    # Dedup trackers
    enr_img_seen = set()
    enr_tbl_seen = set()

    # Enrichment plots: for each group, show up to 4 unique plots
    for g in enr_groups:
        plots = []
        for p in png_by_group.get(g, []):
            fp = file_fingerprint_fast(p)
            if fp in enr_img_seen:
                continue
            enr_img_seen.add(fp)
            uri = b64_image_data_uri(p)
            if not uri:
                continue
            lbl = pretty_label_from_path(p).replace("cell cycle", "").strip()
            plots.append((lbl, uri))
            if len(plots) >= 4:
                break

        if plots:
            enrichment_plot_panels.append(ImagePanel(
                title=f"{ENR_TITLE} — {g.upper()}",
                caption=ENR_CAPTION,
                images=plots,
                layout="grid_4",
                accent_class="accent-enr",
            ))

        # Enrichment tables: show TOP 5 rows for each unique table (dedupe by df fingerprint)
        # To avoid spam, we take up to 2 tables per group (common outputs: complex + gobp)
        picked = 0
        for tpath in txt_by_group.get(g, []):
            df = read_table_any(tpath)
            if df is None or df.empty:
                continue
            df_sorted = best_guess_sort(df)
            top5 = df_sorted.head(5).copy()
            fp_df = df_fingerprint(top5)
            if fp_df in enr_tbl_seen:
                continue
            enr_tbl_seen.add(fp_df)

            enrichment_table_panels.append(TablePanel(
                panel_id=f"enr_{slug(g)}_{picked+1}",
                title=f"TOP ENRICHED CATEGORIES — {g.upper()}",
                caption="Top 5 enriched categories are shown for readability. Please refer to the pipeline outputs for complete enrichment results.",
                df=top5,
                truncated=False,
                font_small=True,
            ))
            picked += 1
            if picked >= 2:
                break

    enr_group_count = len(enr_groups)
    enr_table_count = len(enrichment_table_panels)

    # ------------------ Pathways (dedupe + TOP 5 one-by-one) ------------------
    pathway_paths = discover_pathway_images(base)
    path_seen = set()
    unique_pathways: List[Path] = []
    for p in pathway_paths:
        fp = file_fingerprint_fast(p)
        if fp in path_seen:
            continue
        path_seen.add(fp)
        unique_pathways.append(p)

    pathway_unique_total = len(unique_pathways)

    top5_path_imgs: List[Tuple[str, str]] = []
    for p in unique_pathways[:5]:
        uri = b64_image_data_uri(p)
        if not uri:
            continue
        title = normalize_pathway_display_name(p.stem)
        top5_path_imgs.append((title, uri))

    # ------------------ Summary table (REAL counts) ------------------
    landscape_df = build_landscape_table(
        rra_gene_df_full=rra_gene_df_full,
        rra_sgrna_df_full=rra_sgrna_df_full,
        mle_df_full=mle_df_full,
        qc_unique_count=len(qc_seen),
        sel_unique_count=sel_unique_count if sel_unique_count > 0 else len(sel_images),
        enr_group_count=enr_group_count,
        enr_table_count=enr_table_count,
        pathway_unique_total=pathway_unique_total,
        sgrna_count_label=sgrna_label,
        gene_rra_label=gene_rra_label,
        gene_mle_label=gene_mle_label,
    )
    landscape_cols, landscape_rows = df_to_records(landscape_df)

    # ------------------ Serialize for template ------------------
    env = Environment(
        loader=BaseLoader(),
        autoescape=False,
        variable_start_string="[[",
        variable_end_string="]]",
        block_start_string="[%",   # avoid conflict with JS braces
        block_end_string="%]",
    )

    template = env.from_string(HTML_TEMPLATE)
    html_out = template.render(
        generated=generated,
        header_title=HEADER_TITLE,
        header_paragraph=HEADER_PARAGRAPH,
        logo_left=logo_left,
        logo_right=logo_right,

        # landscape
        landscape_desc=LANDSCAPE_DESC,
        landscape_cols=[html.escape(c) for c in landscape_cols],
        landscape_rows=[[html.escape(x) for x in r] for r in landscape_rows],

        # RRA
        rra_desc=RRA_DESC,
        rra_tables=serialize_tables(rra_tables),
        rra_img_panels=serialize_images(rra_img_panels),

        # MLE/Flute
        mle_desc=MLE_DESC,
        mle_tables=serialize_tables(mle_tables),
        qc_panel=serialize_image_panel(qc_panel) if qc_panel else None,
        sel_panel=serialize_image_panel(sel_panel) if sel_panel else None,
        enr_plot_panels=serialize_images(enrichment_plot_panels),
        enr_table_panels=serialize_tables(enrichment_table_panels),

        # Pathways
        path_title=PATH_TITLE,
        path_caption=PATH_CAPTION,
        pathway_unique_total=pathway_unique_total,
        top5_path_imgs=[{"label": html.escape(lbl), "data_uri": uri} for lbl, uri in top5_path_imgs],

        # References
        refs=[{"name": html.escape(n), "url": html.escape(u)} for n, u in REFS],

        # footer
        base_dir=html.escape(str(base)),
        final_note=FINAL_NOTE,
        sgrna_label=sgrna_label,
        gene_rra_label=gene_rra_label,
        gene_mle_label=gene_mle_label,
    )

    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(html_out, encoding="utf-8")
    print(f"✔ Report generated: {out_html}")
    return ReportResult(
        success=True,
        message=f"Report generated: {out_html}",
        report_path=out_html,
    )


def serialize_tables(panels: List[TablePanel]) -> List[dict]:
    out = []
    for p in panels:
        cols, rows = df_to_records(p.df) if p.df is not None else ([], [])
        out.append(
            {
                "panel_id": p.panel_id,
                "title": p.title,
                "caption": p.caption,
                "cols": [html.escape(c) for c in cols],
                "rows": [[html.escape(x) for x in r] for r in rows],
                "truncated": bool(p.truncated),
                "font_small": bool(p.font_small),
            }
        )
    return out


def serialize_image_panel(panel: Optional[ImagePanel]) -> Optional[dict]:
    if panel is None:
        return None
    return {
        "title": panel.title,
        "caption": panel.caption,
        "layout": panel.layout,
        "accent_class": panel.accent_class,
        "images": [{"label": html.escape(lbl), "data_uri": uri} for lbl, uri in panel.images],
    }


def serialize_images(panels: List[ImagePanel]) -> List[dict]:
    return [serialize_image_panel(p) for p in panels if p is not None]


# -----------------------------
# HTML Template (complete)
# -----------------------------
HTML_TEMPLATE = r"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>CRISPR Based Genetic Screening Report</title>

  <style>
    :root{
      --bg:#f4f7fb;
      --card:#ffffff;
      --ink:#0f172a;
      --muted:#5b6474;
      --line:#e6eaf2;

      /* brand */
      --teal1:#00a7b5;
      --teal2:#0aa37f;
      --teal3:#0b7ea6;

      /* pastel accents */
      --p_land:#EAF7FF;
      --p_rra:#FFF6EA;
      --p_mle:#EEF9F1;
      --p_qc:#F3EEFF;
      --p_sel:#FFEFF3;
      --p_enr:#F0FBF6;
      --p_path:#F7FAFF;
      --shadow: 0 10px 30px rgba(15, 23, 42, .08);
      --shadow2: 0 8px 18px rgba(15, 23, 42, .06);
      --radius: 18px;
      --mono: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
      --sans: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, "Apple Color Emoji","Segoe UI Emoji";
    }

    *{ box-sizing:border-box; }
    body{
      margin:0;
      font-family: var(--sans);
      color:var(--ink);
      background: var(--bg);
    }
    .wrap{ max-width: 1200px; margin: 28px auto 80px; padding: 0 18px; }

    /* Header */
    .hero{
      border-radius: 26px;
      overflow:hidden;
      box-shadow: var(--shadow);
      background: linear-gradient(90deg, var(--teal3), var(--teal2));
      color:#fff;
      position:relative;
    }
    .hero-inner{
      display:grid;
      grid-template-columns: 120px 1fr 120px;
      gap: 18px;
      align-items:center;
      padding: 22px 26px;
      min-height: 170px;
    }
    .logo{
      width: 110px;
      height: 110px;
      object-fit: contain;
      background: rgba(255,255,255,.12);
      border-radius: 18px;
      padding: 10px;
    }
    .hero-title{
      margin:0;
      font-size: 30px;
      letter-spacing: .3px;
      line-height: 1.12;
      font-weight: 900;
    }
    .hero-sub{
      margin: 10px 0 0;
      font-size: 14.5px;
      line-height: 1.55;
      color: rgba(255,255,255,.92);
      max-width: 920px;
    }
    .hero-meta{
      position:absolute;
      right: 18px;
      top: 14px;
      font-size: 12px;
      letter-spacing: .4px;
      color: rgba(255,255,255,.9);
    }

    .pill-row{
      margin-top: 14px;
      display:flex;
      gap:10px;
      flex-wrap:wrap;
    }
    .pill{
      padding: 8px 12px;
      border-radius: 999px;
      font-size: 12px;
      font-weight: 900;
      letter-spacing: .35px;
      border: 1px solid rgba(255,255,255,.22);
      user-select:none;
    }
    .pill.crispr{ background: rgba(234,247,255,.18); }
    .pill.rra{ background: rgba(255,246,234,.18); }
    .pill.flute{ background: rgba(238,249,241,.18); }

    /* Cards */
    .card{
      background: var(--card);
      border-radius: var(--radius);
      box-shadow: var(--shadow2);
      border: 1px solid var(--line);
    }
    .card.pad{ padding: 18px 18px; }

    .grid2{
      display:grid;
      grid-template-columns: 1fr 1fr;
      gap: 16px;
    }
    @media (max-width: 980px){
      .hero-inner{ grid-template-columns: 1fr; text-align:center; }
      .logo{ margin: 0 auto; }
      .grid2{ grid-template-columns: 1fr; }
    }

    /* SECTION HEADING STYLES (PASTEL + CAPITAL) */
    .section{ margin-top: 18px; }
    .section .h2wrap{
      border-radius: 16px;
      padding: 10px 14px;
      border: 1px solid var(--line);
      box-shadow: var(--shadow2);
      margin: 18px 0 12px;
    }
    .section h2{
      margin:0;
      font-size: 16px;
      letter-spacing: .9px;
      text-transform: uppercase;
      font-weight: 950;
      color: #0b4f6c;
    }
    .h2wrap.land{ background: linear-gradient(90deg, var(--p_land), #ffffff); }
    .h2wrap.rra{ background: linear-gradient(90deg, var(--p_rra), #ffffff); }
    .h2wrap.mle{ background: linear-gradient(90deg, var(--p_mle), #ffffff); }
    .h2wrap.refs{ background: linear-gradient(90deg, #ffffff, #ffffff); border-style:dashed; }

    .section p{
      margin: 6px 2px 14px;
      color: var(--muted);
      line-height: 1.65;
      font-size: 14px;
    }

    .accent{
      border-radius: 18px;
      border: 1px solid var(--line);
      padding: 14px 16px;
    }
    .accent-land{ background: var(--p_land); }
    .accent-rra{ background: var(--p_rra); }
    .accent-mle{ background: var(--p_mle); }
    .accent-enr{ background: var(--p_enr); }
    .accent-path{ background: var(--p_path); }

    .subhead{
      display:flex;
      align-items:flex-end;
      justify-content:space-between;
      gap:10px;
      margin: 0 0 10px;
    }
    .subhead h3{
      margin: 0;
      font-size: 18px;
      font-weight: 950;
      letter-spacing:.2px;
      color:#0f172a;
    }
    .subhead .hint{
      font-size: 12px;
      color: var(--muted);
      font-weight: 800;
    }

    /* Table panels */
    .panel{
      border: 1px solid var(--line);
      border-radius: 18px;
      overflow:hidden;
      background:#fff;
    }
    .panel-hd{
      padding: 14px 16px;
      border-bottom: 1px solid var(--line);
      display:flex;
      align-items:center;
      justify-content:space-between;
      gap: 10px;
      flex-wrap:wrap;
    }
    .panel-hd.landscape{ background: linear-gradient(90deg, var(--p_land), #ffffff); }
    .panel-hd.rra{ background: linear-gradient(90deg, var(--p_rra), #ffffff); }
    .panel-hd.mle{ background: linear-gradient(90deg, var(--p_mle), #ffffff); }
    .panel-hd.enr{ background: linear-gradient(90deg, var(--p_enr), #ffffff); }

    .panel-title{
      font-weight: 950;
      font-size: 13.5px;
      letter-spacing: .4px;
      color:#0b4f6c;
      margin:0;
      text-transform: uppercase;
    }
    .panel-tools{
      display:flex;
      gap: 10px;
      align-items:center;
      flex-wrap:wrap;
    }
    .search{
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 8px 12px;
      min-width: 240px;
      outline:none;
      font-size: 13px;
      background:#fff;
    }

    .table-wrap{
      max-height: 360px;
      overflow:auto;
      position:relative;
    }
    table{
      border-collapse: separate;
      border-spacing: 0;
      width: 100%;
      min-width: 760px;
      font-size: 12px;
      line-height: 1.35;
    }
    table.small{ font-size: 11px; }
    th, td{
      padding: 10px 10px;
      border-bottom: 1px solid var(--line);
      white-space: nowrap;
    }
    th{
      position: sticky;
      top:0;
      z-index: 2;
      background: #f1f6ff;
      font-weight: 950;
      cursor:pointer;
      user-select:none;
      color:#1f2a44;
    }
    tr:nth-child(even) td{ background: #fcfdff; }
    tr:hover td{ background: #f3fbff; }

    .panel-ft{
      padding: 12px 16px;
      border-top: 1px solid var(--line);
      color: var(--muted);
      font-size: 12.5px;
      line-height: 1.55;
    }
    .badge{
      display:inline-block;
      padding: 3px 8px;
      border-radius: 999px;
      border: 1px solid var(--line);
      background:#fff;
      font-size: 11px;
      font-weight: 950;
      color:#0b4f6c;
      margin-left: 10px;
      text-transform: uppercase;
      letter-spacing: .25px;
    }

    /* Image panels */
    .imgpanel{
      border: 1px solid var(--line);
      border-radius: 18px;
      overflow:hidden;
      background:#fff;
    }
    .imgpanel-hd{
      padding: 14px 16px;
      border-bottom: 1px solid var(--line);
    }
    .imgpanel-hd.accent-rra{ background: linear-gradient(90deg, var(--p_rra), #ffffff); }
    .imgpanel-hd.accent-qc{ background: linear-gradient(90deg, var(--p_qc), #ffffff); }
    .imgpanel-hd.accent-sel{ background: linear-gradient(90deg, var(--p_sel), #ffffff); }
    .imgpanel-hd.accent-enr{ background: linear-gradient(90deg, var(--p_enr), #ffffff); }
    .imgpanel-hd.accent-path{ background: linear-gradient(90deg, var(--p_path), #ffffff); }

    .imgpanel-hd h4{
      margin:0;
      font-weight: 950;
      font-size: 13.5px;
      letter-spacing: .4px;
      text-transform: uppercase;
      color:#0b4f6c;
    }
    .imggrid{
      padding: 14px;
      display:grid;
      gap: 14px;
    }
    .imgbox{
      border: 1px solid var(--line);
      border-radius: 16px;
      overflow:hidden;
      background: #fff;
    }
    .imgbox .lbl{
      padding: 8px 10px;
      font-size: 12px;
      font-weight: 950;
      color:#1f2a44;
      background:#f7fbff;
      border-bottom: 1px solid var(--line);
      text-transform: none;
    }
    .imgbox img{
      width:100%;
      height: 320px;
      display:block;
      object-fit: contain;
      background: #ffffff;
      cursor: zoom-in;
    }
    .imgcap{
      padding: 12px 16px;
      border-top: 1px solid var(--line);
      color: var(--muted);
      font-size: 12.5px;
      line-height: 1.55;
    }

    /* Layouts */
    .two{ grid-template-columns: 1fr 1fr; }
    @media (max-width: 980px){ .two{ grid-template-columns:1fr; } }

    .grid_2x4{ grid-template-columns: repeat(4, 1fr); }
    @media (max-width: 1200px){ .grid_2x4{ grid-template-columns: repeat(2, 1fr);} }
    @media (max-width: 560px){ .grid_2x4{ grid-template-columns: 1fr;} }

    .grid_4{ grid-template-columns: repeat(2, 1fr); }
    @media (max-width: 780px){ .grid_4{ grid-template-columns:1fr; } }

    /* Pathways: one-by-one */
    .path-item{ margin-top: 14px; }

    /* Modal zoom */
    .modal{
      position:fixed;
      inset:0;
      background: rgba(15,23,42,.86);
      display:none;
      align-items:center;
      justify-content:center;
      z-index: 9999;
      padding: 18px;
    }
    .modal.open{ display:flex; }
    .modal-inner{
      max-width: 1100px;
      width: 100%;
      background:#fff;
      border-radius: 18px;
      overflow:hidden;
      border:1px solid rgba(255,255,255,.2);
    }
    .modal-hd{
      padding: 10px 14px;
      display:flex;
      align-items:center;
      justify-content:space-between;
      border-bottom: 1px solid var(--line);
      background: #f7fbff;
      font-weight: 950;
      color:#0b4f6c;
    }
    .modal-hd button{
      border:1px solid var(--line);
      background:#fff;
      border-radius: 10px;
      padding: 6px 10px;
      font-weight: 950;
      cursor:pointer;
    }
    .modal-img{
      width:100%;
      height: calc(100vh - 160px);
      object-fit: contain;
      background:#fff;
      remembering: no;
    }

    .footer{
      margin-top: 22px;
      color: #7b8799;
      font-size: 12px;
      text-align:center;
      line-height: 1.6;
    }
    .kbd{
      font-family: var(--mono);
      background:#fff;
      border:1px solid var(--line);
      padding: 2px 6px;
      border-radius: 8px;
    }

    a{ color:#0b7ea6; text-decoration:none; font-weight:950; }
    a:hover{ text-decoration:underline; }
  </style>
</head>

<body>
  <div class="wrap">
    <div class="hero">
      <div class="hero-meta">Generated: [[ generated ]]</div>
      <div class="hero-inner">
        <div>
          [% if logo_left %]
            <img class="logo" src="[[ logo_left ]]" alt="Left logo"/>
          [% else %]
            <div class="logo" style="display:flex;align-items:center;justify-content:center;font-weight:950;">LOGO</div>
          [% endif %]
        </div>

        <div>
          <div style="font-size:12px;letter-spacing:.35px;font-weight:950;opacity:.95;">
            AYASS BIOSCIENCE · FUNCTIONAL GENOMICS
          </div>
          <h1 class="hero-title">[[ header_title ]]</h1>
          <p class="hero-sub">[[ header_paragraph ]]</p>
          <div class="pill-row">
            <div class="pill crispr">CRISPR SCREENING</div>
            <div class="pill rra">MAGeCK RRA</div>
            <div class="pill flute">MAGeCK MLE / FLUTE</div>
          </div>
        </div>

        <div>
          [% if logo_right %]
            <img class="logo" src="[[ logo_right ]]" alt="Right logo"/>
          [% else %]
            <div class="logo" style="display:flex;align-items:center;justify-content:center;font-weight:950;">LOGO</div>
          [% endif %]
        </div>
      </div>
    </div>

    <!-- LANDSCAPE -->
    <div class="section">
      <div class="h2wrap land"><h2>OVERALL CRISPR SCREENING LANDSCAPE</h2></div>

      <div class="card pad accent accent-land">
        <div class="subhead">
          <h3>Overall CRISPR screening landscape</h3>
          <div class="hint">Counts derived from detected outputs</div>
        </div>
        <p style="margin:0;">[[ landscape_desc ]]</p>
        <p style="margin:10px 0 0;">
          <b>Note:</b> Counts are derived from detected output files. sgRNA/gene counts are marked as <b>unique</b> when an ID column is detected
          (otherwise shown as <b>rows</b>). This prevents misleading totals when tables contain repeated entries.
        </p>
      </div>

      <div style="height:14px"></div>

      <div class="panel" id="landscape_table">
        <div class="panel-hd landscape">
          <div class="panel-title">OVERALL CRISPR SCREENING LANDSCAPE</div>
          <div class="panel-tools">
            <input class="search" placeholder="Search..." oninput="filterTable('landscape_table', this.value)">
          </div>
        </div>
        <div class="table-wrap" style="max-height:420px;">
          <table data-panel="landscape_table">
            <thead>
              <tr>
                [% for c in landscape_cols %]
                  <th onclick="sortTable('landscape_table', this.cellIndex)">[[ c ]]</th>
                [% endfor %]
              </tr>
            </thead>
            <tbody>
              [% for r in landscape_rows %]
                <tr>
                  [% for x in r %]
                    <td>[[ x ]]</td>
                  [% endfor %]
                </tr>
              [% endfor %]
            </tbody>
          </table>
        </div>
      </div>
    </div>

    <!-- RRA -->
    <div class="section">
      <div class="h2wrap rra"><h2>ROBUST RANK AGGREGATION (COUNTS → RRA)</h2></div>

      <div class="card pad accent accent-rra">
        <div class="subhead">
          <h3>Data incorporated</h3>
          <div class="hint">sgRNA summary, gene summary, and key RRA plots</div>
        </div>
        <p style="margin:0;">[[ rra_desc ]]</p>
      </div>

      <div style="height:14px"></div>

      [% if rra_tables|length > 0 %]
        <div class="grid2">
          [% for t in rra_tables %]
            <div class="panel" id="[[ t.panel_id ]]">
              <div class="panel-hd rra">
                <div class="panel-title">[[ t.title ]]</div>
                <div class="panel-tools">
                  <input class="search" placeholder="Search..." oninput="filterTable('[[ t.panel_id ]]', this.value)">
                </div>
              </div>
              <div class="table-wrap">
                <table class="[% if t.font_small %]small[% endif %]" data-panel="[[ t.panel_id ]]">
                  <thead>
                    <tr>
                      [% for c in t.cols %]
                        <th onclick="sortTable('[[ t.panel_id ]]', this.cellIndex)">[[ c ]]</th>
                      [% endfor %]
                    </tr>
                  </thead>
                  <tbody>
                    [% for r in t.rows %]
                      <tr>
                        [% for x in r %]
                          <td>[[ x ]]</td>
                        [% endfor %]
                      </tr>
                    [% endfor %]
                  </tbody>
                </table>
              </div>
              <div class="panel-ft">
                [[ t.caption ]]
                [% if t.truncated %]
                  <span class="badge">TOP 100 SHOWN</span>
                [% endif %]
              </div>
            </div>
          [% endfor %]
        </div>
      [% endif %]

      <div style="height:14px"></div>

      [% if rra_img_panels|length > 0 %]
        [% for p in rra_img_panels %]
          <div class="imgpanel">
            <div class="imgpanel-hd [[ p.accent_class ]]"><h4>[[ p.title ]]</h4></div>
            <div class="imggrid [[ p.layout ]]">
              [% for im in p.images %]
                <div class="imgbox">
                  <div class="lbl">[[ im.label ]]</div>
                  <img src="[[ im.data_uri ]]" alt="[[ im.label ]]" onclick="openModal('[[ im.label ]]', '[[ im.data_uri ]]')">
                </div>
              [% endfor %]
            </div>
            <div class="imgcap">[[ p.caption ]]</div>
          </div>
        [% endfor %]
      [% endif %]
    </div>

    <!-- MLE / FLUTE -->
    <div class="section">
      <div class="h2wrap mle"><h2>MODEL-BASED ESTIMATION (COUNTS → MLE / FLUTE)</h2></div>

      <div class="card pad accent accent-mle">
        <div class="subhead">
          <h3>Data incorporated</h3>
          <div class="hint">MLE summary, quality control, selection/ranking, enrichment, pathways</div>
        </div>
        <p style="margin:0;">[[ mle_desc ]]</p>
      </div>

      <div style="height:14px"></div>

      [% if mle_tables|length > 0 %]
        <div class="grid2">
          [% for t in mle_tables %]
            <div class="panel" id="[[ t.panel_id ]]">
              <div class="panel-hd mle">
                <div class="panel-title">[[ t.title ]]</div>
                <div class="panel-tools">
                  <input class="search" placeholder="Search..." oninput="filterTable('[[ t.panel_id ]]', this.value)">
                </div>
              </div>
              <div class="table-wrap">
                <table class="[% if t.font_small %]small[% endif %]" data-panel="[[ t.panel_id ]]">
                  <thead>
                    <tr>
                      [% for c in t.cols %]
                        <th onclick="sortTable('[[ t.panel_id ]]', this.cellIndex)">[[ c ]]</th>
                      [% endfor %]
                    </tr>
                  </thead>
                  <tbody>
                    [% for r in t.rows %]
                      <tr>
                        [% for x in r %]
                          <td>[[ x ]]</td>
                        [% endfor %]
                      </tr>
                    [% endfor %]
                  </tbody>
                </table>
              </div>
              <div class="panel-ft">
                [[ t.caption ]]
                [% if t.truncated %]
                  <span class="badge">TOP 100 SHOWN</span>
                [% endif %]
              </div>
            </div>
          [% endfor %]
        </div>
      [% endif %]

      <div style="height:14px"></div>

      [% if qc_panel %]
        <div class="imgpanel">
          <div class="imgpanel-hd [[ qc_panel.accent_class ]]"><h4>[[ qc_panel.title ]]</h4></div>
          <div class="imggrid [[ qc_panel.layout ]]">
            [% for im in qc_panel.images %]
              <div class="imgbox">
                <div class="lbl">[[ im.label ]]</div>
                <img src="[[ im.data_uri ]]" alt="[[ im.label ]]" onclick="openModal('[[ im.label ]]', '[[ im.data_uri ]]')">
              </div>
            [% endfor %]
          </div>
          <div class="imgcap">
            <b>What this shows.</b> [[ qc_panel.caption ]]
          </div>
        </div>
      [% endif %]

      <div style="height:14px"></div>

      [% if sel_panel %]
        <div class="imgpanel">
          <div class="imgpanel-hd [[ sel_panel.accent_class ]]"><h4>[[ sel_panel.title ]]</h4></div>
          <div class="imggrid [[ sel_panel.layout ]]">
            [% for im in sel_panel.images %]
              <div class="imgbox">
                <div class="lbl">[[ im.label ]]</div>
                <img src="[[ im.data_uri ]]" alt="[[ im.label ]]" onclick="openModal('[[ im.label ]]', '[[ im.data_uri ]]')">
              </div>
            [% endfor %]
          </div>
          <div class="imgcap">
            <b>What this shows.</b> [[ sel_panel.caption ]]
          </div>
        </div>
      [% endif %]

      <div style="height:14px"></div>

      [% if enr_plot_panels|length > 0 %]
        <div class="card pad accent accent-enr">
          <div class="subhead">
            <h3>Functional enrichment</h3>
            <div class="hint">Plots + tables are deduplicated</div>
          </div>
          <p style="margin:0;">
            [[ enr_plot_panels[0].caption ]]
          </p>
        </div>

        <div style="height:14px"></div>

        [% for p in enr_plot_panels %]
          <div class="imgpanel">
            <div class="imgpanel-hd [[ p.accent_class ]]"><h4>[[ p.title ]]</h4></div>
            <div class="imggrid [[ p.layout ]]">
              [% for im in p.images %]
                <div class="imgbox">
                  <div class="lbl">[[ im.label ]]</div>
                  <img src="[[ im.data_uri ]]" alt="[[ im.label ]]" onclick="openModal('[[ im.label ]]', '[[ im.data_uri ]]')">
                </div>
              [% endfor %]
            </div>
            <div class="imgcap"><b>What this shows.</b> [[ p.caption ]]</div>
          </div>
          <div style="height:14px"></div>
        [% endfor %]
      [% endif %]

      [% if enr_table_panels|length > 0 %]
        <div class="grid2">
          [% for t in enr_table_panels %]
            <div class="panel" id="[[ t.panel_id ]]">
              <div class="panel-hd enr">
                <div class="panel-title">[[ t.title ]]</div>
                <div class="panel-tools">
                  <input class="search" placeholder="Search..." oninput="filterTable('[[ t.panel_id ]]', this.value)">
                </div>
              </div>
              <div class="table-wrap">
                <table class="small" data-panel="[[ t.panel_id ]]">
                  <thead>
                    <tr>
                      [% for c in t.cols %]
                        <th onclick="sortTable('[[ t.panel_id ]]', this.cellIndex)">[[ c ]]</th>
                      [% endfor %]
                    </tr>
                  </thead>
                  <tbody>
                    [% for r in t.rows %]
                      <tr>
                        [% for x in r %]
                          <td>[[ x ]]</td>
                        [% endfor %]
                      </tr>
                    [% endfor %]
                  </tbody>
                </table>
              </div>
              <div class="panel-ft">[[ t.caption ]]</div>
            </div>
          [% endfor %]
        </div>
      [% endif %]

      <div style="height:14px"></div>

      <!-- PATHWAYS -->
      <div class="card pad accent accent-path">
        <div class="subhead">
          <h3>[[ path_title ]]</h3>
          <div class="hint">Top 5 shown · no pathway table detected/added</div>
        </div>
        <p style="margin:0;">[[ path_caption ]]</p>
        <p style="margin:10px 0 0;">
          <b>Detected in outputs:</b> [[ pathway_unique_total ]] pathway figures (unique).
        </p>
      </div>

      [% if top5_path_imgs|length > 0 %]
        [% for im in top5_path_imgs %]
          <div class="imgpanel path-item">
            <div class="imgpanel-hd accent-path"><h4>PATHWAY FIGURE — [[ im.label ]]</h4></div>
            <div class="imggrid" style="grid-template-columns: 1fr;">
              <div class="imgbox">
                <div class="lbl">[[ im.label ]]</div>
                <img src="[[ im.data_uri ]]" alt="[[ im.label ]]" onclick="openModal('[[ im.label ]]', '[[ im.data_uri ]]')">
              </div>
            </div>
            <div class="imgcap">
              <b>What this shows.</b> Pathway-level mapping of gene-level effects for the pathway above. Use the full outputs for the complete pathway set.
            </div>
          </div>
        [% endfor %]
      [% endif %]
    </div>

    <!-- REFERENCES -->
    <div class="section">
      <div class="h2wrap refs"><h2>REFERENCES</h2></div>
      <div class="card pad">
        <ul style="margin:0 0 0 18px;color:var(--muted);line-height:1.75;">
          [% for r in refs %]
            <li><a href="[[ r.url ]]" target="_blank" rel="noopener">[[ r.name ]]</a></li>
          [% endfor %]
        </ul>
      </div>
    </div>

    <!--div class="footer">
      Base directory: <span class="kbd">[[ base_dir ]]</span><br/>
      [[ final_note ]]
    </div-->

  </div>

  <!-- Zoom Modal -->
  <div class="modal" id="modal">
    <div class="modal-inner">
      <div class="modal-hd">
        <div id="modalTitle">Figure</div>
        <button onclick="closeModal()">Close</button>
      </div>
      <img id="modalImg" class="modal-img" src="" alt="Zoomed figure"/>
    </div>
  </div>

  <script>
    // --- Table Search ---
    function filterTable(panelId, q){
      q = (q || "").toLowerCase();
      const panel = document.getElementById(panelId);
      if(!panel) return;
      const table = panel.querySelector("table");
      const rows = Array.from(table.tBodies[0].rows);
      rows.forEach(r => {
        const txt = r.innerText.toLowerCase();
        r.style.display = txt.includes(q) ? "" : "none";
      });
    }

    // --- Table Sort ---
    const sortState = {}; // panelId -> {col:int, asc:bool}
    function sortTable(panelId, colIndex){
      const panel = document.getElementById(panelId);
      if(!panel) return;
      const table = panel.querySelector("table");
      const tbody = table.tBodies[0];
      const rows = Array.from(tbody.rows);

      const state = sortState[panelId] || {col:-1, asc:true};
      const asc = (state.col === colIndex) ? !state.asc : true;
      sortState[panelId] = {col: colIndex, asc: asc};

      function parseCell(v){
        const s = (v || "").trim();
        const n = Number(s.replace(/,/g,''));
        if(!Number.isNaN(n) && s !== "") return {t:"n", v:n};
        return {t:"s", v:s.toLowerCase()};
      }

      rows.sort((a,b) => {
        const A = parseCell(a.cells[colIndex]?.innerText || "");
        const B = parseCell(b.cells[colIndex]?.innerText || "");
        if(A.t === "n" && B.t === "n"){
          return asc ? (A.v - B.v) : (B.v - A.v);
        }
        if(A.v < B.v) return asc ? -1 : 1;
        if(A.v > B.v) return asc ? 1 : -1;
        return 0;
      });

      rows.forEach(r => tbody.appendChild(r));
    }

    // --- Modal zoom for images ---
    function openModal(title, src){
      const m = document.getElementById("modal");
      const t = document.getElementById("modalTitle");
      const i = document.getElementById("modalImg");
      t.textContent = title || "Figure";
      i.src = src;
      m.classList.add("open");
    }
    function closeModal(){
      const m = document.getElementById("modal");
      const i = document.getElementById("modalImg");
      i.src = "";
      m.classList.remove("open");
    }
    document.addEventListener("keydown", (e) => {
      if(e.key === "Escape") closeModal();
    });
    document.getElementById("modal").addEventListener("click", (e) => {
      if(e.target.id === "modal") closeModal();
    });
  </script>
</body>
</html>
"""


# -----------------------------
# Public API
# -----------------------------
DEFAULT_REPORT_NAME = "CRISPR_Based_Genetic_Screening_Report.html"


def generate_report(
    results_dir: Union[str, Path],
    output_path: Union[str, Path, None] = None,
) -> ReportResult:
    """Generate an HTML report from pipeline results.

    Parameters
    ----------
    results_dir:
        Directory containing nf-core/crisprseq output (the pipeline's
        ``--outdir``).  The function searches recursively for result
        tables and images.
    output_path:
        Full path for the output HTML file.  When ``None`` the report is
        written to ``<results_dir>/CRISPR_Based_Genetic_Screening_Report.html``.

    Returns
    -------
    ReportResult
        ``.success`` indicates whether the report was written.
        ``.report_path`` is the path to the generated HTML file.
    """
    results_dir = Path(results_dir).resolve()

    if not results_dir.exists():
        return ReportResult(
            success=False,
            message=f"Results directory does not exist: {results_dir}",
        )
    if not results_dir.is_dir():
        return ReportResult(
            success=False,
            message=f"Results path is not a directory: {results_dir}",
        )

    if output_path is None:
        out_html = results_dir / DEFAULT_REPORT_NAME
    else:
        out_html = Path(output_path).resolve()

    try:
        return build_report(results_dir, out_html)
    except Exception as exc:
        return ReportResult(
            success=False,
            message=f"Report generation failed: {exc}",
        )


# -----------------------------
# CLI
# -----------------------------
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate a self-contained HTML report from CRISPR screening results.",
    )
    ap.add_argument("--base", type=str, default=".",
                    help="Base directory containing pipeline results. Default: current directory")
    ap.add_argument("--out", type=str, default=None,
                    help="Output HTML path. Default: <base>/CRISPR_Based_Genetic_Screening_Report.html")
    args = ap.parse_args()

    result = generate_report(args.base, args.out)
    if not result.success:
        print(f"ERROR: {result.message}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()

