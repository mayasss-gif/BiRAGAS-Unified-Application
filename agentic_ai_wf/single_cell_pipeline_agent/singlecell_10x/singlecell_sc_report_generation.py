import os, json, base64, argparse, ast, glob, re, subprocess, shutil, sys, textwrap
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Any
from datetime import datetime

import numpy as np
import pandas as pd

# Matplotlib headless backend
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from jinja2 import Template
from dotenv import load_dotenv

# ======================
# === Constants & logos
# ======================

DEFAULT_LOGOS_DIR = Path(__file__).parent / "logos"
LOGO_FILENAMES_PREFERRED = ["left-side-logo.png", "right-side-main.png"]
LOGO_EXTS = (".png", ".jpg", ".jpeg", ".webp")

PDF_PAGE_SIZE = "A4"
PDF_MARGIN_MM = 12


# ======================
# === OpenAI (LLM-only)
# ======================

def _require_openai():
    load_dotenv()
    key = os.getenv("OPENAI_API_KEY", "").strip()
    if not key:
        raise RuntimeError("OPENAI_API_KEY missing in .env (LLM-only interpretations required).")
    try:
        from openai import OpenAI
    except Exception:
        raise RuntimeError("`openai` python package not found. Install: pip install openai")
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    client = OpenAI(api_key=key, timeout=30.0)
    return client, model

_client, _model = None, None

def _client_model():
    global _client, _model
    if _client is None:
        _client, _model = _require_openai()
    return _client, _model


def _json_maybe_repair(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        s = s.strip("`")
        if s.lower().startswith("json"):
            s = s[4:].lstrip(":").strip()
    return s


def llm_json(system: str, prompt: str, max_tokens: int = 400) -> dict:
    client, model = _client_model()
    msg = [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt + "\n\nReturn ONLY valid JSON."},
    ]
    out = client.chat.completions.create(
        model=model,
        temperature=0.0,
        max_tokens=max_tokens,
        messages=msg,
    )
    text = (out.choices[0].message.content or "").strip()
    text = _json_maybe_repair(text)
    try:
        return json.loads(text)
    except Exception:
        try:
            return ast.literal_eval(text)
        except Exception as e:
            raise RuntimeError(
                f"LLM did not return valid JSON. First 300 chars:\n{text[:300]}\nError: {e}"
            )


def llm_text(system: str, prompt: str, max_tokens: int = 420) -> str:
    client, model = _client_model()
    out = client.chat.completions.create(
        model=model,
        temperature=0.35,
        max_tokens=max_tokens,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": prompt}],
    )
    return (out.choices[0].message.content or "").strip()


# ==============
# === IO utils
# ==============

def read_table(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    try:
        suf = path.suffix.lower()
        if suf in {".tsv", ".txt"}:
            return pd.read_csv(path, sep="\t")
        if suf == ".csv":
            return pd.read_csv(path)
        if suf == ".json":
            with open(path, "r", encoding="utf-8") as f:
                obj = json.load(f)
            return pd.json_normalize(obj) if not isinstance(obj, pd.DataFrame) else obj
    except Exception as e:
        print(f"[WARN] Failed to read {path}: {e}")
    return None


def read_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[WARN] Failed to read JSON {path}: {e}")
        return None


def b64_img(path: Optional[Path]) -> Optional[str]:
    if not path or not path.exists():
        return None
    try:
        ext = path.suffix.lower().strip(".")
        mime = "png" if ext == "png" else "jpeg"
        return "data:image/%s;base64,%s" % (
            mime,
            base64.b64encode(path.read_bytes()).decode("utf-8"),
        )
    except Exception as e:
        print(f"[WARN] Could not b64-encode {path}: {e}")
        return None


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def safe_float(x):
    try:
        return float(x)
    except Exception:
        return None


# ================
# === Text clean
# ================

_MD_PAT = re.compile(r"[*_`#>]+")

def strip_md(s: str) -> str:
    if not s:
        return ""
    return _MD_PAT.sub("", s).replace("  ", " ").strip()


def clean_line(s: str) -> str:
    s = strip_md(s or "")
    return re.sub(r"^[\s\-\*\#]+", "", s).strip()


def clean_list(lines: List[str]) -> List[str]:
    return [clean_line(x) for x in lines if clean_line(x)]


# =========================
# === Cell-type coloring
# =========================

def palette(i: int) -> str:
    hue = (i * 53) % 360
    return f"hsl({hue}, 70%, 80%)"


def build_ct_colors_from_counts(ct_counts: Dict[str, int]) -> Dict[str, str]:
    names = [str(k).strip() for k in ct_counts.keys() if str(k).strip()]
    uniq = []
    seen = set()
    for n in names:
        if n not in seen:
            seen.add(n); uniq.append(n)
    return {name: palette(i) for i, name in enumerate(uniq)}


def colorize_text(text: str, ct_colors: Dict[str, str]) -> str:
    if not text:
        return ""
    s = strip_md(text)
    for name in sorted(ct_colors.keys(), key=len, reverse=True):
        pat = r"\b" + re.escape(name).replace(r"\.", r"[._ ]") + r"\b"
        s = re.sub(
            pat,
            f'<span class="ct-tag" style="background:{ct_colors[name]};">'
            + name.replace(".", " ").replace("_", " ")
            + "</span>",
            s,
            flags=re.IGNORECASE,
        )
    return s


def colorize_list(lines: List[str], ct_colors: Dict[str, str]) -> List[str]:
    return [colorize_text(clean_line(l), ct_colors) for l in lines]


# ==========================
# === GEO metadata helpers
# ==========================

def _find_sample_entry(geo_meta: dict, case_id: str) -> Optional[dict]:
    """
    Try to locate a GSM/sample entry matching case_id in the GEO metadata.
    Supports several possible list keys.
    """
    if not case_id:
        return None
    keys = ["samples", "sample_list", "gsm_list", "sample_metadata"]
    for k in keys:
        arr = geo_meta.get(k)
        if isinstance(arr, list):
            for s in arr:
                acc = str(s.get("accession", "")).strip()
                if acc == str(case_id).strip():
                    return s
    return None


def infer_dataset_context_from_geo(
    meta: dict,
    sample_accession: str = "",
    sample_title: str = "",
) -> Tuple[str, str, str]:
    """
    Use LLM to derive (disease_label, biosample_label, context_text) from GEO JSON,
    explicitly taking into account the chosen sample (case_id) when available.
    """
    es = meta.get("esummary_raw", {})
    title = meta.get("title") or es.get("title", "")
    summary = meta.get("summary") or es.get("summary", "")
    taxon = meta.get("taxon") or es.get("taxon", "")
    gdstype = meta.get("gdstype") or es.get("gdstype", "")

    j = llm_json(
        "You are a clinical and single-cell bioinformatics expert. Return ONLY JSON.",
        (
            "Given GEO metadata (title, summary, taxon, gdstype) and a specific sample, "
            "extract a concise high-level context.\n"
            f"Series title: {title}\n"
            f"Series summary: {summary}\n"
            f"Taxon: {taxon}\n"
            f"gdstype: {gdstype}\n"
            f"Sample accession: {sample_accession or 'NA'}\n"
            f"Sample title: {sample_title or 'NA'}\n\n"
            'Return JSON: {"disease":"...","biosample":"...","context":"2-3 sentences summarizing the disease, '
            'the biological role of this specific sample, and why transcriptomic profiling is informative (no invented numbers)."}'
        ),
        max_tokens=260,
    )
    disease = clean_line(j.get("disease", "") or "NA")
    biosample = clean_line(j.get("biosample", "") or "NA")
    context = strip_md(j.get("context", "") or "")
    return disease, biosample, context


# ===============================
# === Single-cell summary parse
# ===============================

def load_analysis_summary(summary_dir: Path) -> Tuple[str, Dict[str, str], Dict[str, int]]:
    """
    Returns:
      analysis_name (str from header),
      kv (dict of key metrics),
      celltype_counts (dict)
    """
    txt_files = sorted(summary_dir.glob("*_analysis_summary.txt"))
    if not txt_files:
        return "singlecell", {}, {}

    path = txt_files[0]
    text = path.read_text(encoding="utf-8", errors="ignore")

    kv = {}
    ct_counts: Dict[str, int] = {}
    analysis_name = "singlecell"

    # Simple key-value parsing
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("===") and line.endswith("==="):
            inside = line.strip("=").strip()
            analysis_name = inside or analysis_name
        m = re.match(r"([^:]+):\s*(.+)", line)
        if m:
            key = m.group(1).strip()
            val = m.group(2).strip()
            kv[key] = val

    # Celltype counts block
    if "Celltype counts:" in text:
        block = text.split("Celltype counts:")[1]
        for line in block.splitlines()[1:]:
            if not line.strip():
                break
            m2 = re.match(r"^\s*(.+?):\s*([0-9]+)\s+cells", line.strip())
            if m2:
                ct = m2.group(1).strip()
                n = int(m2.group(2))
                ct_counts[ct] = n

    return analysis_name, kv, ct_counts


def top_celltypes_line(ct_counts: Dict[str, int]) -> str:
    if not ct_counts:
        return ""
    total = float(sum(ct_counts.values()))
    rows = sorted(ct_counts.items(), key=lambda x: x[1], reverse=True)
    bits = []
    for name, n in rows[:8]:
        pct = 100.0 * n / total if total > 0 else 0.0
        bits.append(f"{name} ({pct:.1f}% of cells)")
    return ", ".join(bits)


def build_preproc_bullets_from_kv(kv: Dict[str, str],
                                  disease: str,
                                  biosample: str) -> List[str]:
    """
    Build a professional, stepwise preprocessing summary using the metrics in kv.
    Mirrors the 1–7 bullets you described.
    """
    bullets: List[str] = []

    init_cells = kv.get("Initial cells")
    init_genes = kv.get("Initial genes")
    cells_after_qc = kv.get("Cells after QC filters")
    genes_after_min = kv.get("Genes after min_cells filter")
    hvgs = kv.get("HVGs used")
    n_clusters = kv.get("Leiden clusters")

    if init_cells and init_genes:
        bullets.append(
            f"Initial Data: The dataset starts with {init_cells} cells and {init_genes} genes "
            f"derived from transcriptomic profiling of {biosample} in the context of {disease}."
        )
    else:
        bullets.append(
            f"Initial Data: Transcriptomic profiling was performed on {biosample} in {disease}, "
            "capturing thousands of cells and genes for downstream analysis."
        )

    if cells_after_qc:
        bullets.append(
            f"Quality Control (QC): Low-quality cells are filtered out based on total counts, number of detected genes, "
            f"and mitochondrial gene percentage, resulting in {cells_after_qc} high-quality cells retained for analysis."
        )
    else:
        bullets.append(
            "Quality Control (QC): Low-quality cells are removed using thresholds on library size, detected genes, and "
            "mitochondrial RNA content, retaining a high-confidence set of cells."
        )

    if genes_after_min:
        bullets.append(
            f"Gene Filtering: Genes expressed in very few cells are removed using a minimum-cells filter, reducing the "
            f"feature space to {genes_after_min} genes while preserving informative transcripts."
        )
    else:
        bullets.append(
            "Gene Filtering: Lowly expressed genes are filtered out to focus the analysis on robustly detected transcripts."
        )

    bullets.append(
        "Normalization: Raw counts are normalized across cells (library-size scaling followed by log-transformation) "
        "to make gene expression levels comparable across the dataset."
    )

    if hvgs:
        bullets.append(
            f"Highly Variable Genes (HVGs): {hvgs} highly variable genes are selected to capture the dominant biological "
            "signals and reduce technical noise before downstream modeling."
        )
    else:
        bullets.append(
            "Highly Variable Genes (HVGs): The most variable genes are selected to emphasise key biological variation "
            "while reducing technical noise."
        )

    bullets.append(
        "Dimensionality Reduction: Latent embeddings are constructed from the HVG space, followed by non-linear methods "
        "such as t-SNE and UMAP to visualise cell states and gradients in two dimensions."
    )

    if n_clusters:
        bullets.append(
            f"Clustering: A graph-based Leiden clustering algorithm is run on the embedding space, yielding {n_clusters} "
            "transcriptionally distinct cell populations for downstream interpretation."
        )
    else:
        bullets.append(
            "Clustering: A graph-based Leiden clustering algorithm identifies transcriptionally distinct cell populations, "
            "which are then annotated using canonical marker genes and, where relevant, supervised reference models."
        )

    return bullets


# =======================
# === Markers & pathways
# =======================

def prepare_sc_markers_table(markers_path: Path,
                             disease: str,
                             biosample: str,
                             ct_colors: Dict[str, str]
                             ) -> Tuple[Optional[pd.DataFrame], str, List[str], str, Dict[str, List[str]]]:
    """
    Use celltype_marker_genes_celltype_ALL.csv (group=celltype, names=gene).
    Returns:
      display_df,
      topline,
      bullets,
      note,
      top_markers_per_ct (dict celltype -> list[genes])
    """
    if not markers_path.exists():
        return None, "Marker table not found.", [], "", {}

    df = pd.read_csv(markers_path)
    if "group" not in df.columns or "names" not in df.columns:
        return None, "Marker table not found (missing columns).", [], "", {}

    # group markers per celltype
    grouped = (
        df[["group", "names"]]
        .dropna()
        .groupby("group")["names"]
        .apply(lambda s: list(dict.fromkeys([str(x) for x in s])))
        .reset_index(name="markers")
    )

    top_markers_per_ct: Dict[str, List[str]] = {}
    # limit each cell type to top 10 markers for display and LLM
    grouped["markers"] = grouped["markers"].apply(
        lambda lst: [g for g in lst][:10]
    )
    for _, row in grouped.iterrows():
        top_markers_per_ct[str(row["group"])] = list(row["markers"])

    display = grouped.copy()
    display["markers"] = display["markers"].apply(lambda lst: ", ".join(lst))
    display = display.rename(columns={"group": "cell_type"}).head(18)

    topline = "Key marker genes are summarised per cell type."

    j = llm_json(
        "You are a clinical single-cell bioinformatics expert. Return ONLY JSON.",
        (
            f"For disease '{disease}' and biosample '{biosample}', interpret the following cell-type marker summary.\n"
            f"Each row has a cell_type and its top markers.\n"
            f"Markers:\n{display.to_dict(orient='records')}\n\n"
            'Return JSON: {"bullets":["...","...","...","..."],'
            '"note":"one concise clinical note focusing on immune/tumour/stromal composition and activation/exhaustion."}\n'
            "Avoid invented percentages or patient numbers."
        ),
        max_tokens=320,
    )

    bullets = colorize_list(j.get("bullets", [])[:6], ct_colors)
    note = colorize_text(clean_line(j.get("note", "")), ct_colors)

    return display, topline, bullets, note, top_markers_per_ct


def summarize_pathways(pathways_combined_dir: Path,
                       disease: str,
                       biosample: str,
                       ct_colors: Dict[str, str]) -> Tuple[Optional[pd.DataFrame], List[str], str]:
    """
    Summarize top pathways across all clusters (global view).
    Show only top 10 pathways, with max 2 per Biological_Database.
    Do NOT display p-values in the final table (only use them internally for ranking).
    """
    if not pathways_combined_dir.exists():
        return None, ["Pathway enrichment tables not found."], ""

    csvs = sorted(pathways_combined_dir.glob("*combined_pathways_DEDUP.csv"))
    if not csvs:
        csvs = sorted(pathways_combined_dir.glob("*combined_pathways*.csv"))
    if not csvs:
        return None, ["Pathway enrichment tables not found."], ""

    dfs = []
    for f in csvs:
        try:
            df = pd.read_csv(f)
            if not df.empty:
                df["source_file"] = f.name
                dfs.append(df)
        except Exception as e:
            print(f"[WARN] Failed to read pathways {f}: {e}")
    if not dfs:
        return None, ["Pathway enrichment tables not found."], ""

    all_df = pd.concat(dfs, axis=0, ignore_index=True)

    # Rank by adj p or combined score
    if "Adjusted P-value" in all_df.columns:
        all_df["__score"] = all_df["Adjusted P-value"].fillna(1.0)
        all_df = all_df.sort_values("__score", ascending=True)
    elif "Combined Score" in all_df.columns:
        all_df["__score"] = -all_df["Combined Score"].fillna(0.0)
        all_df = all_df.sort_values("__score", ascending=True)

    # Keep only descriptive columns for display (no p-values)
    keep_cols = []
    for c in ["Biological_Database", "Pathways", "Genes", "source_file", "__score"]:
        if c in all_df.columns:
            keep_cols.append(c)
    display = all_df[keep_cols].copy()

    # limit to top 2 per database, then overall top 10
    if "Biological_Database" in display.columns:
        top_rows = []
        for db, sub in display.groupby("Biological_Database", sort=False):
            sub_sorted = sub.sort_values("__score", ascending=True)
            top_rows.append(sub_sorted.head(2))
        display = pd.concat(top_rows, ignore_index=True)
        display = display.sort_values("__score", ascending=True).head(10)
    else:
        display = display.sort_values("__score", ascending=True).head(10)

    # Build compact list representation to feed into LLM (we can still send p-values if present)
    mini_list = []
    for _, r in display.iterrows():
        mini_list.append({
            "database": r.get("Biological_Database", ""),
            "pathway": r.get("Pathways", ""),
        })

    j = llm_json(
        "You are a clinical single-cell and immuno-oncology expert. Return ONLY JSON.",
        (
            f"Summarize how the top enriched pathways across clusters relate to disease '{disease}' "
            f"and biosample '{biosample}' in a single-cell immunotherapy context.\n"
            f"Top pathways list (each with database and pathway name):\n{mini_list}\n\n"
            'Return JSON: {"bullets":["...","...","..."],'
            '"note":"one concise clinical note about how these pathways might guide interpretation or future work."}\n'
            "Avoid fabricating exact effect sizes or patient outcomes."
        ),
        max_tokens=320,
    )

    bullets = colorize_list(j.get("bullets", [])[:5], ct_colors)
    note = colorize_text(clean_line(j.get("note", "")), ct_colors)

    display = display.drop(columns=[c for c in display.columns if c == "__score"], errors="ignore")
    return display, bullets, note


def _subset_genes_string(genes_str: str, max_genes: int = 8) -> str:
    """
    Take a long 'Genes' string and keep only first N genes as a subset.
    """
    if not isinstance(genes_str, str):
        return ""
    # split on commas, semicolons, or whitespace slashes
    tokens = re.split(r"[,\s;/]+", genes_str.strip())
    tokens = [t for t in tokens if t]
    return ", ".join(tokens[:max_genes])


def summarize_celltype_pathways(
    pathways_combined_dir: Path,
    top_markers_per_ct: Dict[str, List[str]],
    disease: str,
    biosample: str,
    ct_colors: Dict[str, str],
) -> List[Dict[str, Any]]:
    """
    Per-cluster/cell-type pathway summary.
    For each *combined_pathways_DEDUP.csv, extract top 3 pathways
    and link them to that cell type and its markers.
    Shows a subset of genes per pathway. p-values are used internally
    but not displayed in the final report.
    """
    items: List[Dict[str, Any]] = []
    if not pathways_combined_dir.exists():
        return items

    csvs = sorted(pathways_combined_dir.glob("*combined_pathways_DEDUP.csv"))
    if not csvs:
        csvs = sorted(pathways_combined_dir.glob("*combined_pathways*.csv"))
    if not csvs:
        return items

    for f in csvs:
        try:
            df = pd.read_csv(f)
        except Exception as e:
            print(f"[WARN] Failed to read {f}: {e}")
            continue
        if df.empty or "Pathways" not in df.columns:
            continue

        stem = f.stem  # e.g. single_dataset_cluster_0_Tcell_combined_pathways_DEDUP
        label = (
            stem.replace("single_dataset_cluster_", "")
                .replace("_combined_pathways_DEDUP", "")
                .replace("_combined_pathways_RAW", "")
        )
        pretty_label = label.replace("_", " ")

        # choose best score
        if "Adjusted P-value" in df.columns:
            df = df.sort_values("Adjusted P-value", ascending=True)
        elif "Combined Score" in df.columns:
            df = df.sort_values("Combined Score", ascending=False)

        top_df = df.head(3).copy()
        top_list = []
        for _, r in top_df.iterrows():
            genes_sub = _subset_genes_string(str(r.get("Genes", "")), max_genes=8)
            if "Adjusted P-value" in r and pd.notnull(r["Adjusted P-value"]):
                try:
                    adj_val = float(r["Adjusted P-value"])
                    adj_str = f"{adj_val:.2e}"  # still used for LLM context only
                except Exception:
                    adj_str = ""
            else:
                adj_str = ""
            top_list.append(
                {
                    "db": str(r.get("Biological_Database", "")),
                    "name": str(r.get("Pathways", "")),
                    "adj_p": adj_str,
                    "genes": genes_sub,
                }
            )

        # match markers
        ct_guess = label.split("_", 1)[-1]
        markers = (
            top_markers_per_ct.get(ct_guess)
            or top_markers_per_ct.get(pretty_label)
            or []
        )

        j = llm_json(
            "You are a clinical single-cell bioinformatics expert. Return ONLY JSON.",
            textwrap.dedent(
                f"""
                You are interpreting cluster/cell-type–specific pathway enrichment results
                from a single-cell RNA-seq study.

                Disease: {disease}
                Biosample / tissue context: {biosample}
                Cell type / cluster label: "{pretty_label}"

                Top enriched pathways (database, name, adj_p, subset of genes):
                {top_list}

                Known marker genes for this cell type:
                {markers}

                Task:
                  • Provide 3–5 bullets explaining what these pathways and markers say
                    about this cell population (e.g. cytotoxic T cells, exhausted T cells,
                    suppressive myeloid cells, fibroblasts, endothelial cells, etc.).
                  • Highlight immune activation or suppression, stromal/angiogenic programs,
                    and how this may modulate immunotherapy response or resistance.
                  • Do NOT invent any numeric percentages or hazard ratios.

                Return JSON:
                {{
                  "bullets": ["bullet1","bullet2","bullet3","bullet4","bullet5"],
                  "note": "single concise clinical note linking this cell-type program to therapy response/resistance."
                }}
                """
            ),
            max_tokens=420,
        )

        bullets = colorize_list(j.get("bullets", [])[:5], ct_colors)
        note = colorize_text(clean_line(j.get("note", "")), ct_colors)

        items.append(
            {
                "cell_label": label,
                "pretty_label": pretty_label,
                "top_pathways": top_list,
                "bullets": bullets,
                "note": note,
            }
        )

    return items


# ==========================
# === HTML Template
# ==========================

HTML = Template(r"""
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/><meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Single-Cell Transcriptomic Report — {{ case_id }}</title>
<style>
:root{
  --bg:#f7f8ff;--card:#ffffff;--ink:#0b1020;--muted:#4b5a8a;--edge:#e3e7ff;
  --accent:#6c5ce7;--glow1:#6c8cff;--glow2:#b06afc;--radius:16px;--maxw:1400px
}
html,body{
  margin:0;padding:0;background:var(--bg);color:var(--ink);
  font-family:Inter,system-ui,-apple-system,Segoe UI,Roboto,sans-serif;
  line-height:1.55;
}
.container{max-width:var(--maxw);margin:0 auto;padding:22px 16px}
header{padding:18px 16px 12px;background:#fff;border-bottom:1px solid var(--edge)}
.header-inner{max-width:var(--maxw);margin:0 auto}
.brand{font-weight:800;letter-spacing:.3px;margin-bottom:2px}
h1{margin:4px 0 6px;font-size:24px;letter-spacing:.2px}
.muted{color:var(--muted);font-size:13px}
.pill{
  display:inline-block;padding:4px 8px;border-radius:999px;background:#eef1ff;
  border:1px solid var(--edge);color:#37417a;margin:0 4px;font-size:12px
}
.card{
  background:var(--card);border:1px solid var(--edge);border-radius:var(--radius);
  padding:18px;margin:0 0 16px;box-shadow:0 8px 22px rgba(16,23,51,.10);
  page-break-inside:avoid;break-inside:avoid
}
.page-break-before{
  page-break-before: always;
  break-before: page;
}
h2{
  color:var(--accent);margin:0 0 10px;font-size:18px;letter-spacing:.2px;
  position:relative;page-break-after:avoid;break-after:avoid
}
h2::after{
  content:"";position:absolute;left:0;bottom:-6px;width:48px;height:2px;
  background:linear-gradient(90deg,var(--glow1),var(--glow2));border-radius:2px;opacity:.65
}
.table-wrap{
  overflow:auto;border:1px solid var(--edge);border-radius:12px;
  page-break-inside:avoid;break-inside:avoid
}
table{width:100%;border-collapse:collapse;background:#fff}
th,td{
  border-bottom:1px dashed #dfe5ff;padding:8px 10px;font-size:13px;vertical-align:top
}
th{text-align:left;color:#3a4688;background:#f8f9ff;position:sticky;top:0}
.img-frame{
  border:1px solid var(--edge);border-radius:12px;overflow:hidden;background:#fff;
  page-break-inside:avoid;break-inside:avoid
}
.img-frame img{
  width:100%;height:auto;display:block;image-rendering:auto;
  page-break-inside:avoid;break-inside:avoid
}
.caption{color:#5b69a8;font-size:12px;margin-top:6px}
.panel{
  background:#f3f5ff;border:1px solid #dbe2ff;border-radius:12px;
  padding:12px;margin-top:10px;page-break-inside:avoid;break-inside:avoid
}
.panel h3{margin:0 0 6px;font-size:15px;color:#4957a8}
ul{margin:6px 0 0 18px}
.ct-tag{
  display:inline-block;padding:2px 8px;border-radius:999px;color:#0b1020;
  border:1px solid rgba(0,0,0,.06);margin:0 2px
}

/* two col */
.two-col{display:grid;grid-template-columns:1.05fr 1.35fr;gap:14px}
@media (max-width:900px){.two-col{grid-template-columns:1fr}}

/* Logo header */
.header-flex{
  display:flex;align-items:center;justify-content:space-between;gap:12px;
  page-break-inside:avoid;break-inside:avoid
}
.header-center{flex:1 1 auto;text-align:center;min-width:0}
.header-logo{
  height:50px;max-width:220px;object-fit:contain;flex:0 0 auto;
  page-break-inside:avoid;break-inside:avoid
}
.header-center .muted{
  white-space:nowrap;overflow:hidden;text-overflow:ellipsis
}

/* Print tweaks */
@media print{
  html,body{-webkit-print-color-adjust:exact;print-color-adjust:exact}
  .card{box-shadow:none;border:1px solid #dfe5ff}
  a[href]:after{content:""!important}
  .two-col{display:block;grid-template-columns:none}
  .two-col>div{page-break-inside:avoid;break-inside:avoid}
  .img-frame,.panel,.table-wrap,.figure-block,.gallery-card{
    page-break-inside:avoid;break-inside:avoid
  }
  h2,.panel h3,.caption{page-break-after:avoid;break-after:avoid}
  img{page-break-inside:avoid;break-inside:avoid}
}

/* Page size for headless browser */
@page{
  size: {{ page_size }};
  margin: {{ pdf_margin }}mm;
}
</style>
</head>
<body>
<header>
  <div class="header-inner">
    <div class="header-flex">
      {% if left_logo %}
        <img class="header-logo" src="{{ left_logo }}" alt="Left Logo"/>
      {% else %}
        <div style="width:50px"></div>
      {% endif %}
      <div class="header-center">
        <div class="brand">Ayass Bioscience</div>
        <h1>Single-Cell Transcriptomic Profiling Report</h1>
        <div class="muted">
          Case: <strong>{{ case_id }}</strong>
          · GEO: <strong>{{ accession }}</strong>
          {% if sample_accession %}
            · Sample: <strong>{{ sample_accession }}</strong>{% if sample_title %} — {{ sample_title }}{% endif %}
          {% endif %}
        </div>
        <div class="muted" style="margin-top:4px;">
          Disease: <strong>{{ disease }}</strong>
          {% if n_cells %} · Cells (post-QC): <strong>{{ n_cells }}</strong>{% endif %}
          {% if n_clusters %} · Clusters: <strong>{{ n_clusters }}</strong>{% endif %}
          {% if n_celltypes %} · Cell types: <strong>{{ n_celltypes }}</strong>{% endif %}
        </div>
      </div>
      {% if right_logo %}
        <img class="header-logo" src="{{ right_logo }}" alt="Right Logo"/>
      {% else %}
        <div style="width:50px"></div>
      {% endif %}
    </div>
  </div>
</header>

<div class="container">

  <div class="card">
    <h2>Dataset & Clinical Context</h2>
    <p><strong>Series title:</strong> {{ title }}</p>
    {% if sample_accession %}
      <p><strong>Sample:</strong> {{ sample_accession }}{% if sample_title %} — {{ sample_title }}{% endif %}</p>
    {% endif %}
    <p><strong>GEO Accession:</strong> {{ accession }} &nbsp; · &nbsp;
       <strong>Taxon:</strong> {{ taxon }} &nbsp; · &nbsp;
       <strong>Type:</strong> {{ gdstype }}</p>
    <p>{{ dataset_context|safe }}</p>
  </div>

  <div class="card page-break-before">
    <h2>Preprocessing & Quality Control Pipeline</h2>
    <ul>
      {% for b in preproc_bullets %}
        <li>{{ b }}</li>
      {% endfor %}
    </ul>
    {% if qc_cells_line %}
      <p class="caption"><strong>Key QC metrics:</strong> {{ qc_cells_line }}</p>
    {% endif %}
    {% if qc_gallery %}
      <div class="two-col" style="margin-top:10px;">
        {% for fig in qc_gallery %}
          <div class="img-frame figure-block" style="margin-bottom:10px;">
            <img src="{{ fig.src }}" alt="{{ fig.caption }}"/>
            <div class="caption"><strong>{{ fig.caption }}</strong></div>
          </div>
        {% endfor %}
      </div>
    {% endif %}
  </div>

  <div class="card page-break-before">
    <h2>Embeddings (t-SNE, UMAP) & Clustering</h2>
    {% if embedding_bullets and embedding_bullets|length > 0 %}
      <ul>
        {% for b in embedding_bullets %}
          <li>{{ b|safe }}</li>
        {% endfor %}
      </ul>
    {% endif %}
    {% if embed_gallery %}
      <div class="two-col">
        {% for fig in embed_gallery %}
          <div class="img-frame figure-block" style="margin-bottom:10px;">
            <img src="{{ fig.src }}" alt="{{ fig.caption }}"/>
            <div class="caption"><strong>{{ fig.caption }}</strong></div>
          </div>
        {% endfor %}
      </div>
    {% endif %}
  </div>

  <div class="card page-break-before">
    <h2>Cell-Type Landscape</h2>
    <p><strong>Dominant cell types:</strong> {{ top_celltypes_html|safe }}</p>
    {% if celltype_barplot %}
      <div class="img-frame figure-block">
        <img src="{{ celltype_barplot }}" alt="Cell-type composition barplot"/>
      </div>
    {% endif %}
    {% if celltype_umap %}
      <div class="img-frame figure-block" style="margin-top:10px;">
        <img src="{{ celltype_umap }}" alt="UMAP colored by cell type"/>
      </div>
    {% endif %}
    {% if celltype_bullets and celltype_bullets|length > 0 %}
      <div class="panel">
        <h3>Cell-type interpretation</h3>
        <ul>
          {% for b in celltype_bullets %}
            <li>{{ b|safe }}</li>
          {% endfor %}
        </ul>
        <p class="caption"><strong>Clinical note:</strong> {{ celltype_note|safe }}</p>
      </div>
    {% endif %}
  </div>

  <div class="card page-break-before">
    <h2>Marker Genes by Cell Type</h2>
    {% if markers_topline_items_html and markers_topline_items_html|length > 0 %}
      <ul>
        {% for line in markers_topline_items_html %}
          <li>{{ line|safe }}</li>
        {% endfor %}
      </ul>
    {% endif %}
    {% if markers_table %}
      <div class="table-wrap">{{ markers_table|safe }}</div>
    {% endif %}
    <div class="panel">
      <h3>Marker-based biological interpretation</h3>
      <ul>
        {% for b in markers_bullets_html %}
          <li>{{ b|safe }}</li>
        {% endfor %}
      </ul>
      <p class="caption"><strong>Clinical note:</strong> {{ markers_note_html|safe }}</p>
    </div>
  </div>

  {% if pathways_table %}
  <div class="card page-break-before">
    <h2>Global Pathway Programs & Cell-State Shifts</h2>
    <p class="caption">
      Top enriched pathways across all clusters (maximum 2 pathways per database), highlighting dominant signalling
      programs active in the tumour microenvironment. p-values are used for ranking but not shown here.
    </p>
    <div class="table-wrap">{{ pathways_table|safe }}</div>
    <div class="panel">
      <h3>Pathway interpretation (global view)</h3>
      <ul>
        {% for b in pathways_bullets_html %}
          <li>{{ b|safe }}</li>
        {% endfor %}
      </ul>
      <p class="caption"><strong>Clinical note:</strong> {{ pathways_note_html|safe }}</p>
    </div>
  </div>
  {% endif %}

  {% if celltype_pathways and celltype_pathways|length > 0 %}
  <div class="card page-break-before">
    <h2>Cell-Type–Specific Pathway Programs</h2>
    <p>
      For each major cell population, pathway enrichment of its defining marker genes highlights functional programs
      such as cytotoxicity, exhaustion, immunosuppression, stromal activation and angiogenesis within the tumour
      microenvironment. Only the top pathways per population are shown.
    </p>
  </div>

  {% for ct in celltype_pathways %}
  <div class="card">
    <h2>{{ ct.pretty_label }}</h2>
    <div class="table-wrap">
      <table>
        <tr>
          <th>Biological Database</th>
          <th>Pathway</th>
          <th>Genes (subset)</th>
        </tr>
        {% for p in ct.top_pathways %}
        <tr>
          <td>{{ p.db }}</td>
          <td>{{ p.name }}</td>
          <td>{{ p.genes }}</td>
        </tr>
        {% endfor %}
      </table>
    </div>
    <div class="panel">
      <h3>Functional interpretation</h3>
      <ul>
        {% for b in ct.bullets %}
          <li>{{ b|safe }}</li>
        {% endfor %}
      </ul>
      <p class="caption"><strong>Clinical note:</strong> {{ ct.note|safe }}</p>
    </div>
  </div>
  {% endfor %}
  {% endif %}

  <div class="card page-break-before">
    <h2>Key Takeaways & Clinical Conclusion</h2>
    {% if key_takeaways_html and key_takeaways_html|length > 0 %}
    <ul>
      {% for b in key_takeaways_html %}
        <li>{{ b|safe }}</li>
      {% endfor %}
    </ul>
    {% endif %}
    <p style="margin-top:10px;">
      {{ clinical_conclusion_html|safe }}
    </p>
  </div>

</div>

</body>
</html>
""")


# ======================================
# === HTML → PDF via headless browser
# ======================================

from urllib.parse import urljoin, quote

def _file_url(path: Path) -> str:
    return urljoin("file:", quote(str(path.resolve()).replace("\\", "/")))


def _find_browser_exe() -> Optional[str]:
    candidates = []
    if sys.platform.startswith("win"):
        candidates += [
            r"C:\Program Files\Google\Chrome\Application\chrome.exe",
            r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
            r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
            r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        ]
    elif sys.platform == "darwin":
        candidates += [
            "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
            "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
            shutil.which("chrome"),
            shutil.which("google-chrome"),
            shutil.which("msedge"),
            shutil.which("chromium"),
        ]
    else:
        candidates += [
            shutil.which("google-chrome"),
            shutil.which("chrome"),
            shutil.which("chromium"),
            shutil.which("chromium-browser"),
            shutil.which("msedge"),
        ]
    for c in candidates:
        if c and os.path.exists(c):
            return c
    return None


def _timestamped_pdf_path(outdir: Path, case_id: str) -> Path:
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M")
    pdf_name = f"Single-Cell Scanpy — Ayass Bioscience Report — {case_id} — {ts}.pdf"
    return outdir / pdf_name


def write_pdf_via_browser(html_path: Path, pdf_path: Path) -> bool:
    """
    Generate PDF using Chrome/Edge headless browser.
    Tries Playwright first (supports disabling headers/footers), then falls back to Chrome CDP.
    """
    file_url = _file_url(html_path)
    
    # Try Playwright first (most reliable way to disable headers/footers)
    try:
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            browser_instance = p.chromium.launch(headless=True)
            page = browser_instance.new_page()
            page.goto(file_url, wait_until="networkidle", timeout=30000)
            page.pdf(
                path=str(pdf_path),
                format="A4",
                margin={"top": "12mm", "bottom": "12mm", "left": "12mm", "right": "12mm"},
                print_background=True,
                display_header_footer=False,
            )
            browser_instance.close()
            if pdf_path.exists():
                return True
    except ImportError:
        print("[INFO] Playwright not installed. Install with: pip install playwright && playwright install chromium")
        print("[INFO] Falling back to Chrome print-to-pdf (may include headers/footers)")
    except Exception as e:
        print(f"[WARN] Playwright PDF export failed: {e}")
        print("[INFO] Falling back to Chrome print-to-pdf (may include headers/footers)")
    
    # Fallback: Use Chrome's basic print-to-pdf (will include headers/footers)
    browser = _find_browser_exe()
    if not browser:
        print("[WARN] No Chrome/Edge found. Install one for PDF export.")
        return False
    
    cmd = [
        browser,
        "--headless=new",
        "--disable-gpu",
        "--disable-print-preview",
        f"--print-to-pdf={str(pdf_path)}",
        file_url,
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60)
        if pdf_path.exists():
            print("[INFO] PDF generated. Note: Browser headers/footers may appear.")
            print("[INFO] To remove headers/footers, install Playwright: pip install playwright && playwright install chromium")
            return True
    except Exception as e:
        print(f"[WARN] Browser PDF export failed: {e}")
    
    return False


# =====================
# === Logo helpers
# =====================

def _find_logo_files(dirpath: Path) -> Tuple[Optional[Path], Optional[Path]]:
    if not dirpath or not dirpath.exists():
        return None, None
    left = dirpath / LOGO_FILENAMES_PREFERRED[0]
    right = dirpath / LOGO_FILENAMES_PREFERRED[1]
    if left.exists() and right.exists():
        return left, right

    imgs = [p for p in dirpath.iterdir() if p.suffix.lower() in LOGO_EXTS and p.is_file()]

    left_guess = None
    right_guess = None
    for p in imgs:
        n = p.name.lower()
        if ("left" in n or "ayass" in n) and left_guess is None:
            left_guess = p
        if ("right" in n or "logo" in n) and right_guess is None:
            right_guess = p
    if not left_guess and imgs:
        imgs_sorted = sorted(imgs, key=lambda x: x.name.lower())
        left_guess = imgs_sorted[0]
        if len(imgs_sorted) > 1:
            right_guess = imgs_sorted[1]
    return left_guess, right_guess


def load_logos(logos_dir: Optional[Path]) -> Tuple[Optional[str], Optional[str]]:
    if not logos_dir or not logos_dir.exists():
        logos_dir = DEFAULT_LOGOS_DIR
    l_path, r_path = _find_logo_files(logos_dir)
    left = b64_img(l_path) if l_path else None
    right = b64_img(r_path) if r_path else None
    return left, right


# ============================
# === Figure discovery helper
# ============================

def first_match(dirpath: Path, patterns: List[str]) -> Optional[Path]:
    if not dirpath.exists():
        return None
    for pat in patterns:
        hits = sorted(dirpath.glob(pat))
        if hits:
            return hits[0]
    return None


# ==================
# === Builder main
# ==================

def build_singlecell_report(
    sc_root: Path,
    geo_json_path: Optional[Path] = None,
    case_id: str = "",
    logos_dir: Optional[Path] = None,
):
    """
    sc_root: SC_RESULTS directory from your single-cell pipeline.
    geo_json_path: path to GSE*_metadata.json (or similar). Optional - if None, report will be generated without GEO metadata.
    case_id: the sample/case identifier used in the report (e.g. GSM6360688).
    """
    outdir = sc_root / "singlecell_report"
    ensure_dir(outdir)

    # ---- GEO metadata ----
    geo_meta = {}
    if geo_json_path is not None:
        geo_meta = read_json(geo_json_path) or {}
    es = geo_meta.get("esummary_raw", {})
    accession = geo_meta.get("accession") or es.get("accession", "")
    title = geo_meta.get("title") or es.get("title", "")
    taxon = geo_meta.get("taxon") or es.get("taxon", "")
    gdstype = geo_meta.get("gdstype") or es.get("gdstype", "")

    # Sample-level info from GEO JSON based on case_id
    sample_entry = _find_sample_entry(geo_meta, case_id)
    sample_accession = ""
    sample_title = ""
    if sample_entry:
        sample_accession = str(sample_entry.get("accession", "")).strip()
        sample_title = str(sample_entry.get("title", "")).strip()

    disease, biosample, dataset_context = infer_dataset_context_from_geo(
        geo_meta,
        sample_accession=sample_accession or case_id,
        sample_title=sample_title,
    )

    # ---- Scanpy summary ----
    summary_dir = sc_root / "00_analysis_summary"
    qc_dir      = sc_root / "01_qc_and_filtering"
    dim_dir     = sc_root / "03_dimensionality_reduction_and_embeddings"
    clust_dir   = sc_root / "04_clustering_and_cell_states"
    ct_anno_dir = sc_root / "05_celltype_analysis" / "celltype_annotation"
    ct_markers_dir = sc_root / "05_celltype_analysis" / "celltype_specific_markers"
    pathways_combined_dir = sc_root / "07_pathway_enrichment" / "cluster_marker_enrichment" / "pathways" / "combined"

    analysis_name, kv, ct_counts = load_analysis_summary(summary_dir)

    # Parse key metrics
    n_cells = 0
    n_clusters = 0
    if "Cells after QC filters" in kv:
        try:
            n_cells = int(kv["Cells after QC filters"])
        except Exception:
            pass
    if "Leiden clusters" in kv:
        try:
            n_clusters = int(kv["Leiden clusters"])
        except Exception:
            pass
    n_celltypes = len(ct_counts) if ct_counts else 0

    # Preprocessing bullets (1–7 style)
    preproc_bullets = build_preproc_bullets_from_kv(kv, disease=disease, biosample=biosample)

    # Short QC metrics line
    qc_bits = []
    for key_label in [
        ("Initial cells", "initial cells"),
        ("Initial genes", "initial genes"),
        ("Genes after min_cells filter", "genes after min_cells filter"),
        ("Cells after QC filters", "cells after QC filters"),
        ("HVGs used", "highly variable genes"),
    ]:
        k, label = key_label
        if k in kv:
            qc_bits.append(f"{label}: {kv[k]}")
    qc_cells_line = " · ".join(qc_bits)

    # ---- Cell-type colors & top line ----
    ct_colors = build_ct_colors_from_counts(ct_counts)
    top_ct_line = top_celltypes_line(ct_counts)
    top_celltypes_html = colorize_text(top_ct_line, ct_colors)

    # ---- QC figures (violin + hist) ----
    qc_violin = first_match(qc_dir, ["*qc_violin*.png", "violin*qc*.png", "*qc_violin*.pdf"])
    qc_hist   = first_match(qc_dir, ["*qc_metric_histograms*.png", "*histograms*.png"])
    qc_gallery = []
    if qc_violin:
        qc_gallery.append({"src": b64_img(qc_violin), "caption": "QC violin — genes, counts, mitochondrial %"})
    if qc_hist:
        qc_gallery.append({"src": b64_img(qc_hist), "caption": "QC metric histograms"})

    # ---- Embeddings & clustering figures ----
    pca_plot   = first_match(dim_dir, ["*PCA*variance*.png", "*PCA*.png", "*pc_variance*.png"])
    umap_leiden = first_match(clust_dir, ["umap*leiden*.png", "*UMAP_leiden*.png"])
    tsne_leiden = first_match(clust_dir, ["tsne*leiden*.png", "*TSNE_leiden*.png"])
    umap_samples = first_match(dim_dir, ["umap*UMAP_samples_groups*.png", "umap*group*.png", "*UMAP_samples*.png"])

    embed_gallery = []
    if pca_plot:
        embed_gallery.append({"src": b64_img(pca_plot), "caption": "PCA embedding of major cell populations"})
    if umap_samples:
        embed_gallery.append({"src": b64_img(umap_samples), "caption": "UMAP coloured by sample/group"})
    if umap_leiden:
        embed_gallery.append({"src": b64_img(umap_leiden), "caption": "UMAP coloured by Leiden clusters"})
    if tsne_leiden:
        embed_gallery.append({"src": b64_img(tsne_leiden), "caption": "t-SNE coloured by Leiden clusters"})

    # LLM: embeddings & clustering summary as bullet points
    embedding_j = llm_json(
        "You are a clinical single-cell bioinformatics expert. Return ONLY JSON.",
        (
            f"Describe, in 3–5 bullet points, how t-SNE and UMAP embeddings together with Leiden clustering "
            f"resolve discrete and transitional cell states in this '{disease}' / '{biosample}' dataset. "
            "Highlight separation of immune, stromal and tumour-like populations and potential gradients of "
            "activation or exhaustion. Do NOT discuss PCA variance structure or exact numerical percentages.\n\n"
            'Return JSON: {"bullets":["...","...","..."]}'
        ),
        max_tokens=320,
    )
    embedding_bullets_raw = clean_list(embedding_j.get("bullets", [])[:5])
    embedding_bullets = colorize_list(embedding_bullets_raw, ct_colors)

    # ---- Cell-type figures ----
    celltype_barplot = first_match(ct_anno_dir, ["*celltype_composition_barplot*.png", "*celltype*barplot*.png"])
    celltype_umap    = first_match(ct_anno_dir, ["umap*celltype*.png", "*UMAP_celltypes*.png"])

    # LLM: cell-type landscape summary
    celltype_summary_j = llm_json(
        "You are a clinical single-cell bioinformatics expert. Return ONLY JSON.",
        (
            f"Summarize the cell-type landscape for disease '{disease}' and biosample '{biosample}'. "
            f"Dominant cell types string: {top_ct_line or 'NA'}.\n"
            'Return JSON: {"bullets":["...","...","..."],'
            '"note":"one sentence linking the overall cell-type pattern to the tumour microenvironment and therapy context."}\n'
            "Avoid fabricated fractions; you may refer qualitatively to dominant cell types and infiltration/activation."
        ),
        max_tokens=260,
    )
    celltype_bullets = colorize_list(celltype_summary_j.get("bullets", [])[:4], ct_colors)
    celltype_note = colorize_text(clean_line(celltype_summary_j.get("note", "")), ct_colors)

    # ---- Markers ----
    markers_all_path = ct_markers_dir / "celltype_marker_genes_celltype_ALL.csv"
    markers_display, markers_topline, markers_bullets, markers_note, top_markers_per_ct = prepare_sc_markers_table(
        markers_all_path,
        disease=disease,
        biosample=biosample,
        ct_colors=ct_colors,
    )
    markers_table_html = (
        markers_display.to_html(index=False, border=0)
        if markers_display is not None
        else None
    )

    # Build bullet-style per-cell-type marker lines for the top section
    markers_topline_items_html: List[str] = []
    if markers_display is not None and not markers_display.empty:
        marker_lines = [
            f"{row['cell_type']}: {row['markers']}"
            for _, row in markers_display.iterrows()
        ]
        markers_topline_items_html = colorize_list(marker_lines, ct_colors)

    markers_bullets_html = colorize_list(markers_bullets, ct_colors)
    markers_note_html = colorize_text(markers_note, ct_colors)

    # ---- Pathways: global summary ----
    pathways_table = None
    pathways_bullets_html: List[str] = []
    pathways_note_html = ""
    pw_display, pw_bullets, pw_note = summarize_pathways(
        pathways_combined_dir,
        disease=disease,
        biosample=biosample,
        ct_colors=ct_colors,
    )

    if pw_display is not None:
        pathways_table = pw_display.to_html(index=False, border=0)
        pathways_bullets_html = colorize_list(pw_bullets, ct_colors)
        pathways_note_html = colorize_text(pw_note, ct_colors)

    # ---- Pathways: per cell-type/cluster ----
    celltype_pathways = summarize_celltype_pathways(
        pathways_combined_dir=pathways_combined_dir,
        top_markers_per_ct=top_markers_per_ct,
        disease=disease,
        biosample=biosample,
        ct_colors=ct_colors,
    )

    # ---- Global key takeaways & clinical conclusion ----
    ct_names = sorted(top_markers_per_ct.keys())
    ct_prog_text = "; ".join(
        f"{ct['pretty_label']}: " + ", ".join(p["name"] for p in ct["top_pathways"])
        for ct in celltype_pathways
    )[:2500]

    summary_j = llm_json(
        "You are a clinical single-cell translational oncology expert. Return ONLY JSON.",
        textwrap.dedent(
            f"""
            You are summarising a Scanpy-based single-cell RNA-seq analysis.

            Disease: {disease}
            Biosample: {biosample}
            GEO title: {title}
            Sample accession: {sample_accession or case_id}
            Sample title: {sample_title or 'NA'}

            Annotated cell types (from markers): {ct_names}
            Cell-type–specific pathway programs (name → pathways):
            {ct_prog_text}

            Write:
              1) 4–7 high-level bullet points capturing:
                 • which cell types are most prominent or biologically important,
                 • which pathway programs (e.g. IFN signalling, cytotoxicity, exhaustion, angiogenesis, TGF-β, ECM remodelling)
                   are enriched in which cell populations,
                 • how these patterns could relate to response vs resistance to immunotherapy or combination regimens.
              2) One concise paragraph as a clinical-style conclusion synthesising cell types, pathways and markers.

            Do NOT invent numeric effect sizes, p-values, or patient counts.

            Return JSON:
            {{
              "bullets": ["...","...","..."],
              "conclusion": "final clinical-style conclusion."
            }}
            """
        ),
        max_tokens=520,
    )

    key_takeaways = clean_list(summary_j.get("bullets", [])[:8])
    key_takeaways_html = colorize_list(key_takeaways, ct_colors)
    clinical_conclusion_raw = strip_md(summary_j.get("conclusion", "") or "")
    clinical_conclusion_html = colorize_text(clinical_conclusion_raw, ct_colors)

    # ---- Logos ----
    left_logo, right_logo = load_logos(logos_dir)

    # ---- Render HTML ----
    html = HTML.render(
        case_id=case_id,
        accession=accession,
        title=title,
        taxon=taxon,
        gdstype=gdstype,
        disease=disease,
        biosample=biosample,
        dataset_context=dataset_context,
        n_cells=n_cells,
        n_clusters=n_clusters,
        n_celltypes=n_celltypes,
        preproc_bullets=preproc_bullets,
        qc_cells_line=qc_cells_line,
        qc_gallery=qc_gallery,
        embed_gallery=embed_gallery,
        embedding_bullets=embedding_bullets,
        celltype_barplot=b64_img(celltype_barplot) if celltype_barplot else None,
        celltype_umap=b64_img(celltype_umap) if celltype_umap else None,
        top_celltypes_html=top_celltypes_html,
        celltype_bullets=celltype_bullets,
        celltype_note=celltype_note,
        markers_topline_items_html=markers_topline_items_html,
        markers_table=markers_table_html,
        markers_bullets_html=markers_bullets_html,
        markers_note_html=markers_note_html,
        pathways_table=pathways_table,
        pathways_bullets_html=pathways_bullets_html,
        pathways_note_html=pathways_note_html,
        celltype_pathways=celltype_pathways,
        key_takeaways_html=key_takeaways_html,
        clinical_conclusion_html=clinical_conclusion_html,
        left_logo=left_logo,
        right_logo=right_logo,
        page_size=PDF_PAGE_SIZE,
        pdf_margin=PDF_MARGIN_MM,
        sample_accession=sample_accession or case_id,
        sample_title=sample_title,
    )

    out_html = outdir / "index.html"
    out_html.write_text(html, encoding="utf-8")
    print(f"[INFO] Single-cell report HTML written to: {out_html}")

    pdf_path = _timestamped_pdf_path(outdir, case_id)
    if write_pdf_via_browser(out_html, pdf_path):
        print(f"[INFO] PDF written to: {pdf_path}")
    else:
        print("[WARN] Could not create PDF. Ensure Chrome/Edge is installed and accessible.")


# ========= CLI =========

def parse_args():
    p = argparse.ArgumentParser(
        description="Generate Ayass Bioscience LLM-interpreted single-cell Scanpy report (HTML + PDF)."
    )
    p.add_argument(
        "--root",
        required=True,
        help="Path to SC_RESULTS folder (output of single-cell Scanpy pipeline).",
    )
    p.add_argument(
        "--geo-json",
        required=True,
        help="Path to GEO metadata JSON (e.g. GSE233203_metadata.json).",
    )
    p.add_argument(
        "--case-id",
        default=None,
        help="Case/sample label (e.g. GSM ID); also used to select sample context from GEO JSON when available.",
    )
    p.add_argument(
        "--logos-dir",
        default=None,
        help="Optional directory with logos (defaults to Ayass logo folder).",
    )
    return p.parse_args()


if __name__ == "__main__":

    build_singlecell_report(
        sc_root=Path(r"C:\Ayass Bio Work\Agentic_AI_ABS\single_cell_pipeline\sc_test_run\GSM6360688_ADC\SC_RESULTS"),
        # geo_json_path=Path(r"C:\Ayass Bio Work\Agentic_AI_ABS\single_cell_pipeline\sc_test_run\GSM6360688_ADC\GSE274588_metadata.json"),
        case_id="GSM6360688",
    )
#python singlecell_sc_report_generation.py --root "D:\AyassBio_Workspace_Downloads\SCANPY-SINGLECELL_PIPELINE\Cervical_cancer_sc_test\SINGLE_CELL_10X\singlecell_pipeline\sc_test_run\GSM6360688_ADC\SC_RESULTS" --geo-json "D:\AyassBio_Workspace_Downloads\cohort_single_cell\Cervical_sc\GSE208653\GSE208653_metadata.json" --case-id "GSM6360688" --logos-dir "C:\Users\shahr\Downloads\ayass-bioscn-logo"    