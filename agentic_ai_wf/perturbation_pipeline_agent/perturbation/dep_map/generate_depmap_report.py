#!/usr/bin/env python
"""
generate_depmap_report_llm.py

Self-contained HTML report generator for DepMap functional genomics.

Applied (latest) user updates:
- Codependency section: ADD the data-derived blocks (Positive/Negative/Overlap lists),
  but DO NOT include the removed narrative paragraphs ("Interpretation..." + "Clinical significance...").
- Perturbation section: ADD the PCA/cluster interpretation block + top within-cluster pairs,
  but DO NOT include "Evidence from pairs" or "Clinical validation ideas".
- Remove hardcoded LLM usage for these parts: no LLM calls; only computed + templated text.
- All earlier requested edits remain:
  * Removed "Model linkage rationale" block.
  * Moved FIGURE2B directly below Table 2.
  * Essentiality category blocks show ONLY "Genes: ...".
  * sgRNA section removes "Gene notes" and uses single-line top-5 summaries per direction.

NEW (this request):
- Render top-dependents as PNG panels from DepMap_Dependencies/figs (no iframe => no scroll).
- Insert panel grid directly BELOW Table 2B.
- Make ONLY these panels smaller so 4×4 fits on screen (CSS applies to .panel-grid cards only).
"""

import argparse
import base64
import datetime as dt
import html
import json
import re
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# -------------------------------------------------------------------
# Helpers for paths and safe reading
# -------------------------------------------------------------------

def detect_depmap_run(depmap_root: Path) -> Path:
    if not depmap_root.exists():
        raise FileNotFoundError(f"DepMap root not found: {depmap_root}")

    # First check if DepMap_Dependencies and DepMap_GuideAnalysis are direct children of depmap_root
    ge_direct = depmap_root / "DepMap_Dependencies" / "GeneEssentiality_ByMedian.csv"
    guides_direct = depmap_root / "DepMap_GuideAnalysis" / "CRISPR_GuideLevel_Avana_SelectedModels_long.csv"
    if ge_direct.exists() and guides_direct.exists():
        print(f"[INFO] Found DepMap outputs directly under: {depmap_root}")
        return depmap_root

    # Otherwise, look for DepMap_* subdirectories that contain both as nested folders
    candidates = sorted(
        [d for d in depmap_root.iterdir() if d.is_dir() and d.name.startswith("DepMap_")],
        key=lambda p: p.name,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"No DepMap_* subdirectories under {depmap_root}")

    for d in candidates:
        ge = d / "DepMap_Dependencies" / "GeneEssentiality_ByMedian.csv"
        guides = d / "DepMap_GuideAnalysis" / "CRISPR_GuideLevel_Avana_SelectedModels_long.csv"
        if ge.exists() and guides.exists():
            print(f"[INFO] Selected DepMap run with core outputs: {d}")
            return d

    print("[WARN] No DepMap_* folder with both essentiality + guide tables found; using most recent folder anyway.")
    return candidates[0]


def maybe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        print(f"[WARN] CSV not found: {path}")
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception as e:
        print(f"[WARN] Failed to read CSV {path}: {e}")
        return pd.DataFrame()


def encode_image_to_data_uri(path: Path) -> Optional[str]:
    if not path.exists():
        print(f"[WARN] PNG not found for embedding: {path}")
        return None
    try:
        mime = "image/png"
        if path.suffix.lower() in {".jpg", ".jpeg"}:
            mime = "image/jpeg"
        data = path.read_bytes()
        b64 = base64.b64encode(data).decode("ascii")
        return f"data:{mime};base64,{b64}"
    except Exception as e:
        print(f"[WARN] Could not encode image {path}: {e}")
        return None


def read_html_for_iframe(path: Path) -> Optional[str]:
    if not path or not path.exists():
        print(f"[WARN] HTML figure not found for embedding: {path}")
        return None
    try:
        txt = path.read_text(encoding="utf-8")
        return html.escape(txt, quote=True)
    except Exception as e:
        print(f"[WARN] Could not read HTML {path}: {e}")
        return None


# -------------------------------------------------------------------
# Disease context inference (FinalTesting *_DEGs* files/folders)
# -------------------------------------------------------------------

def infer_disease_context(finaltesting_root: Path) -> str:
    if not finaltesting_root.exists():
        print(f"[WARN] FinalTesting root does not exist: {finaltesting_root}")
        return "Unknown"

    exts = {".csv", ".tsv", ".xlsx", ".xls"}
    candidates: List[Path] = []

    for p in finaltesting_root.iterdir():
        if "_DEGs" not in p.name:
            continue
        if p.is_dir():
            candidates.append(p)
        elif p.is_file() and p.suffix.lower() in exts:
            candidates.append(p)

    if not candidates:
        print(f"[WARN] No *_DEGs* files/folders found under: {finaltesting_root}")
        return "Unknown"

    def key_prioritized(path: Path) -> Tuple[int, float]:
        pri = 1 if "prioritized" in path.name.lower() else 0
        mtime = path.stat().st_mtime
        return (pri, mtime)

    pick = max(candidates, key=key_prioritized)
    base = pick.stem if pick.is_file() else pick.name
    disease_raw = base.split("_DEGs", 1)[0].strip()
    if not disease_raw:
        return "Unknown"

    disease_raw = disease_raw.replace("_", " ").strip()
    return " ".join(w.capitalize() for w in disease_raw.split())


# -------------------------------------------------------------------
# Robust inference of disease + lineage from Table 4 (guides table)
# -------------------------------------------------------------------

def infer_disease_and_lineage_from_guides(guides_df: pd.DataFrame) -> Dict[str, List[str]]:
    out = {"diseases": [], "lineages": []}
    if guides_df.empty:
        return out

    if "OncotreePrimaryDisease" in guides_df.columns:
        ds = guides_df["OncotreePrimaryDisease"].dropna().astype(str).str.strip()
        out["diseases"] = sorted([x for x in ds.unique() if x])

    if "OncotreeLineage" in guides_df.columns:
        ln = guides_df["OncotreeLineage"].dropna().astype(str).str.strip()
        out["lineages"] = sorted([x for x in ln.unique() if x])

    return out


def merge_model_cfg_with_guides(model_cfg: Dict[str, Any], guides_df: pd.DataFrame) -> Dict[str, Any]:
    inferred = infer_disease_and_lineage_from_guides(guides_df)

    if not model_cfg.get("diseases"):
        model_cfg["diseases"] = inferred["diseases"]
    if not model_cfg.get("lineages"):
        model_cfg["lineages"] = inferred["lineages"]

    model_cfg["_inferred_from_guides"] = inferred
    return model_cfg


# -------------------------------------------------------------------
# ModelSelection parsing (optional)
# -------------------------------------------------------------------

def parse_model_selection(finaltesting_root: Path) -> Dict[str, Any]:
    ms_path = finaltesting_root / "ModelSelection.txt"
    ex_path = finaltesting_root / "Explanation.txt"

    mode = ""
    disease_list: List[str] = []
    lineages: List[str] = []
    explanation_text = ""

    if ms_path.exists():
        lines = ms_path.read_text(encoding="utf-8").splitlines()
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("mode=") and not stripped.startswith("#"):
                mode = stripped.split("=", 1)[1].strip()
            elif stripped.startswith("diseases=") and not stripped.startswith("#"):
                disease_list = [x.strip() for x in stripped.split("=", 1)[1].split(",") if x.strip()]
            elif stripped.startswith("lineages=") and not stripped.startswith("#"):
                lineages = [x.strip() for x in stripped.split("=", 1)[1].split(",") if x.strip()]
    else:
        print(f"[WARN] ModelSelection.txt not found at {ms_path}")

    if ex_path.exists():
        lines = ex_path.read_text(encoding="utf-8").splitlines()
        out_lines: List[str] = []
        for line in lines:
            if "Raw JSON configuration" in line:
                break
            out_lines.append(line)
        explanation_text = "\n".join(out_lines).strip()
    else:
        print(f"[WARN] Explanation.txt not found at {ex_path}")

    return {"mode": mode, "diseases": disease_list, "lineages": lineages, "explanation": explanation_text}


# -------------------------------------------------------------------
# LLM interface (kept for compatibility; no longer used for removed blocks)
# -------------------------------------------------------------------

@dataclass
class LLMConfig:
    temperature: float = 0.0
    max_chars: int = 3500
    narrative_len: str = "medium"


class BaseLLM:
    def generate(self, prompt: str, cfg: LLMConfig) -> str:
        raise NotImplementedError


class RuleBasedLLM(BaseLLM):
    def generate(self, prompt: str, cfg: LLMConfig) -> str:
        return "Genes: —"[:cfg.max_chars]


class HTTPLLM(BaseLLM):
    def __init__(self, url: str, header_kv: Optional[str] = None, timeout_s: int = 60):
        self.url = url
        self.header_kv = header_kv
        self.timeout_s = timeout_s

    def generate(self, prompt: str, cfg: LLMConfig) -> str:
        payload = json.dumps({"prompt": prompt, "temperature": 0.0}).encode("utf-8")
        req = urllib.request.Request(self.url, data=payload, method="POST")
        req.add_header("Content-Type", "application/json")
        if self.header_kv and ":" in self.header_kv:
            k, v = self.header_kv.split(":", 1)
            req.add_header(k.strip(), v.strip())

        with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
            raw = resp.read()

        try:
            obj = json.loads(raw.decode("utf-8", errors="replace"))
            text = obj.get("text") or obj.get("output") or obj.get("response")
            if isinstance(text, str) and text.strip():
                return text.strip()[:cfg.max_chars]
            return str(obj)[:cfg.max_chars]
        except Exception:
            return raw.decode("utf-8", errors="replace").strip()[:cfg.max_chars]


def make_llm(backend: str, http_url: Optional[str], http_header: Optional[str]) -> BaseLLM:
    backend = (backend or "none").lower()
    if backend == "http":
        if not http_url:
            raise ValueError("--llm_backend http requires --llm_http_url")
        return HTTPLLM(http_url, header_kv=http_header)
    return RuleBasedLLM()


def split_into_paragraphs(text: str) -> List[str]:
    if not text:
        return []
    t = text.strip().replace("\r\n", "\n")
    parts = [p.strip() for p in t.split("\n\n") if p.strip()]
    return parts if parts else [t]


# -------------------------------------------------------------------
# Data summarization utilities
# -------------------------------------------------------------------

def top_genes_by_essentiality_tag(ge_df: pd.DataFrame, tag: str, top_n: int = 12) -> List[Tuple[str, float, int]]:
    if ge_df.empty or not {"Gene", "median_effect", "BiologicalTag"}.issubset(ge_df.columns):
        return []

    df = ge_df.copy()
    df = df[df["BiologicalTag"].astype(str) == str(tag)].copy()
    if df.empty:
        return []

    df["median_effect"] = pd.to_numeric(df["median_effect"], errors="coerce")
    if "n_models" not in df.columns:
        df["n_models"] = np.nan
    df["n_models"] = pd.to_numeric(df["n_models"], errors="coerce")
    df = df.dropna(subset=["median_effect", "Gene"])

    asc = True
    if "growth-suppressive" in tag.lower() or "non-essential" in tag.lower():
        asc = False

    df = df.sort_values("median_effect", ascending=asc).head(top_n)
    out: List[Tuple[str, float, int]] = []
    for _, r in df.iterrows():
        out.append((str(r["Gene"]), float(r["median_effect"]), int(r["n_models"]) if pd.notna(r["n_models"]) else 0))
    return out


def compute_sgrna_gene_lists(guides_df: pd.DataFrame, top_n: int = 15) -> Dict[str, Any]:
    out: Dict[str, Any] = {"top_genes_by_direction": {"Depleted": [], "Enriched": [], "Neutral": []}}
    if guides_df.empty or not {"Gene", "GuideDirection", "GuideLFC"}.issubset(guides_df.columns):
        return out

    df = guides_df.copy()
    df = df.dropna(subset=["Gene", "GuideDirection", "GuideLFC"])
    df["GuideLFC"] = pd.to_numeric(df["GuideLFC"], errors="coerce")
    df = df.dropna(subset=["GuideLFC"])

    for direction in ["Depleted", "Enriched", "Neutral"]:
        sub = df[df["GuideDirection"] == direction].copy()
        if sub.empty:
            continue
        agg = (
            sub.groupby("Gene", as_index=False)
               .agg(guide_count=("Gene", "size"), median_lfc=("GuideLFC", "median"))
        )
        agg["abs_med"] = agg["median_lfc"].abs()
        agg = agg.sort_values(["guide_count", "abs_med"], ascending=[False, False]).head(top_n)
        out["top_genes_by_direction"][direction] = [(str(r["Gene"]), int(r["guide_count"]), float(r["median_lfc"])) for _, r in agg.iterrows()]

    return out


def compute_cluster_summaries(pert_df: pd.DataFrame, top_genes_per_cluster: int = 8) -> Dict[str, Any]:
    out: Dict[str, Any] = {"cluster_sizes": [], "representative_genes": {}}
    req = {"Gene", "KMeansCluster", "HierCluster", "PC1", "PC2"}
    if pert_df.empty or not req.issubset(pert_df.columns):
        return out

    df = pert_df.copy()
    df = df.dropna(subset=["Gene", "KMeansCluster", "HierCluster", "PC1", "PC2"])
    df["KMeansCluster"] = pd.to_numeric(df["KMeansCluster"], errors="coerce")
    df["PC1"] = pd.to_numeric(df["PC1"], errors="coerce")
    df["PC2"] = pd.to_numeric(df["PC2"], errors="coerce")
    df = df.dropna(subset=["KMeansCluster", "PC1", "PC2"])

    sizes = df["KMeansCluster"].value_counts().sort_index()
    out["cluster_sizes"] = [(int(k), int(v)) for k, v in sizes.items()]

    for cluster_id, sub in df.groupby("KMeansCluster"):
        cid = int(cluster_id)
        c1, c2 = sub["PC1"].mean(), sub["PC2"].mean()
        sub2 = sub.copy()
        sub2["dist"] = (sub2["PC1"] - c1) ** 2 + (sub2["PC2"] - c2) ** 2
        reps = sub2.sort_values("dist", ascending=True).head(top_genes_per_cluster)
        out["representative_genes"][cid] = [(str(r["Gene"]), float(r["PC1"]), float(r["PC2"])) for _, r in reps.iterrows()]

    return out


def compute_within_cluster_top_pairs(
    pert_df: pd.DataFrame,
    codep_df: pd.DataFrame,
    method_preference: str = "Spearman",
    top_n: int = 10
) -> List[Tuple[int, str, str, float]]:
    req_p = {"Gene", "KMeansCluster"}
    req_c = {"Gene1", "Gene2", "Correlation", "Method"}
    if pert_df.empty or codep_df.empty or not req_p.issubset(pert_df.columns) or not req_c.issubset(codep_df.columns):
        return []

    g2c = pert_df.dropna(subset=["Gene", "KMeansCluster"]).copy()
    g2c["KMeansCluster"] = pd.to_numeric(g2c["KMeansCluster"], errors="coerce")
    g2c = g2c.dropna(subset=["KMeansCluster"])
    gene_to_cluster = dict(zip(g2c["Gene"].astype(str), g2c["KMeansCluster"].astype(int)))

    df = codep_df.copy()
    df = df[df["Method"] == method_preference].copy()
    if df.empty:
        return []
    df["Correlation"] = pd.to_numeric(df["Correlation"], errors="coerce")
    df = df.dropna(subset=["Correlation", "Gene1", "Gene2"])
    df["absCorr"] = df["Correlation"].abs()
    df = df.sort_values("absCorr", ascending=False)

    out: List[Tuple[int, str, str, float]] = []
    seen = set()
    for _, r in df.iterrows():
        a, b = str(r["Gene1"]), str(r["Gene2"])
        if a not in gene_to_cluster or b not in gene_to_cluster:
            continue
        ca, cb = gene_to_cluster[a], gene_to_cluster[b]
        if ca != cb:
            continue
        key = (ca, *sorted([a, b]))
        if key in seen:
            continue
        seen.add(key)
        out.append((int(ca), key[1], key[2], float(r["Correlation"])))
        if len(out) >= top_n:
            break
    return out


def compute_codependency_lists(pairs_df: pd.DataFrame, top_n: int = 5) -> Dict[str, Any]:
    """
    Returns:
      - pos_by_method[method] = list of strings (top positive pairs)
      - neg_by_method[method] = list of strings (top negative pairs)
      - overlap = list of strings (pairs appearing in >=2 methods, from the top lists)
    """
    req = {"Gene1", "Gene2", "Correlation", "q_value_fdr_bh", "Method"}
    if pairs_df.empty or not req.issubset(pairs_df.columns):
        return {"pos_by_method": {}, "neg_by_method": {}, "overlap": []}

    df = pairs_df.copy()
    df["Correlation"] = pd.to_numeric(df["Correlation"], errors="coerce")
    df["q_value_fdr_bh"] = pd.to_numeric(df["q_value_fdr_bh"], errors="coerce")
    df = df.dropna(subset=["Gene1", "Gene2", "Correlation", "Method"])

    method_order = ["Spearman", "Pearson", "KendallTau", "Cosine"]
    methods = [m for m in method_order if m in set(df["Method"].unique())]
    for m in sorted(set(df["Method"].unique())):
        if m not in methods:
            methods.append(m)

    pos_by_method: Dict[str, List[str]] = {}
    neg_by_method: Dict[str, List[str]] = {}

    pair_to_methods: Dict[Tuple[str, str], set] = {}

    for method in methods:
        sub = df[df["Method"] == method].copy()
        if sub.empty:
            continue

        pos = sub[sub["Correlation"] > 0].sort_values("Correlation", ascending=False).head(top_n)
        neg = sub[sub["Correlation"] < 0].sort_values("Correlation", ascending=True).head(top_n)

        def fmt_row(r: pd.Series) -> str:
            g1, g2 = str(r["Gene1"]), str(r["Gene2"])
            corr = float(r["Correlation"])
            method = str(r["Method"])

            if method == "Cosine":
                return f"{g1}–{g2} (corr={corr:.3f})"

            q = r.get("q_value_fdr_bh", np.nan)
            if pd.notna(q):
                qtxt = f"q={float(q):.3g}"
                return f"{g1}–{g2} (corr={corr:.3f}, {qtxt})"
            return f"{g1}–{g2} (corr={corr:.3f})"

        pos_list = [fmt_row(r) for _, r in pos.iterrows()]
        neg_list = [fmt_row(r) for _, r in neg.iterrows()]
        pos_by_method[method] = pos_list
        neg_by_method[method] = neg_list

        for _, r in pd.concat([pos, neg], axis=0).iterrows():
            a, b = str(r["Gene1"]), str(r["Gene2"])
            key = tuple(sorted([a, b]))
            pair_to_methods.setdefault(key, set()).add(method)

    overlap_rows = []
    for (a, b), ms in pair_to_methods.items():
        if len(ms) >= 2:
            overlap_rows.append((a, b, len(ms), ", ".join(sorted(ms))))
    overlap_rows.sort(key=lambda x: (-x[2], x[0], x[1]))
    overlap = [f"{a}–{b} (in {k} methods: {ms})" for a, b, k, ms in overlap_rows[:10]]

    return {"pos_by_method": pos_by_method, "neg_by_method": neg_by_method, "overlap": overlap, "methods": methods}


# -------------------------------------------------------------------
# HTML helpers (tables searchable + sortable)
# -------------------------------------------------------------------

def render_searchable_sortable_table(
    table_id: str,
    columns: List[str],
    rows: List[List[Any]],
    caption: str,
    caption_id: Optional[str] = None,
) -> str:
    safe_cols = [html.escape(c) for c in columns]
    header_html = "".join(
        f'<th onclick="sortTable(\'{table_id}\', {i})" title="Click to sort">{c}</th>'
        for i, c in enumerate(safe_cols)
    )

    if rows:
        body_parts = []
        for r in rows:
            tds = "".join(f"<td>{html.escape(str(x))}</td>" for x in r)
            body_parts.append(f"<tr>{tds}</tr>")
        body_html = "\n".join(body_parts)
    else:
        body_html = f'<tr><td colspan="{len(columns)}">No data available</td></tr>'

    caption_label = (
        f'<div class="table-caption pastel-caption" id="{caption_id}">{html.escape(caption)}</div>'
        if caption_id
        else f'<div class="table-caption pastel-caption">{html.escape(caption)}</div>'
    )

    return f"""
    <div class="table-card">
      <div class="table-header">
        <div class="table-title">{html.escape(caption.split('.')[0])}</div>
        <input type="text" class="table-search" placeholder="Search…" oninput="filterTable('{table_id}', this.value)">
      </div>
      <div class="table-wrapper">
        <table id="{table_id}">
          <thead><tr>{header_html}</tr></thead>
          <tbody>{body_html}</tbody>
        </table>
      </div>
      {caption_label}
    </div>
    """


def render_figure_card(
    fig_id: str,
    title: str,
    description: str,
    png_path: Optional[Path],
    html_path: Optional[Path],
    *,
    show_png: bool = True,
    show_html_iframe: bool = True,
    show_html_link: bool = True,
) -> Optional[str]:
    has_png = png_path is not None and png_path.exists() if show_png else False
    has_html = html_path is not None and html_path.exists() if (show_html_iframe or show_html_link) else False
    if not (has_png or has_html):
        return None

    title_safe = html.escape(title)
    desc_safe = html.escape(description)

    if has_png:
        data_uri = encode_image_to_data_uri(png_path)
        img_html = (
            f'<img class="figure-img" src="{data_uri}" alt="{title_safe}">'
            if data_uri
            else '<div class="figure-placeholder">Figure image could not be loaded.</div>'
        )
    else:
        img_html = ""

    iframe_block = ""
    link_html = ""
    if has_html and show_html_iframe:
        escaped_html = read_html_for_iframe(html_path)
        if escaped_html:
            iframe_block = f"""
            <div class="interactive-block">
              <div class="interactive-label">Embedded interactive HTML view</div>
              <iframe class="interactive-frame" srcdoc="{escaped_html}"></iframe>
            </div>
            """
    if has_html and show_html_link:
        href = html.escape(str(html_path.as_posix()))
        link_html = (
            '<div class="figure-link">'
            f'<a href="{href}" target="_blank" rel="noopener">'
            "Open interactive view in a new browser tab."
            "</a></div>"
        )

    return f"""
    <div class="figure-card" id="{fig_id}">
      <div class="figure-label">{html.escape(fig_id.upper())}</div>
      <div class="figure-title">{title_safe}</div>
      <div class="figure-text">{desc_safe}</div>
      {img_html}
      {link_html}
      {iframe_block}
    </div>
    """


def render_fullwidth_figures(cards: List[Optional[str]]) -> str:
    parts: List[str] = []
    for c in cards:
        if c:
            parts.append(f'<div class="figure-row">{c}</div>')
    return "\n".join(parts)


def render_block(title: str, text: str) -> str:
    paragraphs = split_into_paragraphs(text)
    ps_parts = []

    for p in paragraphs:
        if not p.strip():
            continue

        safe = html.escape(p)

        if p.strip() == "Positive correlation: top 5 pairs in each method":
            safe = f'<span class="codep-h codep-pos">{safe}</span>'
        elif p.strip() == "Negative correlation: top 5 pairs in each method":
            safe = f'<span class="codep-h codep-neg">{safe}</span>'
        elif p.strip() == "Pairs recurring across methods":
            safe = f'<span class="codep-h codep-rec">{safe}</span>'

        ps_parts.append(f'<p class="llm-p">{safe}</p>')

    ps = "".join(ps_parts)

    return f"""
    <div class="llm-block">
      <div class="llm-title">{html.escape(title)}</div>
      {ps}
    </div>
    """


# -------------------------------------------------------------------
# Panel grid helper (for top_dependents_*.png gene panels)
# -------------------------------------------------------------------

def render_panel_grid_figures(cards: List[Optional[str]], n_cols: int = 4) -> str:
    parts: List[str] = [c for c in cards if c]
    if not parts:
        return ""
    return f'<div class="panel-grid panel-grid-cols-{int(n_cols)}">' + "\n".join(parts) + "</div>"


def build_top_dependents_panel(dep_dep: Path, max_panels: int = 16, n_cols: int = 4) -> str:
    """
    Render top-dependents as PNG panels from DepMap_Dependencies/figs (no iframe => no scroll).
    Uses files matching: figs/top_dependents_*.png
    Only renders complete rows of n_cols (no leftover last row).
    """
    figs_dir = dep_dep / "figs"
    files = sorted(figs_dir.glob("top_dependents_*.png"), key=lambda p: p.name.lower())

    if not files:
        return ""

    # cap first
    if max_panels and max_panels > 0:
        files = files[:max_panels]

    # drop leftovers so we only show full rows (4,8,12,16...)
    full_count = (len(files) // n_cols) * n_cols
    if full_count == 0:
        return ""
    files = files[:full_count]

    cards: List[Optional[str]] = []
    for p in files:
        gene = p.stem.replace("top_dependents_", "", 1)

        # PNG-only figure card (no html iframe)
        cards.append(
            render_figure_card(
                f"top_dependents_{gene}",
                f"Top dependents – {gene}",
                "",            # keep description empty for compact panels
                p,             # png_path
                None,          # html_path
                show_png=True,
                show_html_iframe=False,
                show_html_link=False,
            )
        )

    grid_html = render_panel_grid_figures(cards, n_cols=n_cols)
    if not grid_html:
        return ""

    return (
        '<div class="clinical-block">'
        '<div class="clinical-title">Top-dependents panels (Top_dependents)</div>'
        '</div>'
        + grid_html
    )


# -------------------------------------------------------------------
# Section builders
# -------------------------------------------------------------------

def build_module_overview_section(
    ge_df: pd.DataFrame,
    guides_df: pd.DataFrame,
    pert_df: pd.DataFrame,
    model_cfg: Dict[str, Any],
) -> str:
    n_ge_genes = int(ge_df["Gene"].nunique()) if not ge_df.empty and "Gene" in ge_df.columns else 0
    n_guides = len(guides_df) if not guides_df.empty else 0
    n_guides_genes = int(guides_df["Gene"].nunique()) if (not guides_df.empty and "Gene" in guides_df.columns) else 0
    n_pert_genes = int(pert_df["Gene"].nunique()) if (not pert_df.empty and "Gene" in pert_df.columns) else 0
    n_kmeans = int(pert_df["KMeansCluster"].nunique()) if (not pert_df.empty and "KMeansCluster" in pert_df.columns) else 0
    n_hier = int(pert_df["HierCluster"].nunique()) if (not pert_df.empty and "HierCluster" in pert_df.columns) else 0

    all_genes: set = set()
    for df in [ge_df, guides_df, pert_df]:
        if not df.empty and "Gene" in df.columns:
            all_genes.update(df["Gene"].dropna().astype(str))
    n_total_genes = len(all_genes)

    table1_rows = [
        ["Gene essentiality", f"{n_ge_genes} genes" if n_ge_genes else "—",
         "Chronos dependency scores, biological tags", "Survival-oriented and growth-suppressive genes."],
        ["sgRNA guide-level (CRISPR)", f"{n_guides:,} guides · {n_guides_genes} genes" if n_guides else "—",
         "Per-guide log-fold changes, depletion/enrichment, direction", "Model-specific sgRNA effects and context-dependent dependencies."],
        ["Gene–gene codependency", "Top correlated gene pairs",
         "Cosine / Pearson / Spearman / Kendall", "Co-essential and antagonistic relationships across models."],
        ["Perturbation similarity & clustering",
         f"{n_pert_genes} genes · {n_kmeans} k-means clusters · {n_hier} hierarchical clusters" if n_pert_genes else "—",
         "PCA, k-means, hierarchical clustering", "Shared dependency signatures and coordinated gene groupings."],
    ]
    table1_caption = f"Table 1. DepMap modules and data coverage (total unique genes across modules: {n_total_genes})."

    table_html = render_searchable_sortable_table(
        "table_modules",
        ["Module", "Data scope", "Key metrics", "Interpretation focus"],
        table1_rows,
        table1_caption,
        caption_id="table1_caption",
    )

    diseases = model_cfg.get("diseases", [])
    lineages = model_cfg.get("lineages", [])
    disease_label = ", ".join(diseases) if diseases else "—"
    lineage_label = ", ".join(lineages) if lineages else "—"

    expl_html = f"""
    <div class="clinical-block">
      <div class="clinical-title">Configured disease models</div>
      <ul>
        <li><strong>Configured diseases:</strong> {html.escape(disease_label)}</li>
        <li><strong>Configured lineages:</strong> {html.escape(lineage_label)}</li>
      </ul>
    </div>
    """

    return f"""
    <div class="section-card">
      <div class="section-eyebrow">Integrated DepMap overview</div>
      <div class="section-title">Overall functional genomics landscape</div>
      <div class="section-lead">
        Patient-derived differential expression profiles are contextualised using DepMap CRISPR-Cas9 dependency data across gene essentiality, sgRNA-level effects, gene–gene codependency and perturbation similarity modules.
      </div>
      <div class="badge-row">
        <span class="badge badge-essentiality">Gene essentiality</span>
        <span class="badge badge-sgrna">sgRNA-level CRISPR effects</span>
        <span class="badge badge-codep">Gene–gene codependency</span>
        <span class="badge badge-pert">Perturbation similarity &amp; clustering</span>
      </div>
      {table_html}
      {expl_html}
    </div>
    """


def build_gene_essentiality_section(
    dep_dep: Path,
    ge_df: pd.DataFrame,
) -> str:
    rows_classes: List[List[Any]] = []
    if not ge_df.empty and "BiologicalTag" in ge_df.columns:
        class_counts = ge_df["BiologicalTag"].value_counts().reset_index()
        class_counts.columns = ["Essentiality category", "Gene count"]
        total = class_counts["Gene count"].sum()
        class_counts["Proportion of analysed genes"] = ((class_counts["Gene count"] / total) * 100).round(1).astype(str) + " %"
        desired_order = ["Core essential", "Strong dependency", "Moderate dependency", "Weak/Contextual", "Non-essential / growth-suppressive"]
        class_counts["order"] = class_counts["Essentiality category"].apply(lambda x: desired_order.index(x) if x in desired_order else len(desired_order))
        class_counts = class_counts.sort_values("order").drop(columns=["order"])
        rows_classes = class_counts[["Essentiality category", "Gene count", "Proportion of analysed genes"]].values.tolist()

    table2_html = render_searchable_sortable_table(
        "table_ge_classes",
        ["Essentiality category", "Gene count", "Proportion of analysed genes"],
        rows_classes,
        "Table 2. Gene essentiality classes based on Chronos dependency estimates.",
        caption_id="table2_caption",
    )

    # FIGURE2B moved directly below Table 2
    fig_bar_html = dep_dep / "html" / "Dependencies_BiologicalTag_bar.html"
    figure2b_html = render_fullwidth_figures([
        render_figure_card(
            "figure1",
            "Distribution of genes by biological essentiality tag",
            "Interactive bar view of gene counts across essentiality categories.",
            None,
            fig_bar_html,
            show_png=False,
            show_html_iframe=True,
            show_html_link=True,
        )
    ])

    # Table 2B
    rows_median: List[List[Any]] = []
    required_cols = {"Gene", "n_models", "q10", "median_effect", "q90", "BiologicalTag"}
    if not ge_df.empty and required_cols.issubset(ge_df.columns):
        ge_sorted = ge_df.copy()
        ge_sorted["median_effect"] = pd.to_numeric(ge_sorted["median_effect"], errors="coerce")
        ge_sorted = ge_sorted.dropna(subset=["median_effect"]).sort_values("median_effect")
        bottom = ge_sorted.head(15)
        top = ge_sorted.tail(15)
        example_ge_df = pd.concat([bottom, top], axis=0)

        for _, r in example_ge_df.iterrows():
            rows_median.append([
                str(r["Gene"]),
                int(r["n_models"]) if pd.notna(r["n_models"]) else 0,
                f"{float(r['q10']):.3f}" if pd.notna(r["q10"]) else "—",
                f"{float(r['median_effect']):.3f}",
                f"{float(r['q90']):.3f}" if pd.notna(r["q90"]) else "—",
                str(r["BiologicalTag"]) if pd.notna(r["BiologicalTag"]) else "—",
            ])

    table2b_html = render_searchable_sortable_table(
        "table_ge_median",
        ["Gene", "Models", "q10 Chronos", "Median Chronos", "q90 Chronos", "Essentiality tag"],
        rows_median,
        "Table 2B. Example gene-level median Chronos effects for strongly essential and growth-suppressive genes.",
        caption_id="table2b_caption",
    )

    # NEW: PNG top_dependents_*.png panel below Table 2B
    top_dependents_panel_html = build_top_dependents_panel(dep_dep, max_panels=16, n_cols=4)

    # Other figures
    fig_heat_png = dep_dep / "figs" / "TopGenes_Essentiality_heatmap_top30.png"
    fig_heat_html = dep_dep / "html" / "TopGenes_Essentiality_heatmap_top30.html"
    fig_med_html = dep_dep / "html" / "gene_summary_median_effect.html"

    figures_html_rest = render_fullwidth_figures([
        render_figure_card(
            "figure2a",
            "Top essential genes – Chronos dependency heatmap",
            "Heatmap of Chronos dependency scores for strongly essential and growth-suppressive genes across selected models.",
            fig_heat_png,
            fig_heat_html,
            show_png=True,
            show_html_iframe=True,
            show_html_link=True,
        ),
        render_figure_card(
            "figure2c",
            "Median Chronos effect – gene-level summary",
            "Interactive summary view of median Chronos effects across models.",
            None,
            fig_med_html,
            show_png=False,
            show_html_iframe=True,
            show_html_link=True,
        ),
    ])

    # Essentiality blocks: ONLY Genes line
    tag_order = [
        "Core essential",
        "Strong dependency",
        "Moderate dependency",
        "Weak/Contextual",
        "Non-essential / growth-suppressive",
    ]
    blocks: List[str] = []
    for tag in tag_order:
        genes = top_genes_by_essentiality_tag(ge_df, tag, top_n=12)
        gene_symbols = [g for g, _, _ in genes]
        txt = "Genes: " + (", ".join(gene_symbols) if gene_symbols else "—")
        blocks.append(render_block(tag, txt))

    return f"""
    <div class="section-card">
      <div class="section-eyebrow">Gene essentiality calculations</div>
      <div class="section-title">Chronos dependency landscape</div>
      <div class="section-lead">
        Chronos scores classify genes into essentiality categories, highlighting core survival programmes and context-dependent vulnerabilities.
      </div>
      {table2_html}
      {figure2b_html}
      {table2b_html}
      {top_dependents_panel_html}
      {figures_html_rest}
      {''.join(blocks)}
    </div>
    """


def build_sgrna_section(
    dep_guides: Path,
    guides_df: pd.DataFrame,
) -> str:
    rows_dir: List[List[Any]] = []
    if not guides_df.empty and "GuideDirection" in guides_df.columns:
        vc = guides_df["GuideDirection"].value_counts()
        desired = ["Depleted", "Enriched", "Neutral"]
        for label in desired:
            if label in vc.index:
                rows_dir.append([label, int(vc[label])])
        for label in vc.index:
            if label not in desired:
                rows_dir.append([label, int(vc[label])])

    table3_html = render_searchable_sortable_table(
        "table_sgrna_direction",
        ["Direction category", "Guide count"],
        rows_dir,
        "Table 3. sgRNA-level coverage and directionality.",
        caption_id="table3_caption",
    )

    rows_sgrna: List[List[Any]] = []
    required_cols = {"sgRNA", "SequenceID", "Gene", "CellLineName", "OncotreeLineage", "OncotreePrimaryDisease", "GuideLFC", "GuideDirection"}
    if not guides_df.empty and required_cols.issubset(guides_df.columns):
        df = guides_df[list(required_cols)].copy()
        df["GuideLFC"] = pd.to_numeric(df["GuideLFC"], errors="coerce")
        df = df.dropna(subset=["GuideLFC"])
        order_map = {"Depleted": 0, "Enriched": 1, "Neutral": 2}
        df["DirectionOrder"] = df["GuideDirection"].map(lambda x: order_map.get(x, 99))
        df["abs_lfc"] = df["GuideLFC"].abs()
        df = df.sort_values(["DirectionOrder", "abs_lfc"], ascending=[True, False]).head(60)

        for _, r in df.iterrows():
            rows_sgrna.append([
                str(r["sgRNA"]), str(r["SequenceID"]), str(r["Gene"]), str(r["CellLineName"]),
                str(r["OncotreeLineage"]), str(r["OncotreePrimaryDisease"]),
                f"{float(r['GuideLFC']):.6f}", str(r["GuideDirection"]),
            ])

    table4_html = render_searchable_sortable_table(
        "table_sgrna_examples",
        ["sgRNA", "SequenceID", "Gene", "Cell line", "Lineage", "Primary disease", "Guide LFC", "Direction"],
        rows_sgrna,
        "Table 4. Example sgRNA (Avana) guide-level profiles.",
        caption_id="table4_caption",
    )

    figures_html = render_fullwidth_figures([
        render_figure_card("figure3a", "sgRNA log-fold change distribution",
                           "Interactive density and spread of sgRNA log-fold changes across DepMap models.",
                           None, dep_guides / "html" / "Guides_LFC_distribution.html",
                           show_png=False, show_html_iframe=True, show_html_link=True),
        render_figure_card("figure3b", "sgRNA log-fold changes by direction",
                           "Interactive histograms stratified by depleted, enriched and neutral guides.",
                           None, dep_guides / "html" / "Guides_LFC_hist_byDirection.html",
                           show_png=False, show_html_iframe=True, show_html_link=True),
        render_figure_card("figure3c", "Chronos effect heatmap",
                           "Interactive map of model-level Chronos dependency scores for selected genes.",
                           None, dep_guides / "html" / "chronos_effect_heatmap.html",
                           show_png=False, show_html_iframe=True, show_html_link=True),
        render_figure_card("figure3d", "Chronos volcano – median effect vs p-like",
                           "Interactive volcano plot highlighting genes with strong depletion or enrichment signatures.",
                           None, dep_guides / "html" / "chronos_volcano_medianEffect_vs_neglog10_pLike.html",
                           show_png=False, show_html_iframe=True, show_html_link=True),
        render_figure_card("figure3e", "Dependency probability heatmap",
                           "Interactive heatmap of dependency probabilities complementing Chronos effect estimates.",
                           None, dep_guides / "html" / "dependency_probability_heatmap.html",
                           show_png=False, show_html_iframe=True, show_html_link=True),
    ])

    sgrna_summ = compute_sgrna_gene_lists(guides_df, top_n=15)
    top_depl = sgrna_summ["top_genes_by_direction"].get("Depleted", [])[:5]
    top_enr = sgrna_summ["top_genes_by_direction"].get("Enriched", [])[:5]
    top_neu = sgrna_summ["top_genes_by_direction"].get("Neutral", [])[:5]

    def top5_line(label: str, lst: List[Tuple[str, int, float]]) -> str:
        genes = [g for g, _, _ in lst]
        return f"{label}: " + (", ".join(genes) if genes else "—")

    summary_text = "\n".join([
        top5_line("Depleted list genes", top_depl),
        top5_line("Enriched list genes", top_enr),
        top5_line("Neutral list genes", top_neu),
    ])

    summary_block = render_block("sgRNA gene summaries", summary_text)

    return f"""
    <div class="section-card">
      <div class="section-eyebrow">Guide-level CRISPR effects</div>
      <div class="section-title">sgRNA-level enrichment profile</div>
      <div class="section-lead">
        Per-guide CRISPR effects across DepMap models link sgRNA log-fold changes to gene-level essentiality and highlight depletion, enrichment, and neutral patterns.
      </div>
      {table3_html}
      {table4_html}
      {figures_html}
      {summary_block}
    </div>
    """


def build_codependency_section(code_dir: Path, pairs_df: pd.DataFrame) -> str:
    method_order = ["Spearman", "Pearson", "KendallTau", "Cosine"]
    table_panels: List[str] = []

    if not pairs_df.empty and {"Gene1", "Gene2", "Correlation", "q_value_fdr_bh", "Method"}.issubset(pairs_df.columns):
        for method in method_order:
            sub = pairs_df[pairs_df["Method"] == method].copy()
            if sub.empty:
                continue

            sub["Correlation"] = pd.to_numeric(sub["Correlation"], errors="coerce")
            sub["q_value_fdr_bh"] = pd.to_numeric(sub["q_value_fdr_bh"], errors="coerce")
            sub = sub.dropna(subset=["Correlation"])

            rows_code: List[List[Any]] = []
            sub_pos = sub[sub["Correlation"] > 0].sort_values("Correlation", ascending=False).head(50)
            sub_neg = sub[sub["Correlation"] < 0].sort_values("Correlation", ascending=True).head(50)

            for sign_label, df_part in [("Positive", sub_pos), ("Negative", sub_neg)]:
                for _, r in df_part.iterrows():
                    rows_code.append([
                        method, sign_label, str(r["Gene1"]), str(r["Gene2"]),
                        f"{float(r['Correlation']):.3f}",
                        f"{float(r['q_value_fdr_bh']):.3g}" if pd.notna(r["q_value_fdr_bh"]) else "—",
                    ])

            table_panels.append(render_searchable_sortable_table(
                f"table_codependency_{method.lower()}",
                ["Method", "Direction", "Gene 1", "Gene 2", "Correlation", "FDR q-value"],
                rows_code,
                f"Table 5 – {method} correlation: top positive and negative gene–gene codependencies.",
                caption_id=f"table5_{method.lower()}_caption",
            ))

    tables_html = '<div class="table-grid">' + "\n".join(table_panels) + "</div>" if table_panels else ""

    figures_html = render_fullwidth_figures([
        render_figure_card("figure4_spearman", "Codependency heatmap – Spearman",
                           "Rank-based Spearman correlation heatmap capturing monotonic relationships among genes.",
                           None, code_dir / "html" / "codependency_heatmap_interactive_spearman.html",
                           show_png=False, show_html_iframe=True, show_html_link=True),
        render_figure_card("figure4_pearson", "Codependency heatmap – Pearson",
                           "Pearson correlation heatmap capturing linear relationships among genes.",
                           None, code_dir / "html" / "codependency_heatmap_interactive_pearson.html",
                           show_png=False, show_html_iframe=True, show_html_link=True),
        render_figure_card("figure4_kendall", "Codependency heatmap – Kendall tau",
                           "Kendall tau correlation heatmap capturing rank-based associations among genes.",
                           None, code_dir / "html" / "codependency_heatmap_interactive_kendall.html",
                           show_png=False, show_html_iframe=True, show_html_link=True),
        render_figure_card("figure4_cosine", "Codependency heatmap – Cosine",
                           "Cosine similarity heatmap capturing angular similarity among gene dependency profiles.",
                           None, code_dir / "html" / "codependency_heatmap_interactive_cosine.html",
                           show_png=False, show_html_iframe=True, show_html_link=True),
    ])

    lists = compute_codependency_lists(pairs_df, top_n=5)
    methods = lists.get("methods", [])
    pos_by_method = lists.get("pos_by_method", {})
    neg_by_method = lists.get("neg_by_method", {})
    overlap = lists.get("overlap", []) or ["—"]

    lines: List[str] = []
    lines.append("Positive correlation: top 5 pairs in each method")
    for m in methods:
        items = pos_by_method.get(m, [])
        joined = "; ".join(items) if items else "—"
        lines.append(f"{m} top positive: {joined}")

    lines.append("")
    lines.append("Negative correlation: top 5 pairs in each method")
    for m in methods:
        items = neg_by_method.get(m, [])
        joined = "; ".join(items) if items else "—"
        lines.append(f"{m} top −: {joined}")

    lines.append("")
    lines.append("Pairs recurring across methods")
    for item in overlap[:10]:
        lines.append(item)

    codep_block = render_block("Interpretation – gene–gene codependency & modules", "\n".join(lines))

    return f"""
    <div class="section-card">
      <div class="section-eyebrow">Gene–gene relationships</div>
      <div class="section-title">Codependency metrics</div>
      <div class="section-lead">
        Correlation-based codependency metrics quantify co-essential and antagonistic relationships between genes using multiple complementary correlation definitions.
      </div>
      {tables_html}
      {figures_html}
      {codep_block}
    </div>
    """


def build_ace_causality_section(
    dep_causality: Path,
) -> str:
    """
    Build ACE (Average Causal Effect) causality analysis section.
    Shows ACE computation results, therapeutic alignment, and causal graphs.
    """
    # Check if ACE analysis was run
    ace_csv = dep_causality / "CausalEffects_ACE.csv"
    ranked_csv = dep_causality / "CausalDrivers_Ranked.csv"
    
    if not ace_csv.exists() or not ranked_csv.exists():
        # ACE was not run or failed - return empty section
        return ""
    
    # Load ACE data
    ace_df = maybe_read_csv(ace_csv)
    ranked_df = maybe_read_csv(ranked_csv)
    
    if ace_df.empty or ranked_df.empty:
        return ""
    
    # Table: Top ACE drivers with therapeutic alignment
    rows_ace: List[List[Any]] = []
    req_cols = {"gene", "ACE", "CI_low", "CI_high", "Stability", "Verdict"}
    
    if req_cols.issubset(ace_df.columns):
        # Get top robust drivers (essential genes)
        robust = ace_df[ace_df["Verdict"] == "Robust"].copy()
        robust["absACE"] = robust["ACE"].abs()
        top_ace = robust.sort_values("absACE", ascending=False).head(30)
        
        # Merge with ranked data for therapeutic alignment
        if not ranked_df.empty and "TherapeuticAlignment" in ranked_df.columns:
            top_ace = top_ace.merge(
                ranked_df[["gene", "TherapeuticAlignment"]].drop_duplicates("gene"),
                on="gene",
                how="left"
            )
        
        for _, r in top_ace.iterrows():
            align = r.get("TherapeuticAlignment", "—")
            rows_ace.append([
                str(r["gene"]),
                f"{float(r['ACE']):.4f}",
                f"{float(r['CI_low']):.4f}",
                f"{float(r['CI_high']):.4f}",
                f"{float(r['Stability']):.3f}",
                str(r["Verdict"]),
                str(align) if pd.notna(align) else "—"
            ])
    
    table_ace_html = render_searchable_sortable_table(
        "table_ace_drivers",
        ["Gene", "ACE", "CI Low", "CI High", "Stability", "Verdict", "Therapeutic Alignment"],
        rows_ace,
        "Table 7. Top causal drivers ranked by Average Causal Effect (ACE) with therapeutic alignment.",
        caption_id="table7_caption",
    )
    
    # Figures from ACE analysis
    figs_html_dir = dep_causality / "figs_html"
    graphs_dir = dep_causality / "graphs_extra"
    
    figures_html = render_fullwidth_figures([
        render_figure_card(
            "figure6a",
            "Top causal drivers – Forest plot",
            "ACE estimates with 95% confidence intervals for top causal driver genes.",
            None,
            figs_html_dir / "01_top_drivers_forest.html",
            show_png=False,
            show_html_iframe=True,
            show_html_link=True,
        ),
        render_figure_card(
            "figure6b",
            "ACE vs Stability",
            "Scatter plot showing relationship between Average Causal Effect and bootstrap sign-stability.",
            None,
            figs_html_dir / "02_ace_vs_stability.html",
            show_png=False,
            show_html_iframe=True,
            show_html_link=True,
        ),
        render_figure_card(
            "figure6c",
            "Volcano-style ACE plot",
            "Volcano-style visualization of ACE magnitude versus statistical significance.",
            None,
            figs_html_dir / "03_volcano_like.html",
            show_png=False,
            show_html_iframe=True,
            show_html_link=True,
        ),
        render_figure_card(
            "figure6d",
            "Signed ACE causal graph",
            "Interactive network showing signed causal relationships between genes and viability.",
            None,
            graphs_dir / "signed_gene_viability_alignment.html",
            show_png=False,
            show_html_iframe=True,
            show_html_link=True,
        ),
    ])
    
    # Summary text about therapeutic alignment
    if not ranked_df.empty and "TherapeuticAlignment" in ranked_df.columns:
        reversal_count = (ranked_df["TherapeuticAlignment"] == "Reversal").sum()
        aggravating_count = (ranked_df["TherapeuticAlignment"] == "Aggravating").sum()
        
        summary_lines = [
            f"Reversal targets: {reversal_count} genes show negative ACE × log2FC product (CRISPR perturbation counteracts patient expression state)",
            f"Aggravating factors: {aggravating_count} genes show positive ACE × log2FC product (CRISPR perturbation reinforces patient expression state)",
            "ACE quantifies the average causal effect of gene perturbation on cell viability across DepMap models",
            "Therapeutic alignment identifies genes where perturbation reverses disease-associated expression patterns"
        ]
        summary_block = render_block("Interpretation – ACE causality & therapeutic alignment", "\n".join(summary_lines))
    else:
        summary_block = ""
    
    return f"""
    <div class="section-card">
      <div class="section-eyebrow">ACE Causality Analysis</div>
      <div class="section-title">Average Causal Effects & Therapeutic Alignment</div>
      <div class="section-lead">
        ACE (Average Causal Effect) quantifies the causal impact of gene perturbations on cell viability, 
        combining DepMap dependency data with patient differential expression to identify therapeutic reversal candidates.
      </div>
      {table_ace_html}
      {figures_html}
      {summary_block}
    </div>
    """


def build_perturbation_section(
    dep_guides: Path,
    pert_df: pd.DataFrame,
    codep_df: pd.DataFrame,
) -> str:
    rows_pert: List[List[Any]] = []
    req = {"Gene", "KMeansCluster", "HierCluster", "PC1", "PC2"}
    if not pert_df.empty and req.issubset(pert_df.columns):
        df = pert_df.copy()
        df["KMeansCluster"] = pd.to_numeric(df["KMeansCluster"], errors="coerce")
        df = df.dropna(subset=["KMeansCluster"]).sort_values("KMeansCluster").head(60)
        for _, r in df.iterrows():
            rows_pert.append([
                str(r["Gene"]),
                int(r["KMeansCluster"]),
                int(r["HierCluster"]),
                f"{float(r['PC1']):.3f}",
                f"{float(r['PC2']):.3f}",
            ])

    table6_html = render_searchable_sortable_table(
        "table_perturbation",
        ["Gene", "K-means cluster", "Hierarchical cluster", "PC1", "PC2"],
        rows_pert,
        "Table 6. Genes with perturbation similarity cluster and PCA coordinates.",
        caption_id="table6_caption",
    )

    figures_html = render_fullwidth_figures([
        render_figure_card("figure5a", "Perturbation similarity – PCA with k-means clusters",
                           "Interactive PCA projection of gene dependency profiles annotated by k-means clusters.",
                           None, dep_guides / "html" / "perturbation_similarity_pca_kmeans.html",
                           show_png=False, show_html_iframe=True, show_html_link=True),
        render_figure_card("figure5b", "Perturbation similarity – PCA with hierarchical clusters",
                           "Interactive PCA projection annotated by hierarchical clustering.",
                           None, dep_guides / "html" / "perturbation_similarity_pca_hierarchical.html",
                           show_png=False, show_html_iframe=True, show_html_link=True),
    ])

    cs = compute_cluster_summaries(pert_df, top_genes_per_cluster=8)
    cluster_sizes = cs["cluster_sizes"]
    top_clusters = sorted(cluster_sizes, key=lambda x: -x[1])[:6] if cluster_sizes else []

    within_pairs = compute_within_cluster_top_pairs(pert_df, codep_df, method_preference="Spearman", top_n=10)
    within_pairs_lines = [f"Cluster {cid}: {a}–{b} (corr={corr:.3f})" for cid, a, b, corr in within_pairs] if within_pairs else ["—"]

    lines: List[str] = []
    lines.append("Cluster:")
    if top_clusters:
        for cid, _sz in top_clusters:
            reps = cs["representative_genes"].get(cid, [])[:4]
            rep_genes = ", ".join([g for g, _, _ in reps]) if reps else "—"
            lines.append(f"**Cluster {cid}:** Genes: {rep_genes}")
    else:
        lines.append("- —")

    lines.append("")
    lines.append("Top within-cluster codependency pairs")
    for x in within_pairs_lines[:10]:
        lines.append(x)

    pert_block = render_block("Interpretation – perturbation similarity & biomarker modules", "\n".join(lines))

    return f"""
    <div class="section-card">
      <div class="section-eyebrow">Perturbation similarity &amp; clustering</div>
      <div class="section-title">Gene modules with shared dependency signatures</div>
      <div class="section-lead">
        PCA and clustering of Chronos dependency profiles group genes into modules with shared perturbation signatures.
      </div>
      {table6_html}
      {figures_html}
      {pert_block}
    </div>
    """


# -------------------------------------------------------------------
# Main HTML template (FIXED: not an f-string)
# -------------------------------------------------------------------

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Ayass Bioscience · Functional Genomics &amp; DepMap Vulnerability Report</title>
  <style>
    :root {{
      --bg-page: #f5f7fb;
      --bg-card: #ffffff;
      --border-soft: #e1e7f5;
      --border-table: #e3ebff;
      --accent-blue: #3b6cff;
      --text-main: #1f2933;
      --text-muted: #6b778c;
      --text-soft: #9fb3c8;
      --text-strong: #102a43;
      --table-header-bg: #f0f3ff;
      --table-row-alt: #f9fbff;
    }}

    .codep-h {{ font-weight: 700; }}
    .codep-pos {{ color: #047857; }}
    .codep-neg {{ color: #b42318; }}
    .codep-rec {{ color: #1d4ed8; }}

    * {{ box-sizing: border-box; }}
    body {{
      font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: var(--bg-page);
      color: var(--text-main);
      margin: 0;
      padding: 0;
    }}
    .page {{
      max-width: 1200px;
      margin: 24px auto 40px auto;
      padding: 0 24px 40px 24px;
    }}
    .report-card {{
      background: var(--bg-card);
      border-radius: 26px;
      padding: 0 0 28px 0;
      box-shadow: 0 18px 45px rgba(15, 35, 52, 0.10);
      border: 1px solid var(--border-soft);
      overflow: hidden;
    }}
    .hero {{
      position: relative;
      background: linear-gradient(93deg, #0099cc, #00b894);
      padding: 20px 28px 22px 28px;
      color: #ffffff;
    }}
    .hero-row {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
    }}
    .ayass-logo {{
      height: 52px;
      object-fit: contain;
    }}
    .hero-center {{
      flex: 1;
      text-align: left;
    }}
    .hero-eyebrow {{
      font-size: 11px;
      letter-spacing: 0.16em;
      text-transform: uppercase;
      opacity: 0.9;
      margin-bottom: 4px;
      font-weight: 600;
    }}
    .hero-title {{
      font-size: 24px;
      font-weight: 700;
      letter-spacing: 0.02em;
      margin: 0;
    }}
    .hero-subtitle {{
      margin-top: 6px;
      font-size: 13px;
      opacity: 0.95;
    }}
    .hero-right {{
      display: flex;
      flex-direction: column;
      gap: 8px;
      align-items: flex-end;
    }}
    .hero-pill {{
      padding: 6px 12px;
      border-radius: 999px;
      font-size: 11px;
      font-weight: 600;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      border: 1px solid rgba(255,255,255,0.7);
      background: rgba(0,0,0,0.05);
      white-space: nowrap;
    }}
    .timestamp {{
      position: absolute;
      top: 10px;
      right: 18px;
      font-size: 11px;
      color: rgba(240,252,255,0.9);
    }}
    .pill-row {{
      display: flex;
      gap: 10px;
      justify-content: center;
      margin: 16px 28px 16px 28px;
      flex-wrap: wrap;
    }}
    .pill {{
      padding: 6px 14px;
      border-radius: 999px;
      border: 1px solid #dde5ff;
      background: #f6f8ff;
      font-size: 12px;
      color: #3d4f6a;
    }}

    .section-card {{
      margin: 0 28px;
      margin-top: 22px;
      background: #ffffff;
      border-radius: 22px;
      padding: 24px 24px 24px 24px;
      border: 1px solid var(--border-soft);
    }}

    .section-eyebrow {{
      font-size: 11px;
      letter-spacing: 0.16em;
      text-transform: uppercase;
      color: var(--text-soft);
      margin-bottom: 4px;
      font-weight: 600;
    }}
    .section-title {{
      font-size: 20px;
      color: var(--text-strong);
      font-weight: 650;
      margin-bottom: 6px;
    }}
    .section-lead {{
      font-size: 13px;
      color: var(--text-muted);
      margin-bottom: 18px;
    }}
    .badge-row {{
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      margin-bottom: 14px;
    }}
    .badge {{
      border-radius: 999px;
      padding: 5px 12px;
      font-size: 12px;
      font-weight: 500;
    }}
    .badge-essentiality {{ background: #ffe4ea; color: #b42318; }}
    .badge-sgrna {{ background: #fff4d5; color: #92400e; }}
    .badge-codep {{ background: #e5fbf2; color: #047857; }}
    .badge-pert {{ background: #e9f1ff; color: #1d4ed8; }}

    .table-card {{
      margin-top: 10px;
      border-radius: 18px;
      border: 1px solid var(--border-soft);
      background: #fbfdff;
      padding: 12px 14px 10px 14px;
    }}
    .table-header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 8px;
      gap: 10px;
      flex-wrap: wrap;
    }}
    .table-title {{
      font-size: 13px;
      font-weight: 600;
      color: #243b53;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }}
    .table-search {{
      border-radius: 999px;
      border: 1px solid #d9e2ec;
      padding: 4px 10px;
      font-size: 12px;
      min-width: 180px;
    }}
    .table-wrapper {{
      max-height: 260px;
      overflow: auto;
      border-radius: 12px;
      border: 1px solid var(--border-table);
      background: #ffffff;
    }}
    table {{
      border-collapse: collapse;
      width: 100%;
      font-size: 12px;
    }}
    thead tr {{
      background: var(--table-header-bg);
      position: sticky;
      top: 0;
      z-index: 1;
    }}
    th, td {{
      padding: 6px 10px;
      border-bottom: 1px solid #edf2ff;
      text-align: left;
      white-space: nowrap;
      cursor: default;
    }}
    th {{
      cursor: pointer;
      user-select: none;
    }}
    tbody tr:nth-child(even) td {{ background: var(--table-row-alt); }}
    tbody tr:hover td {{ background: #eef2ff; }}
    .table-caption {{
      margin-top: 6px;
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.12em;
      font-weight: 600;
    }}
    .pastel-caption {{ color: #c05621; }}

    .figure-row {{
      display: flex;
      flex-wrap: wrap;
      gap: 16px;
      margin-top: 18px;
    }}
    .figure-card {{
      flex: 1 1 260px;
      min-width: 260px;
      max-width: 100%;
      border-radius: 18px;
      border: 1px solid var(--border-soft);
      background: #ffffff;
      padding: 12px 12px 14px 12px;
      display: flex;
      flex-direction: column;
      min-height: 260px;
    }}
    .figure-label {{
      font-size: 11px;
      font-weight: 600;
      color: var(--accent-blue);
      letter-spacing: 0.12em;
      text-transform: uppercase;
      margin-bottom: 4px;
    }}
    .figure-title {{
      font-size: 13px;
      font-weight: 600;
      color: var(--text-strong);
      margin-bottom: 4px;
    }}
    .figure-text {{
      font-size: 12px;
      color: var(--text-muted);
      margin-bottom: 8px;
    }}
    .figure-img {{
      width: 100%;
      border-radius: 10px;
      border: 1px solid #d9e2ec;
      display: block;
    }}
    .figure-link {{
      margin-top: 6px;
      font-size: 11px;
    }}
    .figure-link a {{
      color: var(--accent-blue);
      text-decoration: none;
    }}
    .figure-link a:hover {{ text-decoration: underline; }}

    .interactive-block {{
      margin-top: 10px;
      border-radius: 14px;
      border: 1px dashed #cbd2ff;
      background: #f8f9ff;
      padding: 8px;
    }}
    .interactive-label {{
      font-size: 11px;
      color: #4b5f82;
      margin-bottom: 4px;
    }}
    .interactive-frame {{
      width: 100%;
      min-height: 420px;
      border: 1px solid #d0d7ff;
      border-radius: 8px;
      background: #ffffff;
    }}

    .clinical-block {{
      margin-top: 16px;
      border-radius: 18px;
      background: #f5f7ff;
      padding: 14px 16px 12px 16px;
    }}
    .clinical-title {{
      font-size: 13px;
      font-weight: 650;
      color: var(--text-strong);
      margin-bottom: 6px;
    }}
    .clinical-block ul {{
      margin: 0 0 0 18px;
      padding: 0;
      font-size: 12px;
      color: var(--text-muted);
    }}
    .clinical-block li {{ margin-bottom: 3px; }}

    .llm-block {{
      margin-top: 18px;
      border-radius: 18px;
      background: #ffffff;
      border: 1px solid var(--border-soft);
      padding: 14px 16px 12px 16px;
      box-shadow: 0 10px 22px rgba(15, 35, 52, 0.06);
    }}
    .llm-title {{
      font-size: 13px;
      font-weight: 700;
      color: var(--text-strong);
      margin-bottom: 8px;
      letter-spacing: 0.02em;
    }}
    .llm-p {{
      margin: 0 0 8px 0;
      font-size: 12.5px;
      color: var(--text-muted);
      line-height: 1.45;
      white-space: pre-wrap;
    }}

    .footer {{
      margin-top: 24px;
      margin-left: 28px;
      margin-right: 28px;
      font-size: 11px;
      color: var(--text-soft);
    }}
    .references {{
      margin-top: 8px;
      font-size: 11px;
    }}
    .references a {{
      color: var(--accent-blue);
      text-decoration: none;
    }}
    .references a:hover {{ text-decoration: underline; }}

    .table-grid {{
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
    }}
    .table-grid .table-card {{
      flex: 1 1 260px;
    }}

    /* === Compact 4×4 PNG top-dependents panel grid === */
    .panel-grid {{
      display: grid;
      gap: 10px;
      margin-top: 14px;
    }}

    .panel-grid-cols-4 {{
      grid-template-columns: repeat(4, minmax(0, 1fr));
    }}

    .panel-grid .figure-card {{
      min-width: 0;
      min-height: 0;
      padding: 8px;
    }}

    .panel-grid .figure-label {{ display: none; }}
    .panel-grid .figure-text  {{ display: none; }}

    .panel-grid .figure-title {{
      font-size: 12px;
      margin-bottom: 6px;
    }}

    .panel-grid .figure-img {{
      width: 100%;
      height: auto;
      max-height: 180px;   /* KEY: shrink image to fit screen */
      object-fit: contain;
    }}

    /* Responsive fallback (still no scroll inside panels) */
    @media (max-width: 1100px) {{
      .panel-grid-cols-4 {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
    }}
    @media (max-width: 640px) {{
      .panel-grid-cols-4 {{ grid-template-columns: repeat(1, minmax(0, 1fr)); }}
    }}
  </style>

  <script>
    function filterTable(tableId, query) {{
      const filter = (query || "").toLowerCase();
      const table = document.getElementById(tableId);
      if (!table) return;
      const rows = table.getElementsByTagName("tr");
      for (let i = 1; i < rows.length; i++) {{
        const cells = rows[i].getElementsByTagName("td");
        let show = false;
        for (let j = 0; j < cells.length; j++) {{
          if (cells[j].innerText.toLowerCase().indexOf(filter) !== -1) {{
            show = true;
            break;
          }}
        }}
        rows[i].style.display = show ? "" : "none";
      }}
    }}

    const sortState = {{}};

    function parseCellValue(text) {{
      const t = (text || "").trim();
      const num = parseFloat(t.replace(/[^0-9eE\\-\\.]/g, ""));
      if (!isNaN(num) && /[0-9]/.test(t)) return num;
      return t.toLowerCase();
    }}

    function sortTable(tableId, colIndex) {{
      const table = document.getElementById(tableId);
      if (!table) return;
      const tbody = table.tBodies[0];
      const rows = Array.from(tbody.rows);

      const key = tableId + ":" + colIndex;
      const current = sortState[key] || "desc";
      const next = current === "asc" ? "desc" : "asc";
      sortState[key] = next;

      rows.sort((a, b) => {{
        const va = parseCellValue(a.cells[colIndex].innerText);
        const vb = parseCellValue(b.cells[colIndex].innerText);
        if (va < vb) return next === "asc" ? -1 : 1;
        if (va > vb) return next === "asc" ? 1 : -1;
        return 0;
      }});

      for (const r of rows) tbody.appendChild(r);
    }}
  </script>
</head>

<body>
  <div class="page">
    <div class="report-card">
      <div class="hero">
        <div class="timestamp">Generated: {generated_at}</div>
        <div class="hero-row">
          <div>{left_logo_html}</div>
          <div class="hero-center">
            <div class="hero-eyebrow">AYASS BIOSCIENCE · FUNCTIONAL GENOMICS</div>
            <h1 class="hero-title">CRISPR Effects and gRNA Analysis</h1>
            <div class="hero-subtitle">
              DepMap CRISPR-Cas9 dependencies and codependency modules contextualised for disease-focused biomarker and target discovery.
            </div>
          </div>
          <div class="hero-right">
            <div class="hero-pill">Disease context&nbsp;&nbsp;{disease_context}</div>
            <div class="hero-pill">Platform&nbsp;&nbsp;{depmap_platform}</div>
            <div>{right_logo_html}</div>
          </div>
        </div>
      </div>

      <div class="pill-row">
        <div class="pill">Patient DEGs &amp; DepMap integration</div>
        <div class="pill">Chronos dependency &amp; sgRNA profiling</div>
        <div class="pill">Gene–gene codependency modules</div>
        <div class="pill">Perturbation similarity clustering</div>
        <div class="pill">ACE causality &amp; therapeutic alignment</div>
      </div>

      {section_overview}
      {section_ge}
      {section_sgrna}
      {section_codep}
      {section_pert}
      {section_ace}

      <div class="footer">
        <div>Report generated by the Ayass Bioscience functional genomics pipeline.</div>
        <div class="references">
          Reference for DepMap CRISPR screening and Chronos dependency modelling:
          <a href="https://pmc.ncbi.nlm.nih.gov/articles/PMC5667678/" target="_blank" rel="noopener">
            Meyers et al., Nat Genet 2017
          </a>.
        </div>
      </div>
    </div>
  </div>
</body>
</html>
"""


def strip_unwanted_panels(html_text: str) -> str:
    """
    Remove specific unwanted figure/panel blocks from the final HTML report.
    Targets figure cards by their HTML id attributes.
    """
    remove_ids = [
        "GENES_DEPENDENCIES_GENEOVERVIEW_GRID",
    ]

    for rid in remove_ids:
        html_text = re.sub(
            rf'<div class="figure-card" id="{re.escape(rid)}">.*?</div>\s*',
            "",
            html_text,
            flags=re.DOTALL,
        )

    return html_text


def build_html(
    depmap_run: Path,
    finaltesting_root: Path,
    ayass_logo_left: Optional[Path],
    ayass_logo_right: Optional[Path],
    disease: str,
) -> str:
    dep_dep = depmap_run / "DepMap_Dependencies"
    dep_guides = depmap_run / "DepMap_GuideAnalysis"
    code_dir = dep_guides / "Codependency"
    dep_causality = depmap_run / "DepMap_Causality"

    ge_df = maybe_read_csv(dep_dep / "GeneEssentiality_ByMedian.csv")
    guides_df = maybe_read_csv(dep_guides / "CRISPR_GuideLevel_Avana_SelectedModels_long.csv")
    code_pairs_df = maybe_read_csv(code_dir / "Top_CoDependent_GenePairs_ALL_Methods.csv")
    pert_df = maybe_read_csv(dep_guides / "PerturbationSimilarity_Chronos_Clusters.csv")

    model_cfg = parse_model_selection(finaltesting_root)
    model_cfg = merge_model_cfg_with_guides(model_cfg, guides_df)

    def logo_tag(path: Optional[Path]) -> str:
        if not path:
            return ""
        uri = encode_image_to_data_uri(path)
        if not uri:
            return ""
        return f'<img src="{uri}" alt="Ayass Bioscience logo" class="ayass-logo">'

    left_logo_html = logo_tag(ayass_logo_left)
    right_logo_html = logo_tag(ayass_logo_right)

    disease_context = disease
    depmap_platform = "DepMap CRISPR functional genomics"
    generated_at = dt.datetime.now().strftime("%Y-%m-%d · %H:%M")

    section_overview = build_module_overview_section(ge_df, guides_df, pert_df, model_cfg)
    section_ge = build_gene_essentiality_section(dep_dep, ge_df)
    section_sgrna = build_sgrna_section(dep_guides, guides_df)
    section_codep = build_codependency_section(code_dir, code_pairs_df)
    section_pert = build_perturbation_section(dep_guides, pert_df, code_pairs_df)
    section_ace = build_ace_causality_section(dep_causality)

    html_text = HTML_TEMPLATE.format(
        generated_at=html.escape(generated_at),
        left_logo_html=left_logo_html,
        right_logo_html=right_logo_html,
        disease_context=html.escape(disease_context),
        depmap_platform=html.escape(depmap_platform),
        section_overview=section_overview,
        section_ge=section_ge,
        section_sgrna=section_sgrna,
        section_codep=section_codep,
        section_pert=section_pert,
        section_ace=section_ace,
    )

    # Optional: if you want stripping applied, uncomment next line
    # html_text = strip_unwanted_panels(html_text)

    return html_text


# -------------------------------------------------------------------
# CLI + Entry
# -------------------------------------------------------------------

def generate_depmap_report(depmap_root: Path, out_path: Path, ayass_logo_left: Optional[Path], ayass_logo_right: Optional[Path], disease: str) -> None:
    finaltesting_root = depmap_root
    depmap_run = detect_depmap_run(depmap_root)

    html_text = build_html(
        depmap_run=depmap_run,
        finaltesting_root=finaltesting_root,
        ayass_logo_left=ayass_logo_left,
        ayass_logo_right=ayass_logo_right,
        disease=disease
    )

    out_path.write_text(html_text, encoding="utf-8")
    return out_path


# if __name__ == "__main__":
#     output_dir = Path(r"C:\Ayass Bio Work\Agentic_AI_ABS\perturbation_pipeline\Output_Perturbation_Pipeline\depmap")
#     generate_depmap_report(
#         depmap_root=output_dir,
#         out_path=output_dir / "DepMap_Report.html",
#         ayass_logo_left=Path(r"C:\Ayass Bio Work\Agentic_AI_ABS\perturbation_pipeline\perturbation\logos\Ayass_logo_left.png"),
#         ayass_logo_right=Path(r"C:\Ayass Bio Work\Agentic_AI_ABS\perturbation_pipeline\perturbation\logos\Ayass_logo_right.png"),
#         disease="Lupus",
#     )