#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ayass Bioscience — Cross-Disease Report (CORE)
==============================================

File: core_report.py  (CODE 1 / 3)

What this file does (core orchestrator):
- Discovers diseases under OUT_ROOT
- Loads/normalizes pathway stats → themes, shared pathways, discordance
- Loads similarity (prefer existing image; else build a heatmap from matrix)
- Loads pathway activity tables and selects:
    - default case studies = top 3 pathways
    - selectable pool = top 50 pathways (dropdown in HTML)
- Loads entities + builds CASE_DATA via report_entities.py (CODE 2)
- Builds narrative (optional LLM)
- Delegates HTML rendering to report_render.py (CODE 3)
- Writes:
    OUT_ROOT/Report/index.html
    OUT_ROOT/Report/report_artifact.json
    OUT_ROOT/Report/Report_<cohort>_<timestamp>.pdf

Run:
  python core_report.py --out-root "/mnt/d/temp/A_IPAA4" --q-cutoff 0.05
  python core_report.py --out-root "/mnt/d/temp/A_IPAA4" --q-cutoff 0.05 --no-llm

Notes:
- CODE 2 file name must be: report_entities.py
- CODE 3 file name must be: report_render.py

This file is robust and defensive:
- Works with many pathway_stats column variants
- Works when similarity exists only as an image or as tables
- Works when pathway_activity.tsv is missing (fallback to stats t)
- Always produces HTML + artifact JSON; PDF best-effort
"""

from __future__ import annotations

import os
import re
import sys
import json
import time
import base64
import shutil
import argparse
import textwrap
import subprocess
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any
from datetime import datetime

import numpy as np
import pandas as pd
from openai import OpenAI

from weasyprint import HTML as WPHTML  # type: ignore


from dotenv import load_dotenv  # type: ignore
_HAVE_DOTENV = True



# ======================
# === Logging
# ======================

def _ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log_info(msg: str) -> None:
    print(f"[INFO {_ts()}] {msg}")


def log_warn(msg: str) -> None:
    print(f"[WARN {_ts()}] {msg}")


def log_error(msg: str) -> None:
    print(f"[ERROR {_ts()}] {msg}")


# ======================
# === IO utils
# ======================

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def clean_name(s: Any) -> str:
    return re.sub(r"\s+", " ", str(s or "").strip())


def safe_num(x: Any, default: float = np.nan) -> float:
    try:
        v = float(x)
        if np.isfinite(v):
            return v
    except Exception:
        pass
    return default


def read_json(path: Optional[Path]) -> Optional[dict]:
    if path is None or not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        log_warn(f"Failed to read JSON {path}: {e}")
        return None


def read_table(path: Optional[Path]) -> Optional[pd.DataFrame]:
    if path is None or not path.exists():
        return None
    try:
        suf = path.suffix.lower()
        if suf in {".tsv", ".txt"}:
            return pd.read_csv(path, sep="\t")
        if suf == ".csv":
            return pd.read_csv(path)
        if suf == ".xlsx":
            return pd.read_excel(path)
        if suf == ".json":
            obj = read_json(path)
            if obj is None:
                return None
            if isinstance(obj, dict):
                return pd.json_normalize(obj)
            return pd.DataFrame(obj)
    except pd.errors.EmptyDataError:
        log_warn(f"Failed to read {path}: empty file.")
        return None
    except Exception as e:
        log_warn(f"Failed to read {path}: {e}")
        return None
    return None


def detect_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    if df is None or df.empty:
        return None
    cols = list(df.columns)
    low = {str(c).lower(): str(c) for c in cols}
    for cand in candidates:
        if cand.lower() in low:
            return low[cand.lower()]
    for cand in candidates:
        for c in cols:
            if cand.lower() in str(c).lower():
                return str(c)
    return None


# =========================
# === Filters (pathways/entities)
# =========================

OLF_SENSE_PAT = re.compile(
    r"(olfactory|odorant|odour|smell|taste|phototransduction|sensory\s+perception|sensory\s+system)",
    flags=re.IGNORECASE,
)
NON_HUMAN_PAT = re.compile(r"\b(mouse|mice|rat|mm9|mm10|murine|c2c12)\b", flags=re.IGNORECASE)


def is_allowed_pathway(name: str) -> bool:
    s = str(name or "")
    return not bool(OLF_SENSE_PAT.search(s))


def is_human_entity(name: str) -> bool:
    if not name:
        return False
    return NON_HUMAN_PAT.search(str(name)) is None


# =========================
# === OUT_ROOT discovery (diseases)
# =========================

def _is_noise_dir(name: str) -> bool:
    return name.startswith(("compare", "Report", "INSIGHTS_out", "ANALYSIS_", "baseline_", "_", ".", "__", "jsons_all", "results"))


def discover_disease_dirs(out_root: Path) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    for p in sorted(out_root.glob("*")):
        if not p.is_dir():
            continue
        if _is_noise_dir(p.name):
            continue
        # IPAA: disease dirs typically contain pathway_activity.tsv OR pathway_stats*.tsv
        if (p / "pathway_activity.tsv").exists():
            out[p.name] = p
            continue
        for fn in [
            "pathway_stats_with_baseline_filtered_classified.tsv",
            "pathway_stats_with_baseline_filtered.tsv",
            "pathway_stats_with_baseline_classified.tsv",
            "pathway_stats_with_baseline.tsv",
            "pathway_stats.tsv",
        ]:
            if (p / fn).exists():
                out[p.name] = p
                break
    return out


def pick_pathway_stats(ddir: Path) -> Optional[Path]:
    for fn in [
        "pathway_stats_with_baseline_filtered_classified.tsv",
        "pathway_stats_with_baseline_filtered.tsv",
        "pathway_stats_with_baseline_classified.tsv",
        "pathway_stats_with_baseline.tsv",
        "pathway_stats.tsv",
    ]:
        p = ddir / fn
        if p.exists():
            return p
    return None


# =========================
# === Similarity (prefer images, else tables)
# =========================

def _file_to_data_uri(img_path: Path) -> Optional[str]:
    if img_path is None or not img_path.exists():
        return None
    ext = img_path.suffix.lower()
    if ext == ".png":
        mime = "image/png"
    elif ext == ".svg":
        mime = "image/svg+xml"
    else:
        return None
    data = img_path.read_bytes()
    return f"data:{mime};base64," + base64.b64encode(data).decode("utf-8")


def similarity_fallback_data_uri(out_root: Path) -> Optional[str]:
    candidates = [
        out_root / "compare" / "pathway_t_spearman.png",
        out_root / "compare" / "pathway_t_spearman.svg",
        out_root / "INSIGHTS_out" / "figures" / "shared" / "shared_disease_similarity.png",
        out_root / "INSIGHTS_out" / "figures" / "shared" / "shared_disease_similarity.svg",
        out_root / "INSIGHTS_out" / "figures" / "shared_disease_similarity.png",
        out_root / "INSIGHTS_out" / "figures" / "shared_disease_similarity.svg",
    ]
    for p in candidates:
        uri = _file_to_data_uri(p)
        if uri:
            return uri
    return None


def load_similarity_table(out_root: Path) -> Optional[pd.DataFrame]:
    candidates = [
        out_root / "compare" / "pathway_t_spearman.tsv",
        out_root / "compare" / "pathway_t_spearman.csv",
        out_root / "compare" / "pathway_t_spearman_matrix.tsv",
        out_root / "compare" / "pathway_t_spearman_matrix.csv",
        out_root / "ANALYSIS_category_landscape" / "tables" / "disease_similarity_cosine.tsv",
    ]
    for p in candidates:
        df = read_table(p)
        if df is not None and not df.empty:
            return df
    return None


def normalize_similarity_to_matrix(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    # already matrix-like
    if df.shape[0] == df.shape[1]:
        m = df.copy()
        if str(m.columns[0]).lower() in {"disease", "name", "cohort"}:
            m = m.set_index(m.columns[0])
        return m

    cols = [str(c).lower() for c in df.columns]
    if {"disease_a", "disease_b", "similarity"}.issubset(set(cols)):
        a = df.columns[cols.index("disease_a")]
        b = df.columns[cols.index("disease_b")]
        s = df.columns[cols.index("similarity")]
        m = df.pivot_table(index=a, columns=b, values=s, aggfunc="mean")
        return m

    m = df.copy()
    m = m.set_index(m.columns[0])
    return m


def top_similarity_pairs(sim_mat: pd.DataFrame, top_n: int = 12) -> pd.DataFrame:
    if sim_mat is None or sim_mat.empty:
        return pd.DataFrame(columns=["Disease A", "Disease B", "Similarity"])

    m = sim_mat.copy()
    m.index = m.index.astype(str)
    m.columns = m.columns.astype(str)
    m = m.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    idx = sorted(set(m.index.tolist()) | set(m.columns.tolist()))
    m = m.reindex(index=idx, columns=idx).fillna(0.0)
    m = (m + m.T) / 2.0
    np.fill_diagonal(m.values, np.nan)

    rows = []
    for i, a in enumerate(idx):
        for j, b in enumerate(idx):
            if j <= i:
                continue
            rows.append((a, b, float(m.loc[a, b])))

    out = pd.DataFrame(rows, columns=["Disease A", "Disease B", "Similarity"])
    out = out.sort_values("Similarity", ascending=False).head(top_n)
    out["Similarity"] = out["Similarity"].map(lambda x: f"{x:.3f}")
    return out


# =========================
# === Pathway stats normalization → themes + shared pathways
# =========================

def normalize_pathway_stats(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["Pathway", "Sub_Class", "Main_Class", "t", "qval", "Direction"])

    pcol = detect_col(df, ["Pathway", "pathway", "Term", "term", "NAME", "name"]) or df.columns[0]
    sub = detect_col(df, ["Sub_Class", "sub_class", "Subclass", "subclass"])
    main = detect_col(df, ["Main_Class", "main_class", "MainClass", "Class", "Category"])
    tcol = detect_col(df, ["t", "t_stat", "tstat", "t-stat", "moderated_t", "score"])
    qcol = detect_col(df, ["FDR", "fdr", "padj", "adj_p", "qval", "q-value", "Adjusted P-value"])
    pval = detect_col(df, ["pval", "p_value", "p-value", "P.Value"])

    out = pd.DataFrame()
    out["Pathway"] = df[pcol].astype(str).map(clean_name)
    out["Sub_Class"] = df[sub].astype(str).map(clean_name) if sub else "Unclassified"
    out["Main_Class"] = df[main].astype(str).map(clean_name) if main else "Unclassified"

    out["t"] = pd.to_numeric(df[tcol], errors="coerce") if tcol else np.nan
    if qcol:
        out["qval"] = pd.to_numeric(df[qcol], errors="coerce")
    elif pval:
        out["qval"] = pd.to_numeric(df[pval], errors="coerce")
    else:
        out["qval"] = np.nan

    out["Direction"] = np.where(out["t"].fillna(0.0) >= 0, "UP", "DOWN")
    out.loc[~out["Direction"].isin(["UP", "DOWN"]), "Direction"] = "UP"

    out.loc[out["Sub_Class"].isin(["", "NA", "NAN", "NONE"]), "Sub_Class"] = "Unclassified"
    out.loc[out["Main_Class"].isin(["", "NA", "NAN", "NONE"]), "Main_Class"] = "Unclassified"

    out = out[out["Pathway"].map(is_allowed_pathway)].copy()
    return out


def compute_shared_subclass_shortlist(
    disease_to_df: Dict[str, pd.DataFrame],
    diseases: List[str],
    q_cutoff: float,
    top_n: int = 30,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    best_rows = []
    theme_presence: Dict[str, Dict[str, int]] = {}

    for disease in diseases:
        gdf = disease_to_df.get(disease)
        if gdf is None or gdf.empty:
            continue

        g = gdf.copy()
        if g["qval"].notna().any():
            g = g[g["qval"] <= q_cutoff].copy()

        g = g[g["Sub_Class"].notna()].copy()
        g.loc[g["Sub_Class"].eq(""), "Sub_Class"] = "Unclassified"

        g["abs_t"] = g["t"].abs()
        g["q_sort"] = g["qval"].fillna(1.0)
        g = g.sort_values(["Sub_Class", "q_sort", "abs_t"], ascending=[True, True, False])

        best = g.groupby("Sub_Class", as_index=False).head(1).copy()
        for _, r in best.iterrows():
            best_rows.append(
                {
                    "Disease": disease,
                    "Sub_Class": r["Sub_Class"],
                    "Main_Class": r["Main_Class"],
                    "Best_Pathway": r["Pathway"],
                    "Direction": r["Direction"],
                    "t": "" if pd.isna(r["t"]) else f"{float(r['t']):.2f}",
                    "qval": "" if pd.isna(r["qval"]) else (
                        f"{float(r['qval']):.2e}" if float(r["qval"]) < 0.001 else f"{float(r['qval']):.3f}"
                    ),
                }
            )

        present = set(best["Sub_Class"].tolist())
        for sub in present:
            rows.append({"Sub_Class": sub, "Disease": disease})
            theme_presence.setdefault(sub, {})[disease] = 1

    if not rows:
        return pd.DataFrame(), pd.DataFrame(best_rows), pd.DataFrame()

    pres = pd.DataFrame(rows).drop_duplicates()
    counts = pres.groupby("Sub_Class")["Disease"].nunique().reset_index(name="n_diseases")
    counts = counts.sort_values(["n_diseases", "Sub_Class"], ascending=[False, True]).head(top_n)

    best_long = pd.DataFrame(best_rows)
    if not best_long.empty:
        top_subs = set(counts["Sub_Class"].tolist())
        best_long = best_long[best_long["Sub_Class"].isin(top_subs)].copy()
        best_long = best_long[best_long["Best_Pathway"].map(is_allowed_pathway)].copy()

    top_themes = counts["Sub_Class"].tolist()
    mat = pd.DataFrame(index=top_themes, columns=diseases, data=0)
    for th in top_themes:
        for d in diseases:
            mat.loc[th, d] = int(theme_presence.get(th, {}).get(d, 0))
    return counts, best_long, mat


def compute_shared_pathways_table(
    disease_to_df: Dict[str, pd.DataFrame],
    diseases: List[str],
    q_cutoff: float,
    top_n: int = 25,
) -> pd.DataFrame:
    counts: Dict[str, int] = {}
    best_q: Dict[str, float] = {}

    for d in diseases:
        df = disease_to_df.get(d)
        if df is None or df.empty:
            continue

        g = df.copy()
        if g["qval"].notna().any():
            g = g[g["qval"] <= q_cutoff].copy()
        g = g[g["Pathway"].map(is_allowed_pathway)].copy()

        pw = set(g["Pathway"].astype(str).tolist())
        for p in pw:
            counts[p] = counts.get(p, 0) + 1

        for _, r in g.iterrows():
            p = str(r["Pathway"])
            q = safe_num(r["qval"], default=np.nan)
            if np.isfinite(q):
                if p not in best_q or q < best_q[p]:
                    best_q[p] = q

    if not counts:
        return pd.DataFrame({"Pathway": [], "#Diseases": [], "Best q/FDR": []})

    rows = [{"Pathway": p, "#Diseases": int(n), "Best q/FDR": best_q.get(p, np.nan)} for p, n in counts.items()]
    out = pd.DataFrame(rows)
    out = out[out["Pathway"].map(is_allowed_pathway)].copy()
    out["Best q/FDR"] = pd.to_numeric(out["Best q/FDR"], errors="coerce")
    out = out.sort_values(["#Diseases", "Best q/FDR"], ascending=[False, True]).head(top_n)
    out["Best q/FDR"] = out["Best q/FDR"].map(lambda x: "" if pd.isna(x) else (f"{x:.2e}" if x < 0.001 else f"{x:.3f}"))
    return out


def load_directional_concordance(out_root: Path) -> Optional[pd.DataFrame]:
    candidates = [
        out_root / "compare" / "directional_concordance_by_pathway.csv",
        out_root / "compare" / "directional_concordance_by_pathway.tsv",
        out_root / "INSIGHTS_out" / "tables" / "directional_concordance_by_pathway.csv",
    ]
    for p in candidates:
        df = read_table(p)
        if df is not None and not df.empty:
            return df
    return None


def compute_discordance_table(
    disease_to_df: Dict[str, pd.DataFrame],
    diseases: List[str],
    q_cutoff: float,
    top_n: int = 12,
) -> pd.DataFrame:
    per_path_dirs: Dict[str, List[str]] = {}

    for d in diseases:
        df = disease_to_df.get(d)
        if df is None or df.empty:
            continue

        g = df.copy()
        if g["qval"].notna().any():
            g = g[g["qval"] <= q_cutoff].copy()

        for _, r in g.iterrows():
            pw = str(r["Pathway"])
            if not is_allowed_pathway(pw):
                continue
            dirn = str(r["Direction"])
            per_path_dirs.setdefault(pw, []).append(dirn)

    rows = []
    for pw, dirs in per_path_dirs.items():
        up = sum(1 for x in dirs if x == "UP")
        dn = sum(1 for x in dirs if x == "DOWN")
        if up + dn < 2:
            continue
        rows.append({"Pathway": pw, "Discordance": int(up * dn), "UP": int(up), "DOWN": int(dn), "#Diseases": int(up + dn)})

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows)
    out = out.sort_values(["Discordance", "#Diseases"], ascending=[False, False]).head(top_n)
    return out


# =========================
# === Pathway activity (case studies)
# =========================

def load_pathway_activity_df(disease_dir: Path, fallback_stats: pd.DataFrame) -> pd.DataFrame:
    """
    Prefers <Disease>/pathway_activity.tsv.
    Falls back to stats 't' as differential_score.
    """
    act_path = disease_dir / "pathway_activity.tsv"
    df = read_table(act_path) if act_path.exists() else None

    if df is not None and not df.empty:
        pcol = detect_col(df, ["Pathway", "pathway", "Term", "term", "NAME", "name"])
        if pcol and pcol != "Pathway":
            df = df.rename(columns={pcol: "Pathway"})
        if "Pathway" not in df.columns:
            df = df.rename(columns={df.columns[0]: "Pathway"})
        df["Pathway"] = df["Pathway"].astype(str).map(clean_name)
        df = df[df["Pathway"].map(is_allowed_pathway)].copy()
        return df

    if fallback_stats is None or fallback_stats.empty or "Pathway" not in fallback_stats.columns:
        return pd.DataFrame(columns=["Pathway", "differential_score"])

    out = fallback_stats[["Pathway", "t"]].copy()
    out = out.rename(columns={"t": "differential_score"})
    out["Pathway"] = out["Pathway"].astype(str).map(clean_name)
    out["differential_score"] = pd.to_numeric(out["differential_score"], errors="coerce")
    out = out.dropna(subset=["differential_score"])
    out = out[out["Pathway"].map(is_allowed_pathway)].copy()
    return out


def pick_best_activity_column(all_activity_dfs: Dict[str, pd.DataFrame]) -> Tuple[str, str]:
    """
    Choose best activity score column across diseases.
    Returns (kind, colname) where kind ∈ {"whole","differential"}.
    """
    candidates = [
        ("whole", ["whole_score", "whole", "whole_activity", "activity_score", "score", "WholeScore"]),
        ("differential", ["differential_score", "differential", "delta_score", "t", "DifferentialScore"]),
    ]

    stats: List[Tuple[int, float, int, str, str]] = []
    for kind, cols in candidates:
        for col in cols:
            vals = []
            for _d, df in all_activity_dfs.items():
                if df is None or df.empty:
                    continue
                if col in df.columns:
                    v = pd.to_numeric(df[col], errors="coerce").dropna()
                    if not v.empty:
                        vals.extend(v.values.tolist())
            if vals:
                arr = np.array(vals, dtype=float)
                cov = int(np.isfinite(arr).sum())
                var = float(np.nanvar(arr))
                stats.append((cov, var, 0 if kind == "whole" else 1, kind, col))

    if not stats:
        return ("differential", "differential_score")

    stats.sort(key=lambda x: (x[0], x[1], -x[2]), reverse=True)
    _cov, _var, _pref, kind, col = stats[0]
    return (kind, col)


def top_pathways_pool_by_activity(all_activity_dfs: Dict[str, pd.DataFrame], limit: int = 50) -> Tuple[List[str], Dict[str, str]]:
    """
    Builds a selectable pool (top N pathways) based on aggregation:
      sum_diseases(abs(score)) across diseases where present.
    Returns (pool, meta).
    """
    kind, col = pick_best_activity_column(all_activity_dfs)

    agg: Dict[str, float] = {}
    for _d, df in all_activity_dfs.items():
        if df is None or df.empty or "Pathway" not in df.columns:
            continue
        if col not in df.columns:
            continue
        tmp = df[["Pathway", col]].copy()
        tmp[col] = pd.to_numeric(tmp[col], errors="coerce")
        tmp = tmp.dropna(subset=[col])
        for _, r in tmp.iterrows():
            pw = clean_name(r["Pathway"])
            if not is_allowed_pathway(pw):
                continue
            sc = float(r[col])
            agg[pw] = agg.get(pw, 0.0) + abs(sc)

    if not agg:
        return ([], {"score_kind": kind, "score_col": col})

    ranked = sorted(agg.items(), key=lambda x: x[1], reverse=True)
    pool = []
    seen = set()
    for pw, _v in ranked:
        if pw in seen:
            continue
        seen.add(pw)
        pool.append(pw)
        if len(pool) >= int(limit):
            break

    return (pool, {"score_kind": kind, "score_col": col})


# =========================
# === PDF export
# =========================

def _find_browser_exe() -> Optional[str]:
    # Windows explicit paths + linux which()
    candidates: List[Optional[str]] = []
    if sys.platform.startswith("win"):
        candidates += [
            r"C:\Program Files\Google\Chrome\Application\chrome.exe",
            r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
            r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
            r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        ]
    else:
        candidates += [
            shutil.which("google-chrome"),
            shutil.which("google-chrome-stable"),
            shutil.which("chrome"),
            shutil.which("chromium"),
            shutil.which("chromium-browser"),
            shutil.which("msedge"),
        ]
    for c in candidates:
        if c and os.path.exists(c):
            return c
    return None


def write_pdf(html_path: Path, pdf_path: Path) -> bool:
    """
    Best-effort PDF.
    Prefer Chrome/Edge headless (runs JS), fallback to WeasyPrint (may not run JS).
    """
    browser = _find_browser_exe()
    if browser:
        try:
            file_url = html_path.resolve().as_uri()
            cmd = [
                browser,
                "--headless=new",
                "--disable-gpu",
                "--disable-print-preview",
                "--no-sandbox",
                "--disable-dev-shm-usage",
                "--virtual-time-budget=7000",  # allow JS to populate
                f"--print-to-pdf={str(pdf_path)}",
                file_url,
            ]
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            return pdf_path.exists() and pdf_path.stat().st_size > 10_000
        except Exception as e:
            log_warn(f"Browser PDF export failed: {e}")

    try:
        WPHTML(filename=str(html_path)).write_pdf(str(pdf_path))
        return pdf_path.exists() and pdf_path.stat().st_size > 10_000
    except Exception as e:
        log_warn("PDF export failed. Install Chrome/Chromium/Edge OR `pip install weasyprint`.")
        log_warn(f"WeasyPrint error: {e}")
        return False


# ======================
# === OpenAI (optional narrative)
# ======================

def _require_openai():
    if _HAVE_DOTENV and load_dotenv:
        load_dotenv()

    key = os.getenv("OPENAI_API_KEY", "").strip()
    if not key:
        raise RuntimeError("OPENAI_API_KEY missing (LLM narrative disabled).")

    
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    client = OpenAI(api_key=key, timeout=60.0)
    return client, model


_client, _model = None, None


def _client_model():
    global _client, _model
    if _client is None:
        _client, _model = _require_openai()
    return _client, _model


def _extract_choice_text(resp) -> str:
    try:
        return (resp.choices[0].message.content or "").strip()
    except Exception:
        return ""


def _try_parse_json(text: str) -> dict:
    if not text:
        return {}
    if "```" in text:
        m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
        if m:
            text = m.group(1).strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    try:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(text[start:end + 1])
    except Exception:
        pass
    return {}


def llm_json(
    system: str,
    prompt: str,
    *,
    max_tokens: int = 950,
    retries: int = 3,
    repair_retries: int = 2,
) -> dict:
    """
    Robust LLM->JSON:
      - tries response_format=json_object if supported
      - retries + repair
      - NEVER raises for JSON formatting issues
      - returns {} on total failure
    """
    client, model = _client_model()

    def _call(user_prompt: str, enforce_json: bool = True):
        kwargs = dict(
            model=model,
            temperature=0.0,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system.strip()},
                {"role": "user", "content": user_prompt.strip()},
            ],
        )
        if enforce_json:
            kwargs["response_format"] = {"type": "json_object"}
        return client.chat.completions.create(**kwargs)

    last_text = ""
    last_err = ""
    for attempt in range(max(1, retries)):
        try:
            resp = _call(prompt + "\n\nReturn ONLY valid JSON.", enforce_json=True)
            txt = _extract_choice_text(resp)
            last_text = txt
            parsed = _try_parse_json(txt)
            if isinstance(parsed, dict) and parsed:
                return parsed
        except Exception as e:
            last_err = str(e)

        try:
            resp = _call(prompt + "\n\nReturn ONLY valid JSON.", enforce_json=False)
            txt = _extract_choice_text(resp)
            last_text = txt
            parsed = _try_parse_json(txt)
            if isinstance(parsed, dict) and parsed:
                return parsed
        except Exception as e:
            last_err = str(e)

        time.sleep(0.6 * (attempt + 1))

    bad = (last_text or "")[:12000]
    repair_prompt = (
        "You will be given text intended to be a JSON object but it is invalid.\n"
        "Return ONLY a corrected JSON object.\n\n"
        "INVALID TEXT:\n"
        f"{bad}\n"
    )
    for attempt in range(max(0, repair_retries)):
        try:
            resp = _call(repair_prompt, enforce_json=True)
            txt = _extract_choice_text(resp)
            parsed = _try_parse_json(txt)
            if isinstance(parsed, dict) and parsed:
                return parsed
        except Exception:
            pass
        time.sleep(0.6 * (attempt + 1))

    log_warn("LLM JSON failed after retries+repair. Using deterministic defaults.")
    if last_err:
        log_warn(f"Last LLM error: {last_err}")
    if last_text:
        log_warn("Last LLM text (first 240 chars): " + last_text[:240].replace("\n", " "))
    return {}


# =========================
# === Narrative defaults (NEVER EMPTY)
# =========================

def ensure_report_narrative_defaults(j: dict, *, diseases: List[str], q_cutoff: float) -> dict:
    if not isinstance(j, dict):
        j = {}
    required = [
        "hook_headline", "hook_bullets", "sim_bullets", "subclass_story_bullets",
        "shared_pathways_bullets", "discord_bullets", "engines_bullets", "takeaways", "conclusion"
    ]
    for k in required:
        if k not in j:
            j[k] = [] if k.endswith("_bullets") or k == "takeaways" else ""

    if not str(j.get("hook_headline", "")).strip():
        j["hook_headline"] = f"Evidence-constrained cross-disease report (q ≤ {q_cutoff:.2g})."

    if not j.get("hook_bullets"):
        j["hook_bullets"] = [
            f"Shared programs are computed from significant pathways (q ≤ {q_cutoff:.2g}) across {len(diseases)} diseases.",
            "Similarity is pathway-level proximity (association-only), not diagnosis overlap or causality.",
            "Driver layers (TF / epigenetic / metabolite) are hypotheses requiring replication + orthogonal validation.",
        ]

    if not j.get("sim_bullets"):
        j["sim_bullets"] = [
            "Higher similarity suggests overlapping pathway programs (association-only).",
            "Lower similarity suggests distinct pathway signatures or different upstream regulation.",
        ]

    if not j.get("subclass_story_bullets"):
        j["subclass_story_bullets"] = [
            "Theme bullets are grounded in the evidence table (representative pathways per disease).",
        ]

    if not j.get("shared_pathways_bullets"):
        j["shared_pathways_bullets"] = [
            "The pathway atlas highlights recurrent pathways across diseases under the cutoff (association).",
        ]

    if not j.get("discord_bullets"):
        j["discord_bullets"] = [
            "Divergence axes (direction flips) suggest candidate stratification or disease-separating biology.",
        ]

    if not j.get("engines_bullets"):
        j["engines_bullets"] = [
            "Driver layers prioritize upstream programs but do not prove mechanism; validate independently.",
        ]

    if not j.get("takeaways"):
        j["takeaways"] = [
            "Replicate shared programs/pathways in independent cohorts.",
            "Use divergence axes to separate diseases by mechanism (candidate stratification).",
            "Validate drivers with orthogonal assays (proteomics/metabolomics/TF activity).",
        ]

    if not str(j.get("conclusion", "")).strip():
        j["conclusion"] = (
            "This report summarizes shared and divergent pathway biology across diseases "
            "using evidence-constrained pathway programs and driver layers."
        )
    return j


def _list_clean(x, n: int) -> List[str]:
    xs = [clean_name(v) for v in (x or []) if clean_name(v)]
    return xs[:n]


# =========================
# === Imports for CODE 2 / CODE 3
# =========================

def _import_entities_module():
    try:
        import report_entities as ent  # type: ignore
        return ent
    except Exception as e:
        raise RuntimeError(
            "Missing CODE 2: report_entities.py.\n"
            f"Import error: {e}"
        ) from e


def _import_ui_module():
    try:
        import report_render as ui  # type: ignore
        return ui
    except Exception as e:
        raise RuntimeError(
            "Missing CODE 3: report_render.py.\n"
            f"Import error: {e}"
        ) from e


# =========================
# === Main builder
# =========================

def _safe_make_similarity_heatmap(ui, sim_mat: pd.DataFrame, title: str) -> str:
    """
    CODE 3 may or may not implement make_similarity_heatmap_from_matrix.
    If absent, return empty string.
    """
    try:
        fn = getattr(ui, "make_similarity_heatmap_from_matrix", None)
        if callable(fn):
            return fn(sim_mat, title)
    except Exception as e:
        log_warn(f"Similarity heatmap generation failed: {e}")
    return ""


def _safe_render_html(ui, context: dict) -> str:
    """
    CODE 3 render_report_html might accept:
      - dict context (recommended), OR
      - RenderInputs dataclass
    We support both without requiring edits here.
    """
    try:
        return ui.render_report_html(context)
    except TypeError:
        pass
    except Exception as e:
        log_warn(f"render_report_html(context) failed: {e}")

    # Fallback: build RenderInputs if it exists
    try:
        RenderInputs = getattr(ui, "RenderInputs")
        if RenderInputs is None:
            raise AttributeError("RenderInputs missing")

        # minimal mapping; your updated CODE 3 should accept dict anyway,
        # but we keep this to be safe.
        diseases = context.get("diseases", []) or []
        top_pathways = (context.get("case_pool_pathways", []) or [])[:50]
        default_case = (context.get("case_default_pathways", []) or [])[:3]
        ri = RenderInputs(
            title=f"Ayass Bioscience — Cross-Disease Report ({context.get('cohort_name','')})",
            subtitle=f"Generated: {context.get('generated','')}",
            diseases=diseases,
            q_cutoff=float(context.get("q_cutoff", 0.05)),
            top_pathways=top_pathways,
            default_case_pathways=default_case,
            case_data=context.get("case_data", {}) or {},
            shared_themes_table_html="",
            shared_pathways_table_html="",
            discordance_table_html="",
            key_takeaways=context.get("takeaways", []) or [],
            validation_checklist=[],
            methods_snapshot="",
            pdf_name="",
            report_artifact={},
        )
        return ui.render_report_html(ri)
    except Exception as e:
        raise RuntimeError(f"Unable to render report HTML via CODE 3. Error: {e}") from e


def build_report(
    out_root: Path,
    no_llm: bool = False,
    q_cutoff: float = 0.05,
    pool_limit: int = 50,
) -> None:
    out_root = out_root.resolve()
    if not out_root.exists() or not out_root.is_dir():
        raise RuntimeError(f"--out-root does not exist or is not a directory: {out_root}")

    cohort_name = out_root.name
    outdir = out_root / "Report"
    ensure_dir(outdir)

    ent = _import_entities_module()
    ui = _import_ui_module()

    # --- disease discovery
    disease_dirs = discover_disease_dirs(out_root)
    diseases = sorted(disease_dirs.keys())
    if not diseases:
        raise RuntimeError(
            "No diseases found.\n"
            "Expected OUT_ROOT/<Disease>/ containing pathway_activity.tsv or pathway_stats*.tsv"
        )

    log_info(f"Detected {len(diseases)} diseases: {', '.join(diseases)}")

    # --- per-disease pathway stats
    disease_to_stats: Dict[str, pd.DataFrame] = {}
    for d in diseases:
        p = pick_pathway_stats(disease_dirs[d])
        raw = read_table(p) if p else None
        if raw is None:
            raw = pd.DataFrame()
        disease_to_stats[d] = normalize_pathway_stats(raw)

    # --- shared themes + evidence + presence heatmap matrix
    subclass_shortlist, subclass_best_long, theme_presence_mat = compute_shared_subclass_shortlist(
        disease_to_stats, diseases, q_cutoff=q_cutoff, top_n=30
    )

    if subclass_shortlist is None or subclass_shortlist.empty:
        subclass_counts = pd.Series({"Unclassified": 1})
        kpi_top_subclass = "NA"
        kpi_shared_subclasses = 0
        theme_chips = []
    else:
        subclass_counts = pd.Series(
            data=subclass_shortlist["n_diseases"].values,
            index=subclass_shortlist["Sub_Class"].values
        )
        kpi_top_subclass = str(subclass_shortlist.iloc[0]["Sub_Class"])
        kpi_shared_subclasses = int((subclass_shortlist["n_diseases"] >= 2).sum())
        theme_chips = subclass_shortlist["Sub_Class"].head(10).tolist()

    theme_evidence_df = subclass_best_long.copy() if subclass_best_long is not None else pd.DataFrame()
    if theme_evidence_df is None or theme_evidence_df.empty:
        theme_evidence_df = pd.DataFrame({"Sub_Class": [], "Disease": [], "Best_Pathway": [], "Direction": [], "t": [], "qval": []})

    # --- similarity (prefer existing image)
    sim_uri = similarity_fallback_data_uri(out_root)
    if sim_uri:
        sim_heatmap_uri = sim_uri
        sim_fallback_used = True
        sim_pairs = pd.DataFrame(columns=["Disease A", "Disease B", "Similarity"])
        sim_mat = pd.DataFrame()
    else:
        sim_df = load_similarity_table(out_root)
        sim_mat = normalize_similarity_to_matrix(sim_df) if sim_df is not None else pd.DataFrame()
        sim_heatmap_uri = _safe_make_similarity_heatmap(ui, sim_mat, "Disease similarity")
        sim_pairs = top_similarity_pairs(sim_mat, top_n=12)
        sim_fallback_used = False

    # --- shared pathways
    shared_pathways_df = compute_shared_pathways_table(disease_to_stats, diseases, q_cutoff=q_cutoff, top_n=25)

    # --- discordance
    direction_concord = load_directional_concordance(out_root)
    if direction_concord is not None and not direction_concord.empty:
        pcol = detect_col(direction_concord, ["Pathway", "pathway", "Term", "term"]) or direction_concord.columns[0]
        dcol = detect_col(direction_concord, ["discord", "flip", "inconsistent", "mixed", "both"])
        if dcol is None:
            upc = detect_col(direction_concord, ["up", "n_up", "UP_count"])
            dnc = detect_col(direction_concord, ["down", "n_down", "DOWN_count"])
            if upc and dnc:
                tmp = direction_concord.copy()
                tmp["Discordance"] = (
                    pd.to_numeric(tmp[upc], errors="coerce").fillna(0)
                    * pd.to_numeric(tmp[dnc], errors="coerce").fillna(0)
                )
                disc_df = pd.DataFrame({"Pathway": tmp[pcol].astype(str).map(clean_name), "Discordance": tmp["Discordance"]})
                disc_df = disc_df[disc_df["Pathway"].map(is_allowed_pathway)].sort_values("Discordance", ascending=False).head(12)
            else:
                disc_df = compute_discordance_table(disease_to_stats, diseases, q_cutoff=q_cutoff, top_n=12)
        else:
            disc_df = pd.DataFrame(
                {
                    "Pathway": direction_concord[pcol].astype(str).map(clean_name),
                    "Discordance": pd.to_numeric(direction_concord[dcol], errors="coerce"),
                }
            ).dropna()
            disc_df = disc_df[disc_df["Pathway"].map(is_allowed_pathway)].sort_values("Discordance", ascending=False).head(12)
    else:
        disc_df = compute_discordance_table(disease_to_stats, diseases, q_cutoff=q_cutoff, top_n=12)

    kpi_top_discord = "NA" if disc_df is None or disc_df.empty else str(disc_df.iloc[0]["Pathway"])

    # =========================
    # === Entities via CODE 2
    # =========================
    disease_to_json, json_sources, fallback_entities = ent.load_entities(out_root, diseases, disease_dirs)

    tf_rank = ent.compute_entity_rankings_from_json(disease_to_json, diseases, "tf", "ANY")
    epi_rank = ent.compute_entity_rankings_from_json(disease_to_json, diseases, "epigenetic", "ANY")
    met_rank = ent.compute_entity_rankings_from_json(disease_to_json, diseases, "metabolite", "ANY")

    # deterministic fill if ranking empty (disease-level fallback)
    for d in diseases:
        if tf_rank.get(d) is None or tf_rank[d].empty:
            fb = (fallback_entities.get(d, {}) or {}).get("tf", [])
            tf_rank[d] = pd.Series({x: 1.0 for x in fb}).sort_values(ascending=False) if fb else pd.Series(dtype=float)
        if epi_rank.get(d) is None or epi_rank[d].empty:
            fb = (fallback_entities.get(d, {}) or {}).get("epigenetic", [])
            epi_rank[d] = pd.Series({x: 1.0 for x in fb}).sort_values(ascending=False) if fb else pd.Series(dtype=float)
        if met_rank.get(d) is None or met_rank[d].empty:
            fb = (fallback_entities.get(d, {}) or {}).get("metabolite", [])
            met_rank[d] = pd.Series({x: 1.0 for x in fb}).sort_values(ascending=False) if fb else pd.Series(dtype=float)

    tf_common = ent.shared_entities_summary(tf_rank, diseases, min_diseases=2, top_n=15)
    epi_common = ent.shared_entities_summary(epi_rank, diseases, min_diseases=2, top_n=15)
    met_common = ent.shared_entities_summary(met_rank, diseases, min_diseases=2, top_n=15)

    # =========================
    # === Case studies: default top 3 + pool top 50
    # =========================
    disease_to_activity_df: Dict[str, pd.DataFrame] = {}
    for d in diseases:
        disease_to_activity_df[d] = load_pathway_activity_df(disease_dirs[d], disease_to_stats.get(d, pd.DataFrame()))

    pool50, activity_meta = top_pathways_pool_by_activity(disease_to_activity_df, limit=int(pool_limit))

    # Fallback if activity yields too few pathways: use shared pathways list
    if len(pool50) < 10 and shared_pathways_df is not None and not shared_pathways_df.empty:
        for p in shared_pathways_df["Pathway"].astype(str).tolist():
            p2 = clean_name(p)
            if is_allowed_pathway(p2) and p2 not in pool50:
                pool50.append(p2)
            if len(pool50) >= int(pool_limit):
                break
        pool50 = pool50[:int(pool_limit)]

    # final fallback: pull top abs(t) per disease and union
    if len(pool50) < 10:
        agg = []
        for d in diseases:
            sdf = disease_to_stats.get(d, pd.DataFrame())
            if sdf is None or sdf.empty:
                continue
            tmp = sdf.copy()
            tmp["abs_t"] = tmp["t"].abs()
            tmp = tmp.sort_values("abs_t", ascending=False).head(50)
            agg.extend(tmp["Pathway"].astype(str).tolist())
        seen = set()
        pool50 = []
        for p in agg:
            p2 = clean_name(p)
            if not p2 or not is_allowed_pathway(p2):
                continue
            if p2 in seen:
                continue
            seen.add(p2)
            pool50.append(p2)
            if len(pool50) >= int(pool_limit):
                break

    default3 = pool50[:3]

    # Build CASE_DATA for all pathways in pool50 (dropdown)
    case_data = {}
    if pool50:
        case_data = ent.build_case_data_for_pathways(
            diseases=diseases,
            disease_to_json=disease_to_json,
            pathways=pool50,
            top_k_per_type=25
        )

    # Sanity warning if still empty (this is the real "why are entities blank" signal)
    if pool50:
        empty_ct = 0
        for pw in default3:
            node = (case_data or {}).get(pw, {})
            # if for ALL diseases, all three layers empty → count
            all_empty = True
            for d in diseases[: min(3, len(diseases))]:
                dn = node.get(d, {}) if isinstance(node, dict) else {}
                if any((dn.get(t) or []) for t in ["tf", "metabolite", "epigenetic"]):
                    all_empty = False
                    break
            if all_empty:
                empty_ct += 1
        if empty_ct > 0:
            log_warn(
                f"CASE_DATA looks empty for {empty_ct}/{len(default3)} default pathways. "
                "This usually means the overlap JSON schema did not normalize into pathway→entities. "
                "Fix is typically in CODE 2 normalization for your overlap json format."
            )

    # -------------------------
    # LLM narrative (optional)
    # -------------------------
    narrative = ensure_report_narrative_defaults({}, diseases=diseases, q_cutoff=q_cutoff)

    if not no_llm:
        try:
            evidence = {
                "diseases": diseases,
                "q_cutoff": q_cutoff,
                "shared_themes_top10": (subclass_shortlist.head(10).to_dict(orient="records") if subclass_shortlist is not None and not subclass_shortlist.empty else []),
                "theme_evidence_sample": (theme_evidence_df.head(15).to_dict(orient="records") if theme_evidence_df is not None and not theme_evidence_df.empty else []),
                "shared_pathways_top12": (shared_pathways_df.head(12).to_dict(orient="records") if shared_pathways_df is not None and not shared_pathways_df.empty else []),
                "divergence_pathways_top12": (disc_df.head(12).to_dict(orient="records") if disc_df is not None and not disc_df.empty else []),
                "case_study_default3": default3,
                "case_study_pool_top50": pool50[:50],
                "activity_score_used": activity_meta,
            }

            llm_out = llm_json(
                system=(
                    "You write a blunt clinician-facing cross-disease narrative.\n"
                    "Hard constraints:\n"
                    "- Association only (no causality claims).\n"
                    "- Use ONLY items in the evidence.\n"
                    "- Do NOT output PMIDs, study IDs, cell lines, or ChIP-Seq metadata.\n"
                    "- Do NOT mention olfactory/sensory/taste/odorant/phototransduction content.\n"
                    "Return ONLY JSON."
                ),
                prompt=textwrap.dedent(f"""
                Write a coherent story based on the evidence.

                Evidence:
                {json.dumps(evidence, indent=2)[:9000]}

                Required JSON:
                {{
                  "hook_headline": "1 sentence; name ≥1 concrete theme/pathway",
                  "hook_bullets": ["3-6 bullets; each bullet names ≥1 evidence item + 'so what'"],
                  "sim_bullets": ["2-5 bullets; what similarity suggests and what it does NOT mean"],
                  "subclass_story_bullets": ["Theme 1: interpret + cite representative pathway", "Theme 2: ...", "Theme 3: ..."],
                  "shared_pathways_bullets": ["3-8 bullets naming pathways + interpretation"],
                  "discord_bullets": ["3-8 bullets naming contrasting pathways + stratification meaning"],
                  "engines_bullets": ["4-10 bullets about upstream programs (hypothesis)"],
                  "takeaways": ["5-10 actionable bullets incl validation"],
                  "conclusion": "2-5 sentences; name ≥3 items; end with validation plan"
                }}
                """),
                max_tokens=950,
            )

            if isinstance(llm_out, dict) and llm_out:
                for k, v in llm_out.items():
                    if v is None:
                        continue
                    if isinstance(v, str) and not v.strip():
                        continue
                    if isinstance(v, list) and len(v) == 0:
                        continue
                    narrative[k] = v
        except Exception as e:
            log_warn(f"LLM narrative skipped due to error: {e}")

    narrative = ensure_report_narrative_defaults(narrative, diseases=diseases, q_cutoff=q_cutoff)

    hook_headline = clean_name(narrative.get("hook_headline", ""))
    hook_bullets = _list_clean(narrative.get("hook_bullets", []), 10)
    sim_bullets = _list_clean(narrative.get("sim_bullets", []), 10)
    subclass_story_bullets = _list_clean(narrative.get("subclass_story_bullets", []), 6)
    shared_pathways_bullets = _list_clean(narrative.get("shared_pathways_bullets", []), 12)
    discord_bullets = _list_clean(narrative.get("discord_bullets", []), 12)
    engines_bullets = _list_clean(narrative.get("engines_bullets", []), 14)
    takeaways = _list_clean(narrative.get("takeaways", []), 14)
    conclusion = clean_name(narrative.get("conclusion", ""))

    # =========================
    # === Render via CODE 3
    # =========================
    context = {
        "cohort_name": cohort_name,
        "generated": _ts(),
        "diseases": diseases,
        "q_cutoff": float(q_cutoff),

        # KPIs
        "kpi_top_subclass": kpi_top_subclass,
        "kpi_shared_subclasses": int(kpi_shared_subclasses),
        "kpi_top_discord": kpi_top_discord,

        # Narrative
        "hook_headline": hook_headline,
        "hook_bullets": hook_bullets,
        "sim_bullets": sim_bullets,
        "subclass_story_bullets": subclass_story_bullets,
        "shared_pathways_bullets": shared_pathways_bullets,
        "discord_bullets": discord_bullets,
        "engines_bullets": engines_bullets,
        "takeaways": takeaways,
        "conclusion": conclusion,

        # Data
        "disease_dirs": disease_dirs,
        "disease_to_stats": disease_to_stats,
        "subclass_counts_series": subclass_counts,
        "theme_evidence_df": theme_evidence_df,
        "theme_presence_mat": theme_presence_mat,
        "theme_chips": theme_chips,
        "shared_pathways_df": shared_pathways_df,
        "discord_df": disc_df,

        # Similarity
        "sim_heatmap_uri": sim_heatmap_uri,
        "sim_pairs_df": sim_pairs,
        "sim_mat": sim_mat,
        "similarity_fallback_image_used": bool(sim_fallback_used),

        # Drivers (ranked)
        "tf_rank": tf_rank,
        "epi_rank": epi_rank,
        "met_rank": met_rank,
        "tf_common": tf_common,
        "epi_common": epi_common,
        "met_common": met_common,

        # Entities provenance
        "entity_json_sources": json_sources,
        "drivers_source_note": "Primary: OUT_ROOT/results/all_jsons or OUT_ROOT/jsons_all*. Fallback: disease overlap json. Fallback: ALL_COMBINED.csv.",

        # Case studies
        "case_pool_pathways": pool50[:int(pool_limit)],
        "case_default_pathways": default3,
        "case_data": case_data,
        "activity_score_used": activity_meta,
    }

    html = _safe_render_html(ui, context)
    out_html = outdir / "index.html"
    out_html.write_text(html, encoding="utf-8")
    log_info(f"Report HTML written to: {out_html}")

    artifact = {
        "cohort": cohort_name,
        "generated": _ts(),
        "diseases": diseases,
        "disease_dirs": {d: str(disease_dirs[d]) for d in diseases},
        "q_cutoff": float(q_cutoff),
        "kpis": {
            "top_shared_theme": kpi_top_subclass,
            "shared_themes_ge2": int(kpi_shared_subclasses),
            "top_contrasting_pathway": kpi_top_discord
        },
        "entity_json_sources": json_sources,
        "case_default_pathways_top3": default3,
        "case_pool_top50": pool50[:int(pool_limit)],
        "activity_score_used": activity_meta,
        "similarity_fallback_image_used": bool(sim_fallback_used),
        "notes": {
            "interactive_html": True,
            "pdf_static": True
        }
    }
    (outdir / "report_artifact.json").write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    log_info(f"Debug artifact written to: {outdir / 'report_artifact.json'}")

    pdf_name = f"Report_{cohort_name}_{datetime.now().strftime('%Y-%m-%d_%H-%M')}.pdf"
    pdf_path = outdir / pdf_name
    if write_pdf(out_html, pdf_path):
        log_info(f"PDF written to: {pdf_path}")
    else:
        log_warn("Could not create PDF. Install Chrome/Chromium/Edge OR `pip install weasyprint`.")


# ========= CLI =========

def parse_args():
    p = argparse.ArgumentParser(description="Generate Cross-Disease Report (HTML + PDF) — Core orchestrator.")
    p.add_argument("--out-root", "--out_root", "--counts-root", "--counts_root",
                   required=True, dest="out_root",
                   help="Pipeline output root (e.g., /mnt/d/temp/A_IPAA4).")
    p.add_argument("--no-llm", action="store_true", help="Disable LLM-based interpretation.")
    p.add_argument("--q-cutoff", type=float, default=0.05, help="q/FDR cutoff for theme/pathway presence.")
    p.add_argument("--pool-limit", type=int, default=50, help="Max pathways in dropdown pool (default 50).")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_report(
        Path(args.out_root),
        no_llm=bool(args.no_llm),
        q_cutoff=float(args.q_cutoff),
        pool_limit=int(args.pool_limit),
    )
