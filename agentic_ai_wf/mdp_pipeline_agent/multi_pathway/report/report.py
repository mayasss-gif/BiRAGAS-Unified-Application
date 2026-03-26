#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ayass Bioscience — MDP Cross-Disease Report (Plan2 v4.3 DROP-IN, layout-preserving)
-----------------------------------------------------------------------------------
DROP-IN REPLACEMENT for: mdp_pipeline_3/report/report.py

What this fixes (without changing your report layout/sections):
1) ✅ Pandas crash: removes all `df or pd.DataFrame()` truthiness patterns
2) ✅ Similarity plot fallback:
   - If an INSIGHTS shared similarity image exists, it is used
   - Else fallback to rendering a heatmap from the similarity table
3) ✅ “Same pathway, different mechanism” bullets:
   - Primary: deterministic bullets from your biolinkbert.py (evidence-only)
   - Optional: LLM refines bullets into clean clinician-facing insight
   - Prevents dumping PMIDs/cell lines/ChIP-Seq metadata by sanitizing entities BEFORE LLM
4) ✅ Better LLM narrative “insight”:
   - LLM sees only CLEAN, normalized drivers (TF/epi/metabolite) + top themes/pathways/discordance
   - Forces interpretive linkage (shared theme → clinical meaning → divergence hypothesis → validation)
   - Still evidence-bound (association-only)

Run:
  python -m report.report --counts-root /mnt/d/DEGs_run_099 --q-cutoff 0.05
  python -m report.report --counts-root /mnt/d/DEGs_run_099 --q-cutoff 0.05 --no-llm
"""

import os
import json
import base64
import argparse
import re
import shutil
import sys
import textwrap
import time
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Any
from datetime import datetime

import numpy as np
import pandas as pd
import io
import subprocess
from urllib.parse import urljoin, quote


import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from jinja2 import Template
from dotenv import load_dotenv


# ======================
# === Optional biolinkbert bullets (your deterministic module)
# ======================
_HAS_BIOLINKBERT = False
try:
    # expects your file biolinkbert.py to be importable
    from biolinkbert import build_contrast_bullets, normalize_entity  # type: ignore
    _HAS_BIOLINKBERT = True
except Exception:
    build_contrast_bullets = None
    normalize_entity = None
    _HAS_BIOLINKBERT = False


# ======================
# === OpenAI (LLM-only)
# ======================

def _require_openai():
    load_dotenv()
    key = os.getenv("OPENAI_API_KEY", "").strip()
    if not key:
        raise RuntimeError("OPENAI_API_KEY missing in .env (LLM interpretations enabled).")
    try:
        from openai import OpenAI
    except Exception as e:
        raise RuntimeError("`openai` python package not found. Install: pip install openai") from e

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

    # unwrap ```json ... ```
    if "```" in text:
        m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
        if m:
            text = m.group(1).strip()

    try:
        return json.loads(text)
    except Exception:
        pass

    # substring recovery
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
      - tries API-enforced JSON (response_format=json_object) if supported
      - retries
      - repair loop
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

    print("[WARN] LLM JSON failed after retries+repair. Using deterministic defaults.")
    if last_err:
        print(f"[WARN] Last LLM error: {last_err}")
    if last_text:
        print("[WARN] Last LLM text (first 400 chars):")
        print(last_text[:400].replace("\n", " ") + ("..." if len(last_text) > 400 else ""))
    return {}


# ==============
# === IO utils
# ==============

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def read_table(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
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
            with open(path, "r", encoding="utf-8") as f:
                obj = json.load(f)
            if isinstance(obj, dict):
                return pd.json_normalize(obj)
            return pd.DataFrame(obj)
    except pd.errors.EmptyDataError:
        print(f"[WARN] Failed to read {path}: No columns to parse (empty file).")
        return None
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


def fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")


def clean_name(s: str) -> str:
    return re.sub(r"\s+", " ", str(s or "").strip())


def safe_num(x, default=np.nan) -> float:
    try:
        v = float(x)
        if np.isfinite(v):
            return v
    except Exception:
        pass
    return default


# =========================
# === Hard pathway filters
# =========================

OLF_SENSE_PAT = re.compile(
    r"(olfactory|odorant|odour|smell|taste|phototransduction|sensory\s+perception|sensory\s+system)",
    flags=re.IGNORECASE,
)


def is_allowed_pathway(name: str) -> bool:
    s = str(name or "")
    return not bool(OLF_SENSE_PAT.search(s))


# =========================
# === Non-human filtering (entities)
# =========================

NON_HUMAN_PAT = re.compile(r"\b(mouse|mice|rat|mm9|mm10|murine|c2c12)\b", flags=re.IGNORECASE)


def is_human_entity(name: str) -> bool:
    if not name:
        return False
    return NON_HUMAN_PAT.search(str(name)) is None


# =========================
# === Similarity image fallback (INSIGHTS shared plots)
# =========================

def _file_to_data_uri(img_path: Path) -> Optional[str]:
    if not img_path or not img_path.exists():
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


def similarity_fallback_data_uri(counts_root: Path) -> Optional[str]:
    """
    Prefer any existing INSIGHTS shared similarity figure (so report matches pipeline outputs).
    Tries common locations.
    """
    candidates = [
        counts_root / "INSIGHTS_out" / "figures" / "shared" / "shared_disease_similarity.png",
        counts_root / "INSIGHTS_out" / "figures" / "shared" / "shared_disease_similarity.svg",
        counts_root / "INSIGHTS_out" / "figures" / "shared_disease_similarity.png",
        counts_root / "INSIGHTS_out" / "figures" / "shared_disease_similarity.svg",
        counts_root / "INSIGHTS_out" / "shared_disease_similarity.png",
        counts_root / "INSIGHTS_out" / "shared_disease_similarity.svg",
    ]
    for p in candidates:
        uri = _file_to_data_uri(p)
        if uri:
            return uri
    return None


# =========================
# === Entity sanitization (prevents ChIP-Seq metadata dumps)
# =========================

_STUDY_ID_PAT = re.compile(r"\b(\d{7,8})\b")
_SEP_PAT = re.compile(r"[\|\(\)\[\]\{\}:]+")
_MULTI_WS = re.compile(r"\s+")
_HISTONE_MARK_PAT = re.compile(r"^(H[234]K\d+(AC|ME\d?))$", flags=re.IGNORECASE)
_NONWORD_PAT = re.compile(r"[^A-Za-z0-9_\-]+")


def _basic_clean_entity(s: str) -> str:
    s = str(s or "").strip()
    s = _SEP_PAT.sub(" ", s)
    s = _STUDY_ID_PAT.sub(" ", s)
    s = _MULTI_WS.sub(" ", s).strip()
    return s


def safe_entity_label(raw: str, entity_type: str) -> str:
    """
    Deterministic cleaning for display + LLM evidence payload:
    - TF/EPI: compress to the main symbol/mark (first token or histone mark)
    - Metabolite: keep cleaned string
    Uses biolinkbert.normalize_entity if available.
    """
    et = (entity_type or "").strip().lower()
    raw2 = _basic_clean_entity(raw)
    if not raw2:
        return ""

    if _HAS_BIOLINKBERT and normalize_entity is not None:
        try:
            out = str(normalize_entity(raw2, entity_type=et)).strip()
            return out
        except Exception:
            pass

    toks = raw2.split()
    head = toks[0] if toks else raw2

    # histone mark detection anywhere for epi
    if et == "epigenetic":
        for t in toks:
            t2 = _NONWORD_PAT.sub("", t).upper().strip()
            if _HISTONE_MARK_PAT.match(t2):
                return t2
        head2 = _NONWORD_PAT.sub("", head).upper().strip()
        return head2

    if et == "tf":
        # sometimes histone marks leak into TF
        head2 = _NONWORD_PAT.sub("", head).upper().strip()
        if _HISTONE_MARK_PAT.match(head2):
            return head2
        return head2

    if et == "metabolite":
        # keep readable, remove double spaces
        return raw2.strip()

    return raw2.strip()


# =========================
# === Plot helpers
# =========================

def make_barh(title: str, series: pd.Series, xlabel: str, top_n: int = 12) -> str:
    s = series.dropna()
    if s.empty:
        s = pd.Series({"NA": 1})
    s = s.sort_values(ascending=True).tail(top_n)
    fig = plt.figure(figsize=(9.6, max(3.6, 0.36 * len(s) + 1.9)))
    ax = fig.add_subplot(111)
    ax.barh(s.index.astype(str), s.values)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.grid(True, axis="x", alpha=0.25)
    fig.tight_layout()
    return fig_to_b64(fig)


def make_similarity_heatmap(sim_df: Optional[pd.DataFrame], title: str = "Disease similarity") -> str:
    if sim_df is None or sim_df.empty:
        fig = plt.figure(figsize=(7.2, 5.8))
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, "Similarity table not available", ha="center", va="center", fontsize=12)
        ax.set_axis_off()
        fig.tight_layout()
        return fig_to_b64(fig)

    mat = None
    cols = [c.lower() for c in sim_df.columns]
    if sim_df.shape[0] == sim_df.shape[1] and sim_df.index.is_unique:
        mat = sim_df.copy()
    elif {"disease_a", "disease_b", "similarity"}.issubset(set(cols)):
        a = sim_df.columns[cols.index("disease_a")]
        b = sim_df.columns[cols.index("disease_b")]
        s = sim_df.columns[cols.index("similarity")]
        mat = sim_df.pivot_table(index=a, columns=b, values=s, aggfunc="mean").fillna(0.0)
    else:
        if sim_df.columns[0].lower() in {"disease", "name"}:
            mat = sim_df.set_index(sim_df.columns[0])
        else:
            mat = sim_df.copy()

    mat = mat.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    idx = sorted(set(mat.index.astype(str)) | set(mat.columns.astype(str)))
    mat.index = mat.index.astype(str)
    mat.columns = mat.columns.astype(str)
    mat = mat.reindex(index=idx, columns=idx).fillna(0.0)
    mat = (mat + mat.T) / 2.0
    np.fill_diagonal(mat.values, 1.0)

    fig = plt.figure(figsize=(7.8, 6.2))
    ax = fig.add_subplot(111)
    im = ax.imshow(mat.values, aspect="auto")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xticks(range(len(idx)))
    ax.set_yticks(range(len(idx)))
    ax.set_xticklabels(idx, rotation=45, ha="right")
    ax.set_yticklabels(idx)
    fig.colorbar(im, ax=ax, fraction=0.045, pad=0.04)
    fig.tight_layout()
    return fig_to_b64(fig)


def make_table(df: Optional[pd.DataFrame], max_rows: int = 15) -> str:
    if df is None or df.empty:
        return "<div class='panel'><p style='margin:0;'>No rows available.</p></div>"
    return df.head(max_rows).to_html(index=False, border=0, escape=True)


# =========================
# === PDF export
# =========================

def _find_browser_exe() -> Optional[str]:
    candidates = []
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
    browser = _find_browser_exe()
    if browser:
        try:
            
            file_url = urljoin("file:", quote(str(html_path.resolve()).replace("\\", "/")))
            cmd = [
                browser,
                "--headless=new",
                "--disable-gpu",
                "--disable-print-preview",
                f"--print-to-pdf={str(pdf_path)}",
                file_url,
            ]
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            return pdf_path.exists()
        except Exception as e:
            print(f"[WARN] Browser PDF export failed: {e}")

    try:
        from weasyprint import HTML as WPHTML
        WPHTML(filename=str(html_path)).write_pdf(str(pdf_path))
        return pdf_path.exists()
    except Exception as e:
        print("[WARN] PDF export failed. Install Chrome/Chromium/Edge OR `pip install weasyprint`.")
        print(f"[WARN] WeasyPrint error: {e}")
        return False


# =========================
# === Layout discovery (diseases + per-disease dirs)
# =========================

def _is_noise_dir(name: str) -> bool:
    return name.startswith(("INSIGHTS_out", "ANALYSIS_", "CLIENT_", "baseline_", "comparison",
                            "json_comparison", "results", "_", "mdp_report", "Report"))


def discover_disease_dirs(counts_root: Path) -> Dict[str, Path]:
    """
    Returns {disease_name: disease_dir_path} where disease_dir_path contains gsea_prerank_classified.tsv.

    Supports:
      - <root>/<Disease>/gsea_prerank_classified.tsv
      - <root>/GL_enrich/<Disease>/gsea_prerank_classified.tsv
      - <root>/GC_enrich/<Disease>/gsea_prerank_classified.tsv
      - <root>/*_enrich/<Disease>/gsea_prerank_classified.tsv
    """
    candidates: List[Path] = []

    # Direct children
    for p in sorted(counts_root.glob("*")):
        if p.is_dir() and not _is_noise_dir(p.name):
            candidates.append(p)

    # Enrich containers
    for enrich_dir in sorted(counts_root.glob("*_enrich")):
        if enrich_dir.is_dir():
            for p in sorted(enrich_dir.glob("*")):
                if p.is_dir() and not _is_noise_dir(p.name):
                    candidates.append(p)

    # Explicit GL_enrich / GC_enrich
    for name in ["GL_enrich", "GC_enrich"]:
        e = counts_root / name
        if e.is_dir():
            for p in sorted(e.glob("*")):
                if p.is_dir() and not _is_noise_dir(p.name):
                    candidates.append(p)

    out: Dict[str, Path] = {}
    seen = set()
    for ddir in candidates:
        if ddir in seen:
            continue
        seen.add(ddir)

        gsea = ddir / "gsea_prerank_classified.tsv"
        if gsea.exists():
            out[ddir.name] = ddir

    return out


# =========================
# === Overlap JSON discovery
# =========================

def discover_overlap_json_for_disease(counts_root: Path, disease: str, disease_dir: Path) -> Optional[Path]:
    """
    Try multiple known locations for overlap JSON. Prefer the most specific per-disease overlap file.
    """
    checks = []

    # Legacy counts output location
    checks.append(counts_root / "results" / "all_jsons" / f"{disease}_pathway_entity_overlap.json")

    # Per-disease overlap folder in enrich outputs
    checks.append(disease_dir / "overlap" / "pathway_entity_overlap.json")

    # In case disease_dir is itself nested oddly, try sibling enrich containers
    for enrich_name in ["GL_enrich", "GC_enrich"]:
        checks.append(counts_root / enrich_name / disease / "overlap" / "pathway_entity_overlap.json")

    # Generic *_enrich containers
    for enrich_dir in sorted(counts_root.glob("*_enrich")):
        checks.append(enrich_dir / disease / "overlap" / "pathway_entity_overlap.json")

    # Fallback: jsons_all_folder (best effort; may not be overlap schema)
    for enrich_dir in [counts_root / "GL_enrich", counts_root / "GC_enrich"]:
        checks.append(enrich_dir / "jsons_all_folder" / f"{disease}.json")
    for enrich_dir in sorted(counts_root.glob("*_enrich")):
        checks.append(enrich_dir / "jsons_all_folder" / f"{disease}.json")

    for p in checks:
        if p and p.exists():
            return p
    return None


# =========================
# === Overlap JSON parsing (schema-flex)
# =========================

def parse_overlap_json(path: Path) -> Dict[str, Any]:
    obj = read_json(path)
    if obj is None:
        return {}
    return obj if isinstance(obj, dict) else {}


def _extract_overlap_score(row: Dict[str, Any]) -> float:
    if not isinstance(row, dict):
        return 1.0

    key_candidates = [
        "overlap_genes", "overlapGenes", "overlap_gene_set",
        "overlap_count", "n_overlap", "nOverlaps", "overlap",
        "shared_genes", "n_shared", "intersection_size",
        "gene_overlap", "overlap_n", "k",
    ]
    for k in key_candidates:
        if k in row:
            v = row.get(k)
            if isinstance(v, (list, tuple, set)):
                return float(len(v))
            if isinstance(v, str):
                toks = [t for t in re.split(r"[,\s;]+", v.strip()) if t]
                if len(toks) >= 2:
                    return float(len(toks))
                try:
                    return float(v)
                except Exception:
                    return 1.0
            try:
                return float(v)
            except Exception:
                return 1.0

    for k in ["genes", "overlapping genes", "overlapping_genes", "overlapGenesList"]:
        if k in row:
            v = row.get(k)
            if isinstance(v, (list, tuple, set)):
                return float(len(v))
            if isinstance(v, str):
                toks = [t for t in re.split(r"[,\s;]+", v.strip()) if t]
                if toks:
                    return float(len(toks))

    return 1.0


def _rows_to_entities(rows, human_only=True) -> List[Tuple[str, float]]:
    out = []
    if not isinstance(rows, list):
        return out

    for row in rows:
        if isinstance(row, str):
            ent = clean_name(row)
            if human_only and not is_human_entity(ent):
                continue
            if ent:
                out.append((ent, 1.0))
            continue

        if not isinstance(row, dict):
            continue

        ent = clean_name(
            row.get("entity")
            or row.get("Entity")
            or row.get("name")
            or row.get("Name")
            or row.get("term")
            or row.get("Term")
            or row.get("target")
            or row.get("Target")
            or ""
        )
        if human_only and not is_human_entity(ent):
            continue
        if not ent:
            continue

        score = _extract_overlap_score(row)
        out.append((ent, float(score)))

    agg: Dict[str, float] = {}
    for e, sc in out:
        agg[e] = max(agg.get(e, 0.0), float(sc))
    return list(agg.items())


def _get_pathway_block(overlap_obj: Dict[str, Any], pathway_name: str) -> Optional[Dict[str, Any]]:
    if not isinstance(overlap_obj, dict) or not pathway_name:
        return None
    if pathway_name in overlap_obj and isinstance(overlap_obj[pathway_name], dict):
        return overlap_obj[pathway_name]
    low = pathway_name.strip().lower()
    for k, v in overlap_obj.items():
        if str(k).strip().lower() == low and isinstance(v, dict):
            return v
    return None


def extract_entities_with_scores_for_pathway(
    overlap_obj: Dict[str, Any],
    pathway_name: str,
    entity_type: str,
    direction_mode: str = "ANY",
    human_only: bool = True
) -> List[Tuple[str, float]]:
    """
    Supports BOTH schemas:
    A) Directional: pathway -> UP/DOWN -> tf/epigenetic/metabolite
    B) Non-directional: pathway -> tf/epigenetic/metabolite (NO UP/DOWN)
    """
    blk = _get_pathway_block(overlap_obj, pathway_name)
    if not blk:
        return []

    et = entity_type.lower()

    # Schema A: directional
    if any(k in blk for k in ("UP", "DOWN")):
        out: List[Tuple[str, float]] = []

        def _pull(dir_key: str):
            dd = blk.get(dir_key, {})
            if not isinstance(dd, dict):
                return
            rows = dd.get(et, dd.get(et + "s", []))
            out.extend(_rows_to_entities(rows, human_only=human_only))

        if direction_mode == "UP":
            _pull("UP")
        elif direction_mode == "DOWN":
            _pull("DOWN")
        else:
            _pull("UP")
            _pull("DOWN")

        agg: Dict[str, float] = {}
        for e, sc in out:
            agg[e] = max(agg.get(e, 0.0), float(sc))
        return list(agg.items())

    # Schema B: non-directional
    rows = None
    if et in blk:
        rows = blk[et]
    elif (et + "s") in blk:
        rows = blk[et + "s"]
    elif "entities" in blk and isinstance(blk["entities"], dict):
        rows = blk["entities"].get(et) or blk["entities"].get(et + "s")
    elif "drivers" in blk and isinstance(blk["drivers"], dict):
        rows = blk["drivers"].get(et) or blk["drivers"].get(et + "s")

    if rows is not None:
        return _rows_to_entities(rows, human_only=human_only)

    # last resort: scan nested dicts
    for _, v in blk.items():
        if isinstance(v, dict) and (et in v or (et + "s") in v):
            rows = v.get(et) or v.get(et + "s")
            return _rows_to_entities(rows, human_only=human_only)

    return []


def compute_entity_rankings_from_json(
    disease_to_json: Dict[str, Dict[str, Any]],
    diseases: List[str],
    entity_type: str,
    direction_mode: str = "ANY",
) -> Dict[str, pd.Series]:
    """
    entity_score = sum over pathways of max overlap score for that entity in that pathway.
    NOTE: we sanitize entities here so tables + LLM payload stay clean.
    """
    out: Dict[str, pd.Series] = {}
    et = entity_type.lower().strip()

    for dis in diseases:
        obj = disease_to_json.get(dis, {})
        if not obj:
            out[dis] = pd.Series(dtype=float)
            continue

        freq: Dict[str, float] = {}
        for pathway in obj.keys():
            if not is_allowed_pathway(pathway):
                continue
            ents = extract_entities_with_scores_for_pathway(
                obj, pathway, et, direction_mode=direction_mode, human_only=True
            )
            for ent, sc in ents:
                lab = safe_entity_label(ent, et)
                if not lab:
                    continue
                freq[lab] = freq.get(lab, 0.0) + float(sc)

        out[dis] = pd.Series(freq).sort_values(ascending=False)

    return out


def shared_entities_across_all_diseases(
    per_disease_series: Dict[str, pd.Series],
    diseases: List[str],
    top_n: int = 10
) -> pd.Series:
    sets = []
    for dis in diseases:
        s = per_disease_series.get(dis, pd.Series(dtype=float))
        sets.append(set(s.index.tolist()))
    if not sets:
        return pd.Series(dtype=float)
    shared = set.intersection(*sets) if sets else set()
    if not shared:
        return pd.Series(dtype=float)

    totals: Dict[str, float] = {}
    for ent in shared:
        totals[ent] = float(sum(per_disease_series[d].get(ent, 0.0) for d in diseases))
    return pd.Series(totals).sort_values(ascending=False).head(top_n)


# =========================
# === Fallback entity extraction from sig/all tables
# =========================

def detect_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols = list(df.columns)
    low = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand.lower() in low:
            return low[cand.lower()]
    for cand in candidates:
        for c in cols:
            if cand.lower() in c.lower():
                return c
    return None


def _load_sig_list(disease_dir: Path, filenames: List[str], entity_type: str, max_items: int = 30) -> List[str]:
    for fn in filenames:
        p = disease_dir / fn
        df = read_table(p)
        if df is None or df.empty:
            continue

        name_col = detect_col(df, ["entity", "Entity", "term", "Term", "name", "Name", "target", "Target"])
        if name_col is None:
            name_col = df.columns[0] if len(df.columns) else None
        if name_col is None:
            continue

        pcol = detect_col(df, ["pval", "p-value", "p_value", "P-value", "Adjusted P-value", "FDR", "qval", "padj"])
        if pcol is not None:
            df["__p"] = pd.to_numeric(df[pcol], errors="coerce")
            df = df.sort_values("__p", ascending=True)

        vals = [safe_entity_label(x, entity_type) for x in df[name_col].astype(str).tolist()]
        vals = [x for x in vals if x and is_human_entity(x)]
        if vals:
            out = []
            seen = set()
            for x in vals:
                if x not in seen:
                    seen.add(x)
                    out.append(x)
                if len(out) >= max_items:
                    break
            return out
    return []


def _fallback_entities_from_tables(disease_dir: Path) -> Dict[str, List[str]]:
    return {
        "tf": _load_sig_list(disease_dir, ["tf_sig.tsv", "tf_all.tsv"], "tf", max_items=40),
        "epigenetic": _load_sig_list(disease_dir, ["epigenetic_sig.tsv", "epigenetic_all.tsv"], "epigenetic", max_items=40),
        "metabolite": _load_sig_list(disease_dir, ["metabolites_sig.tsv", "metabolites_all.tsv"], "metabolite", max_items=40),
    }


def _best_effort_from_all_combined(disease_dir: Path) -> Dict[str, List[str]]:
    p = disease_dir / "ALL_COMBINED.csv"
    df = read_table(p)
    if df is None or df.empty:
        return {"tf": [], "epigenetic": [], "metabolite": []}

    src_col = detect_col(df, ["Source", "source", "TYPE", "type", "Category", "category"])
    name_col = detect_col(df, ["Entity", "entity", "Term", "term", "Name", "name"])
    if name_col is None:
        name_col = df.columns[0] if len(df.columns) else None
    if name_col is None:
        return {"tf": [], "epigenetic": [], "metabolite": []}

    out = {"tf": [], "epigenetic": [], "metabolite": []}
    if src_col is None:
        vals = [safe_entity_label(x, "tf") for x in df[name_col].astype(str).tolist()]
        vals = [x for x in vals if x and is_human_entity(x)]
        out["tf"] = vals[:30]
        return out

    def _bucket(s: str) -> Optional[str]:
        s = (s or "").strip().lower()
        if "tf" in s or "transcription" in s or "chea" in s or "jaspar" in s:
            return "tf"
        if "epi" in s or "histone" in s or "chromatin" in s or "encode" in s:
            return "epigenetic"
        if "metab" in s or "hmdb" in s or "compound" in s:
            return "metabolite"
        return None

    for _, r in df.iterrows():
        b = _bucket(str(r.get(src_col, "")))
        if b is None:
            continue
        name = safe_entity_label(r.get(name_col, ""), b)
        if not name or not is_human_entity(name):
            continue
        out[b].append(name)

    for k in out:
        uniq = []
        seen = set()
        for x in out[k]:
            if x not in seen:
                seen.add(x)
                uniq.append(x)
        out[k] = uniq[:40]
    return out


# =========================
# === Metabolite enrich parsing (shared metabolites)
# =========================

def load_significant_metabolites_from_enrich_files(
    disease_dir: Path,
    pval_cutoff: float = 0.1
) -> List[str]:
    mets: List[str] = []
    for fn in ["metabolite_enrich_up.csv", "metabolite_enrich_down.csv"]:
        p = disease_dir / fn
        df = read_table(p)
        if df is None or df.empty:
            continue

        pcol = detect_col(df, ["pval", "p-value", "p_value", "P-value", "P.Value"]) \
            or detect_col(df, ["Adjusted P-value", "adj_p", "padj", "FDR", "qval"])
        ncol = detect_col(df, ["metabolite", "Metabolite", "Term", "Drug", "Compound", "name", "Name"]) \
            or (df.columns[0] if len(df.columns) else None)

        if not ncol:
            continue

        if pcol:
            pv = pd.to_numeric(df[pcol], errors="coerce")
            df = df[pv <= pval_cutoff].copy()

        names = [safe_entity_label(x, "metabolite") for x in df[ncol].astype(str).tolist()]
        for x in names:
            if x:
                mets.append(x)

    out = []
    seen = set()
    for m in mets:
        if m not in seen:
            seen.add(m)
            out.append(m)
    return out


def compute_shared_metabolites_from_enrich(
    disease_dirs: Dict[str, Path],
    diseases: List[str],
    pval_cutoff: float = 0.1
) -> pd.Series:
    per: Dict[str, set] = {}
    for dis in diseases:
        ddir = disease_dirs.get(dis)
        if ddir is None:
            per[dis] = set()
            continue
        mets = load_significant_metabolites_from_enrich_files(ddir, pval_cutoff=pval_cutoff)
        per[dis] = set(mets)

    if not per:
        return pd.Series(dtype=float)

    shared = None
    for dis in diseases:
        shared = per[dis] if shared is None else shared.intersection(per[dis])
    shared = shared or set()

    if not shared:
        return pd.Series(dtype=float)

    return pd.Series({m: len(diseases) for m in sorted(shared)}).sort_values(ascending=False)


# =========================
# === GSEA classified parsing (Sub_Class)
# =========================

def normalize_gsea_classified(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    term_col = detect_col(df, ["Pathway", "Pathways", "Term", "NAME", "pathway"])
    sub_col = detect_col(df, ["Sub_Class", "SubClass", "sub_class", "Subclass"])
    main_col = detect_col(df, ["MainClass", "Main_Class", "Class", "Category", "main_class"])
    nes_col = detect_col(df, ["NES", "nes", "Normalized Enrichment Score"])
    q_col = detect_col(df, ["qval", "FDR q-val", "FDR", "padj", "adj_p", "q-value", "Adjusted P-value"])
    dir_col = detect_col(df, ["Direction", "direction", "Regulation", "UPDOWN", "sign"])

    out = pd.DataFrame()
    out["Pathway"] = df[term_col].astype(str) if term_col else df.iloc[:, 0].astype(str)
    out["Sub_Class"] = df[sub_col].astype(str) if sub_col else ""
    out["Main_Class"] = df[main_col].astype(str) if main_col else ""
    out["NES"] = pd.to_numeric(df[nes_col], errors="coerce") if nes_col else np.nan
    out["qval"] = pd.to_numeric(df[q_col], errors="coerce") if q_col else np.nan

    if dir_col:
        out["Direction"] = df[dir_col].astype(str).str.upper()
    else:
        out["Direction"] = np.where(out["NES"] >= 0, "UP", "DOWN")

    out["Pathway"] = out["Pathway"].map(clean_name)
    out["Sub_Class"] = out["Sub_Class"].map(clean_name)
    out["Main_Class"] = out["Main_Class"].map(clean_name)
    out["Direction"] = out["Direction"].map(clean_name)

    out["Direction"] = out["Direction"].replace(
        {"UPREGULATED": "UP", "DOWNREGULATED": "DOWN", "POSITIVE": "UP", "NEGATIVE": "DOWN"}
    )
    out.loc[~out["Direction"].isin(["UP", "DOWN"]), "Direction"] = np.where(out["NES"] >= 0, "UP", "DOWN")

    out.loc[out["Sub_Class"].isin(["", "NA", "NAN", "NONE"]), "Sub_Class"] = "Unclassified"
    out.loc[out["Main_Class"].isin(["", "NA", "NAN", "NONE"]), "Main_Class"] = "Unclassified"

    out = out[out["Pathway"].map(is_allowed_pathway)].copy()
    return out


def compute_shared_subclass_shortlist(
    disease_to_gsea: Dict[str, pd.DataFrame],
    q_cutoff: float = 0.05,
    top_n: int = 30,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    best_rows = []

    for disease, gdf in disease_to_gsea.items():
        if gdf is None or gdf.empty:
            continue

        g = gdf.copy()
        if g["qval"].notna().any():
            g = g[g["qval"] <= q_cutoff].copy()

        g = g[g["Sub_Class"].notna()].copy()
        g.loc[g["Sub_Class"].eq(""), "Sub_Class"] = "Unclassified"

        g["absNES"] = g["NES"].abs()
        g["q_sort"] = g["qval"].fillna(1.0)
        g = g.sort_values(["Sub_Class", "q_sort", "absNES"], ascending=[True, True, False])

        best = g.groupby("Sub_Class", as_index=False).head(1).copy()
        for _, r in best.iterrows():
            best_rows.append(
                {
                    "Disease": disease,
                    "Sub_Class": r["Sub_Class"],
                    "Main_Class": r["Main_Class"],
                    "Best_Pathway": r["Pathway"],
                    "NES": safe_num(r["NES"]),
                    "qval": safe_num(r["qval"]),
                    "Direction": r["Direction"],
                }
            )

        present = set(best["Sub_Class"].tolist())
        for sub in present:
            rows.append({"Sub_Class": sub, "Disease": disease})

    if not rows:
        return pd.DataFrame(), pd.DataFrame(best_rows)

    pres = pd.DataFrame(rows).drop_duplicates()
    counts = pres.groupby("Sub_Class")["Disease"].nunique().reset_index(name="n_diseases")

    best_long = pd.DataFrame(best_rows)
    if not best_long.empty:
        tmp = best_long.copy()
        tmp["neglog10q"] = -np.log10(tmp["qval"].replace(0, np.nan))
        tmp["neglog10q"] = tmp["neglog10q"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        strength = tmp.groupby("Sub_Class", as_index=False).agg(
            mean_neglog10q=("neglog10q", "mean"),
            mean_absNES=("NES", lambda x: float(np.nanmean(np.abs(x)))),
            directions=("Direction", lambda x: ",".join(sorted(set([str(v) for v in x if str(v)]))))
        )
        counts = counts.merge(strength, on="Sub_Class", how="left")
    else:
        counts["mean_neglog10q"] = 0.0
        counts["mean_absNES"] = 0.0
        counts["directions"] = ""

    counts = counts.sort_values(
        ["n_diseases", "mean_neglog10q", "mean_absNES"],
        ascending=[False, False, False],
    )
    shortlist = counts.head(top_n).copy()

    if not best_long.empty:
        bl = best_long[best_long["Sub_Class"].isin(set(shortlist["Sub_Class"]))].copy()
        bl = bl[bl["Best_Pathway"].map(is_allowed_pathway)].copy()
        bl["NES"] = bl["NES"].map(lambda x: "" if pd.isna(x) else f"{x:.2f}")
        bl["qval"] = bl["qval"].map(lambda x: "" if pd.isna(x) else f"{x:.2e}" if x < 0.001 else f"{x:.3f}")
        best_long = bl

    return shortlist, best_long


# =========================
# === Discordance parsing
# =========================

def compute_discordant_pathways(direction_concord: Optional[pd.DataFrame], top_n: int = 12) -> pd.DataFrame:
    if direction_concord is None or direction_concord.empty:
        return pd.DataFrame()

    df = direction_concord.copy()
    pcol = detect_col(df, ["Pathway", "Term", "pathway", "Pathways"]) or df.columns[0]
    dcol = detect_col(df, ["discord", "flip", "inconsistent", "mixed", "both"])

    if dcol is None:
        upc = detect_col(df, ["up", "n_up", "UP_count", "count_up"])
        dnc = detect_col(df, ["down", "n_down", "DOWN_count", "count_down"])
        if upc and dnc:
            df["__discord"] = (
                pd.to_numeric(df[upc], errors="coerce").fillna(0) *
                pd.to_numeric(df[dnc], errors="coerce").fillna(0)
            )
            dcol = "__discord"

    if dcol is None:
        return pd.DataFrame()

    out = pd.DataFrame({
        "Pathway": df[pcol].astype(str).map(clean_name),
        "Discordance": pd.to_numeric(df[dcol], errors="coerce"),
    }).dropna()

    out = out[out["Pathway"].map(is_allowed_pathway)].copy()
    out = out.sort_values("Discordance", ascending=False).head(top_n)
    return out


# =========================
# === Shared pathways table
# =========================

def load_shared_pathways_table(counts_root: Path) -> pd.DataFrame:
    top200 = counts_root / "INSIGHTS_out" / "tables" / "top200_pathways_cross_disease.csv"
    df = read_table(top200)
    if df is not None and not df.empty:
        pcol = detect_col(df, ["Pathway", "Term", "Pathways"]) or df.columns[0]
        ncol = detect_col(df, ["n_diseases", "num_diseases", "count_diseases"])
        qcol = detect_col(df, ["qval", "FDR", "padj", "q-value", "Adjusted P-value"])
        keep = [pcol]
        if ncol:
            keep.append(ncol)
        if qcol:
            keep.append(qcol)
        out = df[keep].copy()
        out = out.rename(columns={pcol: "Pathway"})
        if ncol:
            out = out.rename(columns={ncol: "#Diseases"})
        if qcol:
            out = out.rename(columns={qcol: "q/FDR"})
        out = out[out["Pathway"].map(is_allowed_pathway)].copy()
        if "#Diseases" in out.columns and "q/FDR" in out.columns:
            out["q/FDR"] = pd.to_numeric(out["q/FDR"], errors="coerce")
            out = out.sort_values(["#Diseases", "q/FDR"], ascending=[False, True])
        elif "#Diseases" in out.columns:
            out = out.sort_values(["#Diseases"], ascending=[False])
        return out.head(25)

    return pd.DataFrame({"Pathway": [], "#Diseases": []})


# =========================
# === Pathway contrast cards
# =========================

def make_entity_contrast_plot(
    title: str,
    disease_a: str,
    disease_b: str,
    shared_counts: Dict[str, int],
    unique_a_counts: Dict[str, int],
    unique_b_counts: Dict[str, int],
) -> str:
    labels = ["TFs", "Metabolites", "Epigenetic"]
    shared = [shared_counts.get("tf", 0), shared_counts.get("metabolite", 0), shared_counts.get("epigenetic", 0)]
    ua = [unique_a_counts.get("tf", 0), unique_a_counts.get("metabolite", 0), unique_a_counts.get("epigenetic", 0)]
    ub = [unique_b_counts.get("tf", 0), unique_b_counts.get("metabolite", 0), unique_b_counts.get("epigenetic", 0)]

    x = np.arange(len(labels))
    w = 0.25

    fig = plt.figure(figsize=(9.2, 3.8))
    ax = fig.add_subplot(111)
    ax.bar(x - w, ua, width=w, label=f"{disease_a} only")
    ax.bar(x, shared, width=w, label="Shared")
    ax.bar(x + w, ub, width=w, label=f"{disease_b} only")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Count")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=3, loc="upper right")
    fig.tight_layout()
    return fig_to_b64(fig)


def compute_pathway_entity_contrast_cards(
    diseases: List[str],
    disease_to_json: Dict[str, Dict[str, Any]],
    pathways: List[str],
) -> List[Dict[str, Any]]:
    cards = []
    if not pathways or not disease_to_json or len(diseases) < 2:
        return cards

    disease_a = diseases[0]
    disease_b = diseases[1]

    for pw in pathways:
        if not is_allowed_pathway(pw):
            continue

        obj_a = disease_to_json.get(disease_a, {})
        obj_b = disease_to_json.get(disease_b, {})

        def ents(obj, etype):
            return set([safe_entity_label(e, etype) for e, _ in extract_entities_with_scores_for_pathway(obj, pw, etype, "ANY", True) if safe_entity_label(e, etype)])

        A = {
            "tf": ents(obj_a, "tf"),
            "metabolite": ents(obj_a, "metabolite"),
            "epigenetic": ents(obj_a, "epigenetic")
        }
        B = {
            "tf": ents(obj_b, "tf"),
            "metabolite": ents(obj_b, "metabolite"),
            "epigenetic": ents(obj_b, "epigenetic")
        }

        shared = {k: sorted(list(A[k].intersection(B[k]))) for k in A}
        only_a = {k: sorted(list(A[k].difference(B[k]))) for k in A}
        only_b = {k: sorted(list(B[k].difference(A[k]))) for k in A}

        shared_counts = {k: len(shared[k]) for k in shared}
        unique_a_counts = {k: len(only_a[k]) for k in only_a}
        unique_b_counts = {k: len(only_b[k]) for k in only_b}

        plot = make_entity_contrast_plot(
            title=pw,
            disease_a=disease_a,
            disease_b=disease_b,
            shared_counts=shared_counts,
            unique_a_counts=unique_a_counts,
            unique_b_counts=unique_b_counts,
        )

        def topn(lst, n=10):
            return ", ".join(lst[:n]) if lst else "—"

        tab = pd.DataFrame([
            {"Entity type": "TFs", "Shared": topn(shared["tf"]), f"{disease_a} only": topn(only_a["tf"]), f"{disease_b} only": topn(only_b["tf"])},
            {"Entity type": "Metabolites", "Shared": topn(shared["metabolite"]), f"{disease_a} only": topn(only_a["metabolite"]), f"{disease_b} only": topn(only_b["metabolite"])},
            {"Entity type": "Epigenetic", "Shared": topn(shared["epigenetic"]), f"{disease_a} only": topn(only_a["epigenetic"]), f"{disease_b} only": topn(only_b["epigenetic"])},
        ])

        cards.append({
            "pathway": pw,
            "title": pw,
            "plot": plot,
            "table": make_table(tab, max_rows=10),
            "raw": {"disease_a": disease_a, "disease_b": disease_b, "shared": shared, "only_a": only_a, "only_b": only_b},
            "bullets": [],
        })

    return cards


# =========================
# === Narrative defaults (NO EMPTY INTERPRETATIONS EVER)
# =========================

def ensure_report_narrative_defaults(j: dict, *, diseases: List[str], q_cutoff: float) -> dict:
    if not isinstance(j, dict):
        j = {}

    required = [
        "hook_headline",
        "hook_bullets",
        "sim_bullets",
        "subclass_story_bullets",
        "shared_pathways_bullets",
        "discord_bullets",
        "engines_bullets",
        "takeaways",
        "conclusion",
        "pathway_contrast_bullets",
    ]
    for k in required:
        if k not in j:
            j[k] = [] if k.endswith("_bullets") or k in {"hook_bullets", "takeaways"} else ({} if k == "pathway_contrast_bullets" else "")

    if not str(j.get("hook_headline", "")).strip():
        j["hook_headline"] = f"Evidence-constrained cross-disease report: shared themes computed at q ≤ {q_cutoff:.2g}."

    if not j.get("hook_bullets"):
        j["hook_bullets"] = [
            f"Shared themes are computed from significant pathways (q ≤ {q_cutoff:.2g}) across {len(diseases)} diseases.",
            "Top shared pathways and top contrasting pathways are shown with evidence tables.",
            "Drivers are normalized (TF/marks/metabolites) and ranked by overlap-evidence support (association-only).",
        ]

    if not j.get("sim_bullets"):
        j["sim_bullets"] = [
            "Similarity reflects pathway-level proximity from the available tables/INSIGHTS plots.",
            "Interpretation is limited to signals shown in this report (association-only).",
        ]

    if not j.get("subclass_story_bullets"):
        j["subclass_story_bullets"] = [
            "Theme bullets are grounded in the Evidence table (per-disease representative pathways).",
        ]

    if not j.get("shared_pathways_bullets"):
        j["shared_pathways_bullets"] = [
            "Shared pathways are recurrent across diseases in the input tables; interpret as shared biology candidates (association).",
        ]

    if not j.get("discord_bullets"):
        j["discord_bullets"] = [
            "Discordant pathways indicate direction flips across diseases under the same pathway label (when concordance evidence exists).",
        ]

    if not j.get("engines_bullets"):
        j["engines_bullets"] = [
            "Driver tables are hypotheses; validate in independent cohorts and orthogonal assays.",
        ]

    if not j.get("takeaways"):
        j["takeaways"] = [
            "Validate shared themes and pathways using literature and independent cohorts.",
            "Use contrasting pathways to prioritize disease-separating mechanisms and stratification hypotheses.",
            "Use driver contrasts as upstream-program hypotheses (TF/chromatin/metabolic axes).",
        ]

    if not str(j.get("conclusion", "")).strip():
        j["conclusion"] = "This report summarizes cross-disease shared biology using evidence-constrained pathway and driver contrasts."

    if not isinstance(j.get("pathway_contrast_bullets", {}), dict):
        j["pathway_contrast_bullets"] = {}

    return j


def _list_clean(x, n: int) -> List[str]:
    xs = [clean_name(v) for v in (x or []) if clean_name(v)]
    return xs[:n]


# =========================
# === LLM refiner for contrast bullets (clean insight, no metadata)
# =========================

def llm_refine_contrast_bullets(
    pathway: str,
    disease_a: str,
    disease_b: str,
    seed_bullets: List[str],
    debug: Dict[str, Any],
    max_bullets: int = 5,
) -> List[str]:
    system = (
        "You write clinician-facing cross-disease pathway bullets.\n"
        "Hard rules:\n"
        "- Association only; do NOT claim causality.\n"
        "- Do NOT include study IDs, PMID-like numbers, cell lines, or 'ChIP-Seq'.\n"
        "- Use only the provided entities (TFs/marks/metabolites).\n"
        "- Bullets must explain what the contrast means clinically (shared label vs different upstream program).\n"
        "- Output JSON only: {\"bullets\": [...]} with 3-5 bullets.\n"
        "- No markdown formatting.\n"
    )

    evidence_obj = {
        "pathway": pathway,
        "disease_a": disease_a,
        "disease_b": disease_b,
        "seed_bullets": seed_bullets,
        "top_ranked": (debug or {}).get("top_ranked", {}),
        "shared_counts": (debug or {}).get("shared_counts", {}),
        "unique_counts": (debug or {}).get("unique_counts", {}),
    }

    out = llm_json(
        system=system,
        prompt=(
            "Rewrite the bullets into clean clinical interpretation.\n"
            "Keep them specific to this pathway and the listed entities.\n"
            "Do NOT list raw strings; interpret them.\n\n"
            f"Evidence:\n{json.dumps(evidence_obj, indent=2)[:8000]}\n\n"
            "Return JSON: {\"bullets\": [\"...\", \"...\", \"...\"]}"
        ),
        max_tokens=260,
    )

    bullets = out.get("bullets", []) if isinstance(out, dict) else []
    bullets = [str(b).strip() for b in bullets if str(b).strip()]
    bullets = bullets[:max(3, min(max_bullets, 5))]

    if len(bullets) < 3:
        bullets = seed_bullets[:max(3, min(max_bullets, 5))]

    return bullets


# =========================
# === HTML Template (UNCHANGED layout)
# =========================

HTML = Template(r"""
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/><meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>MDP Cross-Disease Report — {{ cohort_name }}</title>
<style>
:root{
  --bg:#f7f8ff;--card:#ffffff;--ink:#0b1020;--muted:#4b5a8a;--edge:#e3e7ff;
  --accent:#6c5ce7;--glow1:#6c8cff;--glow2:#b06afc;--radius:18px;--maxw:1320px
}
html,body{margin:0;padding:0;background:var(--bg);color:var(--ink);
  font-family:Inter,system-ui,-apple-system,Segoe UI,Roboto,sans-serif;line-height:1.55;counter-reset: page;}
.container{max-width:var(--maxw);margin:0 auto;padding:22px 16px}
header{padding:18px 16px 12px;background:#fff;border-bottom:1px solid var(--edge)}
.header-inner{max-width:var(--maxw);margin:0 auto}
.brand{font-weight:900;letter-spacing:.3px;margin-bottom:2px}
h1{margin:4px 0 6px;font-size:24px;letter-spacing:.2px}
.muted{color:var(--muted);font-size:13px}
.pill{display:inline-block;padding:5px 10px;border-radius:999px;background:#eef1ff;border:1px solid var(--edge);
  color:#37417a;margin:0 6px 6px 0;font-size:12px}
.card{background:var(--card);border:1px solid var(--edge);border-radius:var(--radius);
  padding:18px;margin:0 0 16px;box-shadow:0 10px 28px rgba(16,23,51,.10);page-break-inside:avoid;break-inside:avoid}
h2{color:var(--accent);margin:0 0 10px;font-size:18px;letter-spacing:.2px;position:relative}
h2::after{content:"";position:absolute;left:0;bottom:-6px;width:54px;height:2px;
  background:linear-gradient(90deg,var(--glow1),var(--glow2));border-radius:2px;opacity:.8}
.two-col{display:grid;grid-template-columns:1.12fr 0.88fr;gap:14px}
@media (max-width:980px){.two-col{grid-template-columns:1fr}}
.grid-3{display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px}
@media (max-width:980px){.grid-3{grid-template-columns:1fr}}
.img-frame{border:1px solid var(--edge);border-radius:14px;overflow:hidden;background:#fff}
.img-frame img{width:100%;height:auto;display:block}
.caption{color:#5b69a8;font-size:12px;margin-top:6px}
.panel{background:#f3f5ff;border:1px solid #dbe2ff;border-radius:14px;padding:12px;margin-top:10px}
.panel h3{margin:0 0 6px;font-size:15px;color:#4957a8}
.small{font-size:12px;color:var(--muted)}
.kpi{display:flex;gap:10px;flex-wrap:wrap;margin-top:8px}
.kpi .box{background:#f8f9ff;border:1px solid var(--edge);border-radius:14px;padding:10px 12px;min-width:220px;flex:1 1 220px}
.kpi .lab{font-size:12px;color:var(--muted)}
.kpi .val{font-size:16px;font-weight:900;color:#2e3568;white-space:normal;word-break:break-word;line-height:1.2}
.table-wrap{overflow:auto;border:1px solid var(--edge);border-radius:14px}
table{width:100%;border-collapse:collapse;background:#fff}
th,td{border-bottom:1px dashed #dfe5ff;padding:8px 10px;font-size:13px;vertical-align:top}
th{text-align:left;color:#3a4688;background:#f8f9ff;position:sticky;top:0;z-index:1}
tr:hover td{background:#fbfcff}
footer{position:fixed;bottom:0;left:0;right:0}
.footer-inner{max-width:var(--maxw);margin:0 auto;padding:4px 16px 6px;border-top:1px solid var(--edge);
  background:#fff;font-size:11px;color:var(--muted)}
.footer-inner::after{content:" · Page " counter(page)}
@page{size:A4;margin:12mm;counter-increment: page;}
</style>
</head>
<body>
<header>
  <div class="header-inner">
    <div class="brand">Ayass Bioscience</div>
    <h1>Cross-Disease Pathway Intelligence Report</h1>
    <div class="muted">Cohort: <strong>{{ cohort_name }}</strong> · Generated: <strong>{{ generated }}</strong></div>
    <div style="margin-top:8px;">
      {% for d in diseases %}
        <span class="pill">{{ d }}</span>
      {% endfor %}
    </div>
    <div class="kpi">
      <div class="box"><div class="lab">Diseases</div><div class="val">{{ diseases|length }}</div></div>
      <div class="box"><div class="lab">Top shared pathway theme</div><div class="val">{{ kpi_top_subclass }}</div></div>
      <div class="box"><div class="lab">Shared pathway themes (≥2)</div><div class="val">{{ kpi_shared_subclasses }}</div></div>
      <div class="box"><div class="lab">Top contrasting pathway</div><div class="val">{{ kpi_top_discord }}</div></div>
    </div>
  </div>
</header>

<div class="container">

  <div class="card">
    <h2>Executive narrative</h2>
    <p style="font-size:16px;margin:0 0 8px;"><strong>{{ hook_headline }}</strong></p>
    <ul>{% for b in hook_bullets %}<li>{{ b }}</li>{% endfor %}</ul>
    <p class="small">Narrative is constrained to the evidence shown in the tables/figures below (association-only).</p>
  </div>

  <div class="card">
    <h2>Disease similarity</h2>
    <div class="two-col">
      <div>
        <div class="img-frame"><img src="{{ sim_heatmap }}" alt="Similarity heatmap"/></div>
        <div class="caption"><strong>Similarity map</strong> derived from pathway-level signals.</div>
      </div>
      <div class="panel">
        <h3>Interpretation</h3>
        <ul>{% for b in sim_bullets %}<li>{{ b }}</li>{% endfor %}</ul>
      </div>
    </div>
  </div>

  <div class="card">
    <h2>Shared pathway themes</h2>
    <div class="two-col">
      <div>
        <div class="img-frame"><img src="{{ subclass_barplot }}" alt="Shared Sub_Class barplot"/></div>
        <div class="caption"><strong>Ranked by</strong> how many diseases show the theme among significant pathways.</div>
      </div>
      <div class="panel">
        <h3>Top 3 shared themes (selected for clinical relevance)</h3>
        <ul>{% for b in subclass_story_bullets %}<li>{{ b }}</li>{% endfor %}</ul>
      </div>
    </div>

    <div class="panel">
      <h3>Evidence</h3>
      <div class="table-wrap">{{ subclass_evidence_table|safe }}</div>
      <p class="small">Representative pathways shown per disease for each theme.</p>
    </div>
  </div>

  <div class="card">
    <h2>Top shared pathways</h2>
    <div class="table-wrap">{{ shared_pathways_table|safe }}</div>
    <div class="panel">
      <h3>Interpretation</h3>
      <ul>{% for b in shared_pathways_bullets %}<li>{{ b }}</li>{% endfor %}</ul>
    </div>
  </div>

  <div class="card">
    <h2>Where the biology diverges</h2>
    <div class="two-col">
      <div>
        <div class="img-frame"><img src="{{ discord_barplot }}" alt="Discordance barplot"/></div>
        <div class="caption"><strong>Highest contrast</strong> pathways that shift direction across diseases (if available).</div>
      </div>
      <div class="panel">
        <h3>Interpretation</h3>
        <ul>{% for b in discord_bullets %}<li>{{ b }}</li>{% endfor %}</ul>
      </div>
    </div>
    <div class="panel">
      <h3>Evidence</h3>
      <div class="table-wrap">{{ discord_table|safe }}</div>
    </div>
  </div>

  <div class="card">
    <h2>Molecular drivers</h2>

    <div class="panel">
      <h3>Transcription factors (evidence-weighted)</h3>
      <div class="table-wrap">{{ tf_table|safe }}</div>
      <p class="small">{{ tf_source_note }}</p>
    </div>

    <div class="panel">
      <h3>Epigenetic factors (evidence-weighted)</h3>
      <div class="table-wrap">{{ epi_table|safe }}</div>
      <p class="small">{{ epi_source_note }}</p>
    </div>

    <div class="panel">
      <h3>Metabolites</h3>
      {% if shared_metabolite_plot %}
        <div class="img-frame"><img src="{{ shared_metabolite_plot }}" alt="Shared metabolites plot"/></div>
        <div class="caption"><strong>Shared metabolites</strong> across diseases (p ≤ 0.1).</div>
      {% else %}
        <p class="small" style="margin:0;">No shared metabolite was detected across all diseases at p ≤ 0.1 (metabolite enrich up/down).</p>
      {% endif %}

      <div style="margin-top:10px;" class="table-wrap">{{ metabolite_table|safe }}</div>
      <p class="small">{{ metabolite_source_note }}</p>
    </div>

    <div class="panel">
      <h3>Interpretation</h3>
      <ul>{% for b in engines_bullets %}<li>{{ b }}</li>{% endfor %}</ul>
    </div>
  </div>

  <div class="card">
    <h2>Same pathway, different mechanism (top 3)</h2>
    <p class="small" style="margin-top:0;">
      For the most shared pathways, we compare which TFs / metabolites / epigenetic factors align vs diverge.
      (This section uses the first two diseases for a clean contrast view when the cohort has many diseases.)
    </p>

    {% for item in pathway_contrast_cards %}
      <div class="panel" style="margin-top:12px;">
        <h3>{{ item.title }}</h3>
        <div class="two-col">
          <div>
            <div class="img-frame"><img src="{{ item.plot }}" alt="Contrast plot"/></div>
            <div class="caption"><strong>Counts</strong> of shared vs disease-specific entities.</div>
          </div>
          <div>
            <ul>
              {% for b in item.bullets %}
                <li>{{ b }}</li>
              {% endfor %}
            </ul>
            <div class="table-wrap" style="margin-top:10px;">{{ item.table|safe }}</div>
          </div>
        </div>
      </div>
    {% endfor %}
  </div>

  <div class="card">
    <h2>Key takeaways</h2>
    <ul>{% for b in takeaways %}<li>{{ b }}</li>{% endfor %}</ul>
    <div class="panel">
      <h3>Conclusion</h3>
      <p style="margin:0;">{{ conclusion }}</p>
    </div>
  </div>

</div>

<footer><div class="footer-inner">Ayass Bioscience · MDP Report Plan2</div></footer>
</body>
</html>
""")


# =========================
# === Main builder
# =========================

def timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M")


def build_report(counts_root: Path, no_llm: bool = False, q_cutoff: float = 0.05):
    cohort_name = counts_root.name
    outdir = counts_root / "Report"
    ensure_dir(outdir)

    # --- disease discovery (robust)
    disease_dirs = discover_disease_dirs(counts_root)
    diseases = sorted(disease_dirs.keys())
    if not diseases:
        raise RuntimeError(
            "No diseases found. Expected gsea_prerank_classified.tsv under either:\n"
            "  <root>/<Disease>/\n"
            "  <root>/GL_enrich/<Disease>/\n"
            "  <root>/GC_enrich/<Disease>/\n"
            "  <root>/*_enrich/<Disease>/\n"
        )
    print(f"[INFO] Detected {len(diseases)} diseases: {', '.join(diseases)}")

    # similarity
    sim_path = counts_root / "ANALYSIS_category_landscape" / "tables" / "disease_similarity_cosine.tsv"
    sim_df = read_table(sim_path)

    # discordance
    concord_path = counts_root / "INSIGHTS_out" / "tables" / "directional_concordance_by_pathway.csv"
    concord_df = read_table(concord_path)

    # per-disease GSEA classified (FIXED: no df truthiness)
    disease_to_gsea: Dict[str, pd.DataFrame] = {}
    for d in diseases:
        p = disease_dirs[d] / "gsea_prerank_classified.tsv"
        raw = read_table(p)
        if raw is None:
            raw = pd.DataFrame()
        disease_to_gsea[d] = normalize_gsea_classified(raw)

    # shared subclass shortlist + evidence
    subclass_shortlist, subclass_best_long = compute_shared_subclass_shortlist(
        disease_to_gsea, q_cutoff=q_cutoff, top_n=30
    )

    if subclass_shortlist is None or subclass_shortlist.empty:
        subclass_counts = pd.Series({"Unclassified": 1})
        kpi_top_subclass = "NA"
        kpi_shared_subclasses = 0
    else:
        subclass_counts = pd.Series(
            data=subclass_shortlist["n_diseases"].values,
            index=subclass_shortlist["Sub_Class"].values
        )
        kpi_top_subclass = str(subclass_shortlist.iloc[0]["Sub_Class"])
        kpi_shared_subclasses = int((subclass_shortlist["n_diseases"] >= 2).sum())

    subclass_barplot = make_barh(
        f"Shared pathway themes (q ≤ {q_cutoff})",
        subclass_counts,
        xlabel="# diseases",
        top_n=12
    )

    if subclass_best_long is not None and not subclass_best_long.empty:
        top_subs = set(subclass_shortlist["Sub_Class"].head(12).tolist()) if not subclass_shortlist.empty else set()
        ev = subclass_best_long[subclass_best_long["Sub_Class"].isin(top_subs)].copy()
        ev = ev[ev["Best_Pathway"].map(is_allowed_pathway)].copy()
        ev = ev[["Sub_Class", "Disease", "Best_Pathway", "Direction", "NES", "qval"]]
        ev["__rank"] = ev["Sub_Class"].map({s: i for i, s in enumerate(subclass_shortlist["Sub_Class"].tolist())})
        ev = ev.sort_values(["__rank", "Disease"]).drop(columns=["__rank"])
    else:
        ev = pd.DataFrame({"Sub_Class": [], "Disease": [], "Best_Pathway": [], "Direction": [], "NES": [], "qval": []})

    subclass_evidence_table = make_table(ev, max_rows=60)

    # similarity heatmap with fallback to INSIGHTS shared image
    sim_heatmap = similarity_fallback_data_uri(counts_root) or make_similarity_heatmap(sim_df, title="Disease similarity")

    # discordance
    disc_df = compute_discordant_pathways(concord_df, top_n=12)
    if disc_df is None or disc_df.empty:
        discord_barplot = make_barh("Top contrasting pathways", pd.Series({"NA": 1}), xlabel="contrast score", top_n=1)
        discord_table = make_table(pd.DataFrame({"Pathway": ["NA"], "Discordance": [1]}), max_rows=8)
        kpi_top_discord = "NA"
    else:
        discord_barplot = make_barh(
            "Top contrasting pathways",
            disc_df.set_index("Pathway")["Discordance"],
            xlabel="contrast score",
            top_n=12
        )
        discord_table = make_table(disc_df, max_rows=12)
        kpi_top_discord = str(disc_df.iloc[0]["Pathway"])

    # shared pathways table (preferred top200)
    shared_pathways_df = load_shared_pathways_table(counts_root)
    if shared_pathways_df is None or shared_pathways_df.empty:
        shared_pathways_df = pd.DataFrame({"Pathway": [], "#Diseases": []})
    else:
        shared_pathways_df = shared_pathways_df[shared_pathways_df["Pathway"].map(is_allowed_pathway)].copy()
    shared_pathways_table = make_table(shared_pathways_df, max_rows=18)

    # overlap jsons (robust discovery)
    disease_to_json: Dict[str, Dict[str, Any]] = {}
    json_sources: Dict[str, str] = {}
    for d in diseases:
        jp = discover_overlap_json_for_disease(counts_root, d, disease_dirs[d])
        if jp is not None and jp.exists():
            disease_to_json[d] = parse_overlap_json(jp)
            json_sources[d] = str(jp)

    tf_rows_payload: Dict[str, Any] = {}
    epi_rows_payload: Dict[str, Any] = {}
    metabolite_payload: Dict[str, Any] = {}

    tf_source_note = ""
    epi_source_note = ""
    metabolite_source_note = ""

    # Always prepare fallbacks per disease
    fallback_entities = {}
    for d in diseases:
        ddir = disease_dirs[d]
        fb = _fallback_entities_from_tables(ddir)

        if not fb["tf"] and not fb["epigenetic"] and not fb["metabolite"]:
            fb2 = _best_effort_from_all_combined(ddir)
            for k in fb:
                fb[k] = fb[k] or fb2.get(k, [])

        fallback_entities[d] = fb

    if disease_to_json:
        tf_rank = compute_entity_rankings_from_json(disease_to_json, diseases, "tf", "ANY")
        epi_rank = compute_entity_rankings_from_json(disease_to_json, diseases, "epigenetic", "ANY")
        met_rank = compute_entity_rankings_from_json(disease_to_json, diseases, "metabolite", "ANY")

        for d in diseases:
            if tf_rank.get(d, pd.Series(dtype=float)).empty and fallback_entities[d]["tf"]:
                tf_rank[d] = pd.Series({x: 1.0 for x in fallback_entities[d]["tf"]}).sort_values(ascending=False)
            if epi_rank.get(d, pd.Series(dtype=float)).empty and fallback_entities[d]["epigenetic"]:
                epi_rank[d] = pd.Series({x: 1.0 for x in fallback_entities[d]["epigenetic"]}).sort_values(ascending=False)
            if met_rank.get(d, pd.Series(dtype=float)).empty and fallback_entities[d]["metabolite"]:
                met_rank[d] = pd.Series({x: 1.0 for x in fallback_entities[d]["metabolite"]}).sort_values(ascending=False)

        tf_shared = shared_entities_across_all_diseases(tf_rank, diseases, top_n=10)
        epi_shared = shared_entities_across_all_diseases(epi_rank, diseases, top_n=10)
        met_shared = shared_entities_across_all_diseases(met_rank, diseases, top_n=10)

        tf_rows = []
        if not tf_shared.empty:
            tf_rows.append({"Section": "Shared (all diseases)", "Entities (top 10, evidence-weighted)": ", ".join(tf_shared.index.tolist())})
        for dis in diseases:
            s = tf_rank.get(dis, pd.Series(dtype=float)).head(10)
            tf_rows.append({"Section": f"{dis}", "Entities (top 10, evidence-weighted)": ", ".join([str(x) for x in s.index.tolist()]) if not s.empty else "—"})
        tf_table_html = make_table(pd.DataFrame(tf_rows), max_rows=60)
        tf_rows_payload = {"shared": tf_shared.to_dict(), "per_disease": {d: tf_rank[d].head(10).to_dict() for d in diseases}}
        tf_source_note = "Primary: pathway–entity overlap JSON (evidence-weighted by overlap-gene support). Entities are normalized to prevent metadata dumps."

        epi_rows = []
        if not epi_shared.empty:
            epi_rows.append({"Section": "Shared (all diseases)", "Entities (top 10, evidence-weighted)": ", ".join(epi_shared.index.tolist())})
        for dis in diseases:
            s = epi_rank.get(dis, pd.Series(dtype=float)).head(10)
            epi_rows.append({"Section": f"{dis}", "Entities (top 10, evidence-weighted)": ", ".join([str(x) for x in s.index.tolist()]) if not s.empty else "—"})
        epi_table_html = make_table(pd.DataFrame(epi_rows), max_rows=60)
        epi_rows_payload = {"shared": epi_shared.to_dict(), "per_disease": {d: epi_rank[d].head(10).to_dict() for d in diseases}}
        epi_source_note = "Primary: pathway–entity overlap JSON (schema-flex). Entities normalized to canonical histone marks when possible."

        met_rows = []
        if not met_shared.empty:
            met_rows.append({"Section": "Shared (all diseases)", "Metabolites (top 10, evidence-weighted)": ", ".join(met_shared.index.tolist())})
        for dis in diseases:
            s = met_rank.get(dis, pd.Series(dtype=float)).head(10)
            met_rows.append({"Section": f"{dis}", "Metabolites (top 10, evidence-weighted)": ", ".join([str(x) for x in s.index.tolist()]) if not s.empty else "—"})
        metabolite_table_html = make_table(pd.DataFrame(met_rows), max_rows=60)
        metabolite_payload = {"shared": met_shared.to_dict(), "per_disease": {d: met_rank[d].head(10).to_dict() for d in diseases}}
        metabolite_source_note = "Primary: overlap JSON; fallback: metabolites_sig/all.tsv or ALL_COMBINED.csv."

    else:
        tf_rows, epi_rows, met_rows = [], [], []
        for dis in diseases:
            fb = fallback_entities[dis]
            tf_rows.append({"Section": f"{dis}", "Entities (top 10)": ", ".join(fb["tf"][:10]) if fb["tf"] else "—"})
            epi_rows.append({"Section": f"{dis}", "Entities (top 10)": ", ".join(fb["epigenetic"][:10]) if fb["epigenetic"] else "—"})
            met_rows.append({"Section": f"{dis}", "Metabolites (top 10)": ", ".join(fb["metabolite"][:10]) if fb["metabolite"] else "—"})
        tf_table_html = make_table(pd.DataFrame(tf_rows), max_rows=60)
        epi_table_html = make_table(pd.DataFrame(epi_rows), max_rows=60)
        metabolite_table_html = make_table(pd.DataFrame(met_rows), max_rows=60)

        tf_source_note = "Overlap JSON not found; using per-disease tf_sig/all.tsv (or ALL_COMBINED.csv fallback)."
        epi_source_note = "Overlap JSON not found; using per-disease epigenetic_sig/all.tsv (or ALL_COMBINED.csv fallback)."
        metabolite_source_note = "Overlap JSON not found; using per-disease metabolites_sig/all.tsv (or ALL_COMBINED.csv fallback)."

    # Shared metabolites plot from metabolite_enrich_up/down.csv (p <= 0.1)
    shared_mets = compute_shared_metabolites_from_enrich(disease_dirs, diseases, pval_cutoff=0.1)
    shared_metabolite_plot = None
    if not shared_mets.empty:
        shared_metabolite_plot = make_barh(
            "Shared metabolites (p ≤ 0.1)",
            pd.Series({k: v for k, v in shared_mets.items()}),
            xlabel="# diseases",
            top_n=min(10, len(shared_mets))
        )

    # Same pathway, different mechanism (candidate pool from shared pathways top 12)
    pathway_contrast_cards: List[Dict[str, Any]] = []
    if len(diseases) >= 2 and disease_to_json and shared_pathways_df is not None and not shared_pathways_df.empty:
        candidate_pathways = [p for p in shared_pathways_df["Pathway"].head(12).tolist() if is_allowed_pathway(p)]
        all_cards = compute_pathway_entity_contrast_cards(diseases, disease_to_json, candidate_pathways)
        pathway_contrast_cards = all_cards[:3]

    # -------------------------
    # LLM narratives (grounded) — KEEP ENABLED; NEVER FAIL
    # -------------------------
    narrative = ensure_report_narrative_defaults({}, diseases=diseases, q_cutoff=q_cutoff)

    if not no_llm:
        subclass_payload = []
        if subclass_shortlist is not None and not subclass_shortlist.empty:
            for _, row in subclass_shortlist.head(30).iterrows():
                sub = str(row["Sub_Class"])
                per_dis = []
                if subclass_best_long is not None and not subclass_best_long.empty:
                    sub_rows = subclass_best_long[subclass_best_long["Sub_Class"] == sub]
                    for _, rr in sub_rows.iterrows():
                        per_dis.append({
                            "disease": str(rr.get("Disease", "")),
                            "best_pathway": str(rr.get("Best_Pathway", "")),
                            "NES": str(rr.get("NES", "")),
                            "qval": str(rr.get("qval", "")),
                            "direction": str(rr.get("Direction", "")),
                        })
                subclass_payload.append({
                    "subclass": sub,
                    "n_diseases": int(row.get("n_diseases", 0)),
                    "per_disease": per_dis[:10],
                })

        shared_pathways_payload = shared_pathways_df.head(12).to_dict(orient="records") if shared_pathways_df is not None and not shared_pathways_df.empty else []
        disc_payload = disc_df.head(12).to_dict(orient="records") if disc_df is not None and not disc_df.empty else []

        # Provide only CLEAN driver names (already sanitized in ranking)
        evidence = {
            "diseases": diseases,
            "q_cutoff": q_cutoff,
            "shared_themes_top10": subclass_payload[:10],
            "shared_pathways_top12": shared_pathways_payload,
            "contrasting_pathways_top12": disc_payload,
            "drivers_top10": {
                "tf": tf_rows_payload,
                "epigenetic": epi_rows_payload,
                "metabolite": metabolite_payload,
            },
            "shared_metabolites_from_enrich_p_le_0p1": shared_mets.head(15).to_dict() if not shared_mets.empty else {},
        }

        llm_out = llm_json(
            system=(
                "You are a blunt clinician-facing translational bioinformatics report writer.\n"
                "You MUST link evidence into a coherent story (shared label vs divergent upstream program).\n"
                "Hard constraints:\n"
                "- Association only (no causality claims).\n"
                "- Use ONLY item names in the evidence.\n"
                "- Do NOT output PMIDs, study IDs, cell lines, or ChIP-Seq metadata.\n"
                "- Do NOT mention olfactory/sensory/taste/odorant/phototransduction pathways.\n"
                "Return ONLY JSON."
            ),
            prompt=textwrap.dedent(f"""
            Write a clinically sensible story for this cohort.

            Evidence (cleaned/normalized):
            {json.dumps(evidence, indent=2)[:9000]}

            Required JSON:
            {{
              "hook_headline": "1 sentence: specific, not generic; name ≥1 theme/pathway",
              "hook_bullets": ["3-6 bullets; each bullet must name ≥1 concrete item AND state why it matters clinically"],
              "sim_bullets": ["2-5 bullets; explain what similarity suggests (shared biology) and what it does NOT mean"],
              "subclass_story_bullets": ["Theme 1: interpret + cite representative pathway", "Theme 2: ...", "Theme 3: ..."],
              "shared_pathways_bullets": ["3-8 bullets: name shared pathways and connect to themes + possible shared axis"],
              "discord_bullets": ["3-8 bullets: name contrasting pathways and explain stratification/clinical split"],
              "engines_bullets": ["4-10 bullets: connect TF/epi/metabolite signals to shared vs diverging biology; label as hypothesis"],
              "pathway_contrast_bullets": {{
                "Pathway1": ["3-6 bullets contrasting drivers (TF/epi/met) as upstream-program hypothesis"],
                "Pathway2": ["3-6 bullets ..."],
                "Pathway3": ["3-6 bullets ..."]
              }},
              "takeaways": ["5-10 bullets: actionable, includes validation & biomarker/target hypotheses"],
              "conclusion": "2-5 sentences; name ≥3 concrete items; end with validation plan"
            }}

            Constraints:
            - Every bullet must include at least one named evidence item.
            - No dumping lists; each bullet must interpret the list (why/so-what).
            - No olfactory/sensory content.
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
                if isinstance(v, dict) and len(v) == 0:
                    continue
                narrative[k] = v

    narrative = ensure_report_narrative_defaults(narrative, diseases=diseases, q_cutoff=q_cutoff)

    hook_headline = clean_name(narrative.get("hook_headline", ""))
    hook_bullets = _list_clean(narrative.get("hook_bullets", []), 10)
    sim_bullets = _list_clean(narrative.get("sim_bullets", []), 10)
    subclass_story_bullets = _list_clean(narrative.get("subclass_story_bullets", []), 5)
    shared_pathways_bullets = _list_clean(narrative.get("shared_pathways_bullets", []), 12)
    discord_bullets = _list_clean(narrative.get("discord_bullets", []), 12)
    engines_bullets = _list_clean(narrative.get("engines_bullets", []), 14)
    takeaways = _list_clean(narrative.get("takeaways", []), 14)
    conclusion = clean_name(narrative.get("conclusion", ""))

    # -------------------------
    # SAME PATHWAY / DIFFERENT MECH bullets:
    # deterministic biolinkbert.py first, then optional LLM refine
    # -------------------------
    if len(diseases) >= 2 and disease_to_json and pathway_contrast_cards:
        disease_a = diseases[0]
        disease_b = diseases[1]
        json_a = disease_to_json.get(disease_a, {})
        json_b = disease_to_json.get(disease_b, {})

        for c in pathway_contrast_cards:
            pw = c.get("pathway", "")
            if not pw:
                c["bullets"] = [
                    "Contrast bullets unavailable (missing pathway key).",
                    "Interpretation remains association-only.",
                    "Validate with an independent cohort."
                ]
                continue

            seed = []
            debug = {}

            if _HAS_BIOLINKBERT and build_contrast_bullets is not None:
                det = build_contrast_bullets(
                    pathway=pw,
                    disease_a=disease_a,
                    disease_b=disease_b,
                    json_a=json_a,
                    json_b=json_b,
                    nonhuman_pat=NON_HUMAN_PAT,
                    max_bullets=5,
                )
                seed = det.bullets or []
                debug = det.debug or {}
            else:
                # fallback seed (still clean)
                raw = c.get("raw", {})
                shared_tf = raw.get("shared", {}).get("tf", [])[:3]
                only_a_tf = raw.get("only_a", {}).get("tf", [])[:3]
                only_b_tf = raw.get("only_b", {}).get("tf", [])[:3]
                seed = [
                    f"{pw}: shared pathway label supported by shared TF scaffold ({', '.join(shared_tf) or 'none'}) (association).",
                    f"{disease_a}-leaning axis: {', '.join(only_a_tf) or 'no unique TF signal detected'} (hypothesis).",
                    f"{disease_b}-leaning axis: {', '.join(only_b_tf) or 'no unique TF signal detected'} (hypothesis).",
                    "Validation: reproduce driver contrast in an independent cohort; confirm overlap genes track the listed upstream signals.",
                ]

            if not seed:
                seed = [
                    f"{pw}: contrast evidence available; interpret as association-only.",
                    "Use the table to compare shared vs disease-specific drivers.",
                    "Validate in an independent cohort."
                ]

            if no_llm:
                c["bullets"] = seed[:5]
            else:
                c["bullets"] = llm_refine_contrast_bullets(
                    pathway=pw,
                    disease_a=disease_a,
                    disease_b=disease_b,
                    seed_bullets=seed,
                    debug=debug,
                    max_bullets=5,
                )

            c["bullets"] = [clean_name(b) for b in c["bullets"] if clean_name(b)]
    else:
        for c in pathway_contrast_cards:
            c["bullets"] = [
                "Driver contrast is shown in the table; interpret shared vs disease-specific mechanisms.",
                "Use disease-specific drivers as hypotheses for follow-up validation.",
                "Interpretation remains association-only."
            ]

    # Render HTML
    html = HTML.render(
        cohort_name=cohort_name,
        generated=timestamp(),
        diseases=diseases,
        kpi_top_subclass=kpi_top_subclass,
        kpi_shared_subclasses=kpi_shared_subclasses,
        kpi_top_discord=kpi_top_discord,
        hook_headline=hook_headline,
        hook_bullets=hook_bullets,
        sim_heatmap=sim_heatmap,
        sim_bullets=sim_bullets,
        subclass_barplot=subclass_barplot,
        subclass_story_bullets=subclass_story_bullets,
        subclass_evidence_table=subclass_evidence_table,
        shared_pathways_table=shared_pathways_table,
        shared_pathways_bullets=shared_pathways_bullets,
        discord_barplot=discord_barplot,
        discord_bullets=discord_bullets,
        discord_table=discord_table,
        tf_table=tf_table_html,
        epi_table=epi_table_html,
        shared_metabolite_plot=shared_metabolite_plot,
        metabolite_table=metabolite_table_html,
        engines_bullets=engines_bullets,
        pathway_contrast_cards=pathway_contrast_cards,
        takeaways=takeaways,
        conclusion=conclusion,
        tf_source_note=tf_source_note,
        epi_source_note=epi_source_note,
        metabolite_source_note=metabolite_source_note,
    )

    out_html = outdir / "index.html"
    out_html.write_text(html, encoding="utf-8")
    print(f"[INFO] Report HTML written to: {out_html}")

    artifact = {
        "cohort": cohort_name,
        "generated": timestamp(),
        "diseases": diseases,
        "disease_dirs": {d: str(disease_dirs[d]) for d in diseases},
        "q_cutoff": q_cutoff,
        "filters": {
            "removed_pathway_patterns": "olfactory|odorant|smell|taste|phototransduction|sensory perception|sensory system",
            "removed_nonhuman_entities_patterns": "mouse|mice|rat|mm9|mm10|murine|c2c12"
        },
        "kpis": {
            "top_shared_theme": kpi_top_subclass,
            "shared_themes_ge2": kpi_shared_subclasses,
            "top_contrasting_pathway": kpi_top_discord
        },
        "overlap_json_sources": json_sources,
        "shared_metabolites_from_enrich_p_le_0p1": list(shared_mets.index)[:30] if not shared_mets.empty else [],
        "paths_checked": {
            "similarity_table": str(sim_path),
            "similarity_fallback_image_used": bool(similarity_fallback_data_uri(counts_root)),
            "direction_concordance": str(concord_path),
        },
        "contrast_bullets_source": "biolinkbert.py (deterministic) + optional LLM refine" if _HAS_BIOLINKBERT else "fallback + optional LLM refine",
    }
    (outdir / "report_artifact.json").write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    print(f"[INFO] Debug artifact written to: {outdir / 'report_artifact.json'}")

    pdf_name = f"Report_{cohort_name}_{datetime.now().strftime('%Y-%m-%d_%H-%M')}.pdf"
    pdf_path = outdir / pdf_name
    if write_pdf(out_html, pdf_path):
        print(f"[INFO] PDF written to: {pdf_path}")
    else:
        print("[WARN] Could not create PDF. Install Chrome/Chromium/Edge OR `pip install weasyprint`.")


# ========= CLI =========

def parse_args():
    p = argparse.ArgumentParser(description="Generate MDP Cross-Disease Report Plan2 v4.3 (HTML + PDF).")
    p.add_argument("--counts-root", "--counts_root", required=True, dest="counts_root",
                   help="Counts/GL/GC output root (e.g., /mnt/d/DEGs_run_099).")
    p.add_argument("--no-llm", action="store_true", help="Disable LLM-based interpretation.")
    p.add_argument("--q-cutoff", type=float, default=0.05, help="q/FDR cutoff for theme presence.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_report(Path(args.counts_root), no_llm=bool(args.no_llm), q_cutoff=float(args.q_cutoff))
