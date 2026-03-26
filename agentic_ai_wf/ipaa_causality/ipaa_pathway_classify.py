#!/usr/bin/env python3
"""
ipaa_pathway_classify.py

Robust pathway classification for IPAA.

Fixes:
- Cohort dir resolution: supports cohort_dir/out_dir + fallbacks.
- Input fallback: prefers pathway_stats_with_baseline.tsv, else uses pathway_stats.tsv.
- No-surprises: does not crash pipeline; produces classified tables whenever possible.
"""

from __future__ import annotations

import csv
import logging
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import pandas as pd
import openai


LOG = logging.getLogger("ipaa_pathway_classify")
if not LOG.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    LOG.addHandler(_h)
LOG.setLevel(logging.INFO)


# ----------------------------
# Robust cohort dir resolution
# ----------------------------

def _resolve_cohort_dir(out_root: Path, cohort_name: str, cr: Dict[str, str]) -> Optional[Path]:
    """
    Resolve cohort directory robustly.

    Accepts keys:
      - cohort_dir (from m6_processing)
      - out_dir (legacy)
    Then falls back to:
      - out_root/cohorts/<name>
      - out_root/<name>

    Picks the first directory that exists AND contains at least one sentinel file.
    """
    out_root = out_root.resolve()

    cand_strs: List[str] = []
    for k in ("cohort_dir", "out_dir", "cohort_path", "dir"):
        v = (cr.get(k) or "").strip()
        if v:
            cand_strs.append(v)

    candidates: List[Path] = []
    for s in cand_strs:
        try:
            candidates.append(Path(s).expanduser().resolve())
        except Exception:
            continue

    candidates.append(out_root / "cohorts" / cohort_name)
    candidates.append(out_root / cohort_name)

    sentinels = (
        "pathway_stats_with_baseline.tsv",
        "pathway_stats.tsv",
        "pathway_activity.tsv",
    )

    for d in candidates:
        if not d.exists() or not d.is_dir():
            continue
        if any((d / s).exists() for s in sentinels):
            return d

    for d in candidates:
        if d.exists() and d.is_dir():
            return d

    return None


# ----------------------------
# Utilities: schema inference
# ----------------------------

def _guess_col(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    cols = {c.lower(): c for c in df.columns}
    for cand in candidates:
        c = cols.get(cand.lower())
        if c:
            return c
    return None


def _infer_fdr_col(df: pd.DataFrame) -> Optional[str]:
    return _guess_col(
        df,
        ["fdr", "fdr_q", "fdr q-val", "fdr.q.val", "qval", "q-value", "padj", "adj p", "adj_p"],
    )


def _infer_pathway_col(df: pd.DataFrame) -> Optional[str]:
    return _guess_col(df, ["pathway", "term", "name", "geneset", "gene set"])


# ----------------------------
# Background pathway filtering
# ----------------------------

def _jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def build_reduced_overlap_backbone(
    gmt: Dict[str, Set[str]],
    expressed_genes: Optional[Set[str]] = None,
    min_genes_expressed: int = 10,
    reduce_overlap: bool = True,
    max_pathway_jaccard: float = 0.50,
) -> Tuple[Set[str], Dict[str, Set[str]]]:
    filtered: Dict[str, Set[str]] = {}
    for p, genes in gmt.items():
        if not genes:
            continue
        if expressed_genes is None:
            filtered[p] = set(genes)
        else:
            g = set(genes) & set(expressed_genes)
            if len(g) >= int(min_genes_expressed):
                filtered[p] = g

    if not reduce_overlap:
        return set(filtered.keys()), filtered

    items = sorted(filtered.items(), key=lambda kv: len(kv[1]), reverse=True)
    kept: Dict[str, Set[str]] = {}
    for p, genes in items:
        ok = True
        for _, kgenes in kept.items():
            if _jaccard(genes, kgenes) > float(max_pathway_jaccard):
                ok = False
                break
        if ok:
            kept[p] = genes

    return set(kept.keys()), kept


# ----------------------------
# Simple cache (CSV)
# ----------------------------

@dataclass(frozen=True)
class ClassRecord:
    main: str
    sub: str
    source: str
    confidence: float
    created_at: str


def _ensure_cache(csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if csv_path.exists():
        return
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["pathway", "main_class", "sub_class", "source", "confidence", "created_at"],
        )
        w.writeheader()


def _load_cache(csv_path: Path) -> Dict[str, ClassRecord]:
    _ensure_cache(csv_path)
    out: Dict[str, ClassRecord] = {}
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            p = (row.get("pathway") or "").strip()
            if not p:
                continue
            out[p] = ClassRecord(
                main=(row.get("main_class") or "").strip(),
                sub=(row.get("sub_class") or "").strip(),
                source=(row.get("source") or "").strip(),
                confidence=float(row.get("confidence") or 0.0),
                created_at=(row.get("created_at") or "").strip(),
            )
    return out


def _upsert_cache(csv_path: Path, pathway: str, main: str, sub: str, source: str, confidence: float) -> None:
    cache = _load_cache(csv_path)
    cache[pathway] = ClassRecord(
        main=main,
        sub=sub,
        source=source,
        confidence=float(confidence),
        created_at=datetime.now().isoformat(),
    )
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["pathway", "main_class", "sub_class", "source", "confidence", "created_at"],
        )
        w.writeheader()
        for p, rec in cache.items():
            w.writerow(
                {
                    "pathway": p,
                    "main_class": rec.main,
                    "sub_class": rec.sub,
                    "source": rec.source,
                    "confidence": rec.confidence,
                    "created_at": rec.created_at,
                }
            )


# ----------------------------
# Ontology inference + rules
# ----------------------------

_CANON_MAIN = [
    "Metabolism",
    "Signal Transduction",
    "Immune System",
    "Cell Cycle",
    "Cancer",
    "Apoptosis",
    "Developmental Biology",
    "Genetic Information Processing",
    "Environmental Information Processing",
    "Cellular Processes",
    "Disease Mechanisms",
]


def infer_ontology_from_name(pathway: str) -> str:
    p = (pathway or "").strip()
    if not p:
        return "UNKNOWN"
    m = re.match(r"^([A-Za-z0-9]+)[\s:_\-]+", p)
    if m:
        token = m.group(1).upper()
        if token in {"REACTOME", "KEGG", "PID", "BIOCARTA", "HALLMARK", "WIKIPATHWAYS", "WP"}:
            return token
    up = p.upper()
    if "REACTOME" in up:
        return "REACTOME"
    if "KEGG" in up:
        return "KEGG"
    if "BIOCARTA" in up:
        return "BIOCARTA"
    if "PID" in up:
        return "PID"
    if "GO_" in up or up.startswith("GO "):
        return "GO"
    return "MSIGDB"


def _rules_classify(pathway: str) -> Tuple[str, str]:
    s = (pathway or "").lower()

    if any(k in s for k in ["immune", "cytokine", "interleukin", "tnf", "nfkb", "t cell", "b cell", "mhc", "antigen"]):
        return "Immune System", "Immune signaling / inflammation"

    if any(k in s for k in ["metabol", "glycol", "oxid", "tca", "lipid", "cholesterol", "fatty acid", "amino acid"]):
        return "Metabolism", "Core metabolic processes"

    if "apopt" in s or "cell death" in s:
        return "Apoptosis", "Programmed cell death"

    if any(k in s for k in ["cell cycle", "mitosis", "checkpoint", "g1", "g2", "s phase", "dna replication"]):
        return "Cell Cycle", "Cell-cycle control / replication"

    if any(k in s for k in ["transcription", "translation", "ribosome", "splice", "rna processing", "dna repair"]):
        return "Genetic Information Processing", "Gene expression / maintenance"

    if any(k in s for k in ["signaling", "mapk", "pi3k", "jak", "stat", "wnt", "notch", "tgf", "hedgehog", "yap", "taz"]):
        return "Signal Transduction", "Cell signaling cascades"

    if any(k in s for k in ["cancer", "tumor", "oncogen", "metast"]):
        return "Cancer", "Oncogenic programs"

    if any(k in s for k in ["development", "differentiation", "morphogen", "neurogenesis"]):
        return "Developmental Biology", "Development / differentiation"

    if any(k in s for k in ["extracellular matrix", "ecm", "focal adhesion", "integrin", "transport", "ion channel", "synapse"]):
        return "Environmental Information Processing", "Cell-environment interface"

    return "Cellular Processes", "General cellular processes"


# ----------------------------
# Optional LLM classifier
# ----------------------------

def _openai_client_available() -> bool:
    if not os.getenv("OPENAI_API_KEY"):
        return False
    return True


def _llm_batch_classify(
    items: List[Tuple[str, str]],
    model: str = "gpt-4o-mini",
    batch_size: int = 8,
    sleep_s: float = 0.8,
) -> Dict[str, Tuple[str, str]]:
    

    results: Dict[str, Tuple[str, str]] = {}
    client = openai.OpenAI()

    for i in range(0, len(items), batch_size):
        batch = items[i : i + batch_size]
        lines = "\n".join([f"{j+1}. {p} ({ont})" for j, (p, ont) in enumerate(batch)])
        prompt = f"""
You are a biomedical ontology expert. Classify each pathway into a Main_Class and Sub_Class.

Main_Class must be exactly one of:
{chr(10).join(["- " + x for x in _CANON_MAIN])}

Rules:
- Sub_Class must be specific and NOT just the pathway name.
- Do not leave any field empty.

Pathways:
{lines}

Return STRICTLY one line per pathway:
1. Main_Class: <...> | Sub_Class: <...>
...
"""
        try:
            res = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=1200,
            )
            txt = (res.choices[0].message.content or "").strip()
        except Exception as e:
            LOG.warning("LLM batch failed (%s). Falling back to rules for this batch.", e)
            for p, _ in batch:
                results[p] = _rules_classify(p)
            continue

        parsed: List[Tuple[str, str]] = []
        for raw in txt.splitlines():
            if "|" not in raw:
                continue
            parts = [x.strip() for x in raw.split("|", 1)]
            if len(parts) != 2:
                continue
            main_part, sub_part = parts
            main = main_part.split(":", 1)[-1].strip()
            sub = sub_part.split(":", 1)[-1].strip()
            main = re.sub(r"^\d+\.?\s*", "", main).strip()
            sub = re.sub(r"^\d+\.?\s*", "", sub).strip()
            if main and sub:
                parsed.append((main, sub))

        if len(parsed) != len(batch):
            LOG.warning("LLM parse mismatch: got %d/%d lines. Using rules for remaining.", len(parsed), len(batch))

        for idx, (p, _) in enumerate(batch):
            if idx < len(parsed):
                results[p] = parsed[idx]
            else:
                results[p] = _rules_classify(p)

        time.sleep(float(sleep_s))

    return results


# ----------------------------
# Core classification
# ----------------------------

def classify_pathway_table(
    df: pd.DataFrame,
    gmt: Dict[str, Set[str]],
    expressed_genes: Optional[Set[str]],
    reduce_overlap: bool,
    max_pathway_jaccard: float,
    min_genes_expressed: int,
    sig_fdr: float,
    sig_top_n: int,
    cache_csv: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = df.copy()

    pcol = _infer_pathway_col(df)
    qcol = _infer_fdr_col(df)
    if not pcol:
        raise ValueError(f"Could not infer pathway column. Columns={list(df.columns)}")
    if not qcol:
        raise ValueError(f"Could not infer FDR/q column. Columns={list(df.columns)}")

    df[pcol] = df[pcol].astype(str)
    df[qcol] = pd.to_numeric(df[qcol], errors="coerce")
    df = df.dropna(subset=[pcol, qcol])

    kept, _ = build_reduced_overlap_backbone(
        gmt=gmt,
        expressed_genes=expressed_genes,
        min_genes_expressed=int(min_genes_expressed),
        reduce_overlap=bool(reduce_overlap),
        max_pathway_jaccard=float(max_pathway_jaccard),
    )

    df_bg = df[df[pcol].isin(kept)].copy()
    if df_bg.empty:
        LOG.warning(
            "After background filter, 0 pathways matched GMT keys. "
            "Proceeding without background filter (likely name mismatch)."
        )
        df_bg = df.copy()

    df_sig = df_bg[df_bg[qcol] <= float(sig_fdr)].copy()
    if df_sig.empty:
        df_sig = df_bg.nsmallest(max(int(sig_top_n), 50), qcol) if len(df_bg) else df_bg
    else:
        if int(sig_top_n) > 0 and len(df_sig) > int(sig_top_n):
            df_sig = df_sig.nsmallest(int(sig_top_n), qcol)

    cache = _load_cache(cache_csv)
    unique_paths = sorted(df_sig[pcol].astype(str).unique().tolist())
    todo = [p for p in unique_paths if p not in cache or not cache[p].main or not cache[p].sub]

    LOG.info("Pathways after filters: %d rows; %d unique; %d need classification", len(df_sig), len(unique_paths), len(todo))

    cls_map: Dict[str, Tuple[str, str]] = {}
    for p in unique_paths:
        if p in cache and cache[p].main and cache[p].sub:
            cls_map[p] = (cache[p].main, cache[p].sub)

    if todo:
        if _openai_client_available():
            model = os.getenv("IPAA_CLASSIFY_MODEL", "gpt-4o-mini")
            items = [(p, infer_ontology_from_name(p)) for p in todo]
            batch_res = _llm_batch_classify(items, model=model, batch_size=8)
            for p, (main, sub) in batch_res.items():
                main = main.strip() or "Cellular Processes"
                sub = sub.strip() or "Unknown Process"
                cls_map[p] = (main, sub)
                _upsert_cache(cache_csv, p, main, sub, "LLM", 0.85)
        else:
            for p in todo:
                main, sub = _rules_classify(p)
                cls_map[p] = (main, sub)
                _upsert_cache(cache_csv, p, main, sub, "RULES", 0.60)

    df_out = df_sig.copy()
    df_out["Ontology_Source"] = df_out[pcol].map(infer_ontology_from_name)
    df_out["Main_Class"] = df_out[pcol].map(lambda x: cls_map.get(str(x), _rules_classify(str(x)))[0])
    df_out["Sub_Class"] = df_out[pcol].map(lambda x: cls_map.get(str(x), _rules_classify(str(x)))[1])

    return df_sig, df_out


def classify_one_cohort(
    cohort_dir: Path,
    gmt: Dict[str, Set[str]],
    cache_csv: Path,
    expressed_genes: Optional[Set[str]] = None,
    reduce_overlap: bool = True,
    max_pathway_jaccard: float = 0.50,
    min_genes_expressed: int = 10,
    sig_fdr: float = 0.05,
    sig_top_n: int = 300,
) -> List[Path]:
    """
    Prefer baseline stats; fallback to pathway_stats.tsv.
    Writes baseline-named outputs for compatibility even if fallback is used.
    """
    cohort_dir = cohort_dir.resolve()

    in_baseline = cohort_dir / "pathway_stats_with_baseline.tsv"
    in_plain = cohort_dir / "pathway_stats.tsv"

    used_input: Optional[Path] = None
    if in_baseline.exists():
        used_input = in_baseline
    elif in_plain.exists():
        used_input = in_plain
        LOG.info("[%s] baseline stats missing; using pathway_stats.tsv for classification", cohort_dir.name)
    else:
        LOG.warning("[%s] missing pathway_stats*.tsv; skipping classification", cohort_dir.name)
        return []

    df = pd.read_csv(used_input, sep="\t", engine="python")
    if df.empty:
        LOG.warning("[%s] %s is empty; skipping", cohort_dir.name, used_input.name)
        return []

    df_filt, df_class = classify_pathway_table(
        df=df,
        gmt=gmt,
        expressed_genes=expressed_genes,
        reduce_overlap=reduce_overlap,
        max_pathway_jaccard=max_pathway_jaccard,
        min_genes_expressed=min_genes_expressed,
        sig_fdr=sig_fdr,
        sig_top_n=sig_top_n,
        cache_csv=cache_csv,
    )

    # Compatibility outputs (what your pipeline expects)
    out_f = cohort_dir / "pathway_stats_with_baseline_filtered.tsv"
    out_c = cohort_dir / "pathway_stats_with_baseline_filtered_classified.tsv"
    df_filt.to_csv(out_f, sep="\t", index=False)
    df_class.to_csv(out_c, sep="\t", index=False)

    return [out_f, out_c]


def classify_ipaa_pathways_all(
    out_root: Path,
    cohort_runs: List[Dict[str, str]],
    gmt: Dict[str, Set[str]],
    cache_csv: Optional[Path] = None,
    expressed_genes_by_cohort: Optional[Dict[str, Set[str]]] = None,
    reduce_overlap: bool = True,
    max_pathway_jaccard: float = 0.50,
    min_genes_expressed: int = 10,
    sig_fdr: float = 0.05,
    sig_top_n: int = 300,
) -> List[Path]:
    out_root = out_root.resolve()
    cache_csv = cache_csv or (out_root / "classification_memory.csv")
    written: List[Path] = []

    for cr in cohort_runs:
        name = str(cr.get("name") or "").strip()
        if not name:
            continue

        cdir = _resolve_cohort_dir(out_root, name, cr)
        if cdir is None or not cdir.exists():
            LOG.warning("[%s] could not resolve cohort directory; skipping classification", name)
            continue

        expressed = None
        if expressed_genes_by_cohort and name in expressed_genes_by_cohort:
            expressed = set(expressed_genes_by_cohort[name])

        try:
            out_paths = classify_one_cohort(
                cohort_dir=cdir,
                gmt=gmt,
                cache_csv=cache_csv,
                expressed_genes=expressed,
                reduce_overlap=reduce_overlap,
                max_pathway_jaccard=max_pathway_jaccard,
                min_genes_expressed=min_genes_expressed,
                sig_fdr=sig_fdr,
                sig_top_n=sig_top_n,
            )
            written.extend(out_paths)
            if out_paths:
                LOG.info("[%s] wrote classified pathway table(s)", name)
            else:
                LOG.info("[%s] classification skipped (no usable pathway stats)", name)
        except Exception as e:
            LOG.warning("[%s] pathway classification failed: %s", name, e)

    return written
