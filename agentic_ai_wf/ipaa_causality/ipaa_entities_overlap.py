#!/usr/bin/env python3
"""
ipaa_entities_overlap.py

Purpose
-------
Runs:
  1) (Optional) TF / Epigenetic / Metabolite enrichments per cohort using DE genes
  2) Builds pathway↔entity overlap JSONs using an IPAA-filtered pathway backbone

No-surprises guarantee:
- Reuses existing enrichment CSVs; only computes missing ones (if enabled).
- If gseapy/Enrichr is unavailable, still attempts overlap JSONs from existing files.
- Robust cohort directory resolution across layouts.

Output
------
Per cohort:
  <cohort_dir>/overlap/pathway_entity_overlap.json

Mirrored to:
  <OUT_ROOT>/jsons_all/<Cohort>.json
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import pandas as pd
import gseapy 


LOG = logging.getLogger("ipaa_entities_overlap")
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

    # fallbacks
    candidates.append(out_root / "cohorts" / cohort_name)
    candidates.append(out_root / cohort_name)

    sentinels = (
        "pathway_stats_with_baseline.tsv",
        "pathway_stats.tsv",
        "de_gene_stats.tsv",
        "pathway_activity.tsv",
    )

    for d in candidates:
        if not d.exists() or not d.is_dir():
            continue
        if any((d / s).exists() for s in sentinels):
            return d

    # if none match sentinels but something exists, return first existing dir
    for d in candidates:
        if d.exists() and d.is_dir():
            return d

    return None


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
    gmt: Dict[str, Iterable[str]],
    expressed_genes: Optional[Set[str]] = None,
    min_genes_expressed: int = 10,
    reduce_overlap: bool = True,
    max_pathway_jaccard: float = 0.50,
) -> Tuple[Set[str], Dict[str, Set[str]]]:
    # 1) expressed gene gate
    filtered: Dict[str, Set[str]] = {}
    for p, genes in gmt.items():
        gset = set(genes) if genes is not None else set()
        if not gset:
            continue
        if expressed_genes is None:
            filtered[p] = set(gset)
        else:
            g = set(gset) & set(expressed_genes)
            if len(g) >= int(min_genes_expressed):
                filtered[p] = g

    if not reduce_overlap:
        return set(filtered.keys()), filtered

    # 2) greedy reduced-overlap keep
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
# DE genes → up/down sets
# ----------------------------

def _guess_col(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    cols = {c.lower(): c for c in df.columns}
    for cand in candidates:
        c = cols.get(cand.lower())
        if c:
            return c
    return None


def read_de_genes_up_down(
    de_path: Path,
    gene_fdr_for_updown: float = 0.05,
) -> Tuple[Set[str], Set[str]]:
    """
    Read de_gene_stats.tsv (or similar) and split into UP/DOWN sets by sign of logFC.
    Robust to column names.
    """
    df = pd.read_csv(de_path, sep="\t", engine="python")
    if df.empty:
        return set(), set()

    gcol = _guess_col(df, ["gene", "symbol", "hgnc", "id", "gene_id", "gene_name"])
    lcol = _guess_col(df, ["log2fc", "log2foldchange", "lfc", "logfc", "log fold change"])
    qcol = _guess_col(df, ["padj", "fdr", "qval", "q-value", "adj p", "adj_p"])

    if not gcol:
        raise ValueError(f"Could not infer gene column in {de_path.name}: {list(df.columns)}")

    df[gcol] = df[gcol].astype(str)

    if qcol:
        df[qcol] = pd.to_numeric(df[qcol], errors="coerce")
        df = df.dropna(subset=[qcol])
        df = df[df[qcol] <= float(gene_fdr_for_updown)]

    if lcol:
        df[lcol] = pd.to_numeric(df[lcol], errors="coerce")
        df = df.dropna(subset=[lcol])
        up = set(df.loc[df[lcol] > 0, gcol].astype(str))
        down = set(df.loc[df[lcol] < 0, gcol].astype(str))
    else:
        # Direction unknown
        up = set(df[gcol].astype(str))
        down = set()

    return up, down


# ----------------------------
# Enrichr (optional via gseapy)
# ----------------------------

@dataclass(frozen=True)
class EnrichConfig:
    tf_lib_patterns: Tuple[str, ...] = ("ChEA", "ENCODE", "JASPAR", "TRANSFAC", "ChIP", "DoRothEA")
    epi_lib_patterns: Tuple[str, ...] = ("Histone", "H3K", "Chromatin", "Enhancer", "Epigen", "Roadmap")
    hmdb_libs: Tuple[str, ...] = ("HMDB_Metabolites",)
    organism: str = "Human"
    min_overlap_genes: int = 2




def _list_enrichr_libs(gp) -> List[str]:
    try:
        return list(gp.get_library_name(organism="Human"))
    except Exception:
        return []


def _pick_libs(all_libs: Sequence[str], patterns: Sequence[str]) -> List[str]:
    out: List[str] = []
    for lib in all_libs:
        up = lib.upper()
        if any(pat.upper() in up for pat in patterns):
            out.append(lib)
    seen = set()
    uniq = []
    for x in out:
        if x not in seen:
            uniq.append(x)
            seen.add(x)
    return uniq


def _run_enrichr(gp, genes: Sequence[str], gene_sets: str, outdir: Path, organism: str = "Human") -> Optional[pd.DataFrame]:
    if not genes:
        return None
    try:
        enr = gp.enrichr(
            gene_list=list(genes),
            gene_sets=gene_sets,
            organism=organism,
            outdir=str(outdir),
            no_plot=True,
        )
        if hasattr(enr, "results") and isinstance(enr.results, pd.DataFrame):
            return enr.results
    except Exception as e:
        LOG.warning("enrichr failed for %s: %s", gene_sets, e)
    return None


def ensure_entity_enrichments(
    cohort_dir: Path,
    gene_fdr_for_updown: float,
    run_if_missing: bool,
    cfg: EnrichConfig,
) -> None:
    """
    Create (or reuse) 6 files:
      tf_enrich_up/down.csv
      epigenetic_enrich_up/down.csv
      metabolite_enrich_up/down.csv

    No surprise behavior:
      - If a file exists, we do NOT overwrite it.
      - If run_if_missing=False, we do nothing unless nothing exists at all.
    """
    cohort_dir = cohort_dir.resolve()
    out_tf_up = cohort_dir / "tf_enrich_up.csv"
    out_tf_dn = cohort_dir / "tf_enrich_down.csv"
    out_epi_up = cohort_dir / "epigenetic_enrich_up.csv"
    out_epi_dn = cohort_dir / "epigenetic_enrich_down.csv"
    out_met_up = cohort_dir / "metabolite_enrich_up.csv"
    out_met_dn = cohort_dir / "metabolite_enrich_down.csv"

    outputs = [out_tf_up, out_tf_dn, out_epi_up, out_epi_dn, out_met_up, out_met_dn]
    have_any = any(p.exists() for p in outputs)
    if have_any and not run_if_missing:
        return

    de_path = cohort_dir / "de_gene_stats.tsv"
    if not de_path.exists():
        LOG.info("[%s] no de_gene_stats.tsv; cannot run enrichments", cohort_dir.name)
        return

    up_genes, down_genes = read_de_genes_up_down(de_path, gene_fdr_for_updown=gene_fdr_for_updown)
    if not up_genes and not down_genes:
        LOG.info("[%s] no significant genes for enrichments", cohort_dir.name)
        return

    gp = gseapy

    all_libs = _list_enrichr_libs(gp)
    tf_libs = _pick_libs(all_libs, cfg.tf_lib_patterns)
    epi_libs = _pick_libs(all_libs, cfg.epi_lib_patterns)
    hmdb_libs = [lib for lib in all_libs if lib in cfg.hmdb_libs] or _pick_libs(all_libs, cfg.hmdb_libs)

    tf_lib = tf_libs[0] if tf_libs else None
    epi_lib = epi_libs[0] if epi_libs else None
    hmdb_lib = hmdb_libs[0] if hmdb_libs else None

    workdir = cohort_dir / "enrichr_tmp"
    workdir.mkdir(parents=True, exist_ok=True)

    def save_if_missing(df: Optional[pd.DataFrame], out_path: Path) -> None:
        if out_path.exists():
            return
        if df is None or df.empty:
            return
        df.to_csv(out_path, index=False)

    if tf_lib:
        save_if_missing(_run_enrichr(gp, sorted(up_genes), tf_lib, workdir / "tf_up", organism=cfg.organism), out_tf_up)
        save_if_missing(_run_enrichr(gp, sorted(down_genes), tf_lib, workdir / "tf_dn", organism=cfg.organism), out_tf_dn)

    if epi_lib:
        save_if_missing(_run_enrichr(gp, sorted(up_genes), epi_lib, workdir / "epi_up", organism=cfg.organism), out_epi_up)
        save_if_missing(_run_enrichr(gp, sorted(down_genes), epi_lib, workdir / "epi_dn", organism=cfg.organism), out_epi_dn)

    if hmdb_lib:
        save_if_missing(_run_enrichr(gp, sorted(up_genes), hmdb_lib, workdir / "met_up", organism=cfg.organism), out_met_up)
        save_if_missing(_run_enrichr(gp, sorted(down_genes), hmdb_lib, workdir / "met_dn", organism=cfg.organism), out_met_dn)


# ----------------------------
# Overlap JSON builder
# ----------------------------

def _read_enrich_table(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, engine="python")
    return df if not df.empty else pd.DataFrame()


def _infer_term_col(df: pd.DataFrame) -> Optional[str]:
    return _guess_col(df, ["term", "name", "pathway", "gene_set", "geneset"])


def _infer_genes_col(df: pd.DataFrame) -> Optional[str]:
    return _guess_col(df, ["genes", "overlapping genes", "overlap genes", "gene", "overlap"])


def _infer_qcol(df: pd.DataFrame) -> Optional[str]:
    return _guess_col(df, ["adjusted p-value", "adj p-value", "adj_p", "fdr", "q-value", "qval", "padj", "p-value"])


def _parse_gene_list(val: str) -> Set[str]:
    if val is None:
        return set()
    s = str(val).strip()
    if not s:
        return set()
    parts = re.split(r"[;,/|\s]+", s)
    return {p.strip() for p in parts if p.strip()}


def _select_sig_pathways(pathway_stats: Path, sig_fdr: float, sig_top_n: int) -> pd.DataFrame:
    df = pd.read_csv(pathway_stats, sep="\t", engine="python")
    if df.empty:
        return df
    pcol = _guess_col(df, ["pathway", "term", "name"])
    qcol = _guess_col(df, ["fdr", "qval", "q-value", "padj", "fdr_q"])
    if not pcol or not qcol:
        return df
    df[qcol] = pd.to_numeric(df[qcol], errors="coerce")
    df = df.dropna(subset=[pcol, qcol])
    df_f = df[df[qcol] <= float(sig_fdr)].copy()
    if df_f.empty:
        df_f = df.nsmallest(max(int(sig_top_n), 50), qcol)
    else:
        if int(sig_top_n) > 0 and len(df_f) > int(sig_top_n):
            df_f = df_f.nsmallest(int(sig_top_n), qcol)
    return df_f


def build_overlap_json(
    cohort_name: str,
    cohort_dir: Path,
    gmt: Dict[str, Set[str]],
    kept_pathways: Set[str],
    sig_fdr: float,
    sig_top_n: int,
    entity_q_thresh: float = 0.05,
    min_overlap_genes: int = 2,
) -> Optional[dict]:
    """
    Build compact JSON:
      pathway -> direction + entities overlaps

    Robustness:
      - Prefer pathway_stats_with_baseline.tsv
      - Else fallback to pathway_stats.tsv (direction from t if present)
    """
    cohort_dir = cohort_dir.resolve()

    ps = cohort_dir / "pathway_stats_with_baseline.tsv"
    if not ps.exists():
        ps = cohort_dir / "pathway_stats.tsv"
    if not ps.exists():
        LOG.info("[%s] missing pathway stats; cannot build overlap JSON", cohort_name)
        return None

    df_sig = _select_sig_pathways(ps, sig_fdr=sig_fdr, sig_top_n=sig_top_n)
    if df_sig.empty:
        return None

    pcol = _guess_col(df_sig, ["pathway", "term", "name"]) or "pathway"
    tcol = _guess_col(df_sig, ["t", "tstat", "t_stat", "t-statistic"])
    df_sig[pcol] = df_sig[pcol].astype(str)

    df_bg = df_sig[df_sig[pcol].isin(kept_pathways)].copy()
    if df_bg.empty:
        df_bg = df_sig.copy()

    direction: Dict[str, str] = {}
    if tcol and tcol in df_bg.columns:
        df_bg[tcol] = pd.to_numeric(df_bg[tcol], errors="coerce")
        for _, r in df_bg.iterrows():
            p = str(r[pcol])
            t = r[tcol]
            if pd.isna(t):
                direction[p] = "ANY"
            else:
                direction[p] = "UP" if float(t) > 0 else "DOWN"
    else:
        for p in df_bg[pcol].tolist():
            direction[str(p)] = "ANY"

    enrich_files = {
        "tf": (cohort_dir / "tf_enrich_up.csv", cohort_dir / "tf_enrich_down.csv"),
        "epigenetic": (cohort_dir / "epigenetic_enrich_up.csv", cohort_dir / "epigenetic_enrich_down.csv"),
        "metabolite": (cohort_dir / "metabolite_enrich_up.csv", cohort_dir / "metabolite_enrich_down.csv"),
    }

    entities: Dict[str, Dict[str, Dict[str, Set[str]]]] = {}
    for etype, (fup, fdn) in enrich_files.items():
        entities[etype] = {"UP": {}, "DOWN": {}}
        for d, fp in [("UP", fup), ("DOWN", fdn)]:
            if not fp.exists():
                continue
            df = _read_enrich_table(fp)
            if df.empty:
                continue
            termc = _infer_term_col(df)
            genec = _infer_genes_col(df)
            qc = _infer_qcol(df)
            if not termc or not genec:
                continue
            if qc:
                df[qc] = pd.to_numeric(df[qc], errors="coerce")
                df = df.dropna(subset=[qc])
                df = df[df[qc] <= float(entity_q_thresh)].copy()
            for _, r in df.iterrows():
                term = str(r[termc])
                genes = _parse_gene_list(r[genec])
                if len(genes) >= int(min_overlap_genes):
                    entities[etype][d][term] = genes

    out: dict = {"cohort": cohort_name, "generated_from": str(cohort_dir), "pathways": {}}

    for p in df_bg[pcol].tolist():
        p = str(p)
        if p not in gmt:
            continue
        pgenes = set(gmt[p])
        pdir = direction.get(p, "ANY")

        out["pathways"].setdefault(p, {})
        out["pathways"][p]["direction"] = pdir
        out["pathways"][p]["entities"] = {}

        dirs_to_use = ["UP", "DOWN"] if pdir == "ANY" else [pdir]

        for etype in ["tf", "epigenetic", "metabolite"]:
            matches: List[dict] = []
            for d in dirs_to_use:
                for term, tgenes in entities.get(etype, {}).get(d, {}).items():
                    overlap = sorted(pgenes & tgenes)
                    if len(overlap) >= int(min_overlap_genes):
                        matches.append({"term": term, "direction": d, "overlap_genes": overlap})
            matches.sort(key=lambda x: len(x.get("overlap_genes", [])), reverse=True)
            out["pathways"][p]["entities"][etype] = matches[:50]

    return out


def run_entities_and_overlap_all(
    out_root: Path,
    cohort_runs: List[Dict[str, str]],
    gmt: Dict[str, Iterable[str]],
    expressed_genes_by_cohort: Optional[Dict[str, Set[str]]] = None,
    gene_fdr_for_updown: float = 0.05,
    reduce_overlap: bool = True,
    max_pathway_jaccard: float = 0.50,
    min_genes_expressed: int = 10,
    run_enrichments_if_missing: bool = True,
    sig_fdr: float = 0.05,
    sig_top_n: int = 300,
) -> List[Path]:
    """
    High-level runner invoked by main.py.

    Writes:
      <cohort_dir>/overlap/pathway_entity_overlap.json
    Mirrors:
      <OUT_ROOT>/jsons_all/<Cohort>.json
    """
    out_root = out_root.resolve()
    json_root = out_root / "jsons_all"
    json_root.mkdir(parents=True, exist_ok=True)

    written: List[Path] = []

    # ensure gmt values are sets for downstream
    gmt_sets: Dict[str, Set[str]] = {k: set(v) for k, v in gmt.items()}

    for cr in cohort_runs:
        name = str(cr.get("name") or "").strip()
        if not name:
            continue

        cdir = _resolve_cohort_dir(out_root, name, cr)
        if cdir is None or not cdir.exists():
            LOG.warning("[%s] could not resolve cohort directory; skipping", name)
            continue

        expressed = None
        if expressed_genes_by_cohort and name in expressed_genes_by_cohort:
            expressed = set(expressed_genes_by_cohort[name])

        kept, gmt_f = build_reduced_overlap_backbone(
            gmt=gmt_sets,
            expressed_genes=expressed,
            min_genes_expressed=min_genes_expressed,
            reduce_overlap=reduce_overlap,
            max_pathway_jaccard=max_pathway_jaccard,
        )

        # 1) enrichments (optional)
        try:
            ensure_entity_enrichments(
                cohort_dir=cdir,
                gene_fdr_for_updown=gene_fdr_for_updown,
                run_if_missing=run_enrichments_if_missing,
                cfg=EnrichConfig(),
            )
        except Exception as e:
            LOG.warning("[%s] enrichment step failed: %s", name, e)

        # 2) overlap json
        payload = build_overlap_json(
            cohort_name=name,
            cohort_dir=cdir,
            gmt=gmt_f,
            kept_pathways=kept,
            sig_fdr=sig_fdr,
            sig_top_n=sig_top_n,
            entity_q_thresh=0.05,
            min_overlap_genes=2,
        )
        if payload is None:
            LOG.info("[%s] no overlap JSON produced", name)
            continue

        odir = cdir / "overlap"
        odir.mkdir(parents=True, exist_ok=True)
        out1 = odir / "pathway_entity_overlap.json"
        out1.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        written.append(out1)

        out2 = json_root / f"{name}.json"
        out2.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        written.append(out2)

        LOG.info("[%s] wrote overlap JSONs", name)

    return written
