# GC_enricher/GC_enrich.py
from __future__ import annotations
import json, re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, Optional, Dict, List, Tuple

import numpy as np
import pandas as pd
import gseapy
from gseapy import prerank, enrichr
from statsmodels.stats.multitest import multipletests

__all__ = [
    "GCEnrichConfig",
    "run_genecards_cohort",
    "write_flat_json_from_outputs",
]

@dataclass(frozen=True)
class GCEnrichConfig:
    enabled: bool = True
    top_k: int = 500
    prerank_permutations: int = 100
    prerank_min_size: int = 10
    prerank_max_size: int = 2000
    seed: int = 13
    fdr_threshold: float = 0.05
    pathway_libs: Tuple[str, ...] = (
        "MSigDB_Hallmark_2020",
        "KEGG_2021_Human",
        "Reactome_2022",
        "GO_Biological_Process_2023",
        "GO_Cellular_Component_2023",
        "GO_Molecular_Function_2023",
    )
    tf_epi_libs: Tuple[str, ...] = ("ChEA_2022","ENCODE_TF_ChIP-seq_2015","ENCODE_TF_ChIP-seq_2014")
    metabolism_libs: Tuple[str, ...] = ("KEGG_2021_Human","Reactome_2022")
    metabolite_libs: Tuple[str, ...] = ("HMDB_Metabolites",)
    collapse_redundancy: bool = True
    jaccard_threshold: float = 0.7
    hgnc_map_file: Optional[str] = None

def _safe_mkdir(p: Path) -> Path:
    p = Path(p); p.mkdir(parents=True, exist_ok=True); return p

def _clean_symbol(x: str) -> str:
    if pd.isna(x): return ""
    x = str(x).strip(); x = re.sub(r"\s+", "", x); return x.upper()

def _load_hgnc_map(hgnc_map_file: Optional[str]) -> Dict[str, str]:
    if not hgnc_map_file: return {}
    p = Path(hgnc_map_file)
    if not p.exists(): return {}
    df = pd.read_csv(p, sep=None, engine="python")
    alias_col = next((c for c in df.columns if re.search(r"(alias|synonym|prev)", c, re.I)), None)
    sym_col   = next((c for c in df.columns if re.search(r"(symbol|hgnc)",   c, re.I)), None)
    if alias_col is None or sym_col is None: return {}
    mp: Dict[str, str] = {}
    for a, s in df[[alias_col, sym_col]].dropna().itertuples(index=False):
        a2, s2 = _clean_symbol(a), _clean_symbol(s)
        if a2 and s2: mp[a2] = s2
    return mp

def _map_to_hgnc(ser: pd.Series, mp: Dict[str, str]) -> pd.Series:
    cleaned = ser.map(_clean_symbol)
    return cleaned.map(lambda g: mp.get(g, g)) if mp else cleaned

def _rank_from_scores(df: pd.DataFrame, gene_col: str, score_col: str) -> pd.Series:
    s = pd.to_numeric(df[score_col], errors="coerce")
    signed = (pd.notna(s.min())) and (s.min() < 0)
    keep_idx = s.abs().groupby(df[gene_col]).transform(lambda col: col.eq(col.max()))
    df_c = df.loc[keep_idx].drop_duplicates(subset=[gene_col], keep="first")
    s = pd.to_numeric(df_c[score_col], errors="coerce")
    w = np.log1p(np.abs(s.values))
    if signed: w = np.sign(s.values) * w
    genes = df_c[gene_col].astype(str)
    eps = genes.map(lambda g: (int.from_bytes(g.encode("utf-8"), "little") & 0xFFFF) / 65535.0 * 1e-9).to_numpy()
    w = w + (np.sign(w) if signed else 1.0) * eps
    return pd.Series(w, index=genes.values, name="weight").sort_values(ascending=False)

def _top_genes(weights: pd.Series, k: int) -> List[str]:
    if weights is None or weights.empty: return []
    k = max(1, int(k))
    return [g for g in list(weights.index[:k]) if isinstance(g, str) and g]

def _collapse(res_df: pd.DataFrame, j: float) -> pd.DataFrame:
    if res_df is None or res_df.empty or "Term" not in res_df.columns: return res_df
    keep = []; seen: List[str] = []
    for _, row in res_df.iterrows():
        term = str(row["Term"]); tset = set(re.split(r"[\W_]+", term.lower()))
        redundant = False
        for prev in seen:
            pset = set(re.split(r"[\W_]+", prev.lower()))
            if len(tset & pset) / max(1, len(tset | pset)) >= j:
                redundant = True; break
        keep.append(not redundant)
        if not redundant: seen.append(term)
    return res_df.loc[keep].reset_index(drop=True)

def _adjust_fdr(df: pd.DataFrame, pcol: str = "P-value", out_col: str = "FDR_q") -> pd.DataFrame:
    if df is None or df.empty or pcol not in df.columns: return df
    p = pd.to_numeric(df[pcol], errors="coerce")
    mask = pd.notna(p); q = np.full(len(df), np.nan)
    if mask.any():
        q[mask.values] = multipletests(p[mask].values, method="fdr_bh")[1]
    df[out_col] = q; return df

def _write_tsv(df: pd.DataFrame, path: Path):
    if df is None or df.empty: return
    Path(path).write_text(df.to_csv(sep="\t", index=False))

def run_genecards_cohort(
    cohort_name: str,
    input_json: Path | str,
    out_root: Path | str,
    cfg: GCEnrichConfig,
    gene_col: str = "gene_symbol",
    score_col: str = "gene_score",
):
    """
    Writes DIRECTLY into `out_root` (NO appending of cohort name here).
    The caller must pass the final folder (e.g., .../GC_enrich/<disease>).
    """
    if not cfg.enabled:
        raise RuntimeError("GC enrichment disabled via config.")

    out_dir = _safe_mkdir(Path(out_root))  # final folder as-is

    obj = json.loads(Path(input_json).read_text())
    if not (isinstance(obj, dict) and "data" in obj and isinstance(obj["data"], dict)):
        raise ValueError("Scores JSON must have a top-level 'data' dict with arrays.")
    data = obj["data"]
    if gene_col not in data:
        raise KeyError(f"'data' missing '{gene_col}'")
    s_key = score_col if score_col in data else ("disorder_score" if "disorder_score" in data else None)
    if s_key is None:
        raise KeyError(f"'data' missing '{score_col}' and 'disorder_score'")

    genes, scores = data[gene_col], data[s_key]
    if not (isinstance(genes, list) and isinstance(scores, list) and len(genes) == len(scores)):
        raise ValueError("Gene and score arrays must be lists of equal length.")

    df = pd.DataFrame({"Gene": genes, "Score": scores})
    mp = _load_hgnc_map(cfg.hgnc_map_file)
    df["Gene"]  = _map_to_hgnc(df["Gene"], mp)
    df["Score"] = pd.to_numeric(df["Score"], errors="coerce")
    df = df.dropna(subset=["Gene", "Score"])
    df = df[df["Gene"] != ""]
    df = df.sort_values("Score", key=lambda s: np.abs(s), ascending=False).drop_duplicates("Gene", keep="first")
    if df.empty: raise ValueError("No genes after cleaning.")

    _write_tsv(df[["Gene","Score"]], out_dir / "clean_scores.tsv")

    weights = _rank_from_scores(df.rename(columns={"Gene":"gene","Score":"score"}), "gene", "score")
    _write_tsv(
        weights.reset_index().rename(columns={"index":"Gene","weight":"RankWeight"}),
        out_dir / "ranked_weights.tsv"
    )

    prerank_dir = _safe_mkdir(out_dir / "prerank")
    res_all: List[pd.DataFrame] = []
    for lib in cfg.pathway_libs:
        try:
            pre = gseapy.prerank(
                rnk=weights,
                gene_sets=lib,
                min_size=cfg.prerank_min_size,
                max_size=cfg.prerank_max_size,
                permutation_num=cfg.prerank_permutations,
                seed=cfg.seed,
                outdir=str(prerank_dir / lib),
                format="png",
                processes=1,
                no_plot=True,
                verbose=False,
            )
            d = pre.res2d.reset_index()
            d["Library"] = lib
            if "FDR q-val" in d.columns and "FDR_q" not in d.columns:
                d = d.rename(columns={"FDR q-val":"FDR_q"})
            if "FDR_q" not in d.columns and "P-value" in d.columns:
                d = _adjust_fdr(d, "P-value", "FDR_q")
            res_all.append(d)
        except Exception as e:
            (prerank_dir / f"{lib}.error.txt").write_text(str(e))

    prerank_all = pd.concat(res_all, ignore_index=True) if res_all else pd.DataFrame()
    if not prerank_all.empty and cfg.collapse_redundancy:
        sort_col = "FDR_q" if "FDR_q" in prerank_all.columns else ("P-value" if "P-value" in prerank_all.columns else None)
        if sort_col: prerank_all = prerank_all.sort_values(sort_col, ascending=True)
        prerank_all = _collapse(prerank_all, cfg.jaccard_threshold)

    if not prerank_all.empty and "FDR_q" in prerank_all.columns:
        prerank_sig = prerank_all[pd.to_numeric(prerank_all["FDR_q"], errors="coerce") <= cfg.fdr_threshold]
    else:
        prerank_sig = pd.DataFrame()

    _write_tsv(prerank_all, out_dir / "prerank_all.tsv")
    _write_tsv(prerank_sig, out_dir / "prerank_sig.tsv")

    def _enrich_block(genes: Iterable[str], libs: Iterable[str], sub: str) -> pd.DataFrame:
        subdir = _safe_mkdir(out_dir / sub)
        parts = []
        for nm in libs:
            try:
                enr = gseapy.enrichr(
                    gene_list=list(genes),
                    gene_sets=[nm],
                    outdir=str(subdir),
                    no_plot=True,
                    cutoff=1.0,
                )
                t = enr.results.copy()
                t["Library"] = nm
                if "Adjusted P-value" in t.columns and "FDR_q" not in t.columns:
                    t = t.rename(columns={"Adjusted P-value":"FDR_q"})
                if "FDR_q" not in t.columns and "P-value" in t.columns:
                    t = _adjust_fdr(t, "P-value", "FDR_q")
                parts.append(t)
            except Exception as e:
                (subdir / f"{nm}.error.txt").write_text(str(e))
        if not parts: return pd.DataFrame()
        out = pd.concat(parts, ignore_index=True)
        if not out.empty and cfg.collapse_redundancy:
            if "FDR_q" in out.columns:
                out = out.sort_values("FDR_q", ascending=True)
            out = _collapse(out, cfg.jaccard_threshold)
        return out

    top_genes = _top_genes(weights, cfg.top_k)

    tf_epi_all = _enrich_block(top_genes, cfg.tf_epi_libs, "enrichr_tf_epi")
    if not tf_epi_all.empty:
        _write_tsv(tf_epi_all, out_dir / "tf_epi_all.tsv")
        sig = tf_epi_all[pd.to_numeric(tf_epi_all["FDR_q"], errors="coerce") <= cfg.fdr_threshold] if "FDR_q" in tf_epi_all.columns else pd.DataFrame()
        _write_tsv(sig, out_dir / "tf_epi_sig.tsv")

    metab_all = _enrich_block(top_genes, cfg.metabolism_libs, "enrichr_metabolism")
    if not metab_all.empty:
        _write_tsv(metab_all, out_dir / "metabolism_all.tsv")
        sig = metab_all[pd.to_numeric(metab_all["FDR_q"], errors="coerce") <= cfg.fdr_threshold] if "FDR_q" in metab_all.columns else pd.DataFrame()
        _write_tsv(sig, out_dir / "metabolism_sig.tsv")

    metabs_all = _enrich_block(top_genes, cfg.metabolite_libs, "enrichr_metabolites")
    if not metabs_all.empty:
        _write_tsv(metabs_all, out_dir / "metabolites_all.tsv")
        sig = metabs_all[pd.to_numeric(metabs_all["FDR_q"], errors="coerce") <= cfg.fdr_threshold] if "FDR_q" in metabs_all.columns else pd.DataFrame()
        _write_tsv(sig, out_dir / "metabolites_sig.tsv")

    (out_dir / "input_meta.json").write_text(json.dumps({
        "cohort": cohort_name,
        "source_file": str(input_json),
        "config": asdict(cfg),
    }, indent=2))

    return {
        "cohort": cohort_name,
        "out_dir": out_dir,
        "prerank_sig": out_dir / "prerank_sig.tsv",
        "tf_epi_sig": (out_dir / "tf_epi_sig.tsv") if (out_dir / "tf_epi_sig.tsv").exists() else None,
        "metabolism_sig": (out_dir / "metabolism_sig.tsv") if (out_dir / "metabolism_sig.tsv").exists() else None,
        "metabolites_sig": (out_dir / "metabolites_sig.tsv") if (out_dir / "metabolites_sig.tsv").exists() else None,
        "config": asdict(cfg),
    }

# (flat JSON helper stays same as before if you need it)
def write_flat_json_from_outputs(*args, **kwargs):
    pass  # omitted for brevity; not related to the nesting bug
