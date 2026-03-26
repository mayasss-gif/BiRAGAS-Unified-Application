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
    "GLEnrichConfig",
    "run_gl_enrichment_for_cohort",
    "combine_cohorts_union_and_shared",
]

# ============================== Config ==============================

@dataclass(frozen=True)
class GLEnrichConfig:
    enabled: bool = True

    # Ranked-list / prerank GSEA
    top_k: int = 500
    prerank_permutations: int = 100
    prerank_min_size: int = 10
    prerank_max_size: int = 2000
    seed: int = 13
    fdr_threshold: float = 0.05

    # --- Pathway libs for prerank (added GO BP/CC/MF as requested) ---
    pathway_libs: Tuple[str, ...] = (
        "MSigDB_Hallmark_2020",
        "KEGG_2021_Human",
        "Reactome_2022",
        "GO_Biological_Process_2021",
#        "GO_Cellular_Component_2021",
        "GO_Molecular_Function_2021",
    )

    # --- Library discovery via regex patterns (split TF vs Epigenetic) ---
    TF_PATTERNS: Tuple[str, ...] = (
        r"ENCODE_TF_ChIP",
        r"ChEA",
        r"JASPAR",
        r"TRANSFAC",
        r"ChIP-X",
        r"DoRothEA",
        r"\bTF\b",
    )
    EPIGENETIC_PATTERNS: Tuple[str, ...] = (
        r"Histone",
        r"\bH3K",
        r"Chromatin",
        r"Enhancer",
        r"Epigenetic",
    )

    # Optional (declared for future use; not needed for this change)
    IMMUNE_PATTERNS: Tuple[str, ...] = (
        r"PanglaoDB",
        r"Azimuth",
        r"Immunologic|Immune|LM22|CIBERSORT",
    )

    # Metabolite libraries (explicit; Enrichr category)
    METABOLIC_LIBS: Tuple[str, ...] = ("HMDB_Metabolites",)

    # Redundancy control
    collapse_redundancy: bool = True
    jaccard_threshold: float = 0.7

    # Null audit
    null_shuffles: int = 0

    # HGNC mapping (CSV with alias + symbol columns)
    hgnc_map_file: Optional[str] = None


# ============================== Utilities ==============================

def _safe_mkdir(p: Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _record_versions(out_dir: Path) -> None:
    try:
        data = {
            "gseapy": getattr(gseapy, "__version__", "unknown"),
            "pandas": getattr(pd, "__version__", "unknown"),
            "numpy": getattr(np, "__version__", "unknown"),
        }
        (Path(out_dir) / "versions.json").write_text(json.dumps(data, indent=2))
    except Exception:
        pass


def _write_tsv(df: pd.DataFrame, path: Path) -> None:
    path = Path(path)
    if df is None or df.empty:
        df.head(0).to_csv(path, sep="\t", index=False)
        return
    df.to_csv(path, sep="\t", index=False)


def _clean_symbol(x: str) -> str:
    if x is None:
        return ""
    x = str(x).strip()
    x = re.sub(r"\s+", "", x)
    return x.upper()


def _load_hgnc_map(hgnc_map_file: Optional[str]) -> Dict[str, str]:
    if not hgnc_map_file:
        return {}
    p = Path(hgnc_map_file)
    if not p.exists():
        return {}
    df = pd.read_csv(p)
    cols = [c.lower() for c in df.columns]
    alias_col = None
    symbol_col = None
    for i, c in enumerate(cols):
        if re.search(r"(alias|synonym|prev)", c):
            alias_col = df.columns[i]
        if re.search(r"(symbol|hgnc)", c):
            symbol_col = df.columns[i]
    if alias_col is None or symbol_col is None:
        alias_col = df.columns[0]
        symbol_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]
    mapping = {}
    for a, s in zip(df[alias_col], df[symbol_col]):
        a = _clean_symbol(a)
        s = _clean_symbol(s)
        if a and s:
            mapping[a] = s
    return mapping


def _map_to_hgnc(gene_series: pd.Series, hgnc_map: Dict[str, str]) -> pd.Series:
    cleaned = gene_series.map(_clean_symbol)
    if not hgnc_map:
        return cleaned
    mapped = cleaned.map(lambda g: hgnc_map.get(g, g))
    return mapped


def _rank_from_scores(df: pd.DataFrame, gene_col: str, score_col: str) -> pd.Series:
    df = df.copy()
    df[score_col] = pd.to_numeric(df[score_col], errors="coerce")
    df = df.dropna(subset=[gene_col, score_col])
    if df.empty:
        raise ValueError("No valid rows to rank from.")
    df = (
        df.sort_values(score_col, key=lambda s: np.abs(s), ascending=False)
          .drop_duplicates(subset=[gene_col], keep="first")
          .reset_index(drop=True)
    )
    signed = (df[score_col].min() < 0)
    w = np.log1p(np.abs(df[score_col].values))
    if signed:
        w = w * np.sign(df[score_col].values)

    def _jit(g: str) -> float:
        b = g.encode("utf-8")
        return (sum(b) % 1000) / 1e9

    jit = np.array([_jit(g) for g in df[gene_col].astype(str)])
    w = w + jit

    ser = pd.Series(w, index=df[gene_col].astype(str))
    ser = ser.sort_values(ascending=False)
    return ser


def _collapse_redundant_sets(res_df: pd.DataFrame, j_threshold: float) -> pd.DataFrame:
    if res_df is None or res_df.empty or "Term" not in res_df.columns:
        return res_df

    def _tok(s: str) -> set:
        toks = re.split(r"[\W_]+", str(s).upper())
        toks = [t for t in toks if t]
        return set(toks)

    keep_rows = []
    seen = []
    for i, row in res_df.iterrows():
        term = row["Term"]
        tset = _tok(term)
        ok = True
        for prev in seen:
            inter = len(tset & prev)
            union = len(tset | prev) if (tset or prev) else 1
            j = inter / union
            if j >= j_threshold:
                ok = False
                break
        if ok:
            keep_rows.append(i)
            seen.append(tset)
    return res_df.loc[keep_rows].reset_index(drop=True)


def _adjust_fdr(df: pd.DataFrame, pcol: str = "P-value", out_col: str = "FDR_q") -> pd.DataFrame:
    df = df.copy()
    if pcol not in df.columns or df.empty:
        return df
    pv = pd.to_numeric(df[pcol], errors="coerce")
    mask = pv.notna()
    if mask.any():
        _, q, _, _ = multipletests(pv[mask].values, method="fdr_bh")
        df.loc[mask, out_col] = q
    return df


def _enrichr_compat(gene_list: Iterable[str], gene_sets, outdir: Path, description: str = "") -> gseapy.enrichr:
    try:
        return enrichr(
            gene_list=list(gene_list),
            gene_sets=gene_sets,
            outdir=str(outdir),
            description=description,
            no_plot=True,
            cutoff=1.0,
        )
    except TypeError:
        return enrichr(
            gene_list=list(gene_list),
            gene_sets=gene_sets,
            outdir=str(outdir),
            no_plot=True,
            cutoff=1.0,
        )


def _top_genes_for_enrichr(weights: pd.Series, k: int) -> List[str]:
    k = max(1, int(k))
    return list(weights.head(k).index)


def _load_scores_table_force_json(path: Path, gene_key: str, score_key: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Scores JSON not found: {p}")
    if p.suffix.lower() != ".json":
        raise ValueError("This loader only accepts .json inputs.")

    raw = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or "data" not in raw or not isinstance(raw["data"], dict):
        raise ValueError("JSON must be an object with a 'data' object containing arrays.")

    data = raw["data"]
    if gene_key not in data:
        raise KeyError(f"Missing required gene_key '{gene_key}' in JSON['data'].")

    score_key_final = None
    if score_key in data:
        score_key_final = score_key
    elif "disorder_score" in data:
        score_key_final = "disorder_score"
    else:
        raise KeyError(f"Missing required score key '{score_key}' (or 'disorder_score') in JSON['data'].")

    genes = data[gene_key]
    scores = data[score_key_final]
    if not isinstance(genes, list) or not isinstance(scores, list):
        raise TypeError("JSON['data'][gene_key] and JSON['data'][score_key] must be lists.")
    if len(genes) != len(scores):
        raise ValueError("Gene and Score arrays must be same length.")

    df = pd.DataFrame({"Gene": genes, "Score": scores})
    return df


# -------- Enrichr library discovery via patterns --------

def _get_enrichr_libs_matching(patterns: Tuple[str, ...]) -> List[str]:
    """Return Enrichr library names that match any of the provided regex patterns."""
    try:
        all_libs = set(gseapy.get_library_name())  # list of lib names
    except Exception:
        # Fallback to a minimal hardcoded set if environment can't fetch
        all_libs = {
            "ChEA_2022", "ENCODE_TF_ChIP-seq_2015", "ENCODE_TF_ChIP-seq_2014",
            "JASPAR_PWMs", "TRANSFAC_and_JASPAR_PWMs", "ChIP-X_Enrichment_Analysis_2016",
            "DoRothEA", "Histone_Modification", "Epigenetic_Landscape_In_Silico",
            "Chromatin_Regulators", "Enhancer_Atlas", "HMDB_Metabolites"
        }
    matched = set()
    for lib in all_libs:
        for pat in patterns:
            if re.search(pat, lib, flags=re.IGNORECASE):
                matched.add(lib)
                break
    return sorted(matched)


# ============================== Main per-cohort driver ==============================

def run_gl_enrichment_for_cohort(
    cohort_name: str,
    scores_table: Path,
    out_root: Path,
    cfg: GLEnrichConfig,
    gene_col: str = "gene_symbol",
    score_col: str = "gene_score",
) -> Dict[str, Optional[Path]]:
    """
    Run GeneList enrichment for a single cohort.
    Changes in this version:
      - Prerank adds GO: BP/CC/MF.
      - TF and Epigenetic Enrichr blocks are separated via regex library discovery.
      - ALL_COMBINED.csv contains Source ∈ {'tf','epigenetic','metabolite'}.
    """
    if not cfg.enabled:
        raise RuntimeError("GL enrichment disabled via config.")

    out_root = Path(out_root)
    out_dir = _safe_mkdir(out_root / str(cohort_name))
    _record_versions(out_dir)

    # ------------ Load & clean inputs ------------
    df = _load_scores_table_force_json(Path(scores_table), gene_key=gene_col, score_key=score_col)

    hgnc_map: Dict[str, str] = _load_hgnc_map(cfg.hgnc_map_file)
    df["Gene"] = _map_to_hgnc(df["Gene"], hgnc_map)
    df["Score"] = pd.to_numeric(df["Score"], errors="coerce")
    df = df.dropna(subset=["Gene", "Score"])
    df = df[df["Gene"] != ""]
    df = (
        df.sort_values("Score", key=lambda s: np.abs(s), ascending=False)
          .drop_duplicates(subset=["Gene"], keep="first")
          .reset_index(drop=True)
    )
    if df.empty:
        raise ValueError("No valid gene-score pairs after cleaning.")

    meta = {
        "cohort": cohort_name,
        "source_file": str(scores_table),
        "score_semantics": "Higher absolute values imply stronger association; sign may be ignored unless negatives exist.",
        "identifier_policy": "Gene symbols cleaned to uppercase; optional alias→symbol mapping via HGNC map.",
        "universe_policy": "All provided symbols considered; GSEA prerank uses all; Enrichr uses top_k.",
        "config": asdict(cfg),
    }
    (out_dir / "input_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    _write_tsv(df[["Gene", "Score"]], out_dir / "clean_scores.tsv")

    # ------------ Ranked weights ------------
    weights = _rank_from_scores(df.rename(columns={"Gene": "gene", "Score": "score"}), "gene", "score")
    wdf = pd.DataFrame({"Gene": weights.index, "RankWeight": weights.values})
    _write_tsv(wdf, out_dir / "ranked_weights.tsv")

    # ------------ Pathway GSEA (prerank) ------------
    prerank_dir = _safe_mkdir(out_dir / "prerank")
    all_prerank = []

    for lib in cfg.pathway_libs:
        lib_dir = _safe_mkdir(prerank_dir / f"{lib}")
        pre = prerank(
            rnk=weights,
            gene_sets=lib,
            min_size=cfg.prerank_min_size,
            max_size=cfg.prerank_max_size,
            permutation_num=cfg.prerank_permutations,
            seed=cfg.seed,
            outdir=str(lib_dir),
            format="png",
            processes=1,
            no_plot=True,
            verbose=False,
        )
        df_res = pre.res2d.reset_index()
        df_res["Library"] = lib

        if "FDR q-val" in df_res.columns and "FDR_q" not in df_res.columns:
            df_res = df_res.rename(columns={"FDR q-val": "FDR_q"})
        if "P-value" in df_res.columns and "FDR_q" not in df_res.columns:
            df_res = _adjust_fdr(df_res, pcol="P-value", out_col="FDR_q")

        all_prerank.append(df_res)

    if all_prerank:
        prerank_all = pd.concat(all_prerank, ignore_index=True)
        if cfg.collapse_redundancy:
            sort_col = "FDR_q" if "FDR_q" in prerank_all.columns else ("P-value" if "P-value" in prerank_all.columns else None)
            if sort_col:
                prerank_all = prerank_all.sort_values(sort_col, ascending=True)
            prerank_all = _collapse_redundant_sets(prerank_all, cfg.jaccard_threshold)
    else:
        prerank_all = pd.DataFrame()

    _write_tsv(prerank_all, out_dir / "prerank_all.tsv")
    if not prerank_all.empty and "FDR_q" in prerank_all.columns:
        prerank_sig = prerank_all[pd.to_numeric(prerank_all["FDR_q"], errors="coerce") <= cfg.fdr_threshold]
    else:
        prerank_sig = pd.DataFrame()
    _write_tsv(prerank_sig, out_dir / "prerank_sig.tsv")

    # ------------ Enrichr (split TF vs Epigenetic, plus Metabolite) ------------
    top_genes = _top_genes_for_enrichr(weights, cfg.top_k)

    def _run_enrichr_block(gmt_names: List[str], subdir: str) -> pd.DataFrame:
        outd = _safe_mkdir(out_dir / subdir)
        all_res = []
        for nm in gmt_names:
            enr = _enrichr_compat(top_genes, nm, outd, description=f"{cohort_name}-{nm}")
            df_res = enr.results.copy() if hasattr(enr, "results") else pd.DataFrame()
            if df_res is None or df_res.empty:
                continue
            df_res["Library"] = nm

            # FDR normalization
            if "Adjusted P-value" in df_res.columns and "FDR_q" not in df_res.columns:
                df_res = df_res.rename(columns={"Adjusted P-value": "FDR_q"})
            if "P-value" in df_res.columns and "FDR_q" not in df_res.columns:
                df_res = _adjust_fdr(df_res, pcol="P-value", out_col="FDR_q")

            # Standardize term column if needed
            if "Term" not in df_res.columns:
                for alt in ("Gene_set", "Term_name", "term_name"):
                    if alt in df_res.columns:
                        df_res = df_res.rename(columns={alt: "Term"})
                        break
            all_res.append(df_res)

        if not all_res:
            return pd.DataFrame()
        merged = pd.concat(all_res, ignore_index=True)
        if cfg.collapse_redundancy and not merged.empty:
            sort_col = "FDR_q" if "FDR_q" in merged.columns else ("P-value" if "P-value" in merged.columns else None)
            if sort_col:
                merged = merged.sort_values(sort_col, ascending=True)
            merged = _collapse_redundant_sets(merged, cfg.jaccard_threshold)
        return merged

    # Resolve TF and Epigenetic libraries via patterns
    tf_libs = _get_enrichr_libs_matching(cfg.TF_PATTERNS)
    epi_libs = _get_enrichr_libs_matching(cfg.EPIGENETIC_PATTERNS)
    metabs_libs = list(cfg.METABOLIC_LIBS)

    tf_all = _run_enrichr_block(tf_libs, "enrichr_tf")
    epi_all = _run_enrichr_block(epi_libs, "enrichr_epigenetic")
    metabs_all = _run_enrichr_block(metabs_libs, "enrichr_metabolites")

    # --- Write enrichment results (ALL + SIG) ---
    if tf_all is not None and not tf_all.empty:
        _write_tsv(tf_all, out_dir / "tf_all.tsv")
        tf_sig = tf_all[pd.to_numeric(tf_all.get("FDR_q", np.nan), errors="coerce") <= cfg.fdr_threshold]
        _write_tsv(tf_sig, out_dir / "tf_sig.tsv")
    else:
        tf_sig = pd.DataFrame()

    if epi_all is not None and not epi_all.empty:
        _write_tsv(epi_all, out_dir / "epigenetic_all.tsv")
        epi_sig = epi_all[pd.to_numeric(epi_all.get("FDR_q", np.nan), errors="coerce") <= cfg.fdr_threshold]
        _write_tsv(epi_sig, out_dir / "epigenetic_sig.tsv")
    else:
        epi_sig = pd.DataFrame()

    if metabs_all is not None and not metabs_all.empty:
        _write_tsv(metabs_all, out_dir / "metabolites_all.tsv")
        metabs_sig = metabs_all[pd.to_numeric(metabs_all.get("FDR_q", np.nan), errors="coerce") <= cfg.fdr_threshold]
        _write_tsv(metabs_sig, out_dir / "metabolites_sig.tsv")
    else:
        metabs_sig = pd.DataFrame()

    # ---------------- ALL_COMBINED.csv (Source first: tf/epigenetic/metabolite) ----------------
    try:
        combined_frames = []
        if tf_all is not None and not tf_all.empty:
            _tf = tf_all.copy()
            _tf["Source"] = "tf"
            combined_frames.append(_tf)
        if epi_all is not None and not epi_all.empty:
            _epi = epi_all.copy()
            _epi["Source"] = "epigenetic"
            combined_frames.append(_epi)
        if metabs_all is not None and not metabs_all.empty:
            _met = metabs_all.copy()
            _met["Source"] = "metabolite"
            combined_frames.append(_met)

        if combined_frames:
            all_combined = pd.concat(combined_frames, ignore_index=True)
            cols = ["Source"] + [c for c in all_combined.columns if c != "Source"]
            all_combined = all_combined[cols]
            (out_dir / "ALL_COMBINED.csv").write_text(all_combined.to_csv(index=False), encoding="utf-8")
    except Exception as _e:
        (out_dir / "ALL_COMBINED.ERROR.txt").write_text(str(_e))

    return {
        "cohort": cohort_name,
        "clean_scores": out_dir / "clean_scores.tsv",
        "ranked_weights": out_dir / "ranked_weights.tsv",
        "prerank_all": out_dir / "prerank_all.tsv",
        "prerank_sig": out_dir / "prerank_sig.tsv",
        "tf_all": (out_dir / "tf_all.tsv") if tf_all is not None and not tf_all.empty else None,
        "tf_sig": (out_dir / "tf_sig.tsv") if tf_sig is not None and not tf_sig.empty else None,
        "epigenetic_all": (out_dir / "epigenetic_all.tsv") if epi_all is not None and not epi_all.empty else None,
        "epigenetic_sig": (out_dir / "epigenetic_sig.tsv") if epi_sig is not None and not epi_sig.empty else None,
        "metabolites_all": (out_dir / "metabolites_all.tsv") if metabs_all is not None and not metabs_all.empty else None,
        "metabolites_sig": (out_dir / "metabolites_sig.tsv") if metabs_sig is not None and not metabs_sig.empty else None,
        "all_combined_csv": (out_dir / "ALL_COMBINED.csv") if (out_dir / "ALL_COMBINED.csv").exists() else None,
    }


# ============================== Cross-cohort combine ==============================

def _load_sig_terms(path: Optional[Path], term_col: str = "Term", fdr_col: str = "FDR_q", thr: float = 0.05) -> pd.DataFrame:
    if path is None:
        return pd.DataFrame()
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p, sep="\t")
    if df is None or df.empty or term_col not in df.columns:
        return pd.DataFrame()
    if fdr_col in df.columns:
        q = pd.to_numeric(df[fdr_col], errors="coerce")
        df = df[q <= thr]
    return df.reset_index(drop=True)


def _union_and_shared(dfs: List[pd.DataFrame], names: List[str], term_col: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    pairs = [(n, d[[term_col]].dropna()) for n, d in zip(names, dfs) if d is not None and not d.empty and term_col in d.columns]
    if not pairs:
        return pd.DataFrame(), pd.DataFrame()

    union: Dict[str, set] = {}
    for nm, d in pairs:
        for t in d[term_col].astype(str).tolist():
            union.setdefault(t, set()).add(nm)

    union_rows = []
    for t, s in union.items():
        union_rows.append({"Term": t, "Cohorts": ",".join(sorted(s)), "n_cohorts": len(s)})
    union_df = pd.DataFrame(union_rows).sort_values(["n_cohorts", "Term"], ascending=[False, True]).reset_index(drop=True)

    all_cohorts = {nm for nm, _ in pairs}
    shared_terms = [t for t, s in union.items() if s == all_cohorts and len(all_cohorts) > 1]
    shared_df = pd.DataFrame({"Term": sorted(shared_terms)})

    return union_df, shared_df


def combine_cohorts_union_and_shared(
    cohort_maps: List[Dict],
    out_dir: Path,
    fdr_threshold: float = 0.05,
) -> Dict[str, Optional[Path]]:
    out_dir = _safe_mkdir(out_dir)

    names = [m.get("cohort", f"C{i}") for i, m in enumerate(cohort_maps)]

    prerank_sig_dfs = [_load_sig_terms(m.get("prerank_sig"), "Term", "FDR_q", fdr_threshold) for m in cohort_maps]
    tf_sig_dfs      = [_load_sig_terms(m.get("tf_sig"), "Term", "FDR_q", fdr_threshold) for m in cohort_maps]
    epi_sig_dfs     = [_load_sig_terms(m.get("epigenetic_sig"), "Term", "FDR_q", fdr_threshold) for m in cohort_maps]
    metabs_sig_dfs  = [_load_sig_terms(m.get("metabolites_sig"), "Term", "FDR_q", fdr_threshold) for m in cohort_maps]

    outputs: Dict[str, Optional[Path]] = {}

    if any(d is not None and not d.empty for d in prerank_sig_dfs):
        u, s = _union_and_shared(prerank_sig_dfs, names, "Term")
        p_u = out_dir / "union_pathways.tsv"
        p_s = out_dir / "shared_pathways.tsv"
        _write_tsv(u, p_u)
        _write_tsv(s, p_s)
        outputs["union_pathways"] = p_u
        outputs["shared_pathways"] = p_s
    else:
        outputs["union_pathways"] = None
        outputs["shared_pathways"] = None

    if any(d is not None and not d.empty for d in tf_sig_dfs):
        u, s = _union_and_shared(tf_sig_dfs, names, "Term")
        p_u = out_dir / "union_tf.tsv"
        p_s = out_dir / "shared_tf.tsv"
        _write_tsv(u, p_u)
        _write_tsv(s, p_s)
        outputs["union_tf"] = p_u
        outputs["shared_tf"] = p_s
    else:
        outputs["union_tf"] = None
        outputs["shared_tf"] = None

    if any(d is not None and not d.empty for d in epi_sig_dfs):
        u, s = _union_and_shared(epi_sig_dfs, names, "Term")
        p_u = out_dir / "union_epigenetic.tsv"
        p_s = out_dir / "shared_epigenetic.tsv"
        _write_tsv(u, p_u)
        _write_tsv(s, p_s)
        outputs["union_epigenetic"] = p_u
        outputs["shared_epigenetic"] = p_s
    else:
        outputs["union_epigenetic"] = None
        outputs["shared_epigenetic"] = None

    if any(d is not None and not d.empty for d in metabs_sig_dfs):
        u, s = _union_and_shared(metabs_sig_dfs, names, "Term")
        p_u = out_dir / "union_metabolites.tsv"
        p_s = out_dir / "shared_metabolites.tsv"
        _write_tsv(u, p_u)
        _write_tsv(s, p_s)
        outputs["union_metabolites"] = p_u
        outputs["shared_metabolites"] = p_s
    else:
        outputs["union_metabolites"] = None
        outputs["shared_metabolites"] = None

    return outputs
