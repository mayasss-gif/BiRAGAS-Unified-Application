from __future__ import annotations
import json, re, argparse, glob
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
    "run_gl_folder",
    "main",
]

# ============================== Config ==============================

@dataclass(frozen=True)
class GLEnrichConfig:
    enabled: bool = True

    # Select input mode:
    #   - True  -> GL mode (plain gene list; accepts inline string or file)
    #   - False -> GC mode (GeneCards-style JSON with data arrays)
    use_gene_list: bool = True

    # Ranked-list / prerank GSEA
    top_k: int = 500
    prerank_permutations: int = 100
    prerank_min_size: int = 10
    prerank_max_size: int = 2000
    seed: int = 13
    fdr_threshold: float = 0.05

    # Pathway libs (includes GO BP/CC/MF)
    pathway_libs: Tuple[str, ...] = (
        "MSigDB_Hallmark_2020",
        "KEGG_2021_Human",
        "Reactome_2022",
        "GO_Biological_Process_2021",
        "GO_Cellular_Component_2021",
        "GO_Molecular_Function_2021",
    )

    # Regex library discovery (split TF vs Epigenetic)
    TF_PATTERNS: Tuple[str, ...] = (
        r"ENCODE_TF_ChIP", r"ChEA", r"JASPAR", r"TRANSFAC", r"ChIP-X", r"DoRothEA", r"\bTF\b",
    )
    EPIGENETIC_PATTERNS: Tuple[str, ...] = (
        r"Histone", r"\bH3K", r"Chromatin", r"Enhancer", r"Epigenetic",
    )

    # Metabolite libs (explicit Enrichr category)
    METABOLIC_LIBS: Tuple[str, ...] = ("HMDB_Metabolites",)

    # Redundancy/Jaccard control
    collapse_redundancy: bool = True
    jaccard_threshold: float = 0.7

    # Null audit
    null_shuffles: int = 0

    # Optional HGNC alias→symbol mapping file (CSV/TSV)
    hgnc_map_file: Optional[str] = None


# ============================== Utilities ==============================

def _safe_mkdir(p: Path) -> Path:
    p = Path(p); p.mkdir(parents=True, exist_ok=True); return p

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
        df.head(0).to_csv(path, sep="\t", index=False); return
    df.to_csv(path, sep="\t", index=False)

def _clean_symbol(x: str) -> str:
    if x is None: return ""
    x = str(x).strip()
    x = re.sub(r"\s+", "", x)
    return x.upper()

def _load_hgnc_map(hgnc_map_file: Optional[str]) -> Dict[str, str]:
    if not hgnc_map_file: return {}
    p = Path(hgnc_map_file)
    if not p.exists(): return {}
    sep = "\t" if p.suffix.lower() in {".tsv", ".tab"} else ","
    df = pd.read_csv(p, sep=sep)
    cols = [c.lower() for c in df.columns]
    alias_col = None; symbol_col = None
    for i, c in enumerate(cols):
        if re.search(r"(alias|synonym|prev)", c): alias_col = df.columns[i]
        if re.search(r"(symbol|hgnc)", c): symbol_col = df.columns[i]
    if alias_col is None or symbol_col is None:
        alias_col = df.columns[0]
        symbol_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]
    mapping = {}
    for a, s in zip(df[alias_col], df[symbol_col]):
        a = _clean_symbol(a); s = _clean_symbol(s)
        if a and s: mapping[a] = s
    return mapping

def _map_to_hgnc(gene_series: pd.Series, hgnc_map: Dict[str, str]) -> pd.Series:
    cleaned = gene_series.map(_clean_symbol)
    if not hgnc_map: return cleaned
    return cleaned.map(lambda g: hgnc_map.get(g, g))

def _rank_from_scores_any(df: pd.DataFrame, gene_col: str, score_col: Optional[str]) -> pd.Series:
    """
    If score_col exists -> use numeric scores with jitter; else
    construct a gentle descending weight by the provided order.
    """
    d = df.copy()
    d = d.dropna(subset=[gene_col])
    d[gene_col] = d[gene_col].astype(str)
    # Case 1: real scores present
    if score_col and score_col in d.columns:
        d[score_col] = pd.to_numeric(d[score_col], errors="coerce")
        d = d.dropna(subset=[score_col])
        if d.empty: raise ValueError("No valid numeric scores found.")
        d = (d.sort_values(score_col, key=lambda s: np.abs(s), ascending=False)
               .drop_duplicates(subset=[gene_col], keep="first")
               .reset_index(drop=True))
        signed = (d[score_col].min() < 0)
        w = np.log1p(np.abs(d[score_col].values))
        if signed: w = w * np.sign(d[score_col].values)
        def _jit(g: str) -> float:
            b = g.encode("utf-8"); return (sum(b) % 1000) / 1e9
        jit = np.array([_jit(g) for g in d[gene_col]])
        w = w + jit
        ser = pd.Series(w, index=d[gene_col])
        return ser.sort_values(ascending=False)
    # Case 2: plain gene list, build descending weights
    d = d.drop_duplicates(subset=[gene_col], keep="first").reset_index(drop=True)
    n = len(d)
    if n == 0: raise ValueError("Empty gene list.")
    base = np.linspace(1.0, 0.01, n)
    def _jit(g: str) -> float:
        b = g.encode("utf-8"); return (sum(b) % 1000) / 1e9
    jit = np.array([_jit(g) for g in d[gene_col]])
    w = base + jit
    ser = pd.Series(w, index=d[gene_col])
    return ser.sort_values(ascending=False)

def _split_inline_genes(s: str) -> List[str]:
    """
    Robustly parse genes from free-form text:
    - accepts JSON-like arrays, comma/semicolon/pipe separated, multi-line, extra spaces
    - strips surrounding quotes and trailing punctuation
    - keeps hyphens and special chars like '@' (e.g., PTPRJ-AS1, IFN1@)
    - de-duplicates, uppercases
    """
    if not isinstance(s, str):
        return []
    s = re.sub(r"[\[\]\{\}]", " ", s)
    parts = re.split(r"[,\s;|]+", s.strip())

    out: List[str] = []
    seen: set = set()
    for p in parts:
        t = p.strip().strip('"\'')
        if not t:
            continue
        if t.endswith(",") or t.endswith("."):
            t = t[:-1].strip()
        t = t.upper()
        if t and t not in seen:
            seen.add(t)
            out.append(t)
    return out


# ------------------------ input loaders ------------------------

def _load_genelist_any(path_or_inline: str, gene_key: Optional[str], score_key: Optional[str]) -> pd.DataFrame:
    """
    GL mode:
      - .txt  -> accepts one-per-line OR JSON-like array with quotes/commas/indents
      - .csv/.tsv -> auto-detect gene column; optional score column
      - .json -> flat list of genes
      - otherwise, treat input as an inline string (same robust splitter)
    Returns DataFrame with columns: Gene (required), Score (optional).
    """
    s = str(path_or_inline)
    p = Path(s)
    if p.exists():
        ext = p.suffix.lower()
        if ext == ".txt":
            content = p.read_text(encoding="utf-8", errors="ignore")
            genes = _split_inline_genes(content)
            return pd.DataFrame({"Gene": genes})

        if ext in {".csv", ".tsv", ".tab"}:
            sep = "\t" if ext in {".tsv", ".tab"} else ","
            df = pd.read_csv(p, sep=sep)
            cols = {c.lower(): c for c in df.columns}
            gcol = gene_key if (gene_key and gene_key in df.columns) else (
                cols.get("gene") or cols.get("symbol") or cols.get("gene_symbol") or list(df.columns)[0]
            )
            scol = score_key if (score_key and score_key in df.columns) else None
            if not scol:
                for k in ("score","gene_score","log2fc","stat","weight","rank","t","wald","beta"):
                    if k in cols:
                        scol = cols[k]; break
            out = pd.DataFrame({"Gene": df[gcol].astype(str)})
            if scol:
                out["Score"] = df[scol]
            return out

        if ext == ".json":
            raw = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(raw, list):
                return pd.DataFrame({"Gene": [str(x) for x in raw]})
            raise ValueError("JSON must be a flat list of genes for GL mode.")

        raise ValueError(f"Unsupported file format in GL mode: {p.suffix}")

    # Not a path -> treat as inline text
    genes = _split_inline_genes(s)
    if not genes:
        raise ValueError("Inline gene list is empty / unparsable.")
    return pd.DataFrame({"Gene": genes})

def _load_genecards_json(path: Path, gene_key: str, score_key: str) -> pd.DataFrame:
    """
    GC mode: GeneCards-like JSON with {"data": {...}} arrays.
    Returns DataFrame with columns: Gene, Score? (optional).
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Scores JSON not found: {p}")
    if p.suffix.lower() != ".json":
        raise ValueError("GC mode expects a .json input.")
    raw = json.loads(p.read_text(encoding="utf-8"))
    if not (isinstance(raw, dict) and "data" in raw and isinstance(raw["data"], dict)):
        raise ValueError("JSON must be an object with a 'data' object containing arrays.")
    data = raw["data"]
    if gene_key not in data:
        raise KeyError(f"Missing required gene_key '{gene_key}' in JSON['data'].")
    genes = data[gene_key]
    out = {"Gene": genes}
    if score_key in data:
        out["Score"] = data[score_key]
    elif "disorder_score" in data:
        out["Score"] = data["disorder_score"]
    return pd.DataFrame(out)

# ------------------------ enrichr helpers ------------------------

def _adjust_fdr(df: pd.DataFrame, pcol: str = "P-value", out_col: str = "FDR_q") -> pd.DataFrame:
    df = df.copy()
    if pcol not in df.columns or df.empty: return df
    pv = pd.to_numeric(df[pcol], errors="coerce")
    mask = pv.notna()
    if mask.any():
        _, q, _, _ = multipletests(pv[mask].values, method="fdr_bh")
        df.loc[mask, out_col] = q
    return df

def _collapse_redundant_sets(res_df: pd.DataFrame, j_threshold: float) -> pd.DataFrame:
    if res_df is None or res_df.empty or "Term" not in res_df.columns: return res_df
    def _tok(s: str) -> set:
        toks = re.split(r"[\W_]+", str(s).upper()); toks = [t for t in toks if t]; return set(toks)
    keep_rows = []; seen = []
    for i, row in res_df.iterrows():
        tset = _tok(row["Term"]); ok = True
        for prev in seen:
            inter = len(tset & prev); union = len(tset | prev) if (tset or prev) else 1
            j = inter / union
            if j >= j_threshold: ok = False; break
        if ok: keep_rows.append(i); seen.append(tset)
    return res_df.loc[keep_rows].reset_index(drop=True)

def _enrichr_compat(gene_list: Iterable[str], gene_sets, outdir: Path, description: str = "") -> gseapy.enrichr:
    # Always call without `description` to support older gseapy versions
    return enrichr(
        gene_list=list(gene_list),
        gene_sets=gene_sets,
        outdir=str(outdir),
        no_plot=True,
        cutoff=1.0,
    )

def _get_enrichr_libs_matching(patterns: Tuple[str, ...]) -> List[str]:
    try:
        all_libs = set(gseapy.get_library_name())
    except Exception:
        all_libs = {
            "ChEA_2022","ENCODE_TF_ChIP-seq_2015","ENCODE_TF_ChIP-seq_2014",
            "JASPAR_PWM_Human_2025","TRANSFAC_and_JASPAR_PWMs","ChIP-X_Enrichment_Analysis_2016",
            "DoRothEA","Histone_Modification","Epigenetic_Landscape_In_Silico",
            "Chromatin_Regulators","Enhancer_Atlas","HMDB_Metabolites"
        }
    matched = set()
    for lib in all_libs:
        for pat in patterns:
            if re.search(pat, lib, flags=re.IGNORECASE):
                matched.add(lib); break
    return sorted(matched)

def _top_genes_for_enrichr(weights: pd.Series, k: int) -> List[str]:
    k = max(1, int(k)); return list(weights.head(k).index)

def _clean_term_name(term: str) -> str:
    """Remove trailing '(GO:1234567)' keeping human-readable name."""
    t = str(term).strip()
    t = re.sub(r"\s*\(GO:\d{7}\)\s*$", "", t)
    return t if t else str(term)

# ============================== Main per-cohort driver ==============================

def run_gl_enrichment_for_cohort(
    cohort_name: str,
    scores_table: Path | str,   # may be a path or an inline gene string
    out_root: Path,
    cfg: GLEnrichConfig,
    gene_col: str = "gene_symbol",   # used in GC mode or CSV/TSV override
    score_col: str = "gene_score",   # used in GC mode or CSV/TSV override
) -> Dict[str, Optional[Path]]:
    if not cfg.enabled:
        raise RuntimeError("GL enrichment disabled via config.")

    out_root = Path(out_root)
    out_dir = _safe_mkdir(out_root / str(cohort_name))
    _record_versions(out_dir)

    # ------------ Load & clean inputs ------------
    if cfg.use_gene_list:
        raw_df = _load_genelist_any(str(scores_table), gene_key=gene_col, score_key=score_col)
    else:
        raw_df = _load_genecards_json(Path(scores_table), gene_key=gene_col, score_key=score_col)

    hgnc_map: Dict[str, str] = _load_hgnc_map(cfg.hgnc_map_file)
    raw_df["Gene"] = _map_to_hgnc(raw_df["Gene"], hgnc_map)
    raw_df = raw_df.dropna(subset=["Gene"])
    raw_df = raw_df[raw_df["Gene"] != ""]
    raw_df = raw_df.drop_duplicates(subset=["Gene"], keep="first").reset_index(drop=True)

    # weights for prerank (fabricated if Score missing)
    weights = _rank_from_scores_any(raw_df, "Gene", "Score" if "Score" in raw_df.columns else None)
    wdf = pd.DataFrame({"Gene": weights.index, "RankWeight": weights.values})
    _write_tsv(wdf, out_dir / "ranked_weights.tsv")
    _write_tsv(raw_df[["Gene"] + (["Score"] if "Score" in raw_df.columns else [])], out_dir / "clean_scores.tsv")

    meta = {
        "cohort": cohort_name,
        "source_file": str(scores_table),
        "input_mode": "genes+scores" if "Score" in raw_df.columns else "genes_only",
        "use_gene_list": cfg.use_gene_list,
        "identifier_policy": "Gene symbols uppercased; optional alias→symbol via HGNC map.",
        "universe_policy": "Prerank uses full ranked list; Enrichr uses top_k.",
        "config": asdict(cfg),
    }
    (out_dir / "input_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

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
            threads=1,      # use threads instead of deprecated processes
            no_plot=True,
            verbose=False,
        )
        df_res = pre.res2d.reset_index()
        df_res["Library"] = lib
        if "FDR q-val" in df_res.columns and "FDR_q" not in df_res.columns:
            df_res = df_res.rename(columns={"FDR q-val": "FDR_q"})
        if "P-value" in df_res.columns and "FDR_q" not in df_res.columns:
            df_res = _adjust_fdr(df_res, pcol="P-value", out_col="FDR_q")
        if "Term" in df_res.columns:
            df_res["Term"] = df_res["Term"].map(_clean_term_name)
        all_prerank.append(df_res)

    prerank_all = pd.concat(all_prerank, ignore_index=True) if all_prerank else pd.DataFrame()
    if cfg.collapse_redundancy and not prerank_all.empty:
        sort_col = "FDR_q" if "FDR_q" in prerank_all.columns else ("P-value" if "P-value" in prerank_all.columns else None)
        if sort_col:
            prerank_all = prerank_all.sort_values(sort_col, ascending=True)
        prerank_all = _collapse_redundant_sets(prerank_all, cfg.jaccard_threshold)
    _write_tsv(prerank_all, out_dir / "prerank_all.tsv")

    if not prerank_all.empty and "FDR_q" in prerank_all.columns:
        prerank_sig = prerank_all[pd.to_numeric(prerank_all["FDR_q"], errors="coerce") <= cfg.fdr_threshold]
    else:
        prerank_sig = pd.DataFrame()
    _write_tsv(prerank_sig, out_dir / "prerank_sig.tsv")

    # ------------ Enrichr blocks (TF / Epigenetic / Metabolites) ------------
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
            if "Adjusted P-value" in df_res.columns and "FDR_q" not in df_res.columns:
                df_res = df_res.rename(columns={"Adjusted P-value": "FDR_q"})
            if "P-value" in df_res.columns and "FDR_q" not in df_res.columns:
                df_res = _adjust_fdr(df_res, pcol="P-value", out_col="FDR_q")
            if "Term" not in df_res.columns:
                for alt in ("Gene_set", "Term_name", "term_name"):
                    if alt in df_res.columns:
                        df_res = df_res.rename(columns={alt: "Term"}); break
            if "Term" in df_res.columns:
                df_res["Term"] = df_res["Term"].map(_clean_term_name)
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

    tf_libs  = _get_enrichr_libs_matching(cfg.TF_PATTERNS)
    epi_libs = _get_enrichr_libs_matching(cfg.EPIGENETIC_PATTERNS)
    met_libs = list(cfg.METABOLIC_LIBS)

    tf_all   = _run_enrichr_block(tf_libs, "enrichr_tf")
    epi_all  = _run_enrichr_block(epi_libs, "enrichr_epigenetic")
    metab_all= _run_enrichr_block(met_libs, "enrichr_metabolites")

    # Write enrichment tables
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

    if metab_all is not None and not metab_all.empty:
        _write_tsv(metab_all, out_dir / "metabolites_all.tsv")
        metab_sig = metab_all[pd.to_numeric(metab_all.get("FDR_q", np.nan), errors="coerce") <= cfg.fdr_threshold]
        _write_tsv(metab_sig, out_dir / "metabolites_sig.tsv")
    else:
        metab_sig = pd.DataFrame()

    # ---------------- ALL_COMBINED.csv ----------------
    try:
        frames = []
        if tf_all is not None and not tf_all.empty:
            _tf = tf_all.copy();  _tf["Source"] = "tf";         frames.append(_tf)
        if epi_all is not None and not epi_all.empty:
            _ep = epi_all.copy(); _ep["Source"] = "epigenetic"; frames.append(_ep)
        if metab_all is not None and not metab_all.empty:
            _mt = metab_all.copy(); _mt["Source"] = "metabolite"; frames.append(_mt)
        if frames:
            all_combined = pd.concat(frames, ignore_index=True)
            cols = ["Source"] + [c for c in all_combined.columns if c != "Source"]
            all_combined = all_combined[cols]
            (out_dir / "ALL_COMBINED.csv").write_text(all_combined.to_csv(index=False), encoding="utf-8")
    except Exception as e:
        (out_dir / "ALL_COMBINED.ERROR.txt").write_text(str(e))

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
        "metabolites_all": (out_dir / "metabolites_all.tsv") if metab_all is not None and not metab_all.empty else None,
        "metabolites_sig": (out_dir / "metabolites_sig.tsv") if metab_sig is not None and not metab_sig.empty else None,
        "all_combined_csv": (out_dir / "ALL_COMBINED.csv") if (out_dir / "ALL_COMBINED.csv").exists() else None,
    }

# ============================== Cross-cohort combine ==============================

def _load_sig_terms(path: Optional[Path], term_col: str = "Term", fdr_col: str = "FDR_q", thr: float = 0.05) -> pd.DataFrame:
    if path is None: return pd.DataFrame()
    p = Path(path)
    if not p.exists(): return pd.DataFrame()
    df = pd.read_csv(p, sep="\t")
    if df is None or df.empty or term_col not in df.columns: return pd.DataFrame()
    if fdr_col in df.columns:
        q = pd.to_numeric(df[fdr_col], errors="coerce"); df = df[q <= thr]
    return df.reset_index(drop=True)

def _union_and_shared(dfs: List[pd.DataFrame], names: List[str], term_col: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    pairs = [(n, d[[term_col]].dropna()) for n, d in zip(names, dfs) if d is not None and not d.empty and term_col in d.columns]
    if not pairs: return pd.DataFrame(), pd.DataFrame()
    union: Dict[str, set] = {}
    for nm, d in pairs:
        for t in d[term_col].astype(str).tolist():
            union.setdefault(t, set()).add(nm)
    union_rows = [{"Term": t, "Cohorts": ",".join(sorted(s)), "n_cohorts": len(s)} for t, s in union.items()]
    union_df   = pd.DataFrame(union_rows).sort_values(["n_cohorts","Term"], ascending=[False, True]).reset_index(drop=True)
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
    met_sig_dfs     = [_load_sig_terms(m.get("metabolites_sig"), "Term", "FDR_q", fdr_threshold) for m in cohort_maps]

    outputs: Dict[str, Optional[Path]] = {}
    if any(d is not None and not d.empty for d in prerank_sig_dfs):
        u, s = _union_and_shared(prerank_sig_dfs, names, "Term")
        _write_tsv(u, out_dir / "union_pathways.tsv")
        _write_tsv(s, out_dir / "shared_pathways.tsv")
        outputs["union_pathways"] = out_dir / "union_pathways.tsv"
        outputs["shared_pathways"] = out_dir / "shared_pathways.tsv"
    else:
        outputs["union_pathways"] = None; outputs["shared_pathways"] = None

    if any(d is not None and not d.empty for d in tf_sig_dfs):
        u, s = _union_and_shared(tf_sig_dfs, names, "Term")
        _write_tsv(u, out_dir / "union_tf.tsv")
        _write_tsv(s, out_dir / "shared_tf.tsv")
        outputs["union_tf"] = out_dir / "union_tf.tsv"
        outputs["shared_tf"] = out_dir / "shared_tf.tsv"
    else:
        outputs["union_tf"] = None; outputs["shared_tf"] = None

    if any(d is not None and not d.empty for d in epi_sig_dfs):
        u, s = _union_and_shared(epi_sig_dfs, names, "Term")
        _write_tsv(u, out_dir / "union_epigenetic.tsv")
        _write_tsv(s, out_dir / "shared_epigenetic.tsv")
        outputs["union_epigenetic"] = out_dir / "union_epigenetic.tsv"
        outputs["shared_epigenetic"] = out_dir / "shared_epigenetic.tsv"
    else:
        outputs["union_epigenetic"] = None; outputs["shared_epigenetic"] = None

    if any(d is not None and not d.empty for d in met_sig_dfs):
        u, s = _union_and_shared(met_sig_dfs, names, "Term")
        _write_tsv(u, out_dir / "union_metabolites.tsv")
        _write_tsv(s, out_dir / "shared_metabolites.tsv")
        outputs["union_metabolites"] = out_dir / "union_metabolites.tsv"
        outputs["shared_metabolites"] = out_dir / "shared_metabolites.tsv"
    else:
        outputs["union_metabolites"] = None; outputs["shared_metabolites"] = None

    return outputs


# ============================== CLI wrapper expected by mdp-gl ==============================

def _infer_table(path: Path) -> pd.DataFrame:
    p = str(path)
    if p.lower().endswith(".csv"):
        return pd.read_csv(p)
    if p.lower().endswith(".tsv") or p.lower().endswith(".txt"):
        try:
            return pd.read_csv(p, sep="\t")
        except Exception:
            return pd.read_csv(p)
    return pd.read_csv(p, sep=None, engine="python")

def _pick_cols(df: pd.DataFrame, gene_col_opt: Optional[str], score_col_opt: Optional[str]) -> tuple[str, Optional[str]]:
    if gene_col_opt and gene_col_opt in df.columns:
        gcol = gene_col_opt
    else:
        lc = {c.lower(): c for c in df.columns}
        gcol = (
            lc.get("gene")
            or lc.get("symbol")
            or lc.get("gene_symbol")
            or lc.get("hgnc_symbol")
            or df.columns[0]
        )
    scol = None
    if score_col_opt and score_col_opt in df.columns:
        scol = score_col_opt
    else:
        lc = {c.lower(): c for c in df.columns}
        for k in ("score","gene_score","log2fc","logfc","stat","weight","rank","t","wald","beta"):
            if k in lc:
                scol = lc[k]
                break
    return gcol, scol

def run_gl_folder(
    input_dir: str,
    out_root: str,
    fdr: float = 0.05,
    topk: int = 500,
    perms: int = 100,
    minset: int = 10,
    maxset: int = 2000,
    seed: int = 13,
    collapse: bool = True,
    jaccard: Optional[float] = None,
    jaccard_thresh: Optional[float] = None,  # accept legacy kw
    gene_col: Optional[str] = None,
    score_col: Optional[str] = None,
    **kwargs,  # absorb any extra args gracefully
) -> bool:
    """
    Entry point expected by mdp-gl. Scans a folder of gene-list files and runs GL enrichment.
    Produces per-cohort outputs under <out_root>/GL_enrich/<cohort>/, including:
      - clean_scores.tsv, ranked_weights.tsv
      - prerank_all.tsv, prerank_sig.tsv
      - gsea_prerank.tsv (compat for mdp-json-make)
      - tf/epigenetic/metabolites tables + ALL_COMBINED.csv
    """
    in_dir = Path(input_dir)
    out_root = Path(out_root)
    if not in_dir.is_dir():
        print(f"[GL_enrich] Input is not a directory: {in_dir}")
        return False
    out_root.mkdir(parents=True, exist_ok=True)

    files = sorted(
        glob.glob(str(in_dir / "*.txt")) +
        glob.glob(str(in_dir / "*.tsv")) +
        glob.glob(str(in_dir / "*.csv"))
    )
    if not files:
        print(f"[GL_enrich] No gene-list files found in: {in_dir}")
        return False

    # Handle both jaccard arg names
    if jaccard is None and jaccard_thresh is not None:
        jaccard = jaccard_thresh
    if jaccard is None:
        jaccard = 0.7

    # Base output once (avoid disease/disease nesting)
    base_out = out_root / "GL_enrich"
    base_out.mkdir(parents=True, exist_ok=True)

    cfg = GLEnrichConfig(
        enabled=True,
        use_gene_list=True,
        top_k=topk,
        prerank_permutations=perms,
        prerank_min_size=minset,
        prerank_max_size=maxset,
        seed=seed,
        fdr_threshold=fdr,
        collapse_redundancy=collapse,
        jaccard_threshold=float(jaccard),
    )

    for fp in files:
        path = Path(fp)
        disease = path.stem

        print(f"[GL_enrich] Running cohort={disease} from {path}")
        # Run the core with base_out (it will create .../GL_enrich/<disease>/)
        m = run_gl_enrichment_for_cohort(
            cohort_name=disease,
            scores_table=str(path),
            out_root=base_out,
            cfg=cfg,
            gene_col=gene_col or "gene_symbol",
            score_col=score_col or "gene_score",
        )

        # Create gsea_prerank.tsv copy for json_maker
        disease_dir = base_out / disease
        src = None
        if m.get("prerank_all") and Path(m["prerank_all"]).exists():
            src = Path(m["prerank_all"])
        elif m.get("prerank_sig") and Path(m["prerank_sig"]).exists():
            src = Path(m["prerank_sig"])

        if src is not None:
            dst = disease_dir / "gsea_prerank.tsv"
            try:
                df = pd.read_csv(src, sep="\t")
                df.to_csv(dst, sep="\t", index=False)
                print(f"[GL_enrich] wrote: {dst}")
            except Exception as e:
                print(f"[GL_enrich] WARN: could not create gsea_prerank.tsv: {e}")

    print(f"[GL_enrich] DONE. Outputs under: {base_out}")
    return True

# -------- module CLI --------

def _parse_args():
    ap = argparse.ArgumentParser(description="Run GL enrichment from a folder of gene-list files.")
    ap.add_argument("--input", required=True, help="Folder with .txt/.tsv/.csv gene lists.")
    ap.add_argument("--out-root", required=True, help="Output root directory.")
    ap.add_argument("--fdr", type=float, default=0.05)
    ap.add_argument("--topk", type=int, default=500)
    ap.add_argument("--perms", type=int, default=100)
    ap.add_argument("--minset", type=int, default=10)
    ap.add_argument("--maxset", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--collapse", action="store_true")
    ap.add_argument("--jaccard", type=float, default=0.7)
    ap.add_argument("--gene-col", default=None)
    ap.add_argument("--score-col", default=None)
    return ap.parse_args()

def main() -> int:
    args = _parse_args()
    ok = run_gl_folder(
        input_dir=args.input,
        out_root=args.out_root,
        fdr=args.fdr,
        topk=args.topk,
        perms=args.perms,
        minset=args.minset,
        maxset=args.maxset,
        seed=args.seed,
        collapse=args.collapse,
        jaccard=args.jaccard,
        gene_col=args.gene_col,
        score_col=args.score_col,
    )
    return 0 if ok else 1

if __name__ == "__main__":
    raise SystemExit(main())
