#!/usr/bin/env python3
from __future__ import annotations
import sys, json, re, traceback
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests
from math import comb
from scipy.stats import fisher_exact

# ================== FILTERS (P-value based) ==================
PATHWAY_P_MAX = 0.10        # keep pathways with P<=0.10 (mild)
ENTITY_P_MAX  = 0.10        # keep entities with P<=0.10 (mild)
MAX_PATHWAYS_FALLBACK = 1000  # if no pathway P-value column is present

# ================== OVERLAP GATES (mild) ==================
REQUIRE_JACCARD_MIN = 0.10
OR_MIN = 1.5
OVERLAP_MIN = 5
DEFAULT_UNIVERSE_N = 20000

# ================== SIZE GUARDS ==================
MIN_SET_SIZE = 5
MAX_SET_SIZE = 2000

CATS_CANON = {"metabolites", "epigenetic", "tf"}

def _eprint(msg: str) -> None:
    sys.stderr.write(str(msg).rstrip() + "\n"); sys.stderr.flush()

def load_table_auto(p: Path) -> pd.DataFrame:
    try:
        if not p.exists(): return pd.DataFrame()
        ext = p.suffix.lower()
        if ext in (".xlsx", ".xls"):
            df = pd.read_excel(p)
        else:
            sep = "\t" if ext == ".tsv" else ","
            df = pd.read_csv(p, sep=sep)
        df.columns = [str(c).strip() for c in df.columns]
        return df
    except Exception as e:
        _eprint(f"[load] {p}: {e}")
        return pd.DataFrame()

def _split_genes_string(s: str) -> List[str]:
    if not isinstance(s, str): return []
    parts = re.split(r"[;,]\s*|\s+\+\s+|\s*\|\s*", s.strip())
    out = [p.strip().upper() for p in parts if p and p.strip()]
    seen=set(); uniq=[]
    for g in out:
        if g not in seen:
            uniq.append(g); seen.add(g)
    return uniq

def _first_present(d: Dict[str,str], keys: List[str]) -> str | None:
    for k in keys:
        if k in d: return d[k]
    return None

def _clean_term(term: str) -> str:
    """Keep readable name; remove GO codes like '(GO:0005777)'; never return just a code."""
    t = str(term).strip()
    t_no_code = re.sub(r"\s*\(GO:\d{7}\)\s*", "", t)
    return t_no_code if t_no_code else t

# -------------------- 1) Pathways (P filter) --------------------
def collect_filtered_pathways(cdir: Path) -> Dict[str, List[str]]:
    candidates = [cdir/"prerank_all.tsv", cdir/"gsea_prerank.tsv", cdir/"gsea_ORA_fallback.tsv"]
    for p in candidates:
        df = load_table_auto(p)
        if df.empty: continue
        cols = {c.lower(): c for c in df.columns}
        term_col = _first_present(cols, ["term","pathway","name"]) or ("Term" if "Term" in df.columns else None)
        le_col   = _first_present(cols, ["lead_genes","ledge_genes","lead_genes_list"])
        genes_col= _first_present(cols, ["genes","overlap_genes"]) or ("Genes" if "Genes" in df.columns else None)
        p_col    = _first_present(cols, ["p-value","pvalue","pval","p_value","p"])
        if not term_col: 
            continue

        df_f = df.copy()
        if p_col:
            with np.errstate(all="ignore"):
                df_f[p_col] = pd.to_numeric(df_f[p_col], errors="coerce")
            df_f = df_f[df_f[p_col] <= PATHWAY_P_MAX]
            if df_f.empty:
                df_f = df.sort_values(p_col, ascending=True).head(MAX_PATHWAYS_FALLBACK)
        else:
            df_f = df.head(MAX_PATHWAYS_FALLBACK)

        out: Dict[str, List[str]] = {}
        if le_col:
            for _,r in df_f.iterrows():
                term = _clean_term(r.get(term_col, ""))
                genes = _split_genes_string(str(r.get(le_col,"")))
                if MIN_SET_SIZE <= len(genes) <= MAX_SET_SIZE and term:
                    out[term] = genes
        if not out and genes_col:
            for _,r in df_f.iterrows():
                term = _clean_term(r.get(term_col, ""))
                genes = _split_genes_string(str(r.get(genes_col,"")))
                if MIN_SET_SIZE <= len(genes) <= MAX_SET_SIZE and term:
                    out[term] = genes

        if out:
            _eprint(f"[{cdir.name}] Pathways kept: {len(out)} (from {p.name})")
            return out

    _eprint(f"[{cdir.name}] No pathway table found.")
    return {}

# -------------------- 2) Entities (P filter) --------------------
def collect_filtered_entities(cdir: Path) -> Dict[str, List[Tuple[str, List[str], float]]]:
    out = { "metabolites": [], "epigenetic": [], "tf": [] }
    df = load_table_auto(cdir/"ALL_COMBINED.csv")
    if df.empty:
        _eprint(f"[{cdir.name}] ALL_COMBINED.csv not found/empty.")
        return out

    cols = {c.lower(): c for c in df.columns}
    cat_col  = _first_present(cols, ["source","category","type","class"])
    ent_col  = _first_present(cols, ["term","entity","name","label"]) or ("Term" if "Term" in df.columns else None)
    genes_col= _first_present(cols, ["genes","overlap_genes","gene_list"]) or ("Genes" if "Genes" in df.columns else None)
    p_col    = _first_present(cols, ["p-value","pvalue","pval","p_value","p","old p-value","old_pvalue"])

    if not (cat_col and ent_col and genes_col):
        _eprint(f"[{cdir.name}] ALL_COMBINED missing essentials; columns: {list(df.columns)}")
        return out

    df_f = df.copy()
    if p_col:
        with np.errstate(all="ignore"):
            df_f[p_col] = pd.to_numeric(df_f[p_col], errors="coerce")
        df_f = df_f[df_f[p_col] <= ENTITY_P_MAX]

    if len(df_f) > 5000:
        df_f = df_f.head(5000)

    for _, r in df_f.iterrows():
        raw_cat = str(r.get(cat_col,"")).strip().lower()
        if "metab" in raw_cat:
            cat="metabolites"
        elif "epi" in raw_cat or "histone" in raw_cat or "chromatin" in raw_cat:
            cat="epigenetic"
        elif "tf" in raw_cat:
            cat="tf"
        else:
            continue

        ent = str(r.get(ent_col,"")).strip()
        genes = _split_genes_string(str(r.get(genes_col,"")))
        if not ent or not (MIN_SET_SIZE <= len(genes) <= MAX_SET_SIZE):
            continue

        pval = r.get(p_col, np.nan) if p_col else np.nan
        try: pval = float(pval)
        except Exception: pval = np.nan
        out[cat].append((ent, genes, pval))
    for k in list(out.keys()):
        if not out[k]:
            _eprint(f"[{cdir.name}] No entities kept for category: {k}")
    return out

# -------------------- Universe size --------------------
def universe_N_from_counts(cdir: Path) -> int:
    df = load_table_auto(cdir / "degs_from_counts.csv")
    if df.empty: return DEFAULT_UNIVERSE_N
    for k in ("Gene","gene","symbol","hgnc_symbol","ensembl","ensembl_id","id"):
        if k in df.columns: gcol = k; break
    else:
        gcol = df.columns[0]
    try:
        return int(pd.Series(df[gcol].astype(str).str.upper().str.strip()).nunique())
    except Exception:
        return DEFAULT_UNIVERSE_N

# -------------------- Stats --------------------
def _fisher_exact_pvalue(k:int,a:int,b:int,N:int)->float:
    try:
        
        x11=k; x12=max(0,a-k); x21=max(0,b-k); x22=max(0,N-a-b+k)
        _, p = fisher_exact([[x11,x12],[x21,x22]], alternative="greater")
        return float(p)
    except Exception:
        try:
            
            def pmf(x): return (comb(a,x)*comb(N-a,b-x))/max(1,comb(N,b))
            p = sum(pmf(t) for t in range(k, min(a,b)+1))
            return float(min(1.0, max(0.0, p)))
        except Exception:
            return 1.0

def _bh_fdr(pvals: List[float]) -> List[float]:
    try:
        return multipletests(pvals, method="fdr_bh")[1].tolist()
    except Exception:
        m=len(pvals); order=np.argsort(pvals); out=[1.0]*m; prev=1.0
        for r,i in enumerate(order, start=1):
            prev=min(prev, pvals[i]*m/r); out[i]=min(1.0, prev)
        return out

# -------------------- Linking (no UP/DOWN) --------------------
def link_category(pathways: Dict[str, List[str]],
                  entities_rows: List[Tuple[str, List[str]]],
                  N: int,
                  cohort_name: str,
                  cat: str) -> List[Dict]:
    if not pathways or not entities_rows:
        return []

    Psets = {pterm: set(g.upper() for g in genes) for pterm, genes in pathways.items()}
    Esets = [(ent, set(g.upper() for g in genes)) for ent, genes in entities_rows]

    prelim = []
    p_count = 0
    for pterm, P in Psets.items():
        a = len(P)
        if a == 0: continue
        p_count += 1
        if p_count % 200 == 0:
            _eprint(f"[{cohort_name}] {cat}: screened {p_count} pathways...")
        for ent_term, E in Esets:
            b = len(E)
            if b == 0: continue
            inter = P & E
            k = len(inter)
            if k < OVERLAP_MIN:
                continue
            union = len(P | E)
            j = k / float(union) if union else 0.0
            if j < REQUIRE_JACCARD_MIN:
                continue
            x11=k; x12=max(0,a-k); x21=max(0,b-k); x22=max(0,N-a-b+k)
            OR = (x11*x22)/max(1e-12, x12*x21)
            if OR <= OR_MIN:
                continue
            prelim.append((pterm, ent_term, k, a, b, j, sorted(inter), OR))

    if not prelim:
        return []

    rows=[]
    for pterm, ent_term, k, a, b, j, overlap, OR in prelim:
        pval = _fisher_exact_pvalue(k, a, b, N)
        rows.append(dict(pathway=pterm, entity=ent_term, k=k, a=a, b=b, N=N,
                         Jaccard=j, OR=OR, pval=pval, overlap_genes=overlap))
    qvals = _bh_fdr([r["pval"] for r in rows])
    for r,q in zip(rows,qvals):
        r["qval"] = q

    out=[r for r in rows if r["pval"] <= PATHWAY_P_MAX
                        and r["k"] >= OVERLAP_MIN
                        and r["OR"] > OR_MIN
                        and r["Jaccard"] >= REQUIRE_JACCARD_MIN]
    out.sort(key=lambda x:(x["pathway"].lower(), x["pval"], -x["OR"], -x["k"]))
    return out

# -------------------- Per-cohort --------------------
def process_cohort(cdir: Path) -> Path | None:
    try:
        pathways = collect_filtered_pathways(cdir)
        if not pathways:
            _eprint(f"[{cdir.name}] Skipping — no pathways after filtering.")
            return None
        entities = collect_filtered_entities(cdir)
        if not any(entities[cat] for cat in entities):
            _eprint(f"[{cdir.name}] Skipping — no entities after filtering.")
            return None

        N = universe_N_from_counts(cdir)

        hits_by_cat: Dict[str, List[Dict]] = {}
        for cat in CATS_CANON:
            rows = [(e, g) for (e, g, _p) in entities[cat]]
            hits_by_cat[cat] = link_category(pathways, rows, N, cdir.name, cat)

        if not any(hits_by_cat[cat] for cat in hits_by_cat):
            _eprint(f"[{cdir.name}] No qualifying overlaps after gates.")
            return None

        result: Dict[str, Dict[str, List[Dict]]] = {}
        def _ensure_term(t: str):
            if t not in result:
                result[t] = {k: [] for k in CATS_CANON}
        for cat in CATS_CANON:
            for h in hits_by_cat[cat]:
                _ensure_term(h["pathway"])
                result[h["pathway"]][cat].append(dict(
                    entity=h["entity"],
                    OR=round(h["OR"],4),
                    pval=h["pval"],
                    qval=h["qval"],
                    Jaccard=round(h["Jaccard"],4),
                    k=h["k"], a=h["a"], b=h["b"], N=h["N"],
                    overlap_genes=h["overlap_genes"]
                ))

        outdir = cdir / "overlap"; outdir.mkdir(parents=True, exist_ok=True)
        outpath = outdir / "pathway_entity_overlap.json"
        with open(outpath, "w") as f:
            json.dump(result, f, indent=2)
        _eprint(f"[{cdir.name}] Wrote JSON: {outpath}")
        return outpath
    except Exception as e:
        _eprint(f"[{cdir.name}] ERROR:\n{e}\n{traceback.format_exc()}")
        return None

# -------------------- Programmatic entrypoint --------------------
def run_json_maker(root: Path | str) -> int:
    root = Path(root).expanduser()
    if not root.exists() or not root.is_dir():
        _eprint(f"Root folder not found or not a directory: {root}")
        return 0

    exclude = {"baseline_consensus","comparison","gl_comparison","jsons_all_folder",
               "gc_comparison"}  # include gc_comparison just in case
    cohorts = sorted([d for d in root.iterdir() if d.is_dir() and d.name not in exclude])
    if not cohorts:
        _eprint("No disease folders found."); return 0

    bundle = root / "jsons_all_folder"
    try: bundle.mkdir(parents=True, exist_ok=True)
    except Exception as e: _eprint(f"[bundle] mkdir failed: {e}")

    written = 0
    for cdir in cohorts:
        outpath = process_cohort(cdir)
        if outpath is not None:
            written += 1
            try:
                target = bundle / f"{cdir.name}.json"   # diseaseName.json
                with open(outpath, "rb") as src, open(target, "wb") as dst:
                    dst.write(src.read())
                _eprint(f"[bundle] {cdir.name} -> {target}")
            except Exception as e:
                _eprint(f"[bundle] copy failed for {cdir.name}: {e}")
    _eprint(f"Done. JSONs written for {written} cohort(s).")
    return written

# -------------------- CLI entrypoint --------------------
def main():
    if len(sys.argv) < 2:
        _eprint("Usage: python json_maker.py /path/to/out_root")
        sys.exit(2)
    n = run_json_maker(sys.argv[1])
    if n < 0: sys.exit(1)

if __name__ == "__main__":
    main()
