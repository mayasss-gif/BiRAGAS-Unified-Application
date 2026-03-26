# GC_enricher/gc_make_jsons.py
from __future__ import annotations
import argparse, json, re
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd

GO_BP_RE = re.compile(r"Biological[_\s]?Process", re.I)
GO_CC_RE = re.compile(r"Cellular[_\s]?Component", re.I)
GO_MF_RE = re.compile(r"Molecular[_\s]?Function", re.I)

def _read_tsv_maybe(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, sep="\t")
    except Exception:
        try:
            return pd.read_csv(path)
        except Exception:
            return pd.DataFrame()

def _terms_only(df: pd.DataFrame) -> List[str]:
    if df is None or df.empty:
        return []
    col = "Term" if "Term" in df.columns else (df.columns[0] if len(df.columns) else "Term")
    return sorted(pd.Series(df[col]).dropna().astype(str).unique().tolist())

def _load_sig_terms(path: Optional[Path], thr: float = 0.05) -> pd.DataFrame:
    if path is None:
        return pd.DataFrame()
    df = _read_tsv_maybe(path)
    if df.empty:
        return df
    if "FDR_q" in df.columns:
        q = pd.to_numeric(df["FDR_q"], errors="coerce")
        df = df.loc[q.notna() & (q <= thr)].copy()
    return df

def _split_pathways(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Split prerank sig results into buckets:
      - pathways (non-GO libs)
      - biological_function (GO:BP)
      - cell_component (GO:CC)
      - mechanism (GO:MF)
    """
    out = {
        "pathways": [],
        "biological_function": [],
        "cell_component": [],
        "mechanism": [],
    }
    if df.empty:
        return out

    term_col = "Term" if "Term" in df.columns else df.columns[0]
    lib_col = "Library" if "Library" in df.columns else None

    if lib_col is None:
        # If we don't know the library, just dump everything under pathways
        out["pathways"] = _terms_only(df)
        return out

    df[term_col] = df[term_col].astype(str)
    df[lib_col] = df[lib_col].astype(str)

    def _get(mask):
        return sorted(df.loc[mask, term_col].dropna().astype(str).unique().tolist())

    m_bp = df[lib_col].str.contains(GO_BP_RE)
    m_cc = df[lib_col].str.contains(GO_CC_RE)
    m_mf = df[lib_col].str.contains(GO_MF_RE)
    not_go = ~(m_bp | m_cc | m_mf)

    out["biological_function"] = _get(m_bp)
    out["cell_component"]     = _get(m_cc)
    out["mechanism"]          = _get(m_mf)
    out["pathways"]           = _get(not_go)
    return out

def build_one_disease_json(disease_dir: Path, out_json_dir: Path, fdr_q: float = 0.05) -> Optional[Path]:
    disease = disease_dir.name
    # inputs produced by GL
    prerank_sig = disease_dir / "prerank_sig.tsv"
    tf_epi_sig  = disease_dir / "tf_epi_sig.tsv"   # optional

    pw_df = _load_sig_terms(prerank_sig, thr=fdr_q)
    tf_df = _load_sig_terms(tf_epi_sig,  thr=fdr_q)

    buckets = _split_pathways(pw_df)
    tf_terms = _terms_only(tf_df)

    obj = {
        "disease": disease,
        "layers": {
            "pathways":            {"terms": buckets["pathways"]},
            "mechanism":           {"terms": buckets["mechanism"]},            # GO:MF
            "biological_function": {"terms": buckets["biological_function"]},  # GO:BP
            "cell_component":      {"terms": buckets["cell_component"]},       # GO:CC
            "tf":                  {"terms": tf_terms},
            "epigenetic":          {"terms": []},  # not produced in GC flow; keep key for schema stability
            "metabolites":         {"terms": []},  # not produced in GC flow; keep key for schema stability
        },
    }

    out_json_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_json_dir / (disease.replace(" ", "_").lower() + ".json")
    out_path.write_text(json.dumps(obj, ensure_ascii=False, indent=2))
    return out_path

def run_all(base_out: Path, fdr_q: float = 0.05) -> Path:
    """
    base_out points to your --out from mdp-gc (the folder that contains GC_enrich/).
    """
    gc_enrich = base_out / "GC_enrich"
    if not gc_enrich.exists():
        raise FileNotFoundError(f"No GC_enrich found under: {base_out}")

    out_json_dir = base_out / "results" / "all_jsons"
    created: int = 0
    for d in sorted([p for p in gc_enrich.iterdir() if p.is_dir()]):
        p = build_one_disease_json(d, out_json_dir, fdr_q=fdr_q)
        if p is not None:
            created += 1
    if created == 0:
        raise RuntimeError("No disease JSON files were created.")
    return out_json_dir

def main():
    ap = argparse.ArgumentParser(description="Build per-disease JSONs from GC_enrich outputs.")
    ap.add_argument("--out-root", required=True, help="Same --out you used for mdp-gc (contains GC_enrich/).")
    ap.add_argument("--fdr", type=float, default=0.05, help="FDR q cutoff for prerank/tf terms.")
    args = ap.parse_args()
    out_dir = run_all(Path(args.out_root), fdr_q=args.fdr)
    print(f"[GC] JSONs written to: {out_dir}")

if __name__ == "__main__":
    main()
