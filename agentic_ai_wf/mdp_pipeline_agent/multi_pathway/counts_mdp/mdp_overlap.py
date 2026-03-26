# mdp_overlap.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple
import json
import re
from scipy.stats import fisher_exact
from math import comb
import numpy as np
import pandas as pd

from .mdp_config import CONFIG, out_root
from .mdp_logging import info, warn, trace
from .mdp_io import load_table_auto

# thresholds (same as original overlap script)
REQUIRE_JACCARD_MIN = 0.10
OR_MIN = 1.5
FDR_MAX = 0.05
OVERLAP_MIN = 5


# ---------- small helpers ----------

def _split_genes_string(s: str) -> list[str]:
    if not isinstance(s, str):
        return []
    s = s.strip()
    if not s:
        return []
    parts = re.split(r"[;,|+\s]+", s)
    return [p.strip().upper() for p in parts if p.strip()]


def _collect_pathway_gene_sets(cohort_dir: Path) -> dict[str, list[str]]:
    """
    Try to recover pathway -> gene list from:
      1) GSEA table (gsea_prerank.tsv / gsea_ORA_fallback.tsv) using leading edge
      2) If no leading edge, fall back to core_enrich_up/down.csv (Genes column)
    """
    def _from_gsea(tsv: Path) -> dict[str, list[str]]:
        df = load_table_auto(tsv)
        if df.empty:
            return {}
        cols_low = {c.lower(): c for c in df.columns}
        term_col = cols_low.get("term") or cols_low.get("name") or cols_low.get("pathway")
        if term_col is None and "Term" in df.columns:
            term_col = "Term"
        if term_col is None:
            return {}
        # leading-edge column
        ledge_col = None
        for k in ["lead_genes", "ledge_genes", "lead_genes_list"]:
            if k in cols_low:
                ledge_col = cols_low[k]
                break
        if ledge_col is None:
            # no leading edge in this table
            return {}
        out: dict[str, list[str]] = {}
        for _, row in df.iterrows():
            term = row.get(term_col)
            if not isinstance(term, str):
                continue
            genes_raw = row.get(ledge_col)
            genes = _split_genes_string(genes_raw)
            if not genes:
                continue
            # normalize term (drop prefix "LIB:" if present)
            t = term.split(":", 1)[-1].strip()
            out[t] = genes
        return out

    # 1) Prefer GSEA tables
    for fname in ("gsea_prerank.tsv", "gsea_ORA_fallback.tsv"):
        p = cohort_dir / fname
        if p.exists():
            out = _from_gsea(p)
            if out:
                info(f"[{cohort_dir.name}] Pathways from {fname}: {len(out)} terms")
                return out
            else:
                warn(
                    f"[{cohort_dir.name}] {fname} had no leading-edge / gene list; "
                    f"columns: {list(load_table_auto(p).columns)}"
                )

    # 2) Fallback: core ORA CSVs (Genes column)
    for fname in ("core_enrich_up.csv", "core_enrich_down.csv"):
        p = cohort_dir / fname
        if not p.exists():
            continue
        df = load_table_auto(p)
        if df.empty:
            continue
        cols_low = {c.lower(): c for c in df.columns}
        term_col = cols_low.get("term") or ("Term" if "Term" in df.columns else None)
        genes_col = cols_low.get("genes")
        if term_col is None or genes_col is None:
            warn(f"[{cohort_dir.name}] {fname}: no (term, Genes) columns; skipping.")
            continue
        out: dict[str, list[str]] = {}
        for _, row in df.iterrows():
            term = row.get(term_col)
            if not isinstance(term, str):
                continue
            genes = _split_genes_string(row.get(genes_col))
            if not genes:
                continue
            t = term.split(":", 1)[-1].strip()
            out.setdefault(t, [])
            out[t] = sorted(set(out[t]).union(genes))
        if out:
            info(f"[{cohort_dir.name}] Pathways from {fname}: {len(out)} terms")
            return out
    warn(f"[{cohort_dir.name}] No pathway gene sets found.")
    return {}


def _read_entity_table(path: Path, filter_q: bool = True) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = load_table_auto(path)
    if df.empty:
        return df
    if not filter_q:
        return df
    cols_low = {c.lower(): c for c in df.columns}
    q_col = None
    for k in [
        "qval", "adjusted_p-value", "fdr", "padj",
        "adj_p", "p.adjust", "p.adj", "qvalue", "q-value",
    ]:
        if k in cols_low:
            q_col = cols_low[k]
            break
    if q_col is None:
        return df
    df[q_col] = pd.to_numeric(df[q_col], errors="coerce")
    return df[df[q_col] <= FDR_MAX].copy()


def _genes_from_rows(df: pd.DataFrame) -> list[tuple[str, list[str]]]:
    if df is None or df.empty:
        return []
    cols_low = {c.lower(): c for c in df.columns}
    term_col = cols_low.get("term") or ("Term" if "Term" in df.columns else None)
    genes_col = cols_low.get("genes")
    if term_col is None or genes_col is None:
        return []
    out: list[tuple[str, list[str]]] = []
    for _, row in df.iterrows():
        term = row.get(term_col)
        if not isinstance(term, str):
            continue
        genes = _split_genes_string(row.get(genes_col))
        if not genes:
            continue
        t = term.split(":", 1)[-1].strip()
        out.append((t, genes))
    return out


def _universe_N_from_degs(cohort_dir: Path) -> int:
    p = cohort_dir / "degs_from_counts.csv"
    if not p.exists():
        warn(f"[{cohort_dir.name}] No degs_from_counts.csv; using default N=20000.")
        return 20000
    df = load_table_auto(p)
    if df.empty:
        return 20000
    cols_low = {c.lower(): c for c in df.columns}
    gene_col = None
    for k in ["gene", "genes", "symbol", "hgnc_symbol", "ensembl", "ensembl_id", "id"]:
        if k in cols_low:
            gene_col = cols_low[k]
            break
    if gene_col is None:
        gene_col = df.columns[0]
    genes = (
        df[gene_col]
        .astype(str)
        .str.upper()
        .str.strip()
        .replace({"": np.nan})
        .dropna()
        .unique()
    )
    N = int(len(genes))
    if N < 1000:
        warn(f"[{cohort_dir.name}] gene universe unusually small (N={N}); using fallback=20000.")
        return 20000
    return N


def _fisher_or_p(k: int, a: int, b: int, N: int) -> tuple[float, float]:
    """
    a = |pathway|
    b = |entity|
    k = |intersection|
    N = universe size
    """
    try:
        
        table = np.array(
            [[k, a - k],
             [b - k, N - a - b + k]],
            dtype=float,
        )
        or_val, p = fisher_exact(table, alternative="greater")
        return float(or_val), float(p)
    except Exception:
        # hypergeometric tail fallback
        def pmf(x: int) -> float:
            return comb(a, x) * comb(N - a, b - x) / comb(N, b)
        p = sum(pmf(x) for x in range(k, min(a, b) + 1))
        or_val = (k * (N - a - b + k)) / max((a - k) * (b - k), 1e-9)
        return float(or_val), float(p)


def _bh_fdr(pvals: np.ndarray) -> np.ndarray:
    m = len(pvals)
    order = np.argsort(pvals)
    ranks = np.empty(m, dtype=float)
    ranks[order] = np.arange(1, m + 1)
    q = pvals * m / np.clip(ranks, 1, None)
    q[order[::-1]] = np.minimum.accumulate(q[order[::-1]])
    return np.clip(q, 0, 1)


def _link_one_side(
    pathways: dict[str, list[str]],
    entities_rows: list[tuple[str, list[str]]],
    N: int,
) -> list[dict]:
    """
    pathways: term -> genes
    entities_rows: list[(entity_name, genes)]
    """
    if not pathways or not entities_rows:
        return []
    rows: list[dict] = []
    for term, p_genes in pathways.items():
        P = set(p_genes)
        a = len(P)
        if a < 2:
            continue
        for ent, e_genes in entities_rows:
            E = set(e_genes)
            b = len(E)
            if b < 2:
                continue
            inter = P & E
            k = len(inter)
            if k < OVERLAP_MIN:
                continue
            or_val, p = _fisher_or_p(k, a, b, N)
            jacc = k / float(len(P | E))
            rows.append(
                dict(
                    pathway=term,
                    entity=ent,
                    k=k,
                    a=a,
                    b=b,
                    N=N,
                    OR=or_val,
                    pval=p,
                    Jaccard=jacc,
                    overlap_genes=sorted(inter),
                )
            )
    if not rows:
        return []
    df = pd.DataFrame(rows)
    df["pval"] = pd.to_numeric(df["pval"], errors="coerce").fillna(1.0)
    df["qval"] = _bh_fdr(df["pval"].to_numpy())
    df = df[
        (df["k"] >= OVERLAP_MIN)
        & (df["OR"] > OR_MIN)
        & (df["qval"] <= FDR_MAX)
        & (df["Jaccard"] >= REQUIRE_JACCARD_MIN)
    ].copy()
    df = df.sort_values(["pathway", "qval", "OR", "k"], ascending=[True, True, False, False])
    return df.to_dict(orient="records")


# ---------- main public function ----------
def build_overlap_jsons(root: Path | None = None) -> None:
    """
    For each cohort directory under OUT_ROOT:
      - Collect pathway gene sets
      - For each category (metabolites, epigenetic, TF) and direction (UP, DOWN),
        find significant overlaps.

    Writes:
      1) Per-cohort JSON:
           <OUT_ROOT>/<COHORT>/overlap/pathway_entity_overlap.json
      2) Global JSONs (one per cohort) collected into:
           <OUT_ROOT>/results/all_jsons/<COHORT>_pathway_entity_overlap.json
    """
    root = root or out_root()
    if not root.exists():
        warn(f"[overlap] OUT_ROOT does not exist: {root}")
        return

    # single folder collecting all JSONs across diseases
    all_jsons_root = root / "results" / "all_jsons"
    all_jsons_root.mkdir(parents=True, exist_ok=True)

    # config for categories: filenames and whether to q-filter
    categories = {
        "metabolites": {
            "files": ("metabolite_enrich_up.csv", "metabolite_enrich_down.csv"),
            "filter_q": False,
        },
        "epigenetic": {
            "files": ("epigenetic_enrich_up.csv", "epigenetic_enrich_down.csv"),
            "filter_q": True,
        },
        "tf": {
            "files": ("tf_enrich_up.csv", "tf_enrich_down.csv"),
            "filter_q": True,
        },
    }

    # treat each subdir under OUT_ROOT as a cohort dir (skip baseline/comparison)
    for cdir in sorted(p for p in root.iterdir() if p.is_dir()):
        if cdir.name in {"baseline_consensus", "comparison"}:
            continue

        info(f"[overlap] Cohort: {cdir.name}")
        pathways = _collect_pathway_gene_sets(cdir)
        if not pathways:
            warn(f"[{cdir.name}] No pathways; skipping overlap.")
            continue
        N = _universe_N_from_degs(cdir)

        # nested dict: pathway -> UP/DOWN -> category -> list[hit]
        result: Dict[str, Dict[str, Dict[str, list]]] = {}

        for cat, cfg in categories.items():
            up_file, down_file = cfg["files"]
            filter_q = cfg["filter_q"]

            df_up = _read_entity_table(cdir / up_file, filter_q=filter_q)
            up_rows = _genes_from_rows(df_up)
            up_hits = _link_one_side(pathways, up_rows, N)

            df_dn = _read_entity_table(cdir / down_file, filter_q=filter_q)
            dn_rows = _genes_from_rows(df_dn)
            dn_hits = _link_one_side(pathways, dn_rows, N)

            for direction, hits in (("UP", up_hits), ("DOWN", dn_hits)):
                if not hits:
                    continue
                for h in hits:
                    term = h["pathway"]
                    result.setdefault(term, {"UP": {}, "DOWN": {}})
                    bucket = result[term][direction].setdefault(cat, [])
                    pack = dict(
                        entity=h["entity"],
                        OR=round(float(h["OR"]), 4),
                        pval=round(float(h["pval"]), 6),
                        qval=round(float(h["qval"]), 6),
                        Jaccard=round(float(h["Jaccard"]), 4),
                        k=int(h["k"]),
                        a=int(h["a"]),
                        b=int(h["b"]),
                        N=int(h["N"]),
                        overlap_genes=h["overlap_genes"],
                    )
                    bucket.append(pack)

        if not result:
            info(f"[{cdir.name}] No pathway–entity hits after filtering.")
            continue

        # 1) per-cohort JSON
        overlap_dir = cdir / "overlap"
        overlap_dir.mkdir(parents=True, exist_ok=True)
        cohort_json = overlap_dir / "pathway_entity_overlap.json"
        with open(cohort_json, "w") as f:
            json.dump(result, f, indent=2)

        # 2) copy into global all_jsons folder, one file per cohort
        global_json = all_jsons_root / f"{cdir.name}_pathway_entity_overlap.json"
        with open(global_json, "w") as f:
            json.dump(result, f, indent=2)

        info(f"[{cdir.name}] Wrote overlap JSON: {cohort_json}")
        info(f"[{cdir.name}] Wrote global JSON: {global_json}")

if __name__ == "__main__":
    build_overlap_jsons()
