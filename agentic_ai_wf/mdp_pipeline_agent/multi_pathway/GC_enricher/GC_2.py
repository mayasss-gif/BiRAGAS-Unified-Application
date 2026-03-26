#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import sys, re, json, math, random
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from math import comb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ================== CONFIG ==================
BASE_DIR = Path("/home/sabahatjamil/testGL_out_fix/Lupus_enrich").expanduser().resolve()

FOLDERS = {
    "pathways":   "mc_prerank",     # main pathways (leading-edge)
    "metabolites":"mc_metabolism",
    "epigenetic": "mc_epigenetic",
    "tf":         "mc_tf",
}
CSV_GLOBS = ("*.csv", "*.tsv", "*.txt")

# ---- Master switch for metabolite gating ----
# True  -> use ALL gates (size/K/OR/Jaccard) + p and q (q uses Q_STAR)
# False -> ONLY use p-value and Jaccard for metabolites (no size/K/OR gates)
MET_USE_STRICT = True

# Global gates (for epi/TF; also used by metabolites when MET_USE_STRICT=True)
MIN_K = 3
MIN_OR = 2.0
MIN_JACCARD = 0.05
Q_STAR = 0.05  # q-value cutoff used when strict mode is ON

# Metabolite relaxed thresholds (used only when MET_USE_STRICT=False)
MET_P_CUTOFF = 0.05
MET_MIN_JACCARD = 0.02

# Set-size gates (apply to epi/TF; and to metabolites only when strict mode ON)
MIN_SET = 10
MAX_SET = 1500

# Optional inputs (if present in BASE_DIR)
HGNC_ALIAS_FILE = "hgnc_alias_map.tsv"   # columns: alias,symbol
UNIVERSE_FILE   = "universe_genes.txt"   # one HGNC symbol per line

# Single JSON output
# OUT_JSON = "pairwise_enrichment.json"  # (unused now)  # CHANGED: we will compute name dynamically

# Quick visuals (optional)
MAKE_PLOTS = True
HEATMAP_PNG = "heatmap_neglog10q.png"
VOLCANO_PNG = "volcano_log2OR.png"
MAX_HEATMAP_COLS = 80

# Permutation check (off by default)
RUN_PERMUTATIONS = False
PERM_N = 1000
# ============================================


NORMALIZATION_RULES: Dict[str, str] = {
    r"^p[\s\-_]*value$": "p_value", r"^p[\s\-_]*val$": "p_value", r"^p$": "p_value",
    r"^(adj|adjusted)[\s\-_]*p[\s\-_]*value$": "q_value", r"^(adjusted)[\s\-_]*p$": "q_value",
    r"^q[\s\-_]*value$": "q_value", r"^fdr(\s*q[\s\-_]*value)?$": "q_value",
    r"^padj$": "q_value", r"^q$": "q_value",
    r"^term$": "term", r"^name$": "term", r"^motif$": "term", r"^pathway$": "term",
    r"^(tf|transcription[\s\-_]*factor)$": "term",
    r"^combined[\s\-_]*score$": "combined_score", r"^overlap$": "overlap",
    r"^genes?$": "genes", r"^gene[\s\-_]*symbols?$": "genes",
    r"^library$": "library", r"^source$": "library", r"^rank$": "rank",
    r"^direction$": "direction", r"^regulation$": "direction", r"^effect$": "direction", r"^(status|state)$":"direction",
    r"^lead[\s\-_]*genes?$": "lead_genes",
    r"^core[\s\-_]*enrichment$": "lead_genes",
    r"^gs$": "term",
}

def normalize_one(col: str) -> str:
    base = re.sub(r"\s+", " ", col.strip().lower()).replace("%", "percent")
    for patt, target in NORMALIZATION_RULES.items():
        if re.match(patt, base):
            return target
    return re.sub(r"[^\w]+", "_", base).strip("_")

def normalize_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str]]:
    mapping = {c: normalize_one(c) for c in df.columns}
    return df.rename(columns=mapping), mapping

def make_unique_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.columns.is_unique:
        return df
    counts: Dict[str, int] = {}
    new_cols = []
    for c in df.columns:
        counts[c] = counts.get(c, 0) + 1
        new_cols.append(f"{c}_{counts[c]}" if counts[c] > 1 else c)
    df.columns = new_cols
    return df

def coerce_numeric(df: pd.DataFrame, colnames: List[str]) -> None:
    for c in colnames:
        if c in df.columns:
            s = df[c].astype(str).str.replace("%", "", regex=False).str.replace(",", "", regex=False)
            df[c] = pd.to_numeric(s, errors="coerce")

def read_table(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, low_memory=False)
    except Exception:
        return pd.read_csv(path, sep="\t", low_memory=False)

# ---------- gene splitters ----------
def split_genes_to_list(s: str) -> List[str]:
    if not isinstance(s, str) or not s.strip():
        return []
    parts = re.split(r"[;,|]\s*|\s+\+\s+|\s{2,}", s)
    if len(parts) == 1:
        parts = [p.strip() for p in re.split(r"[;,|]", s)]
    return [p.strip() for p in parts if p and p.strip()]

def split_genes_semicolon(s: str) -> List[str]:
    if not isinstance(s, str) or not s.strip():
        return []
    return [t.strip() for t in s.split(";") if t.strip()]

# ---------- HGNC standardization ----------
def load_hgnc_alias_map(base: Path) -> Dict[str, str]:
    f = base / HGNC_ALIAS_FILE
    if not f.exists():
        return {}
    df = pd.read_csv(f, sep="\t")
    out: Dict[str, str] = {}
    for _, row in df.iterrows():
        alias = str(row.get("alias", "")).strip().upper()
        symbol = str(row.get("symbol", "")).strip().upper()
        if alias and symbol:
            out[alias] = symbol
    return out

def std_symbol(g: str, alias_map: Dict[str, str]) -> Optional[str]:
    if not g:
        return None
    x = re.sub(r"[^\w\-\.]", "", str(g).strip()).upper()
    if not x:
        return None
    return alias_map.get(x, x)

def standardize_gene_list(genelist: List[str], alias_map: Dict[str, str]) -> List[str]:
    std = [s for s in (std_symbol(g, alias_map) for g in genelist) if s]
    std = [g for g in std if re.match(r"^[A-Z0-9\-\.]{2,}$", g)]
    return sorted(set(std))

# ---------- direction parsing ----------
def parse_direction(val: Optional[str]) -> Optional[str]:
    if not isinstance(val, str):
        return None
    s = val.strip().lower()
    if not s:
        return None
    if any(t in s for t in ["up", "gain", "hyper", "activation", "increase", "induction"]):
        return "up"
    if any(t in s for t in ["down", "loss", "hypo", "repression", "decrease", "reduced"]):
        return "down"
    return None

# ---------- column detection ----------
def detect_term_genes_columns(folder_name: str, df: pd.DataFrame) -> Tuple[Optional[str], Optional[str], str]:
    cols = set(df.columns)

    if folder_name == FOLDERS["pathways"]:
        term_candidates = ["term", "term_2", "gs", "pathway", "name"]
        genes_candidates = ["lead_genes", "core_enrichment", "genes"]
        term_col = next((c for c in term_candidates if c in cols), None)
        if "term" in cols:
            # ensure Series.unique() call works even if name clashes exist
            term_vals = pd.Series(df["term"]).dropna().astype(str)
            if len(term_vals.unique()) <= 1:
                term_col = "term_2" if "term_2" in cols else "term"
        genes_col = next((c for c in genes_candidates if c in cols), None)
        splitter = "semicolon" if genes_col in {"lead_genes", "core_enrichment"} else "generic"
        return term_col, genes_col, splitter

    term_candidates = ["term", "name", "motif", "pathway"]
    genes_candidates = ["genes"]
    term_col = next((c for c in term_candidates if c in cols), None)
    genes_col = next((c for c in genes_candidates if c in cols), None)
    return term_col, genes_col, "generic"

# ---------- Collect gene sets ----------
def collect_entities(base: Path, folder: str, alias_map: Dict[str, str]) -> Dict[str, Dict]:
    fpath = base / folder
    out: Dict[str, Dict] = {}
    if not fpath.exists():
        return out
    files: List[Path] = []
    for patt in CSV_GLOBS:
        files.extend(sorted(fpath.glob(patt)))
    if not files:
        return out

    for p in files:
        try:
            df0 = read_table(p)
        except Exception:
            continue
        df, _ = normalize_columns(df0)
        df = make_unique_columns(df)
        coerce_numeric(df, ["p_value", "q_value"])

        term_col, genes_col, splitter = detect_term_genes_columns(folder, df)
        if term_col is None or genes_col is None:
            continue

        for _, row in df.iterrows():
            term = str(row.get(term_col, "")).strip()
            if not term:
                continue

            raw = row.get(genes_col, "")
            genes_raw = split_genes_semicolon(raw) if splitter == "semicolon" else split_genes_to_list(raw)
            if not genes_raw:
                continue

            genes = set(standardize_gene_list(genes_raw, alias_map))
            if not genes:
                continue

            dir_val = parse_direction(row.get("direction")) if "direction" in df.columns else None

            rec = out.setdefault(term, {"genes": set(), "direction": dir_val})
            rec["genes"].update(genes)

            prior = rec.get("direction")
            if prior != dir_val:
                if prior is None:
                    rec["direction"] = dir_val
                elif dir_val is None:
                    pass
                else:
                    rec["direction"] = None

    return out

# ---------- Universe ----------
def get_universe(base: Path, groups: Dict[str, Dict[str, Dict]], alias_map: Dict[str, str]) -> Set[str]:
    uni_file = base / UNIVERSE_FILE
    if uni_file.exists():
        genes = [line.strip() for line in uni_file.read_text(encoding="utf-8").splitlines() if line.strip()]
        return set(standardize_gene_list(genes, alias_map))
    U: Set[str] = set()
    for cat in groups.values():
        for ent in cat.values():
            U.update(ent["genes"])
    return U

# ---------- Fisher exact + BH-FDR ----------
def odds_ratio(a: int, b: int, k: int, N: int) -> float:
    n11 = k; n10 = a - k; n01 = b - k; n00 = N - a - b + k
    if min(n11, n10, n01, n00) == 0:
        n11 += 0.5; n10 += 0.5; n01 += 0.5; n00 += 0.5
    return (n11 * n00) / (n10 * n01)

def hypergeom_pmf(k: int, K: int, n: int, N: int) -> float:
    if k < 0 or k > K or k > n or n > N or K > N:
        return 0.0
    return comb(K, k) * comb(N - K, n - k) / comb(N, n)

def fishers_exact_two_sided(k: int, a: int, b: int, N: int) -> float:
    lo = max(0, a + b - N); hi = min(a, b)
    p_obs = hypergeom_pmf(k, a, b, N)
    p = 0.0
    for x in range(lo, hi + 1):
        px = hypergeom_pmf(x, a, b, N)
        if px <= p_obs + 1e-12:
            p += px
    return min(1.0, p)

def bh_fdr(pvals: List[float]) -> List[float]:
    m = len(pvals)
    if m == 0: return []
    order = sorted(range(m), key=lambda i: pvals[i])
    q = [0.0] * m; prev = 1.0
    for rank, i in enumerate(order, start=1):
        val = pvals[i] * m / rank
        prev = min(prev, val)
        q[i] = min(prev, 1.0)
    return q

def perm_pvalue_k_at_least(k_obs: int, a: int, b: int, N: int, perms: int = 1000) -> float:
    lo = max(0, a + b - N); hi = min(a, b)
    pmfs = [hypergeom_pmf(x, a, b, N) for x in range(lo, hi+1)]
    exceed = 0
    for _ in range(perms):
        r = random.random(); cum = 0.0; xdraw = hi
        for i, px in enumerate(pmfs):
            cum += px
            if r <= cum:
                xdraw = lo + i; break
        if xdraw >= k_obs:
            exceed += 1
    return (exceed + 1) / (perms + 1)

# ---------- helper: safe disease file name ----------  # CHANGED
def _safe_disease_name_from_base(base: Path) -> str:
    # use last directory name, normalize to snake_case, lowercased
    name = base.name.strip()
    name = re.sub(r"[^\w]+", "_", name).strip("_")
    return name.lower() if name else "disease"

# ---------- MAIN ----------
def run():
    base = BASE_DIR
    if not base.exists():
        sys.exit(1)

    alias_map = load_hgnc_alias_map(base)

    # Collect sets
    pathways    = collect_entities(base, FOLDERS["pathways"],    alias_map)
    metabolites = collect_entities(base, FOLDERS["metabolites"], alias_map)
    epigenetic  = collect_entities(base, FOLDERS["epigenetic"],  alias_map)
    tf          = collect_entities(base, FOLDERS["tf"],          alias_map)

    # Universe
    groups = {"pathways": pathways, "metabolites": metabolites, "epigenetic": epigenetic, "tf": tf}
    U = get_universe(base, groups, alias_map); N = len(U)
    if N == 0:
        sys.exit(1)

    # Clamp sets to U. Size filter depends on strict-mode for metabolites.
    def clamp_only(sets: Dict[str, Dict]) -> Dict[str, Dict]:
        kept = {}
        for name, rec in sets.items():
            g = rec["genes"] & U
            if len(g) > 0:
                kept[name] = {"genes": g, "direction": rec.get("direction")}
        return kept

    def clamp_and_size_filter(sets: Dict[str, Dict]) -> Dict[str, Dict]:
        kept = {}
        for name, rec in sets.items():
            g = rec["genes"] & U
            if len(g) == 0:
                continue
            if len(g) < MIN_SET:
                continue
            if len(g) > MAX_SET:
                continue
            kept[name] = {"genes": g, "direction": rec.get("direction")}
        return kept

    pathways = clamp_only(pathways)  # never size-filter prerank
    if MET_USE_STRICT:
        metabolites = clamp_and_size_filter(metabolites)
    else:
        metabolites = clamp_only(metabolites)
    epigenetic  = clamp_and_size_filter(epigenetic)
    tf          = clamp_and_size_filter(tf)

    # Inverted indices
    def invert_index(sets: Dict[str, Dict]) -> Dict[str, Set[str]]:
        idx: Dict[str, Set[str]] = {}
        for name, rec in sets.items():
            for g in rec["genes"]:
                idx.setdefault(g, set()).add(name)
        return idx

    met_idx = invert_index(metabolites)
    epi_idx = invert_index(epigenetic)
    tf_idx  = invert_index(tf)

    results_per_pathway: Dict[str, Dict[str, List[Dict]]] = {
        pw: {"metabolites": [], "epigenetic": [], "tf": []} for pw in pathways.keys()
    }

    def candidate_entities(A: Set[str], idx: Dict[str, Set[str]]) -> Set[str]:
        cands: Set[str] = set()
        for g in A:
            if g in idx:
                cands |= idx[g]
        return cands

    # ---------- Metabolites evaluation ----------
    def eval_metabolites():
        for pw_name, pw_rec in pathways.items():
            A = pw_rec["genes"]; a = len(A)
            if a == 0: continue
            pw_dir = pw_rec.get("direction")

            for entity in candidate_entities(A, met_idx):
                Brec = metabolites[entity]; B = Brec["genes"]; b = len(B)
                if b == 0: continue
                k = len(A & B)
                if k == 0: continue

                jacc = k / (a + b - k) if (a + b - k) > 0 else 0.0
                or_val = odds_ratio(a, b, k, N)
                pval = fishers_exact_two_sided(k, a, b, N)

                if MET_USE_STRICT:
                    keep = (k >= MIN_K) and (or_val > MIN_OR) and (jacc >= MIN_JACCARD) and (pval < 1.0)
                else:
                    keep = (pval < MET_P_CUTOFF) and (jacc >= MET_MIN_JACCARD)

                if keep:
                    results_per_pathway[pw_name]["metabolites"].append({
                        "entity": entity,
                        "OR": float(round(or_val, 4)),
                        "pval": float(round(pval, 12)),
                        "Jaccard": float(round(jacc, 4)),
                        "k": int(k), "a": int(a), "b": int(b), "N": int(N),
                        "overlap_genes": sorted(A & B),
                        "direction": Brec.get("direction") or pw_dir
                    })

    # ---------- Epi/TF evaluation ----------
    def pass_gates_other(or_val: float, jacc: float, k: int) -> bool:
        return (k >= MIN_K) and (or_val > MIN_OR) and (jacc >= MIN_JACCARD)

    def eval_other(cat_name: str, factors: Dict[str, Dict], idx: Dict[str, Set[str]]):
        for pw_name, pw_rec in pathways.items():
            A = pw_rec["genes"]; a = len(A)
            if a == 0: continue
            pw_dir = pw_rec.get("direction")

            for entity in candidate_entities(A, idx):
                Brec = factors[entity]; B = Brec["genes"]; b = len(B)
                if b == 0: continue
                k = len(A & B)
                if k == 0: continue
                jacc = k / (a + b - k) if (a + b - k) > 0 else 0.0
                or_val = odds_ratio(a, b, k, N)

                if not pass_gates_other(or_val, jacc, k):
                    continue

                pval = fishers_exact_two_sided(k, a, b, N)
                results_per_pathway[pw_name][cat_name].append({
                    "entity": entity,
                    "OR": float(round(or_val, 4)),
                    "pval": float(round(pval, 12)),
                    "Jaccard": float(round(jacc, 4)),
                    "k": int(k), "a": int(a), "b": int(b), "N": int(N),
                    "overlap_genes": sorted(A & B),
                    "direction": Brec.get("direction") or pw_dir
                })

    # Run all
    eval_metabolites()
    eval_other("epigenetic", epigenetic, epi_idx)
    eval_other("tf", tf, tf_idx)

    # ---------- BH-FDR (assign q) ----------
    collected: List[Dict] = []
    for pw, cats in results_per_pathway.items():
        for cat in ["metabolites", "epigenetic", "tf"]:
            collected.extend(cats[cat])
    pvals = [r["pval"] for r in collected]
    qvals = bh_fdr(pvals)
    for r, q in zip(collected, qvals):
        r["qval"] = float(round(q, 12))

    # Optional permutation sanity-check
    if RUN_PERMUTATIONS and len(collected) > 0:
        for r in collected:
            p_perm = perm_pvalue_k_at_least(r["k"], r["a"], r["b"], r["N"], perms=PERM_N)
            r["pval"] = float(round(max(r["pval"], p_perm), 12))
        pvals = [r["pval"] for r in collected]
        qvals = bh_fdr(pvals)
        for r, q in zip(collected, qvals):
            r["qval"] = float(round(q, 12))

    # ---------- STRICT MODE: apply q-value filter to metabolites ----------
    if MET_USE_STRICT:
        for pw in list(results_per_pathway.keys()):
            mets = results_per_pathway[pw]["metabolites"]
            kept = [x for x in mets if x.get("qval", 1.0) < Q_STAR]
            results_per_pathway[pw]["metabolites"] = kept

    # Sorting
    def sort_key_met(x):
        return (-x["OR"], -x["Jaccard"], -x["k"])
    def sort_key_other(x):
        return (0 if x.get("qval", 1.0) < Q_STAR else 1, -x["OR"], -x["Jaccard"], -x["k"])

    for pw, cats in results_per_pathway.items():
        cats["metabolites"].sort(key=sort_key_met)
        cats["epigenetic"].sort(key=sort_key_other)
        cats["tf"].sort(key=sort_key_other)

    # Drop pathways with no kept hits
    kept_results = {
        pw: cats for pw, cats in results_per_pathway.items()
        if any(len(v) > 0 for v in cats.values())
    }

    # ---------- Write JSON as <disease_name>.json ----------  # CHANGED
    disease_file = f"{_safe_disease_name_from_base(base)}.json"
    out_path = base / disease_file
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(kept_results, f, ensure_ascii=False, indent=2)

    # Plots (silent on errors)
    if MAKE_PLOTS and len(kept_results) > 0:
        try:
            

            rows, cols, vals = [], [], {}
            for pw, cats in kept_results.items():
                rows.append(pw)
                for cat in ["metabolites", "epigenetic", "tf"]:
                    for ent in cats[cat]:
                        qv = ent.get("qval", 1.0)
                        col = f"{cat}:{ent['entity']}"
                        cols.append(col)
                        vals[(pw, col)] = max(0.0, -math.log10(max(qv, 1e-300)))
            rows = sorted(set(rows))
            cols = sorted(set(cols))[:MAX_HEATMAP_COLS]

            H = __import__("numpy").zeros((len(rows), len(cols)))
            for i, rname in enumerate(rows):
                for j, cname in enumerate(cols):
                    H[i, j] = vals.get((rname, cname), 0.0)

            # Heatmap
            plt.figure(figsize=(min(16, 2 + 0.12*len(cols)), min(12, 2 + 0.25*len(rows))))
            im = plt.imshow(H, aspect='auto')
            plt.colorbar(im, label='-log10(q)')
            plt.yticks(range(len(rows)), rows, fontsize=6)
            plt.xticks(range(len(cols)), cols, fontsize=6, rotation=90)
            plt.tight_layout()
            plt.savefig(base / HEATMAP_PNG, dpi=250)
            plt.close()

            # Volcano
            X, Y, S = [], [], []
            for pw, cats in kept_results.items():
                for cat in ["metabolites", "epigenetic", "tf"]:
                    for ent in cats[cat]:
                        log2or = math.log2(max(ent["OR"], 1e-12))
                        nlq = max(0.0, -math.log10(max(ent.get("qval", 1.0), 1e-300)))
                        X.append(log2or); Y.append(nlq); S.append(20 + 5*ent["k"])

            plt.figure(figsize=(8, 6))
            plt.scatter(X, Y, s=S, alpha=0.7)
            plt.axvline(1.0, linestyle='--', linewidth=1)
            plt.axhline(-math.log10(Q_STAR), linestyle='--', linewidth=1)
            plt.xlabel("log2(OR)")
            plt.ylabel("-log10(q)")
            plt.title("Volcano of enriched (pathway × factor) associations")
            plt.tight_layout()
            plt.savefig(base / VOLCANO_PNG, dpi=250)
            plt.close()
        except Exception:
            pass

if __name__ == "__main__":
    pd.set_option("display.max_columns", 50)
    run()
