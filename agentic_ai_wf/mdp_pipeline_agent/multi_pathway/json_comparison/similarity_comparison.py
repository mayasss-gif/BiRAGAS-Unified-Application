# similarity_comparison.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd

# Matplotlib is optional and headless-safe
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    MPL = True
except Exception:
    MPL = False

# -------------------- small utils --------------------

def _sanitize_labels(labels: List[object], prefix: str = "unnamed") -> List[str]:
    out: List[str] = []
    for i, v in enumerate(labels):
        try:
            s = "" if v is None else str(v).strip()
            if not s:
                s = f"<{prefix}_{i}>"
            out.append(s)
        except Exception:
            out.append(f"<{prefix}_{i}>")
    return out

def _safe_index(a, default=0):
    try:
        return int(a)
    except Exception:
        return default

def _pair_name(a: str, b: str) -> str:
    """Stable column-safe pair name with real disease labels"""
    a2, b2 = str(a).strip(), str(b).strip()
    if a2 <= b2:
        left, right = a2, b2
    else:
        left, right = b2, a2
    # Excel/CSV-safe
    name = f"{left}__AND__{right}"
    name = name.replace("/", "_").replace("\\", "_")
    return name

# =============================== #
#   Feature matrix & base metrics #
# =============================== #

def _disease_feature_matrix(mat_up: pd.DataFrame,
                            mat_down: pd.DataFrame,
                            mat_any: pd.DataFrame) -> Tuple[np.ndarray, List[str], List[str]]:
    """
    Build a disease-by-feature matrix for global (disease-level) similarity.
    - If directional: rows=diseases, cols = [UP::pw..., DOWN::pw...]
    - Else: rows=diseases, cols = [ANY::pw...]
    """
    try:
        if mat_up is not None and not mat_up.empty and mat_down is not None and not mat_down.empty:
            diseases = _sanitize_labels(list(mat_up.columns), "disease")
            # reindex and align
            mat_up = mat_up.copy()
            mat_down = mat_down.reindex(index=mat_up.index, columns=diseases, fill_value=0)
            mat_up = mat_up.reindex(columns=diseases, fill_value=0)
            M = np.vstack([mat_up.values.astype(float), mat_down.values.astype(float)]).T
            feats = [f"UP::{pw}" for pw in mat_up.index] + [f"DOWN::{pw}" for pw in mat_up.index]
            return M, diseases, feats
        elif mat_any is not None and not mat_any.empty:
            diseases = _sanitize_labels(list(mat_any.columns), "disease")
            mat_any = mat_any.reindex(columns=diseases, fill_value=0)
            M = mat_any.values.astype(float).T
            feats = [f"ANY::{pw}" for pw in mat_any.index]
            return M, diseases, feats
    except Exception:
        pass
    return np.zeros((0, 0), float), [], []


def _cosine_sim(M: np.ndarray) -> np.ndarray:
    """Cosine similarity across rows of M (diseases x features)."""
    try:
        if M.size == 0:
            return M
        norms = np.linalg.norm(M, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        X = M / norms
        return np.clip(X @ X.T, -1.0, 1.0)
    except Exception:
        return np.zeros((0, 0), float)


def _pairwise_jaccard_from_sets(sets: Dict[str, set], diseases: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Given {disease -> set}, compute pairwise intersection counts and Jaccard.
    Returns (counts_df, jaccard_df)
    """
    try:
        diseases = _sanitize_labels(diseases, "disease")
        n = len(diseases)
        counts = np.zeros((n, n), dtype=int)
        jacc   = np.zeros((n, n), dtype=float)
        for i, a in enumerate(diseases):
            A = sets.get(a, set())
            for j, b in enumerate(diseases):
                B = sets.get(b, set())
                inter = len(A & B)
                uni = len(A | B) or 1
                counts[i, j] = inter
                jacc[i, j] = inter / uni if uni else 0.0
        idx = pd.Index(diseases, name="disease")
        return (pd.DataFrame(counts, index=idx, columns=diseases),
                pd.DataFrame(jacc,   index=idx, columns=diseases))
    except Exception:
        return pd.DataFrame(), pd.DataFrame()


def _disease_pathway_sets(mat_any: pd.DataFrame) -> Dict[str, set]:
    """For ANY presence matrix, return {disease -> set(pathways present)}."""
    try:
        if mat_any is None or mat_any.empty:
            return {}
        diseases = _sanitize_labels(list(mat_any.columns), "disease")
        mat_any = mat_any.reindex(columns=diseases, fill_value=0)
        return {d: set(mat_any.index[mat_any[d] == 1].tolist()) for d in diseases}
    except Exception:
        return {}


def _disease_gene_sets(entities_df: pd.DataFrame, diseases: List[str]) -> Dict[str, set]:
    """
    Build disease-level gene sets from entities_df['overlap_genes'] (union across pathways).
    Returns {disease -> set(genes)}.
    """
    out: Dict[str, set] = {d: set() for d in diseases}
    try:
        if entities_df is None or entities_df.empty:
            return out
        for d, sub in entities_df.groupby("disease"):
            key = str(d).strip() if str(d).strip() else "<disease>"
            genes = set()
            if "overlap_genes" in sub.columns:
                for g in sub["overlap_genes"]:
                    try:
                        if isinstance(g, list):
                            genes.update(str(x).upper() for x in g if x is not None)
                        elif isinstance(g, str):
                            genes.update(p.strip().upper() for p in g.split(",") if p.strip())
                    except Exception:
                        continue
            out[key] = genes
    except Exception:
        pass
    return out


def _collapse_entity_sets(entity_sets: dict, diseases: List[str]) -> Dict[str, Dict[str, set]]:
    """
    Collapse pathway->disease->etype->set into disease->etype->set (union across pathways).
    entity_sets: {pathway -> {disease -> {etype -> set(entities)}}}
    """
    diseases = _sanitize_labels(diseases, "disease")
    out = {d: {"metabolites": set(), "epigenetic": set(), "tf": set()} for d in diseases}
    try:
        for _, dd in (entity_sets or {}).items():
            for d, etmap in (dd or {}).items():
                d2 = str(d).strip() if str(d).strip() else "<disease>"
                if d2 not in out or not isinstance(etmap, dict):
                    continue
                for et in ("metabolites", "epigenetic", "tf"):
                    s = etmap.get(et, set())
                    if isinstance(s, set):
                        out[d2][et].update(s)
    except Exception:
        pass
    return out

# ====================== #
#   PCA & cluster plot   #
# ====================== #

def _pca_2d(M: np.ndarray) -> np.ndarray:
    """Simple SVD-based 2D projection for clustering scatter."""
    try:
        if M.size == 0:
            return np.zeros((0, 2))
        X = M - M.mean(axis=0, keepdims=True)
        U, S, _ = np.linalg.svd(X, full_matrices=False)
        if S.size >= 2:
            return U[:, :2] * S[:2]
        return np.hstack([U[:, :1] * S[:1], np.zeros((U.shape[0], 1))])
    except Exception:
        return np.zeros((M.shape[0], 2)) if M.ndim == 2 else np.zeros((0, 2))


def save_cluster_plot(diseases: List[str], coords: np.ndarray, out_png: Path) -> Optional[Path]:
    """Save a PCA-based cluster scatter plot of diseases."""
    if not MPL or not diseases or coords.shape[0] == 0:
        return None
    try:
        plt.figure(figsize=(8, 6))
        plt.scatter(coords[:, 0], coords[:, 1], s=60)
        for i, name in enumerate(diseases):
            plt.text(coords[i, 0], coords[i, 1], f" {name}", fontsize=8, va="center")
        plt.title("Disease Clusters (PCA of pathway embeddings)")
        plt.xlabel("PC1"); plt.ylabel("PC2")
        plt.tight_layout()
        plt.savefig(out_png, dpi=200, bbox_inches="tight")
        plt.close()
        return out_png
    except Exception:
        return None

# ===================================== #
#   Helpers for pathway-level analysis  #
# ===================================== #

def _per_pathway_dir_embeddings(mat_up: pd.DataFrame,
                                mat_down: pd.DataFrame,
                                mat_any: pd.DataFrame) -> Tuple[Dict[str, np.ndarray], List[str]]:
    """
    For each pathway, build a 2-D embedding [UP, DOWN] per disease (non-directional treated as [1,0]).
    Returns ({pathway -> array(n_diseases, 2)}, diseases).
    """
    try:
        if mat_up is not None and not mat_up.empty and mat_down is not None and not mat_down.empty:
            diseases = _sanitize_labels(list(mat_up.columns), "disease")
            mat_up   = mat_up.reindex(columns=diseases, fill_value=0)
            mat_down = mat_down.reindex(index=mat_up.index, columns=diseases, fill_value=0)
            out = {}
            for pw in mat_up.index:
                try:
                    uv = mat_up.loc[pw].values.astype(float)
                    dv = mat_down.loc[pw].values.astype(float)
                    out[pw] = np.stack([uv, dv], axis=1)  # (n_diseases, 2)
                except Exception:
                    continue
            return out, diseases
        elif mat_any is not None and not mat_any.empty:
            diseases = _sanitize_labels(list(mat_any.columns), "disease")
            mat_any = mat_any.reindex(columns=diseases, fill_value=0)
            out = {}
            for pw in mat_any.index:
                try:
                    uv = mat_any.loc[pw].values.astype(float)
                    dv = np.zeros_like(uv)
                    out[pw] = np.stack([uv, dv], axis=1)
                except Exception:
                    continue
            return out, diseases
    except Exception:
        pass
    return {}, []


def _cosine_pair(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1D vectors."""
    try:
        a = np.asarray(a, dtype=float).ravel()
        b = np.asarray(b, dtype=float).ravel()
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        if na == 0 or nb == 0:
            return 0.0
        return float(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))
    except Exception:
        return 0.0


def _per_pathway_entity_sets(entity_sets: dict,
                             diseases: List[str],
                             pathway: str) -> Dict[str, Dict[str, set]]:
    """
    From the nested entity_sets structure, extract for a pathway:
      {disease -> {entity_type -> set(entities)}}
    Guarantees presence of keys and set types.
    """
    diseases = _sanitize_labels(diseases, "disease")
    out = {d: {"metabolites": set(), "epigenetic": set(), "tf": set()} for d in diseases}
    try:
        dd = (entity_sets or {}).get(pathway, {})
        if not isinstance(dd, dict):
            return out
        for d in diseases:
            etmap = dd.get(d, {}) if d in dd else dd.get(str(d), {})
            if not isinstance(etmap, dict):
                continue
            for et in ("metabolites", "epigenetic", "tf"):
                s = etmap.get(et, set())
                if isinstance(s, set):
                    out[d][et].update(s)
    except Exception:
        pass
    return out

# ---------- pathway-level: genes, weighted similarity, and sharing ----------

_W_GENE = 0.50
_W_TF   = 0.20
_W_EPI  = 0.10
_W_MET  = 0.10
_W_DIR  = 0.10

def _per_pathway_gene_sets_from_entities(entities_df: pd.DataFrame,
                                         diseases: List[str],
                                         pathway: str) -> Dict[str, set]:
    """
    Build per-pathway gene sets per disease from entities_df['overlap_genes'].
    """
    diseases = _sanitize_labels(diseases, "disease")
    out = {d: set() for d in diseases}
    try:
        if entities_df is None or entities_df.empty:
            return out
        sub = entities_df[(entities_df["pathway"] == pathway)]
        if sub is None or sub.empty:
            return out
        for d, grp in sub.groupby("disease"):
            key = str(d).strip() if str(d).strip() else "<disease>"
            if key not in out:
                out[key] = set()
            genes = set()
            if "overlap_genes" in grp.columns:
                for g in grp["overlap_genes"]:
                    try:
                        if isinstance(g, list):
                            genes.update(str(x).strip().upper() for x in g if x)
                        elif isinstance(g, str):
                            genes.update(p.strip().upper() for p in g.split(",") if p.strip())
                    except Exception:
                        continue
            out[key].update(genes)
    except Exception:
        pass
    return out


def _safe_jaccard(a: set, b: set) -> float:
    try:
        if not a and not b:
            return 1.0
        if not a or not b:
            return 0.0
        inter = len(a & b)
        uni = len(a | b)
        return float(inter) / float(uni) if uni else 0.0
    except Exception:
        return 0.0


def _shared_sorted_list(a: set, b: set, max_show: int = 200) -> Tuple[int, str]:
    try:
        shared = sorted(a & b)
        n = len(shared)
        if n > max_show:
            text = "; ".join(shared[:max_show]) + f"; (+{n - max_show} more)"
        else:
            text = "; ".join(shared)
        return n, text
    except Exception:
        return 0, ""


def _unordered_pairs(names: List[str]) -> List[Tuple[str, str]]:
    """Combinations without order, using real names."""
    out: List[Tuple[str, str]] = []
    n = len(names)
    for i in range(n):
        for j in range(i + 1, n):
            out.append((names[i], names[j]))
    return out


def _compute_pathway_level_similarity(mat_up: pd.DataFrame,
                                      mat_down: pd.DataFrame,
                                      mat_any: pd.DataFrame,
                                      entity_sets: dict,
                                      entities_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Ordered pairs block kept for backward-compat (now uses disease_1/disease_2).
    """
    try:
        emb_by_pw, diseases = _per_pathway_dir_embeddings(mat_up, mat_down, mat_any)
        if not diseases or not emb_by_pw:
            return pd.DataFrame(), pd.DataFrame()

        diseases = _sanitize_labels(diseases, "disease")
        present_in_series = (mat_any.sum(axis=1)
                             if (mat_any is not None and not mat_any.empty)
                             else pd.Series(dtype=int))

        rows_sim: List[dict] = []
        rows_tf:  List[dict] = []

        for pw, emb in emb_by_pw.items():
            try:
                present_in = int(present_in_series.loc[pw]) if (present_in_series is not None and pw in present_in_series.index) else 0

                per_d_ent = _per_pathway_entity_sets(entity_sets, diseases, pw)
                per_d_genes = _per_pathway_gene_sets_from_entities(entities_df, diseases, pw)

                n = len(diseases)
                for i in range(n):
                    for j in range(n):
                        if i == j:
                            continue
                        d1, d2 = diseases[i], diseases[j]

                        vec_a = emb[i] if i < emb.shape[0] else np.array([0.0, 0.0])
                        vec_b = emb[j] if j < emb.shape[0] else np.array([0.0, 0.0])
                        dir_cos = _cosine_pair(vec_a, vec_b)

                        j_genes = _safe_jaccard(per_d_genes.get(d1, set()), per_d_genes.get(d2, set()))
                        j_tf    = _safe_jaccard(per_d_ent.get(d1, {}).get("tf", set()),
                                                per_d_ent.get(d2, {}).get("tf", set()))
                        j_epi   = _safe_jaccard(per_d_ent.get(d1, {}).get("epigenetic", set()),
                                                per_d_ent.get(d2, {}).get("epigenetic", set()))
                        j_met   = _safe_jaccard(per_d_ent.get(d1, {}).get("metabolites", set()),
                                                per_d_ent.get(d2, {}).get("metabolites", set()))

                        sim_score = (_W_GENE * j_genes +
                                     _W_TF   * j_tf +
                                     _W_EPI  * j_epi +
                                     _W_MET  * j_met +
                                     _W_DIR  * dir_cos)

                        rows_sim.append({
                            "pathway": pw,
                            "disease_1": d1,
                            "disease_2": d2,
                            "dir_cosine": dir_cos,
                            "jaccard_genes": j_genes,
                            "jaccard_tf": j_tf,
                            "jaccard_epigenetic": j_epi,
                            "jaccard_metabolites": j_met,
                            "similarity_score": sim_score,
                            "present_in": present_in,
                        })

                        tf_a = per_d_ent.get(d1, {}).get("tf", set())
                        tf_b = per_d_ent.get(d2, {}).get("tf", set())
                        tf_n, tf_text = _shared_sorted_list(tf_a, tf_b)
                        rows_tf.append({
                            "pathway": pw,
                            "disease_1": d1,
                            "disease_2": d2,
                            "shared_tf_n": tf_n,
                            "shared_tf_list": tf_text,
                        })

            except Exception:
                continue

        sim_df = pd.DataFrame(rows_sim)
        if not sim_df.empty:
            sim_df = sim_df.sort_values(
                ["pathway", "disease_1", "disease_2", "similarity_score"],
                ascending=[True, True, True, False]
            ).reset_index(drop=True)

        tf_df = pd.DataFrame(rows_tf)
        if not tf_df.empty:
            tf_df = tf_df.sort_values(
                ["pathway", "disease_1", "disease_2", "shared_tf_n"],
                ascending=[True, True, True, False]
            ).reset_index(drop=True)

        return sim_df, tf_df

    except Exception:
        return pd.DataFrame(), pd.DataFrame()

# ---------- NEW: all-entities pair tables (unordered) + wide matrices ----------

def _build_pathway_shared_tables_for_type(pathway: str,
                                          per_d_sets: Dict[str, set],
                                          diseases: List[str],
                                          label: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    For a single entity type on one pathway:
      - returns (pairs_df, wide_counts_df, wide_lists_df)
    pairs_df columns: pathway, disease_1, disease_2, shared_n, shared_list
    wide_* columns:  <DiseaseA>__AND__<DiseaseB>
    """
    pairs_rows: List[dict] = []
    pair_cols: Dict[str, int] = {}
    pair_cols_text: Dict[str, str] = {}

    for d1, d2 in _unordered_pairs(diseases):
        a = per_d_sets.get(d1, set())
        b = per_d_sets.get(d2, set())
        n, txt = _shared_sorted_list(a, b, max_show=200)
        pairs_rows.append({
            "pathway": pathway,
            "disease_1": d1,
            "disease_2": d2,
            "shared_n": n,
            "shared_list": txt
        })
        col = _pair_name(d1, d2)
        pair_cols[col] = n
        pair_cols_text[col] = txt

    pairs_df = pd.DataFrame(pairs_rows)

    # wide counts/list are 1-row per pathway
    counts_row = {"pathway": pathway}
    counts_row.update(pair_cols)
    lists_row = {"pathway": pathway}
    lists_row.update(pair_cols_text)

    wide_counts_df = pd.DataFrame([counts_row])
    wide_lists_df  = pd.DataFrame([lists_row])
    return pairs_df, wide_counts_df, wide_lists_df


def _compute_pathway_shared_all_entities(mat_up: pd.DataFrame,
                                         mat_down: pd.DataFrame,
                                         mat_any: pd.DataFrame,
                                         entity_sets: dict,
                                         entities_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Returns a dict of DataFrames:
      - Pathway_Shared_TF_Pairs / _Wide_Counts / _Wide_Lists
      - Pathway_Shared_EPI_Pairs / _Wide_Counts / _Wide_Lists
      - Pathway_Shared_MET_Pairs / _Wide_Counts / _Wide_Lists
      - Pathway_Shared_GENES_Pairs / _Wide_Counts / _Wide_Lists
    """
    out: Dict[str, pd.DataFrame] = {}
    emb_by_pw, diseases = _per_pathway_dir_embeddings(mat_up, mat_down, mat_any)
    if not diseases or not emb_by_pw:
        # return empties with all expected keys
        for tag in ("TF", "EPI", "MET", "GENES"):
            out[f"Pathway_Shared_{tag}_Pairs"] = pd.DataFrame(columns=["pathway","disease_1","disease_2","shared_n","shared_list"])
            out[f"Pathway_Shared_{tag}_Wide_Counts"] = pd.DataFrame()
            out[f"Pathway_Shared_{tag}_Wide_Lists"]  = pd.DataFrame()
        return out

    diseases = _sanitize_labels(diseases, "disease")
    # For per-pathway entity sets
    ETYPES = ("tf", "epigenetic", "metabolites")

    # Accumulators per sheet
    acc_pairs = { "tf": [], "epigenetic": [], "metabolites": [], "genes": [] }
    acc_wide_counts = { "tf": [], "epigenetic": [], "metabolites": [], "genes": [] }
    acc_wide_lists  = { "tf": [], "epigenetic": [], "metabolites": [], "genes": [] }

    for pw in emb_by_pw.keys():
        # entities per pathway
        per_d_ent = _per_pathway_entity_sets(entity_sets, diseases, pw)
        per_d_genes = _per_pathway_gene_sets_from_entities(entities_df, diseases, pw)

        # TF / EPI / MET
        for et in ETYPES:
            per_d_sets = {d: per_d_ent.get(d, {}).get(et, set()) for d in diseases}
            p_df, wc_df, wl_df = _build_pathway_shared_tables_for_type(pw, per_d_sets, diseases, et)
            acc_pairs[et].append(p_df)
            acc_wide_counts[et].append(wc_df)
            acc_wide_lists[et].append(wl_df)

        # GENES
        p_df_g, wc_df_g, wl_df_g = _build_pathway_shared_tables_for_type(pw, per_d_genes, diseases, "genes")
        acc_pairs["genes"].append(p_df_g)
        acc_wide_counts["genes"].append(wc_df_g)
        acc_wide_lists["genes"].append(wl_df_g)

    def _concat_or_empty(frames: List[pd.DataFrame]) -> pd.DataFrame:
        frames = [f for f in frames if f is not None and not f.empty]
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    # stitch per entity type
    out["Pathway_Shared_TF_Pairs"]  = _concat_or_empty(acc_pairs["tf"])
    out["Pathway_Shared_EPI_Pairs"] = _concat_or_empty(acc_pairs["epigenetic"])
    out["Pathway_Shared_MET_Pairs"] = _concat_or_empty(acc_pairs["metabolites"])
    out["Pathway_Shared_GENES_Pairs"] = _concat_or_empty(acc_pairs["genes"])

    out["Pathway_Shared_TF_Wide_Counts"]  = _concat_or_empty(acc_wide_counts["tf"])
    out["Pathway_Shared_EPI_Wide_Counts"] = _concat_or_empty(acc_wide_counts["epigenetic"])
    out["Pathway_Shared_MET_Wide_Counts"] = _concat_or_empty(acc_wide_counts["metabolites"])
    out["Pathway_Shared_GENES_Wide_Counts"] = _concat_or_empty(acc_wide_counts["genes"])

    out["Pathway_Shared_TF_Wide_Lists"]  = _concat_or_empty(acc_wide_lists["tf"])
    out["Pathway_Shared_EPI_Wide_Lists"] = _concat_or_empty(acc_wide_lists["epigenetic"])
    out["Pathway_Shared_MET_Wide_Lists"] = _concat_or_empty(acc_wide_lists["metabolites"])
    out["Pathway_Shared_GENES_Wide_Lists"] = _concat_or_empty(acc_wide_lists["genes"])

    # Sort pairs tables for readability
    for key in list(out.keys()):
        df = out[key]
        if isinstance(df, pd.DataFrame) and not df.empty and "Pairs" in key:
            try:
                out[key] = df.sort_values(["pathway", "disease_1", "disease_2", "shared_n"],
                                          ascending=[True, True, True, False]).reset_index(drop=True)
            except Exception:
                pass

    return out

# =========================== #
#   Public entrypoint (API)   #
# =========================== #

def compute_similarity_block(mat_up: pd.DataFrame,
                             mat_down: pd.DataFrame,
                             mat_any: pd.DataFrame,
                             entities_df: pd.DataFrame,
                             entity_sets: dict,
                             out_dir: Path,
                             prefix: str) -> Tuple[Dict[str, pd.DataFrame], Optional[Path]]:
    """
    Build all similarity/cluster outputs:
      - "Disease_Cosine_Similarity"
      - "Disease_Pathway_Jaccard"
      - "Shared_Genes_Counts" / "Shared_Genes_Jaccard"
      - "Shared_Entities_{et}_Counts" / "Shared_Entities_{et}_Jaccard"
      - "Pathway_Level_Similarity" (ordered pairs; disease_1/disease_2)
      - "Pathway_Level_TF_Shared"  (ordered pairs; disease_1/disease_2)
      - NEW: Pathway_Shared_{TF,EPI,MET,GENES}_{Pairs,Wide_Counts,Wide_Lists}
      - cluster PNG: <prefix>_clusters.png (if matplotlib available)
    """
    sheets: Dict[str, pd.DataFrame] = {}

    # embeddings & cosine (global/disease-level)
    try:
        M, diseases, _ = _disease_feature_matrix(mat_up, mat_down, mat_any)
        cos = _cosine_sim(M) if diseases else np.zeros((0, 0))
        cos_df = pd.DataFrame(cos, index=diseases, columns=diseases)
        sheets["Disease_Cosine_Similarity"] = cos_df
    except Exception:
        sheets["Disease_Cosine_Similarity"] = pd.DataFrame()

    # pathway jaccard (ANY)
    try:
        pw_sets = _disease_pathway_sets(mat_any)
        dlist = _sanitize_labels(diseases if diseases else list(pw_sets.keys()), "disease")
        _, pw_jacc_df = _pairwise_jaccard_from_sets(pw_sets, dlist)
        sheets["Disease_Pathway_Jaccard"] = pw_jacc_df
    except Exception:
        sheets["Disease_Pathway_Jaccard"] = pd.DataFrame()

    # genes (disease-level overlap across all pathways)
    try:
        glist = _sanitize_labels(diseases if diseases else list(pw_sets.keys()), "disease")
        gene_sets = _disease_gene_sets(entities_df, glist)
        g_counts_df, g_jacc_df = _pairwise_jaccard_from_sets(gene_sets, glist)
        sheets["Shared_Genes_Counts"]  = g_counts_df
        sheets["Shared_Genes_Jaccard"] = g_jacc_df
    except Exception:
        sheets["Shared_Genes_Counts"]  = pd.DataFrame()
        sheets["Shared_Genes_Jaccard"] = pd.DataFrame()

    # entities per type (disease-level, collapsed across pathways)
    try:
        clist = _sanitize_labels(diseases if diseases else list(pw_sets.keys()), "disease")
        collapsed = _collapse_entity_sets(entity_sets, clist)
        for et in ("metabolites", "epigenetic", "tf"):
            try:
                et_sets = {d: collapsed.get(d, {}).get(et, set()) for d in clist}
                e_counts, e_jacc = _pairwise_jaccard_from_sets(et_sets, list(et_sets.keys()))
                sheets[f"Shared_Entities_{et}_Counts"]  = e_counts
                sheets[f"Shared_Entities_{et}_Jaccard"] = e_jacc
            except Exception:
                sheets[f"Shared_Entities_{et}_Counts"]  = pd.DataFrame()
                sheets[f"Shared_Entities_{et}_Jaccard"] = pd.DataFrame()
    except Exception:
        for et in ("metabolites", "epigenetic", "tf"):
            sheets[f"Shared_Entities_{et}_Counts"]  = pd.DataFrame()
            sheets[f"Shared_Entities_{et}_Jaccard"] = pd.DataFrame()

    # Pathway-level (ordered A→B kept for compatibility, now using disease_1/2)
    try:
        pw_level_df, tf_shared_df = _compute_pathway_level_similarity(
            mat_up, mat_down, mat_any, entity_sets, entities_df
        )
        sheets["Pathway_Level_Similarity"] = pw_level_df
        sheets["Pathway_Level_TF_Shared"]  = tf_shared_df
    except Exception:
        sheets["Pathway_Level_Similarity"] = pd.DataFrame()
        sheets["Pathway_Level_TF_Shared"]  = pd.DataFrame()

    # NEW: Pathway-level unordered pairs + wide matrices for ALL entity types (incl. genes)
    try:
        extras = _compute_pathway_shared_all_entities(mat_up, mat_down, mat_any, entity_sets, entities_df)
        sheets.update(extras)
    except Exception:
        # ensure keys exist
        for tag in ("TF","EPI","MET","GENES"):
            sheets[f"Pathway_Shared_{tag}_Pairs"] = pd.DataFrame(columns=["pathway","disease_1","disease_2","shared_n","shared_list"])
            sheets[f"Pathway_Shared_{tag}_Wide_Counts"] = pd.DataFrame()
            sheets[f"Pathway_Shared_{tag}_Wide_Lists"]  = pd.DataFrame()

    # clusters (global embedding PCA)
    cluster_png: Optional[Path] = None
    try:
        coords = _pca_2d(M) if (isinstance(M, np.ndarray) and M.size > 0) else np.zeros((0, 2))
        cluster_png = save_cluster_plot(diseases, coords, (out_dir / f"{prefix}_clusters.png")) if (isinstance(coords, np.ndarray) and coords.shape[0] > 0) else None
    except Exception:
        cluster_png = None

    return sheets, cluster_png
