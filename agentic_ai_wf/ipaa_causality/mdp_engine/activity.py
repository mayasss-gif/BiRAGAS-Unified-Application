# mdp_engine/activity.py
from __future__ import annotations

from typing import Dict, List, Optional, Set

import numpy as np
import pandas as pd

from .exceptions import DataError, ValidationError
from .gsea import calculate_enrichment_score
from .graph import SimpleDiGraph
from .topology import pagerank

# Stable alias for type hints (safe even if tools evaluate annotations)
DataFrameT = pd.DataFrame


def _require_df(expr) -> DataFrameT:
    if not isinstance(expr, pd.DataFrame):
        raise ValidationError("expression must be a pandas DataFrame")
    if expr.empty:
        raise DataError("expression DataFrame is empty")
    return expr


def _zscore(x: np.ndarray, axis: int = 0) -> np.ndarray:
    mu = np.mean(x, axis=axis, keepdims=True)
    sd = np.std(x, axis=axis, ddof=0, keepdims=True)
    sd = np.where(sd == 0, 1.0, sd)
    return (x - mu) / sd


def ipaa_activity(
    expression: DataFrameT,
    pathways: Dict[str, Set[str]],
    method: str = "mean",
    standardize_pathways: bool = True,
    min_size: int = 10,
    weight_by_topology: bool = False,
    pathway_graph: Optional[SimpleDiGraph] = None,
    pagerank_damping: float = 0.85,
) -> DataFrameT:
    """
    IPAA/ssGSEA-like activity per sample.

    expression:
      - expected samples x genes (rows=samples, cols=genes).
      - If you pass genes x samples, transpose upstream.

    method:
      - mean: mean expression of pathway genes per sample
      - zscore: z-score of mean per pathway
      - pca1: PC1 score of pathway genes per sample
      - ssgsea: per-sample rank + enrichment score (unweighted hits by default)

    If weight_by_topology=True, gene contributions are weighted by PageRank over pathway_graph subgraph.

    NOTE (bugfix):
      Previously, when method="ssgsea" and weight_by_topology=True, topology weights were computed
      but had no effect because ssGSEA ranked using X[i, :] rather than the weighted pathway slice.
      This fix ONLY affects that specific combination by applying topology weights to the pathway genes
      in the per-sample ranking vector prior to computing enrichment score.
    """
    df = _require_df(expression).copy()
    if not pathways:
        raise DataError("pathways dict is empty")

    method = str(method).lower().strip()
    if method not in {"mean", "zscore", "pca1", "ssgsea"}:
        raise ValidationError("method must be one of: mean, zscore, pca1, ssgsea")

    if weight_by_topology and pathway_graph is None:
        raise ValidationError("pathway_graph is required when weight_by_topology=True")

    # sanitize
    df.columns = [str(c).strip() for c in df.columns]
    df.index = [str(i).strip() for i in df.index]
    df = df.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    genes_all = list(df.columns)
    gene_to_idx = {g: i for i, g in enumerate(genes_all)}
    X = df.to_numpy(dtype=float)  # samples x genes
    samples = df.index.tolist()

    out_cols: List[str] = []
    out_mat: List[np.ndarray] = []

    for pid, gset in pathways.items():
        gset = set(str(g).strip() for g in (gset or set()) if str(g).strip())
        keep = [g for g in gset if g in gene_to_idx]
        if len(keep) < int(min_size):
            continue

        idxs = [gene_to_idx[g] for g in keep]
        sub = X[:, idxs]  # samples x pathway_genes

        # Compute topology weights only once per pathway (reused across samples)
        weights: Optional[np.ndarray] = None
        weights_scaled: Optional[np.ndarray] = None
        if weight_by_topology:
            sg = pathway_graph.subgraph(set(keep))  # type: ignore[union-attr]
            pr = pagerank(sg, damping=float(pagerank_damping))
            weights = np.array([pr.get(g, 0.0) for g in keep], dtype=float)
            if float(weights.sum()) <= 0:
                weights = np.ones_like(weights)
            # keep the original behavior for non-ssgsea branches: normalized weights sum to 1
            weights = weights / float(weights.sum())
            sub = sub * weights.reshape(1, -1)

            # For ssGSEA ranking fix, use a *scale-stable* version of weights
            # so we don't trivially push all pathway genes down due to sum=1 normalization.
            wm = float(np.mean(weights)) if weights.size else 1.0
            if wm <= 0:
                wm = 1.0
            weights_scaled = weights / wm  # mean ~ 1

        if method == "mean":
            score = np.mean(sub, axis=1)
        elif method == "zscore":
            score = _zscore(np.mean(sub, axis=1).reshape(-1, 1), axis=0).reshape(-1)
        elif method == "pca1":
            sub_centered = sub - np.mean(sub, axis=0, keepdims=True)
            try:
                U, S, _Vt = np.linalg.svd(sub_centered, full_matrices=False)
                score = U[:, 0] * S[0]
                if np.corrcoef(score, np.mean(sub, axis=1))[0, 1] < 0:
                    score = -score
            except Exception:
                score = np.mean(sub, axis=1)
        else:
            # ssgsea-like: rank within each sample then ES for pathway gene set
            # Bugfix: when topology weighting is enabled, apply weights to pathway genes in the per-sample vector.
            score = np.zeros(len(samples), dtype=float)
            keep_set = set(keep)

            for i in range(len(samples)):
                vals = X[i, :].copy()

                if weight_by_topology and weights_scaled is not None:
                    # apply weights only to the pathway genes (same indices as idxs)
                    # This changes ranking, which is the only way weights can influence ES when weight=0.
                    vals[idxs] = vals[idxs] * weights_scaled

                order = np.argsort(vals)[::-1]
                ranked = [(genes_all[j], float(vals[j])) for j in order]
                es, _lead, _prof = calculate_enrichment_score(ranked, keep_set, weight=0.0)
                score[i] = float(es)

        out_cols.append(str(pid))
        out_mat.append(score)

    if not out_cols:
        raise DataError("No pathways passed filtering (min_size too high or gene symbols mismatched).")

    M = np.vstack(out_mat).T  # samples x pathways
    if standardize_pathways:
        M = _zscore(M, axis=0)

    return pd.DataFrame(M, index=samples, columns=out_cols)
