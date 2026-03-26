#!/usr/bin/env python3
"""
Stage-10: Bayesian Network / DAG learning (FAST on CPU, FULL on GPU optional)

Key improvements:
- Auto hardware detection (CPU/GPU)
- Auto gene-program scoring (no hardcoded genes)
- CPU-safe mode avoids combinatorial explosion (top-K Z + row subsample)
- GPU mode can run a larger / fuller search (if you enable --full_bn)
- Constrained structure to enforce causality: Z -> (PCs/program scores), no PC->Z, no Z->Z
- Layered DAG visualization

Outputs:
  processed_stage10/
    tables/stage10_data_used.tsv
    tables/stage10_edges.tsv
    tables/stage10_nodes.tsv
    figures/stage10_network.png
"""
from __future__ import annotations

from __future__ import annotations
import argparse
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import anndata as ad
import matplotlib.pyplot as plt
import networkx as nx

from utils_gene_programs import auto_gene_programs, score_programs_scanpy


def log(msg: str) -> None:
    print(msg, flush=True)


def detect_gpu() -> bool:
    # lightweight detection
    # 1) env override
    if os.environ.get("FORCE_CPU", "").strip() == "1":
        return False
    if os.environ.get("FORCE_GPU", "").strip() == "1":
        return True

    # 2) torch
    try:
        import torch  # type: ignore
        return bool(torch.cuda.is_available())
    except Exception:
        pass

    # 3) cupy
    try:
        import cupy  # type: ignore
        _ = cupy.zeros((1,))
        return True
    except Exception:
        return False


def import_pgmpy():
    from pgmpy.estimators import HillClimbSearch
    try:
        from pgmpy.estimators import ExpertKnowledge
    except Exception:
        ExpertKnowledge = None

    try:
        from pgmpy.estimators import BicScore as Score
    except Exception:
        from pgmpy.estimators import BIC as Score

    return HillClimbSearch, ExpertKnowledge, Score


def detect_prediction_cols(obs: pd.DataFrame) -> Tuple[str, str | None]:
    label_candidates = ["pred_label_stage7", "pred_class_stage7", "pred_label", "pred_class"]
    conf_candidates = ["pred_confidence_stage7", "pred_conf_stage7", "pred_confidence", "pred_conf"]
    label_col = next((c for c in label_candidates if c in obs.columns), None)
    if label_col is None:
        raise ValueError(f"Missing prediction label column. Tried: {label_candidates}")
    conf_col = next((c for c in conf_candidates if c in obs.columns), None)
    return label_col, conf_col


def sanitize_numeric(df: pd.DataFrame) -> pd.DataFrame:
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    return df


def build_matrix(
    adata: ad.AnnData,
    label_col: str,
    conf_col: str | None,
    min_conf: float,
    min_cells_per_pert: int,
    include_pcs: int,
    program_prefix: str,
) -> pd.DataFrame:
    """
    Produces matrix with:
      - Z_* one-hot (instrument / predicted perturbation)
      - PC0..PCk from X_pca
      - optional program scores PROG_* (auto discovered)
    """

    # filter confidence
    if conf_col is not None:
        conf = pd.to_numeric(adata.obs[conf_col], errors="coerce")
        keep = conf >= min_conf
        log(f"[INFO] Confidence filter >= {min_conf}: keep {int(keep.sum())}/{adata.n_obs}")
        adata = adata[keep].copy()

    # collapse rare perts
    pert = adata.obs[label_col].astype(str)
    vc = pert.value_counts()
    keep_levels = vc[vc >= min_cells_per_pert].index.tolist()
    pert = pert.where(pert.isin(keep_levels), "__OTHER__")
    adata.obs["Z_pred"] = pert
    log(f"[INFO] Perturbation levels kept: {adata.obs['Z_pred'].nunique()}")

    # base df
    df = pd.DataFrame(index=adata.obs_names)

    # PCs
    if include_pcs > 0:
        if "X_pca" not in adata.obsm:
            raise ValueError("X_pca not found in adata.obsm. Ensure Stage3 computed PCA.")
        Xp = adata.obsm["X_pca"][:, :include_pcs]
        for i in range(Xp.shape[1]):
            df[f"PC{i}"] = Xp[:, i]

    # auto programs + scoring
    programs = auto_gene_programs(adata, max_programs=3, genes_per_program=50, min_present=10)
    if programs:
        scored_cols, skipped = score_programs_scanpy(adata, programs, prefix=program_prefix, min_present=10)
        if scored_cols:
            for c in scored_cols:
                df[c] = pd.to_numeric(adata.obs[c], errors="coerce")
        if skipped:
            log(f"[WARN] Skipped program scoring: {skipped}")
    else:
        log("[WARN] No gene programs discovered -> continuing without program scores")

    # Z one-hot
    Z = pd.get_dummies(adata.obs["Z_pred"].astype(str), prefix="Z", drop_first=True)

    df_all = pd.concat([Z, df], axis=1)
    df_all = sanitize_numeric(df_all)
    return df_all


def reduce_for_speed(
    df: pd.DataFrame,
    top_z: int,
    max_rows: int,
) -> pd.DataFrame:
    z_cols = [c for c in df.columns if c.startswith("Z_")]
    other_cols = [c for c in df.columns if not c.startswith("Z_")]

    if z_cols and top_z > 0 and len(z_cols) > top_z:
        # choose top instruments by frequency (sum of one-hot)
        top = df[z_cols].sum().sort_values(ascending=False).head(top_z).index.tolist()
        log(f"[INFO] Z variables: {len(z_cols)} -> {len(top)} (top_z={top_z})")
        df = df[top + other_cols]

    if max_rows > 0 and df.shape[0] > max_rows:
        log(f"[INFO] Subsampling rows: {df.shape[0]} -> {max_rows}")
        df = df.sample(n=max_rows, random_state=42)

    return df


def build_constraints(df: pd.DataFrame) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    """
    Enforce:
      - No edges into Z_*  (i.e., PC->Z forbidden)
      - No Z->Z edges
    Allow:
      - Z -> PCs / programs
      - PC <-> PC allowed (optional), program <-> PC allowed
    """
    z_nodes = [c for c in df.columns if c.startswith("Z_")]
    non_z = [c for c in df.columns if not c.startswith("Z_")]

    forbidden: List[Tuple[str, str]] = []

    # forbid any -> Z
    for a in non_z:
        for z in z_nodes:
            forbidden.append((a, z))

    # forbid Z -> Z
    for z1 in z_nodes:
        for z2 in z_nodes:
            if z1 != z2:
                forbidden.append((z1, z2))

    required: List[Tuple[str, str]] = []
    return forbidden, required


def learn_bn_pgmpy(
    df: pd.DataFrame,
    expert_constraints: bool,
    max_iter: int,
    max_indegree: int,
) -> List[Tuple[str, str]]:
    HillClimbSearch, ExpertKnowledge, Score = import_pgmpy()

    forbidden, required = build_constraints(df)
    ek = None

    if expert_constraints and ExpertKnowledge is not None:
        ek = ExpertKnowledge(forbidden_edges=forbidden, required_edges=required)
    elif expert_constraints and ExpertKnowledge is None:
        log("[WARN] ExpertKnowledge not available in this pgmpy version -> running unconstrained")

    log(f"[INFO] Learning BN via HillClimb + BIC (max_iter={max_iter}, max_indegree={max_indegree})")
    est = HillClimbSearch(df)
    model = est.estimate(
        scoring_method=Score(df),
        max_iter=max_iter,
        max_indegree=max_indegree,
        expert_knowledge=ek,
        show_progress=False,
    )
    return list(model.edges())


def plot_layered_dag(edges: List[Tuple[str, str]], out_png: Path) -> None:
    G = nx.DiGraph()
    G.add_edges_from(edges)

    z_nodes = [n for n in G.nodes if str(n).startswith("Z_")]
    other_nodes = [n for n in G.nodes if n not in z_nodes]

    # layered positions
    pos = {}
    # left: Z
    for i, n in enumerate(sorted(z_nodes)):
        pos[n] = (0.0, -float(i))

    # right: others
    for i, n in enumerate(sorted(other_nodes)):
        pos[n] = (1.6, -float(i))

    plt.figure(figsize=(14, 10))
    nx.draw_networkx_nodes(G, pos, node_size=900)
    nx.draw_networkx_edges(G, pos, arrows=True, width=1.2, arrowsize=14)
    nx.draw_networkx_labels(G, pos, font_size=7)
    plt.title("Stage-10 DAG (Z -> latent transcriptomic state)")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_h5ad", required=True)
    ap.add_argument("--out_dir", required=True)

    ap.add_argument("--min_conf", type=float, default=0.40)
    ap.add_argument("--min_cells_per_pert", type=int, default=50)

    ap.add_argument("--include_pcs", type=int, default=5)

    # speed controls
    ap.add_argument("--top_z_cpu", type=int, default=25, help="Top Z dummies to keep in CPU mode")
    ap.add_argument("--max_rows_cpu", type=int, default=20000, help="Row cap in CPU mode")

    ap.add_argument("--top_z_gpu", type=int, default=80, help="Top Z dummies to keep in GPU mode")
    ap.add_argument("--max_rows_gpu", type=int, default=80000, help="Row cap in GPU mode")

    ap.add_argument("--full_bn", action="store_true",
                    help="If set: allow more Z and more rows (best used on GPU).")

    ap.add_argument("--max_iter_cpu", type=int, default=200)
    ap.add_argument("--max_iter_gpu", type=int, default=600)

    ap.add_argument("--max_indegree", type=int, default=3)
    ap.add_argument("--no_constraints", action="store_true")

    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    (out_dir / "tables").mkdir(parents=True, exist_ok=True)
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)

    gpu = detect_gpu()
    mode = "GPU" if gpu else "CPU"
    log(f"[INFO] Hardware mode: {mode}")

    log(f"[INFO] Reading h5ad: {args.input_h5ad}")
    adata = ad.read_h5ad(args.input_h5ad)
    adata.obs_names_make_unique()

    label_col, conf_col = detect_prediction_cols(adata.obs)
    log(f"[INFO] Using label={label_col}, conf={conf_col}")

    df = build_matrix(
        adata=adata,
        label_col=label_col,
        conf_col=conf_col,
        min_conf=args.min_conf,
        min_cells_per_pert=args.min_cells_per_pert,
        include_pcs=args.include_pcs,
        program_prefix="PROG_",
    )

    # save full (pre-reduction) data-used snapshot
    df_path = out_dir / "tables" / "stage10_data_used.tsv"
    df.to_csv(df_path, sep="\t")
    log(f"[OK] data used -> {df_path} shape={df.shape}")

    # decide speed plan
    if gpu and args.full_bn:
        top_z = args.top_z_gpu
        max_rows = args.max_rows_gpu
        max_iter = args.max_iter_gpu
        full_flag = True
    else:
        top_z = args.top_z_cpu
        max_rows = args.max_rows_cpu
        max_iter = args.max_iter_cpu
        full_flag = False

    log(f"[INFO] Learning Bayesian Network (mode={mode}, full_bn={full_flag})")

    df_fit = reduce_for_speed(df, top_z=top_z, max_rows=max_rows)

    edges = learn_bn_pgmpy(
        df_fit,
        expert_constraints=(not args.no_constraints),
        max_iter=max_iter,
        max_indegree=args.max_indegree,
    )

    edges_df = pd.DataFrame(edges, columns=["source", "target"])
    edges_df.to_csv(out_dir / "tables" / "stage10_edges.tsv", sep="\t", index=False)

    nodes = sorted(set(edges_df["source"]) | set(edges_df["target"]))
    pd.DataFrame({"node": nodes}).to_csv(out_dir / "tables" / "stage10_nodes.tsv", sep="\t", index=False)

    fig_path = out_dir / "figures" / "stage10_network.png"
    plot_layered_dag(edges, fig_path)

    log("[DONE] Stage-10 complete")


if __name__ == "__main__":
    main()
