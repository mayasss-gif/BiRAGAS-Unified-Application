#!/usr/bin/env python3
"""
Stage-5 Export model-ready dataset (robust).

Inputs:
  processed_stage3/stage3_merged.h5ad

Outputs:
  processed_stage5/
    stage5_cells_metadata.tsv
    stage5_X_pca.npy
    stage5_y_perturbation.npy
    stage5_label_map.tsv
    stage5_summary.tsv

Run:
python stage5_export_ml_dataset.py \
  --input_h5ad processed_stage3/stage3_merged.h5ad \
  --out_dir processed_stage5 \
  --min_cells_per_pert 10 \
  --n_pcs 50
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp


def info(msg):
    print(f"[INFO] {msg}", flush=True)


def warn(msg):
    print(f"[WARN] {msg}", flush=True)


def sanitize_X(adata: sc.AnnData, clip_max: float = 50.0) -> None:
    """
    Clean NaN/inf and optionally clip huge values.
    Works for sparse and dense.
    """
    X = adata.X

    if sp.issparse(X):
        # Replace inf/nan in sparse data array
        data = X.data
        bad = ~np.isfinite(data)
        if bad.any():
            warn(f"Found {bad.sum()} non-finite values in sparse X.data -> setting to 0")
            data[bad] = 0.0
        if clip_max is not None:
            data[:] = np.clip(data, -clip_max, clip_max)
        adata.X = X  # in-place update
    else:
        bad = ~np.isfinite(X)
        if bad.any():
            warn(f"Found {bad.sum()} non-finite values in dense X -> setting to 0")
            X = X.copy()
            X[bad] = 0.0
        if clip_max is not None:
            X = np.clip(X, -clip_max, clip_max)
        adata.X = X


def run_hvg_with_fallback(adata: sc.AnnData, n_top: int = 3000):
    """
    Robust HVG selection.
    seurat_v3 can crash on non-integer log data; fallback safely.
    """
    flavors = ["seurat_v3", "seurat", "cell_ranger"]

    for f in flavors:
        try:
            info(f"HVG trying flavor='{f}' ...")
            sc.pp.highly_variable_genes(adata, n_top_genes=n_top, flavor=f)
            info(f"HVG OK flavor='{f}'")
            return f
        except Exception as e:
            warn(f"HVG flavor='{f}' failed: {type(e).__name__}: {e}")

    raise RuntimeError("All HVG flavors failed. Data may be corrupted or contain extreme values.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_h5ad", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--min_cells_per_pert", type=int, default=10)
    ap.add_argument("--n_pcs", type=int, default=50)
    ap.add_argument("--n_hvg", type=int, default=3000)
    ap.add_argument("--clip_max", type=float, default=50.0)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    info(f"Reading merged h5ad: {args.input_h5ad}")
    adata = sc.read_h5ad(args.input_h5ad)

    # Always make unique observation names
    adata.obs_names_make_unique()

    # Checks
    needed = ["condition_class", "perturbation_id"]
    for c in needed:
        if c not in adata.obs.columns:
            raise RuntimeError(f"Missing required obs column: {c}")

    # Keep only control + perturbed_single
    use = adata[adata.obs["condition_class"].isin(["control", "perturbed_single"])].copy()
    info(f"Subset ctrl+single: {use.n_obs} cells")

    # Filter perts by min size
    counts = use.obs["perturbation_id"].astype(str).value_counts()
    keep = counts[counts >= args.min_cells_per_pert].index.tolist()

    # Ensure control is included if present
    if "control" in counts.index and "control" not in keep:
        keep.append("control")

    use = use[use.obs["perturbation_id"].astype(str).isin(keep)].copy()
    info(f"Kept perts >= {args.min_cells_per_pert}: {len(keep)}")
    info(f"Final cells: {use.n_obs}")

    # Work on .X (merged object likely already normalized/log)
    info("Sanitizing X (NaN/inf removal + clipping)")
    sanitize_X(use, clip_max=args.clip_max)

    # HVG + PCA
    info("Selecting HVGs")
    hvg_flavor = run_hvg_with_fallback(use, n_top=args.n_hvg)

    use = use[:, use.var["highly_variable"]].copy()

    info("Scaling + PCA")
    sc.pp.scale(use, max_value=10)
    sc.tl.pca(use, n_comps=args.n_pcs)

    X_pca = use.obsm["X_pca"].astype(np.float32)

    # Encode labels
    perturbations = use.obs["perturbation_id"].astype(str).values
    uniq = np.unique(perturbations)
    label_map = {p: i for i, p in enumerate(uniq)}
    y = np.array([label_map[p] for p in perturbations], dtype=np.int32)

    # Save outputs
    np.save(out_dir / "stage5_X_pca.npy", X_pca)
    np.save(out_dir / "stage5_y_perturbation.npy", y)

    label_df = pd.DataFrame(
        {"label_id": list(label_map.values()), "perturbation_id": list(label_map.keys())}
    ).sort_values("label_id")
    label_df.to_csv(out_dir / "stage5_label_map.tsv", sep="\t", index=False)

    meta = use.obs[["perturbation_id", "condition_class"]].copy()
    meta["y_label"] = y
    meta.to_csv(out_dir / "stage5_cells_metadata.tsv", sep="\t", index=True)

    summary = (
        meta.groupby("perturbation_id")
        .agg(n_cells=("perturbation_id", "size"))
        .reset_index()
        .sort_values("n_cells", ascending=False)
    )
    summary.to_csv(out_dir / "stage5_summary.tsv", sep="\t", index=False)

    # Log
    info(f"[OK] HVG flavor used: {hvg_flavor}")
    info(f"[OK] X -> {out_dir/'stage5_X_pca.npy'} shape={X_pca.shape}")
    info(f"[OK] y -> {out_dir/'stage5_y_perturbation.npy'} shape={y.shape} classes={len(uniq)}")
    info(f"[OK] label map -> {out_dir/'stage5_label_map.tsv'}")
    info(f"[OK] summary -> {out_dir/'stage5_summary.tsv'}")
    info("[DONE] Stage-5 export complete")


if __name__ == "__main__":
    main()

