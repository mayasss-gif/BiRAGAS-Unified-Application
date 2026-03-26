from __future__ import annotations
from pathlib import Path
import time
import json
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
import pandas as pd

MIN_CELLS_FOR_MARKERS = 10


# =============================================================================
# Logging helpers (GEAR-style)
# =============================================================================

def _ts():
    return time.strftime("%Y-%m-%d %H:%M:%S")

def log(stage: str, msg: str):
    print(f"[{_ts()}] [{stage}] {msg}", flush=True)

def log_skip(stage: str, msg: str):
    print(f"[{_ts()}] [SKIP] [{stage}] {msg}", flush=True)


# =============================================================================
# Utilities
# =============================================================================

def _save(figdir: Path, name: str):
    figdir.mkdir(parents=True, exist_ok=True)
    plt.savefig(figdir / name, dpi=300, bbox_inches="tight")
    plt.close()


def _write_status(figdir: Path, payload: dict):
    figdir.mkdir(parents=True, exist_ok=True)
    (figdir / "STATUS.json").write_text(json.dumps(payload, indent=2))


def _write_markers_csv(adata, group: str, outdir: Path):
    """
    Export rank_genes_groups results as CSV.
    """
    try:
        df = sc.get.rank_genes_groups_df(adata, group=None)
    except Exception:
        return

    outdir.mkdir(parents=True, exist_ok=True)
    out_csv = outdir / f"markers_{group}.csv"
    df.to_csv(out_csv, index=False)


# =============================================================================
# UMAP plots
# =============================================================================

def plot_umaps(adata, figdir: Path, color_cols: list[str]):
    log("STAGE_PLOTTING", "UMAP plots")

    status = {"stage": "umap", "plots": []}

    if "X_umap" not in adata.obsm:
        log_skip("STAGE_PLOTTING", "UMAP embedding not found")
        status["status"] = "skipped_no_umap"
        _write_status(figdir, status)
        return

    for col in color_cols:
        if col not in adata.obs:
            continue
        sc.pl.umap(adata, color=col, show=False)
        fname = f"umap_{col}.png"
        _save(figdir, fname)
        status["plots"].append(fname)

    status["status"] = "ok"
    _write_status(figdir, status)


# =============================================================================
# Marker genes (CLUSTER + CELLTYPE)
# =============================================================================

def plot_markers(
    adata,
    figdir: Path,
    groupby: str = "leiden",
    also_by_celltype: bool = True,
    celltype_col: str = "final_celltype",
):
    """
    Marker genes:
      - always per cluster
      - optionally per cell type
    """

    log("STAGE_MARKERS", f"Marker genes (groupby={groupby})")
    status = {"stage": "markers", "groupby": groupby, "celltype": None}

    if groupby not in adata.obs:
        log_skip("STAGE_MARKERS", f"{groupby} not found in obs")
        status["status"] = "skipped_no_groupby"
        _write_status(figdir, status)
        return

    try:
        valid_groups = [
            g for g in adata.obs[groupby].unique()
            if (adata.obs[groupby] == g).sum() >= MIN_CELLS_FOR_MARKERS
        ]
        if len(valid_groups) < 2:
            log_skip("STAGE_MARKERS", f"cluster markers skipped (< 2 groups with >={MIN_CELLS_FOR_MARKERS} cells)")
        else:
            # Defragment to reduce Scanpy rank_genes_groups PerformanceWarning
            _a = adata.copy()
            sc.tl.rank_genes_groups(
                _a,
                groupby=groupby,
                groups=valid_groups,
                method="wilcoxon",
                pts=True,
            )
            sc.pl.rank_genes_groups(_a, n_genes=20, show=False)
            _save(figdir, f"markers_{groupby}.png")
            _write_markers_csv(_a, groupby, figdir / "tables")
    except Exception as e:
        log_skip("STAGE_MARKERS", f"cluster markers failed ({e})")

    # -------------------------------
    # Markers per cell type (OPTIONAL)
    # -------------------------------
    if also_by_celltype and celltype_col in adata.obs:
        log("STAGE_MARKERS", f"Marker genes (by cell type: {celltype_col})")
        status["celltype"] = celltype_col

        try:
            valid_ct = [
                g for g in adata.obs[celltype_col].unique()
                if (adata.obs[celltype_col] == g).sum() >= MIN_CELLS_FOR_MARKERS
            ]
            if len(valid_ct) < 2:
                log_skip("STAGE_MARKERS", f"celltype markers skipped (< 2 groups with >={MIN_CELLS_FOR_MARKERS} cells)")
            else:
                _a = adata.copy()
                sc.tl.rank_genes_groups(
                    _a,
                    groupby=celltype_col,
                    groups=valid_ct,
                    method="wilcoxon",
                    pts=True,
                )
                sc.pl.rank_genes_groups(_a, n_genes=20, show=False)
                _save(figdir, f"markers_{celltype_col}.png")
                _write_markers_csv(_a, celltype_col, figdir / "tables")
        except Exception as e:
            log_skip("STAGE_MARKERS", f"celltype markers failed ({e})")

    status["status"] = "ok"
    _write_status(figdir, status)


# =============================================================================
# Cell-type composition
# =============================================================================

def plot_composition(
    adata,
    figdir: Path,
    celltype_col: str,
    condition_col: str,
):
    log("STAGE_COMPOSITION", "Cell-type composition")

    status = {"stage": "composition"}

    if celltype_col not in adata.obs or condition_col not in adata.obs:
        log_skip("STAGE_COMPOSITION", "required columns missing")
        status["status"] = "skipped_missing_columns"
        _write_status(figdir, status)
        return

    ct = (
        adata.obs.groupby([condition_col, celltype_col])
        .size()
        .unstack(fill_value=0)
    )
    ct = ct.div(ct.sum(axis=1), axis=0)

    ct.plot(kind="bar", stacked=True, figsize=(10, 5))
    plt.ylabel("Fraction of cells")
    plt.title("Cell-type composition by condition")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    _save(figdir, "composition_celltype_by_condition.png")

    ct.reset_index().to_csv(
        figdir / "tables" / "composition_celltype_by_condition.csv",
        index=False,
    )

    status["status"] = "ok"
    _write_status(figdir, status)


# =============================================================================
# Trajectory: PAGA + DPT (ALWAYS ATTEMPTED)
# =============================================================================

def plot_paga_dpt(
    adata,
    figdir: Path,
    cluster_key: str = "leiden",
):
    log("STAGE_TRAJECTORY", "PAGA + DPT")
    status = {"stage": "trajectory", "paga": False, "dpt": False}

    if cluster_key not in adata.obs:
        log_skip("STAGE_TRAJECTORY", f"{cluster_key} not found")
        status["status"] = "skipped_no_clusters"
        _write_status(figdir, status)
        return

    n_clusters = adata.obs[cluster_key].nunique()
    if n_clusters < 2:
        log_skip("STAGE_TRAJECTORY", f"PAGA requires >=2 clusters, got {n_clusters}")
        status["status"] = "skipped_single_cluster"
        _write_status(figdir, status)
        return

    # PAGA often fails with single-sample (one batch) due to graph structure
    if "sample_id" in adata.obs and adata.obs["sample_id"].nunique() < 2:
        log_skip("STAGE_TRAJECTORY", "PAGA skipped for single-sample data (requires multiple batches)")
        status["status"] = "skipped_single_sample"
        _write_status(figdir, status)
        return

    if "neighbors" not in adata.uns:
        log("STAGE_TRAJECTORY", "computing neighbors")
        sc.pp.neighbors(adata)

    if "connectivities" not in adata.obsp:
        log_skip("STAGE_TRAJECTORY", "neighbors graph (connectivities) missing")
        status["status"] = "skipped_no_connectivities"
        _write_status(figdir, status)
        return

    # -------------------------
    # PAGA
    # -------------------------
    try:
        sc.tl.paga(adata, groups=cluster_key)
        sc.pl.paga(adata, show=False)
        _save(figdir, "paga_graph.png")
        status["paga"] = True
    except Exception as e:
        log_skip("STAGE_TRAJECTORY", f"PAGA failed ({e})")

    # -------------------------
    # Diffusion map
    # -------------------------
    try:
        sc.tl.diffmap(adata)
    except Exception as e:
        log_skip("STAGE_TRAJECTORY", f"diffmap failed ({e})")
        status["status"] = "partial_paga_only"
        _write_status(figdir, status)
        return

    # -------------------------
    # DPT (requires root cell)
    # -------------------------
    try:
        if "iroot" not in adata.uns or adata.uns.get("iroot") is None:
            # Use first cell of largest cluster as root
            counts = adata.obs[cluster_key].value_counts()
            if len(counts) > 0:
                largest = counts.index[0]
                idx = np.flatnonzero(adata.obs[cluster_key] == largest)
                if len(idx) > 0:
                    adata.uns["iroot"] = int(idx[0])
        sc.tl.dpt(adata)
        if "dpt_pseudotime" in adata.obs:
            sc.pl.umap(
                adata,
                color="dpt_pseudotime",
                color_map="viridis",
                show=False,
            )
            _save(figdir, "umap_dpt_pseudotime.png")

            adata.obs[["dpt_pseudotime"]].to_csv(
                figdir / "tables" / "dpt_pseudotime.csv"
            )
            status["dpt"] = True
    except Exception as e:
        log_skip("STAGE_TRAJECTORY", f"DPT failed ({e})")

    status["status"] = "ok"
    _write_status(figdir, status)

