from __future__ import annotations
import time
import json
from pathlib import Path
import pandas as pd
import scanpy as sc

# ------------------------------------------------------------------------------
# Logging helpers (GEAR-style)
# ------------------------------------------------------------------------------

def _ts():
    return time.strftime("%Y-%m-%d %H:%M:%S")

def log_step(msg: str):
    print(f"[{_ts()}] [STAGE_ANNOTATION] {msg}", flush=True)

def log_skip(msg: str):
    print(f"[{_ts()}] [SKIP] [STAGE_ANNOTATION] {msg}", flush=True)

def log_info(msg: str):
    print(f"[{_ts()}] [INFO] {msg}", flush=True)


def _write_csv_if_requested(adata, out_dir: Path | None):
    """
    Optional: dump annotation columns to CSV if out_dir provided.
    This NEVER breaks upstream code.
    """
    if out_dir is None:
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    cols = [
        c for c in adata.obs.columns
        if c.startswith(("celltypist", "scanvi", "final_celltype"))
    ]

    if not cols:
        return

    df = adata.obs[cols].copy()
    df.insert(0, "cell_barcode", adata.obs_names)

    out_csv = out_dir / "annotation_labels.csv"
    df.to_csv(out_csv, index=False)
    log_info(f"Annotation CSV written → {out_csv}")


def _write_status(out_dir: Path | None, payload: dict):
    if out_dir is None:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "STATUS.json").write_text(json.dumps(payload, indent=2))


def _as_1d(x, name: str = "value"):
    """
    Convert CellTypist outputs into a 1D array-like suitable for adata.obs[col] assignment.
    Handles: Series, DataFrame (n×1), numpy arrays (n,), (n×1).
    """
    if x is None:
        return None

    if isinstance(x, pd.Series):
        return x.astype(str).to_numpy()

    if isinstance(x, pd.DataFrame):
        if x.shape[1] == 0:
            raise ValueError(f"{name} DataFrame has 0 columns")
        return x.iloc[:, 0].astype(str).to_numpy()

    arr = getattr(x, "to_numpy", lambda: x)()
    try:
        import numpy as np
        arr = np.asarray(arr)
        if arr.ndim == 2 and arr.shape[1] == 1:
            arr = arr[:, 0]
        if arr.ndim != 1:
            raise ValueError(f"{name} has invalid ndim={arr.ndim}")
        return arr
    except Exception:
        return pd.Series(x).to_numpy()


# ------------------------------------------------------------------------------
# CellTypist annotation (SAFE + CANONICAL)
# ------------------------------------------------------------------------------

def run_celltypist(
    adata,
    model_name: str = "Immune_All_Low.pkl",
    force: bool = False,
    out_dir: Path | None = None,
):
    """
    CellTypist runner.
    - Never silently skips
    - Always logs outcome
    - Optionally writes CSV + STATUS.json
    """

    status = {"engine": "CellTypist", "status": None}

    if not force and "celltypist_label" in adata.obs:
        log_skip("CellTypist already present")
        status["status"] = "skipped_existing"
        _write_status(out_dir, status)
        return adata

    if "counts" not in adata.layers:
        status["status"] = "failed_no_counts"
        _write_status(out_dir, status)
        raise RuntimeError("CellTypist requires raw counts in adata.layers['counts']")

    log_step("CellTypist: preparing expression matrix")

    adata.layers["_X_backup_pre_celltypist"] = adata.X.copy()
    adata.X = adata.layers["counts"].copy()

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    log_step(f"CellTypist: running model = {model_name}")

    import celltypist
    from celltypist import models

    models.download_models(force_update=False)

    pred = celltypist.annotate(
        adata,
        model=model_name,
        majority_voting=True,
    )

    labels_1d = _as_1d(pred.predicted_labels, name="predicted_labels")
    if labels_1d is None or len(labels_1d) != adata.n_obs:
        raise RuntimeError("CellTypist label extraction failed")

    adata.obs["celltypist_label"] = labels_1d

    conf = getattr(pred, "confidence", None)
    if conf is not None:
        try:
            conf_arr = _as_1d(conf, name="confidence")
            adata.obs["celltypist_score"] = pd.to_numeric(conf_arr, errors="coerce")
        except Exception:
            pass

    mv = getattr(pred, "majority_voting", None)
    if mv is not None:
        mv_1d = _as_1d(mv, name="majority_voting")
        if mv_1d is not None:
            adata.obs["celltypist_label_mv"] = mv_1d

    adata.X = adata.layers["_X_backup_pre_celltypist"]
    del adata.layers["_X_backup_pre_celltypist"]

    status["status"] = "ok"
    _write_csv_if_requested(adata, out_dir)
    _write_status(out_dir, status)

    log_step("CellTypist: completed successfully")
    return adata


# ------------------------------------------------------------------------------
# Majority voting
# ------------------------------------------------------------------------------

def majority_vote_labels(
    adata,
    label_cols: list[str],
    out_col: str = "final_celltype",
    out_dir: Path | None = None,
):
    log_step(f"Majority voting across labels: {label_cols}")

    votes = adata.obs[label_cols].copy()

    def mv(row):
        vals = [v for v in row.tolist() if pd.notnull(v)]
        return pd.Series(vals).value_counts().index[0] if vals else "unknown"

    adata.obs[out_col] = votes.apply(mv, axis=1)

    _write_csv_if_requested(adata, out_dir)

    log_step(f"Final cell type stored in obs['{out_col}']")
    return adata


# ------------------------------------------------------------------------------
# scANVI (NEVER SILENT)
# ------------------------------------------------------------------------------

def run_scanvi_optional(
    adata,
    label_key: str,
    allow_weak: bool = True,
    out_dir: Path | None = None,
):
    status = {"engine": "scANVI", "status": None}
    log_step("scANVI: initialization")

    try:
        import scvi
    except Exception as e:
        status["status"] = f"skipped_no_scvi ({e})"
        _write_status(out_dir, status)
        raise

    if label_key not in adata.obs:
        if allow_weak and "celltypist_label_mv" in adata.obs:
            label_key = "celltypist_label_mv"
            log_info("scANVI using weak labels from CellTypist majority voting")
        else:
            status["status"] = "skipped_no_labels"
            _write_status(out_dir, status)
            raise RuntimeError("No labels available for scANVI")

    if "counts" not in adata.layers:
        adata.layers["counts"] = adata.X.copy()

    scvi.model.SCANVI.setup_anndata(
        adata,
        layer="counts",
        batch_key="sample_id" if "sample_id" in adata.obs else None,
        labels_key=label_key,
        unlabeled_category="unknown"
        if "unknown" in adata.obs[label_key].unique()
        else None,
    )

    model = scvi.model.SCANVI(
        adata,
        unlabeled_category="unknown"
        if "unknown" in adata.obs[label_key].unique()
        else None,
    )

    log_step("scANVI: training")
    model.train()

    adata.obs["scanvi_label"] = model.predict(adata)
    adata.obsm["X_scANVI"] = model.get_latent_representation()

    status["status"] = "ok"
    _write_csv_if_requested(adata, out_dir)
    _write_status(out_dir, status)

    log_step("scANVI: completed successfully")
    return adata

