from __future__ import annotations
from pathlib import Path
import time
import json
import scanpy as sc
import pandas as pd


# =============================================================================
# Logging helpers (GEAR-style)
# =============================================================================

def _ts():
    return time.strftime("%Y-%m-%d %H:%M:%S")

def log(stage: str, msg: str):
    print(f"[{_ts()}] [{stage}] {msg}", flush=True)

def log_skip(stage: str, msg: str):
    print(f"[{_ts()}] [SKIP] [{stage}] {msg}", flush=True)


def _write_status(out_dir: Path, payload: dict):
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "STATUS.json").write_text(json.dumps(payload, indent=2))


def _write_qc_csv(adata, out_dir: Path):
    """
    Export per-cell QC metrics if present.
    """
    cols = [
        c for c in [
            "n_genes_by_counts",
            "total_counts",
            "pct_counts_mt",
            "predicted_doublet",
            "doublet_score",
        ]
        if c in adata.obs
    ]

    if not cols:
        return

    df = adata.obs[cols].copy()
    df.insert(0, "cell_barcode", adata.obs_names)

    out_csv = out_dir / "qc_metrics_cells.csv"
    df.to_csv(out_csv, index=False)


# =============================================================================
# QC metrics
# =============================================================================

def add_qc_metrics(adata, out_dir: Path | None = None):
    log("STAGE_QC", "Calculating QC metrics")

    status = {"stage": "qc_metrics"}

    try:
        upper = adata.var_names.str.upper()
        adata.var["mt"] = upper.str.startswith("MT-")
        sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True)

        status["status"] = "ok"
        status["n_cells"] = adata.n_obs
        status["n_genes"] = adata.n_vars

    except Exception as e:
        log_skip("STAGE_QC", f"QC metric calculation failed ({e})")
        status["status"] = "failed"
        status["error"] = str(e)

    if out_dir is not None:
        _write_qc_csv(adata, out_dir)
        _write_status(out_dir, status)

    return adata


def filter_qc(adata, qc_cfg: dict, out_dir: Path | None = None):
    log("STAGE_QC", "Filtering cells / genes")

    status = {"stage": "qc_filtering"}
    n_before = adata.n_obs

    try:
        sc.pp.filter_cells(adata, min_genes=int(qc_cfg["min_genes"]))
        sc.pp.filter_genes(adata, min_cells=int(qc_cfg["min_cells"]))

        adata = adata[adata.obs["pct_counts_mt"] <= float(qc_cfg["max_mt_pct"])].copy()
        adata = adata[adata.obs["n_genes_by_counts"] <= int(qc_cfg["max_genes"])].copy()

        status["status"] = "ok"
        status["n_cells_before"] = n_before
        status["n_cells_after"] = adata.n_obs

    except Exception as e:
        log_skip("STAGE_QC", f"QC filtering failed ({e})")
        status["status"] = "failed"
        status["error"] = str(e)

    if out_dir is not None:
        _write_qc_csv(adata, out_dir)
        _write_status(out_dir, status)

    return adata


def scrub_doublets(adata, qc_cfg: dict, out_dir: Path | None = None):
    log("STAGE_QC", "Doublet detection (Scrublet)")

    status = {"stage": "doublets"}

    if not qc_cfg.get("doublets", {}).get("enabled", True):
        log_skip("STAGE_QC", "Doublet detection disabled")
        status["status"] = "skipped_disabled"
        if out_dir is not None:
            _write_status(out_dir, status)
        return adata

    # Ensure copy to avoid "Received a view" warning from scanpy
    adata = adata.copy()

    try:
        import scanpy.external as sce
        expected = float(qc_cfg["doublets"].get("expected_doublet_rate", 0.06))
        sce.pp.scrublet(adata, expected_doublet_rate=expected)

        n_doublets = int(adata.obs["predicted_doublet"].sum())
        adata = adata[~adata.obs["predicted_doublet"].astype(bool)].copy()

        status["status"] = "ok"
        status["n_doublets_removed"] = n_doublets
        status["n_cells_after"] = adata.n_obs

    except Exception as e:
        log_skip("STAGE_QC", f"Scrublet failed ({e})")
        status["status"] = "failed"
        status["error"] = str(e)

    if out_dir is not None:
        _write_qc_csv(adata, out_dir)
        _write_status(out_dir, status)

    return adata

