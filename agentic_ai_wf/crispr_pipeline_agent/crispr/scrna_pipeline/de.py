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


# =============================================================================
# DEG: disease vs control (GLOBAL)
# =============================================================================

def deg_disease_vs_control(
    adata,
    condition_col: str,
    disease_value: str,
    control_value: str,
    out_csv: str | Path,
):
    """
    Differential expression:
    Disease vs Control (ALWAYS disease - control)

    Guarantees:
      - CSV always attempted
      - STATUS.json always written
      - Never silent
    """

    out_csv = Path(out_csv)
    out_dir = out_csv.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    status = {
        "stage": "DE_global",
        "comparison": f"{disease_value}_vs_{control_value}",
    }

    log("STAGE_DE", f"Running global DE: {disease_value} vs {control_value}")

    if condition_col not in adata.obs:
        log_skip("STAGE_DE", f"{condition_col} not found in obs")
        status["status"] = "skipped_no_condition_col"
        _write_status(out_dir, status)
        return

    if (
        disease_value not in adata.obs[condition_col].values
        or control_value not in adata.obs[condition_col].values
    ):
        log_skip("STAGE_DE", "One or both conditions missing")
        status["status"] = "skipped_missing_groups"
        _write_status(out_dir, status)
        return

    try:
        sc.tl.rank_genes_groups(
            adata,
            groupby=condition_col,
            groups=[disease_value],
            reference=control_value,
            method="wilcoxon",
            pts=True,
        )

        df = sc.get.rank_genes_groups_df(
            adata,
            group=disease_value,
        )

        df.insert(0, "comparison", f"{disease_value}_vs_{control_value}")
        df.to_csv(out_csv, index=False)

        status["status"] = "ok"
        status["n_genes"] = df.shape[0]

    except Exception as e:
        log_skip("STAGE_DE", f"Global DE failed ({e})")
        status["status"] = "failed"
        status["error"] = str(e)

    _write_status(out_dir, status)


# =============================================================================
# DEG: per cell type
# =============================================================================

def deg_by_celltype(
    adata,
    celltype_col: str,
    condition_col: str,
    disease_value: str,
    control_value: str,
    out_dir: str | Path,
):
    """
    Differential expression per cell type:
    Disease vs Control within each cell type

    Guarantees:
      - One CSV per cell type (if computable)
      - STATUS.json summary
      - Never crashes pipeline
    """

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    status = {
        "stage": "DE_by_celltype",
        "comparison": f"{disease_value}_vs_{control_value}",
        "celltypes": {},
    }

    log("STAGE_DE", "Running DE per cell type")

    if celltype_col not in adata.obs:
        log_skip("STAGE_DE", f"{celltype_col} not found in obs")
        status["status"] = "skipped_no_celltype_col"
        _write_status(out_dir, status)
        return

    for ct in sorted(adata.obs[celltype_col].dropna().unique()):
        ct_key = ct.replace(" ", "_")
        status["celltypes"][ct_key] = {}

        sub = adata[adata.obs[celltype_col] == ct].copy()

        if (
            disease_value not in sub.obs[condition_col].values
            or control_value not in sub.obs[condition_col].values
        ):
            log_skip("STAGE_DE", f"{ct}: missing one condition")
            status["celltypes"][ct_key]["status"] = "skipped_missing_groups"
            continue

        try:
            sc.tl.rank_genes_groups(
                sub,
                groupby=condition_col,
                groups=[disease_value],
                reference=control_value,
                method="wilcoxon",
                pts=True,
            )

            df = sc.get.rank_genes_groups_df(
                sub,
                group=disease_value,
            )

            df.insert(0, "celltype", ct)
            df.insert(1, "comparison", f"{disease_value}_vs_{control_value}")

            out_csv = out_dir / f"deg_{ct_key}.csv"
            df.to_csv(out_csv, index=False)

            status["celltypes"][ct_key]["status"] = "ok"
            status["celltypes"][ct_key]["n_genes"] = df.shape[0]

        except Exception as e:
            log_skip("STAGE_DE", f"{ct}: DE failed ({e})")
            status["celltypes"][ct_key]["status"] = "failed"
            status["celltypes"][ct_key]["error"] = str(e)

    status["status"] = "ok"
    _write_status(out_dir, status)

