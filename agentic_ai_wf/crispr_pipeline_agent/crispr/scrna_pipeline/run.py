from __future__ import annotations
from pathlib import Path
import yaml
import anndata as ad
import pandas as pd
import json
import time
import sys
import traceback

from .io import read_gsm_10x_like
from .qc import add_qc_metrics, filter_qc, scrub_doublets
from .integrate import run_none, run_harmony, run_bbknn, run_scvi
from .annotate import run_celltypist, majority_vote_labels, run_scanvi_optional
from .de import deg_disease_vs_control, deg_by_celltype
from .plotting import plot_umaps, plot_markers, plot_composition, plot_paga_dpt
from .utils import safe_mkdir


def _ts():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _log(msg: str):
    print(f"[{_ts()}] {msg}", flush=True)


def run_from_config(config_path: str):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    dataset_dir = cfg["paths"]["dataset_dir"]
    out_root = Path(cfg["paths"]["out_root"]) / cfg["dataset_name"]
    safe_mkdir(out_root)

    gsms = cfg["gsm_selected"]
    run_tag = "_".join(gsms)
    base_out = out_root / f"run_{run_tag}"
    safe_mkdir(base_out)

    _log(f"PIPELINE START :: {cfg['dataset_name']} :: {gsms}")

    # ================= LOAD =================
    adata = ad.concat(
        [read_gsm_10x_like(dataset_dir, g) for g in gsms],
        label="sample_id",
        keys=gsms,
        join="outer",
        fill_value=0,
    )
    adata.var_names_make_unique()
    adata.layers["counts"] = adata.X.copy()

    # ================= QC =================
    adata = add_qc_metrics(adata)
    adata = filter_qc(adata, cfg["qc"])
    adata = scrub_doublets(adata, cfg["qc"])

    # ================= CONDITIONS =================
    if cfg["conditions"]["enabled"]:
        meta = pd.read_csv(cfg["conditions"]["metadata_csv"])
        gsm2grp = dict(zip(meta["GSM"].astype(str), meta["Group"].astype(str)))
        adata.obs["Group"] = adata.obs["sample_id"].map(gsm2grp)

        def map_cond(x):
            if x in cfg["conditions"]["control_groups"]:
                return "control"
            if x in cfg["conditions"]["disease_groups"]:
                return "disease"
            return "unknown"

        adata.obs["condition"] = adata.obs["Group"].map(map_cond)
        adata = adata[adata.obs["condition"].isin(["control", "disease"])]

    methods = cfg["integration"]["methods"]

    for method in methods:
        method_dir = base_out / method
        if (method_dir / "DONE.txt").exists():
            _log(f"SKIP METHOD :: {method} (already completed)")
            continue

        _log(f"METHOD START :: {method}")
        safe_mkdir(method_dir)

        a = adata.copy()

        try:
            if method == "none":
                a = run_none(a, cfg["preprocess"])
            elif method == "harmony":
                a = run_harmony(a, cfg["preprocess"])
            elif method == "bbknn":
                a = run_bbknn(a, cfg["preprocess"])
            elif method == "scvi":
                a = run_scvi(a, cfg["preprocess"])
            else:
                raise ValueError(method)
        except Exception as e:
            _log(f"INTEGRATION FAILED :: {method} :: {e}")
            (method_dir / "FAILED.txt").write_text(traceback.format_exc())
            continue

        # ================= ANNOTATION =================
        labels = []
        try:
            a = run_celltypist(a, cfg["annotation"]["celltypist_model"])
            labels.append("celltypist_label_mv" if "celltypist_label_mv" in a.obs else "celltypist_label")
        except Exception:
            pass

        try:
            if labels:
                a = run_scanvi_optional(a, labels[0], allow_weak=True)
                labels.append("scanvi_label")
        except Exception:
            pass

        if labels:
            a = majority_vote_labels(a, labels)

        # ================= PLOTS =================
        plot_umaps(a, method_dir / "figures", ["sample_id", "leiden", "final_celltype"])
        plot_markers(a, method_dir / "figures", "leiden")
        plot_paga_dpt(a, method_dir / "figures")

        # ================= DEG =================
        if cfg["conditions"]["enabled"]:
            deg_disease_vs_control(
                a,
                "condition",
                "disease",
                "control",
                method_dir / "tables" / "deg_all_cells.csv",
            )
            if "final_celltype" in a.obs:
                deg_by_celltype(
                    a,
                    "final_celltype",
                    "condition",
                    "disease",
                    "control",
                    method_dir / "tables" / "deg_by_celltype",
                )
                plot_composition(
                    a,
                    method_dir / "figures",
                    "final_celltype",
                    "condition",
                )

        a.write(method_dir / "adata.final.h5ad")
        (method_dir / "DONE.txt").write_text("OK\n")
        _log(f"METHOD COMPLETE :: {method}")

    _log(f"PIPELINE COMPLETE :: {base_out}")

