import os
import json
from typing import Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def _load_manifest(step1_dir: str) -> Dict:
    manifest_path = os.path.join(step1_dir, "manifest.json")
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(
            f"Step 1 manifest not found at {manifest_path}. "
            "Run Step 1 (ingestion) before Step 2."
        )
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    print("Loaded manifest from Step 1:")
    print(json.dumps(manifest, indent=2))
    return manifest


def load_layer(step1_dir: str, manifest: Dict, name: str) -> pd.DataFrame:
    """Load samples × features matrix from parquet (Step 1 output)."""
    info = manifest["layers"][name]
    path = os.path.join(step1_dir, info["file"])
    df = pd.read_parquet(path)
    df = df.apply(pd.to_numeric, errors="coerce")
    return df


def zscore_per_feature(df: pd.DataFrame, layer_name: str):
    """Z-score each feature (column): (x - mean) / std; drop zero-variance features."""
    means = df.mean(axis=0)
    stds = df.std(axis=0)

    keep = stds > 0
    dropped = (~keep).sum()
    if dropped > 0:
        print(f"[{layer_name}] Dropping {dropped} zero-variance features before z-scoring.")

    df = df.loc[:, keep]
    means = means[keep]
    stds = stds[keep]

    df_norm = (df - means) / stds
    return df_norm, int(dropped)


def log_transform_if_needed(df: pd.DataFrame, layer_name: str):
    """
    For 'raw_like' layers (primarily metabolomics), apply a log1p-like transform.
    Handles negative values by shifting if necessary.
    """
    vals = df.to_numpy().ravel()
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return df, None

    vmin = np.min(vals)
    shift = 0.0
    if vmin <= 0:
        shift = abs(vmin) + 1e-6
        print(f"[{layer_name}] Detected non-positive values; shifting by {shift:.4g} before log1p.")

    df_shifted = df + shift
    df_log = np.log1p(df_shifted)
    return df_log, shift


def median_impute(df: pd.DataFrame, layer_name: str) -> pd.DataFrame:
    """Median imputation per feature."""
    na_before = df.isna().sum().sum()
    if na_before > 0:
        medians = df.median(axis=0)
        df = df.fillna(medians)
        na_after = df.isna().sum().sum()
        print(f"[{layer_name}] Median-imputed {na_before} missing values (remaining {na_after}).")
    return df


def qc_panel_before_after(df_before: pd.DataFrame, df_after: pd.DataFrame,
                          layer_name: str, outdir: str) -> None:
    """
    Build a 2×2 QC panel:
      (1) Histogram of values before
      (2) Histogram of values after
      (3) Boxplot of sample means before
      (4) Boxplot of sample means after
    """
    os.makedirs(outdir, exist_ok=True)

    vals_before = df_before.to_numpy().ravel()
    vals_before = vals_before[np.isfinite(vals_before)]
    vals_after = df_after.to_numpy().ravel()
    vals_after = vals_after[np.isfinite(vals_after)]

    sample_means_before = df_before.mean(axis=1)
    sample_means_after = df_after.mean(axis=1)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle(f"{layer_name}: Normalization QC", fontsize=14)

    # Colors
    color_vals_before = "#5b8ff9"   # blue
    color_vals_after = "#5ad8a6"    # green
    color_box_before = "#5b8ff9"
    color_box_after = "#9a6df0"     # purple

    # (1) Histogram before
    sns.histplot(vals_before, bins=60, kde=True, ax=axes[0, 0], color=color_vals_before)
    axes[0, 0].set_title("Values (before)")
    axes[0, 0].set_xlabel("Value")
    axes[0, 0].set_ylabel("Count")

    # (2) Histogram after
    sns.histplot(vals_after, bins=60, kde=True, ax=axes[0, 1], color=color_vals_after)
    axes[0, 1].set_title("Values (after)")
    axes[0, 1].set_xlabel("Value")
    axes[0, 1].set_ylabel("Count")

    # (3) Sample means before
    axes[1, 0].boxplot(
        sample_means_before,
        vert=False,
        boxprops=dict(color=color_box_before),
        medianprops=dict(color="black"),
    )
    axes[1, 0].set_title("Sample mean (before)")
    axes[1, 0].set_xlabel("Mean across features")

    # (4) Sample means after
    axes[1, 1].boxplot(
        sample_means_after,
        vert=False,
        boxprops=dict(color=color_box_after),
        medianprops=dict(color="black"),
    )
    axes[1, 1].set_title("Sample mean (after)")
    axes[1, 1].set_xlabel("Mean across features")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    out_path = os.path.join(outdir, f"{layer_name}_normalization_QC_panel.png")
    plt.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"✔ Saved {layer_name} panel → {out_path}")


def run_step2(base_dir: str) -> Dict[str, str]:
    """
    STEP 2 — Adaptive Normalization & Imputation

    Generalized version of your notebook Step 2.

    Parameters
    ----------
    base_dir : str
        Same base directory used in Step 1 (where step_1_ingestion lives).

    Returns
    -------
    dict with:
      - step_dir: path to step_2_preprocessing
      - normalization_summary: path to normalization_summary.json
    """
    step1_dir = os.path.join(base_dir, "step_1_ingestion")
    step2_dir = os.path.join(base_dir, "step_2_preprocessing")
    plots_dir = os.path.join(step2_dir, "plots_norm_compare")

    os.makedirs(step2_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    # Load Step 1 manifest
    manifest = _load_manifest(step1_dir)

    # Main normalization loop
    norm_manifest: Dict[str, Dict] = {
        "base_dir": step2_dir,
        "layers": {},
    }

    for layer_name in manifest["layers"].keys():
        print("\n==============================")
        print(f"Normalizing layer: {layer_name}")
        print("==============================")

        info = manifest["layers"][layer_name]
        guess = info.get("normalization_guess", "unknown")
        print(f"[{layer_name}] Normalization guess from Step 1: {guess}")

        df = load_layer(step1_dir, manifest, layer_name)
        print(f"[{layer_name}] Loaded: {df.shape[0]} samples × {df.shape[1]} features")

        # copy for QC "before"
        df_before = df.copy()

        # 1) Median imputation
        df = median_impute(df, layer_name)

        # 2) Transform based on guess
        applied_shift = None
        applied_log = False

        if guess == "raw_like" or layer_name == "metabolomics":
            df, applied_shift = log_transform_if_needed(df, layer_name)
            applied_log = True
            print(f"[{layer_name}] Applied log1p transform (shift={applied_shift}).")
        else:
            print(f"[{layer_name}] No explicit log transform applied (assumed already log/normalized).")

        # 3) Z-score per feature
        df_norm, dropped = zscore_per_feature(df, layer_name)
        print(f"[{layer_name}] Z-scored features; dropped {dropped} zero-variance columns.")
        print(f"[{layer_name}] Final normalized shape: {df_norm.shape[0]} samples × {df_norm.shape[1]} features")

        # 4) QC panel
        qc_panel_before_after(df_before, df_norm, layer_name, plots_dir)

        # 5) Save normalized matrix
        out_path = os.path.join(step2_dir, f"{layer_name}_normalized.parquet")
        df_norm.to_parquet(out_path)
        print(f"[{layer_name}] Saved normalized matrix → {out_path}")

        # 6) Update norm_manifest
        norm_manifest["layers"][layer_name] = {
            "file": f"{layer_name}_normalized.parquet",
            "n_samples": int(df_norm.shape[0]),
            "n_features": int(df_norm.shape[1]),
            "original_n_features": int(info["n_features"]),
            "normalization_guess": guess,
            "log_transform_applied": applied_log,
            "shift_for_log": float(applied_shift) if applied_shift is not None else None,
            "zero_variance_features_dropped": dropped,
        }

    # Save normalization manifest
    norm_manifest_path = os.path.join(step2_dir, "normalization_summary.json")
    with open(norm_manifest_path, "w", encoding="utf-8") as f:
        json.dump(norm_manifest, f, indent=2)

    print("\n🧾 Normalization summary written to:", norm_manifest_path)
    print(json.dumps(norm_manifest, indent=2))

    print("\n🎉 STEP 2 (Adaptive normalization + imputation) complete.")
    print("QC panels in:", plots_dir)

    return {
        "step_dir": step2_dir,
        "normalization_summary": norm_manifest_path,
    }
