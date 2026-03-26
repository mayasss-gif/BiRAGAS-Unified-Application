import os
import json
from typing import Dict, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from .metadata_analyzer import detect_label_column_smart


def read_feature_table(path: Optional[str], layer_name: str, feature_col: str = "feature") -> Optional[pd.DataFrame]:
    """
    Read a feature x samples CSV and return a samples x features DataFrame.
    Assumes first col named 'feature'; no heavy QC here, just basic cleanup.

    Directly adapted from the original notebook helper.
    """
    if path is None or not os.path.exists(path):
        print(f"⚠️  {layer_name}: path not found or None, skipping: {path}")
        return None

    df = pd.read_csv(path)
    if feature_col not in df.columns:
        raise ValueError(
            f"[{layer_name}] Expected column '{feature_col}' in {path}, got: "
            f"{df.columns[:5].tolist()}"
        )

    # Set index to feature, drop completely empty columns
    df = df.set_index(feature_col)

    # Try to convert all to numeric (non-convertible become NaN)
    df = df.apply(pd.to_numeric, errors="coerce")

    # Drop all-NaN features
    before = df.shape[0]
    df = df.loc[df.notna().any(axis=1)]
    after = df.shape[0]
    if before != after:
        print(f"[{layer_name}] Dropped {before - after} all-NaN features.")

    # Transpose to samples x features
    df = df.T
    df.index = df.index.astype(str)
    df.columns = df.columns.astype(str)

    print(f"[{layer_name}] shape (samples × features): {df.shape}")
    return df


def guess_normalization(df: pd.DataFrame, layer_name: str) -> str:
    """
    Very simple heuristic: just to store in manifest.
    We won't act on this in Step 1; Step 2 will decide what to do.
    """
    vals = df.to_numpy().ravel()
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return "unknown"

    vmin, vmax = np.percentile(vals, [1, 99])
    std = np.std(vals)

    if layer_name == "metabolomics":
        # typically raw-ish intensities, often >1000
        if vmax > 1000:
            return "raw_like"
        else:
            return "normalized_like"
    else:
        # gene / protein / CNV / methylation-type data
        if -8 <= vmin <= -0.1 and 0.1 <= vmax <= 8 and std < 3.5:
            return "log_scaled_or_zscore"
        elif vmax > 100:
            return "raw_like"
        else:
            return "normalized_like"


def basic_qc_plots(df: Optional[pd.DataFrame], layer_name: str, outdir: str) -> None:
    """
    Basic distribution & sample intensity QC plots per layer.
    Nothing fancy yet; main goal is to make sure ingestion looks sane.
    """
    if df is None or df.empty:
        return

    os.makedirs(outdir, exist_ok=True)

    vals = df.to_numpy().ravel()
    vals = vals[np.isfinite(vals)]

    # 1) Value distribution
    plt.figure(figsize=(6, 4))
    sns.histplot(vals, bins=60, kde=True)
    plt.title(f"{layer_name}: Value distribution")
    plt.xlabel("Value")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{layer_name}_value_hist.png"), dpi=300)
    plt.close()

    # 2) Sample-wise total intensity / mean
    sample_means = df.mean(axis=1)
    plt.figure(figsize=(7, 3))
    sns.boxplot(x=sample_means)
    plt.title(f"{layer_name}: Sample mean distribution")
    plt.xlabel("Mean across features")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{layer_name}_sample_mean_box.png"), dpi=300)
    plt.close()

    # 3) Simple heatmap of a small random subset (for sanity)
    n_samples = min(20, df.shape[0])
    n_feats = min(20, df.shape[1])
    df_sub = df.iloc[:n_samples, :n_feats]
    plt.figure(figsize=(7, 6))
    sns.heatmap(df_sub, cmap="vlag", center=0)
    plt.title(f"{layer_name}: Heatmap (subset {n_samples}×{n_feats})")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{layer_name}_heatmap_subset.png"), dpi=300)
    plt.close()


def run_step1(
    base_dir: str,
    layer_paths: Optional[Dict[str, str]] = None,
    layers: Optional[Dict[str, str]] = None,  # backwards-compatible alias
    genomics_path: Optional[str] = None,
    transcriptomics_path: Optional[str] = None,
    epigenomics_path: Optional[str] = None,
    proteomics_path: Optional[str] = None,
    metabolomics_path: Optional[str] = None,
    metadata_path: Optional[str] = None,
    label_column: Optional[str] = None,
) -> Dict[str, str]:
    """
    Step 1 — Multi-Omics Ingestion (generalized version of your notebook STEP 1).

    Parameters
    ----------
    base_dir : str
        Root output folder for the whole pipeline.
    layer_paths : dict[str, str] or config-like object, optional
        Mapping {layer_name -> CSV path} or an object with attributes
        genomics/transcriptomics/epigenomics/proteomics/metabolomics.
        Preferred way (used by CLI).
    layers : dict[str, str], optional
        Backwards-compatible alias for `layer_paths`.
    genomics_path, transcriptomics_path, epigenomics_path, proteomics_path, metabolomics_path : str or None
        CSV paths for each omics layer (legacy style).
    metadata_path : str or None
        Optional metadata CSV. If None or missing, all samples are labeled 'Diseased'.
    label_column : str, optional
        Name of the column in metadata CSV that contains sample labels/classes.
        If provided, this column will be used instead of auto-detection.

    Returns
    -------
    dict with:
      - step_dir: path to step_1_ingestion directory
      - manifest: path to manifest.json
      - metadata: path to metadata.csv
    """
    STEP = os.path.join(base_dir, "step_1_ingestion")
    PLOT_DIR = os.path.join(STEP, "plots")

    os.makedirs(STEP, exist_ok=True)
    os.makedirs(PLOT_DIR, exist_ok=True)

    # ---------------- Normalize layer config ----------------
    # Prefer `layer_paths`; fall back to `layers`; if neither is given,
    # we use the legacy per-layer *_path arguments.
    if layer_paths is None and layers is not None:
        layer_paths = layers

    layers_raw: Dict[str, pd.DataFrame] = {}

    if layer_paths is not None:
        # layer_paths may be:
        #  - a dict, OR
        #  - a config object with attributes genomics/transcriptomics/...
        if hasattr(layer_paths, "items"):
            items_iter = layer_paths.items()
        else:
            # Build a dict from known attribute names
            tmp = {}
            for lname in ["genomics", "transcriptomics", "epigenomics", "proteomics", "metabolomics"]:
                if hasattr(layer_paths, lname):
                    path = getattr(layer_paths, lname)
                    if path is not None:
                        tmp[lname] = path
            items_iter = tmp.items()

        for lname, path in items_iter:
            layers_raw[lname] = read_feature_table(path, lname)

    else:
        # Legacy style: fixed argument names
        layers_raw["genomics"] = read_feature_table(genomics_path, "genomics")
        layers_raw["transcriptomics"] = read_feature_table(transcriptomics_path, "transcriptomics")
        layers_raw["epigenomics"] = read_feature_table(epigenomics_path, "epigenomics")
        layers_raw["proteomics"] = read_feature_table(proteomics_path, "proteomics")
        layers_raw["metabolomics"] = read_feature_table(metabolomics_path, "metabolomics")

    # Drop None layers
    layers_raw = {k: v for k, v in layers_raw.items() if v is not None}

    if not layers_raw:
        raise RuntimeError("No omics layers could be loaded in Step 1.")

    print("\n✅ Loaded layers:")
    for k, v in layers_raw.items():
        print(f" - {k}: {v.shape[0]} samples × {v.shape[1]} features")

    # ---------------- Metadata handling ----------------
    all_samples = sorted(set().union(*[df.index for df in layers_raw.values()]))

    if metadata_path is not None and os.path.exists(metadata_path):
        meta = pd.read_csv(metadata_path)
        cand_cols = [c for c in meta.columns if "sample" in c.lower() or "id" in c.lower()]
        if not cand_cols:
            raise ValueError(
                f"Could not find sample ID column in metadata. "
                f"Columns: {meta.columns.tolist()}"
            )
        sample_col = cand_cols[0]
        meta[sample_col] = meta[sample_col].astype(str)
        meta = meta.set_index(sample_col)

        # Use provided label_column if available, otherwise use intelligent auto-detection
        if label_column is not None and label_column in meta.columns:
            label_col = label_column
            labels = meta[label_col].astype(str)
            print(f"Using provided label column: {label_col}")
        else:
            # Use intelligent detection (LLM-based if available, otherwise rule-based)
            detected_col = detect_label_column_smart(meta, use_llm=True)
            
            if detected_col:
                label_col = detected_col
                labels = meta[label_col].astype(str)
                if label_column is not None:
                    print(f"Warning: Provided label_column '{label_column}' not found. Auto-detected: {label_col}")
                else:
                    print(f"✅ Auto-detected label column: {label_col}")
            else:
                # Fallback to basic keyword-based detection
                label_cols = [
                    c
                    for c in meta.columns
                    if any(x in c.lower() for x in ["label", "status", "class", "group", "phenotype"])
                ]
                if label_cols:
                    label_col = label_cols[0]
                    labels = meta[label_col].astype(str)
                    if label_column is not None:
                        print(f"Warning: Provided label_column '{label_column}' not found. Auto-detected: {label_col}")
                    else:
                        print(f"Auto-detected label column (keyword-based): {label_col}")
                else:
                    labels = pd.Series("Diseased", index=meta.index)
                    if label_column is not None:
                        print(f"Warning: Provided label_column '{label_column}' not found. Using default 'Diseased' labels.")
                    else:
                        print("⚠️  Could not detect label column. Using default 'Diseased' labels.")

        labels = labels.reindex(all_samples).fillna("Diseased")
        metadata_source = "user_provided"
    else:
        labels = pd.Series("Diseased", index=all_samples, name="label")
        meta = labels.to_frame()
        metadata_source = "generated_default"

    meta_out = os.path.join(STEP, "metadata.csv")
    meta.to_csv(meta_out)
    print(f"\n📄 Metadata saved to: {meta_out}")
    print("Label counts:")
    print(labels.value_counts())

    # ---------------- Basic QC plots per layer ----------------
    print("\n📊 Generating basic QC plots...")
    for lname, df in layers_raw.items():
        basic_qc_plots(df, lname, PLOT_DIR)

    print(f"QC plots saved in: {PLOT_DIR}")

    # ---------------- Save per-layer matrices & manifest ----------------
    for lname, df in layers_raw.items():
        out_path = os.path.join(STEP, f"{lname}.parquet")
        df.to_parquet(out_path)
        print(f"💾 Saved {lname} to {out_path}")

    manifest = {
        "base_dir": STEP,
        "layers": {},
        "metadata_source": metadata_source,
    }

    for lname, df in layers_raw.items():
        manifest["layers"][lname] = {
            "n_samples": int(df.shape[0]),
            "n_features": int(df.shape[1]),
            "normalization_guess": guess_normalization(df, lname),
            "file": f"{lname}.parquet",
        }

    manifest_path = os.path.join(STEP, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n🧾 Manifest written to: {manifest_path}")
    print(json.dumps(manifest, indent=2))

    print("\n✅ STEP 1 (Ingestion + basic QC) complete.")

    return {
        "step_dir": STEP,
        "manifest": manifest_path,
        "metadata": meta_out,
    }
