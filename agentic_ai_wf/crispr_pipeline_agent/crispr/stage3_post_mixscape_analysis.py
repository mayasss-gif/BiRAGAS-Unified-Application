#!/usr/bin/env python3
"""
Stage-3 Post Mixscape Analysis (Per-sample compatible)

✅ FIXED:
- Loads stage2 outputs using pattern: stage2_mixscape_*.h5ad
- Works per sample folder: processed/<GSM_ID>/
- Writes outputs inside sample folder:
    processed/<GSM_ID>/processed_stage3/
"""

import argparse
from pathlib import Path
import pandas as pd
import scanpy as sc
import numpy as np


def info(msg):
    print(f"[INFO] {msg}", flush=True)


def load_stage2_files(stage2_dir: Path):
    # ✅ FIXED PATTERN
    files = sorted(stage2_dir.glob("stage2_mixscape_*.h5ad"))

    valid = []
    for f in files:
        try:
            ad = sc.read_h5ad(f)
            if "perturbation_id" in ad.obs.columns:
                valid.append((f.name, ad))
        except Exception as e:
            print(f"[WARN] skipping {f.name}: {e}")
    return valid


def compute_perturbation_summary(adata):
    df = adata.obs.copy()
    df = df[df["condition_class"].astype(str).eq("perturbed_single")].copy()

    if df.empty:
        return pd.DataFrame()

    if "is_responder" not in df.columns:
        df["is_responder"] = 0
    if "mixscape_score" not in df.columns:
        df["mixscape_score"] = np.nan

    out = (
        df.groupby("perturbation_id", observed=False)
        .agg(
            n_cells=("perturbation_id", "size"),
            responder_rate=("is_responder", "mean"),
            mean_mixscape_score=("mixscape_score", "mean"),
        )
        .reset_index()
        .sort_values(["responder_rate", "n_cells"], ascending=False)
    )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage2_dir", required=True, help="Directory containing stage2 output .h5ad")
    ap.add_argument("--out_dir", required=True, help="Output directory for stage3 results")
    args = ap.parse_args()

    stage2_dir = Path(args.stage2_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    info(f"Loading stage2 outputs from {stage2_dir}")
    datasets = load_stage2_files(stage2_dir)

    if len(datasets) == 0:
        raise RuntimeError(f"No valid stage2 files found in: {stage2_dir}")

    info(f"Loaded {len(datasets)} stage2 datasets")

    # Merge
    adatas = []
    names = []

    for name, ad in datasets:
        sample_name = name.replace("stage2_mixscape_", "").replace(".h5ad", "")
        ad.obs["sample"] = sample_name
        adatas.append(ad)
        names.append(sample_name)

    info("Concatenating datasets...")
    merged = sc.concat(adatas, label="batch", keys=names, join="outer", merge="same")

    merged_out = out_dir / "stage3_merged.h5ad"
    merged.write(merged_out)
    info(f"[OK] merged h5ad -> {merged_out}")

    # Perturbation summary
    info("Computing perturbation summary...")
    summary = compute_perturbation_summary(merged)

    summary_path = out_dir / "stage3_perturbation_ranked.tsv"
    summary.to_csv(summary_path, sep="\t", index=False)
    info(f"[OK] perturbation summary -> {summary_path}")

    # responder rate plot
    if not summary.empty:
        import matplotlib.pyplot as plt

        plt.figure()
        plt.hist(summary["responder_rate"].fillna(0), bins=30)
        plt.title("Responder Rate Distribution")
        plt.xlabel("responder_rate")
        plt.ylabel("count")
        fig_path = out_dir / "stage3_responder_rate_hist.png"
        plt.savefig(fig_path, dpi=200, bbox_inches="tight")
        info(f"[OK] saved plot -> {fig_path}")


if __name__ == "__main__":
    main()

