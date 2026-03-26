#!/usr/bin/env python3
"""
Stage-4 DEG + Signature Extraction

Input:
  processed_stage3/stage3_merged.h5ad

Outputs:
  processed_stage4/
    stage4_deg_top_markers.tsv
    stage4_deg_summary.tsv
    stage4_training_table.tsv

Run:
python stage4_deg_signature.py \
  --input_h5ad processed_stage3/stage3_merged.h5ad \
  --out_dir processed_stage4 \
  --min_cells_per_pert 10 \
  --n_top_genes 50
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import scanpy as sc


def info(msg):
    print(f"[INFO] {msg}", flush=True)


def safe_rank_genes_groups(adata, groupby="perturbation_id", reference="control",
                           n_top_genes=50, method="wilcoxon"):
    """
    DEG ranking per perturbation vs control.
    """
    info("Running rank_genes_groups (DEG)...")
    sc.tl.rank_genes_groups(
        adata,
        groupby=groupby,
        reference=reference,
        method=method,
        n_genes=n_top_genes,
        pts=True
    )


def extract_rank_genes_table(adata, n_top_genes=50):
    """
    Convert rank_genes_groups results into long TSV format:
      perturbation_id, gene, score, logFC, pval, pval_adj, pct_in, pct_ref
    """
    rg = adata.uns["rank_genes_groups"]
    groups = rg["names"].dtype.names

    rows = []
    for g in groups:
        names = rg["names"][g][:n_top_genes]
        scores = rg["scores"][g][:n_top_genes]
        logfc = rg["logfoldchanges"][g][:n_top_genes] if "logfoldchanges" in rg else [np.nan] * n_top_genes
        pvals = rg["pvals"][g][:n_top_genes]
        padj = rg["pvals_adj"][g][:n_top_genes]

        # pct expressions (if available)
        pct_in = rg["pts"][g][:n_top_genes] if "pts" in rg else [np.nan] * n_top_genes
        pct_ref = rg["pts_rest"][g][:n_top_genes] if "pts_rest" in rg else [np.nan] * n_top_genes

        for i in range(len(names)):
            rows.append(
                {
                    "perturbation_id": g,
                    "gene": str(names[i]),
                    "score": float(scores[i]) if scores is not None else np.nan,
                    "logfoldchange": float(logfc[i]) if logfc is not None else np.nan,
                    "pval": float(pvals[i]),
                    "pval_adj": float(padj[i]),
                    "pct_in_group": float(pct_in[i]) if pct_in is not None else np.nan,
                    "pct_in_control": float(pct_ref[i]) if pct_ref is not None else np.nan,
                }
            )

    return pd.DataFrame(rows)


def make_training_table(summary_df, top_markers_df, topk=20):
    """
    Build a training table with top marker list per perturbation.
    """
    top_markers_list = (
        top_markers_df.sort_values(["perturbation_id", "score"], ascending=[True, False])
        .groupby("perturbation_id")["gene"]
        .apply(lambda x: ",".join(x.head(topk).astype(str)))
        .reset_index()
        .rename(columns={"gene": f"top_{topk}_markers"})
    )

    merged = summary_df.merge(top_markers_list, on="perturbation_id", how="left")
    return merged


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_h5ad", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--min_cells_per_pert", type=int, default=10)
    ap.add_argument("--n_top_genes", type=int, default=50)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    info(f"Reading merged h5ad: {args.input_h5ad}")
    adata = sc.read_h5ad(args.input_h5ad)

    # fix obs names duplicates warning
    adata.obs_names_make_unique()

    # Ensure required columns exist
    if "condition_class" not in adata.obs.columns:
        raise RuntimeError("Missing 'condition_class' in obs.")
    if "perturbation_id" not in adata.obs.columns:
        raise RuntimeError("Missing 'perturbation_id' in obs.")

    # Only control + perturbed_single
    use = adata[adata.obs["condition_class"].isin(["control", "perturbed_single"])].copy()
    info(f"Subset ctrl+single: {use.n_obs} cells")

    # Filter perturbations with enough cells
    counts = use.obs["perturbation_id"].value_counts()
    keep_perts = counts[counts >= args.min_cells_per_pert].index.tolist()

    # Always keep control
    if "control" not in keep_perts:
        keep_perts.append("control")

    use = use[use.obs["perturbation_id"].isin(keep_perts)].copy()
    info(f"Kept perturbations >= {args.min_cells_per_pert} cells: {len(keep_perts)}")
    info(f"Final cells: {use.n_obs}")

    # Create summary table (basic)
    summary = (
        use.obs.groupby("perturbation_id", observed=False)
        .agg(n_cells=("perturbation_id", "size"))
        .reset_index()
        .sort_values("n_cells", ascending=False)
    )

    # DEG ranking
    safe_rank_genes_groups(use, groupby="perturbation_id", reference="control",
                           n_top_genes=args.n_top_genes, method="wilcoxon")

    markers = extract_rank_genes_table(use, n_top_genes=args.n_top_genes)

    markers_path = out_dir / "stage4_deg_top_markers.tsv"
    markers.to_csv(markers_path, sep="\t", index=False)
    info(f"[OK] top markers -> {markers_path}")

    summary_path = out_dir / "stage4_deg_summary.tsv"
    summary.to_csv(summary_path, sep="\t", index=False)
    info(f"[OK] summary -> {summary_path}")

    # Load stage3 perturbation_ranked.tsv if exists and merge
    stage3_ranked = Path("processed_stage3/stage3_perturbation_ranked.tsv")
    if stage3_ranked.exists():
        ranked = pd.read_csv(stage3_ranked, sep="\t")
        ranked = ranked.rename(columns={"perturbation_id": "perturbation_id"})
        ranked_summary = ranked[["perturbation_id", "n_cells", "responder_rate", "mean_mixscape_score"]].copy()
    else:
        ranked_summary = summary.copy()
        ranked_summary["responder_rate"] = np.nan
        ranked_summary["mean_mixscape_score"] = np.nan

    training = make_training_table(ranked_summary, markers, topk=20)
    training_path = out_dir / "stage4_training_table.tsv"
    training.to_csv(training_path, sep="\t", index=False)
    info(f"[OK] training table -> {training_path}")

    # Write updated deg results h5ad (optional)
    deg_h5ad_path = out_dir / "stage4_deg_results.h5ad"
    use.write(deg_h5ad_path)
    info(f"[OK] wrote h5ad -> {deg_h5ad_path}")

    info("[DONE] Stage-4 complete")


if __name__ == "__main__":
    main()

