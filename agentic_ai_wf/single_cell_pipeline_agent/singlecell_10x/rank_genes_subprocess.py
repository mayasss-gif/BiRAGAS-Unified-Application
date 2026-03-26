#!/usr/bin/env python3
"""
Subprocess wrapper for rank_genes_groups computation and plotting (Celery fork-safe).

This module runs rank_genes_groups computation and plotting in an isolated subprocess
to avoid Celery prefork deadlocks with matplotlib/R resources.

Usage:
    python -m agentic_ai_wf.single_cell_pipeline_agent.singlecell_10x.rank_genes_subprocess \
        --h5ad <input.h5ad> \
        --output_dir <output_dir> \
        --analysis_name <name> \
        --method <t-test|logreg> \
        --n_genes <50>
"""

import os
os.environ["MPLBACKEND"] = "Agg"
import matplotlib
matplotlib.use("Agg")

import argparse
import sys
from pathlib import Path
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import gc


def run_rank_genes_groups_isolated(
    h5ad_path: str,
    output_dir: str,
    analysis_name: str,
    method: str = "t-test",
    n_genes: int = 50,
    groupby: str = "leiden",
):
    """
    Run rank_genes_groups computation and plotting in isolated process.
    
    Args:
        h5ad_path: Path to input h5ad file
        output_dir: Output directory for plots and results
        analysis_name: Analysis name for file naming
        method: Statistical method ('t-test' or 'logreg')
        n_genes: Number of top genes per group
        groupby: Column name for grouping (default: 'leiden')
    """
    h5ad_path = Path(h5ad_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set scanpy figure directory
    sc.settings.figdir = output_dir
    
    print(f"Loading AnnData from: {h5ad_path}")
    adata = sc.read_h5ad(h5ad_path)
    print(f"Loaded {adata.n_obs} cells × {adata.n_vars} genes")
    
    # Verify groupby column exists
    if groupby not in adata.obs.columns:
        raise ValueError(f"Groupby column '{groupby}' not found in adata.obs")
    
    clusters = sorted(adata.obs[groupby].unique().tolist(), key=lambda x: int(x))
    print(f"Found {len(clusters)} clusters: {clusters}")
    
    # Compute rank_genes_groups
    print(f"Computing rank_genes_groups using method='{method}'...")
    try:
        sc.tl.rank_genes_groups(
            adata,
            groupby=groupby,
            method=method,
            n_genes=n_genes,
            use_raw=False,
        )
        print(f"✅ Successfully computed rank_genes_groups using {method}")
    except Exception as e:
        print(f"❌ rank_genes_groups failed with {method}: {e}")
        if method == "t-test":
            print("Trying fallback method 'logreg'...")
            sc.tl.rank_genes_groups(
                adata,
                groupby=groupby,
                method="logreg",
                n_genes=n_genes,
                use_raw=False,
            )
            print("✅ Successfully computed rank_genes_groups using logreg")
        else:
            raise
    
    # Generate plots with cleanup after each
    print("Generating rank plot...")
    sc.pl.rank_genes_groups(
        adata,
        n_genes=20,
        sharey=False,
        show=False,
        save=f"_{analysis_name}_cluster_markers_rankplot.png",
    )
    plt.close("all")
    gc.collect()
    
    print("Generating heatmap...")
    heatmap_height_clusters = max(8.0, 0.6 * len(clusters))
    sc.pl.rank_genes_groups_heatmap(
        adata,
        n_genes=10,
        show=False,
        save=f"_{analysis_name}_cluster_markers_heatmap.png",
        figsize=(10, heatmap_height_clusters),
    )
    plt.close("all")
    gc.collect()
    
    print("Generating dotplot...")
    sc.pl.rank_genes_groups_dotplot(
        adata,
        n_genes=10,
        show=False,
        save=f"_{analysis_name}_cluster_markers_dotplot.png",
    )
    plt.close("all")
    gc.collect()
    
    # Extract markers dataframe
    print("Extracting marker genes dataframe...")
    try:
        markers_all = sc.get.rank_genes_groups_df(adata, None)
    except Exception:
        # Fallback manual extraction
        rg = adata.uns["rank_genes_groups"]
        groups = rg["names"].dtype.names
        rows = []
        for g in groups:
            names = rg["names"][g]
            scores = rg["scores"][g]
            pvals_adj = rg["pvals_adj"][g]
            for rank, (gene, score, padj) in enumerate(
                zip(names, scores, pvals_adj), start=1
            ):
                rows.append(
                    {
                        "group": g,
                        "names": gene,
                        "scores": score,
                        "pvals_adj": padj,
                        "rank": rank,
                    }
                )
        markers_all = pd.DataFrame(rows)
    
    # Save markers CSV
    markers_csv = output_dir / "intercluster_cluster_markers.csv"
    markers_all.to_csv(markers_csv, index=False)
    print(f"✅ Saved markers CSV: {markers_csv}")
    
    # Save updated h5ad with rank_genes_groups results
    updated_h5ad = output_dir / f"{analysis_name}_with_rank_genes.h5ad"
    adata.write_h5ad(updated_h5ad)
    print(f"✅ Saved updated h5ad: {updated_h5ad}")
    
    print("✅ rank_genes_groups computation and plotting completed successfully")


def main():
    parser = argparse.ArgumentParser(
        description="Run rank_genes_groups in isolated subprocess (Celery fork-safe)"
    )
    parser.add_argument("--h5ad", required=True, help="Path to input h5ad file")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--analysis-name", required=True, help="Analysis name")
    parser.add_argument(
        "--method",
        default="t-test",
        choices=["t-test", "logreg"],
        help="Statistical method (default: t-test)",
    )
    parser.add_argument(
        "--n-genes",
        type=int,
        default=50,
        help="Number of top genes per group (default: 50)",
    )
    parser.add_argument(
        "--groupby",
        default="leiden",
        help="Column name for grouping (default: leiden)",
    )
    
    args = parser.parse_args()
    
    try:
        run_rank_genes_groups_isolated(
            h5ad_path=args.h5ad,
            output_dir=args.output_dir,
            analysis_name=args.analysis_name,
            method=args.method,
            n_genes=args.n_genes,
            groupby=args.groupby,
        )
        sys.exit(0)
    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

