from __future__ import annotations
# pipeline.py

import os
os.environ["MPLBACKEND"] = "Agg"
import matplotlib
matplotlib.use("Agg")

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scanpy as sc
from scipy import sparse as sp_sparse
import gc
import subprocess
import sys
import tempfile


from .config_cli import logger, DO_PATHWAY_CLUSTERING as DEFAULT_DO_PATHWAY_CLUSTERING
from .gene_names import update_gene_names
from .markers import compute_celltype_markers
from .group_de import (
    plot_groupwise_celltype_proportions,
    plot_group_specific_umaps,
    compute_de_by_celltype,
)
from .pathway_enrichment import run_cluster_marker_enrichment
from .summary_ct_deg import summarize_celltype_degs_markers_pathways


# =====================================================================
#  MAIN SCANPY PIPELINE (verbatim copy, no logic changes)
# =====================================================================

def run_scanpy_pipeline(
    adata,
    out_dir: Path,
    analysis_name: str,
    batch_key: str | None = None,
    integration_method: str | None = None,  # "bbknn" or None
    do_groupwise_de: bool = False,
    group_col: str = "group",
    cluster_col: str = "leiden",
    do_dpt: bool = False,  # optional pseudotime
    do_pathway_clustering: bool | None = None,  # None = use default from config
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Use provided parameter or fall back to config default
    if do_pathway_clustering is None:
        do_pathway_clustering = DEFAULT_DO_PATHWAY_CLUSTERING

    # Structured subfolders
    summary_dir = out_dir / "00_analysis_summary"
    qc_dir = out_dir / "01_qc_and_filtering"
    hvg_dir = out_dir / "02_highly_variable_genes"
    dimred_dir = out_dir / "03_dimensionality_reduction_and_embeddings"
    clustering_dir = out_dir / "04_clustering_and_cell_states"

    celltype_root_dir = out_dir / "05_celltype_analysis"
    celltype_anno_dir = celltype_root_dir / "celltype_annotation"
    celltype_markers_dir = celltype_root_dir / "celltype_specific_markers"

    deg_root_dir = out_dir / "06_groupwise_deg"
    pathway_root_dir = out_dir / "07_pathway_enrichment"
    reference_dir = out_dir / "08_reference_summary"

    for d in [
        summary_dir,
        qc_dir,
        hvg_dir,
        dimred_dir,
        clustering_dir,
        celltype_root_dir,
        celltype_anno_dir,
        celltype_markers_dir,
        reference_dir,
    ]:
        d.mkdir(parents=True, exist_ok=True)

    logger.info(f"=== Running Scanpy pipeline: {analysis_name} ===")
    logger.info(f"Output root folder: {out_dir}")

    initial_cells = adata.n_obs
    initial_genes = adata.n_vars

    if "counts" not in adata.layers:
        if sp_sparse.issparse(adata.X):
            adata.layers["counts"] = adata.X.copy()
        else:
            adata.layers["counts"] = np.array(adata.X)

    adata = update_gene_names(adata)
    if getattr(adata, "raw", None) is not None:
        adata.raw = None

    sc.pp.filter_genes(adata, min_cells=3)
    genes_after_min_cells = adata.n_vars

    adata.var["mt"] = adata.var_names.str.upper().str.startswith("MT-")
    sc.pp.calculate_qc_metrics(
        adata,
        qc_vars=["mt"],
        percent_top=None,
        log1p=False,
        inplace=True,
    )

    groupby_col = "sample" if "sample" in adata.obs.columns else None
    sc.settings.figdir = qc_dir

    sc.pl.violin(
        adata,
        ["n_genes_by_counts", "total_counts", "pct_counts_mt"],
        jitter=0.4,
        groupby=groupby_col,
        multi_panel=True,
        show=False,
        save=f"_{analysis_name}_qc_violin.png",
    )
    plt.close("all")
    gc.collect()
    
    sc.pl.scatter(
        adata,
        x="total_counts",
        y="pct_counts_mt",
        show=False,
        save=f"_{analysis_name}_qc_total_vs_mito.png",
    )
    plt.close("all")
    gc.collect()
    
    sc.pl.scatter(
        adata,
        x="total_counts",
        y="n_genes_by_counts",
        show=False,
        save=f"_{analysis_name}_qc_total_vs_genes.png",
    )
    plt.close("all")
    gc.collect()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].hist(adata.obs["n_genes_by_counts"], bins=60)
    axes[0].set_title("n_genes_by_counts")
    axes[1].hist(adata.obs["total_counts"], bins=60)
    axes[1].set_title("total_counts")
    axes[2].hist(adata.obs["pct_counts_mt"], bins=60)
    axes[2].set_title("pct_counts_mt")
    plt.tight_layout()
    plt.savefig(qc_dir / f"{analysis_name}_qc_metric_histograms.png", dpi=300)
    plt.close("all")
    gc.collect()

    MIN_GENES = 200
    MAX_GENES = 6000
    MAX_MT_PCT = 15.0
    adata = adata[
        (adata.obs["n_genes_by_counts"] > MIN_GENES)
        & (adata.obs["n_genes_by_counts"] < MAX_GENES)
        & (adata.obs["pct_counts_mt"] < MAX_MT_PCT)
    ].copy()

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata.raw = adata

    n_cells = adata.n_obs
    if n_cells < 4000:
        N_HVG = 2000
    elif n_cells > 200000:
        N_HVG = 4000
    else:
        N_HVG = 4000

    if "sample" in adata.obs.columns:
        sc.pp.highly_variable_genes(
            adata,
            flavor="seurat_v3",
            n_top_genes=N_HVG,
            batch_key="sample",
        )
    else:
        sc.pp.highly_variable_genes(
            adata,
            flavor="seurat_v3",
            n_top_genes=N_HVG,
        )

    sc.settings.figdir = hvg_dir
    sc.pl.highly_variable_genes(adata, show=False, save=f"_{analysis_name}_highly_variable_genes_plot.png")
    plt.close("all")
    gc.collect()
    
    hvg_count = int(adata.var["highly_variable"].sum())
    hvg_table = adata.var.copy()
    hvg_table.to_csv(hvg_dir / f"{analysis_name}_highly_variable_genes_table.csv")

    sc.pp.scale(adata, max_value=10)
    sc.tl.pca(adata, n_comps=50, svd_solver="arpack", use_highly_variable=True)

    sc.settings.figdir = dimred_dir
    sc.pl.pca_variance_ratio(adata, log=True, show=False, save=f"_{analysis_name}_PCA_variance_explained.png")
    plt.close("all")
    gc.collect()

    if integration_method == "bbknn" and batch_key is not None and batch_key in adata.obs.columns:
        try:
            import scanpy.external as sce
            logger.info(f"[INTEGRATION] Using BBKNN with batch_key='{batch_key}'.")
            sce.pp.bbknn(adata, batch_key=batch_key)
        except Exception:
            logger.warning("[INTEGRATION] scanpy.external (bbknn) not available. Falling back to standard neighbors.")
            sc.pp.neighbors(adata, n_neighbors=15, n_pcs=30)
    else:
        logger.info("[INTEGRATION] Using standard neighbors (no explicit integration).")
        sc.pp.neighbors(adata, n_neighbors=15, n_pcs=30)

    sc.tl.umap(adata)

    if adata.n_obs <= 50000:
        sc.tl.tsne(adata, n_pcs=30, use_rep="X_pca")
    else:
        logger.info(f"[TSNE] Skipping t-SNE because n_obs={adata.n_obs} > 50000.")

    if do_dpt:
        try:
            logger.info(f"[{analysis_name}] Computing diffusion map + DPT (trajectory inference)...")
            sc.tl.diffmap(adata)
            sc.tl.dpt(adata)
            sc.pl.umap(
                adata,
                color=["dpt_pseudotime"],
                show=False,
                save=f"_{analysis_name}_UMAP_dpt_pseudotime.png",
            )
            plt.close("all")
            gc.collect()
            
            if "X_tsne" in adata.obsm.keys():
                sc.pl.tsne(
                    adata,
                    color=["dpt_pseudotime"],
                    show=False,
                    save=f"_{analysis_name}_TSNE_dpt_pseudotime.png",
                )
                plt.close("all")
                gc.collect()
                
            sc.pl.diffmap(
                adata,
                color=["dpt_pseudotime"],
                show=False,
                save=f"_{analysis_name}_DIFFMAP_dpt_pseudotime.png",
            )
            plt.close("all")
            gc.collect()
        except Exception as e:
            logger.warning(f"[{analysis_name}] DPT computation/plotting failed: {e}")

    color_cols = []
    if "sample" in adata.obs.columns:
        color_cols.append("sample")
    if "group" in adata.obs.columns:
        color_cols.append("group")

    if color_cols:
        sc.pl.umap(
            adata,
            color=color_cols,
            wspace=0.4,
            size=15,
            show=False,
            save=f"_{analysis_name}_UMAP_samples_groups.png",
        )
        plt.close("all")
        gc.collect()
        
        if "X_tsne" in adata.obsm.keys():
            sc.pl.tsne(
                adata,
                color=color_cols,
                wspace=0.4,
                size=15,
                show=False,
                save=f"_{analysis_name}_TSNE_samples_groups.png",
            )
            plt.close("all")
            gc.collect()
    else:
        sc.pl.umap(adata, size=15, show=False, save=f"_{analysis_name}_UMAP.png")
        plt.close("all")
        gc.collect()
        
        if "X_tsne" in adata.obsm.keys():
            sc.pl.tsne(adata, size=15, show=False, save=f"_{analysis_name}_TSNE.png")
            plt.close("all")
            gc.collect()

    for qc_col in ["n_genes_by_counts", "total_counts", "pct_counts_mt"]:
        if qc_col in adata.obs.columns:
            sc.pl.umap(
                adata,
                color=[qc_col],
                size=15,
                show=False,
                save=f"_{analysis_name}_UMAP_{qc_col}.png",
            )
            plt.close("all")
            gc.collect()
            
            if "X_tsne" in adata.obsm.keys():
                sc.pl.tsne(
                    adata,
                    color=[qc_col],
                    size=15,
                    show=False,
                    save=f"_{analysis_name}_TSNE_{qc_col}.png",
                )
                plt.close("all")
                gc.collect()

    sc.settings.figdir = clustering_dir
    sc.tl.leiden(adata, resolution=0.5)
    clusters = sorted(adata.obs["leiden"].unique().tolist(), key=lambda x: int(x))
    cluster_sizes = adata.obs["leiden"].value_counts().sort_index()
    cluster_sizes.to_csv(
        clustering_dir / f"{analysis_name}_cluster_cell_counts_leiden.csv",
        header=["n_cells"],
    )

    sc.pl.umap(
        adata,
        color=["leiden"],
        legend_loc="on data",
        size=15,
        show=False,
        save=f"_{analysis_name}_UMAP_leiden.png",
    )
    plt.close("all")
    gc.collect()
    
    if "X_tsne" in adata.obsm.keys():
        sc.pl.tsne(
            adata,
            color=["leiden"],
            legend_loc="on data",
            size=15,
            show=False,
            save=f"_{analysis_name}_TSNE_leiden.png",
        )
        plt.close("all")
        gc.collect()
        
    sc.pl.pca(
        adata,
        color=["leiden"],
        show=False,
        save=f"_{analysis_name}_PCA_leiden.png",
    )
    plt.close("all")
    gc.collect()

    # ========= Cell type detection / prediction =========
    celltype_col_raw = None
    celltype_source = None
    standard_celltype_col = "celltype"

    candidate_celltype_cols = [
        "cell_type",
        "celltype",
        "CellType",
        "cell_types",
        "celltype_major",
        "cell_type_major",
        "cell_identity",
        "cell_ontology_class",
        "celltype_new",
        "celltype_celltypist",
    ]
    for col in candidate_celltype_cols:
        if col in adata.obs.columns:
            celltype_col_raw = col
            celltype_source = f"provided_obs_metadata ({col})"
            break

    if celltype_col_raw is None:
        try:
            import celltypist
            from celltypist import models as ct_models
            try:
                ct_models.download_models()
            except Exception as e:
                logger.warning(f"CellTypist model download failed: {e}")
            model_name = "Immune_All_Low.pkl"
            logger.info(
                f"[CELLTYPE-ML] No curated cell-type labels detected; "
                f"applying CellTypist model '{model_name}' ({analysis_name})."
            )
            preds = celltypist.annotate(
                adata,
                model=model_name,
                majority_voting=True,
            )
            plabels = preds.predicted_labels
            if "majority_voting" in plabels.columns:
                adata.obs["celltype_celltypist"] = (
                    plabels["majority_voting"].reindex(adata.obs_names).astype("category")
                )
                celltype_col_raw = "celltype_celltypist"
            elif "predicted_labels" in plabels.columns:
                adata.obs["celltype_celltypist"] = (
                    plabels["predicted_labels"].reindex(adata.obs_names).astype("category")
                )
                celltype_col_raw = "celltype_celltypist"
            if celltype_col_raw is not None:
                celltype_source = "CellTypist (supervised ML cell-type classifier)"
        except Exception as e:
            logger.warning(f"CellTypist failed for {analysis_name}: {e}")

    celltype_col = None
    if celltype_col_raw is not None:
        if not pd.api.types.is_categorical_dtype(adata.obs[celltype_col_raw]):
            adata.obs[celltype_col_raw] = adata.obs[celltype_col_raw].astype("category")
        adata.obs[standard_celltype_col] = adata.obs[celltype_col_raw].astype("category")
        celltype_col = standard_celltype_col

    if celltype_col is not None:
        sc.settings.figdir = celltype_anno_dir
        sc.pl.umap(
            adata,
            color=[celltype_col],
            legend_loc="right margin",
            size=15,
            show=False,
            save=f"_{analysis_name}_UMAP_celltypes.png",
        )
        plt.close("all")
        gc.collect()
        
        if "X_tsne" in adata.obsm.keys():
            sc.pl.tsne(
                adata,
                color=[celltype_col],
                legend_loc="right margin",
                size=15,
                show=False,
                save=f"_{analysis_name}_TSNE_celltypes.png",
            )
            plt.close("all")
            gc.collect()
            
        sc.pl.pca(
            adata,
            color=[celltype_col],
            show=False,
            save=f"_{analysis_name}_PCA_celltypes.png",
        )
        plt.close("all")
        gc.collect()
        
        ct_counts = adata.obs[celltype_col].value_counts().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(ct_counts.index.astype(str), ct_counts.values)
        ax.set_xticklabels(ct_counts.index.astype(str), rotation=90)
        ax.set_ylabel("cells")
        ax.set_title(f"Cell types ({celltype_col})")
        plt.tight_layout()
        plt.savefig(celltype_anno_dir / f"{analysis_name}_celltype_composition_barplot.png", dpi=300)
        plt.close("all")
        gc.collect()

        logger.info(f"[{analysis_name}] Computing celltype-specific marker genes (Scanpy)...")
        compute_celltype_markers(
            adata,
            celltype_col=celltype_col,
            out_dir=celltype_markers_dir,
            analysis_name=analysis_name,
            n_markers_per_type=50,
            reference_dir=reference_dir,
        )

        if "leiden" in adata.obs.columns:
            logger.info(f"[{analysis_name}] Making side-by-side embeddings (leiden vs cell type).")
            sc.settings.figdir = celltype_anno_dir
            sc.pl.pca(
                adata,
                color=["leiden", celltype_col],
                wspace=0.4,
                show=False,
                save=f"_{analysis_name}_PCA_leiden_vs_celltype.png",
            )
            plt.close("all")
            gc.collect()
            
            if "X_tsne" in adata.obsm.keys():
                sc.pl.tsne(
                    adata,
                    color=["leiden", celltype_col],
                    wspace=0.4,
                    show=False,
                    save=f"_{analysis_name}_TSNE_leiden_vs_celltype.png",
                )
                plt.close("all")
                gc.collect()
                
            sc.pl.umap(
                adata,
                color=["leiden", celltype_col],
                wspace=0.4,
                show=False,
                save=f"_{analysis_name}_UMAP_leiden_vs_celltype.png",
            )
            plt.close("all")
            gc.collect()

    else:
        logger.info(
            f"[CELLTYPE] No cell type column found for {analysis_name} "
            f"and no ML-based cell-type prediction could be applied."
        )

    # Build mapping: Leiden cluster → "<clusterID>_<MajorCellType>"
    cluster_celltype_map = None
    if celltype_col is not None and "leiden" in adata.obs.columns:
        tmp = adata.obs[["leiden", celltype_col]].dropna()
        if not tmp.empty:
            cluster_celltype_map = {}
            ref_rows = []
            for cl, sub in tmp.groupby("leiden"):
                top_ct = sub[celltype_col].astype(str).value_counts().idxmax()
                safe_ct = str(top_ct).replace(" ", "_").replace("/", "_")
                label = f"{cl}_{safe_ct}"
                cluster_celltype_map[str(cl)] = label
                ref_rows.append(
                    {
                        "leiden_cluster": cl,
                        "major_celltype_label": top_ct,
                        "cluster_celltype_label": label,
                    }
                )
            logger.info(f"[{analysis_name}] Cluster → celltype map: {cluster_celltype_map}")

            reference_dir.mkdir(parents=True, exist_ok=True)
            ref_df = pd.DataFrame(ref_rows)
            ref_map_file = reference_dir / f"{analysis_name}_cluster_to_celltype_map.csv"
            ref_df.to_csv(ref_map_file, index=False)
            logger.info(
                f"[{analysis_name}] Saved cluster→celltype reference table: {ref_map_file}"
            )

    # ========= Cluster marker genes + intercluster DEG =========
    try:
        logger.info(f"[{analysis_name}] Computing cluster marker genes (leiden, t-test) via subprocess for Celery compatibility...")

        intercluster_dir = clustering_dir / "Intercluster_analysis_deg"
        intercluster_dir.mkdir(parents=True, exist_ok=True)
        
        # CRITICAL FOR CELERY: Run rank_genes_groups in isolated subprocess
        # This avoids deadlocks with matplotlib/R resources in Celery prefork mode
        # Save adata to temporary h5ad file
        temp_h5ad = intercluster_dir / f"{analysis_name}_temp_for_rank_genes.h5ad"
        logger.info(f"[{analysis_name}] Saving temporary h5ad for subprocess: {temp_h5ad}")
        adata.write_h5ad(temp_h5ad)
        
        # Build subprocess command
        cmd = [
            sys.executable,
            "-m",
            "agentic_ai_wf.single_cell_pipeline_agent.singlecell_10x.rank_genes_subprocess",
            "--h5ad", str(temp_h5ad),
            "--output-dir", str(intercluster_dir),
            "--analysis-name", analysis_name,
            "--method", "t-test",
            "--n-genes", "50",
            "--groupby", "leiden",
        ]
        
        logger.info(f"[{analysis_name}] Running rank_genes_groups in isolated subprocess...")
        logger.debug(f"[{analysis_name}] Command: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
                timeout=3600,  # 1 hour timeout
            )
            
            if result.stdout:
                logger.info(f"[{analysis_name}] Subprocess stdout:\n{result.stdout}")
            if result.stderr:
                logger.warning(f"[{analysis_name}] Subprocess stderr:\n{result.stderr}")
            
            logger.info(f"[{analysis_name}] ✅ rank_genes_groups subprocess completed successfully")
            
        except subprocess.TimeoutExpired as e:
            logger.error(f"[{analysis_name}] ❌ rank_genes_groups subprocess timed out after 1 hour")
            raise RuntimeError("rank_genes_groups execution timed out") from e
            
        except subprocess.CalledProcessError as e:
            logger.error(f"[{analysis_name}] ❌ rank_genes_groups subprocess failed with exit code {e.returncode}")
            if e.stdout:
                logger.error(f"[{analysis_name}] Subprocess stdout:\n{e.stdout}")
            if e.stderr:
                logger.error(f"[{analysis_name}] Subprocess stderr:\n{e.stderr}")
            raise RuntimeError(f"rank_genes_groups execution failed: {e.stderr}") from e
        
        # Load results from subprocess output
        markers_csv = intercluster_dir / "intercluster_cluster_markers.csv"
        if not markers_csv.exists():
            raise FileNotFoundError(f"Markers CSV not found: {markers_csv}")
        
        markers_all = pd.read_csv(markers_csv)
        logger.info(f"[{analysis_name}] Loaded {len(markers_all)} marker gene records from subprocess")
        
        # Add cluster column and cluster_celltype_label mapping
        markers_all["cluster"] = markers_all["group"].astype(str)
        if cluster_celltype_map is not None:
            markers_all["cluster_celltype_label"] = markers_all["cluster"].map(
                cluster_celltype_map
            ).fillna(markers_all["cluster"])
        
        # Load updated h5ad with rank_genes_groups results
        updated_h5ad = intercluster_dir / f"{analysis_name}_with_rank_genes.h5ad"
        if updated_h5ad.exists():
            logger.info(f"[{analysis_name}] Loading updated h5ad with rank_genes_groups results...")
            adata_updated = sc.read_h5ad(updated_h5ad)
            # Copy rank_genes_groups results back to main adata
            if "rank_genes_groups" in adata_updated.uns:
                adata.uns["rank_genes_groups"] = adata_updated.uns["rank_genes_groups"]
                logger.info(f"[{analysis_name}] Copied rank_genes_groups results back to main adata")
            # Clean up temporary files
            updated_h5ad.unlink()
        
        # Clean up temporary h5ad
        if temp_h5ad.exists():
            temp_h5ad.unlink()
            logger.debug(f"[{analysis_name}] Cleaned up temporary h5ad: {temp_h5ad}")
        
        # Save markers CSV with cluster mapping (overwrite the one from subprocess)
        intercluster_csv = intercluster_dir / "intercluster_cluster_markers.csv"
        markers_all.to_csv(intercluster_csv, index=False)
        logger.info(f"[{analysis_name}] Wrote intercluster markers with cluster mapping: {intercluster_csv}")

        markers_all["cluster"] = markers_all["group"].astype(str)
        if cluster_celltype_map is not None:
            markers_all["cluster_celltype_label"] = markers_all["cluster"].map(
                cluster_celltype_map
            ).fillna(markers_all["cluster"])

        intercluster_csv = intercluster_dir / "intercluster_cluster_markers.csv"
        markers_all.to_csv(intercluster_csv, index=False)
        logger.info(f"[{analysis_name}] Wrote intercluster markers: {intercluster_csv}")

        for cl, subdf in markers_all.groupby("cluster"):
            label = (
                markers_all.loc[markers_all["cluster"] == cl, "cluster_celltype_label"]
                .iloc[0]
                if "cluster_celltype_label" in markers_all.columns
                else cl
            )
            safe_label = str(label).replace(" ", "_").replace("/", "_")
            out_f = intercluster_dir / f"cluster_{safe_label}_markers.csv"
            subdf.to_csv(out_f, index=False)

        if do_pathway_clustering:
            pathway_root_dir.mkdir(parents=True, exist_ok=True)
            cluster_pathway_dir = pathway_root_dir / "cluster_marker_enrichment"
            logger.info(f"[{analysis_name}] Running pathway enrichment for cluster markers...")
            run_cluster_marker_enrichment(
                markers_all,
                out_dir=cluster_pathway_dir,
                analysis_name=analysis_name,
                pval_col="pvals_adj",
                pval_cutoff=0.05,
                top_n=200,
                cluster_celltype_map=cluster_celltype_map,
            )
        else:
            logger.info(
                f"[{analysis_name}] Skipping cluster-marker pathway enrichment "
                f"(do_pathway_clustering=False)."
            )

    except Exception as e:
        logger.warning(f"[{analysis_name}] intercluster marker computation failed: {e}")

    # ========= Save processed AnnData =========
    processed_h5ad = out_dir / f"{analysis_name}_processed_scanpy_output.h5ad"
    adata.write_h5ad(processed_h5ad)

    # ========= Summary report =========
    summary_lines = [
        f"=== {analysis_name} ===",
        f"Output folder: {out_dir}",
        f"Initial cells: {initial_cells}",
        f"Initial genes: {initial_genes}",
        f"Genes after min_cells filter: {genes_after_min_cells}",
        f"Cells after QC filters: {adata.n_obs}",
        f"HVGs used: {hvg_count}",
        f"Final shape (post-HVG selection for embeddings, ALL genes retained for DE): "
        f"{adata.n_obs} cells x {adata.n_vars} genes",
        f"Leiden clusters: {len(clusters)}",
        f"Processed AnnData (Scanpy): {processed_h5ad}",
    ]
    if "celltype" in adata.obs.columns:
        celltype_col_used = "celltype"
        ct_counts2 = adata.obs[celltype_col_used].value_counts()
        summary_lines.append(f"Celltype column (standard): {celltype_col_used}")
        if celltype_source is not None:
            summary_lines.append(f"Celltype annotation source: {celltype_source}")
        summary_lines.append("Celltype counts:")
        for ct, n in ct_counts2.items():
            summary_lines.append(f"  {ct}: {int(n)} cells")
    else:
        summary_lines.append("Celltype annotation: not available")

    summary_file = summary_dir / f"{analysis_name}_analysis_summary.txt"
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))

    logger.info("\n".join(summary_lines))

    # ========= Group-wise DE & downstream (single-cell only) =========
    celltype_col = "celltype" if "celltype" in adata.obs.columns else None

    if do_groupwise_de:
        deg_root_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"[{analysis_name}] Comparing cell type proportions between groups...")
        plot_groupwise_celltype_proportions(
            adata,
            group_col=group_col,
            celltype_col=celltype_col,
            out_dir=celltype_anno_dir,
        )

        ct_deg_pathway_dir = pathway_root_dir / "celltype_DEG_enrichment"

        if celltype_col is not None:
            logger.info(f"[{analysis_name}] Running single-cell DE per cell type (Scanpy)...")
            compute_de_by_celltype(
                adata,
                celltype_col=celltype_col,
                group_col=group_col,
                deg_root_dir=deg_root_dir,
            )

            if do_pathway_clustering:
                pathway_root_dir.mkdir(parents=True, exist_ok=True)
                logger.info(f"[{analysis_name}] (Optional) run pathway enrichment for celltype-specific DEGs here if desired.")
                # If you want: run_celltype_deg_enrichment_from_files(...) can be added.

            try:
                summarize_celltype_degs_markers_pathways(
                    out_dir=out_dir,
                    analysis_name=analysis_name,
                    deg_dir=deg_root_dir / "celltype_specific_deg",
                    celltype_dir=celltype_markers_dir,
                    ct_deg_pathway_dir=ct_deg_pathway_dir,
                )
            except Exception as e:
                logger.warning(f"[SUMMARY-CT-DEG] Failed to summarise celltype DEGs+markers+pathways: {e}")

        logger.info(f"[{analysis_name}] Building group-specific UMAPs...")
        color_col_for_umap = celltype_col if celltype_col is not None else cluster_col
        group_umap_dir = dimred_dir / "groupwise_embeddings"
        group_umap_dir.mkdir(parents=True, exist_ok=True)
        plot_group_specific_umaps(
            adata,
            group_col=group_col,
            color_col=color_col_for_umap,
            out_dir=group_umap_dir,
        )

    logger.info(f"=== Done Scanpy pipeline: {analysis_name} ===")
