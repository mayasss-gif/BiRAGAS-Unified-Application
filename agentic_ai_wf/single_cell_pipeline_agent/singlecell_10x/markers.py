from __future__ import annotations
# markers.py

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scanpy as sc
from .config_cli import logger


def compute_celltype_markers(
    adata,
    celltype_col: str,
    out_dir: Path,
    analysis_name: str,
    n_markers_per_type: int = 50,
    reference_dir: Path | None = None,
):
    """
    Celltype-specific markers: for each cell type, DE vs all other cell types.

    Writes:
      - global rankplot / heatmap / dotplot
      - celltype_marker_genes_{celltype_col}_ALL.csv
      - per-celltype CSVs + dotplots/rankplots
      - optional copy of ALL markers in reference_dir
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"[CT-MARKERS] compute_celltype_markers called for column '{celltype_col}'.")

    if celltype_col not in adata.obs.columns:
        logger.info(f"[CT-MARKERS] '{celltype_col}' not in obs → skip.")
        return

    sc.settings.figdir = out_dir

    if not pd.api.types.is_categorical_dtype(adata.obs[celltype_col]):
        adata.obs[celltype_col] = adata.obs[celltype_col].astype("category")

    celltypes = adata.obs[celltype_col].cat.categories.tolist()
    logger.info(f"[CT-MARKERS] Found {len(celltypes)} cell types: {celltypes}")

    heatmap_height = max(8.0, 0.6 * len(celltypes))

    dotplot_dir = out_dir / "sc_dot_plot_vis"
    rankplot_dir = out_dir / "sc_rank_plot_vis"
    dotplot_dir.mkdir(parents=True, exist_ok=True)
    rankplot_dir.mkdir(parents=True, exist_ok=True)

    try:
        logger.info(f"[CT-MARKERS] Running rank_genes_groups by '{celltype_col}' (wilcoxon)...")
        sc.tl.rank_genes_groups(
            adata,
            groupby=celltype_col,
            method="wilcoxon",
            n_genes=n_markers_per_type,
        )

        sc.pl.rank_genes_groups(
            adata,
            n_genes=20,
            sharey=False,
            show=False,
            save=f"_{analysis_name}_celltype_markers_rankplot.png",
        )

        sc.pl.rank_genes_groups_heatmap(
            adata,
            n_genes=10,
            show=False,
            save=f"_{analysis_name}_celltype_markers_heatmap.png",
            figsize=(10, heatmap_height),
        )
        sc.pl.rank_genes_groups_dotplot(
            adata,
            n_genes=10,
            show=False,
            save=f"_{analysis_name}_celltype_markers_dotplot.png",
        )

        try:
            markers_all = sc.get.rank_genes_groups_df(adata, None)
        except Exception:
            rg = adata.uns["rank_genes_groups"]
            groups = rg["names"].dtype.names
            rows = []
            for g in groups:
                names = rg["names"][g]
                scores = rg["scores"][g]
                pvals_adj = rg["pvals_adj"][g]
                logfc = rg.get("logfoldchanges", {}).get(g, None)
                for idx, gene in enumerate(names):
                    row = {
                        "group": g,
                        "names": gene,
                        "scores": scores[idx] if scores is not None else np.nan,
                        "pvals_adj": pvals_adj[idx] if pvals_adj is not None else np.nan,
                        "rank": idx + 1,
                    }
                    if logfc is not None:
                        row["logfoldchanges"] = logfc[idx]
                    rows.append(row)
            markers_all = pd.DataFrame(rows)

        if "logfoldchanges" in markers_all.columns:
            markers_all["direction_simple"] = np.where(
                markers_all["logfoldchanges"] > 0,
                "upregulated",
                np.where(
                    markers_all["logfoldchanges"] < 0,
                    "downregulated",
                    "no_expression_change",
                ),
            )

        all_file = out_dir / f"celltype_marker_genes_{celltype_col}_ALL.csv"
        markers_all.to_csv(all_file, index=False)
        logger.info(f"[CT-MARKERS] Wrote all celltype markers: {all_file}")

        if reference_dir is not None:
            reference_dir = Path(reference_dir)
            reference_dir.mkdir(parents=True, exist_ok=True)
            ref_file = reference_dir / f"{analysis_name}_celltype_markers_{celltype_col}_ALL.csv"
            markers_all.to_csv(ref_file, index=False)
            logger.info(f"[CT-MARKERS] Copied celltype markers to reference dir: {ref_file}")

        for ct in celltypes:
            sub = markers_all[markers_all["group"] == ct].copy()
            ct_safe = str(ct).replace(" ", "_").replace("/", "_")
            ct_file = out_dir / f"celltype_marker_genes_{celltype_col}_{ct_safe}.csv"
            sub.to_csv(ct_file, index=False)
            logger.info(f"[CT-MARKERS] Wrote markers for celltype '{ct}': {ct_file}")

            if "pvals_adj" in sub.columns:
                sub_sorted = sub.sort_values("pvals_adj")
            else:
                sub_sorted = sub.sort_values("rank")
            top_genes = sub_sorted["names"].head(min(n_markers_per_type, len(sub_sorted))).tolist()
            if not top_genes:
                continue

            sc.settings.figdir = dotplot_dir
            try:
                sc.pl.dotplot(
                    adata,
                    var_names=top_genes,
                    groupby=celltype_col,
                    show=False,
                    save=f"_{analysis_name}_dotplot_{ct_safe}.png",
                )
            except Exception as e:
                logger.warning(f"[CT-MARKERS] Dotplot failed for {ct}: {e}")

            sc.settings.figdir = rankplot_dir
            try:
                sc.pl.rank_genes_groups(
                    adata,
                    groups=[ct],
                    n_genes=20,
                    sharey=False,
                    show=False,
                    save=f"_{analysis_name}_rankplot_{ct_safe}.png",
                )
            except Exception as e:
                logger.warning(f"[CT-MARKERS] Rankplot (single celltype) failed for {ct}: {e}")

        sc.settings.figdir = out_dir

    except Exception as e:
        logger.warning(f"[CT-MARKERS] Failed to compute celltype markers: {e}")
