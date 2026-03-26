# src/depmap_gene_layers_uif.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Dict

import pandas as pd
from logging import Logger


from .depmap_celllines_uif import DepMapCelllineContext



@dataclass
class GeneLayerResult:
    """
    Container for selected-gene DepMap matrices.
    """
    selected_genes: List[str]
    present_per_layer: Dict[str, List[str]]
    effect_matrix_path: Path
    dep_matrix_path: Path
    expr_matrix_path: Path
    cnv_matrix_path: Path
    gene_presence_path: Path


def _load_selected_genes(path: Path, logger: Optional[Logger] = None) -> List[str]:
    """
    Load selected genes from InputGenes_selected.csv.
    Expects a 'gene' column; falls back to 'genes' or 'symbol' if needed.
    """
    if logger:
        logger.info("Loading selected genes from %s", path)

    if not path.exists():
        raise FileNotFoundError(f"Selected genes file not found: {path}")

    df = pd.read_csv(path)
    cols_lower = {c.lower(): c for c in df.columns}

    gene_col = None
    for key in ["gene", "genes", "symbol"]:
        if key in cols_lower:
            gene_col = cols_lower[key]
            break

    if gene_col is None:
        raise ValueError(
            f"Selected genes file must contain a gene column (gene/genes/symbol). "
            f"Found columns: {list(df.columns)}"
        )

    genes = (
        df[gene_col]
        .astype(str)
        .str.strip()
        .str.upper()
        .dropna()
        .unique()
        .tolist()
    )

    if logger:
        logger.info("Loaded %d unique selected genes.", len(genes))

    return genes


def _standardize_gene_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize matrix column names to uppercase strings for matching with gene symbols.
    Index (ModelID) is preserved.
    """
    df = df.copy()
    df.columns = [str(c).upper() for c in df.columns]
    return df


def build_gene_layer_matrices(
    ctx: DepMapCelllineContext,
    selected_genes_path: Path,
    output_dir: Path,
    logger: Optional[Logger] = None,
) -> GeneLayerResult:
    """
    Subset DepMap effect/dep/expression/CNV matrices to a selected gene list
    and save them as CSVs.

    Parameters
    ----------
    ctx : DepMapCelllineContext
        Context produced by run_depmap_cellline_setup().
    selected_genes_path : Path
        Path to InputGenes_selected.csv (or equivalent).
    logger : logging.Logger, optional
        Shared logger.

    Returns
    -------
    GeneLayerResult
        Paths to saved matrices and presence summary.
    """
    gene_layers_dir = output_dir / "DepMap_GeneLayers"
    gene_layers_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1) Load selected genes
    # ------------------------------------------------------------------
    selected_genes = _load_selected_genes(selected_genes_path, logger=logger)

    # ------------------------------------------------------------------
    # 2) Standardize gene columns in matrices
    # ------------------------------------------------------------------
    effect_df = _standardize_gene_columns(ctx.effect_df)
    dep_df    = _standardize_gene_columns(ctx.dep_df)
    expr_df   = _standardize_gene_columns(ctx.expr_df)
    cnv_df    = _standardize_gene_columns(ctx.cnv_df)

    # ------------------------------------------------------------------
    # 3) Intersect with available genes in each layer
    # ------------------------------------------------------------------
    sel_set = set(selected_genes)

    effect_genes = sorted(sel_set & set(effect_df.columns))
    dep_genes    = sorted(sel_set & set(dep_df.columns))
    expr_genes   = sorted(sel_set & set(expr_df.columns))
    cnv_genes    = sorted(sel_set & set(cnv_df.columns))

    if logger:
        logger.info("Selected genes: %d", len(selected_genes))
        logger.info("Genes with Chronos effect data: %d", len(effect_genes))
        logger.info("Genes with dependency data: %d", len(dep_genes))
        logger.info("Genes with expression data: %d", len(expr_genes))
        logger.info("Genes with CNV data: %d", len(cnv_genes))

    # ------------------------------------------------------------------
    # 4) Subset matrices to selected genes
    # ------------------------------------------------------------------
    effect_sub = effect_df[effect_genes] if effect_genes else effect_df.iloc[:, :0].copy()
    dep_sub    = dep_df[dep_genes]       if dep_genes else dep_df.iloc[:, :0].copy()
    expr_sub   = expr_df[expr_genes]     if expr_genes else expr_df.iloc[:, :0].copy()
    cnv_sub    = cnv_df[cnv_genes]       if cnv_genes else cnv_df.iloc[:, :0].copy()

    # ------------------------------------------------------------------
    # 5) Save matrices
    # ------------------------------------------------------------------
    effect_path = gene_layers_dir / "ChronosEffect_SelectedGenes_matrix.csv"
    dep_path    = gene_layers_dir / "Dependency_SelectedGenes_matrix.csv"
    expr_path   = gene_layers_dir / "Expression_SelectedGenes_matrix.csv"
    cnv_path    = gene_layers_dir / "CNV_SelectedGenes_matrix.csv"

    effect_sub.to_csv(effect_path)
    dep_sub.to_csv(dep_path)
    expr_sub.to_csv(expr_path)
    cnv_sub.to_csv(cnv_path)

    if logger:
        logger.info("Saved Chronos effect matrix to %s", effect_path)
        logger.info("Saved dependency matrix to %s", dep_path)
        logger.info("Saved expression matrix to %s", expr_path)
        logger.info("Saved CNV matrix to %s", cnv_path)

    # ------------------------------------------------------------------
    # 6) Gene presence summary across layers
    # ------------------------------------------------------------------
    presence = []
    for g in sorted(selected_genes):
        presence.append(
            {
                "gene": g,
                "in_effect": g in effect_genes,
                "in_dependency": g in dep_genes,
                "in_expression": g in expr_genes,
                "in_cnv": g in cnv_genes,
            }
        )

    presence_df = pd.DataFrame(presence)
    presence_path = gene_layers_dir / "SelectedGenes_LayerPresence_summary.csv"
    presence_df.to_csv(presence_path, index=False)

    if logger:
        logger.info("Saved gene presence summary to %s", presence_path)

    return GeneLayerResult(
        selected_genes=selected_genes,
        present_per_layer={
            "effect": effect_genes,
            "dependency": dep_genes,
            "expression": expr_genes,
            "cnv": cnv_genes,
        },
        effect_matrix_path=effect_path,
        dep_matrix_path=dep_path,
        expr_matrix_path=expr_path,
        cnv_matrix_path=cnv_path,
        gene_presence_path=presence_path,
    )
