# src/gene_selection_uif.py

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional, Tuple

import pandas as pd
from logging import Logger

from .constants import DATA_DIR

def _load_deg_simple(
    path: Path,
    logger: Optional[Logger] = None,
) -> pd.DataFrame:
    """
    Load prepared_DEGs_Simple.csv and normalize column names.

    Expected columns (case-insensitive):
      - genes / gene / symbol  -> "gene"
      - log2foldchange / log2fc -> "log2fc"
      - p-value / pvalue        -> "pvalue"
      - aliases                 -> "aliases"
    """
    if logger:
        logger.info("Loading DEG simple file from %s", path)

    if not path.exists():
        raise FileNotFoundError(f"DEG simple file not found: {path}")

    df = pd.read_csv(path)
    df.columns = [c.lower().strip() for c in df.columns]

    # Map common names to standardized ones
    rename_map = {}
    if "genes" in df.columns:
        rename_map["genes"] = "gene"
    elif "gene" in df.columns:
        rename_map["gene"] = "gene"
    elif "symbol" in df.columns:
        rename_map["symbol"] = "gene"

    if "log2foldchange" in df.columns:
        rename_map["log2foldchange"] = "log2fc"
    if "log2fc" in df.columns:
        rename_map["log2fc"] = "log2fc"

    if "p-value" in df.columns:
        rename_map["p-value"] = "pvalue"
    if "pvalue" in df.columns:
        rename_map["pvalue"] = "pvalue"

    if "aliases" in df.columns:
        rename_map["aliases"] = "aliases"

    df = df.rename(columns=rename_map)

    # Must have at least gene
    if "gene" not in df.columns:
        raise ValueError(
            f"DEG file must contain a gene column (Genes, Gene, or symbol). "
            f"Found columns: {list(df.columns)}"
        )

    # Clean gene symbols
    df["gene"] = df["gene"].astype(str).str.upper().str.strip()

    # Make sure log2fc exists (if not, set to 0)
    if "log2fc" not in df.columns:
        df["log2fc"] = 0.0
    else:
        df["log2fc"] = pd.to_numeric(df["log2fc"], errors="coerce")

    # Make sure pvalue exists (optional)
    if "pvalue" in df.columns:
        df["pvalue"] = pd.to_numeric(df["pvalue"], errors="coerce")

    # Drop duplicate genes keeping first
    df = df.drop_duplicates(subset=["gene"], keep="first")

    if logger:
        logger.info("Loaded DEG simple table: %d genes", len(df))

    return df


def build_gene_lists_from_degs(
    output_dir: Path,
    deg_simple_path: Path,
    mode: Literal["all", "top"] = "all",
    top_up: int = 50,
    top_down: int = 50,
    logger: Optional[Logger] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build gene lists for downstream DepMap analysis from prepared_DEGs_Simple.csv.

    Parameters
    ----------
    deg_simple_path : Path
        Path to prepared_DEGs_Simple.csv (created by deg_prepare.py).
    mode : {"all", "top"}
        "all"  -> use all genes in deg_simple_path.
        "top"  -> use top_up most upregulated and top_down most downregulated
                  genes by log2fc.
    top_up : int
        Number of top UP genes (highest log2fc) to select when mode="top".
    top_down : int
        Number of top DOWN genes (lowest log2fc) to select when mode="top".
    logger : logging.Logger, optional
        Shared logger.

    Returns
    -------
    (df_full, df_selected) : Tuple[pd.DataFrame, pd.DataFrame]
        df_full     -> all genes with columns [gene, log2fc, pvalue, aliases (if present)]
        df_selected -> selected subset with same columns.
    """
    GENE_OUTDIR = output_dir / "DepMap_Genes"
    GENE_OUTDIR.mkdir(parents=True, exist_ok=True)

    df_full = _load_deg_simple(deg_simple_path, logger=logger)

    # Keep only relevant columns for downstream
    keep_cols = ["gene", "log2fc"]
    if "pvalue" in df_full.columns:
        keep_cols.append("pvalue")
    if "aliases" in df_full.columns:
        keep_cols.append("aliases")

    df_full = df_full[keep_cols]

    # --------------------------------------------
    # Selection logic
    # --------------------------------------------
    if mode == "all":
        if logger:
            logger.info("Gene selection mode: ALL (%d genes)", len(df_full))
        df_selected = df_full.copy()

    elif mode == "top":
        if "log2fc" not in df_full.columns:
            raise ValueError("Top-mode selection requires 'log2fc' column.")

        if logger:
            logger.info(
                "Gene selection mode: TOP (up=%d, down=%d)", top_up, top_down
            )

        # Top UP (largest positive log2fc)
        up = df_full.sort_values("log2fc", ascending=False).head(int(top_up))

        # Top DOWN (most negative log2fc)
        down = df_full.sort_values("log2fc", ascending=True).head(int(top_down))

        df_selected = (
            pd.concat([up, down])
            .drop_duplicates(subset=["gene"])
            .reset_index(drop=True)
        )
    else:
        raise ValueError("mode must be 'all' or 'top'")

    # --------------------------------------------
    # Save outputs in run-specific GENE_OUTDIR
    # --------------------------------------------
    full_path = GENE_OUTDIR / "InputGenes_full.csv"
    sel_path  = GENE_OUTDIR / "InputGenes_selected.csv"

    df_full.to_csv(full_path, index=False)
    df_selected.to_csv(sel_path, index=False)

    if logger:
        logger.info("Saved full gene list to %s (%d genes)", full_path, len(df_full))
        logger.info(
            "Saved selected gene list to %s (%d genes)", sel_path, len(df_selected)
        )

    print(f"💾 Full gene list saved → {full_path}  ({len(df_full)} genes)")
    print(f"💾 Selected gene list saved → {sel_path}  ({len(df_selected)} genes)")

    return df_full, df_selected
