from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union
from logging import Logger
import re  

import pandas as pd




@dataclass
class DegPrepareResult:
    """Container holding paths to DEG files."""
    raw_file: Path
    prepared_long: Path
    prepared_simple: Path


def _load_table(path: Path) -> pd.DataFrame:
    """Load CSV / TSV / Excel into a DataFrame."""
    ext = path.suffix.lower()

    if ext == ".csv":
        return pd.read_csv(path)
    if ext in {".tsv", ".txt"}:
        return pd.read_csv(path, sep="\t")
    if ext in {".xlsx", ".xls", ".xlsm"}:
        return pd.read_excel(path)

    raise ValueError(f"Unsupported file extension for DEG file: {ext}")


def _detect_pvalue_columns(df: pd.DataFrame) -> list[str]:
    """
    Heuristically detect p-value columns by name.

    Matches things like: 'pvalue', 'p-value', 'p_value', 'p.val', 'pval',
    or columns ending in '_p' / '.p'.
    """
    candidates: list[str] = []

    for col in df.columns:
        name = col.lower()
        name_nospace = re.sub(r"\s+", "", name)

        patterns = [
            "pvalue",
            "p-value",
            "p_value",
            "p.val",
            "pval",
        ]

        if any(pat in name_nospace for pat in patterns):
            candidates.append(col)
            continue

        # weak fallback: columns ending in '_p' or '.p'
        if re.search(r"(_p$|\.p$)", name_nospace):
            candidates.append(col)

    return candidates


def prepare_deg_files(
    raw_deg_path: Union[str, Path],
    gene_col: str = "Gene",
    lfc_col: str = "Patient_LFC_mean",
    trend_col: str = "Patient_LFC_Trend",
    synonyms_col: str = "HGNC_Synonyms",
    pval_col: Optional[str] = None,
    logger: Optional[Logger] = None,
    output_dir: Optional[Union[str, Path]] = None,
) -> DegPrepareResult:
    """
    Create the two prepared DEG files used downstream.

    1) prepared_DEGs_File.csv
       Columns: Gene, Log2FC, HGNC_Synonyms, Patient_LFC_Trend

    2) prepared_DEGs_Simple.csv
       Columns: Genes, Log2FoldChange, p-value, Aliases

    Args:
        output_dir: Optional output directory. If provided, creates a "Prepared_DEGs"
                   subdirectory inside it and stores results there. If None, uses
                   the default PREPARED_DIR from paths module.
    """
    
    raw_deg_path = Path(raw_deg_path)

    # Determine output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prepared_dir = output_dir / "Prepared_DEGs"
    prepared_dir.mkdir(parents=True, exist_ok=True)
    if logger:
        logger.info("Using user-specified output directory: %s", prepared_dir)
    
    if logger:
        logger.info("Preparing DEG files from: %s", raw_deg_path)

    df = _load_table(raw_deg_path)

    # ------------------------------------------------------------------
    # 1) Decide which p-value column to use
    # ------------------------------------------------------------------
    if pval_col is None:
        # Auto-detect p-value-like columns by keyword
        pval_cols = _detect_pvalue_columns(df)

        if not pval_cols:
            # Strict: we do NOT silently skip p-values
            raise ValueError(
                "Could not automatically detect a p-value column in the raw DEG file.\n"
                f"Available columns: {list(df.columns)}\n\n"
                "Expected a column whose name contains something like 'pvalue', "
                "'p-value', 'p_value', 'p.val', or 'pval'.\n"
                "Please either rename the p-value column accordingly or call "
                "prepare_deg_files(..., pval_col='YourPvalueColumn')."
            )

        # Convert to numeric and compute row-wise minimum (best/lowest p-value)
        numeric_pvals = df[pval_cols].apply(pd.to_numeric, errors="coerce")
        df["Combined_pvalue"] = numeric_pvals.min(axis=1, skipna=True)
        pval_col = "Combined_pvalue"

        if logger:
            logger.info("Built combined p-value from columns: %s", pval_cols)
    else:
        # User gave a specific p-value column
        if pval_col not in df.columns:
            raise ValueError(
                f"Requested p-value column '{pval_col}' not found in raw DEG file.\n"
                f"Available columns: {list(df.columns)}"
            )

    # ------------------------------------------------------------------
    # 2) Build prepared_DEGs_File.csv (long format)
    # ------------------------------------------------------------------
    needed_long = [gene_col, lfc_col, synonyms_col, trend_col]
    missing_long = [c for c in needed_long if c not in df.columns]
    if missing_long:
        raise ValueError(f"Missing columns for long DEG file: {missing_long}")

    long_df = (
        df[needed_long]
        .rename(
            columns={
                gene_col: "Gene",
                lfc_col: "Log2FC",
                synonyms_col: "HGNC_Synonyms",
                trend_col: "Patient_LFC_Trend",
            }
        )
        .dropna(subset=["Gene"])
    )

    # Drop duplicate genes, keep first occurrence
    long_df = long_df.drop_duplicates(subset=["Gene"], keep="first")

    prepared_long_path = prepared_dir / "prepared_DEGs_File.csv"
    long_df.to_csv(prepared_long_path, index=False)

    if logger:
        logger.info("Wrote prepared long DEG file: %s", prepared_long_path)

    # ------------------------------------------------------------------
    # 3) Build prepared_DEGs_Simple.csv (simple format)
    # ------------------------------------------------------------------
    needed_simple = [gene_col, lfc_col, pval_col, synonyms_col]  # type: ignore[list-item]
    missing_simple = [c for c in needed_simple if c not in df.columns]
    if missing_simple:
        raise ValueError(f"Missing columns for simple DEG file: {missing_simple}")

    simple_df = (
        df[needed_simple]
        .rename(
            columns={
                gene_col: "Genes",
                lfc_col: "Log2FoldChange",
                pval_col: "p-value",      # type: ignore[arg-type]
                synonyms_col: "Aliases",
            }
        )
        .dropna(subset=["Genes"])
    )

    simple_df = simple_df.drop_duplicates(subset=["Genes"], keep="first")

    prepared_simple_path = prepared_dir / "prepared_DEGs_Simple.csv"
    simple_df.to_csv(prepared_simple_path, index=False)

    if logger:
        logger.info("Wrote prepared simple DEG file: %s", prepared_simple_path)


    return DegPrepareResult(
        raw_file=raw_deg_path,
        prepared_long=prepared_long_path,
        prepared_simple=prepared_simple_path,
    )


def build_gene_list_from_prepared_simple(
    prepared_simple_path: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
    logger: Optional[Logger] = None,
) -> Path:
    """
    Build a unique gene list from prepared_DEGs_Simple.csv and save to
    GENES_DIR / 'InputGenes_selected.csv'.
    """
    prepared_simple_path = Path(prepared_simple_path)

    if logger:
        logger.info("Building gene list from: %s", prepared_simple_path)

    if not prepared_simple_path.exists():
        msg = f"DEG simple file not found: {prepared_simple_path}"
        if logger:
            logger.error(msg)
        raise FileNotFoundError(msg)

    df = pd.read_csv(prepared_simple_path)

    # Normalise column names
    df.columns = [c.lower().strip() for c in df.columns]

    # Find a gene column: genes / gene / symbol
    gene_col = None
    for candidate in ["genes", "gene", "symbol"]:
        if candidate in df.columns:
            gene_col = candidate
            break

    if gene_col is None:
        msg = (
            f"No gene-like column in {prepared_simple_path}. "
            f"Columns: {list(df.columns)}"
        )
        if logger:
            logger.error(msg)
        raise ValueError(
            "DEG file must contain a 'Genes', 'Gene' or 'symbol' column."
        )

    # Clean gene list
    df["gene"] = df[gene_col].astype(str).str.upper().str.strip()

    # Remove duplicates
    df_unique = df[["gene"]].drop_duplicates()

    # Save to DepMap_Genes directory (run-specific)
    output_dir = Path(output_dir)
    outpath = output_dir / "Prepared_DEGs" / "InputGenes_selected.csv"
    df_unique.to_csv(outpath, index=False)

    if logger:
        logger.info("Saved gene list → %s", outpath)
        logger.info("Number of unique genes: %d", len(df_unique))

    print(df_unique.head())
    logger.info(
        f"\nExtracted {len(df_unique)} unique genes from prepared_DEGs_Simple.csv"
    )

    return outpath
