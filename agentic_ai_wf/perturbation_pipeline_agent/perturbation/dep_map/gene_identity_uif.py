# src/gene_identity_uif.py

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import plotly.express as px
from logging import Logger


# Try display() if in notebook; otherwise fallback to print
try:
    from IPython.display import display  # type: ignore
except Exception:  # pragma: no cover
    def display(x):  # type: ignore
        print(x)


# ---------------------------------------------------------------------
# Paths for DepMap Gene reference and Gene QC outputs
# ---------------------------------------------------------------------
from .constants import DATA_DIR
from ..plotly_mpl_export import save_plotly_png_with_mpl

def save_plotly(fig, png_path: Path, html_path: Path, scale: int = 2) -> None:
    """
    Save a Plotly figure as PNG and HTML.
    PNG export requires kaleido; if not available, only HTML is saved.
    """
    # Always save HTML
    html_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(html_path))

    # Try PNG (optional)
    try:
        png_path.parent.mkdir(parents=True, exist_ok=True)
        ok = save_plotly_png_with_mpl(fig, png_path, scale=scale)
        if not ok:
            fig.write_image(str(png_path), scale=scale)
    except Exception as e:
        print(f"⚠️ Could not save PNG for {png_path.name}: {e}")


def run_gene_identity_check(
    output_dir: Path,
    logger: Logger,
    gene_ref_path: Optional[Path] = None,
    selected_genes_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Run HGNC symbol approval check on selected gene list.

    - Reads Gene.csv (DepMap gene reference) for HGNC-approved symbols.
    - Reads InputGenes_selected.csv from the run-specific GENES_DIR.
    - Produces:
        * GeneIdentity_Check.csv (table of gene + is_approved_symbol)
        * HGNC symbol status pie chart (HTML + PNG where possible)

    Parameters
    ----------
    output_dir : Path
        Output directory where final tables will be written.
    logger : logging.Logger, optional
        Shared logger to write info/warnings.
    gene_ref_path : Path, optional
        Path to DepMap Gene.csv; defaults to DepMap_Repository/Gene.csv.
    selected_genes_path : Path, optional
        Path to InputGenes_selected.csv; defaults to GENES_DIR/InputGenes_selected.csv.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: ['gene', 'is_approved_symbol'].
    """
    OUTDIR = output_dir / "DepMap_Genes"
    OUTDIR.mkdir(parents=True, exist_ok=True)

    if gene_ref_path is None:
        gene_ref_path = DATA_DIR / "Gene.csv"
    if selected_genes_path is None:
        # Use run-specific selected genes from GENES_DIR
        selected_genes_path = OUTDIR / "InputGenes_selected.csv"

    if logger:
        logger.info("Running gene identity check.")
        logger.info("Gene reference: %s", gene_ref_path)
        logger.info("Selected genes file: %s", selected_genes_path)

    if not gene_ref_path.exists():
        msg = f"Gene reference file not found: {gene_ref_path}"
        if logger:
            logger.error(msg)
        raise FileNotFoundError(msg)

    if not selected_genes_path.exists():
        msg = f"Selected genes file not found: {selected_genes_path}"
        if logger:
            logger.error(msg)
        raise FileNotFoundError(msg)

    # -------------------------------------------------------------
    # Load DepMap gene reference (only relevant columns)
    # -------------------------------------------------------------
    gene_cols = pd.read_csv(gene_ref_path, nrows=0).columns.tolist()
    use_genecols = [
        c
        for c in ["symbol", "entrez_id", "alias_symbol", "prev_symbol"]
        if c in gene_cols
    ]

    if not use_genecols:
        raise ValueError(
            f"No expected gene columns found in Gene reference. "
            f"Available columns: {gene_cols}"
        )

    genes_ref = pd.read_csv(gene_ref_path, usecols=use_genecols)
    if "symbol" not in genes_ref.columns:
        raise ValueError("Gene reference must contain a 'symbol' column.")

    # Set of approved symbols (uppercased)
    approved = set(genes_ref["symbol"].dropna().astype(str).str.upper())

    # -------------------------------------------------------------
    # Load selected gene list from previous step
    # -------------------------------------------------------------
    sel_df = pd.read_csv(selected_genes_path)
    # Expect a 'gene' column (from your selection pipeline)
    if "gene" not in sel_df.columns:
        # try some common alternatives
        possible_cols = [
            c for c in sel_df.columns if c.lower() in ["gene", "genes", "symbol"]
        ]
        if possible_cols:
            sel_df = sel_df.rename(columns={possible_cols[0]: "gene"})
        else:
            raise ValueError(
                f"Selected genes file must contain a 'gene' column. "
                f"Found columns: {list(sel_df.columns)}"
            )

    selected_series = (
        sel_df["gene"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    selected_status = pd.DataFrame(
        {
            "gene": selected_series,
            "is_approved_symbol": selected_series.isin(approved),
        }
    )

    n_approved = int(selected_status["is_approved_symbol"].sum())
    n_total = len(selected_status)

    print(f"✅ Approved HGNC symbols: {n_approved}/{n_total}")
    display(selected_status.head(20))

    # Save table
    out_csv = OUTDIR / "GeneIdentity_Check.csv"
    selected_status.to_csv(out_csv, index=False)

    if logger:
        logger.info(
            "Gene identity check: %d/%d approved symbols. Saved table to %s",
            n_approved,
            n_total,
            out_csv,
        )

    # -------------------------------------------------------------
    # Pie chart for approved vs unrecognized
    # -------------------------------------------------------------
    plotting_df = selected_status.replace(
        {"is_approved_symbol": {True: "Approved", False: "Unrecognized"}}
    )

    fig3 = px.pie(
        plotting_df,
        names="is_approved_symbol",
        title="HGNC symbol status in selected genes",
    )

    # ❗ DO NOT CALL fig3.show() -> avoids WSL browser issues and blocking
    print("ℹ️ Skipping fig.show() to avoid WSL browser issue. HTML/PNG will still be saved.")

    png_path = OUTDIR / "figs" / "hgnc_symbol_status.png"
    html_path = OUTDIR / "html" / "hgnc_symbol_status.html"
    save_plotly(fig3, png_path, html_path, scale=2)

    if logger:
        logger.info(
            "Saved HGNC symbol status pie chart to %s and %s",
            png_path,
            html_path,
        )

    return selected_status
