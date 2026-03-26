from __future__ import annotations
from pathlib import Path
import gzip
import pandas as pd
import numpy as np
from scipy import io as spio
from scipy import sparse as sp_sparse
import anndata as ad
from .config_cli import logger

def _find_first_matching(dir_path: Path, patterns) -> Path | None:
    """
    Return the first file in dir_path matching any of the glob patterns in `patterns`.
    """
    for pat in patterns:
        matches = sorted(dir_path.glob(pat))
        if matches:
            return matches[0]
    return None


def _mmread_auto(path: Path):
    """
    Read a Matrix Market file, supporting optional .gz compression.
    """
    path = Path(path)
    if str(path).endswith(".gz"):
        with gzip.open(path, "rb") as f:
            return spio.mmread(f)
    else:
        return spio.mmread(str(path))


def load_10x_feature_barcode_matrix(tenx_dir: Path):
    """
    Load a single 10x feature-barcode matrix from a folder containing:
      - matrix.mtx[.gz]
      - barcodes.tsv[.gz]
      - features.tsv/genes.tsv[.gz]

    Returns
    -------
    AnnData
        Cells x genes matrix with barcodes in .obs and gene info in .var.
    """
    # local import to avoid circular dependencies
    import anndata as ad

    tenx_dir = Path(tenx_dir)
    if not tenx_dir.exists():
        raise FileNotFoundError(f"10X folder not found: {tenx_dir}")

    logger.info(f"Loading 10X feature-barcode matrix from: {tenx_dir}")

    matrix_path = _find_first_matching(
        tenx_dir,
        [
            "matrix.mtx",
            "matrix.mtx.gz",
            "*.matrix.mtx",
            "*.matrix.mtx.gz",
            "*.mtx",
            "*.mtx.gz",
        ],
    )
    if matrix_path is None:
        raise FileNotFoundError(
            f"No matrix.mtx[.gz] file found in {tenx_dir}."
        )

    barcodes_path = _find_first_matching(
        tenx_dir,
        [
            "barcodes.tsv",
            "barcodes.tsv.gz",
            "*barcodes.tsv",
            "*barcodes.tsv.gz",
            "barcode.tsv",
            "barcode.tsv.gz",
            "*barcode.tsv",
            "*barcode.tsv.gz",
        ],
    )
    if barcodes_path is None:
        raise FileNotFoundError(
            f"No barcodes.tsv[.gz] file found in {tenx_dir}."
        )

    features_path = _find_first_matching(
        tenx_dir,
        [
            "features.tsv",
            "features.tsv.gz",
            "*features.tsv",
            "*features.tsv.gz",
            "genes.tsv",
            "genes.tsv.gz",
            "*genes.tsv",
            "*genes.tsv.gz",
        ],
    )
    if features_path is None:
        raise FileNotFoundError(
            f"No features.tsv[.gz] or genes.tsv[.gz] file found in {tenx_dir}."
        )

    logger.info(f"matrix:   {matrix_path.name}")
    logger.info(f"barcodes: {barcodes_path.name}")
    logger.info(f"features: {features_path.name}")

    # Read matrix
    M = _mmread_auto(matrix_path)
    if not sp_sparse.issparse(M):
        M = sp_sparse.coo_matrix(M)
    M = M.tocsr()
    X = M.T  # cells x genes

    # Read barcodes
    barcodes_df = pd.read_csv(barcodes_path, sep="\t", header=None, compression="infer")
    barcodes = barcodes_df.iloc[:, 0].astype(str).values

    # Read features / genes
    feat_df = pd.read_csv(features_path, sep="\t", header=None, compression="infer")
    ncols = feat_df.shape[1]
    colnames = []
    if ncols >= 1:
        colnames.append("feature_id")
    if ncols >= 2:
        colnames.append("feature_name")
    if ncols >= 3:
        colnames.append("feature_type")
    while len(colnames) < ncols:
        colnames.append(f"extra_{len(colnames)}")
    feat_df.columns = colnames

    gene_ids = feat_df["feature_id"].astype(str).values
    if "feature_name" in feat_df.columns:
        gene_names = feat_df["feature_name"].astype(str).values
    else:
        gene_names = gene_ids

    # Build AnnData
    adata_ = ad.AnnData(X=X)
    adata_.obs_names = barcodes
    adata_.obs["barcode"] = barcodes
    adata_.var_names = gene_names
    adata_.var["feature_id"] = gene_ids
    adata_.var["gene_symbol"] = gene_names
    if "feature_type" in feat_df.columns:
        adata_.var["feature_type"] = feat_df["feature_type"].astype(str).values
    adata_.var_names_make_unique()

    logger.info(f"Loaded AnnData from 10x: {adata_}")
    return adata_
