from __future__ import annotations

from pathlib import Path
import gzip
import pandas as pd
import scanpy as sc
import anndata as ad
import scipy.sparse as sp

from .utils import die


# ============================================================
# helpers
# ============================================================

def _open_text(path: Path):
    return gzip.open(path, "rt") if path.suffix == ".gz" else open(path, "r")


def _maybe_read_tsv(path: Path) -> pd.DataFrame:
    if path.suffix == ".gz":
        return pd.read_csv(path, sep="\t", header=None, compression="gzip")
    return pd.read_csv(path, sep="\t", header=None)


def _find_file(dataset_dir: Path, gsm: str, key: str) -> Path:
    """
    key ∈ {barcodes, genes/features, matrix, cell_identities}
    """
    candidates = []
    for f in dataset_dir.iterdir():
        if not f.name.startswith(gsm):
            continue

        name = f.name.lower()

        if key == "barcodes" and "barcodes" in name:
            candidates.append(f)
        elif key in ("genes", "features") and (
            "genes" in name or "features" in name
        ):
            candidates.append(f)
        elif key == "matrix" and "matrix" in name and "mtx" in name:
            candidates.append(f)
        elif key == "cell_identities" and "cell_identities" in name:
            candidates.append(f)

    if not candidates:
        die(f"Missing {key} file for {gsm} in {dataset_dir}")

    # prefer gz + shorter name
    candidates = sorted(candidates, key=lambda p: (p.suffix != ".gz", len(p.name)))
    return candidates[0]


# ============================================================
# main loader
# ============================================================

def read_gsm_10x_like(dataset_dir: str, gsm: str) -> ad.AnnData:
    """
    Robust loader for GEO / 10x-like / Perturb-seq datasets.

    Guarantees:
    - X is cells × genes
    - obs rows == X.shape[0]
    - var rows == X.shape[1]
    - Detects and fixes transposed matrices
    """

    d = Path(dataset_dir)

    barcodes_f = _find_file(d, gsm, "barcodes")
    genes_f = _find_file(d, gsm, "genes")
    matrix_f = _find_file(d, gsm, "matrix")

    # --------------------------------------------------------
    # read matrix
    # --------------------------------------------------------
    X = sc.read_mtx(str(matrix_f)).X
    if not sp.issparse(X):
        X = sp.csr_matrix(X)

    # --------------------------------------------------------
    # read barcodes
    # --------------------------------------------------------
    bc_df = _maybe_read_tsv(barcodes_f)
    barcodes = bc_df.iloc[:, 0].astype(str).tolist()

    # --------------------------------------------------------
    # read genes / features
    # --------------------------------------------------------
    gn_df = _maybe_read_tsv(genes_f)

    if gn_df.shape[1] >= 2:
        gene_names = gn_df.iloc[:, 1].astype(str).tolist()
        gene_ids = gn_df.iloc[:, 0].astype(str).tolist()
    else:
        gene_names = gn_df.iloc[:, 0].astype(str).tolist()
        gene_ids = gene_names

    n_cells = len(barcodes)
    n_genes = len(gene_names)

    # --------------------------------------------------------
    # CRITICAL: orientation detection
    # --------------------------------------------------------
    if X.shape == (n_genes, n_cells):
        print(f"[INFO] {gsm}: detected genes×cells matrix → transposing")
        X = X.T

    if X.shape != (n_cells, n_genes):
        raise ValueError(
            f"{gsm}: matrix shape {X.shape} incompatible with "
            f"barcodes={n_cells}, genes={n_genes}"
        )

    # --------------------------------------------------------
    # build obs / var (unique var names to avoid anndata warning)
    # --------------------------------------------------------
    obs = pd.DataFrame(index=barcodes)
    obs["GSM"] = gsm

    # Make var names unique (pandas <2.0 lacks Index.make_unique)
    seen: dict[str, int] = {}
    unique_names = []
    for g in gene_names:
        if g in seen:
            seen[g] += 1
            unique_names.append(f"{g}-{seen[g]}")
        else:
            seen[g] = 0
            unique_names.append(g)
    var = pd.DataFrame(index=unique_names)
    var["gene_ids"] = gene_ids

    adata = ad.AnnData(X=X, obs=obs, var=var)

    # --------------------------------------------------------
    # optional cell identities
    # --------------------------------------------------------
    try:
        ci_f = _find_file(d, gsm, "cell_identities")
        ci = pd.read_csv(ci_f, compression="infer")

        barcode_cols = [c for c in ci.columns if "barcode" in c.lower()]
        if barcode_cols:
            bc_col = barcode_cols[0]
            ci[bc_col] = ci[bc_col].astype(str)
            ci = ci.set_index(bc_col)
            adata.obs = adata.obs.join(ci, how="left")
        else:
            adata.uns["cell_identities_raw"] = ci.to_dict(orient="list")

    except Exception:
        pass

    return adata

