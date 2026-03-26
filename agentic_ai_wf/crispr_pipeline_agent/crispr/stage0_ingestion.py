#!/usr/bin/env python3
import argparse
import hashlib
import sys
from pathlib import Path
from typing import Dict, Optional, List

import numpy as np
import pandas as pd
import scanpy as sc
from scipy.io import mmread
from scipy.sparse import csr_matrix


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(block_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()

def mmread_maybe_gz(path: Path) -> csr_matrix:
    if str(path).endswith(".gz"):
        import gzip, io
        with gzip.open(path, "rb") as f:
            data = f.read()
        return mmread(io.BytesIO(data)).tocsr()
    return mmread(path).tocsr()

def read_table(path: Path, **kwargs) -> pd.DataFrame:
    # pandas can read .gz directly
    return pd.read_csv(path, **kwargs)

def locate_geo_files(gse_dir: Path, gsm_id: str) -> Dict[str, Path]:
    # allow .gz or plain
    def find_one(patterns: List[str]) -> Optional[Path]:
        for pat in patterns:
            hits = list(gse_dir.glob(f"{gsm_id}*{pat}*"))
            if hits:
                return hits[0]
        return None

    matrix = find_one(["matrix.mtx", "matrix.mtx.txt"])
    genes = find_one(["genes.tsv"])
    barcodes = find_one(["barcodes.tsv"])
    cell_id = find_one(["cell_identities.csv"])

    if matrix is None or genes is None or barcodes is None:
        raise FileNotFoundError(f"{gsm_id}: missing matrix/genes/barcodes in {gse_dir}")

    return {"matrix": matrix, "genes": genes, "barcodes": barcodes, "cell_identities": cell_id}

def orient_matrix(X: csr_matrix, n_genes: int, n_cells: int, gsm_id: str) -> csr_matrix:
    if X.shape == (n_cells, n_genes):
        return X
    if X.shape == (n_genes, n_cells):
        return X.T.tocsr()
    raise ValueError(f"[{gsm_id}] Matrix shape {X.shape} mismatch vs cells={n_cells}, genes={n_genes}")

def join_cell_identities_into_obs(adata: sc.AnnData, cell_id_path: Optional[Path]) -> sc.AnnData:
    if cell_id_path is None or (not cell_id_path.exists()):
        return adata

    meta = pd.read_csv(cell_id_path)
    lower = {c.lower(): c for c in meta.columns}

    # barcode column (GEO uses: "cell BC")
    bc_col = None
    for cand in ["cell bc", "barcode", "barcodes", "cell_barcode", "cell bc "]:
        if cand in lower:
            bc_col = lower[cand]
            break

    if bc_col is None:
        # can't join reliably
        adata.uns["cell_identities_table"] = meta
        return adata

    meta = meta.copy()
    meta[bc_col] = meta[bc_col].astype(str)
    meta = meta.set_index(bc_col)

    # join by barcode (adata.obs_names are barcodes)
    adata.obs = adata.obs.join(meta, how="left")
    return adata

def sanitize_obs_for_h5ad(adata: sc.AnnData) -> sc.AnnData:
    obs = adata.obs.copy()
    for col in obs.columns:
        s = obs[col]
        if pd.api.types.is_numeric_dtype(s):
            continue
        if pd.api.types.is_bool_dtype(s):
            obs[col] = s.fillna(False).astype(np.int8)
            continue
        s = s.astype("object")
        s = s.where(~pd.isna(s), "")
        obs[col] = s.map(lambda x: "" if x is None else str(x))
    adata.obs = obs
    return adata

def load_geo_to_anndata(gse_dir: Path, gsm_id: str) -> sc.AnnData:
    paths = locate_geo_files(gse_dir, gsm_id)

    X = mmread_maybe_gz(paths["matrix"]).tocsr()
    genes_df = read_table(paths["genes"], header=None, sep="\t")
    barcodes_df = read_table(paths["barcodes"], header=None, sep="\t")

    genes = genes_df.iloc[:, 0].astype(str).values
    barcodes = barcodes_df.iloc[:, 0].astype(str).values

    X = orient_matrix(X, n_genes=len(genes), n_cells=len(barcodes), gsm_id=gsm_id)

    adata = sc.AnnData(X)
    adata.obs_names = barcodes
    adata.var_names = genes
    adata.layers["counts"] = adata.X.copy()

    # join metadata directly into obs (so Stage 1 can be independent)
    adata = join_cell_identities_into_obs(adata, paths["cell_identities"])

    # standard identifiers
    adata.obs["gsm_id"] = gsm_id
    adata.obs["bundle_id"] = gse_dir.name.replace("_RAW", "")
    adata.obs["source"] = "GEO_RAW"

    return adata

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gse_dir", required=True, help="Path to input_data/<GSE..._RAW> folder containing GSM files")
    ap.add_argument("--out_dir", default="./processed", help="Output directory")
    ap.add_argument("--samples", default="all", help="Comma-separated GSM list or 'all'")
    ap.add_argument("--make_combined", action="store_true", help="Also write combined_raw_geo.h5ad")
    args = ap.parse_args()

    gse_dir = Path(args.gse_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    ensure_dir(out_dir)
    ensure_dir(out_dir / "manifests")

    if not gse_dir.exists():
        raise SystemExit(f"❌ gse_dir not found: {gse_dir}")

    # detect gsm ids
    gsm_ids = sorted({p.name.split("_")[0] + "_" + p.name.split("_")[1]
                      for p in gse_dir.iterdir() if p.is_file() and p.name.startswith("GSM") and "_" in p.name})

    if not gsm_ids:
        raise SystemExit(f"❌ No GSM files detected in {gse_dir}")

    if args.samples.strip().lower() != "all":
        wanted = [x.strip() for x in args.samples.split(",") if x.strip()]
        gsm_ids = [g for g in gsm_ids if g in set(wanted)]
        if not gsm_ids:
            raise SystemExit("❌ None of the requested GSM IDs were found in this GSE folder.")

    loaded = []
    for gsm in gsm_ids:
        print(f"[INFO] loading GEO {gsm}")
        adata = load_geo_to_anndata(gse_dir, gsm)

        # quick QC stats
        mat = adata.layers["counts"] if "counts" in adata.layers else adata.X
        adata.obs["n_counts"] = np.asarray(mat.sum(axis=1)).ravel()
        adata.obs["n_genes"] = np.asarray((mat > 0).sum(axis=1)).ravel()

        adata = sanitize_obs_for_h5ad(adata)

        out_path = out_dir / f"raw_{gsm}.h5ad"
        adata.write(out_path)
        print(f"[OK] wrote {out_path}")
        loaded.append(adata)

    if args.make_combined and loaded:
        print("[INFO] concatenating selected GSMs (inner join on genes)")
        combined = sc.concat(loaded, join="inner", label="gsm_id", keys=[a.obs["gsm_id"].iloc[0] for a in loaded])
        combined = sanitize_obs_for_h5ad(combined)
        combined.write(out_dir / "combined_raw_geo.h5ad")
        print(f"[OK] wrote {out_dir / 'combined_raw_geo.h5ad'}")

    print("[DONE] Stage 0 complete.")

if __name__ == "__main__":
    main()

