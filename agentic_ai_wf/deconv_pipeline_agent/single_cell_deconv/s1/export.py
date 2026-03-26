#!/usr/bin/env python3
"""
S1 (prep) — Bisque export utilities
export_bisque_tsvs
"""
from __future__ import annotations

from __future__ import annotations

import os
import time
from typing import Optional

import numpy as np
import pandas as pd
import scipy.sparse as sp
import anndata as ad

def export_bisque_tsvs(
    adata: ad.AnnData,
    outdir: str,
    cell_type_key: str,
    donor_key: Optional[str] = None,
    chunk_size: int = 500,
    memmap: bool = True,
):
    """
    Optimized, memory-safe, and Bisque-compatible TSV exporter.

    Outputs (written to `outdir`):
      - sc_counts.tsv               (genes × cells; first column = 'gene')
      - sc_metadata.tsv             (cell_barcode, cell_type, individual_id)
      - reference_by_celltype.tsv   (genes × cell_types; mean CPM-like signal per cell type)

    Features
    --------
    - Handles sparse AnnData.X safely (CSR path).
    - Writes sc_counts.tsv in gene-chunks with a single header line (no duplication).
    - Optional memmap for large reference matrix to reduce RAM usage.
    - Final integrity check for counts row/col alignment.

    Parameters
    ----------
    adata : AnnData
        Prepared single-cell object; `adata.obs_names` are cell barcodes,
        `adata.var_names` are genes.
    outdir : str
        Output directory (created if missing).
    cell_type_key : str
        obs column holding cell-type labels (e.g., 'cell_type').
    donor_key : str | None
        obs column for donor/subject labels; if None/absent → 'D0'.
    chunk_size : int
        Number of genes per write chunk when streaming sc_counts.tsv.
    memmap : bool
        Use a memory-mapped file while building reference_by_celltype (saves RAM).
    """
    os.makedirs(outdir, exist_ok=True)

    n_genes, n_cells = adata.n_vars, adata.n_obs
    start_time = time.time()

    print(f"\n[FAST EXPORT] Starting Bisque export: {n_cells:,} cells × {n_genes:,} genes")
    print(f"[FAST EXPORT] Chunk size: {chunk_size} genes")
    print(f"[FAST EXPORT] Output directory: {outdir}")

    X = adata.X
    sparse_mode = sp.issparse(X)
    if sparse_mode:
        print(f"[FAST EXPORT] Detected sparse matrix. Using CSR for chunked export.")
        X = X.tocsr()

    # ------------------------------------------------------------------
    # STEP 1 — sc_counts.tsv  (WRITE 'gene' HEADER + cell barcodes ONCE)
    # ------------------------------------------------------------------
    sc_counts_path = os.path.join(outdir, "sc_counts.tsv")
    print(f"\n[FAST EXPORT] Step 1/3: Writing sc_counts.tsv...")

    # Header: 'gene' + cell barcodes
    with open(sc_counts_path, "w", encoding="utf-8") as f:
        f.write("gene\t" + "\t".join(map(str, adata.obs_names)) + "\n")

    # Stream rows in gene-chunks (transpose per chunk: cells×genes → genes×cells)
    for i, start in enumerate(range(0, n_genes, chunk_size), 1):
        end = min(start + chunk_size, n_genes)
        gene_names = adata.var_names[start:end]

        if sparse_mode:
            block = X[:, start:end].toarray()
        else:
            block = np.asarray(X[:, start:end])

        block = block.T  # genes × cells

        df = pd.DataFrame(block, index=gene_names, columns=adata.obs_names)
        df.insert(0, "gene", df.index)  # first column = gene
        df.to_csv(
            sc_counts_path,
            sep="\t",
            index=False,
            header=False,
            mode="a",
            float_format="%.6g",
        )

        pct = (end / n_genes) * 100
        print(f"[FAST EXPORT]   Chunk {i}: genes {start}-{end} ({end-start}) | {pct:.1f}%")

    print(f"[FAST EXPORT] ✓ sc_counts.tsv complete ({n_genes:,} × {n_cells:,})")

    # ------------------------------------------------------------------
    # STEP 2 — sc_metadata.tsv
    # ------------------------------------------------------------------
    print(f"\n[FAST EXPORT] Step 2/3: Writing sc_metadata.tsv...")
    obs = adata.obs
    meta = pd.DataFrame({
        "cell_barcode": adata.obs_names,
        "cell_type": (
            obs[cell_type_key].astype(str).values
            if cell_type_key in obs.columns else "Unknown"
        ),
        "individual_id": (
            obs[donor_key].astype(str).values
            if donor_key and donor_key in obs.columns else "D0"
        ),
    })
    meta_path = os.path.join(outdir, "sc_metadata.tsv")
    meta.to_csv(meta_path, sep="\t", index=False)
    print(f"[FAST EXPORT] ✓ sc_metadata.tsv complete ({len(meta):,} cells)")

    # ------------------------------------------------------------------
    # STEP 3 — reference_by_celltype.tsv  (mean expression per CT)
    # ------------------------------------------------------------------
    print(f"\n[FAST EXPORT] Step 3/3: Computing reference_by_celltype (memory-safe)...")
    cell_types = meta["cell_type"].values
    unique_types = np.unique(cell_types)
    print(f"[FAST EXPORT] Found {len(unique_types)} unique cell types")

    dtype = np.float32
    ref_path = os.path.join(outdir, "reference_by_celltype.tsv")

    if memmap:
        memmap_path = os.path.join(outdir, "ref_memmap.npy")
        ref_data = np.memmap(memmap_path, mode="w+", dtype=dtype,
                             shape=(n_genes, len(unique_types)))
    else:
        ref_data = np.zeros((n_genes, len(unique_types)), dtype=dtype)

    for j, ct in enumerate(unique_types, 1):
        ct_start = time.time()
        mask = cell_types == ct
        n_ct = mask.sum()
        if n_ct == 0:
            ref_data[:, j-1] = 0
            continue

        X_ct = X[mask, :]
        if sp.issparse(X_ct):
            ref_ct = np.array(X_ct.mean(axis=0)).ravel()
        else:
            ref_ct = X_ct.mean(axis=0)
        ref_data[:, j-1] = ref_ct

        elapsed = time.time() - ct_start
        print(f"[FAST EXPORT]   {j}/{len(unique_types)} '{ct}' ({n_ct:,} cells) | {elapsed:.1f}s")

    ref_df = pd.DataFrame(ref_data, index=adata.var_names, columns=unique_types)
    ref_df.to_csv(ref_path, sep="\t", float_format="%.6g")
    print(f"[FAST EXPORT] ✓ reference_by_celltype.tsv written ({n_genes:,}×{len(unique_types)})")

    if memmap:
        del ref_data  # release memmap handle

    # ------------------------------------------------------------------
    # STEP 4 — Integrity checks
    # ------------------------------------------------------------------
    print(f"\n[FAST EXPORT] Verifying export integrity...")

    # sc_counts.tsv: expect 1 header + one line per gene
    with open(sc_counts_path, "r", encoding="utf-8") as f:
        n_lines = sum(1 for _ in f)
    expected = n_genes + 1
    if n_lines != expected:
        raise ValueError(
            f"Counts file mismatch: expected {expected} lines, found {n_lines}. "
            f"Check header duplication or write interruption."
        )

    # sc_metadata.tsv: expect one row per cell (excluding header)
    meta_lines = sum(1 for _ in open(meta_path, "r", encoding="utf-8")) - 1
    if meta_lines != n_cells:
        raise ValueError(
            f"Metadata mismatch: expected {n_cells} cells, found {meta_lines} rows. "
            f"Check metadata generation."
        )

    total = time.time() - start_time
    print(f"[FAST EXPORT] ✓ Validation passed — files aligned ({n_genes} genes, {n_cells} cells)")
    print(f"\n[FAST EXPORT] 🚀 EXPORT COMPLETE in {total/60:.2f} min")
    print(f"[FAST EXPORT] Output directory: {outdir}")
