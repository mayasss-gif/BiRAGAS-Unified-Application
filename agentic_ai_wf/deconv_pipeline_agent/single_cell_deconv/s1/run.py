# src/bisque_deconv/s1/run.py
#!/usr/bin/env python3
"""
S1 (prep) — Entrypoint
s1_qc_and_bisque_prep
"""
from __future__ import annotations
import os
import json
from typing import Any, Dict, List, Optional, Tuple

import anndata as ad
import numpy as np
import pandas as pd
# --- robust adapters to tolerate different function signatures ---
def _robust_read_bulk_genes(read_fn, bulk_path, idcol):
    import pandas as pd
    # 1) try positional
    try:
        return read_fn(bulk_path, idcol)
    except TypeError:
        pass
    # 2) try common keyword names
    for kw in ("id_col", "index_col", "gene_col", "col", "column"):
        try:
            return read_fn(bulk_path, **{kw: idcol})
        except TypeError:
            continue
    # 3) last resort: pandas
    df = pd.read_csv(bulk_path, sep=None, engine="python")
    if isinstance(idcol, int):
        return df.iloc[:, idcol]
    return df[idcol]

def _robust_read_bulk_counts(read_fn, bulk_path, idcol):
    import pandas as pd
    # 1) try positional
    try:
        return read_fn(bulk_path, idcol)
    except TypeError:
        pass
    # 2) try common keyword names
    for kw in ("index_col", "id_col", "gene_col", "col", "column"):
        try:
            return read_fn(bulk_path, **{kw: idcol})
        except TypeError:
            continue
    # 3) last resort: pandas + set index
    df = pd.read_csv(bulk_path, sep=None, engine="python")
    if isinstance(idcol, int):
        df = df.set_index(df.columns[idcol])
    else:
        df = df.set_index(idcol)
    return df

# ---- Use the top-level adata_utils for .h5ad handling ----
from ..adata_utils import find_h5ad_files, read_h5ad, concat_adatas

# ---- S1 local helpers (adjust these if your package names differ) ----
# If you DO have these modules, keep these imports.
# If you DON'T, comment them out and inline/replace accordingly.
from .qc import calc_qc, qc_filter
from .tissue import filter_adatas_by_tissue
from .mapping import harmonize_to_bulk_ids
from .export import export_bisque_tsvs
from .markers import compute_markers

# If you have IO helpers for the bulk side in .io, import them; otherwise implement here.
try:
    from .io import read_bulk_genes, read_bulk_counts
except Exception:
    # Simple fallbacks if .io is missing:
    def read_bulk_genes(bulk_path: str, id_col=0):
        df = pd.read_csv(bulk_path, sep=None, engine="python")
        if isinstance(id_col, int):
            return df.iloc[:, id_col]
        return df[id_col]

    def read_bulk_counts(bulk_path: str, index_col=0):
        df = pd.read_csv(bulk_path, sep=None, engine="python")
        df = df.set_index(df.columns[index_col] if isinstance(index_col, int) else index_col)
        return df

# -------------------- Defaults --------------------
DEF_SPECIES = "human"
DEF_USE_SCRUBLET = False
DEF_MIN_GENES_PER_CELL = 200
DEF_MIN_CELLS_PER_GENE = 3
DEF_MITO_PCT_MAX: Optional[float] = 15.0
DEF_TOP_N_MARKERS = 100
DEF_FINAL_H5AD_NAME = "Bisque_ready.h5ad"

DEF_TISSUE_KEYS = ["tissue"]
DEF_TISSUE_MATCH_MODE = "substring"
DEF_TISSUE_CASE_SENSITIVE = False


def _detect_keys_for_labels(adata: ad.AnnData) -> Tuple[Optional[str], Optional[str]]:
    """Heuristically detect cell-type and donor keys from .obs."""
    ct_candidates = [
        "cell_type", "celltype", "cluster", "leiden", "louvain",
        "seurat_clusters", "annotation", "celltypes"
    ]
    donor_candidates = [
        "donor", "sample", "individual", "subject", "patient",
        "orig.ident", "batch", "donor_id", "Sample"
    ]
    cell_key = next((k for k in ct_candidates if k in adata.obs.columns), None)
    donor_key = next((k for k in donor_candidates if k in adata.obs.columns), None)
    return cell_key, donor_key


def s1_qc_and_bisque_prep(
    root: str,
    outdir: str,
    bulk: str,
    bulk_gene_col: Optional[str] = None,
    species: str = DEF_SPECIES,
    tissue_include: Optional[List[str]] = None,
    tissue_keys: Optional[List[str]] = None,
    tissue_match_mode: str = DEF_TISSUE_MATCH_MODE,
    tissue_case_sensitive: bool = DEF_TISSUE_CASE_SENSITIVE,
    use_scrublet: bool = DEF_USE_SCRUBLET,
    min_genes_per_cell: int = DEF_MIN_GENES_PER_CELL,
    min_cells_per_gene: int = DEF_MIN_CELLS_PER_GENE,
    mito_pct_max: Optional[float] = DEF_MITO_PCT_MAX,
    top_n_markers: int = DEF_TOP_N_MARKERS,
    final_h5ad_name: str = DEF_FINAL_H5AD_NAME,
) -> Tuple[Dict[str, str], Dict[str, Any]]:

    tissue_include = tissue_include or []
    tissue_keys = tissue_keys or DEF_TISSUE_KEYS
    os.makedirs(outdir, exist_ok=True)

    # 1) Load single-cell references (.h5ad; single file or directory tree)
    files = find_h5ad_files(root)
    if not files:
        raise FileNotFoundError(f"No .h5ad files found under: {root}")
    print(f"[S1] Found {len(files)} .h5ad file(s)")
    adatas = [read_h5ad(p) for p in files]

    # 2) Optional tissue filter (per file)
    adatas = filter_adatas_by_tissue(
        adatas=adatas,
        include=tissue_include,
        keys=tissue_keys,
        mode=tissue_match_mode,
        case_sens=tissue_case_sensitive,
    )

    # 3) Merge, QC metrics + filtering
    merged = concat_adatas(adatas)
    _ = calc_qc(merged, species=species)
    filt, qc_sum = qc_filter(
        merged,
        min_genes=min_genes_per_cell,
        min_cells_per_gene=min_cells_per_gene,
        mito_pct_max=mito_pct_max,
        use_scrublet=use_scrublet,
    )

    # 4) Bulk side: genes + counts (normalize IDs to uppercase/no version)
    # bulk_genes = read_bulk_genes(bulk, id_col=(bulk_gene_col if bulk_gene_col not in {None, "", "None"} else 0))
    # bulk_genes = pd.Series(pd.Series(bulk_genes).astype(str).str.strip().str.replace(r"\.\d+$", "", regex=True).str.upper())

    # bulk_counts = read_bulk_counts(bulk, index_col=(bulk_gene_col if bulk_gene_col not in {None, "", "None"} else 0))
    # bulk_counts.index = bulk_counts.index.astype(str).str.strip().str.replace(r"\.\d+$", "", regex=True).str.upper()
    idcol = (bulk_gene_col if bulk_gene_col not in {None, "", "None"} else 0)

    # genes
    bulk_genes = _robust_read_bulk_genes(read_bulk_genes, bulk, idcol)
    bulk_genes = pd.Series(pd.Series(bulk_genes).astype(str).str.strip()
                        .str.replace(r"\.\d+$", "", regex=True).str.upper())

    # counts
    bulk_counts = _robust_read_bulk_counts(read_bulk_counts, bulk, idcol)
    bulk_counts.index = bulk_counts.index.astype(str).str.strip() \
                            .str.replace(r"\.\d+$", "", regex=True).str.upper()

    # 5) Map single-cell gene IDs to bulk and subset to the overlap
    final, map_sum = harmonize_to_bulk_ids(filt, bulk_genes)

    # 6) Detect likely label columns for exports/markers
    CELL_TYPE_KEY, DONOR_KEY = _detect_keys_for_labels(final)

    # 7) Exports for Bisque (counts, metadata, reference)
    export_bisque_tsvs(final, outdir, cell_type_key=CELL_TYPE_KEY or "cell_type", donor_key=DONOR_KEY)

    # 8) Optional marker table
    markers = compute_markers(final, cell_type_key=CELL_TYPE_KEY or "cell_type", top_n=top_n_markers)
    markers_path = os.path.join(outdir, "markers.tsv")
    markers.to_csv(markers_path, sep="\t", index=False, encoding="utf-8")

    # 9) Save final h5ad
    final_path = os.path.join(outdir, final_h5ad_name)
    final.write(final_path)

    # 10) Bulk overlap/unmapped splits (helpful for S2 sanity)
    overlap_genes = [g for g in final.var_names if g in bulk_counts.index]
    unmapped_genes = [g for g in bulk_counts.index if g not in final.var_names]
    bulk_overlap = bulk_counts.loc[overlap_genes].copy()
    bulk_unmapped = bulk_counts.loc[unmapped_genes].copy()

    bulk_overlap_path = os.path.join(outdir, "bulk_counts_overlap.tsv")
    bulk_unmapped_path = os.path.join(outdir, "bulk_counts_unmapped.tsv")
    bulk_overlap.to_csv(bulk_overlap_path, sep="\t", encoding="utf-8")
    bulk_unmapped.to_csv(bulk_unmapped_path, sep="\t", encoding="utf-8")

    # 11) JSON summaries
    qc_json_path = os.path.join(outdir, "qc_summary.json")
    map_json_path = os.path.join(outdir, "mapping_summary.json")
    bulk_json_path = os.path.join(outdir, "bulk_mapping_counts.json")

    bulk_sum = {
        "bulk_total_genes": int(bulk_counts.shape[0]),
        "overlap_genes": int(len(overlap_genes)),
        "unmapped_genes": int(len(unmapped_genes)),
        "overlap_file": os.path.basename(bulk_overlap_path),
        "unmapped_file": os.path.basename(bulk_unmapped_path),
    }

    with open(qc_json_path, "w", encoding="utf-8") as f:
        json.dump(qc_sum, f, indent=2, ensure_ascii=False)
    with open(map_json_path, "w", encoding="utf-8") as f:
        json.dump(map_sum, f, indent=2, ensure_ascii=False)
    with open(bulk_json_path, "w", encoding="utf-8") as f:
        json.dump(bulk_sum, f, indent=2, ensure_ascii=False)

    # 12) Assemble outputs
    paths = {
        "final_h5ad": final_path,
        "sc_counts": os.path.join(outdir, "sc_counts.tsv"),
        "sc_metadata": os.path.join(outdir, "sc_metadata.tsv"),
        "reference_by_celltype": os.path.join(outdir, "reference_by_celltype.tsv"),
        "markers": markers_path,
        "bulk_overlap": bulk_overlap_path,
        "bulk_unmapped": bulk_unmapped_path,
        "qc_summary": qc_json_path,
        "mapping_summary": map_json_path,
        "bulk_mapping_counts": bulk_json_path,
    }

    summaries = {
        "qc_summary": qc_sum,
        "mapping_summary": map_sum,
        "bulk_summary": bulk_sum,
        "bisque_prep": {
            "root": root,
            "bulk_counts_file": bulk,
            "bulk_genes_column": bulk_gene_col,
            "cell_type_key": CELL_TYPE_KEY,
            "donor_key": DONOR_KEY,
            "species": species,
            "tissue_filter": {"include": tissue_include, "keys_used": tissue_keys},
            "notes": "QC-filtered, optional tissue-filtered, and bulk-mapped; ready for Bisque.",
        },
    }

    print("[S1] Final Bisque-ready H5AD:", final_path)
    print("[S1] TSVs written to:", outdir)
    print("[S1] Overlap genes:", summaries["mapping_summary"]["n_overlap"] if "n_overlap" in summaries["mapping_summary"] else "n/a")

    return paths, summaries
