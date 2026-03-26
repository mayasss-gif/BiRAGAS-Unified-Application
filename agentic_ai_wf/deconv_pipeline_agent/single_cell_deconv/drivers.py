from __future__ import annotations

def run_s1(*, root_for_s1, out_s1, bulk_abs, bulk_gene_col, species, tissue_include, tissue_keys, tissue_match_mode):
    from .s1.run import s1_qc_and_bisque_prep  # import late
    s1_qc_and_bisque_prep(
        root=root_for_s1,
        outdir=out_s1,
        bulk=bulk_abs,
        bulk_gene_col=(None if str(bulk_gene_col).lower() in {"", "none"} else bulk_gene_col),
        species=species,
        tissue_include=tissue_include,
        tissue_keys=tissue_keys,
        tissue_match_mode=tissue_match_mode,
    )

def run_s2(*, out_s1, out_s2, drop_mito, allow_duplicate_genes, warn_min_overlap, max_prop_deviation,
           top_markers_per_ct, top_genes_per_sample, auto_install):
    from .s2.deconv import s2_bisque_deconv  # import late
    s2_bisque_deconv(
        ref_dir=out_s1,
        out_dir=out_s2,
        drop_mito=drop_mito,
        allow_duplicate_genes=allow_duplicate_genes,
        warn_min_overlap=int(warn_min_overlap),
        max_prop_deviation=float(max_prop_deviation),
        top_markers_per_ct=int(top_markers_per_ct),
        top_genes_per_sample=int(top_genes_per_sample),
        auto_install=auto_install,
    )

def run_s3_adapter(**kwargs):
    from .s3.run import run_s3 as _run_s3  # import late
    return _run_s3(**kwargs)
