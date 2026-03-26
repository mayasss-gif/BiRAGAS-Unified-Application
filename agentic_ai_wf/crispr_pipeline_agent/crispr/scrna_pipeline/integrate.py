from __future__ import annotations
import scanpy as sc
import numpy as np


# =========================
# Base preprocessing
# =========================

def _base_preprocess(adata, pp_cfg: dict):
    # HVG on raw counts (seurat_v3 expects counts)
    sc.pp.highly_variable_genes(
        adata,
        n_top_genes=int(pp_cfg["hvg_n"]),
        flavor="seurat_v3",
        batch_key="sample_id" if "sample_id" in adata.obs else None,
    )
    adata = adata[:, adata.var["highly_variable"]].copy()

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.scale(adata, max_value=10)
    sc.tl.pca(adata, n_comps=int(pp_cfg["n_pcs"]))

    return adata


def _neighbors_cluster_umap(adata, pp_cfg: dict, use_rep: str | None = None):
    sc.pp.neighbors(
        adata,
        n_neighbors=int(pp_cfg["neighbors_k"]),
        n_pcs=None if use_rep else int(pp_cfg["n_pcs"]),
        use_rep=use_rep,
    )
    sc.tl.leiden(adata, resolution=float(pp_cfg["leiden_resolution"]))
    sc.tl.umap(adata)
    return adata


# =========================
# Integration methods
# =========================

def run_none(adata, pp_cfg: dict, out_dir=None):
    adata = _base_preprocess(adata, pp_cfg)
    adata = _neighbors_cluster_umap(adata, pp_cfg)
    return adata


def run_harmony(adata, pp_cfg: dict, out_dir=None):
    """
    SAFE Harmony implementation.
    Never writes malformed X_pca_harmony.
    """
    try:
        import harmonypy
    except Exception as e:
        raise RuntimeError("harmonypy not installed") from e

    adata = _base_preprocess(adata, pp_cfg)

    if "X_pca" not in adata.obsm:
        raise RuntimeError("X_pca missing")

    X = np.asarray(adata.obsm["X_pca"])
    meta = adata.obs.copy()

    ho = harmonypy.run_harmony(
        X,
        meta_data=meta,
        vars_use=["sample_id"],
        verbose=True,
    )

    Z = np.asarray(ho.Z_corr)

    n_cells, n_pcs = X.shape

    if Z.ndim != 2:
        raise RuntimeError(f"Harmony returned invalid shape {Z.shape}")

    if Z.shape == (n_pcs, n_cells):
        Xh = Z.T
    elif Z.shape == (n_cells, n_pcs):
        Xh = Z
    else:
        raise RuntimeError(
            f"Harmony shape mismatch: X_pca={X.shape}, Z_corr={Z.shape}"
        )

    if Xh.shape != (n_cells, n_pcs):
        raise RuntimeError(f"Invalid harmony embedding {Xh.shape}")

    adata.obsm["X_pca_harmony"] = Xh

    adata = _neighbors_cluster_umap(
        adata,
        pp_cfg,
        use_rep="X_pca_harmony",
    )
    return adata


def run_bbknn(adata, pp_cfg: dict, out_dir=None):
    try:
        import bbknn
    except Exception as e:
        raise RuntimeError("bbknn not installed") from e

    adata = _base_preprocess(adata, pp_cfg)
    bbknn.bbknn(adata, batch_key="sample_id")
    sc.tl.leiden(adata, resolution=float(pp_cfg["leiden_resolution"]))
    sc.tl.umap(adata)
    return adata


def run_scvi(adata, pp_cfg: dict, out_dir=None):
    try:
        import scvi
    except Exception as e:
        raise RuntimeError("scvi-tools not installed") from e

    if "counts" not in adata.layers:
        raise RuntimeError("scVI requires adata.layers['counts']")

    scvi.model.SCVI.setup_anndata(
        adata,
        layer="counts",
        batch_key="sample_id",
    )

    model = scvi.model.SCVI(adata, n_latent=30)
    model.train()

    adata.obsm["X_scVI"] = model.get_latent_representation()

    adata = _neighbors_cluster_umap(
        adata,
        pp_cfg,
        use_rep="X_scVI",
    )

    # Ensure log-normalized X for downstream rank_genes_groups (scVI skips _base_preprocess)
    if "counts" in adata.layers:
        adata.X = adata.layers["counts"].copy()
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)

    return adata

