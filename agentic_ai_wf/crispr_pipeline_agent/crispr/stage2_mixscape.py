from __future__ import annotations
#!/usr/bin/env python3
"""
Stage 2 — Mixscape responder deconvolution (Pertpy Mixscape)

✅ Works across pertpy versions (auto-detects supported args)
✅ No scipy/anndata sparse slicing crash
✅ Treats '*' as control
✅ Filters perturbations with too few cells
✅ Runs Mixscape ONLY on control + perturbed_single
✅ Produces: stage2_mixscape_<sample>.h5ad + tables + UMAPs

🔧 Added (vignette parity; safe/optional):
✅ Mixscape vignette-style figures when possible:
   - barplot (target/guide efficiency proxy)
   - perturbscore plot for a chosen perturbation
   - posterior/probability violin plot (if available)
   - heatmap (Mixscape ordering / DE-like view)
   - LDA + plot_lda
✅ Extra UMAPs for confounders (replicate/phase/etc) if present
✅ Extra tables summarizing KO/NP per perturbation (if available)

NOTE: Does NOT change CLI args, required inputs, or primary output names.
"""

import argparse
from pathlib import Path
import inspect
import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import pertpy as pt

import anndata as ad
from scipy.sparse import issparse, csr_matrix


# -----------------------------
# Utilities
# -----------------------------
def ensure_dirs(out_dir: Path):
    (out_dir / "tables").mkdir(parents=True, exist_ok=True)
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)


def safe_numeric(series, default=np.nan):
    return pd.to_numeric(series, errors="coerce").fillna(default)


def parse_perturbation_label(x: str):
    """
    Examples:
      "OST4_pDS353" -> ("OST4", "pDS353")
      "63(mod)_pBA580" -> ("63(mod)", "pBA580")
      "*" -> ("*", "")
    """
    x = str(x) if x is not None else ""
    x = x.strip()
    if x == "" or x.lower() == "nan":
        return ("", "")
    if "_" not in x:
        return (x, "")
    parts = x.split("_", 1)
    return (parts[0], parts[1])


def safe_subset_anndata(adata: sc.AnnData, idx: np.ndarray) -> sc.AnnData:
    """
    Robust subsetting for sparse matrices.
    Avoids scipy/anndata sparse slicing crashes.
    """
    X = adata.X
    if issparse(X):
        X = X.tocsr()
    else:
        X = csr_matrix(np.asarray(X))

    sub = ad.AnnData(
        X=X[idx, :],
        obs=adata.obs.iloc[idx].copy(),
        var=adata.var.copy(),
    )

    # copy layers safely
    for lname in list(adata.layers.keys()):
        L = adata.layers[lname]
        if issparse(L):
            sub.layers[lname] = L.tocsr()[idx, :].copy()
        else:
            sub.layers[lname] = np.asarray(L)[idx, :].copy()

    # copy embeddings if exist
    for k in list(getattr(adata, "obsm", {}).keys()):
        try:
            sub.obsm[k] = adata.obsm[k][idx, :].copy()
        except Exception:
            pass

    return sub


def figsave(path: Path):
    """Uniform save helper."""
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()


def pick_best_split_by(obs: pd.DataFrame):
    """
    Choose a split_by column for perturbation_signature if present.
    This helps reduce confounders like replicate/batch effects (like the vignette).
    """
    for cand in ["replicate", "batch", "donor", "sample", "library", "lane"]:
        if cand in obs.columns:
            return cand
    return None


def pick_top_perturbation(use: sc.AnnData) -> str | None:
    """
    Choose a perturbation_id (non-control) to use for vignette-like inspection plots.
    Preference:
      - highest KO fraction if mixscape_class exists
      - otherwise highest cell count
    """
    df = use.obs.copy()
    if "perturbation_id" not in df.columns:
        return None

    df = df[df["perturbation_id"] != "control"].copy()
    if df.shape[0] == 0:
        return None

    if "mixscape_class" in df.columns:
        # some pertpy versions label global vs local differently; try a few
        cls = df["mixscape_class"].astype(str)
        ko_mask = cls.str.upper().str.contains("KO")
        grp = df.assign(_is_ko=ko_mask).groupby("perturbation_id").agg(
            n=("perturbation_id", "size"),
            ko_rate=("_is_ko", "mean"),
        )
        grp = grp.sort_values(["ko_rate", "n"], ascending=False)
        return grp.index[0] if grp.shape[0] else None

    # fallback: most frequent perturbation
    return df["perturbation_id"].value_counts().index[0]


# -----------------------------
# Define condition classes
# -----------------------------
def define_condition_classes(adata: sc.AnnData) -> sc.AnnData:
    """
    Stage1 guide logic:
      guide_id == "*"  -> control
      guide_id == ""   -> unknown
      else             -> perturbed_single
      multiplets -> multipert (if is_multiplet exists)
    """
    adata.obs["condition_class"] = "unknown"

    if "guide_id" in adata.obs.columns:
        gids = adata.obs["guide_id"].astype(str).fillna("").str.strip()

        adata.obs.loc[gids == "*", "condition_class"] = "control"
        adata.obs.loc[gids == "", "condition_class"] = "unknown"
        adata.obs.loc[(gids != "") & (gids != "*"), "condition_class"] = "perturbed_single"

        if "is_multiplet" in adata.obs.columns:
            mp = adata.obs["is_multiplet"].astype(bool)
            adata.obs.loc[mp, "condition_class"] = "multipert"

        return adata

    # fallback
    if "perturbation" not in adata.obs.columns:
        raise ValueError("Missing obs['guide_id'] and obs['perturbation'] in input")

    pert = adata.obs["perturbation"].astype(str).fillna("")
    control_mask = pert.str.upper().str.contains(r"NTC|CONTROL|SCRAMBLE|NEG|\*", regex=True)
    adata.obs.loc[control_mask, "condition_class"] = "control"
    adata.obs.loc[~control_mask, "condition_class"] = "perturbed_single"
    return adata


# -----------------------------
# Standardize perturbation_id
# -----------------------------
def standardize_perturbation_columns(adata: sc.AnnData) -> sc.AnnData:
    """
    perturbation_id must be stable and unique per perturbation group.
    """
    if "guide_id" in adata.obs.columns:
        gids = adata.obs["guide_id"].astype(str).fillna("").str.strip()
        tg, gt = zip(*[parse_perturbation_label(x) for x in gids.tolist()])
        adata.obs["target_gene"] = pd.Series(tg, index=adata.obs_names).astype(str)
        adata.obs["guide_token"] = pd.Series(gt, index=adata.obs_names).astype(str)
    else:
        pert_raw = adata.obs.get("perturbation", "").astype(str).fillna("").str.strip()
        tg, gt = zip(*[parse_perturbation_label(x) for x in pert_raw.tolist()])
        adata.obs["target_gene"] = pd.Series(tg, index=adata.obs_names).astype(str)
        adata.obs["guide_token"] = pd.Series(gt, index=adata.obs_names).astype(str)

    adata.obs["perturbation_id"] = adata.obs["target_gene"].astype(str)

    mask_has_guide = adata.obs["guide_token"].astype(str).str.strip() != ""
    adata.obs.loc[mask_has_guide, "perturbation_id"] = (
        adata.obs.loc[mask_has_guide, "target_gene"].astype(str)
        + "_"
        + adata.obs.loc[mask_has_guide, "guide_token"].astype(str)
    )

    adata.obs.loc[adata.obs["condition_class"] == "control", "perturbation_id"] = "control"
    adata.obs.loc[adata.obs["condition_class"] == "multipert", "perturbation_id"] = "MULTIPERT"
    adata.obs.loc[adata.obs["perturbation_id"].astype(str).str.strip() == "", "perturbation_id"] = "unknown"

    return adata


# -----------------------------
# QC filtering
# -----------------------------
def qc_filtering(adata: sc.AnnData, min_genes=200, max_mito=30.0) -> sc.AnnData:
    keep = np.ones(adata.n_obs, dtype=bool)

    if "n_genes" in adata.obs.columns:
        keep &= safe_numeric(adata.obs["n_genes"], default=0) >= min_genes
    elif "ngenes" in adata.obs.columns:
        keep &= safe_numeric(adata.obs["ngenes"], default=0) >= min_genes

    if "percent_mito" in adata.obs.columns:
        keep &= safe_numeric(adata.obs["percent_mito"], default=0) <= max_mito

    idx = np.where(np.asarray(keep, dtype=bool))[0]
    return safe_subset_anndata(adata, idx)


# -----------------------------
# Preprocess + UMAP
# -----------------------------
def preprocess(adata: sc.AnnData, n_hvg=3000, n_pcs=50, neighbors_k=30) -> sc.AnnData:
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    sc.pp.highly_variable_genes(adata, n_top_genes=n_hvg, flavor="seurat")
    if "highly_variable" in adata.var.columns:
        adata = adata[:, adata.var["highly_variable"]].copy()

    sc.tl.pca(adata, n_comps=n_pcs, svd_solver="arpack")
    sc.pp.neighbors(
        adata,
        n_neighbors=neighbors_k,
        n_pcs=min(n_pcs, adata.obsm["X_pca"].shape[1]),
    )
    sc.tl.umap(adata)
    return adata


# -----------------------------
# Filter low-count perturbations
# -----------------------------
def filter_low_count_perts(use: sc.AnnData, min_cells_per_pert: int) -> sc.AnnData:
    """
    Keep:
      - all controls
      - only perturbations with >= min_cells_per_pert cells
    """
    df = use.obs.copy()
    ctrl_mask = df["perturbation_id"] == "control"

    pert_df = df[~ctrl_mask].copy()
    pert_counts = pert_df["perturbation_id"].value_counts()

    keep_perts = set(pert_counts[pert_counts >= min_cells_per_pert].index.tolist())
    keep_mask = ctrl_mask | df["perturbation_id"].isin(keep_perts)

    dropped = pert_counts[pert_counts < min_cells_per_pert]
    if dropped.shape[0] > 0:
        print(f"[WARN] dropping {dropped.shape[0]} perturbations with < {min_cells_per_pert} cells")
        print(dropped.head(15).to_string())

    idx = np.where(keep_mask.values)[0]
    return safe_subset_anndata(use, idx)


# -----------------------------
# Mixscape (AUTO-DETECT pertpy signature)
# -----------------------------
def run_mixscape_version_safe(use: sc.AnnData) -> tuple[sc.AnnData, pt.tl.Mixscape]:
    """
    pertpy Mixscape API differs across versions.
    We detect supported args and call safely.

    Returns:
      (use, ms)
    """
    ms = pt.tl.Mixscape()

    # ---- perturbation signature (vignette-style, optionally split_by replicate/batch)
    split_by = pick_best_split_by(use.obs)
    sig_fn = ms.perturbation_signature
    sig_params = set(inspect.signature(sig_fn).parameters.keys())
    sig_kwargs = dict(pert_key="perturbation_id", control="control")

    if split_by is not None and "split_by" in sig_params:
        sig_kwargs["split_by"] = split_by

    # Some versions take n_neighbors
    if "n_neighbors" in sig_params:
        sig_kwargs["n_neighbors"] = 30

    ms.perturbation_signature(use, **sig_kwargs)

    # ---- mixscape call
    mix_fn = ms.mixscape
    mix_params = set(inspect.signature(mix_fn).parameters.keys())

    # Build kwargs based on API version
    # Newer versions (0.10.0+) use 'labels' instead of 'pert_key'
    if "labels" in mix_params:
        kwargs = dict(labels="perturbation_id", control="control")
    else:
        kwargs = dict(pert_key="perturbation_id", control="control")

    # encourage stable behavior across versions
    if "n_neighbors" in mix_params:
        kwargs["n_neighbors"] = 30
    if "neighbors" in mix_params:
        kwargs["neighbors"] = 30
    if "use_rep" in mix_params:
        kwargs["use_rep"] = "X_pca"

    # If mixscape can take a layer, we want X_pert from signature step
    if "layer" in mix_params:
        # pertpy vignette uses layer="X_pert"
        # perturbation_signature typically writes to layers["X_pert"] or similar
        if "X_pert" in use.layers.keys():
            kwargs["layer"] = "X_pert"

    try:
        ms.mixscape(use, **kwargs)
        return use, ms
    except TypeError as e:
        print(f"[WARN] mixscape() failed with kwargs={kwargs}")
        print(f"[WARN] {e}")

    # Fallback: only required args (try both API versions)
    try:
        if "labels" in mix_params:
            ms.mixscape(use, labels="perturbation_id", control="control")
        else:
            ms.mixscape(use, pert_key="perturbation_id", control="control")
        return use, ms
    except Exception as e:
        print("[ERROR] mixscape fallback failed too.")
        raise e


# -----------------------------
# Output tables
# -----------------------------
def write_tables(adata: sc.AnnData, out_dir: Path):
    df = adata.obs.copy()

    summary = (
        df.groupby("condition_class")
        .size()
        .reset_index(name="n_cells")
        .sort_values("n_cells", ascending=False)
    )
    summary.to_csv(out_dir / "tables" / "stage2_condition_class_summary.tsv", sep="\t", index=False)

    if "mixscape_score" not in df.columns:
        df["mixscape_score"] = np.nan
    if "is_responder" not in df.columns:
        df["is_responder"] = 0

    df_single = df[df["condition_class"] == "perturbed_single"].copy()
    if df_single.shape[0] == 0:
        print("[WARN] No perturbed_single cells detected. Skipping perturbation summary.")
        return

    pert_summary = (
        df_single.groupby("perturbation_id")
        .agg(
            n_cells=("perturbation_id", "size"),
            responder_rate=("is_responder", "mean"),
            mean_mixscape_score=("mixscape_score", "mean"),
        )
        .sort_values(["n_cells"], ascending=False)
        .reset_index()
    )
    pert_summary.to_csv(out_dir / "tables" / "stage2_mixscape_summary.tsv", sep="\t", index=False)

    # Extra table: KO/NP counts per perturbation if available
    if "mixscape_class" in df.columns:
        mcls = df["mixscape_class"].astype(str)
        # standardize: KO vs NP vs other
        label = np.where(mcls.str.upper().str.contains("KO"), "KO",
                 np.where(mcls.str.upper().str.contains("NP"), "NP", mcls))
        df2 = df.copy()
        df2["_mixscape_simple"] = label

        t = (
            df2[df2["condition_class"] == "perturbed_single"]
            .groupby(["perturbation_id", "_mixscape_simple"])
            .size()
            .reset_index(name="n_cells")
        )
        t.to_csv(out_dir / "tables" / "stage2_mixscape_class_counts.tsv", sep="\t", index=False)


# -----------------------------
# Figures
# -----------------------------
def save_umaps(adata: sc.AnnData, out_dir: Path):
    fig_dir = out_dir / "figures"

    # existing
    sc.pl.umap(adata, color=["condition_class"], show=False)
    figsave(fig_dir / "umap_condition_class.png")

    if "mixscape_class" in adata.obs.columns:
        sc.pl.umap(adata, color=["mixscape_class"], show=False)
        figsave(fig_dir / "umap_mixscape_class.png")

    # additional (safe if columns exist)
    extra_cols = []
    for c in ["perturbation_id", "target_gene", "replicate", "batch", "phase", "S.Score", "G2M.Score"]:
        if c in adata.obs.columns:
            extra_cols.append(c)

    # save as separate images to keep consistent output
    for c in extra_cols:
        try:
            sc.pl.umap(adata, color=[c], show=False)
            figsave(fig_dir / f"umap_{c}.png")
        except Exception:
            pass


def save_mixscape_vignette_plots(use: sc.AnnData, ms: pt.tl.Mixscape, out_dir: Path):
    """
    Try to reproduce the main vignette outputs in a robust, pipeline-safe manner.
    Writes into out_dir/figures.
    """
    fig_dir = out_dir / "figures"

    # 1) barplot (if available in this pertpy version)
    try:
        # Prefer the column that represents gRNA/guide if present; otherwise use perturbation_id.
        guide_col = "guide_id" if "guide_id" in use.obs.columns else "perturbation_id"
        ms.plot_barplot(use, guide_rna_column=guide_col)
        figsave(fig_dir / f"mixscape_barplot_{guide_col}.png")
    except Exception as e:
        print(f"[WARN] Could not generate mixscape barplot: {e}")

    # Choose a perturbation to inspect like IFNGR2 in the vignette
    target = pick_top_perturbation(use)
    if target is None:
        print("[WARN] No perturbation available for vignette-style inspection plots.")
        return

    # 2) perturbscore distribution (if available)
    try:
        # Check which API version: newer uses 'labels', older uses 'pert_key'
        plot_ps_fn = ms.plot_perturbscore
        plot_ps_params = set(inspect.signature(plot_ps_fn).parameters.keys())
        if "labels" in plot_ps_params:
            ms.plot_perturbscore(adata=use, labels="perturbation_id", target_gene=target, color="orange")
        else:
            ms.plot_perturbscore(adata=use, pert_key="perturbation_id", target_gene=target, color="orange")
        figsave(fig_dir / f"mixscape_perturbscore_{target}.png")
    except Exception as e:
        print(f"[WARN] Could not generate perturbscore plot for {target}: {e}")

    # 3) violin / posterior-style
    # Different versions expose different columns; try to find a reasonable groupby column.
    try:
        # This matches the vignette style: groupby="mixscape_class"
        if "mixscape_class" in use.obs.columns:
            ms.plot_violin(
                adata=use,
                target_gene_idents=["control", f"{target} NP", f"{target} KO"],
                groupby="mixscape_class",
            )
            figsave(fig_dir / f"mixscape_violin_{target}.png")
        else:
            # fallback: scanpy violin if mixscape_score exists
            if "mixscape_score" in use.obs.columns:
                sc.pl.violin(use, keys="mixscape_score", groupby="perturbation_id", rotation=90, show=False)
                figsave(fig_dir / f"mixscape_score_violin_by_perturbation.png")
    except Exception as e:
        print(f"[WARN] Could not generate violin/posterior plot for {target}: {e}")

    # 4) heatmap (if available)
    try:
        # heatmap requires layer in vignette; prefer X_pert if present
        layer = "X_pert" if "X_pert" in use.layers.keys() else None
        if layer is not None:
            # Check which API version: newer uses 'labels', older uses 'pert_key'
            plot_hm_fn = ms.plot_heatmap
            plot_hm_params = set(inspect.signature(plot_hm_fn).parameters.keys())
            if "labels" in plot_hm_params:
                ms.plot_heatmap(
                    adata=use,
                    labels="perturbation_id",
                    target_gene=target,
                    layer=layer,
                    control="control",
                )
            else:
                ms.plot_heatmap(
                    adata=use,
                    pert_key="perturbation_id",
                    target_gene=target,
                    layer=layer,
                    control="control",
                )
            figsave(fig_dir / f"mixscape_heatmap_{target}.png")
    except Exception as e:
        print(f"[WARN] Could not generate heatmap for {target}: {e}")

    # 5) LDA + plot_lda (if available)
    try:
        lda_fn = getattr(ms, "lda", None)
        plot_lda_fn = getattr(ms, "plot_lda", None)
        if callable(lda_fn) and callable(plot_lda_fn):
            lda_params = set(inspect.signature(lda_fn).parameters.keys())
            # Check which API version: newer uses 'labels', older uses 'pert_key'
            if "labels" in lda_params:
                lda_kwargs = dict(adata=use, control="control", labels="perturbation_id")
            else:
                lda_kwargs = dict(adata=use, control="control", pert_key="perturbation_id")
            if "layer" in lda_params and "X_pert" in use.layers.keys():
                lda_kwargs["layer"] = "X_pert"
            ms.lda(**lda_kwargs)
            ms.plot_lda(adata=use, control="control")
            figsave(fig_dir / "mixscape_lda.png")
    except Exception as e:
        print(f"[WARN] Could not generate LDA plots: {e}")


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_h5ad", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default="processed")
    ap.add_argument("--sample_name", type=str, default="sample")
    ap.add_argument("--min_cells_per_pert", type=int, default=10)
    ap.add_argument("--min_genes", type=int, default=200)
    ap.add_argument("--max_mito", type=float, default=30.0)
    ap.add_argument("--n_hvg", type=int, default=3000)
    ap.add_argument("--n_pcs", type=int, default=50)
    ap.add_argument("--neighbors_k", type=int, default=30)
    args = ap.parse_args()

    input_h5ad = Path(args.input_h5ad).resolve()
    out_dir = Path(args.out_dir).resolve()
    ensure_dirs(out_dir)

    print(f"[INFO] reading {input_h5ad}")
    adata = sc.read_h5ad(input_h5ad)

    print("[INFO] defining condition classes")
    adata = define_condition_classes(adata)

    print("[INFO] standardizing perturbation columns")
    adata = standardize_perturbation_columns(adata)

    print("[INFO] QC filtering")
    adata = qc_filtering(adata, min_genes=args.min_genes, max_mito=args.max_mito)

    print("[INFO] preprocessing + UMAP")
    adata = preprocess(adata, n_hvg=args.n_hvg, n_pcs=args.n_pcs, neighbors_k=args.neighbors_k)

    # subset control + perturbed_single
    mask = adata.obs["condition_class"].isin(["control", "perturbed_single"]).values
    idx = np.where(mask)[0]
    use = safe_subset_anndata(adata, idx)

    print(f"[INFO] Mixscape subset before filtering: {use.n_obs} cells")

    use = filter_low_count_perts(use, min_cells_per_pert=args.min_cells_per_pert)
    print(f"[INFO] Mixscape subset after filtering: {use.n_obs} cells")

    print("[INFO] running Mixscape (pertpy auto-version safe)")
    use, ms = run_mixscape_version_safe(use)

    # merge results back
    for col in ["mixscape_class", "mixscape_score"]:
        if col in use.obs.columns:
            # keep existing behavior (string fill)
            adata.obs[col] = ""
            adata.obs.loc[use.obs_names, col] = use.obs[col].astype(str)

    if "mixscape_class" in adata.obs.columns:
        adata.obs["is_responder"] = (adata.obs["mixscape_class"].astype(str).str.upper().str.contains("KO")).astype(int)
    else:
        adata.obs["is_responder"] = 0

    print("[INFO] writing tables")
    write_tables(adata, out_dir)

    print("[INFO] saving figures (UMAPs + vignette-style Mixscape plots where possible)")
    save_umaps(adata, out_dir)
    # vignette-style plots are best generated on the `use` object (control + perturbed_single)
    save_mixscape_vignette_plots(use, ms, out_dir)

    out_h5ad = out_dir / f"stage2_mixscape_{args.sample_name}.h5ad"
    adata.write(out_h5ad)

    print(f"[OK] wrote {out_h5ad}")
    print(f"[OK] tables -> {out_dir / 'tables'}")
    print(f"[OK] figures -> {out_dir / 'figures'}")


if __name__ == "__main__":
    main()
