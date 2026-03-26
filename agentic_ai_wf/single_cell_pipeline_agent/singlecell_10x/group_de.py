from __future__ import annotations
# group_de.py

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scanpy as sc
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scanpy as sc
from .config_cli import logger



# ============================================================
# 1. compute_de_between_groups_per_cluster
# ============================================================

def compute_de_between_groups_per_cluster(
    adata,
    group_col: str = "group",
    cluster_col: str = "leiden",
    out_dir: Path | None = None,
):
    """
    For each cluster, run DE for ALL group pairs (focus vs ref).
    Output: one CSV with all comparisons per cluster.
    """
    if out_dir is None:
        raise ValueError("out_dir must be provided for compute_de_between_groups_per_cluster")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if group_col not in adata.obs.columns:
        logger.info(f"[DE-CLUSTER-GROUP] '{group_col}' not in obs → skipping.")
        return

    if cluster_col not in adata.obs.columns:
        logger.info(f"[DE-CLUSTER-GROUP] '{cluster_col}' not in obs → skipping.")
        return

    groups = sorted(adata.obs[group_col].astype(str).unique().tolist())
    if len(groups) < 2:
        logger.info(
            f"[DE-CLUSTER-GROUP] Need at least 2 groups in '{group_col}', found: {groups}"
        )
        return

    logger.info(
        f"[DE-CLUSTER-GROUP] Comparing ALL group pairs within each cluster ({cluster_col}). "
        f"Groups: {groups}"
    )

    all_rows = []
    clusters = sorted(adata.obs[cluster_col].astype(str).unique().tolist())

    for cl in clusters:
        sub = adata[adata.obs[cluster_col].astype(str) == cl].copy()
        sub_group_counts = sub.obs[group_col].astype(str).value_counts()

        if sub_group_counts.size < 2:
            logger.info(
                f"[DE-CLUSTER-GROUP] Cluster {cl}: only one group present "
                f"{sub_group_counts.to_dict()} → skip."
            )
            continue

        for i in range(len(groups) - 1):
            for j in range(i + 1, len(groups)):
                group_ref = groups[i]
                group_focus = groups[j]

                if group_ref not in sub_group_counts.index or group_focus not in sub_group_counts.index:
                    continue

                logger.info(
                    f"[DE-CLUSTER-GROUP] Cluster {cl}: {group_focus} vs {group_ref}, "
                    f"counts={sub_group_counts.to_dict()}"
                )

                try:
                    sc.tl.rank_genes_groups(
                        sub,
                        groupby=group_col,
                        groups=[group_focus],
                        reference=group_ref,
                        method="wilcoxon",
                        n_genes=sub.n_vars,
                    )

                    try:
                        df = sc.get.rank_genes_groups_df(sub, group_focus)
                    except Exception:
                        rg = sub.uns["rank_genes_groups"]
                        names = rg["names"][group_focus]
                        scores = rg["scores"][group_focus]
                        pvals_adj = rg["pvals_adj"][group_focus]
                        logfc = None
                        if "logfoldchanges" in rg:
                            logfc = rg["logfoldchanges"][group_focus]
                        df_dict = {
                            "names": names,
                            "scores": scores,
                            "pvals_adj": pvals_adj,
                        }
                        if logfc is not None:
                            df_dict["logfoldchanges"] = logfc
                        df = pd.DataFrame(df_dict)

                    df["cluster"] = cl
                    df["group_focus"] = group_focus
                    df["group_ref"] = group_ref
                    df["comparison"] = f"{group_focus}_vs_{group_ref}"

                    if "logfoldchanges" in df.columns and "pvals_adj" in df.columns:
                        df["regulation"] = "no_change"
                        up_mask = (df["logfoldchanges"] > 1.0) & (df["pvals_adj"] < 0.05)
                        down_mask = (df["logfoldchanges"] < -1.0) & (df["pvals_adj"] < 0.05)
                        df.loc[up_mask, "regulation"] = "up"
                        df.loc[down_mask, "regulation"] = "down"

                        df["direction_simple"] = np.where(
                            df["logfoldchanges"] > 0,
                            "upregulated",
                            np.where(
                                df["logfoldchanges"] < 0,
                                "downregulated",
                                "no_expression_change",
                            ),
                        )

                    all_rows.append(df)

                except Exception as e:
                    logger.warning(
                        f"[DE-CLUSTER-GROUP] DE failed for cluster {cl}, "
                        f"{group_focus} vs {group_ref}: {e}"
                    )
                    continue

    if all_rows:
        de_all = pd.concat(all_rows, axis=0, ignore_index=True)
        out_file = out_dir / f"DE_{cluster_col}_ALL_GROUP_PAIRS.csv"
        de_all.to_csv(out_file, index=False)
        logger.info(
            f"[DE-CLUSTER-GROUP] Wrote group-wise DE per cluster (all pairs): {out_file}"
        )
    else:
        logger.info("[DE-CLUSTER-GROUP] No DE tables were generated.")
        

# ============================================================
# 2. plot_groupwise_celltype_proportions
# ============================================================

def plot_groupwise_celltype_proportions(
    adata,
    group_col: str = "group",
    celltype_col: str | None = None,
    out_dir: Path | None = None,
):
    if out_dir is None:
        raise ValueError("out_dir must be provided for plot_groupwise_celltype_proportions")
    out_dir = Path(out_dir)

    if group_col not in adata.obs.columns:
        logger.info(f"[CT-PROP] '{group_col}' not in obs → skip.")
        return

    if celltype_col is None or celltype_col not in adata.obs.columns:
        logger.info("[CT-PROP] celltype column missing → skip.")
        return

    df = (
        adata.obs[[group_col, celltype_col]]
        .astype(str)
        .value_counts()
        .reset_index(name="n_cells")
    )
    total_per_group = df.groupby(group_col)["n_cells"].sum().rename("total_cells")
    df = df.merge(total_per_group, on=group_col, how="left")
    df["fraction"] = df["n_cells"] / df["total_cells"]

    prop_file = out_dir / "celltype_proportions_by_group.csv"
    df.to_csv(prop_file, index=False)
    logger.info(f"[CT-PROP] cell type proportions: {prop_file}")

    pivot = df.pivot(
        index=celltype_col,
        columns=group_col,
        values="fraction"
    ).fillna(0.0)

    fig, ax = plt.subplots(figsize=(12, 5))
    pivot.plot(kind="bar", ax=ax)
    ax.set_ylabel("fraction of cells")
    ax.set_title("Cell type proportions by group")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(out_dir / "celltype_proportions_by_group.png", dpi=300)
    plt.close(fig)


# ============================================================
# 3. plot_group_specific_umaps
# ============================================================

def plot_group_specific_umaps(
    adata,
    group_col: str = "group",
    color_col: str = "leiden",
    out_dir: Path | None = None,
):
    if out_dir is None:
        raise ValueError("out_dir must be provided for plot_groupwise_celltype_proportions")
    out_dir = Path(out_dir)

    if group_col not in adata.obs.columns:
        logger.info(f"[UMAP-BY-GROUP] '{group_col}' not in obs → skip.")
        return

    sc.settings.figdir = out_dir

    groups = sorted(adata.obs[group_col].astype(str).unique().tolist())
    for g in groups:
        sub = adata[adata.obs[group_col].astype(str) == g, :].copy()
        logger.info(f"[UMAP-BY-GROUP] Plotting group={g}, color={color_col}")
        try:
            sc.pl.umap(
                sub,
                color=[color_col],
                size=15,
                show=False,
                save=f"_group_{g}_{color_col}.png",
            )
        except Exception as e:
            logger.warning(f"[UMAP-BY-GROUP] Failed to plot group {g}: {e}")


# ============================================================
# 4. compute_de_by_celltype
# ============================================================

def compute_de_by_celltype(
    adata,
    celltype_col: str,
    group_col: str = "group",
    deg_root_dir: Path | None = None,
):
    """
    Single-cell DE per cell type (group comparisons).
    """
    if deg_root_dir is None:
        raise ValueError("deg_root_dir must be provided for compute_de_by_celltype")

    if celltype_col not in adata.obs.columns or group_col not in adata.obs.columns:
        logger.info(f"[DE-CELLTYPE] Missing '{celltype_col}' or '{group_col}' → skip.")
        return

    groups = adata.obs[group_col].astype(str).unique().tolist()
    if len(groups) < 2:
        logger.info(f"[DE-CELLTYPE] Need >=2 groups in '{group_col}', found: {groups}")
        return

    celltype_deg_root = deg_root_dir / "celltype_specific_deg"
    celltype_deg_root.mkdir(parents=True, exist_ok=True)

    group_ref = groups[0]
    group_focus = groups[1]

    celltypes = adata.obs[celltype_col].astype(str).unique().tolist()
    for ct in celltypes:
        sub = adata[adata.obs[celltype_col].astype(str) == ct, :].copy()
        tab = sub.obs[group_col].value_counts()

        if (tab < 2).any() or tab.size < 2:
            logger.info(
                f"[DE-CELLTYPE] celltype={ct}: not enough cells per group {tab.to_dict()} → skip."
            )
            continue

        logger.info(
            f"[DE-CELLTYPE] celltype={ct}: running {group_focus} vs {group_ref} ..."
        )
        try:
            sc.tl.rank_genes_groups(
                sub,
                groupby=group_col,
                groups=[group_focus],
                reference=group_ref,
                method="wilcoxon",
                n_genes=sub.n_vars,
            )
            df = sc.get.rank_genes_groups_df(sub, group_focus)

            keep_cols = ["names", "logfoldchanges", "pvals_adj"]
            for c in keep_cols:
                if c not in df.columns:
                    logger.warning(
                        f"[DE-CELLTYPE] Missing column '{c}' in DE results for {ct}; skipping."
                    )
                    continue

            df = df[keep_cols].copy()
            df["celltype"] = ct
            df["comparison"] = f"{group_focus}_vs_{group_ref}"
            df["logFC_str"] = df["logfoldchanges"].map(
                lambda x: f"{x:+.2f}" if pd.notnull(x) else ""
            )

            ct_safe = str(ct).replace(" ", "_").replace("/", "_")
            ct_dir = celltype_deg_root / ct_safe
            ct_dir.mkdir(parents=True, exist_ok=True)

            out_file = ct_dir / f"{ct_safe}_{group_focus}_vs_{group_ref}.csv"
            df.to_csv(out_file, index=False)
            logger.info(f"[DE-CELLTYPE] Wrote celltype-specific DEG file: {out_file}")

        except Exception as e:
            logger.warning(f"[DE-CELLTYPE] Failed for celltype={ct}: {e}")
            continue
