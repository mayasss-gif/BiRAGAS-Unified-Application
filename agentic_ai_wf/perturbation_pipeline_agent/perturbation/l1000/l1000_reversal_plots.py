
import re
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Thread-safe backend
import matplotlib.pyplot as plt
from logging import Logger


# ---------------------------------------------------------------------
# Helpers to load causal summary for reversal drugs
# ---------------------------------------------------------------------
def _pick_rank_metric(causal: pd.DataFrame, requested: Optional[str]) -> str:
    """
    Decide which column to use as ranking metric.

    Priority:
      1) requested (if present & has non-NaN)
      2) CIS_causal
      3) CIS
      4) Sensitivity Index
      5) amp_abs (if present)
    """
    candidates: List[str] = []
    if requested:
        candidates.append(requested)
    candidates.extend(["CIS_causal", "CIS", "Sensitivity Index", "amp_abs"])

    for col in candidates:
        if col in causal.columns:
            vals = pd.to_numeric(causal[col], errors="coerce")
            if vals.notna().any():
                causal[col] = vals
                return col
    raise ValueError(
        f"No usable ranking metric found. Tried: {', '.join(candidates)}. "
        f"Available columns: {list(causal.columns)}"
    )


def _load_reversal_causal_summary(
    output_dir: Path,
    logger: Logger,
    rank_metric: Optional[str] = "CIS_causal",
    r2_min: float = 0.80,
    genes_of_interest: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, str, List[str]]:
    """
    Load causal_link_table*_with_relevance.csv from the current RUN_DIR,
    keep only 'Reversal' rows, apply R² filter, and pick a ranking metric.
    """
    run_dir = output_dir
    causal_qc_path = run_dir / "causal_link_table_qc_with_relevance.csv"
    causal_path = run_dir / "causal_link_table_with_relevance.csv"

    if causal_qc_path.exists():
        path = causal_qc_path
        logger.info("[REVERSAL PLOTS] Using QC causal table: %s", causal_qc_path.name)
    elif causal_path.exists():
        path = causal_path
        logger.info("[REVERSAL PLOTS] Using causal table: %s", causal_path.name)
    else:
        raise FileNotFoundError(
            f"No causal link table found in {run_dir}. "
            "Expected 'causal_link_table_qc_with_relevance.csv' or "
            "'causal_link_table_with_relevance.csv'."
        )

    causal = pd.read_csv(path)

    # Ensure canonical column names if there are stray spaces
    causal.columns = [c.strip() for c in causal.columns]

    # Filter to reversal only if the column exists
    if "Therapeutic_Relevance" in causal.columns:
        before = len(causal)
        causal = causal[
            causal["Therapeutic_Relevance"].astype(str).str.lower() == "reversal"
        ].copy()
        logger.info(
            "[REVERSAL PLOTS] Filtered by Therapeutic_Relevance == 'Reversal': %d → %d rows.",
            before,
            len(causal),
        )

    # R² filter, if available
    if "R²" in causal.columns:
        causal["R²"] = pd.to_numeric(causal["R²"], errors="coerce")
        before = len(causal)
        causal = causal[causal["R²"] >= float(r2_min)].copy()
        logger.info(
            "[REVERSAL PLOTS] Filtered by R² ≥ %.2f: %d → %d rows.",
            r2_min,
            before,
            len(causal),
        )
    else:
        logger.warning("[REVERSAL PLOTS] Column 'R²' not found; skipping R² filter.")

    if causal.empty:
        logger.warning("[REVERSAL PLOTS] No rows left after filters.")
        return causal, rank_metric or "", []

    # Make sure Gene column is string
    if "Gene" not in causal.columns:
        raise ValueError("Causal table has no 'Gene' column – cannot group by gene.")
    causal["Gene"] = causal["Gene"].astype(str)

    # Decide ranking metric
    metric_used = _pick_rank_metric(causal, rank_metric)
    logger.info("[REVERSAL PLOTS] Ranking drugs by: %s", metric_used)

    # Gene subset
    if genes_of_interest:
        gene_set = set(str(g) for g in genes_of_interest)
        genes = sorted([g for g in causal["Gene"].unique() if g in gene_set])
    else:
        genes = sorted(causal["Gene"].unique())

    if not genes:
        logger.warning("[REVERSAL PLOTS] No genes available for plotting after filtering.")
        return causal, metric_used, []

    logger.info(
        "[REVERSAL PLOTS] Will generate plots for %d genes, top K drugs each.", len(genes)
    )
    return causal, metric_used, genes


# ---------------------------------------------------------------------
# Utility to get dose ranges from existing selection meta
# ---------------------------------------------------------------------
def _load_dose_ranges(output_dir: Path, logger: Logger) -> Optional[pd.DataFrame]:
    """
    Read selected_signatures_meta.csv if present, and return a dataframe with
    per (Drug, DoseUnit) min/max dose. Used purely for defining the x-axis
    range of the model curves (summary-only plotting; no GCTX).
    """
    meta_path = output_dir / "selected_signatures_meta.csv"
    if not meta_path.exists():
        logger.warning(
            "[REVERSAL PLOTS] selected_signatures_meta.csv not found in %s – "
            "will use EC50-based heuristic dose ranges.",
            output_dir,
        )
        return None

    meta = pd.read_csv(meta_path)
    # Standardise column names
    meta.columns = [c.strip() for c in meta.columns]

    if "pert_iname" not in meta.columns or "pert_dose_unit" not in meta.columns:
        logger.warning(
            "[REVERSAL PLOTS] selected_signatures_meta.csv missing 'pert_iname' or "
            "'pert_dose_unit' – will use EC50-based heuristic dose ranges."
        )
        return None

    meta["pert_dose_unit"] = meta["pert_dose_unit"].astype(str)
    if "dose_num" not in meta.columns:
        # compute from pert_dose if not present
        if "pert_dose" not in meta.columns:
            logger.warning(
                "[REVERSAL PLOTS] No 'dose_num' or 'pert_dose' in selection meta – "
                "will use EC50-based heuristic dose ranges."
            )
            return None
        meta["dose_num"] = pd.to_numeric(meta["pert_dose"], errors="coerce")

    meta["dose_num"] = pd.to_numeric(meta["dose_num"], errors="coerce")
    meta = meta.dropna(subset=["dose_num"])

    dose_ranges = (
        meta.groupby(["pert_iname", "pert_dose_unit"])["dose_num"]
        .agg(["min", "max"])
        .reset_index()
        .rename(columns={"pert_iname": "Drug", "pert_dose_unit": "DoseUnit"})
    )
    return dose_ranges


# ---------------------------------------------------------------------
# 1) Multi-drug, per-gene model curves (summary-only)
# ---------------------------------------------------------------------
def plot_top_reversal_drugs_per_gene_from_summary(
    output_dir: Path,
    logger: Logger,
    top_k: int = 30,
    genes_of_interest: Optional[List[str]] = None,
    rank_metric: str = "CIS_causal",
    r2_min: float = 0.80,
    out_dir_name: str = "reversal_multi_drug_from_summary",
) -> None:
    """
    For each gene, plot modelled Hill curves for the top K reversal drugs.

    * Uses ONLY the causal_link_table*_with_relevance.csv summary.
    * DOES NOT read the GCTX or re-fit curves.
    * x-axis range per drug is inferred from:
        - selected_signatures_meta.csv (if available), otherwise
        - a heuristic around EC50.
    """
    logger.info("[REVERSAL PLOTS] Starting generation of multi-drug reversal plots (summary only).")

    causal, metric_used, genes = _load_reversal_causal_summary(
        output_dir=output_dir,
        logger=logger,
        rank_metric=rank_metric,
        r2_min=r2_min,
        genes_of_interest=genes_of_interest,
    )
    if causal.empty or not genes:
        logger.warning("[REVERSAL PLOTS] Nothing to plot – exiting.")
        return

    out_dir = output_dir / "figures" / out_dir_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Dose ranges for better x-axis support
    dose_ranges = _load_dose_ranges(output_dir, logger)

    # Ensure numeric columns we need
    for col in ["EC50 (µM)", "Hill Slope", "Emax", "Baseline", metric_used]:
        if col in causal.columns:
            causal[col] = pd.to_numeric(causal[col], errors="coerce")

    # Helper Hill function
    def _hill(dose, ec50, h, emax, b):
        return b + (emax - b) * (dose**h / (ec50**h + dose**h + 1e-12))

    for gene in genes:
        gdf = causal[causal["Gene"] == gene].copy()
        gdf = gdf.dropna(subset=[metric_used, "EC50 (µM)", "Hill Slope", "Emax", "Baseline"])
        if gdf.empty:
            continue

        # Rank by metric and keep top_k
        gdf = gdf.sort_values(metric_used, ascending=False).head(int(top_k))
        n_drugs = len(gdf)
        if n_drugs == 0:
            continue

        # Colour scale by ranking metric (higher = darker)
        metric_vals = gdf[metric_used].values
        vmin, vmax = float(np.nanmin(metric_vals)), float(np.nanmax(metric_vals))
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
            cmap = plt.cm.viridis
            norm = None
        else:
            cmap = plt.cm.viridis_r  # reversed: top drug darkest
            norm = plt.Normalize(vmin=vmin, vmax=vmax)

        fig, ax = plt.subplots(figsize=(7, 4 + 0.25 * n_drugs))

        for _, row in gdf.iterrows():
            drug = str(row["Drug"])
            unit = str(row.get("DoseUnit", "uM")).replace("µ", "u").replace("μ", "u")

            ec50 = float(row["EC50 (µM)"])
            h = float(row["Hill Slope"])
            emax = float(row["Emax"])
            base = float(row["Baseline"])
            score = float(row[metric_used])

            # Determine dose range
            if dose_ranges is not None:
                subset = dose_ranges[
                    (dose_ranges["Drug"] == drug) & (dose_ranges["DoseUnit"] == unit)
                ]
                if not subset.empty:
                    d_min = float(subset["min"].iloc[0])
                    d_max = float(subset["max"].iloc[0])
                else:
                    d_min = np.nan
                    d_max = np.nan
            else:
                d_min = d_max = np.nan

            # Fallback heuristic if metadata not available / bad
            if not np.isfinite(d_min) or not np.isfinite(d_max) or d_min <= 0 or d_max <= 0:
                if np.isfinite(ec50) and ec50 > 0:
                    d_min = ec50 / 30.0
                    d_max = ec50 * 30.0
                else:
                    d_min = 1e-3
                    d_max = 10.0

            d_min = max(d_min, 1e-6)
            d_max = max(d_max, d_min * 1.5)

            x = np.logspace(np.log10(d_min), np.log10(d_max), 200)
            y = _hill(x, ec50, h, emax, base)

            if norm is not None:
                color = cmap(norm(score))
            else:
                color = cmap(0.5)

            label = f"{drug} ({metric_used}={score:.2f})"
            ax.plot(x, y, label=label, color=color, linewidth=2)

        ax.set_xscale("log")
        ax.set_xlabel(f"Dose ({unit})")
        ax.set_ylabel(f"{gene} z-score (Hill model)")
        ax.set_title(f"{gene} – top {n_drugs} reversal drugs\nranked by {metric_used}")

        # Colourbar
        if norm is not None:
            try:
                sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
                sm.set_array([])
                cbar = fig.colorbar(sm, ax=ax)
                cbar.set_label(metric_used)
            except Exception as e:
                logger.warning(f"[REVERSAL PLOTS] Could not add colorbar: {e}")

        ax.legend(fontsize=8, loc="best")
        fig.tight_layout()

        safe_gene = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(gene))
        out_path = out_dir / f"{safe_gene}_top{n_drugs}_reversal_multi_drug.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        logger.info("[REVERSAL PLOTS] Saved multi-drug plot: %s", out_path.name)

    logger.info("[REVERSAL PLOTS] Finished multi-drug reversal plots (summary only).")


# ---------------------------------------------------------------------
# 2) Per-gene barplots, coloured by another metric (e.g. EC50)
# ---------------------------------------------------------------------
def plot_top_reversal_barplots_per_gene_from_summary(
    output_dir: Path,
    logger: Logger,
    top_k: int = 30,
    genes_of_interest: Optional[List[str]] = None,
    rank_metric: str = "CIS_causal",
    r2_min: float = 0.80,
    color_by: str = "EC50 (µM)",  # or "Sensitivity Index", "ATE", etc.
    out_dir_name: str = "reversal_barplots_from_summary",
) -> None:
    """
    For each gene, make a horizontal barplot of the top K reversal drugs,
    ranked by `rank_metric` and coloured by `color_by`.

    Uses only causal_link_table*_with_relevance.csv (summary, no GCTX).
    """
    logger.info("[REVERSAL BARPLOTS] Starting per-gene barplots (summary only).")

    causal, metric_used, genes = _load_reversal_causal_summary(
        output_dir=output_dir,
        logger=logger,
        rank_metric=rank_metric,
        r2_min=r2_min,
        genes_of_interest=genes_of_interest,
    )
    if causal.empty or not genes:
        logger.warning("[REVERSAL BARPLOTS] No data for plotting.")
        return

    out_dir = output_dir / "figures" / out_dir_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Ensure numeric
    causal[metric_used] = pd.to_numeric(causal[metric_used], errors="coerce")
    if color_by in causal.columns:
        causal[color_by] = pd.to_numeric(causal[color_by], errors="coerce")

    for gene in genes:
        gdf = causal[causal["Gene"].astype(str) == str(gene)].copy()
        gdf = gdf.dropna(subset=[metric_used])
        if gdf.empty:
            continue

        gdf = gdf.sort_values(metric_used, ascending=False).head(int(top_k))
        n_drugs = len(gdf)
        if n_drugs == 0:
            continue

        y_pos = np.arange(n_drugs)
        values = gdf[metric_used].values
        labels = gdf["Drug"].astype(str).tolist()

        # Colour from color_by (e.g. EC50); use log10 scale so differences are visible
        colors = "tab:blue"
        cbar_norm = None
        if color_by in gdf.columns:
            cb_vals = gdf[color_by].replace([np.inf, -np.inf], np.nan)
            with np.errstate(invalid="ignore", divide="ignore"):
                log_cb = np.log10(cb_vals)
            vmin = np.nanmin(log_cb)
            vmax = np.nanmax(log_cb)
            if np.isfinite(vmin) and np.isfinite(vmax) and vmin != vmax:
                cmap = plt.cm.Blues_r  # darker = more potent (lower EC50 if that is color_by)
                norm = plt.Normalize(vmin=vmin, vmax=vmax)
                colors = cmap(norm(log_cb))
                cbar_norm = norm

        fig, ax = plt.subplots(figsize=(7, 4 + 0.3 * n_drugs))
        ax.barh(y_pos, values, color=colors)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels)
        ax.invert_yaxis()  # best drug at top
        ax.set_xlabel(metric_used)
        ax.set_title(f"{gene} – top {n_drugs} reversal drugs\nranked by {metric_used}")

        if color_by in gdf.columns and isinstance(colors, np.ndarray) and cbar_norm is not None:
            try:
                sm = plt.cm.ScalarMappable(cmap=plt.cm.Blues_r, norm=cbar_norm)
                sm.set_array([])
                cbar = fig.colorbar(sm, ax=ax)
                cbar.set_label(f"log10({color_by})")
            except Exception as e:
                logger.warning(f"[REVERSAL PLOTS] Could not add colorbar: {e}")

        fig.tight_layout()

        safe_gene = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(gene))
        out_path = out_dir / f"{safe_gene}_top{n_drugs}_reversal_barplot.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        logger.info("[REVERSAL BARPLOTS] Saved %s", out_path.name)

    logger.info("[REVERSAL BARPLOTS] Finished per-gene barplots (summary only).")


# =====================================================================
# 3) Gene × Drug heatmap (Reversal / Aggravating / both)
# =====================================================================
def plot_gene_drug_heatmap_from_causal(
    output_dir: Path,
    logger: Logger,
    therapeutic_classes: Optional[List[str]] = None,
    rank_metric: Optional[str] = "CIS_causal",
    value_col: Optional[str] = None,
    r2_min: float = 0.80,
    out_dir_name: str = "gene_drug_heatmaps",
) -> None:
    """
    Make a single heatmap of Genes (rows) × Drugs (columns) using the
    causal_link_table*_with_relevance.csv.

    You can choose which therapeutic classes to include:
      - therapeutic_classes=None      → uses both ['Reversal', 'Aggravating'] if present
      - ['Reversal']                  → only reversal
      - ['Aggravating']               → only aggravating
      - ['Reversal', 'Aggravating']   → both

    Colour values (heat) are taken from:
      1) value_col      (if given and exists)
      2) rank_metric    (if exists)
      3) Sensitivity Index
      4) CIS_causal
      5) CIS
      6) ATE
      7) EC50 (µM)      (but transformed to -log10(EC50))
    """
    qc_path = output_dir / "causal_link_table_qc_with_relevance.csv"
    base_path = output_dir / "causal_link_table_with_relevance.csv"

    if qc_path.exists():
        path = qc_path
        logger.info("[HEATMAP] Using QC causal table: %s", qc_path.name)
    elif base_path.exists():
        path = base_path
        logger.info("[HEATMAP] Using causal table: %s", base_path.name)
    else:
        raise FileNotFoundError(
            f"No causal link table found in {output_dir}. "
            "Expected 'causal_link_table_qc_with_relevance.csv' or "
            "'causal_link_table_with_relevance.csv'."
        )

    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    if "Gene" not in df.columns or "Drug" not in df.columns:
        logger.warning("[HEATMAP] Table has no 'Gene' or 'Drug' column – cannot build heatmap.")
        return

    df["Gene"] = df["Gene"].astype(str)
    df["Drug"] = df["Drug"].astype(str)

    # Filter by therapeutic class(es)
    if therapeutic_classes is None:
        therapeutic_classes = ["Reversal", "Aggravating"]
    therapeutic_classes = [str(t).lower() for t in therapeutic_classes]

    if "Therapeutic_Relevance" in df.columns:
        before = len(df)
        df = df[
            df["Therapeutic_Relevance"]
            .astype(str)
            .str.lower()
            .isin(therapeutic_classes)
        ].copy()
        logger.info(
            "[HEATMAP] Filtered by Therapeutic_Relevance in %s: %d → %d rows.",
            therapeutic_classes,
            before,
            len(df),
        )
    else:
        logger.warning("[HEATMAP] Column 'Therapeutic_Relevance' not found – using all rows.")

    # R² filter if available
    if "R²" in df.columns:
        df["R²"] = pd.to_numeric(df["R²"], errors="coerce")
        before = len(df)
        df = df[df["R²"] >= float(r2_min)].copy()
        logger.info(
            "[HEATMAP] Filtered by R² ≥ %.2f: %d → %d rows.",
            r2_min,
            before,
            len(df),
        )

    if df.empty:
        logger.warning("[HEATMAP] No rows left after filtering – nothing to plot.")
        return

    # Decide which column to use as the heat value
    candidates: List[str] = []
    if value_col:
        candidates.append(value_col)
    if rank_metric:
        candidates.append(rank_metric)
    candidates.extend(["Sensitivity Index", "CIS_causal", "CIS", "ATE", "EC50 (µM)"])

    metric_used = None
    for col in candidates:
        if col in df.columns:
            metric_used = col
            break

    if metric_used is None:
        logger.warning(
            "[HEATMAP] No suitable value column found. Tried: %s. Available: %s",
            candidates,
            list(df.columns),
        )
        return

    df[metric_used] = pd.to_numeric(df[metric_used], errors="coerce")

    # Prepare value for heatmap
    if metric_used == "EC50 (µM)":
        val = df[metric_used].replace([np.inf, -np.inf], np.nan)
        with np.errstate(divide="ignore", invalid="ignore"):
            heat_val = -np.log10(val)
        cbar_label = "-log10(EC50 (µM))  (higher = more potent)"
    else:
        heat_val = df[metric_used]
        cbar_label = metric_used

    df["__heat_val__"] = heat_val

    # Pivot: Gene × Drug
    pivot = df.pivot_table(
        index="Gene",
        columns="Drug",
        values="__heat_val__",
        aggfunc="mean",  # if multiple rows per gene-drug, average
    )

    if pivot.empty:
        logger.warning("[HEATMAP] Pivot table is empty – nothing to plot.")
        return

    n_genes, n_drugs = pivot.shape
    fig_w = max(8, min(0.4 * n_drugs, 40))
    fig_h = max(6, min(0.35 * n_genes, 40))

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    
    # Validate pivot values before plotting
    pivot_values = pivot.values
    if pivot_values.size == 0 or not np.isfinite(pivot_values).any():
        logger.warning("[HEATMAP] Pivot values are empty or all NaN – skipping heatmap.")
        plt.close(fig)
        return
    
    im = ax.imshow(pivot_values, aspect="auto", cmap="viridis")

    ax.set_xticks(np.arange(n_drugs))
    ax.set_xticklabels(pivot.columns.tolist(), rotation=90, fontsize=6)

    ax.set_yticks(np.arange(n_genes))
    ax.set_yticklabels(pivot.index.tolist(), fontsize=6)

    classes_str = "+".join(sorted(set(therapeutic_classes)))
    ax.set_xlabel("Drug")
    ax.set_ylabel("Gene")
    ax.set_title(
        f"Gene × Drug heatmap – {classes_str}\nvalue = {cbar_label}"
    )

    # Only add colorbar if im is valid
    if im is not None:
        try:
            cbar = fig.colorbar(im, ax=ax)
            cbar.set_label(cbar_label)
        except Exception as e:
            logger.warning(f"[HEATMAP] Could not add colorbar: {e}")

    fig.tight_layout()

    out_dir = output_dir / "figures" / out_dir_name
    out_dir.mkdir(parents=True, exist_ok=True)

    safe_classes = re.sub(r"[^A-Za-z0-9_.-]+", "_", classes_str)
    safe_metric = re.sub(r"[^A-Za-z0-9_.-]+", "_", metric_used)
    out_path = out_dir / f"gene_drug_heatmap_{safe_classes}_{safe_metric}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    logger.info("[HEATMAP] Saved gene × drug heatmap → %s", out_path)


# =====================================================================
# 4) Drug barplot: number of genes affected per drug
# =====================================================================
def plot_drug_gene_count_barplot_from_causal(
    output_dir: Path,
    logger: Logger,
    therapeutic_classes: Optional[List[str]] = None,
    r2_min: float = 0.80,
    top_k: int = 30,
    out_dir_name: str = "gene_drug_heatmaps",
) -> None:
    """
    Make a global barplot: for each drug, how many genes it affects
    (after filtering by Therapeutic_Relevance and R²).

    - therapeutic_classes=None  → uses both ['Reversal', 'Aggravating'] if present
    - ['Reversal']              → only reversal
    - ['Aggravating']           → only aggravating

    Bars are sorted by gene-count, and only the top_k drugs are shown.
    """
    qc_path = output_dir / "causal_link_table_qc_with_relevance.csv"
    base_path = output_dir / "causal_link_table_with_relevance.csv"

    if qc_path.exists():
        path = qc_path
        logger.info("[BARPLOT] Using QC causal table: %s", qc_path.name)
    elif base_path.exists():
        path = base_path
        logger.info("[BARPLOT] Using causal table: %s", base_path.name)
    else:
        raise FileNotFoundError(
            f"No causal link table found in {output_dir}. "
            "Expected 'causal_link_table_qc_with_relevance.csv' or "
            "'causal_link_table_with_relevance.csv'."
        )

    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    if "Gene" not in df.columns or "Drug" not in df.columns:
        logger.warning("[BARPLOT] Table has no 'Gene' or 'Drug' column – cannot build barplot.")
        return

    df["Gene"] = df["Gene"].astype(str)
    df["Drug"] = df["Drug"].astype(str)

    # Filter by therapeutic class(es)
    if therapeutic_classes is None:
        therapeutic_classes = ["Reversal", "Aggravating"]
    therapeutic_classes = [str(t).lower() for t in therapeutic_classes]

    if "Therapeutic_Relevance" in df.columns:
        before = len(df)
        df = df[
            df["Therapeutic_Relevance"]
            .astype(str)
            .str.lower()
            .isin(therapeutic_classes)
        ].copy()
        logger.info(
            "[BARPLOT] Filtered by Therapeutic_Relevance in %s: %d → %d rows.",
            therapeutic_classes,
            before,
            len(df),
        )
    else:
        logger.warning("[BARPLOT] Column 'Therapeutic_Relevance' not found – using all rows.")

    # R² filter
    if "R²" in df.columns:
        df["R²"] = pd.to_numeric(df["R²"], errors="coerce")
        before = len(df)
        df = df[df["R²"] >= float(r2_min)].copy()
        logger.info(
            "[BARPLOT] Filtered by R² ≥ %.2f: %d → %d rows.",
            r2_min,
            before,
            len(df),
        )

    if df.empty:
        logger.warning("[BARPLOT] No rows left after filtering – nothing to plot.")
        return

    # Count number of UNIQUE genes per drug
    counts = (
        df.groupby("Drug")["Gene"]
        .nunique()
        .reset_index(name="n_genes")
        .sort_values("n_genes", ascending=False)
    )

    if counts.empty:
        logger.warning("[BARPLOT] No gene counts per drug – nothing to plot.")
        return

    # Take top_k (if fewer than top_k, it'll just use all)
    counts_top = counts.head(int(top_k))
    n_drugs = len(counts_top)

    fig_w = max(8, min(0.5 * n_drugs, 30))
    fig_h = 6

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    x_pos = np.arange(n_drugs)

    ax.bar(x_pos, counts_top["n_genes"].values, color="tab:purple")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(counts_top["Drug"].tolist(), rotation=90, fontsize=7)
    ax.set_ylabel("Number of genes affected")
    classes_str = "+".join(sorted(set(therapeutic_classes)))
    ax.set_title(f"Top {n_drugs} drugs by number of genes ({classes_str})")

    fig.tight_layout()

    out_dir = output_dir / "figures" / out_dir_name
    out_dir.mkdir(parents=True, exist_ok=True)

    safe_classes = re.sub(r"[^A-Za-z0-9_.-]+", "_", classes_str)
    out_path = out_dir / f"drug_gene_count_barplot_top{n_drugs}_{safe_classes}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    logger.info("[BARPLOT] Saved drug gene-count barplot → %s", out_path)



def generate_reversal_plots(output_dir: Path, logger: Logger):
    """
    Generate reversal plots for the given output directory.
    """
    logger.info("=== Generating reversal plots (summary only, NO GCTX, NO re-fitting) ===")

    # ------------------------------------------------------------------
    # 1) Per-gene multi-drug Hill curves (summary-only, top 30 / gene)
    # ------------------------------------------------------------------
    plot_top_reversal_drugs_per_gene_from_summary(
        output_dir=output_dir,
        logger=logger,
        top_k=30,               # if fewer than 30, it will just use what's available
        genes_of_interest=None, # None = all genes with reversal after filters
        rank_metric="CIS_causal",
        r2_min=0.80,
        out_dir_name="reversal_multi_drug_from_summary",
    )

    # ------------------------------------------------------------------
    # 2) Per-gene barplots (top 30 drugs, coloured by EC50)
    # ------------------------------------------------------------------
    plot_top_reversal_barplots_per_gene_from_summary(
        output_dir=output_dir,
        logger=logger,
        top_k=30,
        genes_of_interest=None,
        rank_metric="CIS_causal",
        r2_min=0.80,
        color_by="EC50 (µM)",   # can change to "Sensitivity Index", "ATE", etc.
        out_dir_name="reversal_barplots_from_summary",
    )

    # ------------------------------------------------------------------
    # 3) Gene × Drug heatmap (all genes, all drugs, Reversal only)
    # ------------------------------------------------------------------
    plot_gene_drug_heatmap_from_causal(
        output_dir=output_dir,
        logger=logger,
        therapeutic_classes=["Reversal"],  # or ["Reversal", "Aggravating"]
        rank_metric="CIS_causal",
        value_col=None,                    # None → use CIS_causal
        r2_min=0.80,
        out_dir_name="gene_drug_heatmaps",
    )

    # ------------------------------------------------------------------
    # 4) Global drug barplot: how many genes each drug reverses (top 30)
    # ------------------------------------------------------------------
    plot_drug_gene_count_barplot_from_causal(
        output_dir=output_dir,
        logger=logger,
        therapeutic_classes=["Reversal"],  # or ["Reversal", "Aggravating"]
        r2_min=0.80,
        top_k=30,
        out_dir_name="gene_drug_heatmaps",
    )

    logger.info("=== Finished all summary-based reversal visualizations ===")
