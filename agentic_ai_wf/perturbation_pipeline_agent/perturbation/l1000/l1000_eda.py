# src/l1000_eda.py
from pathlib import Path
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Thread-safe backend
import matplotlib.pyplot as plt
import seaborn as sns  # safe: ensured via env_setup
from logging import Logger



from .constants import CORE_SIG_PATH, CORE_PERT_PATH, CORE_CELL_PATH, CORE_GENE_PATH


def _read_tsv(path: Path, logger: Logger):
    if not path.exists():
        logger.warning(f"EDA: file not found, skipping: {path}")
        return None
    logger.info(f"EDA: reading {path.name}")
    return pd.read_csv(path, sep="\t", low_memory=False)


def _save_table(df: pd.DataFrame, base_name: str, table_dir: Path, logger: Logger):
    """
    Save full table as CSV in table_dir. No heads/preview.
    """
    table_out = table_dir / f"{base_name}.csv"
    df.to_csv(table_out, index=False)
    logger.info(f"EDA: saved table → {table_out}")


def _setup_plot_style():
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except Exception:
        plt.style.use("default")
    plt.rcParams["figure.dpi"] = 110


def run_l1000_eda(output_dir: Path, logger: Logger):
    """
    Optional L1000 exploratory summaries.
    - Saves summary tables under TABLE_DIR
    - Saves figures under FIG_DIR
    - No downstream step depends on these outputs.
    """

    logger.info("=== STEP EDA: L1000 exploratory summaries ===")
    output_dir.mkdir(parents=True, exist_ok=True)
    TABLE_DIR = output_dir / "tables"
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR = output_dir / "figures"
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    _setup_plot_style()

    # ------------------------------------------------------------------
    # 1) SIG_INFO
    # ------------------------------------------------------------------
    sig_info = _read_tsv(CORE_SIG_PATH, logger)
    if sig_info is not None:
        _save_table(sig_info, "34_sig_info_full", TABLE_DIR, logger)

        sig_unique_counts = (
            sig_info.nunique()
            .sort_values(ascending=False)
            .reset_index()
        )
        sig_unique_counts.columns = ["Column Name", "Unique Value Count"]
        _save_table(sig_unique_counts, "35_sig_info_unique_counts", TABLE_DIR, logger)

        if "pert_itime" in sig_info.columns:
            itime_counts = (
                sig_info["pert_itime"]
                .value_counts()
                .sort_index()
                .reset_index()
            )
            itime_counts.columns = ["pert_itime", "Count"]
            _save_table(itime_counts, "36_sig_info_pert_itime_counts", TABLE_DIR, logger)

            # barplot of pert_itime
            plt.figure(figsize=(9, 5))
            sns.barplot(data=itime_counts, x="pert_itime", y="Count")
            plt.xlabel("Perturbation Time (pert_itime)")
            plt.ylabel("Count of Signatures")
            plt.title("Distribution of Treatment Times (pert_itime)")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            fig_path = FIG_DIR / f"36_fig_pert_itime_distribution_{time.strftime('%Y%m%d-%H%M%S')}.png"
            plt.savefig(fig_path, dpi=150, bbox_inches="tight")
            plt.close()
            logger.info(f"EDA: saved figure → {fig_path}")
        else:
            logger.info("EDA: 'pert_itime' not found in SIG; skipping its distribution.")

    # ------------------------------------------------------------------
    # 2) PERT_INFO
    # ------------------------------------------------------------------
    pert_info = _read_tsv(CORE_PERT_PATH, logger)
    if pert_info is not None:
        # Unique counts per column
        pert_unique_counts = (
            pert_info.nunique()
            .sort_values(ascending=False)
            .reset_index()
        )
        pert_unique_counts.columns = ["Column Name", "Unique Value Count"]
        _save_table(pert_unique_counts, "38_pert_info_unique_counts", TABLE_DIR, logger)

        # Distribution of perturbation types
        if "pert_type" in pert_info.columns:
            pert_type_counts = pert_info["pert_type"].value_counts().reset_index()
            pert_type_counts.columns = ["Perturbation Type", "Count"]
            total = pert_type_counts["Count"].sum()
            pert_type_counts["Percentage (%)"] = (
                pert_type_counts["Count"] / total * 100
            ).round(2)

            description_map = {
                "trt_cp": "Chemical compounds (drugs/inhibitors)",
                "trt_sh": "shRNA knockdowns (loss-of-function)",
                "trt_sh.cgs": "Consensus gene signatures from shRNAs",
                "trt_sh.css": "Composite shRNA signatures",
                "trt_oe": "Gene overexpression (gain-of-function)",
                "trt_lig": "Ligands (growth factors, cytokines)",
                "trt_oe.mut": "Mutant overexpression constructs",
                "ctl_vector": "Empty vector control",
                "ctl_vector.cns": "Consensus vector control",
                "ctl_vehicle": "Vehicle control (e.g., DMSO)",
                "ctl_vehicle.cns": "Consensus vehicle control",
                "ctl_untrt": "Untreated control",
                "ctl_untrt.cns": "Consensus untreated control",
            }
            pert_type_counts["Biological Meaning"] = pert_type_counts[
                "Perturbation Type"
            ].map(description_map)

            _save_table(pert_type_counts, "39_pert_types_summary", TABLE_DIR, logger)
        else:
            logger.info("EDA: 'pert_type' not found in PERT; skipping type summary.")

    # ------------------------------------------------------------------
    # 3) CELL_INFO
    # ------------------------------------------------------------------
    cell_info = _read_tsv(CORE_CELL_PATH, logger)
    if cell_info is not None:
        # clean "-666" sentinels
        for c in cell_info.columns:
            if cell_info[c].dtype == object:
                cell_info[c] = (
                    cell_info[c]
                    .replace("-666", np.nan)
                    .replace(" -666", np.nan)
                )

        _save_table(cell_info, "40_cell_info_cleaned", TABLE_DIR, logger)

        cell_unique_counts = (
            cell_info.nunique()
            .sort_values(ascending=False)
            .reset_index()
        )
        cell_unique_counts.columns = ["Column Name", "Unique Value Count"]
        _save_table(cell_unique_counts, "41_cell_info_unique_counts", TABLE_DIR, logger)

        # core dictionaries
        keep_cols = [
            "cell_id",
            "cell_type",
            "base_cell_id",
            "precursor_cell_id",
            "modification",
            "sample_type",
            "primary_site",
            "subtype",
            "original_growth_pattern",
            "provider_catalog_id",
            "original_source_vendor",
            "donor_age",
            "donor_sex",
            "donor_ethnicity",
        ]
        existing_keep = [c for c in keep_cols if c in cell_info.columns]
        core_cell_dict = (
            cell_info[existing_keep].drop_duplicates().reset_index(drop=True)
        )
        _save_table(core_cell_dict, "42_cell_core_dictionary", TABLE_DIR, logger)

        # primary_site distribution
        if "primary_site" in cell_info.columns:
            primary_site_counts = (
                cell_info["primary_site"]
                .value_counts(dropna=False)
                .reset_index()
            )
            primary_site_counts.columns = ["Primary Site", "Number of Cell Lines"]
            primary_site_counts["Percentage (%)"] = (
                primary_site_counts["Number of Cell Lines"]
                / primary_site_counts["Number of Cell Lines"].sum()
                * 100
            ).round(2)
            _save_table(primary_site_counts, "43_cell_primary_site_counts", TABLE_DIR, logger)

        # subtype distribution
        if "subtype" in cell_info.columns:
            subtype_counts = (
                cell_info["subtype"]
                .value_counts(dropna=False)
                .reset_index()
            )
            subtype_counts.columns = ["Subtype", "Number of Cell Lines"]
            _save_table(subtype_counts, "44_cell_subtype_counts", TABLE_DIR, logger)

        # other categorical distributions
        cat_cols = [
            "cell_type",
            "sample_type",
            "original_growth_pattern",
            "original_source_vendor",
            "donor_sex",
            "donor_ethnicity",
            "modification",
        ]
        out_idx = 45
        for col in cat_cols:
            if col in cell_info.columns:
                tbl = cell_info[col].value_counts(dropna=False).reset_index()
                tbl.columns = [col.replace("_", " ").title(), "Count"]
                _save_table(tbl, f"{out_idx}_cell_distribution_by_{col}", TABLE_DIR, logger)
                out_idx += 1

        # variants per base_cell_id
        if {"base_cell_id", "cell_id"}.issubset(cell_info.columns):
            variants_per_base = (
                cell_info.groupby("base_cell_id")["cell_id"]
                .nunique()
                .reset_index()
                .sort_values("cell_id", ascending=False)
            )
            variants_per_base.columns = [
                "Base Cell ID",
                "Number of Variants (cell_id)",
            ]
            _save_table(variants_per_base, f"{out_idx}_cell_variants_per_base", TABLE_DIR, logger)
            out_idx += 1

        # minimal join key
        join_cols = [
            c
            for c in [
                "cell_id",
                "base_cell_id",
                "primary_site",
                "subtype",
                "sample_type",
                "original_growth_pattern",
                "donor_sex",
                "donor_ethnicity",
            ]
            if c in cell_info.columns
        ]
        cell_key = cell_info[join_cols].drop_duplicates()
        _save_table(cell_key, f"{out_idx}_cell_minimal_join_key", TABLE_DIR, logger)

    # ------------------------------------------------------------------
    # 4) GENE_INFO
    # ------------------------------------------------------------------
    gene_info = _read_tsv(CORE_GENE_PATH, logger)
    if gene_info is not None:
        out_idx = 50  # just an arbitrary offset
        _save_table(gene_info, f"{out_idx}_gene_info_full", TABLE_DIR, logger)
        out_idx += 1

        gene_unique_counts = (
            gene_info.nunique()
            .sort_values(ascending=False)
            .reset_index()
        )
        gene_unique_counts.columns = ["Column Name", "Unique Value Count"]
        _save_table(gene_unique_counts, f"{out_idx}_gene_info_unique_counts", TABLE_DIR, logger)
        out_idx += 1

        if "feature_space" in gene_info.columns:
            space_counts = (
                gene_info["feature_space"]
                .value_counts(dropna=False)
                .reset_index()
            )
            space_counts.columns = ["Feature Type", "Count"]
            space_counts["Percentage (%)"] = (
                space_counts["Count"] / space_counts["Count"].sum() * 100
            ).round(2)
            _save_table(
                space_counts,
                f"{out_idx}_gene_feature_space_distribution",
                TABLE_DIR,
                logger,
            )
        elif "is_landmark" in gene_info.columns:
            landmark_counts = (
                gene_info["is_landmark"]
                .value_counts(dropna=False)
                .reset_index()
            )
            landmark_counts.columns = ["Is Landmark (1=yes)", "Count"]
            landmark_counts["Percentage (%)"] = (
                landmark_counts["Count"] / landmark_counts["Count"].sum() * 100
            ).round(2)
            _save_table(
                landmark_counts,
                f"{out_idx}_gene_landmark_distribution",
                TABLE_DIR,
                logger,
            )
        else:
            logger.info(
                "EDA: neither 'feature_space' nor 'is_landmark' in GENE; "
                "skipping landmark summary."
            )

    logger.info(
        "✅ EDA: exploratory summaries completed. "
        "All tables saved under TABLE_DIR; figures under FIG_DIR."
    )

