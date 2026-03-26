from __future__ import annotations
# summary_ct_deg.py
from pathlib import Path
import numpy as np
import pandas as pd
from .config_cli import logger


def summarize_celltype_degs_markers_pathways(
    out_dir: Path,
    analysis_name: str,
    deg_dir: Path,
    celltype_dir: Path,
    ct_deg_pathway_dir: Path | None = None,
    top_n_genes: int = 30,
    top_n_pathways: int = 10,
):
    """
    Summary across:
      - celltype-specific DEGs (group A vs group B)
      - whether each DEG is also a cell-type marker
      - a simple score (2 = DEG+marker, 1 = DEG only)
      - top pathways per celltype comparison
    Outputs in reference_summary.
    """
    out_dir = Path(out_dir)
    deg_dir = Path(deg_dir)
    celltype_dir = Path(celltype_dir)
    ref_dir = out_dir / "08_reference_summary"
    ref_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"[SUMMARY-CT-DEG] Building celltype DEG + marker + pathway summary for {analysis_name}")

    markers_all_path = celltype_dir / "celltype_marker_genes_celltype_ALL.csv"
    marker_pairs = set()
    if markers_all_path.exists():
        try:
            markers_all = pd.read_csv(markers_all_path)
            if {"group", "names"}.issubset(markers_all.columns):
                for _, row in markers_all.iterrows():
                    ct = str(row["group"])
                    g = str(row["names"])
                    marker_pairs.add((ct, g))
                logger.info(f"[SUMMARY-CT-DEG] Loaded {len(markers_all)} marker rows from {markers_all_path.name}")
            else:
                logger.warning(
                    f"[SUMMARY-CT-DEG] Marker file {markers_all_path.name} missing 'group' or 'names' columns."
                )
        except Exception as e:
            logger.warning(f"[SUMMARY-CT-DEG] Failed to read {markers_all_path}: {e}")
    else:
        logger.info(
            f"[SUMMARY-CT-DEG] No marker ALL file found at {markers_all_path}; "
            "DEGs will still be summarised but without marker scores."
        )

    pathways_combined_dir = None
    if ct_deg_pathway_dir is not None:
        ct_deg_pathway_dir = Path(ct_deg_pathway_dir)
        candidate = ct_deg_pathway_dir / "pathways" / "combined"
        if candidate.exists():
            pathways_combined_dir = candidate
            logger.info(f"[SUMMARY-CT-DEG] Using combined pathway dir: {pathways_combined_dir}")

    ct_deg_files = sorted(deg_dir.glob("*/*.csv"))
    if not ct_deg_files:
        logger.info("[SUMMARY-CT-DEG] No celltype-specific DEG CSVs found.")
        return

    genelevel_rows = []
    overview_rows = []
    summary_txt_lines = [
        f"=== Cell-type DEGs + markers + pathways summary for {analysis_name} ===",
        "",
        "Score legend:",
        "  2 = gene is DEG AND cell-type marker",
        "  1 = gene is DEG only",
        "",
    ]

    for f in ct_deg_files:
        try:
            df = pd.read_csv(f)
        except Exception as e:
            logger.warning(f"[SUMMARY-CT-DEG] Failed to read {f.name}: {e}")
            continue

        if not {"names", "celltype", "comparison", "logfoldchanges", "pvals_adj"}.issubset(df.columns):
            logger.warning(
                f"[SUMMARY-CT-DEG] File {f.name} missing required columns; skipping."
            )
            continue

        ct_name = str(df["celltype"].iloc[0])
        comparison_label = str(df["comparison"].iloc[0])

        pcol = "pvals_adj"
        lfc_col = "logfoldchanges"

        df_deg = df.dropna(subset=[pcol, lfc_col]).copy()
        df_deg = df_deg[df_deg[pcol] < 0.05]
        df_deg = df_deg[df_deg[lfc_col].abs() > 0.25]

        if df_deg.empty:
            summary_txt_lines.append(
                f"Example: {ct_name} DEGs ({comparison_label}) → 0 DEGs passing filters."
            )
            summary_txt_lines.append("")
            continue

        is_marker_list = []
        score_list = []
        for _, row in df_deg.iterrows():
            gene = str(row["names"])
            is_marker = int((ct_name, gene) in marker_pairs)
            score = 2 if is_marker == 1 else 1
            is_marker_list.append(is_marker)
            score_list.append(score)

        df_deg["is_marker_for_celltype"] = is_marker_list
        df_deg["celltype_DEG_marker_score"] = score_list

        genelevel_rows.append(
            df_deg.assign(
                celltype=ct_name,
                comparison=comparison_label,
            )[["celltype", "comparison", "names", lfc_col, pcol,
               "is_marker_for_celltype", "celltype_DEG_marker_score"]].rename(
                columns={
                    "names": "gene",
                    lfc_col: "logFC",
                    pcol: "pval_adj_or_p",
                }
            )
        )

        n_deg = df_deg.shape[0]
        n_markers = int(df_deg["is_marker_for_celltype"].sum())

        df_markers_only = df_deg[df_deg["is_marker_for_celltype"] == 1].copy()
        df_markers_only = df_markers_only.sort_values(pcol)
        top_marker_genes = df_markers_only["names"].head(min(10, len(df_markers_only))).tolist()
        top_marker_genes_str = "; ".join(map(str, top_marker_genes)) if top_marker_genes else ""

        top_pathway_str_list = []
        if pathways_combined_dir is not None:
            prefix = f.stem
            candidate_files = [
                pathways_combined_dir / f"{prefix}_combined_pathways_DEDUP.csv",
                pathways_combined_dir / f"{prefix}_combined_pathways_RAW.csv",
                pathways_combined_dir / f"{prefix}_combined_pathways.csv",
            ]
            comb_file = None
            for cf in candidate_files:
                if cf.exists():
                    comb_file = cf
                    break

            if comb_file is not None:
                try:
                    pdf = pd.read_csv(comb_file)
                    if not pdf.empty:
                        if "Combined Score" in pdf.columns:
                            pdf = pdf.sort_values("Combined Score", ascending=False)
                        elif "Adjusted P-value" in pdf.columns:
                            pdf = pdf.sort_values("Adjusted P-value", ascending=True)
                        pdf_top = pdf.head(top_n_pathways)
                        for _, prow in pdf_top.iterrows():
                            gs = str(prow.get("Biological_Database", "NA"))
                            term = str(prow.get("Pathways", "NA"))
                            adjp = float(prow.get("Adjusted P-value", np.nan)) if "Adjusted P-value" in pdf.columns else np.nan
                            if np.isnan(adjp):
                                top_pathway_str_list.append(f"{gs}:: {term}")
                            else:
                                top_pathway_str_list.append(f"{gs}:: {term} (adj_p={adjp:.2e})")
                except Exception as e:
                    logger.warning(f"[SUMMARY-CT-DEG] Failed to load pathways for {prefix}: {e}")

        top_pathways_str = "; ".join(top_pathway_str_list) if top_pathway_str_list else ""

        overview_rows.append(
            {
                "celltype": ct_name,
                "comparison": comparison_label,
                "n_deg_filtered": n_deg,
                "n_deg_markers": n_markers,
                "top_deg_marker_genes": top_marker_genes_str,
                "top_pathways": top_pathways_str,
            }
        )

        summary_txt_lines.append(
            f"Example: {ct_name} DEGs ({comparison_label}) "
            f"→ {n_deg} DEGs passing filters (|logFC|>0.25, p<0.05)"
        )
        summary_txt_lines.append(
            f"  marker DEGs (also cell-type markers): {n_markers}"
        )
        if top_marker_genes:
            summary_txt_lines.append("  Top marker DEGs:")
            for g in top_marker_genes[: min(5, len(top_marker_genes))]:
                summary_txt_lines.append(f"    - {g} (score=2)")
        else:
            summary_txt_lines.append("  Top marker DEGs: none (no DEG overlaps with markers).")

        if top_pathway_str_list:
            summary_txt_lines.append("  Top pathways:")
            for pw in top_pathway_str_list[: min(5, len(top_pathway_str_list))]:
                summary_txt_lines.append(f"    - {pw}")
        else:
            summary_txt_lines.append("  Top pathways: none (no enriched terms found).")

        summary_txt_lines.append("")

    if genelevel_rows:
        genelevel_df = pd.concat(genelevel_rows, axis=0, ignore_index=True)
        genelevel_file = ref_dir / "celltype_DEG_marker_genelevel_summary.csv"
        genelevel_df.to_csv(genelevel_file, index=False)
        logger.info(f"[SUMMARY-CT-DEG] Wrote gene-level DEG+marker summary: {genelevel_file}")

    if overview_rows:
        overview_df = pd.DataFrame(overview_rows)
        overview_file = ref_dir / "celltype_DEG_marker_pathway_overview.csv"
        overview_df.to_csv(overview_file, index=False)
        logger.info(f"[SUMMARY-CT-DEG] Wrote celltype DEG+marker+pathway overview: {overview_file}")

    summary_txt_file = ref_dir / "celltype_DEG_marker_pathway_summary.txt"
    summary_txt_file.write_text("\n".join(summary_txt_lines), encoding="utf-8")
    logger.info(f"[SUMMARY-CT-DEG] Wrote text summary: {summary_txt_file}")
