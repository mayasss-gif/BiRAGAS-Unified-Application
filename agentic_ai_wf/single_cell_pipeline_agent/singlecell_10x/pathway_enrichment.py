# pathway_enrichment.py

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Optional
import time
from .config_cli import logger

# Optional gseapy for multi-database pathway enrichment (Enrichr)
try:
    import gseapy as gp
    GSEAPY_AVAILABLE = True
except ImportError:
    GSEAPY_AVAILABLE = False

# Configuration for enrichment API calls
ENRICHR_TIMEOUT = 10  # seconds per API call
ENRICHR_MAX_RETRIES = 2  # max retries per database
ENRICHR_MAX_CONSECUTIVE_FAILURES = 5  # stop after N consecutive failures


def deduplicate_pathways_semantic(
    df: pd.DataFrame,
    combined_dir: Path,
    prefix: str,
    sim_threshold: float = 0.9,
) -> pd.DataFrame:
    """
    Optional semantic deduplication using SentenceTransformer (MiniLM) + FAISS.
    Falls back to simple string-based dedup if libraries are not available.
    Writes a log file describing which pathways were removed and why.
    """
    combined_dir = Path(combined_dir)
    combined_dir.mkdir(parents=True, exist_ok=True)
    log_file = combined_dir / f"{prefix}_pathway_dedup_log.txt"

    df = df.reset_index(drop=True)

    def _score(row):
        p = row.get("Adjusted P-value", np.nan)
        cs = row.get("Combined Score", 0.0)
        if pd.isna(p):
            p = 1.0
        return (p, -cs)

    order = sorted(range(len(df)), key=lambda i: _score(df.iloc[i]))

    log_lines = []
    log_lines.append("=== Pathway deduplication (semantic) ===")
    log_lines.append(f"Original rows: {len(df)}")
    log_lines.append(f"Similarity threshold: {sim_threshold}")
    log_lines.append("Priority: min(Adjusted P-value), then max(Combined Score)")
    log_lines.append("")

    try:
        from sentence_transformers import SentenceTransformer
        import faiss

        model = SentenceTransformer("all-MiniLM-L6-v2")

        names = df["Pathways"].astype(str).tolist()
        embeddings = model.encode(names, convert_to_numpy=True, show_progress_bar=False)
        embeddings = embeddings.astype("float32")

        faiss.normalize_L2(embeddings)
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)

        keep_mask = np.zeros(len(df), dtype=bool)

        kept_indices = []
        for idx in order:
            if not kept_indices:
                index.add(embeddings[idx:idx+1])
                kept_indices.append(idx)
                keep_mask[idx] = True
                log_lines.append(f"KEEP idx={idx}: {df.loc[idx, 'Biological_Database']} :: {df.loc[idx, 'Pathways']}")
                continue

            D, I = index.search(embeddings[idx:idx+1], k=1)
            sim = float(D[0][0])
            dup_idx = int(I[0][0])

            if sim >= sim_threshold:
                log_lines.append(
                    f"REMOVE idx={idx} (sim={sim:.3f} vs idx={dup_idx}) → "
                    f"{df.loc[idx, 'Biological_Database']} :: {df.loc[idx, 'Pathways']}"
                )
            else:
                index.add(embeddings[idx:idx+1])
                kept_indices.append(idx)
                keep_mask[idx] = True
                log_lines.append(f"KEEP idx={idx}: {df.loc[idx, 'Biological_Database']} :: {df.loc[idx, 'Pathways']}")

        new_df = df.loc[keep_mask].reset_index(drop=True)
        log_lines.append("")
        log_lines.append(f"Final rows after semantic dedup: {len(new_df)}")

    except Exception as e:
        log_lines.append("")
        log_lines.append(f"Semantic dedup skipped (reason: {e}).")
        log_lines.append("Using simple exact-string dedup instead.")
        df["_score_adj"] = df["Adjusted P-value"].fillna(1.0)
        df["_score_comb"] = -df.get("Combined Score", 0.0).fillna(0.0)
        df = df.sort_values(["Biological_Database", "Pathways", "_score_adj", "_score_comb"])
        new_df = df.drop_duplicates(subset=["Biological_Database", "Pathways"], keep="first").copy()
        new_df = new_df.drop(columns=["_score_adj", "_score_comb"])
        new_df = new_df.reset_index(drop=True)
        log_lines.append(f"Final rows after string dedup: {len(new_df)}")

    log_file.write_text("\n".join(log_lines), encoding="utf-8")
    logger.info(f"[ENRICHR-DEDUP] Wrote pathway deduplication log: {log_file}")
    return new_df


def run_enrichr_multidb(
    gene_list,
    out_dir: Path,
    prefix: str,
    pval_cutoff: float = 0.05,
):
    """
    Run Enrichr-based enrichment on a list of gene symbols across multiple databases.

    Output columns (per file):

        Biological_Database    Pathways    Overlap    P-value    Adjusted P-value
        Odds Ratio    Combined Score    Genes
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not GSEAPY_AVAILABLE:
        logger.warning(f"[ENRICHR] gseapy not available; skipping enrichment for {prefix}.")
        return

    genes = [g for g in set(map(str, gene_list)) if g not in (None, "", "nan")]
    if len(genes) < 5:
        logger.info(f"[ENRICHR] Not enough genes for enrichment ({len(genes)}) for {prefix}. Skipping.")
        return

    gene_sets = [
        "GO_Biological_Process_2021",
        "GO_Molecular_Function_2021",
        "GO_Cellular_Component_2021",
        "KEGG_2021_Human",
        "Reactome_2022",
        "WikiPathways_2019_Human",
    ]

    gs_meta = {
        "GO_Biological_Process_2021": (
            "Gene Ontology – Biological Process (GO BP)",
            "GO_BP",
        ),
        "GO_Molecular_Function_2021": (
            "Gene Ontology – Molecular Function (GO MF)",
            "GO_MF",
        ),
        "GO_Cellular_Component_2021": (
            "Gene Ontology – Cellular Component (GO CC)",
            "GO_CC",
        ),
        "KEGG_2021_Human": (
            "KEGG Pathway Database",
            "KEGG",
        ),
        "Reactome_2022": (
            "Reactome Pathway Database",
            "Reactome",
        ),
        "WikiPathways_2019_Human": (
            "WikiPathways Database",
            "WikiPathways",
        ),
    }

    logger.info(f"[ENRICHR] Running enrichment for {prefix} on {len(genes)} genes...")

    base_pathway_dir = out_dir / "pathways"
    base_pathway_dir.mkdir(parents=True, exist_ok=True)
    combined_dir = base_pathway_dir / "combined"
    combined_dir.mkdir(parents=True, exist_ok=True)

    combined_results = []
    consecutive_failures = 0

    def _plot_top_bar(df: pd.DataFrame, out_png: Path, top_n: int = 20):
        if "Combined Score" not in df.columns:
            return
        df_plot = df.sort_values("Combined Score", ascending=False).head(top_n)
        if df_plot.empty:
            return

        plt.figure(figsize=(12, max(5.0, 0.5 * len(df_plot))))
        colors = plt.cm.tab20(np.linspace(0, 1, len(df_plot)))
        plt.barh(df_plot["Pathways"], df_plot["Combined Score"], color=colors)
        plt.gca().invert_yaxis()
        plt.xlabel("Combined Score", fontsize=13)
        plt.ylabel("Pathways", fontsize=12)
        plt.title(out_png.stem.replace("_", " "), fontsize=14)
        plt.xticks(fontsize=11)
        plt.yticks(fontsize=9)
        plt.tight_layout()
        plt.savefig(out_png, dpi=300)
        plt.close()

    for gs in gene_sets:
        # Early exit if too many consecutive failures (likely network issue)
        if consecutive_failures >= ENRICHR_MAX_CONSECUTIVE_FAILURES:
            logger.warning(
                f"[ENRICHR] Stopping enrichment for {prefix} after {consecutive_failures} "
                f"consecutive failures (likely network/server issue). "
                f"Remaining databases will be skipped."
            )
            break
        
        friendly_name, db_folder = gs_meta.get(gs, (gs, "Other"))
        
        # Retry logic with timeout
        enr = None
        last_error = None
        start_time = time.time()
        for attempt in range(ENRICHR_MAX_RETRIES + 1):
            try:
                # Set a timeout for the API call
                enr = gp.enrichr(
                    gene_list=genes,
                    gene_sets=gs,
                    outdir=None,
                    cutoff=pval_cutoff,
                )
                elapsed = time.time() - start_time
                if elapsed > ENRICHR_TIMEOUT:
                    logger.warning(
                        f"[ENRICHR] Enrichment for {prefix}, gene_set={gs} took {elapsed:.1f}s "
                        f"(exceeded timeout of {ENRICHR_TIMEOUT}s)"
                    )
                consecutive_failures = 0  # Reset on success
                break
            except Exception as e:
                last_error = e
                elapsed = time.time() - start_time
                if attempt < ENRICHR_MAX_RETRIES:
                    wait_time = min(2 ** attempt, 5)  # Exponential backoff, max 5s
                    logger.info(
                        f"[ENRICHR] Attempt {attempt + 1}/{ENRICHR_MAX_RETRIES + 1} failed for "
                        f"{prefix}, gene_set={gs} (took {elapsed:.1f}s). Retrying in {wait_time}s..."
                    )
                    time.sleep(wait_time)
                    start_time = time.time()  # Reset timer for retry
                else:
                    consecutive_failures += 1
                    logger.warning(
                        f"[ENRICHR] Enrichment failed for {prefix}, gene_set={gs} after "
                        f"{ENRICHR_MAX_RETRIES + 1} attempts (took {elapsed:.1f}s): {last_error}"
                    )
        
        if enr is None:
            continue

        if enr is None or getattr(enr, "results", None) is None or enr.results.empty:
            logger.info(f"[ENRICHR] No significant terms for {prefix}, gene_set={gs}.")
            continue

        df_res = enr.results.copy()

        for col in ["Old P-value", "Old Adjusted P-value"]:
            if col in df_res.columns:
                df_res = df_res.drop(columns=[col])

        if "Gene_set" not in df_res.columns:
            df_res["Gene_set"] = gs

        if "Genes" not in df_res.columns:
            df_res["Genes"] = ""

        df_res["Biological_Database"] = friendly_name
        if "Term" in df_res.columns:
            df_res = df_res.rename(columns={"Term": "Pathways"})

        if "Gene_set" in df_res.columns:
            df_res = df_res.drop(columns=["Gene_set"])

        desired_cols = [
            "Biological_Database",
            "Pathways",
            "Overlap",
            "P-value",
            "Adjusted P-value",
            "Odds Ratio",
            "Combined Score",
            "Genes",
        ]
        cols = [c for c in desired_cols if c in df_res.columns] + [
            c for c in df_res.columns if c not in desired_cols
        ]
        df_res = df_res[cols]

        db_dir = base_pathway_dir / db_folder
        db_dir.mkdir(parents=True, exist_ok=True)

        out_file = db_dir / f"{prefix}_{db_folder}_enrichment.csv"
        df_res.to_csv(out_file, index=False)
        logger.info(f"[ENRICHR] Saved {out_file} with {df_res.shape[0]} rows.")

        barplot_file = db_dir / f"{prefix}_{db_folder}_top20_barplot.png"
        _plot_top_bar(df_res, barplot_file, top_n=20)

        combined_results.append(df_res)

    if combined_results:
        combined_df = pd.concat(combined_results, axis=0, ignore_index=True)

        combined_dedup = deduplicate_pathways_semantic(
            combined_df,
            combined_dir=combined_dir,
            prefix=prefix,
        )

        combined_file_raw = combined_dir / f"{prefix}_combined_pathways_RAW.csv"
        combined_df.to_csv(combined_file_raw, index=False)

        combined_file_clean = combined_dir / f"{prefix}_combined_pathways_DEDUP.csv"
        combined_dedup.to_csv(combined_file_clean, index=False)

        logger.info(
            f"[ENRICHR] Saved combined pathways for {prefix}: "
            f"{combined_file_raw.name} (raw), {combined_file_clean.name} (deduplicated)"
        )
    else:
        logger.info(f"[ENRICHR] No enrichment results to combine for {prefix}.")


def run_cluster_marker_enrichment(
    markers_all: pd.DataFrame,
    out_dir: Path,
    analysis_name: str,
    pval_col: str = "pvals_adj",
    pval_cutoff: float = 0.05,
    top_n: int = 200,
    cluster_celltype_map: Optional[Dict[str, str]] = None,
    max_clusters: Optional[int] = None,
):
    """
    For each Leiden cluster (markers_all['group']),
    run multi-DB Enrichr enrichment on top marker genes.
    
    Parameters
    ----------
    max_clusters : int, optional
        Maximum number of clusters to process. If None, processes all clusters.
        Useful for testing or when network is unreliable.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if "group" not in markers_all.columns or "names" not in markers_all.columns:
        logger.warning("[CLUSTER-ENRICH] markers_all must have 'group' and 'names' columns.")
        return

    use_pval = pval_col in markers_all.columns
    
    clusters = sorted(markers_all["group"].unique())
    if max_clusters is not None:
        clusters = clusters[:max_clusters]
        logger.info(f"[CLUSTER-ENRICH] Processing {len(clusters)} clusters (limited from {len(markers_all['group'].unique())})")
    
    total_failures = 0
    max_total_failures = len(clusters) * 2  # Allow some failures but not all

    for idx, cl in enumerate(clusters):
        # Early exit if too many total failures
        if total_failures > max_total_failures:
            logger.warning(
                f"[CLUSTER-ENRICH] Stopping enrichment after {total_failures} failures "
                f"(processed {idx}/{len(clusters)} clusters). Network/server may be unavailable."
            )
            break
        sub = markers_all[markers_all["group"] == cl].copy()

        if use_pval:
            sub = sub.sort_values(pval_col).dropna(subset=[pval_col])
            sig = sub[sub[pval_col] < pval_cutoff]
            if sig.empty:
                logger.info(f"[CLUSTER-ENRICH] No significant markers for cluster {cl}. Skipping.")
                continue
            gene_list = sig["names"].head(top_n).tolist()
        else:
            if "rank" not in sub.columns:
                logger.warning(f"[CLUSTER-ENRICH] No '{pval_col}' or 'rank' column for cluster {cl}. Skipping.")
                continue
            sub = sub.sort_values("rank")
            gene_list = sub["names"].head(top_n).tolist()

        if not gene_list:
            logger.info(f"[CLUSTER-ENRICH] Empty gene list for cluster {cl}. Skipping.")
            continue

        cl_str = str(cl)
        if cluster_celltype_map is not None and cl_str in cluster_celltype_map:
            cl_label = cluster_celltype_map[cl_str]
        else:
            cl_label = cl_str

        prefix = f"{analysis_name}_cluster_{cl_label}"
        try:
            run_enrichr_multidb(
                gene_list=gene_list,
                out_dir=out_dir,
                prefix=prefix,
                pval_cutoff=pval_cutoff,
            )
        except Exception as e:
            total_failures += 1
            logger.warning(
                f"[CLUSTER-ENRICH] Failed to run enrichment for cluster {cl}: {e}"
            )


def run_group_cluster_deg_enrichment_from_file(
    deg_table_path: Path,
    out_dir: Path,
    analysis_name: str,
    pval_col: str = "pvals_adj",
    logfc_col: str = "logfoldchanges",
    pval_cutoff: float = 0.05,
    logfc_cutoff: float = 0.25,
):
    """
    Use the table DE_<cluster_col>_ALL_GROUP_PAIRS.csv (from
    compute_de_between_groups_per_cluster) and run Enrichr (gseapy)
    for EACH (cluster, comparison) pair.
    """
    deg_table_path = Path(deg_table_path)
    if not deg_table_path.exists():
        logger.info(f"[GRP-CL-DEG-ENRICH] DEG table not found: {deg_table_path}")
        return

    df = pd.read_csv(deg_table_path)
    required_cols = {"names", "cluster", "comparison", pval_col, logfc_col}
    missing = required_cols - set(df.columns)
    if missing:
        logger.warning(
            f"[GRP-CL-DEG-ENRICH] Missing columns {missing} in {deg_table_path.name}; skipping."
        )
        return

    wrote_any = False
    for (cl, comp), sub in df.groupby(["cluster", "comparison"]):
        sub = sub.dropna(subset=[pval_col, logfc_col])
        sub = sub[sub[pval_col] < pval_cutoff]
        sub = sub[sub[logfc_col].abs() >= logfc_cutoff]

        if sub.empty:
            continue

        if not wrote_any:
            out_dir = Path(out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            wrote_any = True

        prefix = f"{analysis_name}_cluster{cl}_{comp}"
        gene_list = sub["names"].tolist()
        logger.info(
            f"[GRP-CL-DEG-ENRICH] Running enrichment for cluster={cl}, comparison={comp} "
            f"({len(gene_list)} genes, prefix={prefix})"
        )
        run_enrichr_multidb(
            gene_list=gene_list,
            out_dir=out_dir,
            prefix=prefix,
            pval_cutoff=pval_cutoff,
        )
