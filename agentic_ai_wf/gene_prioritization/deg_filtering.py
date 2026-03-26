from pathlib import Path
import asyncio
import datetime
import time
import pandas as pd
import numpy as np
import os
from typing import Any, Dict, List, Optional, Tuple

from .string_ppi import string_ppi_lookup
from .aligner import aligner
from .helpers import logger
from .config import cgc_file, jl_zscore_file, DEG_FILTERING_OUTPUT_DIR
from .hormonizer import run_hormonizer
from .graph_data import get_disease_scores
from .llm_gene_prioritization import add_llm_scores


def _emit_ui_log(
    workflow_logger: Any,
    event_loop: Optional[asyncio.AbstractEventLoop],
    level: str,
    message: str,
    **kwargs: Any,
) -> None:
    """Emit log to UI via workflow_logger when running in thread pool."""
    if not workflow_logger or not event_loop:
        return
    try:
        agent_name = "Gene Prioritization Agent"
        step = "gene_prioritization"

        async def _do_log() -> None:
            try:
                if level == "info":
                    await workflow_logger.info(agent_name=agent_name, message=message, step=step, **kwargs)
                elif level == "warning":
                    await workflow_logger.warning(agent_name=agent_name, message=message, step=step, **kwargs)
                elif level == "error":
                    await workflow_logger.error(agent_name=agent_name, message=message, step=step, **kwargs)
            except Exception:
                pass

        def _schedule() -> None:
            try:
                asyncio.create_task(_do_log())
            except Exception:
                pass

        event_loop.call_soon_threadsafe(_schedule)
    except Exception:
        pass


def _extract_disease_gene_score_payload(disease_response: dict) -> Optional[dict]:
    """
    Parse GeneCard score payload from get_disease_scores().

    DRF wraps genecards_scorer output as JSON ``{ ..., "data": { "data": { ... } } }``.
    On Neo4j failure, genecards_scorer returns ``{}`` but the view still returns HTTP 200,
    so the inner dict may lack ``gene_symbol`` — must not KeyError.
    """
    if disease_response.get("status") != "success":
        return None
    body = disease_response.get("data")
    if not isinstance(body, dict):
        return None
    inner = body.get("data", body)
    if not isinstance(inner, dict):
        return None
    symbols = inner.get("gene_symbol")
    if not isinstance(symbols, list):
        return None
    return inner


def _gene_score_dicts_from_payload(payload: dict) -> Tuple[Dict[Any, Any], Dict[Any, Any], Dict[Any, Any], int]:
    """Build symbol -> score maps; tolerate missing or short parallel lists."""
    syms: List[Any] = [s for s in (payload.get("gene_symbol") or []) if s is not None]
    n = len(syms)
    if n == 0:
        return {}, {}, {}, 0

    def _aligned(name: str) -> List[Any]:
        v = payload.get(name)
        if not isinstance(v, list):
            return [None] * n
        if len(v) < n:
            return list(v) + [None] * (n - len(v))
        return v[:n]

    gsc = _aligned("gene_score")
    dsc = _aligned("disorder_score")
    dty = _aligned("disorder_type")
    return (
        dict(zip(syms, gsc)),
        dict(zip(syms, dsc)),
        dict(zip(syms, dty)),
        n,
    )


def load_data(df, patient_prefix='patient'):
    """
    Load the data from the dataframe.
    Args:
        df (pd.DataFrame): The dataframe containing the data.
    Returns:
        df (pd.DataFrame): The dataframe containing the data.
        log2fc_column (str): The column name of the log2FC column.
        p_value_column (str): The column name of the p-value column.

    """
    log2fc_column = [col for col in df.columns if col.lower().startswith(
        patient_prefix) and col.lower().endswith('log2fc')][0]
    p_value_column = [col for col in df.columns if col.lower().startswith(
        patient_prefix) and col.lower().endswith('p-value')][0]

    # print(log2fc_column, p_value_column)

    return df, log2fc_column, p_value_column


def significance_cutoff(df, log2fc_column, p_value_column, lfc=1, p_value=0.05):
    """
    Apply significance cutoff to the dataframe.
    Args:
        df (pd.DataFrame): The dataframe containing the data.
        log2fc_column (str): The column name of the log2FC column.
        p_value_column (str): The column name of the p-value column.
        lfc (float): The log2FC threshold.
        p_value (float): The p-value threshold.
    Returns:
        df (pd.DataFrame): The dataframe containing the data.
    """
    df = df[df[log2fc_column].abs() > lfc]
    df = df[df[p_value_column] < p_value]

    return df


def expression_concordance(df):
    """
    Apply expression concordance to the dataframe.
    Args:
        df (pd.DataFrame): The dataframe containing the data.
    Returns:
        df (pd.DataFrame): The dataframe containing the data.
    """
    # cohort_columns = [col for col in df.columns if col.startswith('cohort') and col.endswith('log2FC')]
    cohort_log2fc_cols = [col for col in df.columns if col.lower(
    ).startswith('gse') and col.lower().endswith('log2fc')]

    def get_dcs(row):
        if row['Trend_Consensus'] == 'ALIGNED':
            return 1
        else:
            return -1

        # dcs = []
        # patient_direction = np.sign(row[patient_log2fc_col])
        # for col in cohort_log2fc_cols:
        #     # only consider the cohorts that are not nan
        #     if pd.isna(row[col]):
        #         continue
        #     cohort_direction = np.sign(row[col])
        #     if cohort_direction != 0:
        #         dcs.append(1 if cohort_direction == patient_direction else 0)
        # # if more than 50% of the cohorts are concordant, return 1, otherwise return -1
        # if len(dcs) > 0:
        #     if sum(dcs) / len(dcs) >= 0.5:
        #         return 1
        #     else:
        #         return -1
        # else:
        #     return 1

    def get_csc(row):
        # only consider the cohorts that are not nan
        return sum(abs(row[col]) > 1 for col in cohort_log2fc_cols if not pd.isna(row[col]))

    def get_cfc(row):
        valid = [row[col]
                 for col in cohort_log2fc_cols if not pd.isna(row[col])]
        return np.mean(valid) if valid else np.nan

    df['DCS'] = df.apply(get_dcs, axis=1)
    df['CSC'] = df.apply(get_csc, axis=1)
    df['CFC'] = df.apply(get_cfc, axis=1)

    # Apply concordance filtering
    df['keep'] = df.apply(lambda row: row['CSC'] >=
                          2 and row['DCS'] == 1, axis=1)
    df['novel'] = df['CSC'] == 0

    filtered_df = df[df['keep']]

    # return filtered_df, df[df['novel']]
    return df


z_score_file = pd.read_csv(jl_zscore_file)


def build_zscore_lookup(z_df):
    """
    Build a lookup table for the z-scores.
    Args:
        z_df (pd.DataFrame): The dataframe containing the z-scores.
    Returns:
        lookup (dict): A dictionary containing the z-scores for each biomarker and disease.
    """
    lookup = {}
    for _, row in z_df.iterrows():
        biomarker = row['biomarker_name']
        diseases = str(row['disease']).split('; ')
        z_scores = str(row['z-score']).split('; ')
        if len(diseases) != len(z_scores):
            continue
        disease_z = dict(zip([d.lower()
                         for d in diseases], map(float, z_scores)))
        lookup[biomarker] = disease_z
    return lookup


zscore_lookup = build_zscore_lookup(z_score_file)

# Fast single lookup


def fast_zscore_lookup(gene, disease):
    """
    Fast lookup for the z-scores.
    Args:
        gene (str): The gene name.
        disease (str): The disease name.
    Returns:
        z_score (float): The z-score for the gene and disease.
    """
    disease = disease.lower()
    return zscore_lookup.get(gene, {}).get(disease, None)

# Vectorized application


def add_relevance_score(df, disease):
    """
    Add the relevance score to the dataframe.
    Args:
        df (pd.DataFrame): The dataframe containing the data.
        disease (str): The disease name.
    Returns:
        df (pd.DataFrame): The dataframe containing the data.
    """

    df['JL_score'] = df['Gene'].apply(lambda g: fast_zscore_lookup(g, disease))

    # add CGC column
    cgc = pd.read_csv(cgc_file, sep='\t')
    cgc.set_index('ID', inplace=True)
    cgc.columns = ['CGC']

    cgc['CGC'] = cgc['CGC'].apply(lambda x: 1 if x == "CGC" else 0)

    df = df.merge(cgc, left_on='Gene', right_index=True, how='left')
    df['CGC'] = df['CGC'].fillna(0)

    return df



def assign_tier(row):
    """
    Assign a tier to the row based on a flag-based composite logic.
    
    Args:
        row (pd.Series): The row containing the data.

    Returns:
        str: The assigned tier.
    """
    # # Extract values safely
    # jl_score = row.get("JL_score", None)
    # gc_score = row.get("GC_score", None)
    # degree = row.get("PPI_Degree", 0)
    # csc = row.get("CSC", 0)
    # dcs = row.get("DCS", 0)
    # abs_lfc = abs(row.get("Patient_LFC_mean", 0))

    # # Compute individual flags
    # has_gc = gc_score is not None and gc_score >= 10
    # has_jl = jl_score is not None and jl_score >= 3
    # has_ppi = degree >= 3
    # has_cohort = csc >= 2 and dcs == 1

    # # Compute flag sum
    # flag_sum = sum([has_gc, has_jl, has_ppi, has_cohort])

    # # Assign tier based on flag sum
    # if flag_sum >= 2:
    #     return "Tier 1"
    # elif flag_sum == 1:
    #     return "Tier 2"
    # elif flag_sum == 0 and abs_lfc >= 1:
    #     return "Tier 3"
    # else:
    #     return "Non-Significant"
    composite_score = row['Composite_Score']
    if composite_score > 0.5:
        return "Tier 1"
    elif composite_score > 0.25:
        return "Tier 2"
    else:
        return "Tier 3"


# import time to measure the time taken by the function
import time
import datetime
from pathlib import Path

def run_deg_filtering(
    patient_prefix: Optional[str] = None,
    deg_base_dir: Optional[Path] = None,
    disease_name: Optional[str] = None,
    analysis_id: Optional[str] = None,
    output_dir: Optional[Path] = None,
    causal: bool = False,
    workflow_logger: Any = None,
    event_loop: Optional[asyncio.AbstractEventLoop] = None,
) -> Path:
    """
    Run the deg filtering pipeline.
    Returns:
        Path to the final prioritized DEG file.
    """
    start_time = time.time()
    emit = lambda level, msg, **kw: _emit_ui_log(workflow_logger, event_loop, level, msg, **kw)

    if output_dir is None:
        output_dir = Path(DEG_FILTERING_OUTPUT_DIR)
    if analysis_id is None:
        analysis_id = "analysis_id"
    if disease_name is None:
        disease_name = "disease_name"

    if deg_base_dir is None:
        deg_base_dir = Path(DEG_FILTERING_OUTPUT_DIR)

    sanitized_disease_name = disease_name.replace(
        " ", "_").replace("/", "_").replace("\\", "_")
    final_df_path = f"{output_dir}/{sanitized_disease_name}_DEGs_prioritized.csv"

    emit("info", "Starting gene prioritization pipeline")

    if os.path.exists(final_df_path):
        logger.info(f"Found existing prioritized file: {final_df_path}")
        emit("info", "Found existing prioritized file; applying LLM scoring only")

        df = pd.read_csv(final_df_path)

        if causal:
            logger.info("Causal flag set; skipping LLM-based gene prioritization.")
            emit("info", "Causal mode: skipping LLM-based scoring")
        else:
            logger.info("Adding LLM-based gene prioritization to existing data...")
            emit("info", "Adding LLM-based gene prioritization to existing data")
            df = add_llm_scores(df, disease_name)

            df.to_csv(final_df_path, index=False, encoding='utf-8-sig')

        elapsed = time.time() - start_time
        emit("info", f"Gene prioritization completed in {elapsed:.1f}s")
        return Path(final_df_path).resolve()

    logger.info("No existing prioritized file found. Running full DEG filtering pipeline...")
    emit("info", "Merging and aligning DEG files with harmonizer")

    final_output_path = run_hormonizer(
        deg_base_dir, output_dir=output_dir, analysis_id=analysis_id, causal=causal)

    df = pd.read_csv(final_output_path)
    emit("info", "Applying sample alignment and loading expression data")
    df = aligner(df, patient_prefix=analysis_id)
    df, log2fc_column, p_value_column = load_data(
        df, patient_prefix=analysis_id)

    if causal:
        logger.info("Causal flag set; skipping significance cutoff (p<0.05, |log2FC|>1).")
        emit("info", "Causal mode: skipping significance cutoff")
    else:
        n_before = len(df)
        df = significance_cutoff(df, log2fc_column, p_value_column, lfc=1, p_value=0.05)
        n_after = len(df)
        emit("info", f"Applied significance cutoff (p<0.05, |log2FC|>1): {n_after} genes retained from {n_before}")

    emit("info", "Computing expression concordance across cohorts")
    df = expression_concordance(df)

    emit("info", "Adding relevance scores (JL, CGC) and disease annotations")
    df = add_relevance_score(df, disease=disease_name)
    # score_dict = fetch_genecards_scores(disease)
    # score_dict = gene_card_score(disease_name)
    # df['GC_score'] = df['Gene'].map(score_dict)  # type: ignore


    emit("info", "Fetching disease-specific gene scores from knowledge graph")
    disease_response = get_disease_scores(disease_name)
    payload = _extract_disease_gene_score_payload(disease_response)
    if payload is not None:
        gene_card_score_dict, disorder_score_dict, disorder_type_dict, n_genes = (
            _gene_score_dicts_from_payload(payload)
        )
        df["Gene_Score"] = df["Gene"].map(gene_card_score_dict)
        df["Disorder_Score"] = df["Gene"].map(disorder_score_dict)
        df["Disorder_Type"] = df["Gene"].map(disorder_type_dict)
        emit("info", f"Mapped disease scores for {n_genes} genes from knowledge graph")
        logger.info("Mapped graph disease scores for %s genes", n_genes)
    else:
        msg = disease_response.get("message", "Unknown error")
        if disease_response.get("status") == "success":
            logger.warning(
                "Gene-card-score response missing usable gene_symbol list (empty Neo4j result or malformed payload)"
            )
            emit(
                "warning",
                "Disease graph scores missing or empty; continuing without Gene_Score mapping",
            )
        else:
            logger.warning("Failed to get disease scores: %s", msg)
            emit("warning", f"Disease scores unavailable: {msg}")
        df["Gene_Score"] = None
        df["Disorder_Score"] = None
        df["Disorder_Type"] = None

    emit("info", "Looking up protein–protein interactions (STRING)")
    df = string_ppi_lookup(df)

    df.to_csv(
        f"{output_dir}/{sanitized_disease_name}_DEGs_filtered.csv", index=False)
    # add composite evidence score
    # df.to_csv("deg_filtered.csv", index=False)
    # add composite evidence score
    patient_adj_pval_col = [col for col in df.columns if col.lower().startswith(patient_prefix) and col.lower().endswith('adj-p-value')][0]
    # print(patient_adj_pval_col)
    # Calculate normalization factors with safety checks to avoid division by zero
    patient_lfc_max = abs(df['Patient_LFC_mean'].fillna(0)).max()
    ppi_degree_max = df['PPI_Degree'].fillna(0).max()
    gene_score_max = df['Gene_Score'].fillna(0).max()
    disorder_score_max = df['Disorder_Score'].fillna(0).max()
    
    p_value = -np.log10(df[patient_adj_pval_col].fillna(0))
    pi_value = abs(df['Patient_LFC_mean'].fillna(0)) * p_value
    df['Composite_Score'] = (
        # (abs(df['Patient_LFC_mean'].fillna(0)) / patient_lfc_max if patient_lfc_max > 0 else 0) * 0.45 +
        ((pi_value- pi_value.min()) / (pi_value.max() - pi_value.min())) * 0.45 +
        (df['PPI_Degree'].fillna(0) / ppi_degree_max if ppi_degree_max > 0 else 0) * 0.1 +
        df['CGC'].fillna(0) * 0.2 +
        (df['Gene_Score'].fillna(0) / gene_score_max if gene_score_max > 0 else 0) * 0.1 +
        (df['Disorder_Score'].fillna(0) / disorder_score_max if disorder_score_max > 0 else 0) * 0.15
    )
    # df['Composite_Score'] = (
    #     df['GC_score'].fillna(0) * 0.30 +
    #     df['JL_score'].fillna(0) * 0.30 +
    #     df['Cohort_LFC_mean'].fillna(0) * 0.15 +
    #     df['CGC'].fillna(0) * 0.10 +
    #     df['PPI_Degree'].fillna(0) * 0.15
    # )
    df['Tier'] = df.apply(assign_tier, axis=1)

    # Step 1: Copy original Composite_Score to a new column
    df['Non_CGC_Composite_Score'] = df['Composite_Score'].copy()

    # Step 2: Count how many of the four columns exceed thresholds
    cond1_count = (
        # ((df['Patient_LFC_mean'].fillna(0)/ patient_lfc_max) > 0.25).astype(int) +
        (((pi_value- pi_value.min()) / (pi_value.max() - pi_value.min())) > 0.25).astype(int) +
        ((df['PPI_Degree'].fillna(0)/ ppi_degree_max) > 0.25).astype(int) +
        ((df['Gene_Score'].fillna(0)/ gene_score_max) > 0.25).astype(int) +
        ((df['Disorder_Score'].fillna(0)/ disorder_score_max) > 0.25).astype(int)
    )

    cond2_count = (
        # ((df['Patient_LFC_mean'].fillna(0)/ patient_lfc_max) > 0.5).astype(int) +
        (((pi_value- pi_value.min()) / (pi_value.max() - pi_value.min())) > 0.5).astype(int) +
        ((df['PPI_Degree'].fillna(0)/ ppi_degree_max) > 0.5).astype(int) +
        ((df['Gene_Score'].fillna(0)/ gene_score_max) > 0.5).astype(int) +
        ((df['Disorder_Score'].fillna(0)/ disorder_score_max) > 0.5).astype(int)
    )
    # print(cond1_count)
    # print(cond2_count)
    # Step 3: Define the two adjustment conditions
    condition1 = (
        (df['CGC'].fillna(0) == 0) &
        (df['Composite_Score'] < 0.25) &
        (cond1_count >= 2)
    )

    condition2 = (
        (df['CGC'].fillna(0) == 0) &
        (df['Composite_Score'] >= 0.25) &
        (df['Composite_Score'] <= 0.5) &
        (cond2_count >= 2)
    )

    # Step 4: Apply adjustments to the new column
    df.loc[condition1 | condition2, 'Non_CGC_Composite_Score'] *= 1.2
    
    df['Non_CGC_Tier'] = df['Non_CGC_Composite_Score'].apply(lambda x: "Tier 1" if x > 0.5 else ("Tier 2" if x > 0.25 else "Tier 3"))

    emit("info", "Computing composite evidence score and assigning tiers (Tier 1/2/3)")

    if causal:
        logger.info("Causal flag set; skipping LLM-based gene prioritization.")
        emit("info", "Causal mode: skipping LLM-based gene prioritization")
    else:
        logger.info("Adding LLM-based gene prioritization...")
        emit("info", "Running LLM-based gene prioritization (this may take a moment)")
        df = add_llm_scores(df, disease_name)

    df.to_csv(final_df_path, index=False)

    end_time = time.time()
    elapsed = end_time - start_time
    n_genes = len(df)
    emit("info", f"Gene prioritization complete: {n_genes} genes saved in {elapsed:.1f}s")
    logger.info(f"Time taken: {elapsed} seconds")

    return Path(final_df_path).resolve()


# main(r"C:\Ayass Bio Work\Agentic_AI_ABS\GenePrioritization\agentic_ai_abs\combined_DEGs_matrix_annotated (1).csv")
