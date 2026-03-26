import os
import pandas as pd
from tqdm import tqdm
from datetime import datetime
# from sentence_transformers import SentenceTransformer
# import faiss
from agentic_ai_wf.utils.lazy_loader import get_sentence_model, get_faiss, get_sklearn_cosine_similarity
from .helpers import logger
import numpy as np

# ===============================================
# Sheryar's Deduplication Code
# ===============================================

SIM_THRESHOLD = 0.90
TOP_K = 10



# === UTILITIES ===
def load_table(file_path):
    ext = file_path.lower().split('.')[-1]
    return pd.read_csv(file_path) if ext == 'csv' else pd.read_excel(file_path)

def save_table(df, file_path):
    ext = file_path.lower().split('.')[-1]
    if ext == 'csv':
        df.to_csv(file_path, index=False)
    else:
        df.to_excel(file_path, index=False)

def count_genes(gene_string):
    return len(str(gene_string).split(",")) if pd.notna(gene_string) and str(gene_string).strip() else 0

# === STEP 1: SUMMARIZE DATABASE-LEVEL OCCURRENCES ===
def summarize_pathways_unique(file_path, output_path=None):
    logger.info("Starting pathway summarization by database redundancy...")
    df = pd.read_csv(file_path)
    df["normalized_pathway"] = df["Pathway"].astype(str).str.lower().str.strip()
    df["Calculated_Num_Genes"] = df["inputGenes"].apply(count_genes)

    summary_rows = []
    grouped = df.groupby("normalized_pathway")

    for pathway_name, group in grouped:
        top_entry = group.sort_values("Calculated_Num_Genes", ascending=False).iloc[0].copy()

        unique_dbs = group["DB_ID"].unique().tolist()

        intra_counts = group["DB_ID"].value_counts()
        intra_repetition = intra_counts.sum()
        intra_breakdown = "; ".join([f"{db}: {count}" for db, count in intra_counts.items()])


        top_entry["IntraDatabaseRepetition"] = intra_repetition
        top_entry["IntraDatabaseBreakdown"] = intra_breakdown

        summary_rows.append(top_entry)

    final_df = pd.DataFrame(summary_rows)
    final_df.drop(columns=["normalized_pathway"], inplace=True)

    if output_path:
        final_df.to_csv(output_path, index=False)
        logger.info(f"Summary saved to {output_path}")

    return final_df



def rule_based_dedup(df):
    logger.info("Starting rule-based deduplication...")

    # Normalize names
    df["normalized_name"] = df["Pathway"].astype(str).str.lower().str.strip()

    # Temporarily add Calculated_Num_Genes if not already present
    calc_genes_added = False
    if "Calculated_Num_Genes" not in df.columns:
        df["Calculated_Num_Genes"] = df["inputGenes"].apply(count_genes)
        calc_genes_added = True

    # Sort by normalized name, gene count (descending), and FDR (ascending)
    df.sort_values(['normalized_name', 'Calculated_Num_Genes', 'fdr'], ascending=[True, False, True], inplace=True)

    # Deduplicate
    df_deduped = df.drop_duplicates(subset=['normalized_name'], keep='first').copy()

    # Drop temporary columns
    drop_cols = ['normalized_name']
    if calc_genes_added:
        drop_cols.append('Calculated_Num_Genes')

    df_deduped.drop(columns=drop_cols, inplace=True)

    removed = len(df) - len(df_deduped)
    logger.info(f"Rule-based deduplication removed {removed} exact/case-insensitive duplicates.")
    return df_deduped


# === STEP 3: FAISS + MiniLM SEMANTIC DEDUPLICATION ===
def embedding_dedup(df):
    logger.info("Starting semantic deduplication using FAISS + MiniLM...")
    original_count = len(df)
    pathway_texts = df["Pathway"].astype(str).tolist()

    # Use lazy loaded models
    embedding_model = get_sentence_model()
    faiss_lib = get_faiss()

    # embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    # faiss_lib = faiss
    
    embeddings = embedding_model.encode(pathway_texts, convert_to_numpy=True, batch_size=64).astype('float32')
    faiss_lib.normalize_L2(embeddings)

    index = faiss_lib.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    sim_scores, sim_indices = index.search(embeddings, TOP_K)

    keep_flags = [True] * len(df)
    removed_log = set()

    for i in range(len(df)):
        if not keep_flags[i]:
            continue

        for rank in range(1, TOP_K):
            j = sim_indices[i][rank]
            sim = sim_scores[i][rank]

            if sim < SIM_THRESHOLD or not keep_flags[j] or (i, j) in removed_log or (j, i) in removed_log:
                continue

            pathway_i = df.iloc[i]["Pathway"]
            pathway_j = df.iloc[j]["Pathway"]
            count_i = count_genes(df.iloc[i]["inputGenes"])
            count_j = count_genes(df.iloc[j]["inputGenes"])

            if count_i >= count_j:
                keep_flags[j] = False
                removed_log.add((i, j))
                logger.info(f"[FAISS SIM={sim:.4f}] Kept: '{pathway_i}' ({count_i} genes) | Removed: '{pathway_j}' ({count_j} genes)")
            else:
                keep_flags[i] = False
                removed_log.add((j, i))
                logger.info(f"[FAISS SIM={sim:.4f}] Kept: '{pathway_j}' ({count_j} genes) | Removed: '{pathway_i}' ({count_i} genes)")
                break

    # df_deduped = df[pd.Series(keep_flags)].reset_index(drop=True)
    df_deduped = df[np.array(keep_flags)].reset_index(drop=True)
    logger.info(f"Semantic deduplication removed {original_count - len(df_deduped)} entries using FAISS.")
    return df_deduped

# === MASTER RUN ===
def run_deduplication(input_file: str):
    logger.info(f"=== Starting full pathway processing on: {input_file} ===")

    # Step 2: Deduplication
    df = load_table(input_file)
    logger.info(f"Loaded {len(df)} rows from input file.")
    df = rule_based_dedup(df)
    df = embedding_dedup(df)

    save_table(df, input_file)
    logger.info(f"Deduplicated pathways saved to {input_file}")
    logger.info(f"Final row count after deduplication: {len(df)}")
    logger.info("=" * 60)

    return input_file