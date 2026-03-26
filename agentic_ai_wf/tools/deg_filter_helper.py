import pandas as pd
import re

def extract_gene_names(target_str):
    """
    Extracts gene names from complex target strings like:
    "DPP4 (CD26) [HSA:1803]; ..." → returns ['DPP4']
    """
    if pd.isna(target_str):
        return []
    
    # Match leading gene symbols (capital letters/digits/underscores) before space or (
    matches = re.findall(r'\b[A-Z0-9_-]+(?=\s|\()', target_str)
    return [gene.strip().upper() for gene in matches]

def filter_drugs_by_degs(
    drug_df: pd.DataFrame,
    deg_df: pd.DataFrame,
    target_column: str = "target",
    deg_column: str = "Normalized_Gene"
) -> pd.DataFrame:
    """
    Filters drugs whose extracted targets match with DEGs.
    """
    deg_genes = set(deg_df[deg_column].astype(str).str.upper())

    def has_matching_target(targets_str):
        gene_names = extract_gene_names(targets_str)
        return any(gene in deg_genes for gene in gene_names)

    filtered_df = drug_df[drug_df[target_column].apply(has_matching_target)]
    return filtered_df


