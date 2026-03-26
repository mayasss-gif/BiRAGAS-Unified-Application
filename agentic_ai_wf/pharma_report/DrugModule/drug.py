import pandas as pd
import openai
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global config
REQUIRED_COLUMNS = ['pathway_id', 'drug_id', 'drug_name', 'pathway_name', 'llm_score',
                    'score_justification', 'target_mechanism', 'fda_approved_status', 'molecular_evidence_score', 'justification']
openai.api_key = os.getenv("OPENAI_API_KEY")

# Helper to load and clean
def load_and_clean_csv(csv_path):
    """Load and clean CSV file with basic validation."""
    try:
        df = pd.read_csv(csv_path)
        
        # Handle empty DataFrame
        if df.empty:
            logger.warning(f"CSV file is empty: {csv_path}")
            return df
        
        # Normalize FDA_Approved_Status column if it exists
        if 'fda_approved_status' in df.columns:
            df['fda_approved_status'] = df['fda_approved_status'].astype(str).str.strip()
        
        # Check for missing required columns
        missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
        if missing_cols:
            logger.error(f"Missing columns in CSV: {missing_cols}")
            raise ValueError(f"Missing columns in CSV: {missing_cols}")
        
        return df
        
    except FileNotFoundError:
        logger.error(f"CSV file not found: {csv_path}")
        raise
    except pd.errors.EmptyDataError:
        logger.error(f"CSV file is empty or corrupted: {csv_path}")
        raise
    except Exception as e:
        logger.error(f"Error loading CSV file {csv_path}: {str(e)}")
        raise

# 1️⃣ Enhanced Summary Count Function
def load_csv_flexible(csv_path):
    """Load CSV with flexible column detection instead of strict requirements."""
    try:
        df = pd.read_csv(csv_path)
        
        # Handle empty DataFrame
        if df.empty:
            logger.warning(f"CSV file is empty: {csv_path}")
            return df
        
        return df
        
    except FileNotFoundError:
        logger.error(f"CSV file not found: {csv_path}")
        raise
    except pd.errors.EmptyDataError:
        logger.error(f"CSV file is empty or corrupted: {csv_path}")
        raise
    except Exception as e:
        logger.error(f"Error loading CSV file {csv_path}: {str(e)}")
        raise

def get_drug_summary_counts(csv_path):
    """
    Get comprehensive drug summary counts with robust FDA approval status handling.
    
    Args:
        csv_path (str): Path to the CSV file containing drug data
        
    Returns:
        dict: Dictionary containing drug counts and statistics, or None if no data
    """
    try:
        df = load_csv_flexible(csv_path)  # Use flexible loader instead
        
        if df.empty:
            logger.warning("DataFrame is empty, returning None")
            return None
        
        # Check if required columns exist - be more flexible with column names
        drug_name_col = None
        for col in df.columns:
            if 'drug' in col.lower() and 'name' in col.lower():
                drug_name_col = col
                break
        
        if drug_name_col is None:
            logger.error(f"No drug name column found in DataFrame. Available columns: {list(df.columns)}")
            return None
        
        # Rename to standard column name
        if drug_name_col != 'drug_name':
            df = df.rename(columns={drug_name_col: 'drug_name'})
        
        fda_status_col = None
        for col in df.columns:
            if 'fda' in col.lower() or ('approved' in col.lower() and 'status' in col.lower()):
                fda_status_col = col
                break
        
        if fda_status_col is None:
            logger.warning("No FDA status column found, creating default status")
            df['fda_approved_status'] = 'Under Investigation'
        elif fda_status_col != 'fda_approved_status':
            df = df.rename(columns={fda_status_col: 'fda_approved_status'})
        
        # Remove rows with null Drug_Name
        df = df.dropna(subset=['drug_name'])
        
        if df.empty:
            logger.warning("No valid drug names found after cleaning")
            return None
        
        # Get total unique drugs
        total_drugs = df['drug_name'].nunique()
        
        if total_drugs == 0:
            logger.warning("No unique drugs found")
            return None
        
        # Normalize FDA approval status for matching
        df['fda_status_normalized'] = df['fda_approved_status'].str.lower().str.strip()
        
        # Log unique FDA statuses for debugging
        unique_statuses = df['fda_status_normalized'].unique()
        logger.info(f"Unique FDA approval statuses found: {unique_statuses}")
        
        # Define approval status mappings (case-insensitive)
        approved_statuses = {
            'approved', 'yes', 'true', '1', 'fda approved', 
            'approved by fda', 'fda-approved'
        }
        
        not_found_statuses = {
            'not found in fda database', 'not found', 'no', 'false', '0',
            'not approved', 'not fda approved', 'unknown', 'na', 'n/a'
        }
        
        error_statuses = {
            'error', 'error in verification', 'api error', 'timeout',
            'verification failed', 'data unavailable'
        }
        
        # Count drugs by approval status
        approved_count = df[df['fda_status_normalized'].isin(approved_statuses)]['drug_name'].nunique()
        not_found_count = df[df['fda_status_normalized'].isin(not_found_statuses)]['drug_name'].nunique()
        error_count = df[df['fda_status_normalized'].isin(error_statuses)]['drug_name'].nunique()
        
        # Count unclassified statuses
        all_known_statuses = approved_statuses | not_found_statuses | error_statuses
        unclassified_count = df[~df['fda_status_normalized'].isin(all_known_statuses)]['drug_name'].nunique()
        
        # Get pathway statistics - be flexible with pathway column names
        pathway_col = None
        for col in df.columns:
            if 'pathway' in col.lower():
                pathway_col = col
                break
        
        total_pathways = df[pathway_col].nunique() if pathway_col and not df[pathway_col].isna().all() else 0
        
        # Calculate percentages
        approved_percentage = (approved_count / total_drugs * 100) if total_drugs > 0 else 0
        not_found_percentage = (not_found_count / total_drugs * 100) if total_drugs > 0 else 0
        
        # Get top pathways by drug count
        top_pathways = []
        if pathway_col and not df[pathway_col].isna().all():
            pathway_counts = df[pathway_col].value_counts().head(5)
            top_pathways = [{"pathway": pathway, "drug_count": count} 
                          for pathway, count in pathway_counts.items()]
        
        # Log unclassified statuses for debugging
        if unclassified_count > 0:
            unclassified_statuses = df[~df['fda_status_normalized'].isin(all_known_statuses)]['fda_status_normalized'].unique()
            logger.warning(f"Unclassified FDA statuses found: {unclassified_statuses}")
        
        result = {
            "total_drugs": total_drugs,
            "approved_drugs": approved_count,
            "not_approved_drugs": not_found_count,
            "error_drugs": error_count,
            "unclassified_drugs": unclassified_count,
            "total_pathways": total_pathways,
            "approved_percentage": round(approved_percentage, 1),
            "not_found_percentage": round(not_found_percentage, 1),
            "top_pathways": top_pathways,
            "data_quality": {
                "total_rows": len(df),
                "unique_drugs": total_drugs,
                "coverage": round((approved_count + not_found_count) / total_drugs * 100, 1) if total_drugs > 0 else 0
            }
        }
        
        logger.info(f"Drug summary completed: {total_drugs} unique drugs, {approved_count} approved, {not_found_count} not found")
        return result
        
    except Exception as e:
        logger.error(f"Error in get_drug_summary_counts: {str(e)}")
        return None

# 2️⃣ Top 5 Most Frequent Drugs
# def get_top_5_drugs(csv_path):
#     df = load_and_clean_csv(csv_path)
#     if df.empty:
#         return None

#     top_5 = df['Drug_Name'].value_counts().head(5).index.tolist()
#     if not top_5:
#         return None

#     return df[df['Drug_Name'].isin(top_5)][REQUIRED_COLUMNS].reset_index(drop=True)

# 3️⃣ Top 5 Approved Drugs
def get_top_5_approved_drugs(csv_path):
    """
    Get top 5 FDA approved drugs with unique pathway_id and drug_name, sorted by scores.
    
    Args:
        csv_path (str): Path to the CSV file containing drug data
        
    Returns:
        DataFrame: Top 5 approved drugs with unique pathway_id and drug_name, or None if no data
    """
    try:
        df = load_and_clean_csv(csv_path)
        
        if df.empty:
            logger.warning("DataFrame is empty, returning None")
            return None
        
        # Normalize FDA approval status
        df['fda_status_normalized'] = df['fda_approved_status'].str.lower().str.strip()
        
        # Define approved statuses (case-insensitive)
        approved_statuses = {
            'approved', 'yes', 'true', '1', 'fda approved', 
            'approved by fda', 'fda-approved'
        }
        
        # Filter for approved drugs
        approved_df = df[df['fda_status_normalized'].isin(approved_statuses)]
        
        if approved_df.empty:
            logger.info("No FDA approved drugs found")
            return None
        
        # Sort by molecular_evidence_score and llm_score (both descending)
        # Using na_position='last' to handle potential NaN values
        sorted_df = approved_df.sort_values(
            by=['molecular_evidence_score', 'llm_score'], 
            ascending=[False, False],
            na_position='last'
        ).reset_index(drop=True)
        
        # Get unique combinations of pathway_id and drug_name, keeping the first occurrence
        # (which will be the highest scoring due to sorting)
        unique_combinations = sorted_df.drop_duplicates(
            subset=['pathway_id', 'drug_name'], 
            keep='first'
        ).reset_index(drop=True)
        
        # Get top 5 drugs
        top_5_df = unique_combinations.head(20)
        
        if top_5_df.empty:
            logger.info("No top approved drugs found")
            return None
        
        logger.info(f"Found {len(top_5_df)} top approved drugs with unique pathway_id and drug_name")
        return top_5_df
        
    except Exception as e:
        logger.error(f"Error in get_top_5_approved_drugs: {str(e)}")
        return None
# def get_top_5_approved_drugs(csv_path):
#     """
#     Get top 5 FDA approved drugs with robust status handling.
    
#     Args:
#         csv_path (str): Path to the CSV file containing drug data
        
#     Returns:
#         DataFrame: Top 5 approved drugs with required columns, or None if no data
#     """
#     try:
#         df = load_and_clean_csv(csv_path)
        
#         if df.empty:
#             logger.warning("DataFrame is empty, returning None")
#             return None
        
#         # Normalize FDA approval status
#         df['fda_status_normalized'] = df['fda_approved_status'].str.lower().str.strip()
        
#         # Define approved statuses (case-insensitive)
#         approved_statuses = {
#             'approved', 'yes', 'true', '1', 'fda approved', 
#             'approved by fda', 'fda-approved'
#         }
        
#         # Filter for approved drugs
#         approved_df = df[df['fda_status_normalized'].isin(approved_statuses)]
        
#         if approved_df.empty:
#             logger.info("No FDA approved drugs found")
#             return None
        
#         # Get drug counts and select top 5
#         drug_counts = approved_df['drug_name'].value_counts().head(5)
#         top_drugs = drug_counts.index.tolist()
        
#         if not top_drugs:
#             logger.info("No top approved drugs found")
#             return None
        
#         # Return data for top drugs with required columns
#         result_df = approved_df[approved_df['drug_name'].isin(top_drugs)][REQUIRED_COLUMNS].reset_index(drop=True)

#         # get unique drugs rows by drug_id
#         unique_drugs = result_df['drug_id'].unique()
#         unique_drugs_df = result_df[result_df['drug_id'].isin(unique_drugs)]
#         result_df = unique_drugs_df
        
#         logger.info(f"Found {len(top_drugs)} top approved drugs")
#         return result_df
        
#     except Exception as e:
#         logger.error(f"Error in get_top_5_approved_drugs: {str(e)}")
#         return None

# 4️⃣ Top 5 Not Approved Drugs
def get_top_5_not_approved_drugs(csv_path):
    """
    Get top 5 not approved drugs with robust status handling.
    
    Args:
        csv_path (str): Path to the CSV file containing drug data
        
    Returns:
        DataFrame: Top 5 not approved drugs with required columns, or None if no data
    """
    try:
        df = load_and_clean_csv(csv_path)
        
        if df.empty:
            logger.warning("DataFrame is empty, returning None")
            return None
        
        # Normalize FDA approval status
        df['fda_status_normalized'] = df['fda_approved_status'].str.lower().str.strip()
        
        # Define not approved statuses (case-insensitive)
        not_approved_statuses = {
            'not found in fda database', 'not found', 'no', 'false', '0',
            'not approved', 'not fda approved', 'unknown', 'na', 'n/a'
        }
        
        # Filter for not approved drugs
        not_approved_df = df[df['fda_status_normalized'].isin(not_approved_statuses)]
        
        if not_approved_df.empty:
            logger.info("No not approved drugs found")
            return None
        
        # Get drug counts and select top 5
        drug_counts = not_approved_df['drug_name'].value_counts().head(5)
        top_drugs = drug_counts.index.tolist()
        
        if not top_drugs:
            logger.info("No top not approved drugs found")
            return None
        
        # Return data for top drugs with required columns
        result_df = not_approved_df[not_approved_df['drug_name'].isin(top_drugs)][REQUIRED_COLUMNS].reset_index(drop=True)
        
        logger.info(f"Found {len(top_drugs)} top not approved drugs")
        return result_df
        
    except Exception as e:
        logger.error(f"Error in get_top_5_not_approved_drugs: {str(e)}")
        return None


def generate_table_summary(df):
    """
    Generate a technical, biomedical summary of a given DataFrame table using GPT-4o.
    """
    # Convert DataFrame to markdown table (limited to top rows for brevity)
    num_rows = min(20, len(df))
    markdown_table = df.head(num_rows).to_markdown(index=False)

    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a biomedical data analyst tasked with summarizing tabular cohort or pathway data "
                    "in a domain-specific, research-ready format. Use precise technical language and concise narrative tone."
                    "Keep it within 3 to 4 lines"
                )
            },
            {
                "role": "user",
                "content": (
                    "Please generate a brief summary of the key insights from the following table:\n\n"
                    f"{markdown_table}"
                )
            }
        ],
        max_tokens=500
    )

    return response.choices[0].message.content
