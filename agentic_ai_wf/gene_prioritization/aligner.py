import pandas as pd
import numpy as np
from typing import Optional, List
from .helpers import logger

# ======================================
# Sabahat's code
# ======================================

def aligner(df: pd.DataFrame, 
           excluded_columns: Optional[List[str]] = None,
           log_fc_identifier: str = "log2FC",
           cohort_prefix: str = "gse",
           patient_prefix: str = "patient",
           up_threshold: float = 0.0,
           down_threshold: float = 0.0,
           handle_ties: str = "neutral") -> pd.DataFrame:
    """
    Aligns patient and cohort log fold change data and determines trend consensus.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe containing gene expression data
    excluded_columns : List[str], optional
        Columns to exclude from analysis. Default: ['Gene','Status','Normalized_Gene','HGNC_Synonyms']
    log_fc_identifier : str, default "log2FC"
        String identifier for log fold change columns
    cohort_prefix : str, default "gse"
        Prefix for cohort columns
    patient_prefix : str, default "patient"
        Prefix for patient columns
    up_threshold : float, default 0.0
        Threshold above which values are considered "UP"
    down_threshold : float, default 0.0
        Threshold below which values are considered "DOWN"
    handle_ties : str, default "neutral"
        How to handle values exactly at threshold: "neutral", "up", "down"
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with added columns for trend analysis
        
    Raises:
    -------
    ValueError: If input validation fails
    TypeError: If input types are incorrect
    """
    
    # Input validation
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input 'df' must be a pandas DataFrame")
    
    if df.empty:
        raise ValueError("Input DataFrame cannot be empty")
    
    if excluded_columns is None:
        excluded_columns = ['Gene', 'Status', 'Normalized_Gene', 'HGNC_Synonyms']
    
    if not isinstance(excluded_columns, (list, tuple)):
        raise TypeError("excluded_columns must be a list or tuple")
    
    if not isinstance(log_fc_identifier, str) or not log_fc_identifier:
        raise ValueError("log_fc_identifier must be a non-empty string")
    
    if handle_ties not in ["neutral", "up", "down"]:
        raise ValueError("handle_ties must be one of: 'neutral', 'up', 'down'")
    
    # Create a copy to avoid modifying original DataFrame
    df_copy = df.copy()
    
    try:
        # Get filtered columns (excluding specified columns)
        available_excluded = [col for col in excluded_columns if col in df_copy.columns]
        filtered_columns = df_copy.columns.difference(available_excluded).tolist()
        
        if not filtered_columns:
            logger.warning("No columns remaining after exclusion")
            return df_copy
        
        # Filter log fold change columns
        filtered_lfc_cols = [col for col in filtered_columns if log_fc_identifier in col]
        if not filtered_lfc_cols:
            logger.warning(f"No columns found containing '{log_fc_identifier}'")
            return df_copy
        
        # Separate cohort and patient columns with improved parsing
        filtered_lfc_cohort_cols = []
        filtered_lfc_patient_cols = []
        
        for col in filtered_lfc_cols:
            filtered_lfc_patient_cols = [col for col in filtered_lfc_cols if col.lower().startswith(patient_prefix.lower())]
            filtered_lfc_cohort_cols = [col for col in filtered_lfc_cols if col.lower().startswith(cohort_prefix.lower())]

        # Validate that we have columns for both groups
        if not filtered_lfc_patient_cols:
            logger.warning(f"No patient columns found with prefix '{patient_prefix}'")
        
        if not filtered_lfc_cohort_cols:
            logger.warning(f"No cohort columns found with prefix '{cohort_prefix}'")
        
        # Calculate means with proper handling of missing values
        if filtered_lfc_patient_cols:
            # Ensure all columns are numeric
            for col in filtered_lfc_patient_cols:
                df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')
            
            df_copy["Patient_LFC_mean"] = df_copy[filtered_lfc_patient_cols].mean(axis=1, skipna=True)
        else:
            df_copy["Patient_LFC_mean"] = np.nan
        
        if filtered_lfc_cohort_cols:
            # Ensure all columns are numeric
            for col in filtered_lfc_cohort_cols:
                df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce')
            
            df_copy["Cohort_LFC_mean"] = df_copy[filtered_lfc_cohort_cols].mean(axis=1, skipna=True)
        else:
            df_copy["Cohort_LFC_mean"] = np.nan
        
        # Determine trends with improved logic and tie handling
        def determine_trend(value, up_thresh=up_threshold, down_thresh=down_threshold, tie_handling=handle_ties):
            if pd.isna(value):
                return "UNKNOWN"
            elif value > up_thresh:
                return "UP"
            elif value < down_thresh:
                return "DOWN"
            else:  # value == threshold
                if tie_handling == "up":
                    return "UP"
                elif tie_handling == "down":
                    return "DOWN"
                else:  # neutral
                    return "NEUTRAL"
        
        # Apply trend determination vectorized
        df_copy["Patient_LFC_Trend"] = df_copy["Patient_LFC_mean"].apply(determine_trend)
        df_copy["Cohort_LFC_Trend"] = df_copy["Cohort_LFC_mean"].apply(determine_trend)
        
        # Determine consensus with improved logic
        def determine_consensus(patient_trend, cohort_trend):
            if patient_trend == "UNKNOWN" or cohort_trend == "UNKNOWN":
                return "INSUFFICIENT_DATA"
            elif patient_trend == cohort_trend:
                return "ALIGNED"
            else:
                return "CONTRADICTORY"
        
        df_copy['Trend_Consensus'] = df_copy.apply(
            lambda row: determine_consensus(row["Patient_LFC_Trend"], row["Cohort_LFC_Trend"]), 
            axis=1
        )
        
        # Log summary statistics
        if filtered_lfc_patient_cols or filtered_lfc_cohort_cols:
            aligned_count = (df_copy['Trend_Consensus'] == 'ALIGNED').sum()
            contradictory_count = (df_copy['Trend_Consensus'] == 'CONTRADICTORY').sum()
            insufficient_data_count = (df_copy['Trend_Consensus'] == 'INSUFFICIENT_DATA').sum()
            
            logger.info(f"Trend Analysis Summary:")
            logger.info(f"  Aligned: {aligned_count}")
            logger.info(f"  Contradictory: {contradictory_count}")
            logger.info(f"  Insufficient Data: {insufficient_data_count}")
            logger.info(f"  Patient columns used: {len(filtered_lfc_patient_cols)}")
            logger.info(f"  Cohort columns used: {len(filtered_lfc_cohort_cols)}")

        return df_copy
        
    except Exception as e:
        logger.error(f"Error in aligner function: {str(e)}")
        raise RuntimeError(f"Failed to process DataFrame: {str(e)}") from e
