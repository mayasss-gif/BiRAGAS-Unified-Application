import pandas as pd
import os
import logging
from typing import Optional, List
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress only specific pandas warnings
import warnings
warnings.filterwarnings('ignore', category=pd.errors.SettingWithCopyWarning)


class DataProcessingError(Exception):
    """Custom exception for data processing errors."""
    pass


def validate_pathways_directory(pathways_dir: str) -> Path:
    """
    Validate that the pathways directory exists and contains CSV files.
    
    Args:
        pathways_dir: Path to the directory containing CSV files
        
    Returns:
        Path object for the validated directory
        
    Raises:
        DataProcessingError: If directory doesn't exist or contains no CSV files
    """
    if not pathways_dir:
        raise DataProcessingError("Pathways directory path cannot be empty")
    
    path = Path(pathways_dir)
    if not path.exists():
        raise DataProcessingError(f"Directory does not exist: {pathways_dir}")
    
    if not path.is_dir():
        raise DataProcessingError(f"Path is not a directory: {pathways_dir}")
    
    csv_files = list(path.glob("*.csv"))
    if not csv_files:
        raise DataProcessingError(f"No CSV files found in directory: {pathways_dir}")
    
    logger.info(f"Found {len(csv_files)} CSV files in {pathways_dir}")
    return path


def load_and_combine_csv_files(pathways_dir: Path) -> pd.DataFrame:
    """
    Load and combine all CSV files from the specified directory.
    
    Args:
        pathways_dir: Path to the directory containing CSV files
        
    Returns:
        Combined DataFrame from all CSV files
        
    Raises:
        DataProcessingError: If unable to load or combine files
    """
    csv_files = list(pathways_dir.glob("*.csv"))
    dataframes = []
    
    for csv_file in csv_files:
        try:
            logger.info(f"Loading file: {csv_file.name}")
            df = pd.read_csv(csv_file)
            if df.empty:
                logger.warning(f"Empty CSV file: {csv_file.name}")
                continue
            dataframes.append(df)
        except Exception as e:
            logger.error(f"Error loading {csv_file.name}: {str(e)}")
            raise DataProcessingError(f"Failed to load {csv_file.name}: {str(e)}")
    
    if not dataframes:
        raise DataProcessingError("No valid data found in CSV files")
    
    # Combine all dataframes
    try:
        combined_df = pd.concat(dataframes, ignore_index=True)
        logger.info(f"Successfully combined {len(dataframes)} files into DataFrame with {len(combined_df)} rows")
        return combined_df
    except Exception as e:
        raise DataProcessingError(f"Failed to combine CSV files: {str(e)}")


def clean_target_value(value: str) -> str:
    """
    Clean target values by removing HSA annotations.
    
    Args:
        value: Target value to clean
        
    Returns:
        Cleaned target value
    """
    if pd.isna(value):
        return value
    if '[HSA:' in str(value):
        return str(value).split('[HSA:')[0].strip()
    return str(value)


def extract_efficacy_value(value: str) -> str:
    """
    Extract efficacy value from between commas.
    
    Args:
        value: Efficacy value to extract from
        
    Returns:
        Extracted efficacy value
    """
    if pd.isna(value):
        return value
    
    parts = str(value).split(',')
    if len(parts) >= 2:
        return parts[1].strip()
    return str(value)


def create_target_mechanism_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a combined target-mechanism column from target and efficacy data.
    
    Args:
        df: DataFrame containing target-updated and efficacy-updated columns
        
    Returns:
        DataFrame with added target-mechanism column
    """
    df = df.copy()
    
    def merge_values(row):
        target = row.get('target-updated')
        efficacy = row.get('efficacy-updated')
        
        if pd.notna(target) and pd.notna(efficacy):
            return f"{target} / {efficacy}"
        elif pd.notna(target):
            return target
        elif pd.notna(efficacy):
            return efficacy
        else:
            return None
    
    df['target-mechanism'] = df.apply(merge_values, axis=1)
    return df


def prep_data(pathways_dir: str) -> pd.DataFrame:
    """
    Prepare and clean data from CSV files in the specified directory.
    
    Args:
        pathways_dir: Path to directory containing CSV files
        
    Returns:
        Processed DataFrame
        
    Raises:
        DataProcessingError: If data processing fails
    """
    try:
        # Validate directory and load data
        validated_path = validate_pathways_directory(pathways_dir)
        main_df = load_and_combine_csv_files(validated_path)
        
        # Validate required columns
        required_columns = ['target', 'efficacy', 'pathway_name', 'pathway_id', 'approved']
        missing_columns = [col for col in required_columns if col not in main_df.columns]
        if missing_columns:
            raise DataProcessingError(f"Missing required columns: {missing_columns}")
        
        # Clean and process data
        logger.info("Processing target values")
        main_df['target-updated'] = main_df['target'].apply(clean_target_value)
        
        logger.info("Processing efficacy values")
        main_df['efficacy-updated'] = main_df['efficacy'].apply(extract_efficacy_value)
        
        logger.info("Creating target-mechanism column")
        main_df = create_target_mechanism_column(main_df)
        
        # Clean pathway names
        logger.info("Cleaning pathway names")
        main_df["pathway_name"] = main_df["pathway_name"].str.replace(
            r"\s*-\s*Homo sapiens \(human\)$", "", regex=True
        )
        
        # Create pathway column
        main_df["pathway"] = main_df["pathway_id"].astype(str) + ", " + main_df["pathway_name"]
        
        # Standardize approved column
        main_df['approved'] = main_df['approved'].replace({True: 'Yes', False: 'No'})
        
        logger.info(f"Data preparation completed. Final DataFrame shape: {main_df.shape}")
        return main_df
        
    except DataProcessingError:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in prep_data: {str(e)}")
        raise DataProcessingError(f"Data preparation failed: {str(e)}")


def select_representative_drug(group: pd.DataFrame) -> pd.Series:
    """
    Select a representative drug entry from a group, preferring approved drugs.
    
    Args:
        group: DataFrame group containing drug entries
        
    Returns:
        Series representing the selected drug entry
    """
    # Try to get the first approved entry
    approved = group[group['approved'] == 'Yes']
    if not approved.empty:
        return approved.iloc[0]
    else:
        return group.iloc[0]


def format_data(main_df: pd.DataFrame) -> pd.DataFrame:
    """
    Format data by selecting representative entries for each drug.
    
    Args:
        main_df: DataFrame containing drug data
        
    Returns:
        Formatted DataFrame with representative drug entries
        
    Raises:
        DataProcessingError: If data formatting fails
    """
    try:
        if main_df.empty:
            raise DataProcessingError("Input DataFrame is empty")
        
        if 'name' not in main_df.columns:
            raise DataProcessingError("Missing 'name' column in DataFrame")
        
        # Create a helper column with the first word of the drug name
        main_df = main_df.copy()
        main_df['drug_key'] = main_df['name'].str.split().str[0]
        
        # Group by drug_key and select representative entries
        logger.info("Selecting representative drug entries")
        formatted_df = main_df.groupby('drug_key', as_index=False).apply(
            select_representative_drug, include_groups=False
        ).reset_index(drop=True)
        
        # Remove the helper column
        if 'drug_key' in formatted_df.columns:
            formatted_df.drop(columns=['drug_key'], inplace=True)
        
        logger.info(f"Data formatting completed. Final DataFrame shape: {formatted_df.shape}")
        return formatted_df
        
    except Exception as e:
        logger.error(f"Error in format_data: {str(e)}")
        raise DataProcessingError(f"Data formatting failed: {str(e)}")


def data_processing(pathways_dir: str) -> pd.DataFrame:
    """
    Main function to process drug pathway data.
    
    Args:
        pathways_dir: Path to directory containing CSV files
        
    Returns:
        Processed and formatted DataFrame
        
    Raises:
        DataProcessingError: If data processing fails
    """
    try:
        logger.info(f"Starting data processing for directory: {pathways_dir}")
        
        # Prepare data
        df = prep_data(pathways_dir)
        
        # Format data
        df_final = format_data(df)
        
        logger.info("Data processing completed successfully")
        return df_final
        
    except DataProcessingError:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in data_processing: {str(e)}")
        raise DataProcessingError(f"Data processing failed: {str(e)}")


def get_processing_summary(df: pd.DataFrame) -> dict:
    """
    Get a summary of the processed data.
    
    Args:
        df: Processed DataFrame
        
    Returns:
        Dictionary containing data summary
    """
    if df.empty:
        return {"status": "empty", "message": "No data processed"}
    
    return {
        "status": "success",
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "approved_drugs": len(df[df['approved'] == 'Yes']) if 'approved' in df.columns else 0,
        "unique_drugs": df['name'].nunique() if 'name' in df.columns else 0,
        "columns": list(df.columns)
    }