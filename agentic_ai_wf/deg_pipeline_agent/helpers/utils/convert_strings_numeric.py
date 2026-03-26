import pandas as pd
from typing import List, Optional
import numpy as np

def convert_strings_to_numeric(
    df: pd.DataFrame, 
    columns: Optional[List[str]] = None,
    errors: str = 'coerce',
    downcast: Optional[str] = 'integer',
    preserve_dtypes: List[str] = None
) -> pd.DataFrame:
    """
    Convert string columns to numeric types where possible.
    
    Args:
        df: Input DataFrame
        columns: Specific columns to convert. If None, attempts all columns.
        errors: How to handle conversion errors ('coerce', 'raise', 'ignore')
        downcast: Downcast integer types ('integer', 'signed', 'unsigned', 'float', None)
        preserve_dtypes: List of dtypes to skip conversion (e.g., ['datetime64', 'category'])
    
    Returns:
        DataFrame with converted numeric columns
    """
    df_copy = df.copy()
    preserve_dtypes = preserve_dtypes or ['datetime64', 'timedelta64', 'category']
    
    # Determine columns to process
    cols_to_check = columns if columns else df_copy.columns.tolist()
    
    for col in cols_to_check:
        if col not in df_copy.columns:
            continue
            
        # Skip if column dtype should be preserved
        if any(preserve_type in str(df_copy[col].dtype) for preserve_type in preserve_dtypes):
            continue
            
        # Skip if already numeric
        if pd.api.types.is_numeric_dtype(df_copy[col]):
            continue
            
        # Try converting to numeric
        original_col = df_copy[col].copy()
        
        # Handle common string representations
        if df_copy[col].dtype == 'object':
            # Clean common string patterns
            cleaned_col = df_copy[col].astype(str).str.strip()
            
            # Remove common currency symbols and thousands separators
            cleaned_col = cleaned_col.str.replace(r'[$,€£¥]', '', regex=True)
            cleaned_col = cleaned_col.str.replace(r'[()]', '', regex=True)  # Remove parentheses
            
            # Handle percentage
            is_percentage = cleaned_col.str.endswith('%')
            cleaned_col = cleaned_col.str.replace('%', '')
            
            # Replace common string representations of missing values
            cleaned_col = cleaned_col.replace(['', 'N/A', 'NA', 'null', 'NULL', 'None', 'nan', 'NaN'], np.nan)
            
            try:
                # Attempt conversion
                numeric_col = pd.to_numeric(cleaned_col, errors=errors)
                
                # Apply percentage conversion if needed
                if is_percentage.any() and not numeric_col.isna().all():
                    percentage_mask = is_percentage & ~numeric_col.isna()
                    numeric_col.loc[percentage_mask] = numeric_col.loc[percentage_mask] / 100
                
                # Check if conversion was successful (i.e., we have some non-null numeric values)
                if not numeric_col.isna().all() and (numeric_col.notna().sum() > 0):
                    # Downcast if specified and all values are integers
                    if downcast and numeric_col.notna().any():
                        if numeric_col.dropna().apply(lambda x: float(x).is_integer()).all():
                            numeric_col = pd.to_numeric(numeric_col, downcast=downcast)
                        else:
                            numeric_col = pd.to_numeric(numeric_col, downcast='float')
                    
                    df_copy[col] = numeric_col
                    
            except (ValueError, TypeError):
                # If conversion fails, keep original column
                continue
    
    return df_copy