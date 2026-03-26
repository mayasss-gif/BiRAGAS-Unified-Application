import pandas as pd

def filter_log2fc_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter out rows where patient_log2fc contains 'No log2FC data available'
    
    Parameters:
    df (pd.DataFrame): Input dataframe
    
    Returns:
    pd.DataFrame: Filtered dataframe
    """
    try:
        print("Filtering out rows with 'No log2FC data available'...")
        initial_count = len(df)
        
        # Check if patient_log2fc column exists
        if 'patient_log2fc' not in df.columns:
            print("Warning: 'patient_log2fc' column not found in dataframe. Returning original dataframe.")
            return df
        
        # Filter out rows with "No log2FC data available" (case-insensitive)
        df_filtered = df[~df['patient_log2fc'].str.contains('No log2FC data available', case=False, na=False)]
        
        final_count = len(df_filtered)
        removed_count = initial_count - final_count
        
        print(f"Removed {removed_count} rows with no log2FC data")
        print(f"Remaining rows: {final_count}")
        
        return df_filtered
        
    except Exception as e:
        print(f"Error during log2FC filtering: {str(e)}. Returning original dataframe.")
        return df