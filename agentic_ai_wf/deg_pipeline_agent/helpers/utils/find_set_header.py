import pandas as pd

def find_and_set_proper_header(df, min_numeric_threshold=2):
    """
    Find the row that should be the header (row before first row with >= min_numeric_threshold numbers)
    and return a new dataframe with proper headers set
    """
    print(f"\n🔍 SEARCHING FOR PROPER HEADER ROW (min {min_numeric_threshold} numeric values required):")
    
    for i in range(len(df)):
        row = df.iloc[i]
        # Count numeric values in this row
        numeric_count = 0
        for val in row:
            try:
                # Try to convert to float, if successful and not NaN, it's numeric
                float_val = float(val)
                if not pd.isna(float_val):
                    numeric_count += 1
            except (ValueError, TypeError):
                continue
        
        print(f"  Row {i}: {numeric_count} numeric values")
        
        # If this row has enough numeric values, the previous row should be header
        if numeric_count >= min_numeric_threshold:
            if i == 0:
                print("  ✅ First row already has numeric data, keeping current headers")
                return df
            else:
                header_row_idx = i - 1
                print(f"  ✅ Found data row at index {i}, setting header to row {header_row_idx}")
                
                # Create new dataframe with proper header
                new_df = df.iloc[i:].copy()  # Data starts from row i
                new_df.columns = df.iloc[header_row_idx].values  # Set header from row i-1
                new_df.reset_index(drop=True, inplace=True)
                
                print(f"  📊 New dataframe shape: {new_df.shape}")
                print(f"  📋 New headers (first 5): {list(new_df.columns[:5])}")
                return new_df
    
    print("  ⚠️  No row found with sufficient numeric values, keeping original dataframe")
    return df