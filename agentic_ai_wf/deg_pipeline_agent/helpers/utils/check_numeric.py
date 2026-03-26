import pandas as pd

def check_actual_numeric_content(value):
    """
    Check if a value is actually numeric (even if stored as string)
    Returns True if it's a number, False otherwise
    """
    if pd.isna(value):
        return False
    
    # If already numeric type
    if pd.api.types.is_numeric_dtype(type(value)):
        return True
    
    # Convert to string and check if it represents a number
    str_val = str(value).strip()
    
    # Handle empty strings
    if not str_val:
        return False
    
    # Check if it's a valid number (int or float)
    try:
        float(str_val)
        return True
    except ValueError:
        return False

def count_actual_numeric_values(row):
    """Count values that are actually numeric (including those stored as strings)"""
    return sum(1 for val in row if check_actual_numeric_content(val))