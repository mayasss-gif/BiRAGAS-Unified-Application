import pandas as pd

def fix_missing_first_header_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Renames the first column to 'gene' if it appears to be unnamed but contains gene-like identifiers.
    """
    current_columns = list(df.columns)
    data_columns = df.shape[1]

    print(f"Current columns: {current_columns}")
    print(f"Data columns: {data_columns}")

    # Heuristic: If the first column has no name or is empty
    if current_columns[0].lower().startswith('unnamed') or not current_columns[0].strip():
        df.columns = ['gene'] + current_columns[1:]
        print("✅ Renamed first unnamed column to 'gene'")
    elif len(current_columns) == data_columns - 1:
        df.columns = ['gene'] + current_columns
        print("✅ Added 'gene' as missing first column header")
    else:
        print("⚠️ No fix applied — column mismatch not matching expected patterns.")

    return df