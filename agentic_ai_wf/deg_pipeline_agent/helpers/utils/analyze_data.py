import pandas as pd
import numpy as np
from typing import Tuple, List
from collections import Counter
from .check_numeric import count_actual_numeric_values

def analyze_dataframe(df, file_path, file_name)-> Tuple[pd.DataFrame, List[str]]:
    """Comprehensive data analysis function"""
    print("=" * 80)
    print(f"ANALYZING: {file_name}")
    print(f"FILE PATH: {file_path}")
    print("=" * 80)
    
    # Basic Info
    print("\n📊 BASIC INFORMATION:")
    print(f"Shape: {df.shape} (rows × columns)")
    print(f"Total cells: {df.shape[0] * df.shape[1]:,}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # Column Analysis
    print("\n📋 COLUMN ANALYSIS:")
    print(f"Total columns: {len(df.columns)}")
    print(f"Column names length vs actual columns: {'✅ Match' if len(df.columns) == df.shape[1] else '❌ Mismatch'}")
    
    # Check for duplicate column names
    duplicate_cols = [col for col, count in Counter(df.columns).items() if count > 1]
    if duplicate_cols:
        print(f"❌ Duplicate column names found: {duplicate_cols}")
    else:
        print("✅ No duplicate column names")
    
    # Check for unnamed columns
    unnamed_cols = [col for col in df.columns if 'Unnamed' in str(col)]
    if unnamed_cols:
        print(f"⚠️  Unnamed columns found: {len(unnamed_cols)} columns")
        print(f"   Unnamed columns: {unnamed_cols[:5]}{'...' if len(unnamed_cols) > 5 else ''}")
        if len(unnamed_cols) == 1:
            df.columns = ['gene'] + list(df.columns[:-1])
            print(f"✅ Fixed unnamed column by setting 'gene' as first column {df.columns}")
    else:
        print("✅ No unnamed columns")
    
    # Data Types Analysis
    print("\n🔢 DATA TYPES ANALYSIS:")
    dtype_counts = df.dtypes.value_counts()
    print("Data type distribution:")
    for dtype, count in dtype_counts.items():
        print(f"  {dtype}: {count} columns ({count/len(df.columns)*100:.1f}%)")
    
    # Numerical columns analysis
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    
    print(f"\nNumerical columns: {len(numeric_cols)} ({len(numeric_cols)/len(df.columns)*100:.1f}%)")
    print(f"Non-numerical columns: {len(non_numeric_cols)} ({len(non_numeric_cols)/len(df.columns)*100:.1f}%)")
    
    # Missing Values Analysis
    print("\n🕳️  MISSING VALUES ANALYSIS:")
    total_missing = df.isnull().sum().sum()
    total_cells = df.shape[0] * df.shape[1]
    missing_percentage = (total_missing / total_cells) * 100
    
    print(f"Total missing values: {total_missing:,} ({missing_percentage:.2f}% of all cells)")
    
    if total_missing > 0:
        missing_by_col = df.isnull().sum()
        cols_with_missing = missing_by_col[missing_by_col > 0].sort_values(ascending=False)
        
        print(f"Columns with missing values: {len(cols_with_missing)}")
        print("Top 10 columns with most missing values:")
        for col, missing_count in cols_with_missing.head(10).items():
            missing_pct = (missing_count / len(df)) * 100
            print(f"  {col}: {missing_count:,} ({missing_pct:.1f}%)")

            # drop the columns with missing values
            if missing_pct > 0.5:
                print(f"Dropping column {col} with {missing_pct:.1f}% missing values")
                df.drop(columns=col, inplace=True)

        
        # Rows with missing values
        rows_with_missing = df.isnull().any(axis=1).sum()
        print(f"Rows with at least one missing value: {rows_with_missing} ({rows_with_missing/len(df)*100:.1f}%)")
    else:
        print("✅ No missing values found")
    
    # Row Analysis - Check specific rows (1 and 100)
    print("\n📏 ROW ANALYSIS:")
    
    # Row 1 (index 0) analysis
    if len(df) > 0:
        row_1 = df.iloc[0]
        numeric_in_row_1 = count_actual_numeric_values(row_1)
        print(f"Row 1 (index 0): {numeric_in_row_1} numerical values out of {len(row_1)} columns")
        
        # Show sample values from row 1
        print(f"Row 1 sample values: {dict(list(row_1.items())[:5])}")
    
    # Row 100 (index 99) analysis
    if len(df) >= 100:
        row_100 = df.iloc[99]
        numeric_in_row_100 = count_actual_numeric_values(row_100)
        print(f"Row 100 (index 99): {numeric_in_row_100} numerical values out of {len(row_100)} columns")
        
        # Show sample values from row 100
        print(f"Row 100 sample values: {dict(list(row_100.items())[:5])}")
    else:
        print(f"❌ Dataset has only {len(df)} rows, cannot analyze row 100")
    
    # Data Quality Issues Detection
    print("\n🔍 DATA QUALITY ISSUES:")
    issues = []
    
    # Check for completely empty rows
    empty_rows = df.isnull().all(axis=1).sum()
    if empty_rows > 0:
        issues.append(f"Empty rows: {empty_rows}")
    
    # Check for completely empty columns
    empty_cols = df.isnull().all(axis=0).sum()
    if empty_cols > 0:
        issues.append(f"Empty columns: {empty_cols}")
    
    # Check for single-value columns (no variance)
    if len(numeric_cols) > 0:
        single_value_cols = []
        for col in numeric_cols[:20]:  # Check first 20 numeric columns to avoid long processing
            if df[col].nunique() <= 1:
                single_value_cols.append(col)
        if single_value_cols:
            issues.append(f"Single-value columns: {len(single_value_cols)} columns")
    
    # Check for potential header rows in data
    if len(df) > 1:
        # Check if first few rows have mostly string values while rest have numbers
        first_row_types = [type(val).__name__ for val in df.iloc[0]]
        if len(df) > 5:
            fifth_row_types = [type(val).__name__ for val in df.iloc[4]]
            str_in_first = sum(1 for t in first_row_types if t == 'str')
            str_in_fifth = sum(1 for t in fifth_row_types if t == 'str')
            if str_in_first > str_in_fifth * 2:
                issues.append("Potential header row mixed in data")
    
    # Check for inconsistent data types in columns
    inconsistent_cols = []
    for col in df.columns[:20]:  # Check first 20 columns
        if df[col].dtype == 'object':
            # Check if column has mix of numbers and strings
            sample_vals = df[col].dropna().head(100)
            if len(sample_vals) > 0:
                numeric_count = sum(1 for val in sample_vals if str(val).replace('.', '').replace('-', '').isdigit())
                if 0 < numeric_count < len(sample_vals):
                    inconsistent_cols.append(col)
    
    if inconsistent_cols:
        issues.append(f"Columns with mixed data types: {len(inconsistent_cols)}")
    
    if issues:
        print("❌ Issues found:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("✅ No major data quality issues detected")
    
    # Summary Statistics for Numeric Columns
    if len(numeric_cols) > 0:
        print(f"\n📈 NUMERIC COLUMNS SUMMARY (showing first 5 of {len(numeric_cols)}):")
        numeric_summary = df[numeric_cols[:5]].describe()
        print(numeric_summary)
        
        # Check for potential outliers in numeric columns
        print("\n🎯 OUTLIER DETECTION (first 5 numeric columns):")
        for col in numeric_cols[:5]:
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
            print(f"  {col}: {len(outliers)} potential outliers ({len(outliers)/len(df)*100:.1f}%)")
    
    # Sample Data Preview
    print(f"\n👀 DATA PREVIEW:")
    print("First 3 rows:")
    print(df.head(3))
    print("\nLast 3 rows:")
    print(df.tail(3))
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80 + "\n")

    return df, numeric_cols
