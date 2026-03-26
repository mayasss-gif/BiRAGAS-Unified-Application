import pandas as pd
import numpy as np
import logging
from typing import Optional, Union, List, Dict, Any, Tuple
import warnings

from ..helpers import logger

def validate_numeric_column(
    series: pd.Series, 
    column_name: str, 
    require_all_finite: bool = True,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
    allow_zero: bool = True,
    return_stats: bool = False
) -> Union[pd.Series, Tuple[pd.Series, Dict[str, Any]]]:
    """
    Enhanced numeric column validation with range checking and statistical reporting.
    
    Args:
        series: Input pandas Series to validate
        column_name: Name of the column for logging
        require_all_finite: Whether to require all values to be finite
        min_value: Minimum allowed value (inclusive)
        max_value: Maximum allowed value (inclusive)
        allow_zero: Whether zero values are allowed
        return_stats: Whether to return validation statistics
        
    Returns:
        Validated Series or tuple of (Series, stats_dict) if return_stats=True
        
    Raises:
        ValueError: If validation fails and require_all_finite=True
    """
    if series.empty:
        logger.warning("Column '%s' is empty.", column_name)
        if return_stats:
            return series, {"empty": True, "nan_count": 0, "total": 0}
        return series
    
    # Store original info
    original_count = len(series)
    original_dtype = series.dtype
    
    # Coerce to numeric
    coerced = pd.to_numeric(series, errors="coerce")
    nan_count = coerced.isna().sum()
    
    # Enhanced logging for non-numeric values
    if nan_count > 0:
        non_numeric_samples = series[coerced.isna()].unique()[:5]  # Show up to 5 examples
        logger.warning(
            "Column '%s' has %d/%d non-numeric entries (%.2f%%); these will be treated as NaN. "
            "Examples: %s", 
            column_name, nan_count, original_count, 
            (nan_count / original_count) * 100,
            list(non_numeric_samples)
        )
    
    # Check for infinite values
    finite_mask = np.isfinite(coerced)
    inf_count = (~finite_mask & ~coerced.isna()).sum()
    if inf_count > 0:
        logger.warning(
            "Column '%s' contains %d infinite values.", 
            column_name, inf_count
        )
    
    # Apply finite filter
    if require_all_finite:
        result = coerced[finite_mask]
        if result.empty:
            logger.error("Column '%s' contains no finite numeric values.", column_name)
            raise ValueError(f"Column '{column_name}' must contain finite numeric values.")
    else:
        result = coerced
    
    # Range validation
    range_violations = 0
    if not result.empty:
        if min_value is not None:
            below_min = (result < min_value).sum()
            if below_min > 0:
                logger.warning(
                    "Column '%s' has %d values below minimum threshold %.2f",
                    column_name, below_min, min_value
                )
                range_violations += below_min
                result = result[result >= min_value]
        
        if max_value is not None:
            above_max = (result > max_value).sum()
            if above_max > 0:
                logger.warning(
                    "Column '%s' has %d values above maximum threshold %.2f",
                    column_name, above_max, max_value
                )
                range_violations += above_max
                result = result[result <= max_value]
        
        if not allow_zero:
            zero_count = (result == 0).sum()
            if zero_count > 0:
                logger.warning(
                    "Column '%s' has %d zero values which are not allowed",
                    column_name, zero_count
                )
                range_violations += zero_count
                result = result[result != 0]
    
    # Compile statistics
    if return_stats or logger.isEnabledFor(logging.INFO):
        valid_count = len(result)
        stats = {
            "original_count": original_count,
            "original_dtype": str(original_dtype),
            "nan_count": nan_count,
            "inf_count": inf_count,
            "range_violations": range_violations,
            "valid_count": valid_count,
            "retention_rate": valid_count / original_count if original_count > 0 else 0,
            "empty": result.empty
        }
        
        if not result.empty:
            stats.update({
                "min": float(result.min()),
                "max": float(result.max()),
                "mean": float(result.mean()),
                "std": float(result.std()),
                "median": float(result.median())
            })
        
        if logger.isEnabledFor(logging.INFO):
            logger.info(
                "Column '%s' validation: %d/%d valid (%.2f%% retention)",
                column_name, valid_count, original_count, stats["retention_rate"] * 100
            )
    
    if return_stats:
        return result, stats
    return result


def drop_na_in_columns(
    df: pd.DataFrame, 
    columns: Union[str, List[str]],
    how: str = "any",
    threshold: Optional[int] = None,
    report_details: bool = True,
    inplace: bool = False
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, Dict[str, Any]]]:
    """
    Enhanced function to drop rows with NaN values in specified columns.
    
    Args:
        df: Input DataFrame
        columns: Column name(s) to check for NaN values
        how: 'any' or 'all' - drop if any/all specified columns have NaN
        threshold: Minimum number of non-NaN values required in specified columns
        report_details: Whether to log detailed information about dropped rows
        inplace: Whether to modify DataFrame in place
        
    Returns:
        Cleaned DataFrame or tuple of (DataFrame, report_dict) if report_details=True
        
    Raises:
        ValueError: If specified columns don't exist in DataFrame
    """
    if df.empty:
        logger.warning("Input DataFrame is empty.")
        return df.copy() if not inplace else df
    
    # Handle single column input
    if isinstance(columns, str):
        columns = [columns]
    
    # Validate columns exist
    missing_cols = set(columns) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Columns not found in DataFrame: {list(missing_cols)}")
    
    # Store original info
    original_shape = df.shape
    original_index = df.index.copy()
    
    # Analyze NaN patterns before dropping
    if report_details:
        nan_analysis = {}
        for col in columns:
            nan_mask = df[col].isna()
            nan_analysis[col] = {
                "nan_count": nan_mask.sum(),
                "nan_percentage": (nan_mask.sum() / len(df)) * 100
            }
        
        # Find rows that will be dropped
        if how == "any":
            drop_mask = df[columns].isna().any(axis=1)
        elif how == "all":
            drop_mask = df[columns].isna().all(axis=1)
        else:
            raise ValueError("Parameter 'how' must be 'any' or 'all'")
        
        if threshold is not None:
            non_na_counts = df[columns].notna().sum(axis=1)
            drop_mask = drop_mask | (non_na_counts < threshold)
    
    # Perform the actual dropping
    if threshold is not None:
        cleaned = df.dropna(subset=columns, how=how, thresh=threshold, inplace=inplace)
    else:
        cleaned = df.dropna(subset=columns, how=how, inplace=inplace)
    
    if inplace:
        cleaned = df
    
    # Calculate and report results
    final_shape = cleaned.shape
    dropped_count = original_shape[0] - final_shape[0]
    retention_rate = final_shape[0] / original_shape[0] if original_shape[0] > 0 else 1.0
    
    if report_details and dropped_count > 0:
        dropped_indices = original_index.difference(cleaned.index)
        
        # Log summary
        logger.info(
            "Dropped %d/%d rows (%.2f%%) due to NaN in columns %s (method: %s).",
            dropped_count, original_shape[0], (1 - retention_rate) * 100, 
            columns, how
        )
        
        # Log per-column NaN statistics
        for col, stats in nan_analysis.items():
            if stats["nan_count"] > 0:
                logger.info(
                    "  Column '%s': %d NaN values (%.2f%%)",
                    col, stats["nan_count"], stats["nan_percentage"]
                )
        
        # Create detailed report
        report = {
            "original_shape": original_shape,
            "final_shape": final_shape,
            "dropped_count": dropped_count,
            "retention_rate": retention_rate,
            "columns_checked": columns,
            "method": how,
            "threshold": threshold,
            "nan_analysis": nan_analysis,
            "dropped_indices": dropped_indices.tolist()[:100]  # Limit to first 100 for memory
        }
        
        return (cleaned, report) if not inplace else report
    
    elif dropped_count > 0:
        logger.info(
            "Dropped %d rows due to NaN in columns %s.", 
            dropped_count, columns
        )
    
    return cleaned


def validate_dataframe_integrity(
    df: pd.DataFrame,
    required_columns: Optional[List[str]] = None,
    numeric_columns: Optional[List[str]] = None,
    categorical_columns: Optional[List[str]] = None,
    min_rows: int = 1,
    allow_duplicates: bool = True,
    memory_efficient: bool = False
) -> Dict[str, Any]:
    """
    Comprehensive DataFrame integrity validation.
    
    Args:
        df: DataFrame to validate
        required_columns: Columns that must be present
        numeric_columns: Columns that should be numeric
        categorical_columns: Columns that should be categorical/string
        min_rows: Minimum number of rows required
        allow_duplicates: Whether duplicate rows are allowed
        memory_efficient: Whether to use memory-efficient operations
        
    Returns:
        Dictionary with validation results and recommendations
    """
    validation_results = {
        "valid": True,
        "issues": [],
        "warnings": [],
        "info": {
            "shape": df.shape,
            "memory_usage_mb": df.memory_usage(deep=True).sum() / 1024**2,
            "dtypes": df.dtypes.to_dict()
        }
    }
    
    # Check basic requirements
    if df.empty:
        validation_results["valid"] = False
        validation_results["issues"].append("DataFrame is empty")
        return validation_results
    
    if len(df) < min_rows:
        validation_results["valid"] = False
        validation_results["issues"].append(f"DataFrame has {len(df)} rows, minimum required: {min_rows}")
    
    # Check required columns
    if required_columns:
        missing_cols = set(required_columns) - set(df.columns)
        if missing_cols:
            validation_results["valid"] = False
            validation_results["issues"].append(f"Missing required columns: {list(missing_cols)}")
    
    # Validate numeric columns
    if numeric_columns:
        for col in numeric_columns:
            if col in df.columns:
                non_numeric = pd.to_numeric(df[col], errors="coerce").isna().sum()
                if non_numeric > 0:
                    validation_results["warnings"].append(
                        f"Column '{col}' has {non_numeric} non-numeric values"
                    )
    
    # Check for duplicates
    if not allow_duplicates:
        duplicate_count = df.duplicated().sum()
        if duplicate_count > 0:
            validation_results["valid"] = False
            validation_results["issues"].append(f"Found {duplicate_count} duplicate rows")
    
    # Memory and performance warnings
    if validation_results["info"]["memory_usage_mb"] > 1000:  # > 1GB
        validation_results["warnings"].append(
            f"Large memory usage: {validation_results['info']['memory_usage_mb']:.2f} MB"
        )
    
    # Check for completely empty columns
    empty_cols = [col for col in df.columns if df[col].isna().all()]
    if empty_cols:
        validation_results["warnings"].append(f"Completely empty columns: {empty_cols}")
    
    # Log results
    if not validation_results["valid"]:
        logger.error("DataFrame validation failed: %s", validation_results["issues"])
    elif validation_results["warnings"]:
        logger.warning("DataFrame validation warnings: %s", validation_results["warnings"])
    else:
        logger.info("DataFrame validation passed successfully")
    
    return validation_results