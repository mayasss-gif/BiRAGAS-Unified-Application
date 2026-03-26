"""
Data loader tool for DEG Pipeline Agent.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Union, Tuple

from .base_tool import BaseTool
from ..exceptions import DataLoadError, RecoverableError, ValidationError


class DataLoaderTool(BaseTool):
    """Tool for loading and sanitizing count data."""
    
    @property
    def name(self) -> str:
        return "DataLoader"
    
    @property
    def description(self) -> str:
        return "Load and sanitize count data from various file formats"
    
    def execute(self, counts_file: Union[str, Path], **kwargs) -> pd.DataFrame:
        """
        Load and sanitize count data.
        
        Args:
            counts_file: Path to the counts file
            **kwargs: Additional parameters
            
        Returns:
            Sanitized DataFrame with samples as rows and genes as columns
            
        Raises:
            DataLoadError: If loading fails
        """
        counts_file = Path(counts_file)
        
        # Load the data
        counts_df = self._load_counts_file(counts_file)
        
        # Sanitize the data
        counts_df = self._sanitize_counts(counts_df)
        
        # Orient correctly (samples as rows, genes as columns)
        counts_df = self._orient_dataframe(counts_df)
        
        # Filter low-information genes
        counts_df = self._filter_genes(counts_df)
        
        # Subsample if too many genes
        counts_df = self._subsample_genes(counts_df)
        
        self.logger.info(f"✅ Loaded counts: {counts_df.shape[0]} samples × {counts_df.shape[1]} genes")
        return counts_df
    
    def _load_counts_file(self, counts_file: Path) -> pd.DataFrame:
        """
        Load counts file with automatic format detection.
        
        Supports: CSV, TSV, TXT, XLSX, XLS
        
        Args:
            counts_file: Path to the counts file
            
        Returns:
            Raw counts DataFrame
            
        Raises:
            DataLoadError: If file cannot be loaded
        """
        if not counts_file.exists():
            raise DataLoadError(f"Counts file does not exist: {counts_file}")
        
        if counts_file.stat().st_size == 0:
            raise DataLoadError(f"Counts file is empty: {counts_file}")
        
        file_ext = counts_file.suffix.lower()
        
        # Handle Excel files (.xlsx, .xls)
        if file_ext in ['.xlsx', '.xls']:
            return self._load_excel_file(counts_file)
        
        # Handle text-based formats (CSV, TSV, TXT)
        # Determine separator based on file extension
        if file_ext == '.csv':
            sep = ','
        elif file_ext in ['.tsv', '.txt']:
            sep = '\t'
        else:
            sep = None  # Let pandas auto-detect
        
        try:
            # Try loading with determined separator
            if sep:
                counts_df = pd.read_csv(counts_file, sep=sep, compression='infer', index_col=0)
            else:
                counts_df = pd.read_csv(counts_file, sep=None, engine='python', compression='infer', index_col=0)
            
            if counts_df.empty:
                raise DataLoadError(f"Loaded DataFrame is empty: {counts_file}")
            
            return counts_df
            
        except Exception as e:
            # Try fallback approaches
            return self._load_with_fallback(counts_file, str(e))
    
    def _load_excel_file(self, counts_file: Path) -> pd.DataFrame:
        """
        Load Excel file (.xlsx or .xls) with intelligent sheet and header detection.
        
        Args:
            counts_file: Path to the Excel file
            
        Returns:
            Raw counts DataFrame
            
        Raises:
            DataLoadError: If file cannot be loaded
        """
        try:
            # Try to read Excel file with multiple strategies
            excel_file = pd.ExcelFile(counts_file)
            
            # Strategy 1: Try first sheet with default header (row 0)
            try:
                counts_df = pd.read_excel(counts_file, sheet_name=0, index_col=0, engine='openpyxl' if counts_file.suffix.lower() == '.xlsx' else None)
                
                if not counts_df.empty:
                    self.logger.info(f"✅ Loaded Excel file: {counts_file.name} (sheet 0, header 0)")
                    return counts_df
            except Exception as e:
                self.logger.debug(f"Failed to load with index_col=0: {e}")
            
            # Strategy 2: Try first sheet without index_col (will set index later)
            try:
                counts_df = pd.read_excel(counts_file, sheet_name=0, index_col=None, engine='openpyxl' if counts_file.suffix.lower() == '.xlsx' else None)
                
                if not counts_df.empty:
                    # Try to identify gene column (first column is often gene ID)
                    if counts_df.shape[1] > 1:
                        # Use first column as index if it looks like gene IDs
                        first_col = counts_df.columns[0]
                        first_col_values = counts_df[first_col].astype(str)
                        
                        # Check if first column looks like gene identifiers
                        # (contains gene-like patterns or is mostly unique)
                        is_gene_col = (
                            first_col_values.nunique() > counts_df.shape[0] * 0.8 or  # Mostly unique
                            any(keyword in str(first_col).lower() for keyword in ['gene', 'id', 'symbol', 'ensembl', 'entrez'])
                        )
                        
                        if is_gene_col:
                            counts_df = counts_df.set_index(first_col)
                            self.logger.info(f"✅ Loaded Excel file: {counts_file.name} (sheet 0, auto-indexed first column)")
                        else:
                            # Create default index
                            counts_df.index.name = 'Gene'
                            self.logger.info(f"✅ Loaded Excel file: {counts_file.name} (sheet 0, default index)")
                    
                    return counts_df
            except Exception as e:
                self.logger.debug(f"Failed to load without index_col: {e}")
            
            # Strategy 3: Try other sheets if first sheet failed
            if len(excel_file.sheet_names) > 1:
                for sheet_name in excel_file.sheet_names[1:]:
                    try:
                        counts_df = pd.read_excel(counts_file, sheet_name=sheet_name, index_col=0, engine='openpyxl' if counts_file.suffix.lower() == '.xlsx' else None)
                        
                        if not counts_df.empty:
                            self.logger.info(f"✅ Loaded Excel file: {counts_file.name} (sheet: {sheet_name})")
                            return counts_df
                    except Exception as e:
                        self.logger.debug(f"Failed to load sheet {sheet_name}: {e}")
                        continue
            
            # Strategy 4: Try with header=None and infer structure
            try:
                counts_df = pd.read_excel(counts_file, sheet_name=0, header=None, engine='openpyxl' if counts_file.suffix.lower() == '.xlsx' else None)
                
                if not counts_df.empty and counts_df.shape[0] > 1:
                    # Use first row as column names, first column as index
                    counts_df.columns = counts_df.iloc[0]
                    counts_df = counts_df.iloc[1:].set_index(counts_df.columns[0])
                    self.logger.info(f"✅ Loaded Excel file: {counts_file.name} (inferred structure)")
                    return counts_df
            except Exception as e:
                self.logger.debug(f"Failed to load with inferred structure: {e}")
            
            raise DataLoadError(f"Failed to load Excel file after all strategies: {counts_file}")
            
        except ImportError as e:
            raise DataLoadError(
                f"Excel file support requires 'openpyxl' package. "
                f"Install with: pip install openpyxl. Error: {e}"
            )
        except Exception as e:
            raise DataLoadError(f"Failed to load Excel file {counts_file}: {str(e)}")
    
    def _load_with_fallback(self, counts_file: Path, original_error: str) -> pd.DataFrame:
        """
        Try multiple approaches to load the file.
        
        Args:
            counts_file: Path to the counts file
            original_error: Original error message
            
        Returns:
            Loaded DataFrame
            
        Raises:
            DataLoadError: If all approaches fail
        """
        # Don't try Excel fallbacks for non-Excel files
        file_ext = counts_file.suffix.lower()
        if file_ext in ['.xlsx', '.xls']:
            # Excel files should have been handled in _load_excel_file
            raise DataLoadError(f"Failed to load Excel file. Original error: {original_error}")
        
        fallback_attempts = [
            {'sep': '\t', 'engine': 'python'},
            {'sep': ',', 'engine': 'python'},
            {'sep': ' ', 'engine': 'python'},
            {'sep': None, 'engine': 'python', 'index_col': None},
            {'sep': '\t', 'engine': 'python', 'index_col': None},
            {'sep': ',', 'engine': 'python', 'index_col': None},
        ]
        
        for i, params in enumerate(fallback_attempts):
            try:
                self.logger.warning(f"🔄 Fallback attempt {i+1}: {params}")
                counts_df = pd.read_csv(counts_file, compression='infer', **params)
                
                if counts_df.empty:
                    continue
                
                # If no index was set, use first column as index
                if params.get('index_col') is None and counts_df.shape[1] > 1:
                    counts_df = counts_df.set_index(counts_df.columns[0])
                
                self.logger.info(f"✅ Fallback successful on attempt {i+1}")
                return counts_df
                
            except Exception as e:
                self.logger.warning(f"❌ Fallback attempt {i+1} failed: {e}")
                continue
        
        raise DataLoadError(f"Failed to load counts file after all attempts. Original error: {original_error}")
    
    def _sanitize_counts(self, counts_df: pd.DataFrame) -> pd.DataFrame:
        """
        Sanitize count data by handling inf values and converting to integers.
        
        Args:
            counts_df: Raw counts DataFrame
            
        Returns:
            Sanitized counts DataFrame
        """
        # Replace inf/-inf with NaN, then fill NaN with 0
        counts_df = counts_df.replace([np.inf, -np.inf], np.nan)
        counts_df = counts_df.fillna(0)
        
        # Convert to numeric, handling any remaining non-numeric values
        numeric_columns = []
        for col in counts_df.columns:
            try:
                counts_df[col] = pd.to_numeric(counts_df[col], errors='coerce')
                numeric_columns.append(col)
            except Exception:
                self.logger.warning(f"⚠️  Skipping non-numeric column: {col}")
        
        if not numeric_columns:
            raise DataLoadError("No numeric columns found in counts data")
        
        # Keep only numeric columns
        counts_df = counts_df[numeric_columns]
        
        # Fill any remaining NaN values with 0
        counts_df = counts_df.fillna(0)
        
        # Round and convert to integer
        counts_df = counts_df.round().astype(int)
        
        return counts_df
    
    def _orient_dataframe(self, counts_df: pd.DataFrame) -> pd.DataFrame:
        """
        Orient DataFrame so samples are rows and genes are columns.
        
        Args:
            counts_df: Counts DataFrame
            
        Returns:
            Properly oriented DataFrame
        """
        # If there are more rows than columns, assume genes are rows (need to transpose)
        if counts_df.shape[0] > counts_df.shape[1]:
            self.logger.info("🔄 Transposing DataFrame (genes as rows → samples as rows)")
            counts_df = counts_df.T
        
        return counts_df
    
    def _filter_genes(self, counts_df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter out genes with low information content.
        
        Args:
            counts_df: Counts DataFrame
            
        Returns:
            Filtered DataFrame
        """
        min_samples = self.config.min_samples_per_gene
        
        # Count non-zero samples per gene
        non_zero_counts = (counts_df > 0).sum(axis=0)
        
        # Keep genes detected in at least min_samples_per_gene samples
        genes_to_keep = non_zero_counts >= min_samples
        
        filtered_df = counts_df.loc[:, genes_to_keep]
        
        removed_genes = counts_df.shape[1] - filtered_df.shape[1]
        if removed_genes > 0:
            self.logger.info(f"🗑️  Removed {removed_genes} low-information genes")
        
        if filtered_df.empty:
            raise DataLoadError("No genes pass the minimum sample threshold")
        
        return filtered_df
    
    def _subsample_genes(self, counts_df: pd.DataFrame) -> pd.DataFrame:
        """
        Subsample genes if there are too many.
        
        Args:
            counts_df: Counts DataFrame
            
        Returns:
            Subsampled DataFrame
        """
        max_genes = self.config.max_genes
        
        if counts_df.shape[1] <= max_genes:
            return counts_df
        
        # Calculate variance for each gene
        gene_variances = counts_df.var(axis=0)
        
        # Keep top variable genes
        top_genes = gene_variances.nlargest(max_genes).index
        subsampled_df = counts_df[top_genes]
        
        self.logger.info(f"📊 Subsampled to top {max_genes} most variable genes")
        
        return subsampled_df
    
    def validate_input(self, counts_file: Union[str, Path], **kwargs) -> None:
        """
        Validate input parameters.
        
        Args:
            counts_file: Path to counts file
            **kwargs: Additional parameters
            
        Raises:
            ValidationError: If validation fails
        """
        if not counts_file:
            raise ValidationError("Counts file path cannot be empty")
        
        counts_file = Path(counts_file)
        
        if not counts_file.exists():
            raise ValidationError(f"Counts file does not exist: {counts_file}")
        
        if counts_file.suffix.lower() not in self.config.supported_formats:
            raise ValidationError(f"Unsupported file format: {counts_file.suffix}")
    
    def validate_output(self, result: pd.DataFrame) -> pd.DataFrame:
        """
        Validate output DataFrame.
        
        Args:
            result: Output DataFrame
            
        Returns:
            Validated DataFrame
            
        Raises:
            ValidationError: If validation fails
        """
        if result.empty:
            raise ValidationError("Output DataFrame is empty")
        
        if result.shape[0] < 2:
            raise ValidationError("Need at least 2 samples for analysis")
        
        if result.shape[1] < 10:
            raise ValidationError("Need at least 10 genes for meaningful analysis")
        
        # Check for negative values
        if (result < 0).any().any():
            raise ValidationError("Count data contains negative values")
        
        # Check for all-zero columns
        zero_genes = (result == 0).all(axis=0)
        if zero_genes.any():
            raise ValidationError(f"Found {zero_genes.sum()} genes with all zero counts")
        
        return result
    
    def _apply_fix(self, fix_suggestion: str) -> None:
        """
        Apply automatic fixes for common issues.
        
        Args:
            fix_suggestion: Fix to apply
        """
        if fix_suggestion == "try_different_separator":
            self.logger.info("🔧 Will try different separators in next attempt")
        elif fix_suggestion == "handle_encoding_error":
            self.logger.info("🔧 Will try different encodings in next attempt")
        else:
            super()._apply_fix(fix_suggestion) 