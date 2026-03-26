"""
File validator tool for DEG Pipeline Agent.
"""

import pandas as pd
from pathlib import Path
from typing import Union, Dict, List

from .base_tool import BaseTool
from ..exceptions import ValidationError


class FileValidatorTool(BaseTool):
    """Tool for validating processed files."""
    
    @property
    def name(self) -> str:
        return "FileValidator"
    
    @property
    def description(self) -> str:
        return "Validate processed counts and metadata files"
    
    def execute(self, counts_file: Union[str, Path], metadata_file: Union[str, Path], **kwargs) -> Dict:
        """
        Validate processed files.
        
        Args:
            counts_file: Path to counts file
            metadata_file: Path to metadata file
            **kwargs: Additional parameters
            
        Returns:
            Validation results
            
        Raises:
            ValidationError: If validation fails
        """
        counts_file = Path(counts_file)
        metadata_file = Path(metadata_file)
        
        validation_results = {
            "counts_validation": self._validate_counts_file(counts_file),
            "metadata_validation": self._validate_metadata_file(metadata_file),
            "consistency_validation": self._validate_consistency(counts_file, metadata_file)
        }
        
        # Check if all validations passed
        all_passed = all(
            result["status"] == "valid" 
            for result in validation_results.values()
        )
        
        overall_status = "valid" if all_passed else "invalid"
        
        return {
            "status": overall_status,
            "results": validation_results,
            "summary": self._generate_validation_summary(validation_results)
        }
    
    def _validate_counts_file(self, counts_file: Path) -> Dict:
        """
        Validate counts file.
        
        Args:
            counts_file: Path to counts file
            
        Returns:
            Validation result
        """
        try:
            # Check file exists
            if not counts_file.exists():
                return {"status": "invalid", "error": "File does not exist"}
            
            # Check file size
            if counts_file.stat().st_size == 0:
                return {"status": "invalid", "error": "File is empty"}
            
            # Load and validate data (support Excel formats)
            file_ext = counts_file.suffix.lower()
            if file_ext in ['.xlsx', '.xls']:
                try:
                    counts_df = pd.read_excel(counts_file, index_col=0, engine='openpyxl' if file_ext == '.xlsx' else None)
                except Exception as e:
                    # Fallback: try without index_col
                    counts_df = pd.read_excel(counts_file, index_col=None, engine='openpyxl' if file_ext == '.xlsx' else None)
                    if counts_df.shape[1] > 1:
                        counts_df = counts_df.set_index(counts_df.columns[0])
            else:
                sep = "," if file_ext == ".csv" else "\t"
                counts_df = pd.read_csv(counts_file, sep=sep, index_col=0)
            
            issues = []
            
            # Check DataFrame is not empty
            if counts_df.empty:
                issues.append("DataFrame is empty")
            
            # Check minimum samples
            if counts_df.shape[1] < 2:
                issues.append(f"Too few samples: {counts_df.shape[1]} (minimum 2)")
            
            # Check minimum genes
            if counts_df.shape[0] < 10:
                issues.append(f"Too few genes: {counts_df.shape[0]} (minimum 10)")
            
            # Check for non-numeric data
            non_numeric_cols = []
            for col in counts_df.columns:
                if not pd.api.types.is_numeric_dtype(counts_df[col]):
                    non_numeric_cols.append(col)
            
            if non_numeric_cols:
                issues.append(f"Non-numeric columns: {non_numeric_cols}")
            
            # Check for negative values
            if (counts_df < 0).any().any():
                issues.append("Contains negative values")
            
            # Check for all-zero genes
            zero_genes = (counts_df == 0).all(axis=1)
            if zero_genes.any():
                issues.append(f"Found {zero_genes.sum()} genes with all zero counts")
            
            # Check for all-zero samples
            zero_samples = (counts_df == 0).all(axis=0)
            if zero_samples.any():
                issues.append(f"Found {zero_samples.sum()} samples with all zero counts")
            
            if issues:
                return {
                    "status": "invalid",
                    "issues": issues,
                    "shape": counts_df.shape
                }
            else:
                return {
                    "status": "valid",
                    "shape": counts_df.shape,
                    "summary": f"{counts_df.shape[0]} genes × {counts_df.shape[1]} samples"
                }
                
        except Exception as e:
            return {"status": "invalid", "error": f"Failed to validate: {e}"}
    
    def _validate_metadata_file(self, metadata_file: Path) -> Dict:
        """
        Validate metadata file.
        
        Args:
            metadata_file: Path to metadata file
            
        Returns:
            Validation result
        """
        try:
            # Check file exists
            if not metadata_file.exists():
                return {"status": "invalid", "error": "File does not exist"}
            
            # Check file size
            if metadata_file.stat().st_size == 0:
                return {"status": "invalid", "error": "File is empty"}
            
            # Load and validate data (support Excel formats)
            file_ext = metadata_file.suffix.lower()
            if file_ext in ['.xlsx', '.xls']:
                metadata_df = pd.read_excel(metadata_file, index_col=None, engine='openpyxl' if file_ext == '.xlsx' else None)
            else:
                sep = "," if file_ext == ".csv" else "\t"
                metadata_df = pd.read_csv(metadata_file, sep=sep)
            
            issues = []
            
            # Check DataFrame is not empty
            if metadata_df.empty:
                issues.append("DataFrame is empty")
            
            # Check required columns
            required_columns = ["sample", "condition"]
            missing_columns = [col for col in required_columns if col not in metadata_df.columns]
            if missing_columns:
                issues.append(f"Missing required columns: {missing_columns}")
            
            if "condition" in metadata_df.columns:
                # Check for at least 2 conditions
                unique_conditions = metadata_df["condition"].unique()
                if len(unique_conditions) < 2:
                    issues.append(f"Need at least 2 conditions, found: {unique_conditions}")
                
                # Check for missing condition values
                missing_conditions = metadata_df["condition"].isna().sum()
                if missing_conditions > 0:
                    issues.append(f"Found {missing_conditions} missing condition values")
            
            # Check for duplicate samples
            if "sample" in metadata_df.columns:
                duplicate_samples = metadata_df["sample"].duplicated().sum()
                if duplicate_samples > 0:
                    issues.append(f"Found {duplicate_samples} duplicate samples")
            
            if issues:
                return {
                    "status": "invalid",
                    "issues": issues,
                    "shape": metadata_df.shape
                }
            else:
                n_conditions = len(metadata_df["condition"].unique()) if "condition" in metadata_df.columns else 0
                return {
                    "status": "valid",
                    "shape": metadata_df.shape,
                    "n_conditions": n_conditions,
                    "summary": f"{len(metadata_df)} samples, {n_conditions} conditions"
                }
                
        except Exception as e:
            return {"status": "invalid", "error": f"Failed to validate: {e}"}
    
    def _validate_consistency(self, counts_file: Path, metadata_file: Path) -> Dict:
        """
        Validate consistency between counts and metadata files.
        
        Args:
            counts_file: Path to counts file
            metadata_file: Path to metadata file
            
        Returns:
            Consistency validation result
        """
        try:
            # Load both files (support Excel formats)
            counts_ext = counts_file.suffix.lower()
            meta_ext = metadata_file.suffix.lower()
            
            if counts_ext in ['.xlsx', '.xls']:
                try:
                    counts_df = pd.read_excel(counts_file, index_col=0, engine='openpyxl' if counts_ext == '.xlsx' else None)
                except Exception:
                    counts_df = pd.read_excel(counts_file, index_col=None, engine='openpyxl' if counts_ext == '.xlsx' else None)
                    if counts_df.shape[1] > 1:
                        counts_df = counts_df.set_index(counts_df.columns[0])
            else:
                sep_counts = "," if counts_ext == ".csv" else "\t"
                counts_df = pd.read_csv(counts_file, sep=sep_counts, index_col=0)
            
            if meta_ext in ['.xlsx', '.xls']:
                metadata_df = pd.read_excel(metadata_file, index_col=None, engine='openpyxl' if meta_ext == '.xlsx' else None)
            else:
                sep_meta = "," if meta_ext == ".csv" else "\t"
                metadata_df = pd.read_csv(metadata_file, sep=sep_meta)
            
            issues = []
            
            # Get sample lists
            counts_samples = set(counts_df.columns)
            
            if "sample" in metadata_df.columns:
                metadata_samples = set(metadata_df["sample"])
            else:
                metadata_samples = set()
                issues.append("No 'sample' column in metadata")
            
            # Check sample overlap
            common_samples = counts_samples.intersection(metadata_samples)
            
            if not common_samples:
                issues.append("No common samples between counts and metadata")
            elif len(common_samples) < 2:
                issues.append(f"Too few common samples: {len(common_samples)} (minimum 2)")
            
            # Check for missing samples
            missing_from_metadata = counts_samples - metadata_samples
            missing_from_counts = metadata_samples - counts_samples
            
            if missing_from_metadata:
                issues.append(f"Samples in counts but not metadata: {len(missing_from_metadata)}")
            
            if missing_from_counts:
                issues.append(f"Samples in metadata but not counts: {len(missing_from_counts)}")
            
            if issues:
                return {
                    "status": "invalid",
                    "issues": issues,
                    "common_samples": len(common_samples),
                    "counts_samples": len(counts_samples),
                    "metadata_samples": len(metadata_samples)
                }
            else:
                return {
                    "status": "valid",
                    "common_samples": len(common_samples),
                    "counts_samples": len(counts_samples),
                    "metadata_samples": len(metadata_samples),
                    "summary": f"{len(common_samples)} common samples"
                }
                
        except Exception as e:
            return {"status": "invalid", "error": f"Failed to validate consistency: {e}"}
    
    def _generate_validation_summary(self, validation_results: Dict) -> List[str]:
        """
        Generate validation summary.
        
        Args:
            validation_results: Validation results
            
        Returns:
            List of summary messages
        """
        summary = []
        
        for validation_type, result in validation_results.items():
            status = result.get("status", "unknown")
            
            if status == "valid":
                summary.append(f"✅ {validation_type}: PASSED")
                if "summary" in result:
                    summary.append(f"   {result['summary']}")
            else:
                summary.append(f"❌ {validation_type}: FAILED")
                if "error" in result:
                    summary.append(f"   Error: {result['error']}")
                if "issues" in result:
                    for issue in result["issues"]:
                        summary.append(f"   - {issue}")
        
        return summary
    
    def validate_input(self, counts_file: Union[str, Path], metadata_file: Union[str, Path], **kwargs) -> None:
        """
        Validate input parameters.
        
        Args:
            counts_file: Path to counts file
            metadata_file: Path to metadata file
            **kwargs: Additional parameters
            
        Raises:
            ValidationError: If validation fails
        """
        if not counts_file:
            raise ValidationError("Counts file path cannot be empty")
        
        if not metadata_file:
            raise ValidationError("Metadata file path cannot be empty")
    
    def validate_output(self, result: Dict) -> Dict:
        """
        Validate output result.
        
        Args:
            result: Output result
            
        Returns:
            Validated result
            
        Raises:
            ValidationError: If validation fails
        """
        required_fields = ["status", "results", "summary"]
        
        for field in required_fields:
            if field not in result:
                raise ValidationError(f"Missing required field in result: {field}")
        
        if result["status"] not in ["valid", "invalid"]:
            raise ValidationError(f"Invalid status: {result['status']}")
        
        return result 