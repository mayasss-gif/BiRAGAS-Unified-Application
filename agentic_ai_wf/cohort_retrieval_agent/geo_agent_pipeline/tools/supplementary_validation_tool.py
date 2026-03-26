"""
Supplementary File Validation tool for the Cohort Retrieval Agent system.

This tool provides comprehensive validation for supplementary files with
configurable rules, intelligent categorization, and quality assessment.
"""

import asyncio
import gzip
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import numpy as np

# Project Imports
from   ..base.base_tool import AsyncContextTool, ToolResult
from   ..config import CohortRetrievalConfig
from   ..exceptions import ValidationError


class ValidationSeverity(Enum):
    """Severity levels for validation issues."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ValidationIssue:
    """Represents a validation issue."""
    severity: ValidationSeverity
    message: str
    category: str
    file_path: Optional[Path] = None
    line_number: Optional[int] = None
    column_name: Optional[str] = None
    suggested_fix: Optional[str] = None


@dataclass
class FileValidationResult:
    """Result of validating a single file."""
    file_path: Path
    category: Optional[str]
    is_valid: bool
    file_size: int
    row_count: Optional[int] = None
    column_count: Optional[int] = None
    data_type: Optional[str] = None
    issues: List[ValidationIssue] = field(default_factory=list)
    quality_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DatasetValidationResult:
    """Result of validating an entire dataset."""
    dataset_id: str
    total_files: int
    valid_files: int
    invalid_files: int
    files_by_category: Dict[str, List[FileValidationResult]]
    overall_quality_score: float = 0.0
    issues: List[ValidationIssue] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    is_suitable_for_analysis: bool = False


class SupplementaryValidationTool(AsyncContextTool[DatasetValidationResult]):
    """
    Tool for comprehensive validation of supplementary files.
    
    Features:
    - Configurable validation rules
    - File format detection and validation
    - Data quality assessment
    - Statistical validation for count matrices
    - Metadata consistency checking
    - Quality scoring and recommendations
    """
    
    def __init__(self, config: CohortRetrievalConfig):
        super().__init__(config)
        self.geo_config = config.geo_config
        self.validation_rules = self.geo_config.supplementary_config["validation_rules"]
        self.file_categories = self.geo_config.supplementary_config["file_categories"]
        
        # Quality thresholds (configurable)
        self.quality_thresholds = {
            "min_rows": 1000,           # Minimum rows for count matrices
            "min_columns": 10,          # Minimum samples
            "max_zero_fraction": 0.9,   # Maximum fraction of zeros
            "min_mean_counts": 1.0,     # Minimum mean counts per gene
            "correlation_threshold": 0.7 # Minimum correlation for replicates
        }
    
    async def create_context(self) -> None:
        """Create validation context (no external resources needed)."""
        return None
    
    async def close_context(self, context: None):
        """Close validation context (no cleanup needed)."""
        pass
    
    async def execute(self, 
                     dataset_id: str,
                     files_to_validate: List[Path],
                     categories: Optional[Dict[Path, str]] = None) -> ToolResult[DatasetValidationResult]:
        """
        Validate supplementary files for a dataset.
        
        Args:
            dataset_id: GEO dataset ID
            files_to_validate: List of file paths to validate
            categories: Optional mapping of files to categories
            
        Returns:
            ToolResult with validation results
        """
        if not self.validate_input(dataset_id, files_to_validate):
            return ToolResult(
                success=False,
                error="Invalid input parameters",
                details={"dataset_id": dataset_id, "file_count": len(files_to_validate)}
            )
        
        try:
            self.logger.info(f"Starting validation for {dataset_id} with {len(files_to_validate)} files")
            
            # Validate individual files
            file_results = await self._validate_files(files_to_validate, categories or {})
            
            # Categorize results
            files_by_category = self._categorize_results(file_results)
            
            # Perform dataset-level validation
            dataset_issues = await self._validate_dataset_consistency(file_results, dataset_id)
            
            # Calculate overall quality score
            overall_quality = self._calculate_overall_quality(file_results)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(file_results, dataset_issues)
            
            # Determine if suitable for analysis
            is_suitable = self._assess_analysis_suitability(file_results, overall_quality)
            
            result = DatasetValidationResult(
                dataset_id=dataset_id,
                total_files=len(files_to_validate),
                valid_files=sum(1 for r in file_results if r.is_valid),
                invalid_files=sum(1 for r in file_results if not r.is_valid),
                files_by_category=files_by_category,
                overall_quality_score=overall_quality,
                issues=dataset_issues,
                recommendations=recommendations,
                is_suitable_for_analysis=is_suitable
            )
            
            self.logger.info(f"Validation completed for {dataset_id}: "
                           f"{result.valid_files}/{result.total_files} valid files, "
                           f"quality score: {result.overall_quality_score:.2f}")
            
            return ToolResult(success=True, data=result)
            
        except Exception as e:
            self.logger.error(f"Validation failed for {dataset_id}: {e}")
            return ToolResult(
                success=False,
                error=str(e),
                details={"dataset_id": dataset_id, "exception": type(e).__name__}
            )
    
    async def _validate_files(self, 
                            files: List[Path], 
                            categories: Dict[Path, str]) -> List[FileValidationResult]:
        """Validate individual files concurrently."""
        # Create semaphore for concurrent validation
        semaphore = asyncio.Semaphore(5)  # Limit concurrent file operations
        
        # Create validation tasks
        tasks = []
        for file_path in files:
            category = categories.get(file_path)
            task = self._validate_single_file(file_path, category, semaphore)
            tasks.append(task)
        
        # Execute validations concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        file_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                self.logger.error(f"Validation error for {files[i]}: {result}")
                # Create error result
                file_results.append(FileValidationResult(
                    file_path=files[i],
                    category=categories.get(files[i]),
                    is_valid=False,
                    file_size=0,
                    issues=[ValidationIssue(
                        severity=ValidationSeverity.CRITICAL,
                        message=f"Validation failed: {result}",
                        category="system_error",
                        file_path=files[i]
                    )]
                ))
            else:
                file_results.append(result)
        
        return file_results
    
    async def _validate_single_file(self, 
                                  file_path: Path, 
                                  category: Optional[str],
                                  semaphore: asyncio.Semaphore) -> FileValidationResult:
        """Validate a single file."""
        async with semaphore:
            issues = []
            metadata = {}
            
            # Basic file validation
            if not file_path.exists():
                return FileValidationResult(
                    file_path=file_path,
                    category=category,
                    is_valid=False,
                    file_size=0,
                    issues=[ValidationIssue(
                        severity=ValidationSeverity.CRITICAL,
                        message="File does not exist",
                        category="file_system",
                        file_path=file_path
                    )]
                )
            
            file_size = file_path.stat().st_size
            
            # File size validation
            if not self._validate_file_size(file_size):
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    message=f"File size {file_size} bytes is outside expected range",
                    category="file_size",
                    file_path=file_path
                ))
            
            # Detect file type and validate format
            data_type = self._detect_data_type(file_path)
            metadata["data_type"] = data_type
            
            # Content validation based on file type
            row_count = None
            column_count = None
            quality_score = 0.0
            
            try:
                if data_type in ["count_matrix", "expression_matrix"]:
                    content_result = await self._validate_matrix_file(file_path, category)
                    row_count = content_result.get("row_count")
                    column_count = content_result.get("column_count")
                    quality_score = content_result.get("quality_score", 0.0)
                    issues.extend(content_result.get("issues", []))
                    metadata.update(content_result.get("metadata", {}))
                
                elif data_type == "metadata":
                    content_result = await self._validate_metadata_file(file_path)
                    row_count = content_result.get("row_count")
                    column_count = content_result.get("column_count")
                    quality_score = content_result.get("quality_score", 0.0)
                    issues.extend(content_result.get("issues", []))
                
                elif data_type == "archive":
                    content_result = await self._validate_archive_file(file_path)
                    quality_score = content_result.get("quality_score", 0.0)
                    issues.extend(content_result.get("issues", []))
                
                else:
                    # Generic validation for unknown file types
                    quality_score = 0.5  # Neutral score
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.INFO,
                        message=f"Unknown file type: {data_type}",
                        category="file_type",
                        file_path=file_path
                    ))
                
            except Exception as e:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    message=f"Content validation failed: {e}",
                    category="content_validation",
                    file_path=file_path
                ))
                quality_score = 0.0
            
            # Determine if file is valid
            critical_issues = [i for i in issues if i.severity == ValidationSeverity.CRITICAL]
            is_valid = len(critical_issues) == 0
            
            return FileValidationResult(
                file_path=file_path,
                category=category,
                is_valid=is_valid,
                file_size=file_size,
                row_count=row_count,
                column_count=column_count,
                data_type=data_type,
                issues=issues,
                quality_score=quality_score,
                metadata=metadata
            )
    
    def _detect_data_type(self, file_path: Path) -> str:
        """Detect the type of data in a file."""
        filename = file_path.name.lower()
        
        # Count matrices
        if any(keyword in filename for keyword in ["count", "raw_count", "readcount", "featurecount"]):
            return "count_matrix"
        
        # Expression matrices
        if any(keyword in filename for keyword in ["fpkm", "rpkm", "tpm", "cpm", "expression"]):
            return "expression_matrix"
        
        # Metadata files
        if any(keyword in filename for keyword in ["metadata", "annotation", "phenotype", "clinical"]):
            return "metadata"
        
        # Archive files
        if any(filename.endswith(ext) for ext in [".tar", ".zip", ".gz", ".bz2"]):
            return "archive"
        
        # Differential expression
        if any(keyword in filename for keyword in ["diff", "deseq", "edger", "limma"]):
            return "differential_expression"
        
        return "unknown"
    
    async def _validate_matrix_file(self, file_path: Path, category: Optional[str]) -> Dict[str, Any]:
        """Validate count/expression matrix files."""
        issues = []
        metadata = {}
        
        try:
            # Read file (handle compressed files)
            if file_path.suffix == '.gz':
                df = pd.read_csv(file_path, sep='\t', compression='gzip', index_col=0, nrows=1000)
            else:
                df = pd.read_csv(file_path, sep='\t', index_col=0, nrows=1000)
            
            row_count = len(df)
            column_count = len(df.columns)
            
            # Basic structure validation
            if row_count < self.quality_thresholds["min_rows"]:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    message=f"Low gene count: {row_count} (expected >= {self.quality_thresholds['min_rows']})",
                    category="data_quality",
                    file_path=file_path
                ))
            
            if column_count < self.quality_thresholds["min_columns"]:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    message=f"Low sample count: {column_count} (expected >= {self.quality_thresholds['min_columns']})",
                    category="data_quality",
                    file_path=file_path
                ))
            
            # Data type validation
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            if len(numeric_columns) != column_count:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.ERROR,
                    message=f"Non-numeric data found in {column_count - len(numeric_columns)} columns",
                    category="data_type",
                    file_path=file_path
                ))
            
            # Statistical validation for count data
            if category == "raw_counts":
                # Check for negative values
                if (df < 0).any().any():
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        message="Negative values found in count matrix",
                        category="data_integrity",
                        file_path=file_path
                    ))
                
                # Check for excessive zeros
                zero_fraction = (df == 0).sum().sum() / (df.shape[0] * df.shape[1])
                if zero_fraction > self.quality_thresholds["max_zero_fraction"]:
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        message=f"High zero fraction: {zero_fraction:.2f}",
                        category="data_quality",
                        file_path=file_path
                    ))
                
                # Check mean counts
                mean_counts = df.mean().mean()
                if mean_counts < self.quality_thresholds["min_mean_counts"]:
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.WARNING,
                        message=f"Low mean counts: {mean_counts:.2f}",
                        category="data_quality",
                        file_path=file_path
                    ))
            
            # Calculate quality score
            quality_score = self._calculate_matrix_quality_score(df, issues)
            
            # Store metadata
            metadata.update({
                "zero_fraction": zero_fraction if 'zero_fraction' in locals() else None,
                "mean_counts": mean_counts if 'mean_counts' in locals() else None,
                "numeric_columns": len(numeric_columns),
                "sample_names": df.columns.tolist()[:10]  # First 10 sample names
            })
            
            return {
                "row_count": row_count,
                "column_count": column_count,
                "quality_score": quality_score,
                "issues": issues,
                "metadata": metadata
            }
            
        except Exception as e:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.CRITICAL,
                message=f"Failed to read matrix file: {e}",
                category="file_format",
                file_path=file_path
            ))
            
            return {
                "row_count": 0,
                "column_count": 0,
                "quality_score": 0.0,
                "issues": issues,
                "metadata": {}
            }
    
    async def _validate_metadata_file(self, file_path: Path) -> Dict[str, Any]:
        """Validate metadata files."""
        issues = []
        metadata = {}
        
        try:
            # Try to read as CSV/TSV
            if file_path.suffix == '.gz':
                df = pd.read_csv(file_path, sep=None, compression='gzip', engine='python')
            else:
                df = pd.read_csv(file_path, sep=None, engine='python')
            
            row_count = len(df)
            column_count = len(df.columns)
            
            # Check for required columns
            required_columns = ["sample_id", "group", "condition"]
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    message=f"Missing recommended columns: {missing_columns}",
                    category="metadata_structure",
                    file_path=file_path
                ))
            
            # Check for duplicate sample IDs
            if "sample_id" in df.columns:
                duplicates = df["sample_id"].duplicated().sum()
                if duplicates > 0:
                    issues.append(ValidationIssue(
                        severity=ValidationSeverity.ERROR,
                        message=f"Found {duplicates} duplicate sample IDs",
                        category="data_integrity",
                        file_path=file_path
                    ))
            
            quality_score = 0.8 - (len(issues) * 0.1)  # Reduce score for each issue
            quality_score = max(0.0, min(1.0, quality_score))
            
            metadata.update({
                "columns": df.columns.tolist(),
                "missing_columns": missing_columns,
                "duplicate_samples": duplicates if 'duplicates' in locals() else 0
            })
            
            return {
                "row_count": row_count,
                "column_count": column_count,
                "quality_score": quality_score,
                "issues": issues,
                "metadata": metadata
            }
            
        except Exception as e:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.CRITICAL,
                message=f"Failed to read metadata file: {e}",
                category="file_format",
                file_path=file_path
            ))
            
            return {
                "row_count": 0,
                "column_count": 0,
                "quality_score": 0.0,
                "issues": issues,
                "metadata": {}
            }
    
    async def _validate_archive_file(self, file_path: Path) -> Dict[str, Any]:
        """Validate archive files."""
        issues = []
        
        try:
            # Basic validation - check if file can be opened
            if file_path.suffix == '.gz':
                with gzip.open(file_path, 'rb') as f:
                    f.read(1024)  # Try to read first 1KB
            
            quality_score = 0.6  # Neutral score for archives
            
            return {
                "quality_score": quality_score,
                "issues": issues,
                "metadata": {"archive_type": file_path.suffix}
            }
            
        except Exception as e:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.ERROR,
                message=f"Archive file is corrupted: {e}",
                category="file_integrity",
                file_path=file_path
            ))
            
            return {
                "quality_score": 0.0,
                "issues": issues,
                "metadata": {}
            }
    
    def _calculate_matrix_quality_score(self, df: pd.DataFrame, issues: List[ValidationIssue]) -> float:
        """Calculate quality score for matrix files."""
        score = 1.0
        
        # Penalize for issues
        for issue in issues:
            if issue.severity == ValidationSeverity.CRITICAL:
                score -= 0.5
            elif issue.severity == ValidationSeverity.ERROR:
                score -= 0.3
            elif issue.severity == ValidationSeverity.WARNING:
                score -= 0.1
        
        # Bonus for good characteristics
        if df.shape[0] >= self.quality_thresholds["min_rows"]:
            score += 0.1
        if df.shape[1] >= self.quality_thresholds["min_columns"]:
            score += 0.1
        
        return max(0.0, min(1.0, score))
    
    def _validate_file_size(self, size_bytes: int) -> bool:
        """Validate file size against configuration."""
        min_size = self.validation_rules["min_file_size_bytes"]
        max_size = self.validation_rules["max_file_size_mb"] * 1024 * 1024
        return min_size <= size_bytes <= max_size
    
    def _categorize_results(self, results: List[FileValidationResult]) -> Dict[str, List[FileValidationResult]]:
        """Categorize validation results."""
        categorized = {}
        
        for result in results:
            category = result.category or "unknown"
            if category not in categorized:
                categorized[category] = []
            categorized[category].append(result)
        
        return categorized
    
    async def _validate_dataset_consistency(self, 
                                          results: List[FileValidationResult], 
                                          dataset_id: str) -> List[ValidationIssue]:
        """Validate consistency across the entire dataset."""
        issues = []
        
        # Check if we have required file types
        categories = set(r.category for r in results if r.category)
        required_categories = set(self.geo_config.get_required_categories())
        
        missing_required = required_categories - categories
        if missing_required:
            issues.append(ValidationIssue(
                severity=ValidationSeverity.WARNING,
                message=f"Missing required file categories: {missing_required}",
                category="dataset_completeness"
            ))
        
        # Check sample consistency across files
        matrix_files = [r for r in results if r.data_type in ["count_matrix", "expression_matrix"]]
        if len(matrix_files) > 1:
            sample_counts = [r.column_count for r in matrix_files if r.column_count]
            if len(set(sample_counts)) > 1:
                issues.append(ValidationIssue(
                    severity=ValidationSeverity.WARNING,
                    message=f"Inconsistent sample counts across files: {sample_counts}",
                    category="dataset_consistency"
                ))
        
        return issues
    
    def _calculate_overall_quality(self, results: List[FileValidationResult]) -> float:
        """Calculate overall dataset quality score."""
        if not results:
            return 0.0
        
        # Weight by file importance (category priority)
        weighted_scores = []
        for result in results:
            if result.category and result.category in self.file_categories:
                priority = self.file_categories[result.category]["priority"]
                weight = priority / 10.0  # Normalize to 0-1
            else:
                weight = 0.5  # Default weight
            
            weighted_scores.append(result.quality_score * weight)
        
        return sum(weighted_scores) / len(weighted_scores) if weighted_scores else 0.0
    
    def _generate_recommendations(self, 
                                results: List[FileValidationResult], 
                                dataset_issues: List[ValidationIssue]) -> List[str]:
        """Generate recommendations based on validation results."""
        recommendations = []
        
        # Check for missing high-priority files
        categories = set(r.category for r in results if r.category)
        high_priority_categories = [cat for cat, info in self.file_categories.items() 
                                  if info["priority"] >= 8]
        
        missing_high_priority = set(high_priority_categories) - categories
        if missing_high_priority:
            recommendations.append(f"Consider obtaining {missing_high_priority} files for better analysis")
        
        # Check for low-quality files
        low_quality_files = [r for r in results if r.quality_score < 0.5]
        if low_quality_files:
            recommendations.append(f"Review {len(low_quality_files)} low-quality files before analysis")
        
        # Dataset-specific recommendations
        if any(issue.severity == ValidationSeverity.CRITICAL for issue in dataset_issues):
            recommendations.append("Critical issues found - dataset may not be suitable for analysis")
        
        return recommendations
    
    def _assess_analysis_suitability(self, 
                                   results: List[FileValidationResult], 
                                   overall_quality: float) -> bool:
        """Assess if dataset is suitable for analysis."""
        # Must have at least one valid file
        valid_files = [r for r in results if r.is_valid]
        if not valid_files:
            return False
        
        # Overall quality must be above threshold
        if overall_quality < 0.3:
            return False
        
        # Must have at least one matrix file
        matrix_files = [r for r in results if r.data_type in ["count_matrix", "expression_matrix"]]
        if not matrix_files:
            return False
        
        # No critical issues
        critical_issues = [r for r in results for issue in r.issues 
                          if issue.severity == ValidationSeverity.CRITICAL]
        if critical_issues:
            return False
        
        return True
    
    def validate_input(self, dataset_id: str, files: List[Path]) -> bool:
        """Validate input parameters."""
        if not dataset_id:
            self.logger.error("Dataset ID is required")
            return False
        
        if not files:
            self.logger.error("No files provided for validation")
            return False
        
        return True
    
    def validate_output(self, result: DatasetValidationResult) -> bool:
        """Validate output result."""
        return isinstance(result, DatasetValidationResult) and result.dataset_id 