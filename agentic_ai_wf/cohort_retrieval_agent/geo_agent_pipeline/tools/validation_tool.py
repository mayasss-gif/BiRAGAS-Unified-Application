"""
Validation tool for the Cohort Retrieval Agent system.

This tool handles validation of downloaded files and datasets.
"""

import hashlib
import gzip
import zipfile
import re
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass

# Project Imports
from   ..base.base_tool import BaseTool, ToolResult
from   ..config import CohortRetrievalConfig
from   ..exceptions import ValidationError


@dataclass
class ValidationResult:
    """Result of a validation operation."""
    file_path: Path
    is_valid: bool
    file_size: int
    file_type: str
    checksum: Optional[str] = None
    error_message: Optional[str] = None
    warnings: List[str] = None


class ValidationTool(BaseTool[List[ValidationResult]]):
    """
    Tool for validating downloaded files and datasets.
    
    Validates:
    - File integrity (checksums, compression)
    - File formats
    - Content structure
    - Completeness
    """
    
    def __init__(self, config: CohortRetrievalConfig):
        super().__init__(config, "ValidationTool")
    
    async def execute(self, 
                     file_paths: List[Path], 
                     validation_types: Optional[List[str]] = None) -> ToolResult[List[ValidationResult]]:
        """
        Validate multiple files.
        
        Args:
            file_paths: List of file paths to validate
            validation_types: Optional list of validation types to perform
            
        Returns:
            ToolResult with validation results
        """
        if not self.validate_input(file_paths):
            return ToolResult(
                success=False,
                error="Invalid input parameters",
                details={"file_count": len(file_paths) if file_paths else 0}
            )
        
        validation_types = validation_types or ["integrity", "format", "content"]
        
        return await self.run_with_retry(self._validate_files, file_paths, validation_types)
    
    async def _validate_files(self, 
                             file_paths: List[Path], 
                             validation_types: List[str]) -> List[ValidationResult]:
        """Internal method to validate files."""
        results = []
        
        for file_path in file_paths:
            try:
                result = await self._validate_single_file(file_path, validation_types)
                results.append(result)
                
                if result.is_valid:
                    self.logger.debug(f"File validation passed: {file_path.name}")
                else:
                    self.logger.warning(f"File validation failed: {file_path.name} - {result.error_message}")
                    
            except Exception as e:
                self.logger.error(f"Error validating file {file_path}: {e}")
                results.append(ValidationResult(
                    file_path=file_path,
                    is_valid=False,
                    file_size=0,
                    file_type="unknown",
                    error_message=str(e)
                ))
        
        valid_count = sum(1 for r in results if r.is_valid)
        self.logger.info(f"Validated {len(results)} files: {valid_count} valid, {len(results) - valid_count} invalid")
        
        return results
    
    async def _validate_single_file(self, 
                                   file_path: Path, 
                                   validation_types: List[str]) -> ValidationResult:
        """Validate a single file."""
        if not file_path.exists():
            return ValidationResult(
                file_path=file_path,
                is_valid=False,
                file_size=0,
                file_type="missing",
                error_message="File does not exist"
            )
        
        file_size = file_path.stat().st_size
        file_type = self._detect_file_type(file_path)
        warnings = []
        
        # Basic integrity checks
        if "integrity" in validation_types:
            integrity_result = await self._validate_integrity(file_path, file_size)
            if not integrity_result["valid"]:
                return ValidationResult(
                    file_path=file_path,
                    is_valid=False,
                    file_size=file_size,
                    file_type=file_type,
                    error_message=integrity_result["error"]
                )
            warnings.extend(integrity_result.get("warnings", []))
        
        # Format validation
        if "format" in validation_types:
            format_result = await self._validate_format(file_path, file_type)
            if not format_result["valid"]:
                return ValidationResult(
                    file_path=file_path,
                    is_valid=False,
                    file_size=file_size,
                    file_type=file_type,
                    error_message=format_result["error"]
                )
            warnings.extend(format_result.get("warnings", []))
        
        # Content validation
        if "content" in validation_types:
            content_result = await self._validate_content(file_path, file_type)
            if not content_result["valid"]:
                return ValidationResult(
                    file_path=file_path,
                    is_valid=False,
                    file_size=file_size,
                    file_type=file_type,
                    error_message=content_result["error"]
                )
            warnings.extend(content_result.get("warnings", []))
        
        # Calculate checksum if requested
        checksum = None
        if "checksum" in validation_types:
            checksum = await self._calculate_checksum(file_path)
        
        return ValidationResult(
            file_path=file_path,
            is_valid=True,
            file_size=file_size,
            file_type=file_type,
            checksum=checksum,
            warnings=warnings if warnings else None
        )
    
    def _detect_file_type(self, file_path: Path) -> str:
        """Detect file type based on extension and content."""
        extension = file_path.suffix.lower()
        
        # Map extensions to types
        type_mapping = {
            '.gz': 'gzip',
            '.zip': 'zip',
            '.tar': 'tar',
            '.txt': 'text',
            '.csv': 'csv',
            '.tsv': 'tsv',
            '.json': 'json',
            '.xml': 'xml',
            '.xlsx': 'excel',
            '.h5': 'hdf5',
            '.rds': 'rds',
            '.rda': 'rdata'
        }
        
        return type_mapping.get(extension, 'unknown')
    
    async def _validate_integrity(self, file_path: Path, file_size: int) -> Dict[str, Any]:
        """Validate file integrity."""
        warnings = []
        
        # Check file size
        if file_size == 0:
            return {"valid": False, "error": "File is empty"}
        
        if file_size < 100:  # Suspiciously small
            warnings.append("File is very small (< 100 bytes)")
        
        # Try to read the file to check for corruption
        try:
            with open(file_path, 'rb') as f:
                # Read first and last chunks to check accessibility
                f.read(1024)  # First 1KB
                if file_size > 2048:
                    f.seek(-1024, 2)  # Last 1KB
                    f.read(1024)
        except Exception as e:
            return {"valid": False, "error": f"File appears corrupted: {e}"}
        
        return {"valid": True, "warnings": warnings}
    
    async def _validate_format(self, file_path: Path, file_type: str) -> Dict[str, Any]:
        """Validate file format."""
        warnings = []
        
        try:
            if file_type == 'gzip':
                # Test gzip decompression
                with gzip.open(file_path, 'rb') as f:
                    f.read(1024)  # Try to read first chunk
            
            elif file_type == 'zip':
                # Test zip file
                with zipfile.ZipFile(file_path, 'r') as zf:
                    zf.testzip()  # Test integrity
            
            elif file_type in ['csv', 'tsv']:
                # Basic CSV/TSV validation
                delimiter = ',' if file_type == 'csv' else '\t'
                with open(file_path, 'r', encoding='utf-8') as f:
                    first_line = f.readline()
                    if not first_line.strip():
                        warnings.append("File appears to be empty or has no header")
                    elif delimiter not in first_line:
                        warnings.append(f"Expected delimiter '{delimiter}' not found in first line")
            
            elif file_type == 'json':
                # Basic JSON validation
                import json
                with open(file_path, 'r', encoding='utf-8') as f:
                    json.load(f)  # This will raise an exception if invalid JSON
            
            elif file_type == 'xml':
                # Basic XML validation
                import xml.etree.ElementTree as ET
                ET.parse(file_path)
            
            # Add more format validations as needed
            
        except Exception as e:
            return {"valid": False, "error": f"Format validation failed: {e}"}
        
        return {"valid": True, "warnings": warnings}
    
    async def _validate_content(self, file_path: Path, file_type: str) -> Dict[str, Any]:
        """Validate file content structure."""
        warnings = []
        
        try:
            if file_type in ['csv', 'tsv']:
                # Check for reasonable content in CSV/TSV files
                line_count = 0
                column_count = None
                delimiter = ',' if file_type == 'csv' else '\t'
                
                with open(file_path, 'r', encoding='utf-8') as f:
                    for i, line in enumerate(f):
                        line_count += 1
                        if i < 10:  # Check first 10 lines for consistency
                            cols = len(line.split(delimiter))
                            if column_count is None:
                                column_count = cols
                            elif cols != column_count:
                                warnings.append(f"Inconsistent column count at line {i+1}")
                        
                        if i > 1000:  # Don't read entire large files
                            break
                
                if line_count < 2:
                    warnings.append("File has fewer than 2 lines (header + data)")
                
                if column_count and column_count < 2:
                    warnings.append("File has fewer than 2 columns")
            
            elif file_type == 'json':
                # Check JSON structure
                import json
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if not data:
                        warnings.append("JSON file is empty or null")
                    elif isinstance(data, dict) and len(data) == 0:
                        warnings.append("JSON object is empty")
                    elif isinstance(data, list) and len(data) == 0:
                        warnings.append("JSON array is empty")
            
            # Add more content validations as needed
            
        except Exception as e:
            return {"valid": False, "error": f"Content validation failed: {e}"}
        
        return {"valid": True, "warnings": warnings}
    
    async def _calculate_checksum(self, file_path: Path, algorithm: str = "md5") -> str:
        """Calculate file checksum."""
        hash_func = hashlib.new(algorithm)
        
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hash_func.update(chunk)
        
        return hash_func.hexdigest()
    
    async def validate_dataset_completeness(self, 
                                          dataset_dir: Path, 
                                          expected_files: Optional[List[str]] = None) -> ToolResult[Dict[str, Any]]:
        """
        Validate that a dataset directory contains expected files.
        
        Args:
            dataset_dir: Directory containing dataset files
            expected_files: Optional list of expected file patterns
            
        Returns:
            ToolResult with completeness validation
        """
        try:
            if not dataset_dir.exists():
                return ToolResult(
                    success=False,
                    error=f"Dataset directory does not exist: {dataset_dir}"
                )
            
            result = {
                "dataset_dir": str(dataset_dir),
                "total_files": 0,
                "expected_files_found": 0,
                "missing_files": [],
                "unexpected_files": [],
                "is_complete": False
            }
            
            # Get all files in directory
            all_files = list(dataset_dir.rglob('*'))
            result["total_files"] = len([f for f in all_files if f.is_file()])
            
            if expected_files:
                # Check for expected files
                found_files = set()
                for file_path in all_files:
                    if file_path.is_file():
                        found_files.add(file_path.name)
                
                for expected in expected_files:
                    if any(expected in filename for filename in found_files):
                        result["expected_files_found"] += 1
                    else:
                        result["missing_files"].append(expected)
                
                result["is_complete"] = len(result["missing_files"]) == 0
            else:
                # Basic completeness check - look for common file types
                essential_patterns = ["matrix", "count", "metadata", "sample", "raw", "counts", "expression", "counts.txt", "counts.tsv", "counts.csv", "counts.mtx", "counts.mtx.gz", "counts.mtx.zip", "counts.mtx.tar", "counts.mtx.tar.gz", "counts.mtx.tar.zip", "counts.mtx.tar.gz.zip", "counts.mtx.tar.gz.tar", "counts.mtx.tar.gz.tar.gz", "counts.mtx.tar.gz.tar.gz.zip"]
                found_essential = False
                
                for file_path in all_files:
                    if file_path.is_file():
                        filename = file_path.name.lower()
                        if any(pattern in filename for pattern in essential_patterns):
                            found_essential = True
                            break
                
                result["is_complete"] = found_essential and result["total_files"] > 0
            
            return ToolResult(success=True, data=result)
            
        except Exception as e:
            return ToolResult(
                success=False,
                error=f"Dataset completeness validation failed: {e}",
                details={"dataset_dir": str(dataset_dir)}
            )
    
    def validate_input(self, file_paths: List[Path]) -> bool:
        """Validate input parameters."""
        if not isinstance(file_paths, list):
            self.logger.error("file_paths must be a list")
            return False
        
        if not file_paths:
            self.logger.error("file_paths list cannot be empty")
            return False
        
        for file_path in file_paths:
            if not isinstance(file_path, Path):
                self.logger.error("All file_paths must be Path objects")
                return False
        
        return True
    
    def validate_output(self, result: List[ValidationResult]) -> bool:

        """Validate output result."""
        if not isinstance(result, list):
            self.logger.error("Result must be a list")
            return False
        
        for validation_result in result:
            if not isinstance(validation_result, ValidationResult):
                self.logger.error("All result items must be ValidationResult objects")
                return False
        
        return True 
    
    