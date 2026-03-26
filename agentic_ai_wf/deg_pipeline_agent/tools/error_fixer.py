"""
Error fixer tool for DEG Pipeline Agent.
"""

import os
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional

from .base_tool import BaseTool
from ..exceptions import ValidationError


class ErrorFixerTool(BaseTool):
    """Tool for automatic error detection and fixing."""
    
    @property
    def name(self) -> str:
        return "ErrorFixer"
    
    @property
    def description(self) -> str:
        return "Automatically detect and fix common pipeline errors"
    
    def execute(self, error_type: str, error_message: str, context: Dict[str, Any], **kwargs) -> Dict:
        """
        Attempt to fix an error automatically.
        
        Args:
            error_type: Type of error (e.g., "pipeline_error", "dataset_error")
            error_message: Error message
            context: Error context information
            **kwargs: Additional parameters
            
        Returns:
            Fix result
        """
        self.logger.info(f"🔧 Attempting to fix {error_type}: {error_message}")
        
        fix_result = {
            "error_type": error_type,
            "error_message": error_message,
            "fix_attempted": False,
            "fix_successful": False,
            "fix_actions": [],
            "recommendations": []
        }
        
        try:
            if error_type == "pipeline_error":
                fix_result = self._fix_pipeline_error(error_message, context, fix_result)
            elif error_type == "dataset_error":
                fix_result = self._fix_dataset_error(error_message, context, fix_result)
            else:
                fix_result = self._fix_generic_error(error_message, context, fix_result)
            
            return fix_result
            
        except Exception as e:
            self.logger.error(f"❌ Error fixing failed: {e}")
            fix_result["fix_error"] = str(e)
            return fix_result
    
    def _fix_pipeline_error(self, error_message: str, context: Dict[str, Any], fix_result: Dict) -> Dict:
        """
        Fix pipeline-level errors.
        
        Args:
            error_message: Error message
            context: Pipeline context
            fix_result: Fix result dictionary
            
        Returns:
            Updated fix result
        """
        error_lower = error_message.lower()
        
        # Directory permission errors
        if "permission" in error_lower or "access denied" in error_lower:
            return self._fix_permission_error(context, fix_result)
        
        # Disk space errors
        if "no space" in error_lower or "disk full" in error_lower:
            return self._fix_disk_space_error(context, fix_result)
        
        # Memory errors
        if "memory" in error_lower or "out of memory" in error_lower:
            return self._fix_memory_error(context, fix_result)
        
        # No datasets found
        if "no datasets" in error_lower or "no valid dataset" in error_lower:
            return self._fix_no_datasets_error(context, fix_result)
        
        # Configuration errors
        if "configuration" in error_lower or "config" in error_lower:
            return self._fix_configuration_error(context, fix_result)
        
        # Add general recommendations
        fix_result["recommendations"].extend([
            "Check log files for detailed error information",
            "Verify input directories exist and are accessible",
            "Ensure sufficient system resources (memory, disk space)",
            "Check file permissions"
        ])
        
        return fix_result
    
    def _fix_dataset_error(self, error_message: str, context: Dict[str, Any], fix_result: Dict) -> Dict:
        """
        Fix dataset-specific errors.
        
        Args:
            error_message: Error message
            context: Dataset context
            fix_result: Fix result dictionary
            
        Returns:
            Updated fix result
        """
        error_lower = error_message.lower()
        
        # File not found errors
        if "file not found" in error_lower or "does not exist" in error_lower:
            return self._fix_file_not_found_error(context, fix_result)
        
        # File format errors
        if "format" in error_lower or "separator" in error_lower or "delimiter" in error_lower:
            return self._fix_file_format_error(context, fix_result)
        
        # Empty file errors
        if "empty" in error_lower:
            return self._fix_empty_file_error(context, fix_result)
        
        # Sample mismatch errors
        if "sample" in error_lower and ("mismatch" in error_lower or "alignment" in error_lower):
            return self._fix_sample_mismatch_error(context, fix_result)
        
        # Metadata extraction errors
        if "metadata" in error_lower:
            return self._fix_metadata_error(context, fix_result)
        
        # Analysis errors
        if "deseq" in error_lower or "analysis" in error_lower:
            return self._fix_analysis_error(context, fix_result)
        
        return fix_result
    
    def _fix_permission_error(self, context: Dict[str, Any], fix_result: Dict) -> Dict:
        """Fix permission-related errors."""
        fix_result["fix_attempted"] = True
        
        try:
            # Try to create directories with proper permissions
            if "output_dir" in context:
                output_dir = Path(context["output_dir"])
                output_dir.mkdir(parents=True, exist_ok=True, mode=0o755)
                fix_result["fix_actions"].append(f"Created output directory: {output_dir}")
                fix_result["fix_successful"] = True
        except Exception as e:
            fix_result["fix_actions"].append(f"Failed to fix permissions: {e}")
        
        fix_result["recommendations"].extend([
            "Check directory permissions",
            "Ensure write access to output directories",
            "Run with appropriate user permissions"
        ])
        
        return fix_result
    
    def _fix_disk_space_error(self, context: Dict[str, Any], fix_result: Dict) -> Dict:
        """Fix disk space errors."""
        fix_result["fix_attempted"] = True
        
        try:
            # Clean up temporary files
            if "work_dir" in context:
                work_dir = Path(context["work_dir"])
                temp_files = list(work_dir.glob("*.tmp"))
                for temp_file in temp_files:
                    temp_file.unlink()
                    fix_result["fix_actions"].append(f"Removed temp file: {temp_file}")
                
                if temp_files:
                    fix_result["fix_successful"] = True
        except Exception as e:
            fix_result["fix_actions"].append(f"Failed to clean temp files: {e}")
        
        fix_result["recommendations"].extend([
            "Free up disk space",
            "Clean temporary files",
            "Use a different output directory with more space"
        ])
        
        return fix_result
    
    def _fix_memory_error(self, context: Dict[str, Any], fix_result: Dict) -> Dict:
        """Fix memory-related errors."""
        fix_result["fix_attempted"] = True
        
        # Suggest memory optimization
        fix_result["recommendations"].extend([
            "Reduce max_genes parameter in configuration",
            "Process datasets individually instead of in batch",
            "Increase system memory or use a machine with more RAM",
            "Filter genes more aggressively before analysis"
        ])
        
        return fix_result
    
    def _fix_no_datasets_error(self, context: Dict[str, Any], fix_result: Dict) -> Dict:
        """Fix no datasets found errors."""
        fix_result["fix_attempted"] = True
        
        fix_result["recommendations"].extend([
            "Check that input directories contain data files",
            "Verify file naming conventions (should contain 'count' or 'metadata')",
            "Ensure files have supported extensions (.csv, .tsv, .txt)",
            "Check for compressed files that need decompression"
        ])
        
        return fix_result
    
    def _fix_file_not_found_error(self, context: Dict[str, Any], fix_result: Dict) -> Dict:
        """Fix file not found errors."""
        fix_result["fix_attempted"] = True
        
        try:
            # Try to find similar files
            if "counts_file" in context:
                counts_file = Path(context["counts_file"])
                if not counts_file.exists():
                    # Look for files with similar names
                    parent_dir = counts_file.parent
                    if parent_dir.exists():
                        similar_files = list(parent_dir.glob(f"*{counts_file.stem}*"))
                        if similar_files:
                            fix_result["fix_actions"].append(f"Found similar files: {[str(f) for f in similar_files]}")
                            fix_result["recommendations"].append("Check if file path is correct")
        except Exception as e:
            fix_result["fix_actions"].append(f"Failed to search for similar files: {e}")
        
        fix_result["recommendations"].extend([
            "Verify file paths are correct",
            "Check if files have been moved or renamed",
            "Ensure files are in the expected directory structure"
        ])
        
        return fix_result
    
    def _fix_file_format_error(self, context: Dict[str, Any], fix_result: Dict) -> Dict:
        """Fix file format errors."""
        fix_result["fix_attempted"] = True
        
        fix_result["recommendations"].extend([
            "Check file format (CSV vs TSV)",
            "Verify column headers are correct",
            "Ensure files use standard delimiters (comma or tab)",
            "Check for special characters or encoding issues"
        ])
        
        return fix_result
    
    def _fix_empty_file_error(self, context: Dict[str, Any], fix_result: Dict) -> Dict:
        """Fix empty file errors."""
        fix_result["fix_attempted"] = True
        
        try:
            # Check if files are actually empty or just appear empty
            files_to_check = []
            if "counts_file" in context:
                files_to_check.append(context["counts_file"])
            if "metadata_file" in context:
                files_to_check.append(context["metadata_file"])
            
            for file_path in files_to_check:
                file_path = Path(file_path)
                if file_path.exists():
                    size = file_path.stat().st_size
                    fix_result["fix_actions"].append(f"File {file_path.name}: {size} bytes")
        except Exception as e:
            fix_result["fix_actions"].append(f"Failed to check file sizes: {e}")
        
        fix_result["recommendations"].extend([
            "Verify files contain data",
            "Check if files were properly downloaded/transferred",
            "Ensure files are not corrupted",
            "Try re-downloading or re-generating the files"
        ])
        
        return fix_result
    
    def _fix_sample_mismatch_error(self, context: Dict[str, Any], fix_result: Dict) -> Dict:
        """Fix sample mismatch errors."""
        fix_result["fix_attempted"] = True
        
        fix_result["recommendations"].extend([
            "Check sample IDs in counts and metadata files match",
            "Verify sample naming conventions",
            "Ensure no extra spaces or special characters in sample names",
            "Check if samples are in the same order"
        ])
        
        return fix_result
    
    def _fix_metadata_error(self, context: Dict[str, Any], fix_result: Dict) -> Dict:
        """Fix metadata extraction errors."""
        fix_result["fix_attempted"] = True
        
        fix_result["recommendations"].extend([
            "Check metadata file format",
            "Verify required columns exist (sample, condition)",
            "Ensure at least 2 different conditions are present",
            "Check for missing or invalid condition values"
        ])
        
        return fix_result
    
    def _fix_analysis_error(self, context: Dict[str, Any], fix_result: Dict) -> Dict:
        """Fix analysis errors."""
        fix_result["fix_attempted"] = True
        
        fix_result["recommendations"].extend([
            "Check that PyDESeq2 is properly installed",
            "Verify input data has sufficient samples per condition",
            "Ensure count data is properly formatted (non-negative integers)",
            "Check for genes with all zero counts"
        ])
        
        return fix_result
    
    def _fix_configuration_error(self, context: Dict[str, Any], fix_result: Dict) -> Dict:
        """Fix configuration errors."""
        fix_result["fix_attempted"] = True
        
        try:
            # Check common configuration issues
            if "disease_name" not in context or not context.get("disease_name"):
                fix_result["fix_actions"].append("Missing disease_name in configuration")
            
            if "geo_dir" not in context and "patient_dir" not in context:
                fix_result["fix_actions"].append("No input directories specified")
        except Exception as e:
            fix_result["fix_actions"].append(f"Failed to check configuration: {e}")
        
        fix_result["recommendations"].extend([
            "Verify all required configuration parameters are set",
            "Check that directories exist and are accessible",
            "Ensure disease_name is specified",
            "Validate configuration parameters are within valid ranges"
        ])
        
        return fix_result
    
    def _fix_generic_error(self, error_message: str, context: Dict[str, Any], fix_result: Dict) -> Dict:
        """Fix generic/unknown errors."""
        fix_result["fix_attempted"] = True
        
        fix_result["recommendations"].extend([
            "Check log files for detailed error information",
            "Verify all dependencies are installed",
            "Ensure input data is valid and accessible",
            "Try running with verbose logging enabled",
            "Contact support if the error persists"
        ])
        
        return fix_result
    
    def validate_input(self, error_type: str, error_message: str, context: Dict[str, Any], **kwargs) -> None:
        """
        Validate input parameters.
        
        Args:
            error_type: Type of error
            error_message: Error message
            context: Error context
            **kwargs: Additional parameters
        """
        if not error_type:
            raise ValidationError("Error type cannot be empty")
        
        if not error_message:
            raise ValidationError("Error message cannot be empty")
        
        if context is None:
            raise ValidationError("Context cannot be None")
    
    def validate_output(self, result: Dict) -> Dict:
        """
        Validate output result.
        
        Args:
            result: Output result
            
        Returns:
            Validated result
        """
        required_fields = ["error_type", "error_message", "fix_attempted", "fix_successful", "fix_actions", "recommendations"]
        
        for field in required_fields:
            if field not in result:
                raise ValidationError(f"Missing required field in result: {field}")
        
        return result 