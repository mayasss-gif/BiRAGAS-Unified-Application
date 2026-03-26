"""
Dataset detector tool for DEG Pipeline Agent.
"""

import os
import gzip
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple

from .base_tool import BaseTool
from .tar_processor import process_tar_counts
from ..exceptions import DataLoadError, ValidationError, FileSystemError


class DatasetDetectorTool(BaseTool):
    """Tool for detecting and pairing count/metadata files."""
    
    @property
    def name(self) -> str:
        return "DatasetDetector"
    
    @property
    def description(self) -> str:
        return "Detect and pair count/metadata files in directories"
    
    def execute(self, root_dir: Union[str, Path], disease_name: str, **kwargs) -> Dict:
        """
        Detect dataset pairs in a directory.
        
        Args:
            root_dir: Root directory to scan
            disease_name: Disease name for organizing outputs
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with status and list of dataset pairs
            
        Raises:
            DataLoadError: If detection fails
        """
        root_dir = Path(root_dir).expanduser().resolve()
        
        if not root_dir.exists():
            raise DataLoadError(f"Directory does not exist: {root_dir}")
        
        self.logger.info(f"📁 Scanning directory: {root_dir}")
        
        dataset_pairs = []
        processed_dirs = 0
        
        # Walk through all subdirectories
        for current_dir in self._walk_directories(root_dir):
            processed_dirs += 1
            try:
                # Decompress any .gz files (silent unless errors)
                self._decompress_files(current_dir)
                
                # Find count/metadata pairs
                pair = self._find_dataset_pair(current_dir, disease_name)
                if pair:
                    dataset_pairs.append(pair)
                    
            except Exception as e:
                self.logger.warning(f"⚠️  Error processing directory {current_dir}: {e}")
                continue
        
        self.logger.info(f"📦 Found {len(dataset_pairs)} dataset pairs from {processed_dirs} directories")
        
        return {
            'status': 'success',
            'pairs': dataset_pairs,
            'total_pairs': len(dataset_pairs)
        }
    
    def _walk_directories(self, root_dir: Path) -> List[Path]:
        """
        Walk through directory structure.
        
        Args:
            root_dir: Root directory to walk
            
        Returns:
            List of directories to process
        """
        directories = []
        
        for root, dirs, files in os.walk(root_dir):
            # Skip hidden directories
            dirs[:] = [d for d in dirs if not d.startswith('.')]
            
            # Only process directories that contain files
            if files:
                directories.append(Path(root))
        
        return directories
    
    def _decompress_files(self, directory: Path) -> None:
        """
        Decompress .gz files in a directory.
        
        Args:
            directory: Directory containing files to decompress
        """
        gz_files = list(directory.glob("*.gz"))
        
        if not gz_files:
            return
        
        for gz_file in gz_files:
            output_file = gz_file.with_suffix('')
            
            # Skip if already decompressed
            if output_file.exists():
                continue
            
            try:
                self.logger.debug(f"🗜️  Decompressing {gz_file.name}")
                with gzip.open(gz_file, 'rt') as f_in:
                    with open(output_file, 'w') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                        
            except Exception as e:
                self.logger.warning(f"⚠️  Failed to decompress {gz_file.name}: {e}")
    
    def _find_dataset_pair(self, directory: Path, disease_name: str) -> Optional[Dict]:
        """
        Find count/metadata pair in a directory.
        
        Args:
            directory: Directory to search
            disease_name: Disease name for output organization
            
        Returns:
            Dataset pair dictionary or None if no valid pair found
        """
        files = list(directory.glob("*"))
        file_names = [f.name for f in files if f.is_file()]
        
        # Only log if we have files to process
        if not file_names:
            return None
        
        # Check for tar files first (e.g., GSE97263_RAW.tar)
        tar_file = self._find_tar_file(files)
        
        # If tar file exists, process it first to create prep/prep_counts.csv
        if tar_file:
            self.logger.info(f"📦 Found tar file: {tar_file.name}")
            
            # Generate sample name from directory (for GSE, use directory name)
            sample_name = self._generate_sample_name_from_dir(directory)
            
            # Create work directory (for GSE, this will be shared_gse/GSE_ID)
            work_dir = self._create_work_directory(sample_name)
            prep_dir = work_dir / "prep"
            prep_dir.mkdir(parents=True, exist_ok=True)
            
            # Check if prep_counts.csv already exists (already processed)
            if (prep_dir / "prep_counts.csv").exists():
                self.logger.info(f"⏭️  Tar file already processed, using existing prep_counts.csv")
            else:
                # Process tar file to create prep_counts.csv
                self.logger.info(f"🔄 Processing tar file to extract counts...")
                counts_path = process_tar_counts(
                    tar_path=tar_file,
                    output_dir=prep_dir,
                    output_name="prep_counts.csv",
                    verbose=True
                )
                
                if not counts_path or not counts_path.exists():
                    self.logger.error(f"❌ Failed to extract counts from tar file: {tar_file}")
                    return None
            
            # Now find metadata file
            metadata_file = self._find_metadata_file(files)
            
            if not metadata_file:
                self.logger.warning(f"⚠️  No metadata file found for {sample_name}. Will need to extract metadata.")
                # Still return pair, metadata extraction will handle it
                metadata_file = None
            
            pair = {
                'counts_file': str(prep_dir / "prep_counts.csv"),
                'metadata_file': str(metadata_file) if metadata_file else None,
                'sample_name': sample_name,
                'work_dir': str(work_dir),
                'disease_name': disease_name,
                'source_directory': str(directory),
                'from_tar': True  # Flag to indicate this came from tar
            }
            
            self.logger.info(f"✅ Found dataset from tar: {sample_name}")
            return pair
        
        # Normal processing: Find count and metadata files
        counts_file = self._find_counts_file(files)
        metadata_file = self._find_metadata_file(files)
        
        if not counts_file or not metadata_file:
            return None
        
        if counts_file.resolve() == metadata_file.resolve():
            return None
        
        # Generate unique sample name from directory and files
        sample_name = self._generate_sample_name(directory, counts_file, metadata_file)
        
        # Create work directory with unique sample name
        work_dir = self._create_work_directory(sample_name)
        
        pair = {
            'counts_file': str(counts_file),
            'metadata_file': str(metadata_file),
            'sample_name': sample_name,
            'work_dir': str(work_dir),
            'disease_name': disease_name,
            'source_directory': str(directory)
        }
        
        self.logger.info(f"✅ Found dataset: {sample_name}")
        return pair
    
    def _generate_sample_name(self, directory: Path, counts_file: Path, metadata_file: Path) -> str:
        """
        Generate a unique sample name from directory and file information.
        
        Args:
            directory: Source directory
            counts_file: Counts file path
            metadata_file: Metadata file path
            
        Returns:
            Unique sample name
        """
        # Use directory name as base
        base_name = directory.name
        
        # If directory name is generic, use file names
        if base_name.lower() in ['data', 'files', 'dataset', 'samples']:
            # Try to extract meaningful name from file names
            counts_stem = counts_file.stem.split('_')[0]
            meta_stem = metadata_file.stem.split('_')[0]
            
            # Use the common part or the shorter name
            if counts_stem.lower() in meta_stem.lower():
                base_name = counts_stem
            elif meta_stem.lower() in counts_stem.lower():
                base_name = meta_stem
            else:
                base_name = f"{counts_stem}_{meta_stem}"
        
        # Sanitize the name
        import re
        sanitized = re.sub(r'[<>:"/\\|?*]', '_', base_name)
        sanitized = re.sub(r'_+', '_', sanitized)
        sanitized = sanitized.strip('_')
        
        # Ensure it's not empty
        if not sanitized:
            sanitized = f"sample_{directory.parts[-1]}"
        
        return sanitized
    
    def _create_work_directory(self, sample_name: str) -> Path:
        """
        Create work directory for a sample.
        
        Args:
            sample_name: Unique sample name
            
        Returns:
            Path to created work directory
            
        Raises:
            FileSystemError: If directory creation fails
        """
        try:
            # # Check if this is a GSE directory that should use shared location
            # should_use_shared, shared_path = self.config.should_use_shared_gse(sample_name)
            
            # if should_use_shared:
            #     # Create shared GSE directory
            #     shared_path.mkdir(parents=True, exist_ok=True)
            #     self.logger.info(f"📁 Using shared GSE directory: {shared_path}")
            #     return shared_path
            # else:
                # # Create work directory in analysis-specific location
                # base_output_dir = self.config._get_output_dir()
                # work_dir = base_output_dir / sample_name
                # work_dir.mkdir(parents=True, exist_ok=True)
                # return work_dir

            # Create work directory in analysis-specific location
            base_output_dir = self.config._get_output_dir()
            work_dir = base_output_dir / sample_name
            work_dir.mkdir(parents=True, exist_ok=True)
            return work_dir
            
        except Exception as e:
            raise FileSystemError(f"Failed to create work directory: {e}")
    
    def _find_counts_file(self, files: List[Path]) -> Optional[Path]:
        """
        Find the counts file in a list of files.
        
        Supports: CSV, TSV, TXT, XLSX, XLS formats
        
        Args:
            files: List of files to search
            
        Returns:
            Path to counts file or None if not found
        """
        # Enhanced counts indicators including matrix patterns
        counts_indicators = [
            'count', 'expression', 'fpkm', 'tpm', 'rpkm',
            'matrix', 'gene_count', 'count_matrix', 'expression_matrix',
            'gene_expression', 'read_count', 'read_count_matrix'
        ]
        
        # Priority 1: Files with explicit count indicators
        for file_path in files:
            if not file_path.is_file():
                continue
            
            file_name = file_path.name.lower()
            file_suffix = file_path.suffix.lower()
            
            # Check if file has appropriate extension (including Excel)
            if file_suffix not in self.config.supported_formats:
                continue
            
            # Check if filename contains count indicators
            if any(indicator in file_name for indicator in counts_indicators):
                # Exclude metadata files even if they match indicators
                if not self._is_metadata_file(file_path):
                    self.logger.debug(f"Found counts file by indicator: {file_path.name}")
                    return file_path
        
        # Priority 2: Excel files that are likely counts (not metadata)
        excel_files = [
            f for f in files 
            if f.is_file() and f.suffix.lower() in ['.xlsx', '.xls']
            and f.suffix.lower() in self.config.supported_formats
            and not self._is_metadata_file(f)
        ]
        
        # Prefer Excel files with "gene" or "all" in name (common pattern)
        for excel_file in excel_files:
            file_name_lower = excel_file.name.lower()
            if 'gene' in file_name_lower or 'all' in file_name_lower:
                if 'metadata' not in file_name_lower and 'meta' not in file_name_lower:
                    self.logger.debug(f"Found Excel counts file: {excel_file.name}")
                    return excel_file
        
        # Priority 3: Generic data files (non-metadata, supported formats)
        for file_path in files:
            if not file_path.is_file():
                continue
            
            file_suffix = file_path.suffix.lower()
            
            if file_suffix in self.config.supported_formats:
                # Check if it's not a metadata file
                if not self._is_metadata_file(file_path):
                    self.logger.debug(f"Found generic counts file: {file_path.name}")
                    return file_path
        
        return None
    
    def _find_metadata_file(self, files: List[Path]) -> Optional[Path]:
        """
        Find the metadata file in a list of files.
        
        Args:
            files: List of files to search
            
        Returns:
            Path to metadata file or None if not found
        """
        metadata_indicators = [
            'metadata', 'meta', 'series_matrix', 'phenotype', 'pheno',
            'sample_info', 'clinical', 'annotation', 'design'
        ]
        
        for file_path in files:
            if not file_path.is_file():
                continue
            
            file_name = file_path.name.lower()
            file_suffix = file_path.suffix.lower()
            
            # Check if file has appropriate extension
            if file_suffix not in self.config.supported_formats:
                continue
            
            # Check if filename contains metadata indicators
            if any(indicator in file_name for indicator in metadata_indicators):
                return file_path
        
        return None
    
    def _is_metadata_file(self, file_path: Path) -> bool:
        """
        Check if a file is likely a metadata file.
        
        Args:
            file_path: Path to check
            
        Returns:
            True if likely metadata file
        """
        metadata_indicators = [
            'metadata', 'meta', 'series_matrix', 'phenotype', 'pheno',
            'sample_info', 'clinical', 'annotation', 'design'
        ]
        
        file_name = file_path.name.lower()
        return any(indicator in file_name for indicator in metadata_indicators)
    
    def _find_tar_file(self, files: List[Path]) -> Optional[Path]:
        """
        Find tar file in a list of files.
        
        Args:
            files: List of files to search
            
        Returns:
            Path to tar file or None if not found
        """
        tar_extensions = ['.tar', '.tar.gz', '.tgz']
        
        for file_path in files:
            if not file_path.is_file():
                continue
            
            file_name_lower = file_path.name.lower()
            # Check for tar extensions
            if any(file_name_lower.endswith(ext) for ext in tar_extensions):
                return file_path
            
            # Also check if filename contains "raw" or "tar" (common in GEO datasets)
            if 'raw' in file_name_lower and file_path.suffix.lower() in ['.tar', '.gz']:
                return file_path
        
        return None
    
    def _generate_sample_name_from_dir(self, directory: Path) -> str:
        """
        Generate sample name from directory name (used for tar file processing).
        
        Args:
            directory: Source directory
            
        Returns:
            Sample name (typically the directory name, e.g., GSE1234)
        """
        base_name = directory.name
        
        # Sanitize the name
        import re
        sanitized = re.sub(r'[<>:"/\\|?*]', '_', base_name)
        sanitized = re.sub(r'_+', '_', sanitized)
        sanitized = sanitized.strip('_')
        
        # Ensure it's not empty
        if not sanitized:
            sanitized = f"sample_{directory.parts[-1]}"
        
        return sanitized
    
    
    def validate_input(self, root_dir: Union[str, Path], disease_name: str = 'unknown_disease', **kwargs) -> None:
        """
        Validate input parameters.
        
        Args:
            root_dir: Root directory to scan
            disease_name: Disease name
            **kwargs: Additional parameters
            
        Raises:
            ValidationError: If validation fails
        """
        if not root_dir:
            raise ValidationError("Root directory cannot be empty")
        
        root_dir = Path(root_dir)
        
        if not root_dir.exists():
            raise ValidationError(f"Root directory does not exist: {root_dir}")
        
        if not root_dir.is_dir():
            raise ValidationError(f"Root path is not a directory: {root_dir}")
    
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
        if not isinstance(result, dict):
            raise ValidationError("Output must be a dictionary")
        
        if 'status' not in result:
            raise ValidationError("Output must contain 'status' field")
        
        if 'pairs' not in result:
            raise ValidationError("Output must contain 'pairs' field")
        
        if not isinstance(result['pairs'], list):
            raise ValidationError("'pairs' must be a list")
        
        # Validate each pair
        for i, pair in enumerate(result['pairs']):
            if not isinstance(pair, dict):
                raise ValidationError(f"Pair {i} must be a dictionary")
            
            required_fields = ['counts_file', 'metadata_file', 'sample_name', 'work_dir', 'disease_name']
            for field in required_fields:
                if field not in pair:
                    raise ValidationError(f"Pair {i} missing required field: {field}")
        
        return result
    
    def get_detection_summary(self, result: Dict) -> Dict:
        """
        Get a summary of detection results.
        
        Args:
            result: Detection result
            
        Returns:
            Summary dictionary
        """
        if not result or 'pairs' not in result:
            return {"error": "Invalid result"}
        
        pairs = result['pairs']
        
        # Count by source type
        geo_count = sum(1 for pair in pairs if 'series_matrix' in pair.get('metadata_file', '').lower())
        patient_count = len(pairs) - geo_count
        
        # Count by disease
        disease_counts = {}
        for pair in pairs:
            disease = pair.get('disease_name', 'unknown')
            disease_counts[disease] = disease_counts.get(disease, 0) + 1
        
        return {
            "total_pairs": len(pairs),
            "geo_datasets": geo_count,
            "patient_datasets": patient_count,
            "disease_counts": disease_counts,
            "sample_names": [pair.get('sample_name', 'unknown') for pair in pairs]
        } 