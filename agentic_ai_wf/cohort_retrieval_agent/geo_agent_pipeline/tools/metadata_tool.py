"""
Metadata tool for the Cohort Retrieval Agent system.

This tool handles extracting and processing metadata from various data sources.
"""

import json
import gzip
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

# Project Imports
from   ..base.base_tool import BaseTool, ToolResult
from   ..config import CohortRetrievalConfig
from   ..exceptions import MetadataError


@dataclass
class SampleMetadata:
    """Metadata for a single sample."""
    sample_id: str
    characteristics: Dict[str, Any]
    library_strategy: str
    library_source: str
    organism: str
    tissue_type: str
    experiment_type: str
    additional_info: Dict[str, Any]


class MetadataTool(BaseTool[Dict[str, Any]]):
    """
    Tool for extracting metadata from datasets.
    
    Handles:
    - GEO series matrix files
    - Sample metadata files
    - Supplementary file parsing
    - Metadata validation and normalization
    """
    
    def __init__(self, config: CohortRetrievalConfig):
        super().__init__(config, "MetadataTool")
    
    async def execute(self, 
                     source_file: Path, 
                     source_type: str = "geo") -> ToolResult[Dict[str, Any]]:
        """
        Extract metadata from a source file.
        
        Args:
            source_file: Path to the metadata source file
            source_type: Type of source (geo, sra, etc.)
            
        Returns:
            ToolResult with extracted metadata
        """
        if not self.validate_input(source_file, source_type):
            return ToolResult(
                success=False,
                error="Invalid input parameters",
                details={"source_file": str(source_file), "source_type": source_type}
            )
        
        return await self.run_with_retry(self._extract_metadata, source_file, source_type)
    
    async def _extract_metadata(self, source_file: Path, source_type: str) -> Dict[str, Any]:
        """Internal method to extract metadata."""
        if source_type == "geo":
            return await self._extract_geo_metadata(source_file)
        elif source_type == "sra":
            return await self._extract_sra_metadata(source_file)
        else:
            raise MetadataError(f"Unsupported source type: {source_type}")
    
    async def _extract_geo_metadata(self, source_file: Path) -> Dict[str, Any]:
        """Extract metadata from GEO series matrix file."""
        try:
            # Handle compressed files
            if source_file.suffix == '.gz':
                with gzip.open(source_file, 'rt', encoding='utf-8') as f:
                    content = f.read()
            else:
                with open(source_file, 'r', encoding='utf-8') as f:
                    content = f.read()
            
            metadata = {
                "source": "geo",
                "file_path": str(source_file),
                "series_info": {},
                "samples": [],
                "platform_info": {},
                "processing_info": {}
            }
            
            lines = content.split('\n')
            current_section = None
            
            for line in lines:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                if line.startswith('!Series_'):
                    key = line.split('\t')[0].replace('!Series_', '')
                    value = '\t'.join(line.split('\t')[1:])
                    metadata["series_info"][key] = value
                
                elif line.startswith('!Sample_'):
                    # Parse sample information
                    parts = line.split('\t')
                    if len(parts) >= 2:
                        key = parts[0].replace('!Sample_', '')
                        values = parts[1:]
                        
                        # Handle sample characteristics
                        if key == 'characteristics_ch1':
                            self._parse_sample_characteristics(metadata, values)
                        elif key == 'geo_accession':
                            for i, sample_id in enumerate(values):
                                if i >= len(metadata["samples"]):
                                    metadata["samples"].append({"sample_id": sample_id})
                                else:
                                    metadata["samples"][i]["sample_id"] = sample_id
                        else:
                            for i, value in enumerate(values):
                                if i >= len(metadata["samples"]):
                                    metadata["samples"].append({})
                                metadata["samples"][i][key] = value
                
                elif line.startswith('!Platform_'):
                    key = line.split('\t')[0].replace('!Platform_', '')
                    value = '\t'.join(line.split('\t')[1:])
                    metadata["platform_info"][key] = value
            
            # Post-process and validate samples
            metadata["samples"] = self._validate_and_clean_samples(metadata["samples"])
            
            return metadata
            
        except Exception as e:
            raise MetadataError(f"Failed to extract GEO metadata: {e}", dataset_id=source_file.stem)
    
    async def _extract_sra_metadata(self, source_file: Path) -> Dict[str, Any]:
        """Extract metadata from SRA files (placeholder)."""
        # This would be implemented when SRA agent is added
        return {
            "source": "sra",
            "file_path": str(source_file),
            "status": "not_implemented"
        }
    
    def _parse_sample_characteristics(self, metadata: Dict[str, Any], characteristics: List[str]):
        """Parse sample characteristics into structured format."""
        for i, char_string in enumerate(characteristics):
            if i >= len(metadata["samples"]):
                metadata["samples"].append({})
            
            sample = metadata["samples"][i]
            if "characteristics" not in sample:
                sample["characteristics"] = {}
            
            # Parse characteristics (usually key: value format)
            for char in char_string.split(';'):
                char = char.strip()
                if ':' in char:
                    key, value = char.split(':', 1)
                    sample["characteristics"][key.strip()] = value.strip()
                else:
                    sample["characteristics"]["raw"] = char
    
    def _validate_and_clean_samples(self, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Validate and clean sample metadata."""
        cleaned_samples = []
        
        for sample in samples:
            if not sample.get("sample_id"):
                continue
            
            # Extract tissue type
            tissue_type = "unknown"
            characteristics = sample.get("characteristics", {})
            
            # Look for tissue information in characteristics
            for key, value in characteristics.items():
                if any(keyword in key.lower() for keyword in ["tissue", "organ", "cell_type"]):
                    tissue_type = value
                    break
            
            # Create normalized sample metadata
            normalized_sample = {
                "sample_id": sample.get("sample_id"),
                "title": sample.get("title", ""),
                "characteristics": characteristics,
                "library_strategy": sample.get("library_strategy", ""),
                "library_source": sample.get("library_source", ""),
                "organism": sample.get("organism_ch1", ""),
                "tissue_type": tissue_type,
                "experiment_type": sample.get("type", ""),
                "platform": sample.get("platform_id", ""),
                "raw_data": sample
            }
            
            cleaned_samples.append(normalized_sample)
        
        return cleaned_samples
    
    async def extract_supplementary_metadata(self, 
                                           dataset_dir: Path, 
                                           dataset_id: str) -> ToolResult[Dict[str, Any]]:
        """
        Extract metadata from supplementary files in a dataset directory.
        
        Args:
            dataset_dir: Directory containing dataset files
            dataset_id: Dataset identifier
            
        Returns:
            ToolResult with supplementary metadata
        """
        try:
            metadata = {
                "dataset_id": dataset_id,
                "directory": str(dataset_dir),
                "files": [],
                "file_types": [],
                "total_size": 0
            }
            
            if not dataset_dir.exists():
                return ToolResult(
                    success=False,
                    error=f"Dataset directory does not exist: {dataset_dir}"
                )
            
            # Scan directory for files
            for file_path in dataset_dir.rglob('*'):
                if file_path.is_file():
                    file_info = {
                        "name": file_path.name,
                        "path": str(file_path),
                        "size": file_path.stat().st_size,
                        "extension": file_path.suffix.lower()
                    }
                    
                    metadata["files"].append(file_info)
                    metadata["total_size"] += file_info["size"]
                    
                    if file_info["extension"] not in metadata["file_types"]:
                        metadata["file_types"].append(file_info["extension"])
            
            # Categorize files
            metadata["count_files"] = self._find_count_files(metadata["files"])
            metadata["matrix_files"] = self._find_matrix_files(metadata["files"])
            metadata["metadata_files"] = self._find_metadata_files(metadata["files"])
            
            return ToolResult(success=True, data=metadata)
            
        except Exception as e:
            return ToolResult(
                success=False,
                error=f"Failed to extract supplementary metadata: {e}",
                details={"dataset_id": dataset_id}
            )
    
    def _find_count_files(self, files: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Find count/expression files."""
        count_keywords = ["count", "counts", "raw_counts", "readcounts", "featurecounts"]
        count_files = []
        
        for file_info in files:
            filename_lower = file_info["name"].lower()
            if any(keyword in filename_lower for keyword in count_keywords):
                count_files.append(file_info)
        
        return count_files
    
    def _find_matrix_files(self, files: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Find matrix/expression files."""
        matrix_keywords = ["matrix", "expression", "fpkm", "rpkm", "tpm", "cpm"]
        matrix_files = []
        
        for file_info in files:
            filename_lower = file_info["name"].lower()
            if any(keyword in filename_lower for keyword in matrix_keywords):
                matrix_files.append(file_info)
        
        return matrix_files
    
    def _find_metadata_files(self, files: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Find metadata files."""
        metadata_keywords = ["metadata", "sample", "phenotype", "clinical"]
        metadata_files = []
        
        for file_info in files:
            filename_lower = file_info["name"].lower()
            if any(keyword in filename_lower for keyword in metadata_keywords):
                metadata_files.append(file_info)
        
        return metadata_files
    
    def validate_input(self, source_file: Path, source_type: str) -> bool:
        """Validate input parameters."""
        if not isinstance(source_file, Path):
            self.logger.error("source_file must be a Path object")
            return False
        
        if not source_file.exists():
            self.logger.error(f"Source file does not exist: {source_file}")
            return False
        
        if not source_type or not isinstance(source_type, str):
            self.logger.error("source_type must be a non-empty string")
            return False
        
        return True
    
    def validate_output(self, result: Dict[str, Any]) -> bool:
        """Validate output result."""
        if not isinstance(result, dict):
            self.logger.error("Result must be a dictionary")
            return False
        
        required_keys = ["source"]
        for key in required_keys:
            if key not in result:
                self.logger.error(f"Result missing required key: {key}")
                return False
        
        return True 