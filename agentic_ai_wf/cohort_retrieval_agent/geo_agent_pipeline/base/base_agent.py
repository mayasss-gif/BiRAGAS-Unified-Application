"""
Base agent class for the Cohort Retrieval Agent system.

This module provides the abstract base class for all data source agents,
defining the common interface and functionality.
"""
import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import re
import shutil
from collections import Counter
import time
import os
import json

# Project Imports
from   ..config import CohortRetrievalConfig
from   ..tools.evaluator import EvaluationTool
from   ..tools.evaluatorrag import LLMEvaluator, load_datasets_from_folder

@dataclass
class AgentResult:
    """Result container for agent operations."""
    success: bool
    agent_name: str
    disease_name: str
    datasets_found: int = 0
    datasets_downloaded: int = 0
    files_downloaded: int = 0
    total_size_mb: float = 0.0
    execution_time: float = 0.0
    error: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


@dataclass
class DatasetInfo:
    """Information about a discovered dataset."""
    dataset_id: str
    source: str
    title: str
    description: str
    overall_design : str
    sample_count: int
    file_types: List[str]
    tissuecategorization : Dict[str, Any]
    metadata: Dict[str, Any]
    download_urls: List[str]
    estimated_size_mb: float


class BaseRetrievalAgent(ABC):
    """
    Abstract base class for all data source agents.
    
    Each agent is responsible for:
    - Querying a specific data source
    - Filtering and validating datasets
    - Downloading files and metadata
    - Providing progress updates
    """
    
    def __init__(self, config: CohortRetrievalConfig, name: str = ""):
        self.config = config
        self.name = name or self.__class__.__name__
        self.logger = logging.getLogger(f"cohort_retrieval.agent.{self.name}")
        
        # Tools will be initialized by subclasses
        self.tools = {}
        
        # Statistics
        self.total_queries = 0
        self.successful_queries = 0
        self.total_downloads = 0
        self.successful_downloads = 0
        self.total_files_downloaded = 0
        self.total_size_downloaded_mb = 0.0
        
        # Progress tracking
        self.progress_callback = None
        self.current_progress = 0.0
        self.status_message = "Initialized"
    
    @abstractmethod
    async def query_datasets(self, disease_name: str, filters: Optional[Dict[str, Any]] = None, max_datasets:Optional[int] = None) -> List[DatasetInfo]:
        """
        Query the data source for datasets related to a disease.
        
        Args:
            disease_name: Name of the disease to search for
            filters: Optional filters to apply to the search
            
        Returns:
            List of DatasetInfo objects
        """
        print(f"Querying datasets for {disease_name} using {self.name} with filters: {filters}")
        
        pass
    
    @abstractmethod
    async def download_dataset(self, dataset_info: DatasetInfo, output_dir: Path) -> bool:
        """
        Download a specific dataset.
        
        Args:
            dataset_info: Information about the dataset to download
            output_dir: Directory to download files to
            
        Returns:
            True if successful, False otherwise
        """
        print(f"Downloading dataset {dataset_info.dataset_id} using {self.name} to {output_dir}")
        pass
    
    @abstractmethod
    def validate_dataset(self, dataset_info: DatasetInfo) -> bool:
        """
        Validate that a dataset meets the criteria for this agent.
        
        Args:
            dataset_info: Dataset information to validate
            
        Returns:
            True if valid, False otherwise
        """
        print(f"Validating dataset {dataset_info.dataset_id} using {self.name}")
        pass
    
    @abstractmethod
    def get_supported_file_types(self) -> List[str]:
        """Get list of file types supported by this agent."""
        pass
    
    @abstractmethod
    def get_data_source_info(self) -> Dict[str, Any]:
        """Get information about the data source."""
        pass
    
    # -----------------------------
    # Tissue grouping helpers
    # -----------------------------
    _TISSUE_PAT = re.compile(r"^\s*tissue\s*:\s*(.+?)\s*$", re.IGNORECASE)

    def _normalize_tissue(self, name: str) -> str:
        """
        Normalize a tissue string into a clean folder name.
        Examples:
            'brain' -> 'brain'
            'tissue: brain' -> 'brain'
            'Brain (cortex)' -> 'brain_cortex'
        """
        name = name.strip()
        m = self._TISSUE_PAT.match(name)
        if m:
            name = m.group(1)
        name = re.sub(r"[^a-zA-Z0-9]+", "_", name.strip().lower()).strip("_")
        return name or "unknown"

    def _extract_tissues_from_list(self, values) -> list:
        """
        From a list like ['tissue: brain', 'tissue: brain'] -> ['brain', 'brain'] (normalized).
        Accepts either 'brain' or 'tissue: brain' forms.
        """
        tissues = []
        for v in values or []:
            if isinstance(v, str) and v.strip():
                tissues.append(self._normalize_tissue(v))
        return tissues

    def _infer_tissue(self, dataset_info) -> Optional[str]:
        """
        Infer tissue from:
          1) dataset_info.tissuecategorization (if it contains tissue-like info)
          2) dataset_info.metadata['tissue_type'] (list of 'tissue: X' strings)
          3) scan dataset_info.metadata['Characteristics'] for 'tissue:' entries

        Returns normalized string (e.g., 'brain') or None.
        """
        candidates = []

        # 1) If you already prepared a tissuecategorization dict, try to read from there.
        tc = getattr(dataset_info, "tissuecategorization", None)
        if isinstance(tc, dict):
            if "tissue" in tc:
                val = tc["tissue"]
                if isinstance(val, str):
                    candidates.append(self._normalize_tissue(val))
                elif isinstance(val, (list, tuple)):
                    candidates.extend(self._extract_tissues_from_list(val))

        md = getattr(dataset_info, "metadata", {}) or {}

        # 2) metadata['tissue_type'] (list)
        if isinstance(md.get("tissue_type"), list) and md["tissue_type"]:
            candidates.extend(self._extract_tissues_from_list(md["tissue_type"]))

        # 3) scan metadata['Characteristics'] (list[list[str]])
        chars = md.get("Characteristics")
        if isinstance(chars, list):
            for sample_list in chars:
                if isinstance(sample_list, list):
                    for entry in sample_list:
                        if isinstance(entry, str) and entry.lower().startswith("tissue:"):
                            candidates.append(self._normalize_tissue(entry))

        if not candidates:
            return None

        return Counter(candidates).most_common(1)[0][0]

    def _guess_dataset_dir(self, output_dir: Path, dataset_info) -> Path:
        """
        Best-effort: where did download_dataset put files?
        Conventionally: <output_dir>/<dataset_id>
        If not found, fallback to any folder that startswith dataset_id.
        """
        default_path = output_dir / dataset_info.dataset_id
        if default_path.exists():
            return default_path

        # Fallback: look for any folder that starts with dataset_id
        candidates = [p for p in output_dir.iterdir() if p.is_dir() and p.name.startswith(dataset_info.dataset_id)]
        if candidates:
            return candidates[0]

        # Last resort: assume output_dir itself (won't move if we can't find a subdir)
        return output_dir

    def _group_dataset_by_tissue(self, output_dir: Path, dataset_info) -> Optional[Path]:
        """
        Create <output_dir>/<tissue>/ and move the dataset directory under it.
        Returns the new path if moved, or None if not moved (e.g., tissue not found).
        """
        try:
            tissue = self._infer_tissue(dataset_info)
            if not tissue:
                self.logger.info(f"[{self.name}] Tissue inference failed for {dataset_info.dataset_id}; skipping grouping.")
                return None

            src_dir = self._guess_dataset_dir(output_dir, dataset_info)
            if not src_dir.exists() or not src_dir.is_dir():
                self.logger.warning(f"[{self.name}] Dataset dir not found for {dataset_info.dataset_id}: {src_dir}")
                return None

            # If already under the right tissue dir, do nothing
            if src_dir.parent.name == tissue:
                return src_dir

            tissue_dir = output_dir / tissue
            tissue_dir.mkdir(parents=True, exist_ok=True)

            dest_dir = tissue_dir / src_dir.name
            if dest_dir.exists():
                # If it already exists and is the same, do nothing; else add suffix
                try:
                    if dest_dir.samefile(src_dir):
                        return dest_dir
                except Exception:
                    pass
                dest_dir = tissue_dir / f"{src_dir.name}_{dataset_info.dataset_id}"

            self.logger.info(f"[{self.name}] Moving {src_dir} -> {dest_dir}")
            shutil.move(str(src_dir), str(dest_dir))
            return dest_dir
        except Exception as e:
            self.logger.warning(f"[{self.name}] Tissue grouping error for {dataset_info.dataset_id}: {e}")
            return None

    async def retrieve_cohort(self, 
                            disease_name: str, 
                            max_datasets: Optional[int] = None,
                            filters: Optional[Dict[str, Any]] = None,
                            query : Optional[str] = None,
                            output_dir: Optional[Path] = None) -> AgentResult:
        """
        Main method to retrieve cohort data for a disease.
        
        Args:
            disease_name: Name of the disease to search for
            max_datasets: Maximum number of datasets to process
            filters: Optional filters to apply
            output_dir: Output directory (defaults to config output_dir)
            
        Returns:
            AgentResult with operation results
        """

        tissue_filter = filters['tissue_filter']
        experiment_filter = filters['experiment_filter']

        print(f"Retrieving cohort for {disease_name} using {self.name} with max_datasets: {max_datasets} and tissue filters: {tissue_filter} and experiment_filter: {experiment_filter} and output_dir: {output_dir}")
        start_time = time.time()
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output directory: {output_dir}")
        
        # max_datasets = max_datasets or getattr(self.config, f'{self.name.lower()}_config', None)
        # if hasattr(max_datasets, 'max_datasets'):
        #     max_datasets = max_datasets.max_datasets
        # else:
        #     max_datasets = 10  # Default
        
        try:
            self.logger.info(f"Starting cohort retrieval for {disease_name} using {self.name}")
            
            self.total_queries += 1
            self._update_progress(0.0, f"Querying {self.name} for {disease_name}")
            
            # Step 1: Query datasets
            print(query)
            datasets = await self.query_datasets(disease_name, filters, query, output_dir, max_datasets)
            self.logger.info(f"Found {len(datasets)} datasets from {self.name}")
            print(f"Found {len(datasets)} datasets from {self.name}")
            # Save dictionary to a JSON file
            
            if not datasets:
                self.logger.warning(f"No datasets found for {disease_name} in {self.name}")
                print(f"No datasets found for {disease_name} in {self.name}")
                return AgentResult(
                    success=True,
                    agent_name=self.name,
                    disease_name=disease_name,
                    datasets_found=0,
                    execution_time=time.time() - start_time
                )
            
            # Step 2: Filter and validate datasets
            self._update_progress(0.2, f"Filtering {len(datasets)} datasets")
            valid_datasets = []
            for dataset in datasets:
                valid_datasets.append(dataset)
                
            self.logger.info(f"Selected {len(valid_datasets)} valid datasets")
            print(f"Selected {len(valid_datasets)} valid datasets")
            
            # Step 3: Download datasets
            downloaded_count = 0
            total_files = 0
            total_size = 0.0

            for i, dataset in enumerate(valid_datasets):
                progress = 0.2 + (0.8 * (i + 1) / len(datasets))
                self._update_progress(progress, f"Downloading dataset {i+1}/{len(datasets)}: {dataset.dataset_id}")
                
                try:
                    success = await self.download_dataset(dataset, output_dir)
                    if success:
                        downloaded_count += 1
                        total_files += len(dataset.download_urls)
                        total_size += dataset.estimated_size_mb
                        self.successful_downloads += 1
                    else:
                        self.logger.warning(f"Failed to download dataset {dataset.dataset_id}")
                    
                    self.total_downloads += 1
                    
                except Exception as e:
                    self.logger.error(f"Error downloading dataset {dataset.dataset_id}: {e}")
                    self.total_downloads += 1
            
            # Update statistics
            self.successful_queries += 1
            self.total_files_downloaded += total_files
            self.total_size_downloaded_mb += total_size
            
            execution_time = time.time() - start_time
            self._update_progress(1.0, f"Completed: {downloaded_count} datasets downloaded")
            
            self.logger.info(f"Cohort retrieval completed for {disease_name}: {downloaded_count}/{len(valid_datasets)} datasets downloaded")
            
            # if len(datasets) != 0:
            #     config = CohortRetrievalConfig()  # Assuming you have a valid config setup

            #     evaluator = LLMEvaluator()
            #     evaluation_tool = EvaluationTool(config, evaluator)
            #     datasets = load_datasets_from_folder(base_path=output_dir)
            #     print(datasets)
            #     series_meta= datasets.series_matrix
            #     samples = datasets.meta
            #     print("Before ealuations")
            #     evaluation_results = await evaluation_tool.evaluate_datasets(datasets, series_meta, samples, disease_name, query, filters)
            #     result_data = []
            #     for result in evaluation_results:
            #         result_data.append({
            #             "dataset_id": result.dataset_id,
            #             "primary_metrics": result.primary_metrics,
            #             "composite_scores": result.composite_scores,
            #             "detailed_metrics": result.detailed_metrics,
            #             "supporting_references": result.supporting_references,
            #             "justification": result.justification,
            #             "error_message": result.error_message
            #         })
            #     # Save the results to a JSON file
            #     filepath = os.path.join(output_dir, "evaluation_results.json")
            #     with open(filepath, "w", encoding="utf-8") as json_file:
            #         json.dump(result_data, json_file, ensure_ascii=False, indent=4)

            #     print(f"Evaluation results saved to {filepath}")
 
            return AgentResult(
                success=True,
                agent_name=self.name,
                disease_name=disease_name,
                datasets_found=len(datasets),
                datasets_downloaded=downloaded_count,
                files_downloaded=total_files,
                total_size_mb=total_size,
                execution_time=execution_time,
                details={
                    "valid_datasets": len(valid_datasets),
                    "downloaded_datasets": [d.dataset_id for d in valid_datasets[:downloaded_count]],
                    "output_directory": str(output_dir)
                }
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"Error in {self.name} cohort retrieval: {e}"
            self.logger.error(error_msg)
            
            return AgentResult(
                success=False,
                agent_name=self.name,
                disease_name=disease_name,
                execution_time=execution_time,
                error=error_msg,
                details={"exception": str(e)}
            )
    
    def set_progress_callback(self, callback):
        """Set a callback function for progress updates."""
        self.progress_callback = callback
    
    def _update_progress(self, progress: float, message: str):
        """Update progress and notify callback if set."""
        self.current_progress = progress
        self.status_message = message
        
        if self.progress_callback:
            self.progress_callback(self.name, progress, message)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get agent execution statistics."""
        query_success_rate = (self.successful_queries / self.total_queries * 100) if self.total_queries > 0 else 0
        download_success_rate = (self.successful_downloads / self.total_downloads * 100) if self.total_downloads > 0 else 0
        
        return {
            "agent_name": self.name,
            "total_queries": self.total_queries,
            "successful_queries": self.successful_queries,
            "query_success_rate": round(query_success_rate, 2),
            "total_downloads": self.total_downloads,
            "successful_downloads": self.successful_downloads,
            "download_success_rate": round(download_success_rate, 2),
            "total_files_downloaded": self.total_files_downloaded,
            "total_size_downloaded_mb": round(self.total_size_downloaded_mb, 2),
            "current_progress": self.current_progress,
            "status_message": self.status_message
        }
    
    def reset_statistics(self):
        """Reset agent execution statistics."""
        self.total_queries = 0
        self.successful_queries = 0
        self.total_downloads = 0
        self.successful_downloads = 0
        self.total_files_downloaded = 0
        self.total_size_downloaded_mb = 0.0
        self.current_progress = 0.0
        self.status_message = "Statistics reset"
        
        # Reset tool statistics
        for tool in self.tools.values():
            tool.reset_statistics()
        
        self.logger.info(f"Statistics reset for {self.name}")
    
    def log_performance(self):
        """Log performance statistics."""
        stats = self.get_statistics()
        self.logger.info(f"Performance stats for {self.name}: {stats}")
        
        # Log tool statistics
        for tool in self.tools.values():
            tool.log_performance()
    
    async def cleanup(self):
        """Cleanup resources used by the agent."""
        self.logger.info(f"Cleaning up {self.name} agent")
        
        # Close any tools that need cleanup
        for tool in self.tools.values():
            if hasattr(tool, 'cleanup'):
                try:
                    await tool.cleanup()
                except Exception as e:
                    self.logger.warning(f"Error cleaning up tool {tool.name}: {e}")
        
        # Cleanup temporary files if configured
        if self.config.cleanup_temp_files:
            temp_dir = Path(self.config.temp_dir) / self.name.lower()
            if temp_dir.exists():
                try:
                    shutil.rmtree(temp_dir)
                    self.logger.info(f"Cleaned up temporary directory: {temp_dir}")
                except Exception as e:
                    self.logger.warning(f"Error cleaning up temp directory: {e}")
    
    def __str__(self):
        return f"{self.name}Agent(queries={self.total_queries}, downloads={self.successful_downloads})"
    
    def __repr__(self):
        return f"{self.__class__.__name__}(name='{self.name}', config={self.config})" 