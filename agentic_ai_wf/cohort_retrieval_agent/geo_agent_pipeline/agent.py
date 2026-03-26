"""
Main Cohort Retrieval Agent orchestrator.

This module provides the main agent that coordinates multiple data source agents
to retrieve comprehensive cohort data for biomedical research.
"""

import asyncio
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
import json
import datetime
import os
import json

# Project Imports
from .config import CohortRetrievalConfig
from .base.base_agent import AgentResult 
from .agents import GEORetrievalAgent
from .config import AgentName

@dataclass
class CohortResult:
    """Result of cohort retrieval across all agents."""
    success: bool
    disease_name: str
    total_datasets_found: int = 0
    total_datasets_downloaded: int = 0
    total_files_downloaded: int = 0
    total_size_mb: float = 0.0
    execution_time: float = 0.0
    agent_results: Dict[str, AgentResult] = field(default_factory=dict)
    output_directory: str = ""
    error: Optional[str] = None


class CohortRetrievalAgent:
    """
    Main orchestrator for cohort retrieval across multiple data sources.

    Coordinates:
    - GEO Agent
    - SRA Agent (future)
    - TCGA Agent (future) 
    - GTEx Agent (future)
    - ArrayExpress Agent (future)

    Provides:
    - Progress tracking across all agents
    - Error handling and recovery
    - Result aggregation and reporting
    - Resource management
    """

    def __init__(self, config: Optional[CohortRetrievalConfig] = None):
        self.config = config or CohortRetrievalConfig()
        self.logger = logging.getLogger("cohort_retrieval.main")

         # Suppress noisy external logs
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("openai").setLevel(logging.WARNING)

        # Add handler and formatter if not already added
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

        self.logger.setLevel(logging.INFO)
        # Initialize agents based on configuration
        self.agents = {}
        self._initialize_agents()

        # Progress tracking
        self.progress_callbacks = []
        self.current_progress = {}

        # Statistics
        self.total_cohort_retrievals = 0
        self.successful_cohort_retrievals = 0
        self.total_execution_time = 0.0

        # Setup logging
        self._setup_logging()

    def _initialize_agents(self):
        """Initialize data source agents based on configuration."""
        # Always initialize GEO agent
        self.agents["geo"] = GEORetrievalAgent(self.config)
        self.logger.info("Initialized GEO agent")

        # Initialize other agents based on configuration
        enabled_agents = self.config.get_enabled_agents()

        # Skip abstract agents that aren't fully implemented yet
        # TODO: Implement these agents properly
        if "sra" in enabled_agents:
            self.logger.info("SRA agent not yet implemented (placeholder)")

        if "tcga" in enabled_agents:
            self.logger.info("TCGA agent not yet implemented (placeholder)")

        if "gtex" in enabled_agents:
            self.logger.info("GTEx agent not yet implemented (placeholder)")

        if "arrayexpress" in enabled_agents:
            self.logger.info(
                "ArrayExpress agent not yet implemented (placeholder)")

        self.logger.info(
            f"Initialized {len(self.agents)} agents: {list(self.agents.keys())}")

        print(
            f"Initialized {len(self.agents)} agents: {list(self.agents.keys())}")

    def _setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=getattr(logging, self.config.log_level),
            format=self.config.log_format,
            handlers=[
                logging.FileHandler(
                    Path(self.config.log_dir) / "cohort_retrieval.log"),
                logging.StreamHandler()
            ]
        )

    async def retrieve_cohort(self,
                              disease_name: str,
                              max_datasets_per_source: Optional[int] = None,
                              filters: Optional[Dict[str, Any]] = None,
                              query : Optional[str] = None,
                              output_dir: Optional[Path] = None,
                              agents_to_use: Optional[List[str]] = None) -> CohortResult:
        """
        Main method to retrieve cohort data for a disease across all sources.

        Args:
            disease_name: Name of the disease to search for
            max_datasets_per_source: Maximum datasets per data source
            filters: Optional filters to apply
            output_dir: Output directory (defaults to config output_dir)
            agents_to_use: Specific agents to use (defaults to all enabled)

        Returns:
            CohortResult with aggregated results
        """
        print(
            f"Retrieving cohort for '{disease_name}' with max_datasets_per_source: {max_datasets_per_source} and filters: {filters} and output_dir: {output_dir} and agents_to_use: {agents_to_use}")
        start_time = time.time()
        self.total_cohort_retrievals += 1
        
        # Setup output directory
        # If output_dir is already a full path, use it directly
        # Otherwise, use get_disease_path to construct the path from disease_name
        if output_dir is None:
            # No output_dir provided, use default from config
            output_dir = self.config.directory_paths.get_disease_path(
                AgentName.GEO.value, disease_name)
        else:
            # Convert to Path if it's a string
            output_dir = Path(output_dir) if isinstance(output_dir, str) else output_dir
            
            # Check if it's already a full path (contains path separators indicating it's a directory path)
            # vs just a disease name (single word/identifier)
            output_str = str(output_dir)
            
            # If it's an absolute path OR contains path separators (indicating it's a directory path)
            # use it directly, otherwise treat as disease name and use get_disease_path
            is_full_path = (
                output_dir.is_absolute() or 
                (os.sep in output_str or '/' in output_str) or
                output_str.startswith('.')  # Relative paths like ./path/to/dir
            )
            
            if is_full_path:
                # Already a full path, use it directly (don't modify)
                pass
            else:
                # Looks like just a disease name, use get_disease_path to construct full path
                output_dir = self.config.directory_paths.get_disease_path(
                    AgentName.GEO.value, output_str)

        output_dir.mkdir(parents=True, exist_ok=True)
        print("After agent setup directory")
        print("Before output directory setup in agent")
        print(output_dir)
        agents_to_use = agents_to_use or list(self.agents.keys())
        agents_to_use = [
            agent for agent in agents_to_use if agent in self.agents]

        self.logger.info(
            f"Starting cohort retrieval for '{disease_name}' using agents: {agents_to_use}")
        print(
            f"Starting cohort retrieval for '{disease_name}' using agents: {agents_to_use}")

        try:
            # Initialize progress tracking
            self.current_progress = {agent: 0.0 for agent in agents_to_use}
            self._update_overall_progress(
                0.0, f"Starting cohort retrieval for {disease_name}")
            print(f"Starting cohort retrieval for {disease_name}")

            # Run agents concurrently or sequentially based on configuration
            if self.config.max_concurrent_agents > 1 and len(agents_to_use) > 1:
                print(f"Running {len(agents_to_use)} agents concurrently")
                agent_results = await self._run_agents_concurrently(
                    disease_name, agents_to_use, max_datasets_per_source, filters, output_dir
                )
            else:
                print(f"Running {len(agents_to_use)} agents sequentially")
                agent_results = await self._run_agents_sequentially(
                    disease_name, agents_to_use, max_datasets_per_source, query, filters, output_dir
                )

            
            print("Agrreation Results using Output Directory")
            print(output_dir)
            # Aggregate results
            cohort_result = self._aggregate_results(
                disease_name, agent_results, output_dir, time.time() - start_time
            )
            print(
                f"Cohort retrieval completed for '{disease_name}': {cohort_result.total_datasets_downloaded} datasets downloaded")
            # Update statistics
            if cohort_result.success:
                self.successful_cohort_retrievals += 1
            self.total_execution_time += cohort_result.execution_time

            # Save cohort summary
            await self._save_cohort_summary(cohort_result, output_dir)

            self._update_overall_progress(1.0, "Cohort retrieval completed")
            self.logger.info(
                f"Cohort retrieval completed for '{disease_name}': {cohort_result.total_datasets_downloaded} datasets downloaded")

            return cohort_result

        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"Cohort retrieval failed for '{disease_name}': {e}"
            self.logger.error(error_msg)

            return CohortResult(
                success=False,
                disease_name=disease_name,
                execution_time=execution_time,
                output_directory=str(output_dir),
                error=error_msg
            )

    async def _run_agents_concurrently(self,
                                       disease_name: str,
                                       agents_to_use: List[str],
                                       max_datasets_per_source: Optional[int],
                                       filters: Optional[Dict[str, Any]],
                                       output_dir: Path) -> Dict[str, AgentResult]:
        """Run multiple agents concurrently."""
        self.logger.info(f"Running {len(agents_to_use)} agents concurrently")

        # Limit concurrent agents
        semaphore = asyncio.Semaphore(self.config.max_concurrent_agents)

        async def run_agent_with_semaphore(agent_name: str) -> AgentResult:
            async with semaphore:
                return await self._run_single_agent(
                    agent_name, disease_name, max_datasets_per_source, filters, output_dir
                )

        # Create tasks for all agents
        tasks = {
            agent_name: asyncio.create_task(
                run_agent_with_semaphore(agent_name))
            for agent_name in agents_to_use
        }

        # Wait for all agents to complete
        results = {}
        for agent_name, task in tasks.items():
            try:
                results[agent_name] = await task
            except Exception as e:
                self.logger.error(f"Agent {agent_name} failed: {e}")
                results[agent_name] = AgentResult(
                    success=False,  # type: ignore
                    agent_name=agent_name,
                    disease_name=disease_name,
                    error=str(e)
                )

        return results

    async def _run_agents_sequentially(self,
                                       disease_name: str,
                                       agents_to_use: List[str],
                                       max_datasets_per_source: Optional[int],
                                       output_dir: Path,
                                       query : Optional[str] = None,
                                       filters: Optional[Dict[str, Any]] = None
                                       ) -> Dict[str, AgentResult]:
        """Run agents sequentially."""
        self.logger.info(f"Running {len(agents_to_use)} agents sequentially")

        results = {}
        for i, agent_name in enumerate(agents_to_use):
            try:
                progress = i / len(agents_to_use)
                print(f"Running {agent_name} agent progress: {progress}")

                self._update_overall_progress(
                    progress, f"Running {agent_name} agent")

                print(f"Running {agent_name} agent with run_single_agent")
                result = await self._run_single_agent(
                    agent_name, disease_name, max_datasets_per_source, query, filters, output_dir
                )
                results[agent_name] = result

            except Exception as e:
                self.logger.error(f"Agent {agent_name} failed: {e}")
                results[agent_name] = AgentResult(
                    success=False,
                    agent_name=agent_name,
                    disease_name=disease_name,
                    error=str(e)
                )

        return results

    async def _run_single_agent(self,
                                agent_name: str,
                                disease_name: str,
                                max_datasets_per_source: Optional[int],
                                filters: Optional[Dict[str, Any]],
                                output_dir: Path,
                                query: Optional[str] = None) -> AgentResult:
        """Run a single agent."""
        agent = self.agents[agent_name]

        # Set up progress callback for this agent
        def progress_callback(agent_name: str, progress: float, message: str):
            print(
                f"Progress callback for {agent_name}: {progress} - {message}")
            self.current_progress[agent_name] = progress
            overall_progress = sum(
                self.current_progress.values()) / len(self.current_progress)
            self._update_overall_progress(
                overall_progress, f"{agent_name}: {message}")

        print(f"Setting progress callback for {agent_name}")
        agent.set_progress_callback(progress_callback)
        print("Before calling retreive cohort from inside ")
        print(output_dir)
        # Run the agent
        # GEO Retrieve Cohort
        result = await agent.retrieve_cohort(
            disease_name=disease_name,
            max_datasets=max_datasets_per_source,
            filters=filters,
            query = query, 
            output_dir=output_dir
        )

        return result

    def _aggregate_results(self,
                           disease_name: str,
                           agent_results: Dict[str, AgentResult],
                           output_dir: Path,
                           execution_time: float) -> CohortResult:
        """Aggregate results from all agents."""
        total_datasets_found = sum(
            result.datasets_found for result in agent_results.values())
        total_datasets_downloaded = sum(
            result.datasets_downloaded for result in agent_results.values())
        total_files_downloaded = sum(
            result.files_downloaded for result in agent_results.values())
        total_size_mb = sum(
            result.total_size_mb for result in agent_results.values())

        # Check if any agent succeeded
        success = any(result.success for result in agent_results.values())

        # Collect errors
        errors = [result.error for result in agent_results.values()
                  if result.error]
        error_message = "; ".join(errors) if errors else None

        return CohortResult(
            success=success,
            disease_name=disease_name,
            total_datasets_found=total_datasets_found,
            total_datasets_downloaded=total_datasets_downloaded,
            total_files_downloaded=total_files_downloaded,
            total_size_mb=total_size_mb,
            execution_time=execution_time,
            agent_results=agent_results,
            output_directory=str(output_dir),
            error=error_message if not success else None
        )

    async def _save_cohort_summary(self, cohort_result: CohortResult, output_dir: Path):
        """Save cohort retrieval summary to JSON file."""
        try:
            summary = {
                "cohort_retrieval_summary": {
                    "disease_name": cohort_result.disease_name,
                    "timestamp": datetime.datetime.now().isoformat(),
                    "success": cohort_result.success,
                    "execution_time_seconds": cohort_result.execution_time,
                    "total_datasets_found": cohort_result.total_datasets_found,
                    "total_datasets_downloaded": cohort_result.total_datasets_downloaded,
                    "total_files_downloaded": cohort_result.total_files_downloaded,
                    "total_size_mb": cohort_result.total_size_mb,
                    "output_directory": cohort_result.output_directory,
                    "error": cohort_result.error
                },
                "agent_results": {}
            }

            # Add individual agent results
            for agent_name, result in cohort_result.agent_results.items():
                summary["agent_results"][agent_name] = {
                    "success": result.success,
                    "datasets_found": result.datasets_found,
                    "datasets_downloaded": result.datasets_downloaded,
                    "files_downloaded": result.files_downloaded,
                    "total_size_mb": result.total_size_mb,
                    "execution_time": result.execution_time,
                    "error": result.error
                }

            summary_file = output_dir / \
                f"{cohort_result.disease_name}_cohort_summary.json"
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)

            self.logger.info(f"Saved cohort summary to {summary_file}")

            # Also save detailed results
            await self._save_cohort_detailed_results(cohort_result, output_dir)

        except Exception as e:
            self.logger.warning(f"Failed to save cohort summary: {e}")

    async def _save_cohort_detailed_results(self, cohort_result: CohortResult, output_dir: Path):
        """Save detailed cohort results similar to old literature_module.py format."""
        try:
            print("Inside saving save cohort details")
            print(output_dir)
            # Collect detailed results from all agents
            detailed_results = []
            print(cohort_result.agent_results.items())
            for agent_name, agent_result in cohort_result.agent_results.items():
                if not agent_result.success:
                    continue

                # Get detailed dataset information from individual dataset metadata files
                agent_datasets = await self._load_agent_detailed_datasets(agent_name, cohort_result.disease_name, output_dir)
                for dataset_info in agent_datasets:
                    print("Dataset Info")
                    # Format to match old literature_module.py results structure
                    result_entry = {
                        "Dataset ID": dataset_info.get("dataset_id", ""),
                        "Source": dataset_info.get("source", agent_name),
                        "Title": dataset_info.get("title", ""),
                        "Description": dataset_info.get("description", ""),
                        "Cohort Size": dataset_info.get("metadata", {}).get("sample_id", 0),
                        "Tissue Types": dataset_info.get("metadata", {}).get("tissue_type", []),
                        "Library Sources": dataset_info.get("metadata", {}).get("library_source", []),
                        "Library Strategies": dataset_info.get("metadata", {}).get("library_strategy", []),
                        "Molecule": dataset_info.get("metadata", {}).get("molecule", []),
                        "Estimated Size MB": dataset_info.get("estimated_size_mb", 0),
                        "Download Timestamp": dataset_info.get("download_timestamp", ""),
                        "Agent Version": dataset_info.get("agent_version", "1.0.0")
                    }
                    detailed_results.append(result_entry)

            # Create detailed results file structure
            detailed_results_data = {
                "cohort_detailed_results": {
                    "disease_name": cohort_result.disease_name,
                    "timestamp": datetime.datetime.now().isoformat(),
                    "total_datasets": len(detailed_results),
                    "agents_used": list(cohort_result.agent_results.keys()),
                    "success": cohort_result.success,
                    "execution_time_seconds": cohort_result.execution_time,
                    "output_directory": cohort_result.output_directory
                },
                "results": detailed_results
            }

            # Save to _cohort_details.json file
            details_file = output_dir / \
                f"{cohort_result.disease_name}_cohort_details.json"
            with open(details_file, 'w') as f:
                json.dump(detailed_results_data, f, indent=4)

            self.logger.info(
                f"Saved detailed cohort results to {details_file}")
            self.logger.info(
                f"Detailed results contain {len(detailed_results)} datasets from {len(cohort_result.agent_results)} agents")

        except Exception as e:
            self.logger.warning(f"Failed to save detailed cohort results: {e}")

    async def _load_agent_detailed_datasets(self, agent_name: str, disease_name: str, output_dir: Path) -> List[Dict[str, Any]]:
        """Load detailed dataset information from agent metadata files."""
        detailed_datasets = []

        try:
            # Look for agent-specific directories
            agent_dir = output_dir / agent_name.lower()
            if not agent_dir.exists():
                agent_dir = output_dir  # Fallback to output_dir if agent-specific dir doesn't exist

            # Find all dataset directories
            for item in agent_dir.iterdir():
                if item.is_dir() and (item.name.startswith("GSE") or item.name.startswith("SRP") or item.name.startswith("TCGA")):
                    dataset_id = item.name
                    # Look for metadata files
                    metadata_files = [
                        item / f"{dataset_id}_metadata.json",
                        item / f"{dataset_id}_enhanced_metadata.json"
                    ]

                    for metadata_file in metadata_files:
                        if metadata_file.exists():
                            try:
                                with open(metadata_file, 'r') as f:
                                    metadata = json.load(f)
                                detailed_datasets.append(metadata)
                                break  # Use first found metadata file
                            except Exception as e:
                                self.logger.warning(
                                    f"Error loading metadata from {metadata_file}: {e}")
                                continue

            self.logger.debug(
                f"Loaded {len(detailed_datasets)} detailed datasets for {agent_name}")

        except Exception as e:
            self.logger.warning(
                f"Error loading detailed datasets for {agent_name}: {e}")

        return detailed_datasets

    def add_progress_callback(self, callback: Callable[[float, str], None]):
        """Add a progress callback function."""
        self.progress_callbacks.append(callback)

    def _update_overall_progress(self, progress: float, message: str):
        """Update overall progress and notify callbacks."""
        print(f"Updating overall progress: {progress} - {message}")
        for callback in self.progress_callbacks:
            try:
                callback(progress, message)
            except Exception as e:
                self.logger.warning(f"Progress callback failed: {e}")

    def get_statistics(self) -> Dict[str, Any]:
        """Get overall statistics."""
        success_rate = (self.successful_cohort_retrievals /
                        self.total_cohort_retrievals * 100) if self.total_cohort_retrievals > 0 else 0
        avg_execution_time = self.total_execution_time / \
            max(self.total_cohort_retrievals, 1)

        stats = {
            "main_agent": {
                "total_cohort_retrievals": self.total_cohort_retrievals,
                "successful_cohort_retrievals": self.successful_cohort_retrievals,
                "success_rate": round(success_rate, 2),
                "average_execution_time": round(avg_execution_time, 2),
                "total_execution_time": round(self.total_execution_time, 2)
            },
            "agents": {}
        }

        # Add individual agent statistics
        for agent_name, agent in self.agents.items():
            stats["agents"][agent_name] = agent.get_statistics()

        return stats

    def reset_statistics(self):
        """Reset all statistics."""
        self.total_cohort_retrievals = 0
        self.successful_cohort_retrievals = 0
        self.total_execution_time = 0.0

        for agent in self.agents.values():
            agent.reset_statistics()

        self.logger.info("Statistics reset for all agents")

    async def cleanup(self):
        """Cleanup resources used by all agents."""
        self.logger.info("Cleaning up cohort retrieval agent")

        for agent_name, agent in self.agents.items():
            try:
                await agent.cleanup()
            except Exception as e:
                self.logger.warning(
                    f"Error cleaning up agent {agent_name}: {e}")

    def __str__(self):
        return f"CohortRetrievalAgent(agents={list(self.agents.keys())}, retrievals={self.total_cohort_retrievals})"

    def __repr__(self):
        return f"CohortRetrievalAgent(config={self.config}, agents={len(self.agents)})"
