"""
Main DEG Pipeline Agent with self-healing capabilities.
"""

import asyncio
import time
import logging
import sys
import io
from typing import Dict, List, Optional, Union, Any
from pathlib import Path
from dataclasses import asdict
import pandas as pd

from .config import DEGPipelineConfig
from .exceptions import DEGPipelineError, RecoverableError, NonRecoverableError  # pyright: ignore[reportUnusedImport, reportUnusedImport]
from .tools import (
    DataLoaderTool, DatasetDetectorTool, MetadataExtractorTool,
    DESeq2AnalyzerTool, GeneMapperTool, FileValidatorTool, ErrorFixerTool,
    DEGPlotterTool
)


class DEGPipelineAgent:
    """
    Main agent for DEG pipeline execution with self-healing capabilities.
    
    This agent orchestrates the entire DEG analysis workflow using specialized tools,
    with comprehensive error handling and automatic recovery mechanisms.
    """
    
    def __init__(self, config: Optional[DEGPipelineConfig] = None):
        """
        Initialize the DEG Pipeline Agent.
        
        Args:
            config: Configuration object. If None, uses default configuration.
        """
        self.config = config or DEGPipelineConfig.default()
        self.config._validate()
        
        # Initialize logging
        self.logger = self._setup_logging()
        
        # Initialize tools
        self.tools = self._initialize_tools()
        
        # Execution state
        self.execution_state = {
            "current_step": None,
            "completed_steps": [],
            "failed_steps": [],
            "start_time": None,
            "end_time": None,
            "total_datasets": 0,
            "successful_datasets": 0,
            "failed_datasets": 0,
            "results": []
        }
        self._ui_logger = None
        self._ui_loop = None

        self.logger.info(f"🤖 {self.config.agent_name} initialized successfully")
    
    def _setup_logging(self) -> logging.Logger:
        """
        Setup logging configuration.
        
        Returns:
            Configured logger
        """
        logger = logging.getLogger(self.config.agent_name)
        logger.setLevel(getattr(logging, self.config.log_level))
        
        # Remove existing handlers
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, self.config.log_level))
        formatter = logging.Formatter(self.config.log_format)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # File handler if enabled
        if self.config.log_to_file:
            log_dir = self.config._get_output_dir() / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            print(f"🔍 Log directory: {log_dir}")
            log_file = log_dir / f"{self.config.agent_name}.log"
            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_handler.setLevel(getattr(logging, self.config.log_level))
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        return logger
    
    def _initialize_tools(self) -> Dict[str, Any]:
        """
        Initialize all pipeline tools.
        
        Returns:
            Dictionary of initialized tools
        """
        tools = {
            "data_loader": DataLoaderTool(self.config, self.logger),
            "dataset_detector": DatasetDetectorTool(self.config, self.logger),
            "metadata_extractor": MetadataExtractorTool(self.config, self.logger),
            "deseq2_analyzer": DESeq2AnalyzerTool(self.config, self.logger),
            "gene_mapper": GeneMapperTool(self.config, self.logger),
            # "file_validator": FileValidatorTool(self.config, self.logger),
            "error_fixer": ErrorFixerTool(self.config, self.logger),
            "deg_plotter": DEGPlotterTool(self.config, self.logger)
        }
        
        self.logger.info(f"🔧 Initialized {len(tools)} tools")
        return tools
    
    def _emit_ui_log(self, level: str, message: str, **kwargs: Any) -> None:
        """Emit log to UI via workflow_logger when running in thread pool."""
        workflow_logger = getattr(self, "_ui_logger", None)
        event_loop = getattr(self, "_ui_loop", None)
        if not workflow_logger or not event_loop:
            if self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug(
                    "DEG UI log skipped (no workflow_logger=%s, event_loop=%s)",
                    workflow_logger is not None, event_loop is not None
                )
            return
        try:
            agent_name = "DEG Analysis Agent"
            step = "deg_analysis"

            async def _do_log() -> None:
                try:
                    if level == "info":
                        await workflow_logger.info(agent_name=agent_name, message=message, step=step, **kwargs)
                    elif level == "warning":
                        await workflow_logger.warning(agent_name=agent_name, message=message, step=step, **kwargs)
                    elif level == "error":
                        await workflow_logger.error(agent_name=agent_name, message=message, step=step, **kwargs)
                except Exception:
                    pass

            def _schedule() -> None:
                try:
                    asyncio.create_task(_do_log())
                except Exception:
                    pass

            event_loop.call_soon_threadsafe(_schedule)
        except Exception:
            pass

    def run_pipeline(self, geo_dir: Optional[Union[str, Path]] = None,
                    analysis_transcriptome_dir: Optional[Union[str, Path]] = None,
                    disease_name: Optional[str] = None,
                    _pipeline_retry_count: int = 0,
                    workflow_logger: Any = None,
                    event_loop: Optional[asyncio.AbstractEventLoop] = None) -> Dict[str, Any]:
        """
        Run the complete DEG pipeline.
        
        Args:
            geo_dir: Directory containing GEO datasets
            analysis_transcriptome_dir: Directory containing analysis transcriptome datasets
            disease_name: Disease name for analysis
            _pipeline_retry_count: Internal retry counter (do not set manually)
            
        Returns:
            Pipeline execution results
            
        Raises:
            DEGPipelineError: If pipeline execution fails
        """
        start_time = time.time()
        self.execution_state["start_time"] = start_time
        self._ui_logger = workflow_logger
        self._ui_loop = event_loop

        # Max pipeline-level retries: 2 attempts total (initial + 1 retry)
        max_pipeline_retries = 2

        try:
            self.logger.info("🚀 Starting DEG Pipeline execution")
            self._emit_ui_log("info", "Starting differential expression analysis pipeline")

            # Update configuration with provided parameters
            self._update_config(geo_dir, analysis_transcriptome_dir, disease_name)
            
            # # Create work directories
            # self.config.create_work_directories()
            
            # Step 1: Detect datasets
            dataset_pairs = self._detect_datasets()
            
            # Step 2: Process each dataset
            results = self._process_datasets(dataset_pairs)
            
            # Step 3: Generate summary
            summary = self._generate_summary(results)
            
            self.execution_state["end_time"] = time.time()
            self.execution_state["results"] = results
            elapsed = self.execution_state['end_time'] - start_time
            self.logger.info(f"✅ Pipeline completed successfully in {elapsed:.2f}s")
            self._emit_ui_log("info", f"Differential expression pipeline completed successfully in {elapsed:.1f}s")
            
            return {
                "status": "success",
                "execution_time": self.execution_state["end_time"] - start_time,
                "summary": summary,
                "results": results,
                "config": self.config._to_dict()
            }
            
        except Exception as e:
            self.execution_state["end_time"] = time.time()
            self.logger.error(f"❌ Pipeline failed: {e}")
            self._emit_ui_log("error", f"Pipeline failed: {str(e)}", exception=e)
            
            # Check if this is a non-recoverable error (index mismatches, data alignment issues)
            is_non_recoverable = (
                "Index are different" in str(e) or
                "index does not match" in str(e).lower() or
                "sample mismatch" in str(e).lower() or
                "no common samples" in str(e).lower() or
                isinstance(e, NonRecoverableError)
            )
            
            if is_non_recoverable:
                self.logger.error(f"❌ Non-recoverable error detected. Stopping pipeline: {e}")
                raise DEGPipelineError(f"Pipeline execution failed with non-recoverable error: {e}") from e
            
            # Try to fix the error if possible (only for recoverable errors)
            if self.config.enable_auto_fix and _pipeline_retry_count < max_pipeline_retries:
                try:
                    attempt = _pipeline_retry_count + 1
                    self.logger.info(f"🔧 Attempting pipeline recovery (attempt {attempt}/{max_pipeline_retries})...")
                    self._emit_ui_log(
                        "info",
                        f"Attempting pipeline recovery (attempt {attempt}/{max_pipeline_retries})",
                    )
                    self._attempt_pipeline_recovery(e)
                    # If recovery successful, retry with incremented counter
                    return self.run_pipeline(
                        geo_dir,
                        analysis_transcriptome_dir,
                        disease_name,
                        _pipeline_retry_count=_pipeline_retry_count + 1,
                        workflow_logger=workflow_logger,
                        event_loop=event_loop,
                    )
                except Exception as recovery_error:
                    self.logger.error(f"❌ Recovery failed: {recovery_error}")
                    if _pipeline_retry_count >= max_pipeline_retries - 1:
                        raise DEGPipelineError(
                            f"Pipeline execution failed after {max_pipeline_retries} attempts: {e}"
                        ) from e
            elif _pipeline_retry_count >= max_pipeline_retries:
                self.logger.error(f"❌ Max pipeline retries ({max_pipeline_retries}) exceeded. Stopping.")
            
            raise DEGPipelineError(f"Pipeline execution failed: {e}") from e
        finally:
            self._ui_logger = None
            self._ui_loop = None

    def _update_config(self, geo_dir: Optional[Union[str, Path]], 
                      analysis_transcriptome_dir: Optional[Union[str, Path]], 
                      disease_name: Optional[str]) -> None:
        """
        Update configuration with provided parameters.
        
        Args:
            geo_dir: GEO directory
            analysis_transcriptome_dir: Analysis transcriptome directory  
            disease_name: Disease name
        """
        if geo_dir:
            self.config.geo_dir = str(Path(geo_dir).expanduser().resolve())
        if analysis_transcriptome_dir:
            self.config.analysis_transcriptome_dir = str(Path(analysis_transcriptome_dir).expanduser().resolve())
        if disease_name:
            self.config.disease_name = disease_name
        
        # Re-validate configuration
        self.config._validate()
    
    def _detect_datasets(self) -> List[Dict]:
        """
        Detect all available datasets.
        
        Returns:
            List of dataset pairs
            
        Raises:
            DEGPipelineError: If dataset detection fails
        """
        self.execution_state["current_step"] = "dataset_detection"
        self._emit_ui_log("info", "Agent scanning for datasets...")

        all_pairs = []

        # Detect analysis transcriptome datasets
        if self.config.analysis_transcriptome_dir:
            try:
                self.logger.info(f"🔍 Scanning analysis transcriptome directory: {self.config.analysis_transcriptome_dir}")
                analysis_transcriptome_result = self.tools["dataset_detector"].safe_execute(
                    self.config.analysis_transcriptome_dir, self.config.disease_name
                )
                all_pairs.extend(analysis_transcriptome_result["pairs"])
                if analysis_transcriptome_result["pairs"]:
                    self.logger.info(f"📊 Found {len(analysis_transcriptome_result['pairs'])} analysis transcriptome datasets")
                
            except Exception as e:
                self.logger.error(f"❌ Analysis transcriptome dataset detection failed: {e}")
                if not self.config.geo_dir:
                    raise DEGPipelineError(f"No datasets available: {e}")
        
        # Detect GEO datasets
        if self.config.geo_dir:
            try:
                self.logger.info(f"🔍 Scanning GEO directory: {self.config.geo_dir}")
                geo_result = self.tools["dataset_detector"].safe_execute(
                    self.config.geo_dir, self.config.disease_name
                )
                all_pairs.extend(geo_result["pairs"])
                if geo_result["pairs"]:
                    self.logger.info(f"📊 Found {len(geo_result['pairs'])} GEO datasets")
                
            except Exception as e:
                self.logger.error(f"❌ GEO dataset detection failed: {e}")
                if not all_pairs:
                    raise DEGPipelineError(f"No datasets available: {e}")
        
        if not all_pairs:
            raise DEGPipelineError("No valid dataset pairs found")
        
        self.execution_state["total_datasets"] = len(all_pairs)
        self.execution_state["completed_steps"].append("dataset_detection")
        n = len(all_pairs)
        self._emit_ui_log("info", f"Found {n} dataset(s) ready for analysis" if n != 1 else "Found 1 dataset ready for analysis")

        self.logger.info(f"📦 Total datasets detected: {len(all_pairs)}")
        return all_pairs
    
    def _rank_datasets(self, dataset_pairs: List[Dict]) -> List[Dict]:
        """
        Rank datasets by quality, file availability, metadata completeness, and relevancy.
        
        Args:
            dataset_pairs: List of dataset pairs to rank
            
        Returns:
            List of ranked dataset pairs with scores and reasoning
        """
        ranked_pairs = []
        
        disease_name_lower = (self.config.disease_name or "").lower()
        user_query_lower = (getattr(self.config, 'user_query', None) or "").lower()
        
        for pair in dataset_pairs:
            score = 0.0
            reasons = []
            
            # 1. File availability (40 points max)
            counts_file = Path(pair.get("counts_file", ""))
            metadata_file = pair.get("metadata_file")
            
            if counts_file.exists():
                score += 20.0
                reasons.append("counts_file_exists")
                
                # Bonus for file size (larger = more data)
                try:
                    size_mb = counts_file.stat().st_size / (1024 * 1024)
                    if size_mb > 10:
                        score += 5.0
                        reasons.append(f"large_counts_file({size_mb:.1f}MB)")
                    elif size_mb > 1:
                        score += 2.0
                        reasons.append(f"medium_counts_file({size_mb:.1f}MB)")
                except Exception:
                    pass
            else:
                reasons.append("missing_counts_file")
            
            if metadata_file:
                metadata_path = Path(metadata_file)
                if metadata_path.exists():
                    score += 15.0
                    reasons.append("metadata_file_exists")
                    
                    # Bonus for metadata completeness
                    try:
                        size_kb = metadata_path.stat().st_size / 1024
                        if size_kb > 100:
                            score += 5.0
                            reasons.append(f"rich_metadata({size_kb:.1f}KB)")
                        elif size_kb > 10:
                            score += 2.0
                            reasons.append("basic_metadata")
                    except Exception:
                        pass
                else:
                    reasons.append("metadata_file_missing")
            else:
                reasons.append("no_metadata_file")
            
            # 2. Disease relevancy (30 points max)
            sample_name_lower = (pair.get("sample_name", "") or "").lower()
            source_dir_lower = (pair.get("source_directory", "") or "").lower()
            
            # Exact disease name match in sample name
            if disease_name_lower and disease_name_lower in sample_name_lower:
                score += 20.0
                reasons.append("disease_name_in_sample")
            elif disease_name_lower:
                # Partial match
                disease_words = disease_name_lower.split()
                matches = sum(1 for word in disease_words if word in sample_name_lower)
                if matches > 0:
                    score += 10.0 * (matches / len(disease_words))
                    reasons.append(f"partial_disease_match({matches}/{len(disease_words)})")
            
            # Disease name in source directory path
            if disease_name_lower and disease_name_lower in source_dir_lower:
                score += 5.0
                reasons.append("disease_in_path")
            
            # User query relevancy (if available)
            if user_query_lower:
                query_words = set(user_query_lower.split())
                sample_words = set(sample_name_lower.split())
                common_words = query_words.intersection(sample_words)
                if common_words:
                    score += 5.0 * min(len(common_words) / 3, 1.0)
                    reasons.append(f"query_relevancy({len(common_words)} words)")
            
            # 3. Dataset quality indicators (20 points max)
            # GSE datasets are typically well-structured
            if sample_name_lower.startswith("gse"):
                score += 10.0
                reasons.append("geo_dataset")
            
            # Pre-processed data (from tar) is ready
            if pair.get("from_tar"):
                score += 5.0
                reasons.append("preprocessed_tar")
            
            # Series matrix indicates GEO metadata
            if metadata_file and "series_matrix" in metadata_file.lower():
                score += 5.0
                reasons.append("geo_series_matrix")
            
            # 4. Source directory quality (10 points max)
            source_dir = Path(pair.get("source_directory", ""))
            if source_dir.exists():
                # Check for multiple files (indicates complete dataset)
                try:
                    file_count = len(list(source_dir.glob("*")))
                    if file_count > 5:
                        score += 5.0
                        reasons.append(f"complete_dataset({file_count} files)")
                    elif file_count > 2:
                        score += 2.0
                        reasons.append(f"partial_dataset({file_count} files)")
                except Exception:
                    pass
            
            ranked_pairs.append({
                **pair,
                "_score": score,
                "_reasons": reasons
            })
        
        # Sort by score (highest first)
        ranked_pairs.sort(key=lambda x: x.get("_score", 0.0), reverse=True)
        
        # Log ranking summary
        self.logger.info(f"🎯 Ranked {len(ranked_pairs)} datasets:")
        for i, ranked_pair in enumerate(ranked_pairs[:5], 1):  # Top 5
            score = ranked_pair.get("_score", 0.0)
            sample_name = ranked_pair.get("sample_name", "unknown")
            reasons_str = ", ".join(ranked_pair.get("_reasons", [])[:3])
            self.logger.info(
                f"  {i}. {sample_name} (score: {score:.1f}) - {reasons_str}"
            )
        
        return ranked_pairs
    
    def _process_datasets(self, dataset_pairs: List[Dict]) -> List[Dict]:
        """
        Process datasets agentically: rank first, then process one at a time.
        Stop on first success.
        
        Args:
            dataset_pairs: List of dataset pairs to process
            
        Returns:
            List of processing results (typically one successful result)
        """
        self.execution_state["current_step"] = "dataset_processing"
        total = len(dataset_pairs)
        self._emit_ui_log("info", f"Agent ranking {total} dataset(s) by quality and relevancy")

        if not dataset_pairs:
            self.logger.warning("⚠️  No datasets to process")
            return []
        
        # Rank datasets by quality and relevancy
        self.logger.info(f"🧠 Ranking {len(dataset_pairs)} datasets for optimal selection...")
        ranked_pairs = self._rank_datasets(dataset_pairs)
        
        results = []
        successful_result = None
        
        # Process datasets one at a time, stopping on first success
        for i, ranked_pair in enumerate(ranked_pairs, 1):
            sample_name = ranked_pair["sample_name"]
            score = ranked_pair.get("_score", 0.0)
            reasons = ranked_pair.get("_reasons", [])
            
            # Remove internal scoring fields
            pair = {k: v for k, v in ranked_pair.items() if not k.startswith("_")}
            
            self.logger.info(
                f"🎯 Attempt {i}/{len(ranked_pairs)}: {sample_name} "
                f"(score: {score:.1f}, reasons: {', '.join(reasons[:3])})"
            )
            self._emit_ui_log(
                "info",
                f"Processing dataset {i}/{len(ranked_pairs)}: {sample_name}",
            )
            
            try:
                result = self._process_single_dataset(pair)
                result["status"] = "success"
                result["ranking_score"] = score
                result["ranking_reasons"] = reasons
                result["attempt_number"] = i
                
                results.append(result)
                successful_result = result
                self.execution_state["successful_datasets"] += 1
                
                self.logger.info(
                    f"✅ Successfully processed {sample_name} on attempt {i}. "
                    f"Skipping remaining {len(ranked_pairs) - i} datasets."
                )
                n_sig = result.get("deseq_result", {}).get("n_significant", 0)
                self._emit_ui_log("info", f"Dataset {sample_name} processed successfully — {n_sig} significant genes")
                
                # Stop on first success
                break
                
            except Exception as e:
                error_msg = str(e)
                self.logger.warning(
                    f"❌ Dataset {sample_name} failed (attempt {i}): {error_msg}"
                )
                self._emit_ui_log("warning", f"Dataset {sample_name} failed, trying next")
                
                result = {
                    "sample_name": sample_name,
                    "work_dir": pair["work_dir"],
                    "status": "failed",
                    "error": error_msg,
                    "ranking_score": score,
                    "ranking_reasons": reasons,
                    "attempt_number": i
                }
                results.append(result)
                
                self.execution_state["failed_datasets"] += 1
                
                # Continue to next dataset
                continue
        
        self.execution_state["completed_steps"].append("dataset_processing")
        
        if successful_result:
            self.logger.info(
                f"✅ Agentic processing complete: Selected best dataset "
                f"({successful_result['sample_name']}) after {len(results)} attempt(s). "
                f"Total datasets available: {len(ranked_pairs)}"
            )
            return results
        else:
            # All datasets failed - raise exception with details
            error_summary = []
            for result in results:
                error_summary.append(
                    f"{result.get('sample_name', 'unknown')}: {result.get('error', 'unknown error')}"
                )
            
            error_msg = (
                f"All {len(ranked_pairs)} ranked datasets failed. "
                f"Errors: {'; '.join(error_summary[:3])}"
            )
            
            self.logger.error(f"❌ {error_msg}")
            raise DEGPipelineError(error_msg)
    
    def _process_single_dataset(self, pair: Dict, _dataset_retry_count: int = 0) -> Dict:
        """
        Process a single dataset pair with retry logic.
        
        Args:
            pair: Dataset pair information
            _dataset_retry_count: Internal retry counter (do not set manually)
            
        Returns:
            Processing result
            
        Raises:
            DEGPipelineError: If processing fails after max retries
        """
        sample_name = pair["sample_name"]
        work_dir = Path(pair["work_dir"])
        prep_dir = work_dir / "prep"
        
        # Max dataset-level retries: 3 attempts total (initial + 2 retries)
        max_dataset_retries = 3
        
        # # Check if this is a shared GSE directory that's already been processed
        # if sample_name.startswith('GSE'):
        #     should_use_shared, shared_path = self.config.should_use_shared_gse(sample_name)
        #     if should_use_shared and shared_path.exists():
        #         shared_prep_dir = shared_path / "prep"
        #         if (shared_prep_dir / "prep_counts.csv").exists():
        #             self.logger.info(f"🔄 Using existing shared GSE analysis: {sample_name}")
        #             # Update work_dir to point to shared location
        #             work_dir = shared_path
        #             prep_dir = shared_prep_dir
        #             pair["work_dir"] = str(work_dir)
        #             return self._load_existing_result(pair)
        
        # # Skip if already processed
        # if (prep_dir / "prep_counts.csv").exists() and not self._should_reprocess(pair):
        #     self.logger.info(f"⏭️ Skipping already processed dataset: {sample_name}")
        #     return self._load_existing_result(pair)
        
        try:
            # Create prep directory
            prep_dir.mkdir(parents=True, exist_ok=True)
            
            # Step 1: Extract metadata
            self._emit_ui_log("info", f"Extracting metadata for {sample_name}")
            metadata_result = self._extract_metadata(pair, prep_dir)
            n_samp = metadata_result.get("n_samples")
            if n_samp is None and metadata_result.get("metadata_df") is not None:
                md = metadata_result["metadata_df"]
                n_samp = len(md) if hasattr(md, "__len__") else getattr(md, "shape", (0,))[0]
            conds = metadata_result.get("conditions") or []
            if n_samp:
                cond_str = f" — conditions: {' vs '.join(str(c) for c in conds[:3])}" if conds else ""
                self._emit_ui_log("info", f"Metadata extracted: {n_samp} samples{cond_str}")

            # Step 2: Validate and load data (with sample alignment check)
            self._emit_ui_log("info", "Validating sample alignment (counts vs metadata)")
            self._validate_processed_data(prep_dir)
            self._emit_ui_log("info", "Sample alignment validated — ready for differential expression")

            # Step 3: Run DEG analysis (tool selected automatically: edgeR, DESeq2, or limma)
            self._emit_ui_log("info", "Running differential expression analysis (agent selecting optimal method)")
            deseq_result = self._run_deseq2_analysis(pair, prep_dir)
            tool_used = deseq_result.get("tool_used", "deseq2")
            n_sig = deseq_result.get("n_significant", 0)
            tool_display = {"edger": "edgeR", "deseq2": "DESeq2", "limma-voom": "limma-voom"}.get(tool_used, tool_used)
            self._emit_ui_log("info", f"Differential expression completed with {tool_display}: {n_sig} significant genes")

            # Step 4: Generate plots
            self._emit_ui_log("info", "Generating visualization plots")
            plot_result = self._generate_plots(pair, prep_dir, work_dir)
            n_plots = plot_result.get("n_plots", 0)
            if n_plots:
                self._emit_ui_log("info", f"Generated {n_plots} visualization plots (volcano, heatmap, MA, etc.)")
            
            # Step 5: Create summary logs
            # self._create_summary_logs(pair, work_dir, metadata_result, deseq_result)
            
            return {
                "sample_name": sample_name,
                "work_dir": str(work_dir),
                "metadata_result": metadata_result,
                "deseq_result": deseq_result,
                "plot_result": plot_result,
                "files_created": self._list_created_files(work_dir)
            }
            
        except Exception as e:
            error_msg = str(e)
            
            # Check if this is a non-recoverable error (index mismatches, data alignment issues)
            is_non_recoverable = (
                "Index are different" in error_msg or
                "index does not match" in error_msg.lower() or
                "sample mapping count mismatch" in error_msg.lower() or
                "no common samples" in error_msg.lower() or
                isinstance(e, NonRecoverableError)
            )
            
            if is_non_recoverable:
                self.logger.error(
                    f"❌ Non-recoverable error for dataset {sample_name} (attempt {_dataset_retry_count + 1}): {error_msg}"
                )
                raise DEGPipelineError(
                    f"Dataset {sample_name} failed with non-recoverable error: {error_msg}"
                ) from e
            
            # Retry for recoverable errors
            if _dataset_retry_count < max_dataset_retries - 1:
                self.logger.warning(
                    f"⚠️ Dataset {sample_name} failed (attempt {_dataset_retry_count + 1}/{max_dataset_retries}): {error_msg}"
                )
                self.logger.info(f"🔄 Retrying dataset {sample_name}...")
                
                # Attempt dataset recovery if enabled
                if self.config.enable_auto_fix:
                    try:
                        self._attempt_dataset_recovery(pair, e)
                    except Exception as recovery_error:
                        self.logger.warning(f"⚠️ Dataset recovery attempt failed: {recovery_error}")
                
                # Retry with incremented counter
                return self._process_single_dataset(pair, _dataset_retry_count=_dataset_retry_count + 1)
            else:
                self.logger.error(
                    f"❌ Dataset {sample_name} failed after {max_dataset_retries} attempts: {error_msg}"
                )
                raise DEGPipelineError(
                    f"Dataset {sample_name} failed after {max_dataset_retries} attempts: {error_msg}"
                ) from e
    
    def _extract_metadata(self, pair: Dict, prep_dir: Path) -> Dict:
        """
        Extract metadata for a dataset pair.
        
        Args:
            pair: Dataset pair information
            prep_dir: Preparation directory
            
        Returns:
            Metadata extraction result
        """
        metadata_file = pair.get("metadata_file")
        
        # If metadata_file is None (e.g., from tar processing), try to find it in source directory
        if metadata_file is None:
            source_dir = Path(pair.get("source_directory", ""))
            if source_dir.exists():
                # Look for series_matrix or metadata files
                metadata_files = list(source_dir.glob("*series_matrix*"))
                if not metadata_files:
                    metadata_files = list(source_dir.glob("*metadata*"))
                if not metadata_files:
                    metadata_files = list(source_dir.glob("*meta*"))
                
                if metadata_files:
                    metadata_file = str(metadata_files[0])
                    self.logger.info(f"📋 Found metadata file in source directory: {metadata_files[0].name}")
                else:
                    # If still no metadata, create a basic one from sample IDs in counts
                    self.logger.warning(f"⚠️  No metadata file found. Creating basic metadata from counts file.")
                    metadata_file = self._create_basic_metadata(pair["counts_file"], prep_dir)
        
        if metadata_file is None:
            raise DEGPipelineError(f"Cannot proceed without metadata file for {pair['sample_name']}")
        
        return self.tools["metadata_extractor"].safe_execute(
            counts_file=pair["counts_file"],
            metadata_file=metadata_file,
            output_dir=str(prep_dir)
        )
    
    def _create_basic_metadata(self, counts_file: str, prep_dir: Path) -> str:
        """
        Create a basic metadata file from sample IDs in counts file.
        
        Supports: CSV, TSV, TXT, XLSX, XLS formats
        
        Args:
            counts_file: Path to counts file
            prep_dir: Preparation directory
            
        Returns:
            Path to created metadata file
        """
        
        # Load counts to get sample IDs (support Excel formats)
        counts_path = Path(counts_file)
        file_ext = counts_path.suffix.lower()
        
        if file_ext in ['.xlsx', '.xls']:
            try:
                counts_df = pd.read_excel(counts_file, index_col=0, engine='openpyxl' if file_ext == '.xlsx' else None)
            except Exception:
                # Fallback: try without index_col
                counts_df = pd.read_excel(counts_file, index_col=None, engine='openpyxl' if file_ext == '.xlsx' else None)
                if counts_df.shape[1] > 1:
                    counts_df = counts_df.set_index(counts_df.columns[0])
        else:
            counts_df = pd.read_csv(counts_file, index_col=0)
        
        sample_ids = list(counts_df.columns)
        
        # Create basic metadata with all samples as "Unknown" condition
        metadata_df = pd.DataFrame({
            'sample_id': sample_ids,
            'condition': ['Unknown'] * len(sample_ids)
        })
        
        metadata_file = prep_dir / "basic_metadata.csv"
        metadata_df.to_csv(metadata_file, index=False)
        
        self.logger.warning(f"⚠️  Created basic metadata file. You may need to manually edit: {metadata_file}")
        return str(metadata_file)
    
    def _validate_processed_data(self, prep_dir: Path) -> None:
        """
        Validate processed data files and check sample alignment.
        
        Args:
            prep_dir: Preparation directory
            
        Raises:
            DEGPipelineError: If validation fails
            NonRecoverableError: If sample alignment fails (non-recoverable)
        """
        counts_file = prep_dir / "prep_counts.csv"
        metadata_file = prep_dir / "prep_meta.csv"
        
        if not counts_file.exists():
            raise DEGPipelineError("Processed counts file not found")
        
        if not metadata_file.exists():
            raise DEGPipelineError("Processed metadata file not found")
        
        # Validate sample alignment before proceeding
        try:
            counts_df = pd.read_csv(counts_file, index_col=0)
            metadata_df = pd.read_csv(metadata_file, index_col=0)
            
            # Check if metadata has 'sample' column (not index)
            if 'sample' in metadata_df.columns:
                metadata_samples = set(metadata_df['sample'].astype(str))
            else:
                metadata_samples = set(metadata_df.index.astype(str))
            
            counts_samples = set(counts_df.columns.astype(str))
            common_samples = counts_samples.intersection(metadata_samples)
            
            if not common_samples:
                raise NonRecoverableError(
                    f"No common samples found between counts ({len(counts_samples)} samples) "
                    f"and metadata ({len(metadata_samples)} samples). "
                    f"This indicates a data alignment issue that cannot be automatically fixed."
                )
            
            if len(common_samples) < 2:
                raise NonRecoverableError(
                    f"Insufficient common samples ({len(common_samples)}) for DEG analysis. "
                    f"Need at least 2 samples. Counts: {len(counts_samples)}, Metadata: {len(metadata_samples)}"
                )
            
            # Check for significant mismatch (>20% missing samples)
            missing_in_metadata = counts_samples - metadata_samples
            missing_in_counts = metadata_samples - counts_samples
            
            if len(missing_in_metadata) > len(counts_samples) * 0.2:
                self.logger.warning(
                    f"⚠️ Significant sample mismatch: {len(missing_in_metadata)}/{len(counts_samples)} "
                    f"({len(missing_in_metadata)/len(counts_samples)*100:.1f}%) samples in counts missing from metadata"
                )
                # If >50% missing, treat as non-recoverable
                if len(missing_in_metadata) > len(counts_samples) * 0.5:
                    raise NonRecoverableError(
                        f"Critical sample mismatch: {len(missing_in_metadata)}/{len(counts_samples)} "
                        f"({len(missing_in_metadata)/len(counts_samples)*100:.1f}%) samples in counts missing from metadata. "
                        f"Missing samples: {list(missing_in_metadata)[:10]}"
                    )
            
            if len(missing_in_counts) > len(metadata_samples) * 0.2:
                self.logger.warning(
                    f"⚠️ {len(missing_in_counts)}/{len(metadata_samples)} samples in metadata missing from counts"
                )
            
            self.logger.info(
                f"✅ Sample alignment validated: {len(common_samples)} common samples "
                f"(counts: {len(counts_samples)}, metadata: {len(metadata_samples)})"
            )
            
        except NonRecoverableError:
            raise
        except Exception as e:
            raise DEGPipelineError(f"Failed to validate sample alignment: {e}") from e
        
        # Validate with file validator tool
        # self.tools["file_validator"].safe_execute(
        #     counts_file=str(counts_file),
        #     metadata_file=str(metadata_file)
        # )
    
    def _run_deseq2_analysis(self, pair: Dict, prep_dir: Path) -> Dict:
        """
        Run DESeq2 analysis.
        
        Args:
            pair: Dataset pair information
            prep_dir: Preparation directory
            
        Returns:
            DESeq2 analysis result
        """
        work_dir = Path(pair["work_dir"])
        output_file = work_dir / f"{pair['sample_name']}_DEGs.csv"
        
        return self.tools["deseq2_analyzer"].safe_execute(
            counts_file=str(prep_dir / "prep_counts.csv"),
            metadata_file=str(prep_dir / "prep_meta.csv"),
            output_file=str(output_file)
        )
    
    def _generate_plots(self, pair: Dict, prep_dir: Path, work_dir: Path) -> Dict:
        """
        Generate visualization plots from DEG analysis results.
        
        Args:
            pair: Dataset pair information
            prep_dir: Preparation directory containing counts and metadata
            work_dir: Work directory containing DEG results
            
        Returns:
            Plot generation result
        """
        deg_file = work_dir / f"{pair['sample_name']}_DEGs.csv"
        
        if not deg_file.exists():
            self.logger.warning(f"⚠️ DEG file not found, skipping plot generation: {deg_file}")
            return {
                "status": "skipped",
                "reason": "DEG file not found",
                "plots_created": []
            }
        
        # Create plots directory in the same location as DEGs.csv (work_dir/plots/)
        plots_output_dir = work_dir
        
        try:
            plot_result = self.tools["deg_plotter"].safe_execute(
                deg_file=str(deg_file),
                counts_file=str(prep_dir / "prep_counts.csv"),
                metadata_file=str(prep_dir / "prep_meta.csv"),
                output_dir=str(plots_output_dir)
            )
            self.logger.info(f"✅ Generated {plot_result.get('n_plots', 0)} plots")
            return plot_result
        except Exception as e:
            self.logger.warning(f"⚠️ Plot generation failed: {e}")
            return {
                "status": "failed",
                "error": str(e),
                "plots_created": []
            }
    
    def _create_summary_logs(self, pair: Dict, work_dir: Path, metadata_result: Dict, 
                           deseq_result: Dict) -> None:
        """
        Create summary log files.
        
        Args:
            pair: Dataset pair information
            work_dir: Work directory
            metadata_result: Metadata extraction result
            deseq_result: DESeq2 analysis result
        """
        # Data summary
        data_summary = f"Samples: {metadata_result['n_samples']}, Genes: {deseq_result.get('n_genes', 'unknown')}"
        (work_dir / "data_summary.log").write_text(data_summary + "\n")
        
        # Processing decision
        method = metadata_result.get("method", "unknown")
        decision = "GEO metadata extracted" if "geo" in method else "Analysis transcriptome metadata used"
        (work_dir / "decision.log").write_text(decision + "\n")
        
        # DESeq2 summary
        comparisons = deseq_result.get("comparisons", [])
        n_significant = deseq_result.get("n_significant", 0)
        deseq_summary = f"DESeq2: comparisons={comparisons}, n_sig={n_significant}"
        (work_dir / "deseq2_summary.log").write_text(deseq_summary + "\n")
    
    def _should_reprocess(self, pair: Dict) -> bool:
        """
        Determine if a dataset should be reprocessed.
        
        Args:
            pair: Dataset pair information
            
        Returns:
            True if should reprocess
        """
        # For now, always reprocess if explicitly requested
        # This could be enhanced with timestamp checking, etc.
        return False
    
    def _load_existing_result(self, pair: Dict) -> Dict:
        """
        Load existing processing result.
        
        Args:
            pair: Dataset pair information
            
        Returns:
            Existing result
        """
        return {
            "sample_name": pair["sample_name"],
            "work_dir": pair["work_dir"],
            "status": "skipped_existing",
            "message": "Already processed"
        }
    
    def _list_created_files(self, work_dir: Path) -> List[str]:
        """
        List files created during processing.
        
        Args:
            work_dir: Work directory
            
        Returns:
            List of created file paths
        """
        created_files = []
        for file_path in work_dir.rglob("*"):
            if file_path.is_file():
                created_files.append(str(file_path.relative_to(work_dir)))
        return created_files
    
    def _generate_summary(self, results: List[Dict]) -> Dict:
        """
        Generate execution summary.
        
        Args:
            results: Processing results
            
        Returns:
            Summary dictionary
        """
        successful = [r for r in results if r.get("status") == "success"]
        failed = [r for r in results if r.get("status") == "failed"]
        skipped = [r for r in results if r.get("status") == "skipped_existing"]
        
        return {
            "total_datasets": len(results),
            "successful": len(successful),
            "failed": len(failed),
            "skipped": len(skipped),
            "success_rate": len(successful) / len(results) if results else 0,
            "failed_datasets": [r["sample_name"] for r in failed],
            "successful_datasets": [r["sample_name"] for r in successful],
            "execution_state": self.execution_state
        }
    
    def _attempt_pipeline_recovery(self, error: Exception) -> None:
        """
        Attempt to recover from pipeline-level errors.
        
        Args:
            error: Error that occurred
            
        Raises:
            DEGPipelineError: If recovery is not possible
        """
        self.logger.info(f"🔧 Attempting pipeline recovery from: {error}")
        
        try:
            self.tools["error_fixer"].safe_execute(
                error_type="pipeline_error",
                error_message=str(error),
                context=self.execution_state
            )
        except Exception as e:
            raise DEGPipelineError(f"Pipeline recovery failed: {e}")
    
    def _attempt_dataset_recovery(self, pair: Dict, error: Exception) -> None:
        """
        Attempt to recover from dataset-specific errors.
        
        Args:
            pair: Dataset pair information
            error: Error that occurred
            
        Raises:
            DEGPipelineError: If recovery is not possible
        """
        self.logger.info(f"🔧 Attempting dataset recovery for {pair['sample_name']}: {error}")
        
        try:
            self.tools["error_fixer"].safe_execute(
                error_type="dataset_error",
                error_message=str(error),
                context=pair
            )
        except Exception as e:
            self.logger.warning(f"Dataset recovery failed: {e}")
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """
        Get current pipeline status.
        
        Returns:
            Status dictionary
        """
        return {
            "agent_name": self.config.agent_name,
            "current_step": self.execution_state["current_step"],
            "completed_steps": self.execution_state["completed_steps"],
            "failed_steps": self.execution_state["failed_steps"],
            "total_datasets": self.execution_state["total_datasets"],
            "successful_datasets": self.execution_state["successful_datasets"],
            "failed_datasets": self.execution_state["failed_datasets"],
            "tool_stats": {name: tool.get_stats() for name, tool in self.tools.items()},
            "config": self.config._to_dict()
        }
    
    def reset_pipeline(self) -> None:
        """Reset pipeline state for fresh execution."""
        self.execution_state = {
            "current_step": None,
            "completed_steps": [],
            "failed_steps": [],
            "start_time": None,
            "end_time": None,
            "total_datasets": 0,
            "successful_datasets": 0,
            "failed_datasets": 0,
            "results": []
        }
        
        # Reset tool statistics
        for tool in self.tools.values():
            tool.reset_stats()
        
        self.logger.info("🔄 Pipeline state reset")
    
    def __str__(self) -> str:
        return f"{self.config.agent_name} (DEG Pipeline Agent)"
    
    def __repr__(self) -> str:
        return f"<DEGPipelineAgent: {self.config.disease_name or 'No disease set'}>" 