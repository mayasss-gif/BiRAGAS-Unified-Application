"""
Agent State Management for LangGraph Pathway Analysis Workflow

Defines the shared state that flows through all pipeline stages with
comprehensive tracking for progress, errors, and retry logic.
"""

from typing import TypedDict, Optional, List, Dict, Any
from pathlib import Path


class PipelineState(TypedDict):
    """Shared state object that flows through all pipeline stages"""
    
    # Input parameters
    user_query: str
    deg_file_path: Optional[Path]
    disease_name: Optional[str]
    patient_prefix: Optional[str]
    output_dir: Optional[Path]
    causal: bool
    
    # Stage outputs
    enrichment_output: Optional[Path]
    deduplication_output: Optional[Path]
    categorization_output: Optional[Path]
    literature_output: Optional[Path]
    consolidation_output: Optional[Path]
    
    # Error tracking and retry
    errors: List[Dict[str, Any]]
    retry_counts: Dict[str, int]
    max_retries: int
    
    # Progress tracking
    current_stage: str
    completed_stages: List[str]
    progress_messages: List[str]
    
    # Additional metadata
    start_time: Optional[str]
    end_time: Optional[str]
    total_processing_time: Optional[float]

    # Run-scoped UI logging context (set by agent_runner; must flow to all nodes)
    _pathway_ui_ctx_key: Optional[str]
