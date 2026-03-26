# src/pathway_prioritization/models/config_models.py
from dataclasses import dataclass
from typing import Dict, Any
from collections import defaultdict

@dataclass
class WorkerStats:
    processed: int = 0
    failed: int = 0
    total_time: float = 0.0

@dataclass
class ParallelProcessingStats:
    """Statistics for parallel processing monitoring"""
    total_pathways: int = 0
    processed_pathways: int = 0
    failed_pathways: int = 0
    active_workers: int = 0
    completed_workers: int = 0
    average_processing_time: float = 0.0
    start_time: float = 0.0
    worker_stats: Dict[str, Dict] = None
    
    def __post_init__(self):
        if self.worker_stats is None:
            self.worker_stats = defaultdict(WorkerStats)

@dataclass
class ProgressReport:
    total_pathways: int
    processed: int
    failed: int
    progress_percentage: float
    success_rate: float
    average_processing_time: float
    elapsed_time: float
    active_workers: int