# src/pathway_prioritization/utils/progress_tracker.py
import time
import threading
from typing import Dict, Any
from threading import Lock, RLock
from collections import defaultdict

from ..models import ParallelProcessingStats, ProgressReport

class ThreadSafeProgressTracker:
    """Thread-safe progress tracking for parallel pathway processing"""
    
    def __init__(self, total_pathways: int):
        self.stats = ParallelProcessingStats(total_pathways=total_pathways)
        self.stats.start_time = time.time()
        self._lock = RLock()
        self._processed_pathways = []
        self._failed_pathways = []
        
    def update_worker_stats(self, worker_id: str, pathway_name: str, processing_time: float, success: bool):
        """Update statistics for a worker"""
        with self._lock:
            if worker_id not in self.stats.worker_stats:
                self.stats.worker_stats[worker_id] = {
                    'processed': 0,
                    'failed': 0, 
                    'total_time': 0.0,
                    'pathways': []
                }
            
            self.stats.worker_stats[worker_id]['pathways'].append({
                'name': pathway_name,
                'time': processing_time,
                'success': success
            })
            
            if success:
                self.stats.worker_stats[worker_id]['processed'] += 1
                self.stats.processed_pathways += 1
                self._processed_pathways.append(pathway_name)
            else:
                self.stats.worker_stats[worker_id]['failed'] += 1
                self.stats.failed_pathways += 1
                self._failed_pathways.append(pathway_name)
                
            self.stats.worker_stats[worker_id]['total_time'] += processing_time
            
            # Update average processing time
            total_processed = self.stats.processed_pathways + self.stats.failed_pathways
            if total_processed > 0:
                total_time = sum(worker['total_time'] for worker in self.stats.worker_stats.values())
                self.stats.average_processing_time = total_time / total_processed
    
    def get_progress_report(self) -> ProgressReport:
        """Get current progress report"""
        with self._lock:
            elapsed_time = time.time() - self.stats.start_time
            total_processed = self.stats.processed_pathways + self.stats.failed_pathways
            progress_percentage = (total_processed / self.stats.total_pathways * 100) if self.stats.total_pathways > 0 else 0
            
            success_rate = (self.stats.processed_pathways / total_processed * 100) if total_processed > 0 else 0
            
            return ProgressReport(
                total_pathways=self.stats.total_pathways,
                processed=self.stats.processed_pathways,
                failed=self.stats.failed_pathways,
                progress_percentage=round(progress_percentage, 1),
                success_rate=round(success_rate, 1),
                average_processing_time=round(self.stats.average_processing_time, 2),
                elapsed_time=round(elapsed_time, 1),
                active_workers=len([w for w in self.stats.worker_stats.values() if len(w['pathways']) > 0])
            )