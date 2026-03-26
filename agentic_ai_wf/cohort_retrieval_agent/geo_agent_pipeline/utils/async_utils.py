"""
Async utilities for the Cohort Retrieval Agent system.
"""

import asyncio
import time
from typing import Any, Callable, Dict, List, Optional
from dataclasses import dataclass


@dataclass
class ProgressUpdate:
    """Progress update information."""
    current: int
    total: int
    percentage: float
    message: str
    elapsed_time: float
    estimated_total_time: Optional[float] = None


class AsyncProgressTracker:
    """
    Async progress tracker for long-running operations.
    """
    
    def __init__(self, total: int, update_interval: float = 1.0):
        self.total = total
        self.current = 0
        self.update_interval = update_interval
        self.start_time = time.time()
        self.last_update_time = 0
        self.callbacks = []
        self.message = ""
    
    def add_callback(self, callback: Callable[[ProgressUpdate], None]):
        """Add a progress callback."""
        self.callbacks.append(callback)
    
    def update(self, increment: int = 1, message: str = ""):
        """Update progress."""
        self.current = min(self.current + increment, self.total)
        if message:
            self.message = message
        
        # Check if we should send update
        current_time = time.time()
        if current_time - self.last_update_time >= self.update_interval or self.current >= self.total:
            self._send_update()
            self.last_update_time = current_time
    
    def _send_update(self):
        """Send progress update to callbacks."""
        elapsed_time = time.time() - self.start_time
        percentage = (self.current / self.total) * 100 if self.total > 0 else 0
        
        # Estimate total time
        estimated_total_time = None
        if self.current > 0 and percentage > 0:
            estimated_total_time = elapsed_time / (percentage / 100)
        
        update = ProgressUpdate(
            current=self.current,
            total=self.total,
            percentage=percentage,
            message=self.message,
            elapsed_time=elapsed_time,
            estimated_total_time=estimated_total_time
        )
        
        for callback in self.callbacks:
            try:
                callback(update)
            except Exception:
                pass  # Don't let callback errors break progress tracking


async def run_with_progress(tasks: List[asyncio.Task], 
                          progress_callback: Optional[Callable[[ProgressUpdate], None]] = None,
                          update_interval: float = 1.0) -> List[Any]:
    """
    Run async tasks with progress tracking.
    
    Args:
        tasks: List of async tasks
        progress_callback: Optional progress callback
        update_interval: Update interval in seconds
        
    Returns:
        List of task results
    """
    tracker = AsyncProgressTracker(len(tasks), update_interval)
    if progress_callback:
        tracker.add_callback(progress_callback)
    
    results = []
    for i, task in enumerate(tasks):
        tracker.update(0, f"Running task {i+1}/{len(tasks)}")
        result = await task
        results.append(result)
        tracker.update(1, f"Completed task {i+1}/{len(tasks)}")
    
    return results 