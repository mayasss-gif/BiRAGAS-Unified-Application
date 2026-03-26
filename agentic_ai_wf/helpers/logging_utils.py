"""
Agentic AI Logging Utilities

This module provides convenient logging functions that can be used throughout
the project to create structured log entries that are saved to the database
and broadcasted to real-time channels.

Usage Examples:
    # From Celery tasks
    log_workflow_step(
        agent_name="Data Retrieval Agent",
        message="Starting patient data retrieval",
        user_id=str(user.id),
        analysis_id=str(analysis.id),
        step="data_retrieval",
        progress=10
    )
    
    # From coordinators with error handling
    log_error(
        agent_name="Gene Prioritization Agent",
        message="Failed to process gene data",
        user_id=str(user.id),
        analysis_id=str(analysis.id),
        error_code="GENE_PROC_001",
        error_message=str(e),
        traceback=traceback.format_exc()
    )
"""

import logging
import traceback
import uuid
from typing import Optional, Dict, Any, List
from datetime import datetime, timezone

# Import the synchronous logging function and create async wrapper
try:
    from fastapi_app.utils import create_and_broadcast_log_sync
    from agenticaib.db_pool import sync_to_async_with_cleanup
    
    # FIXED: Use sync_to_async_with_cleanup to prevent connection leaks
    # This is critical for long-running workflows that create many log entries
    create_and_broadcast_log_async = sync_to_async_with_cleanup(create_and_broadcast_log_sync)
except ImportError:
    # Fallback for environments where FastAPI utils aren't available
    try:
        from asgiref.sync import sync_to_async
        from django.db import close_old_connections
        
        # Fallback wrapper that closes connections
        async def create_and_broadcast_log_async(*args, **kwargs):
            try:
                result = await sync_to_async(create_and_broadcast_log_sync, thread_sensitive=True)(*args, **kwargs)
                close_old_connections()
                return result
            except Exception as e:
                close_old_connections()
                raise
    except ImportError:
        create_and_broadcast_log_sync = None
        create_and_broadcast_log_async = None

logger = logging.getLogger(__name__)

import asyncio

# Global guards to suppress any logs after terminal completion
_FINALIZED_CORRELATION_IDS = set()
_FINALIZED_ANALYSIS_IDS = set()

def _is_async_context():
    """Check if we're currently in an async context"""
    try:
        asyncio.current_task()
        return True
    except RuntimeError:
        return False

async def _log_async(
    agent_name: str,
    message: str,
    user_id: str,
    analysis_id: str,
    log_level: str = "INFO",
    status: str = "running",
    step: str = "",
    step_index: int = 0,
    progress: int = 0,
    current_step: int = 0,
    total_steps: int = 6,
    task_id: str = "",
    enable_llm_enhancement: bool = True,
    **kwargs
) -> Optional[str]:
    # Drop any logs emitted after terminal completion for same workflow
    try:
        is_terminal = kwargs.get("status") == "completed" and kwargs.get("step") == "workflow_complete"
        corr_id = kwargs.get("correlation_id")
        if not is_terminal:
            # Instance-level guard
            caller = kwargs.get("caller_instance")
            if getattr(caller, "_finalized", False):
                return None
            # Correlation/analysis guards
            if corr_id and corr_id in _FINALIZED_CORRELATION_IDS:
                return None
            if analysis_id in _FINALIZED_ANALYSIS_IDS:
                return None
    except Exception:
        pass
    """Async logging function with optional LLM enhancement"""
    if not create_and_broadcast_log_async:
        logger.warning("Async logging function not available - using fallback")
        return None
    
    # Send original log immediately (no blocking)
    log_id = await create_and_broadcast_log_async(
        agent_name=agent_name,
        log_message=message,
        user_id=user_id,
        analysis_id=analysis_id,
        log_level=log_level,
        status=status,
        step=step,
        step_index=step_index,
        progress_percentage=progress,
        current_step=current_step,
        total_steps=total_steps,
        task_id=task_id,
        source="workflow",
        **kwargs
    )
    
    # Schedule LLM enhancement in background (fire-and-forget)
    if enable_llm_enhancement and log_id:
        try:
            from agentic_ai_wf.helpers.log_enhancer import schedule_log_enhancement
            from agentic_ai_wf.helpers.enhancement_config import should_enhance_log
            
            # Check if this log should be enhanced (based on config)
            if should_enhance_log(step=step, agent_name=agent_name, log_level=log_level):
                # Build log data for enhancement
                log_data = {
                    "agent_name": agent_name,
                    "log_message": message,
                    "user_id": user_id,
                    "analysis_id": analysis_id,
                    "log_level": log_level,
                    "status": status,
                    "step": step,
                    "current_step": current_step,
                    "total_steps": total_steps,
                    **kwargs
                }
                
                # Fire-and-forget enhancement task
                schedule_log_enhancement(
                    log_id=log_id,
                    log_data=log_data,
                    update_callback=update_log_with_enhancement
                )
        except ImportError:
            logger.debug("Log enhancer not available")
        except Exception as e:
            logger.debug(f"Enhancement scheduling failed: {e}")
            # Graceful degradation - original log already sent
    
    return log_id

def _log_sync(
    agent_name: str,
    message: str,
    user_id: str,
    analysis_id: str,
    log_level: str = "INFO",
    status: str = "running",
    step: str = "",
    step_index: int = 0,
    progress: int = 0,
    current_step: int = 0,
    total_steps: int = 6,
    task_id: str = "",
    **kwargs
) -> Optional[str]:
    """Sync logging function"""
    if not create_and_broadcast_log_sync:
        logger.warning("Sync logging function not available - using fallback")
        return None
    
    # Drop any logs emitted after terminal completion for same workflow
    try:
        # status/step are provided by caller; guard non-terminal logs
        is_terminal = (locals().get('status') == "completed" and locals().get('step') == "workflow_complete")
        if not is_terminal:
            corr_id = locals().get('kwargs', {}).get('correlation_id')
            if corr_id and corr_id in _FINALIZED_CORRELATION_IDS:
                return None
            if analysis_id in _FINALIZED_ANALYSIS_IDS:
                return None
    except Exception:
        pass

    return create_and_broadcast_log_sync(
        agent_name=agent_name,
        log_message=message,
        user_id=user_id,
        analysis_id=analysis_id,
        log_level=log_level,
        status=status,
        step=step,
        step_index=step_index,
        progress_percentage=progress,
        current_step=current_step,
        total_steps=total_steps,
        task_id=task_id,
        source="workflow",
        **kwargs
    )

def log_workflow_step(
    agent_name: str,
    message: str,
    user_id: str,
    analysis_id: str,
    step: str,
    step_index: int = 0,
    progress: int = 0,
    current_step: int = 0,
    total_steps: int = 6,  # Default for transcriptome analysis workflow
    log_level: str = "INFO",
    task_id: str = "",
    **kwargs
) -> Optional[str]:
    """
    Context-aware workflow step logging.
    Automatically detects if called from async context and routes appropriately.
    
    Args:
        agent_name: Name of the agent performing the step
        message: Descriptive message about the step
        user_id: User identifier
        analysis_id: Analysis identifier
        step: Step identifier (e.g., "data_retrieval", "gene_prioritization")
        step_index: Sequential index of the step (0-based)
        progress: Progress percentage (0-100)
        current_step: Current step number (1-based)
        total_steps: Total number of steps in the workflow
        log_level: Log level (INFO, WARNING, ERROR, etc.)
        task_id: Celery task ID if applicable
        **kwargs: Additional fields
        
    Returns:
        Log message ID if successful, None otherwise
    """
    if _is_async_context():
        # We're in an async context but called a sync function
        # Fall back to basic logging to avoid database sync issues
        logger.info(f"[{agent_name}] {message} (step: {step})")
        logger.warning("Sync logging called from async context - using fallback logging")
        return None
    
    # Remove status from kwargs to avoid conflict with explicit status parameter
    clean_kwargs = kwargs.copy()
    status = clean_kwargs.pop("status", "running")  # Use status from kwargs if provided, otherwise "running"
    
    # Use sync version for synchronous contexts
    return _log_sync(
        agent_name=agent_name,
        message=message,
        user_id=user_id,
        analysis_id=analysis_id,
        log_level=log_level,
        status=status,
        step=step,
        step_index=step_index,
        progress=progress,
        current_step=current_step,
        total_steps=total_steps,
        task_id=task_id,
        **clean_kwargs
    )

async def log_workflow_step_async(
    agent_name: str,
    message: str,
    user_id: str,
    analysis_id: str,
    step: str,
    step_index: int = 0,
    progress: int = 0,
    current_step: int = 0,
    total_steps: int = 6,  # Default for transcriptome analysis workflow
    log_level: str = "INFO",
    task_id: str = "",
    **kwargs
) -> Optional[str]:
    """
    Async version of log_workflow_step for use in async contexts.
    
    Args:
        Same as log_workflow_step
        
    Returns:
        Log message ID if successful, None otherwise
    """
    # Remove status from kwargs to avoid conflict with explicit status parameter
    clean_kwargs = kwargs.copy()
    status = clean_kwargs.pop("status", "running")  # Use status from kwargs if provided, otherwise "running"
    
    return await _log_async(
        agent_name=agent_name,
        message=message,
        user_id=user_id,
        analysis_id=analysis_id,
        log_level=log_level,
        status=status,
        step=step,
        step_index=step_index,
        progress=progress,
        current_step=current_step,
        total_steps=total_steps,
        task_id=task_id,
        **clean_kwargs
    )

def log_step_start(
    agent_name: str,
    step: str,
    user_id: str,
    analysis_id: str,
    step_index: int = 0,
    current_step: int = 0,
    total_steps: int = 6,
    task_id: str = "",
    **kwargs
) -> Optional[str]:
    """
    Log the start of a workflow step.
    
    Args:
        agent_name: Name of the agent
        step: Step identifier
        user_id: User identifier
        analysis_id: Analysis identifier
        step_index: Sequential index of the step
        current_step: Current step number
        total_steps: Total number of steps
        task_id: Celery task ID if applicable
        **kwargs: Additional fields
        
    Returns:
        Log message ID if successful, None otherwise
    """
    progress = int((current_step / total_steps) * 100) if current_step and total_steps else None
    
    return log_workflow_step(
        agent_name=agent_name,
        message=f"Starting {step.replace('_', ' ').title()}",
        user_id=user_id,
        analysis_id=analysis_id,
        step=step,
        step_index=step_index,
        progress=int(progress) if progress else 0,
        current_step=current_step,
        total_steps=total_steps,
        log_level="INFO",
        task_id=task_id,
        **kwargs
    )

async def log_step_start_async(
    agent_name: str,
    step: str,
    user_id: str,
    analysis_id: str,
    step_index: int = 0,
    current_step: int = 0,
    total_steps: int = 6,
    task_id: str = "",
    **kwargs
) -> Optional[str]:
    """
    Async version of log_step_start.
    """
    progress = int((current_step / total_steps) * 100) if current_step and total_steps else None
    
    return await log_workflow_step_async(
        agent_name=agent_name,
        message=f"Starting {step.replace('_', ' ').title()}",
        user_id=user_id,
        analysis_id=analysis_id,
        step=step,
        step_index=step_index,
        progress=int(progress) if progress else 0,
        current_step=current_step,
        total_steps=total_steps,
        log_level="INFO",
        task_id=task_id,
        **kwargs
    )

def log_step_complete(
    agent_name: str,
    step: str,
    user_id: str,
    analysis_id: str,
    step_index: int = 0,
    current_step: int = 0,
    total_steps: int = 6,
    elapsed_time_ms: int = 0,
    task_id: str = "",
    **kwargs
) -> Optional[str]:
    """
    Log the completion of a workflow step.
    
    Args:
        agent_name: Name of the agent
        step: Step identifier
        user_id: User identifier
        analysis_id: Analysis identifier
        step_index: Sequential index of the step
        current_step: Current step number
        total_steps: Total number of steps
        elapsed_time_ms: Time taken for the step in milliseconds
        task_id: Celery task ID if applicable
        **kwargs: Additional fields
        
    Returns:
        Log message ID if successful, None otherwise
    """
    progress = int((current_step / total_steps) * 100) if current_step and total_steps else None
    
    # Remove status from kwargs to avoid conflict with explicit status="completed"
    clean_kwargs = kwargs.copy()
    clean_kwargs.pop("status", None)
    
    return log_workflow_step(
        agent_name=agent_name,
        message=f"Completed {step.replace('_', ' ').title()}",
        user_id=user_id,
        analysis_id=analysis_id,
        step=step,
        step_index=step_index,
        progress=int(progress) if progress else 0,
        current_step=current_step,
        total_steps=total_steps,
        log_level="INFO",
        status="completed",
        elapsed_time_ms=elapsed_time_ms,
        task_id=task_id,
        **clean_kwargs
    )

def log_error(
    agent_name: str,
    message: str,
    user_id: str,
    analysis_id: str,
    error_code: str = "",
    error_message: str = "",
    step: str = "",
    task_id: str = "",
    exception: Exception = Exception(),
    **kwargs
) -> Optional[str]:
    """
    Log an error with detailed error information.
    
    Args:
        agent_name: Name of the agent where the error occurred
        message: Descriptive error message
        user_id: User identifier
        analysis_id: Analysis identifier
        error_code: Standardized error code
        error_message: Detailed error message
        step: Step where the error occurred
        task_id: Celery task ID if applicable
        exception: Exception object for automatic traceback extraction
        **kwargs: Additional fields
        
    Returns:
        Log message ID if successful, None otherwise
    """
    # Extract traceback if exception is provided
    error_data = kwargs.copy()
    if exception:
        error_data["error_traceback"] = traceback.format_exc()
        if not error_message:
            error_message = str(exception)
    
    # Remove status from error_data to avoid conflict with explicit status="failed"
    error_data.pop("status", None)
    
    return _log_sync(
        agent_name=agent_name,
        message=message,
        user_id=user_id,
        analysis_id=analysis_id,
        log_level="ERROR",
        status="failed",
        step=step,
        task_id=task_id,
        error_code=error_code,
        error_message=error_message,
        **error_data
    )

async def log_error_async(
    agent_name: str,
    message: str,
    user_id: str,
    analysis_id: str,
    error_code: str = "",
    error_message: str = "",
    step: str = "",
    task_id: str = "",
    exception: Exception = Exception(),
    **kwargs
) -> Optional[str]:
    """
    Async version of log_error.
    """
    # Extract traceback if exception is provided
    error_data = kwargs.copy()
    if exception:
        error_data["error_traceback"] = traceback.format_exc()
        if not error_message:
            error_message = str(exception)
    
    # Remove status from error_data to avoid conflict with explicit status="failed"
    error_data.pop("status", None)
    
    return await _log_async(
        agent_name=agent_name,
        message=message,
        user_id=user_id,
        analysis_id=analysis_id,
        log_level="ERROR",
        status="failed",
        step=step,
        task_id=task_id,
        error_code=error_code,
        error_message=error_message,
        **error_data
    )

def log_warning(
    agent_name: str,
    message: str,
    user_id: str,
    analysis_id: str,
    step: str = "",
    task_id: str = "",
    **kwargs
) -> Optional[str]:
    """
    Log a warning message.
    
    Args:
        agent_name: Name of the agent
        message: Warning message
        user_id: User identifier
        analysis_id: Analysis identifier
        step: Step where the warning occurred
        task_id: Celery task ID if applicable
        **kwargs: Additional fields
        
    Returns:
        Log message ID if successful, None otherwise
    """
    if not create_and_broadcast_log_sync:
        logger.warning("Logging function not available - using fallback")
        return None
    
    # Remove status from kwargs to avoid conflict with explicit status="running"
    clean_kwargs = kwargs.copy()
    clean_kwargs.pop("status", None)
    
    return create_and_broadcast_log_sync(
        agent_name=agent_name,
        log_message=message,
        user_id=user_id,
        analysis_id=analysis_id,
        log_level="WARNING",
        status="running",
        step=step,
        task_id=task_id,
        source="workflow",
        **clean_kwargs
    )

def log_debug(
    agent_name: str,
    message: str,
    user_id: str,
    analysis_id: str,
    step: str = "",
    task_id: str = "",
    **kwargs
) -> Optional[str]:
    """
    Log a debug message.
    
    Args:
        agent_name: Name of the agent
        message: Debug message
        user_id: User identifier
        analysis_id: Analysis identifier
        step: Step being debugged
        task_id: Celery task ID if applicable
        **kwargs: Additional fields
        
    Returns:
        Log message ID if successful, None otherwise
    """
    if not create_and_broadcast_log_sync:
        logger.warning("Logging function not available - using fallback")
        return None
    
    # Remove status from kwargs to avoid conflict with explicit status="running"
    clean_kwargs = kwargs.copy()
    clean_kwargs.pop("status", None)
    
    return create_and_broadcast_log_sync(
        agent_name=agent_name,
        log_message=message,
        user_id=user_id,
        analysis_id=analysis_id,
        log_level="DEBUG",
        status="running",
        step=step,
        task_id=task_id,
        source="workflow",
        **clean_kwargs
    )

def log_workflow_start(
    user_id: str,
    analysis_id: str,
    workflow_name: str = "Agentic AI Transcriptome Analysis",
    total_steps: int = 6,
    task_id: str = "",
    **kwargs
) -> Optional[str]:
    """
    Log the start of a complete workflow.
    
    Args:
        user_id: User identifier
        analysis_id: Analysis identifier
        workflow_name: Name of the workflow
        total_steps: Total number of steps in the workflow
        task_id: Celery task ID if applicable
        **kwargs: Additional fields
        
    Returns:
        Log message ID if successful, None otherwise
    """
    return log_workflow_step(
        agent_name="Workflow Coordinator",
        message=f"Starting {workflow_name}",
        user_id=user_id,
        analysis_id=analysis_id,
        step="workflow_start",
        step_index=0,
        progress=0,
        current_step=0,
        total_steps=total_steps,
        log_level="INFO",
        task_id=task_id,
        workflow_name=workflow_name,
        **kwargs
    )

def log_workflow_complete(
    user_id: str,
    analysis_id: str,
    workflow_name: str = "Agentic AI Transcriptome Analysis",
    total_steps: int = 6,
    elapsed_time_ms: int = 0,
    task_id: str = "",
    **kwargs
) -> Optional[str]:
    """
    Log the completion of a complete workflow.
    
    Args:
        user_id: User identifier
        analysis_id: Analysis identifier
        workflow_name: Name of the workflow
        total_steps: Total number of steps in the workflow
        elapsed_time_ms: Total time taken for the workflow
        task_id: Celery task ID if applicable
        **kwargs: Additional fields
        
    Returns:
        Log message ID if successful, None otherwise
    """
    # Remove status from kwargs to avoid conflict with explicit status="completed"
    clean_kwargs = kwargs.copy()
    clean_kwargs.pop("status", None)
    
    result_id = log_workflow_step(
        agent_name="Workflow Coordinator",
        message=f"Completed {workflow_name}",
        user_id=user_id,
        analysis_id=analysis_id,
        step="workflow_complete",
        step_index=total_steps,
        progress=100,
        current_step=total_steps,
        total_steps=total_steps,
        log_level="INFO",
        status="completed",
        elapsed_time_ms=elapsed_time_ms,
        task_id=task_id,
        workflow_name=workflow_name,
        **clean_kwargs
    )
    # Mark workflow as finalized to suppress any subsequent logs
    try:
        corr_id = kwargs.get("correlation_id")
        if corr_id:
            _FINALIZED_CORRELATION_IDS.add(corr_id)
        _FINALIZED_ANALYSIS_IDS.add(analysis_id)
    except Exception:
        pass
    return result_id

def log_workflow_failed(
    user_id: str,
    analysis_id: str,
    error_message: str,
    workflow_name: str = "Agentic AI Transcriptome Analysis",
    error_code: str = "",
    current_step: int = 0,
    total_steps: int = 6,
    task_id: str = "",
    exception: Exception = Exception(),
    **kwargs
) -> Optional[str]:
    """
    Log the failure of a workflow.
    
    Args:
        user_id: User identifier
        analysis_id: Analysis identifier
        error_message: Description of the failure
        workflow_name: Name of the workflow
        error_code: Standardized error code
        current_step: Step where the failure occurred
        total_steps: Total number of steps in the workflow
        task_id: Celery task ID if applicable
        exception: Exception object for automatic traceback extraction
        **kwargs: Additional fields
        
    Returns:
        Log message ID if successful, None otherwise
    """
    return log_error(
        agent_name="Workflow Coordinator",
        message=f"Failed {workflow_name}: {error_message}",
        user_id=user_id,
        analysis_id=analysis_id,
        error_code=error_code,
        error_message=error_message,
        step="workflow_failed",
        task_id=task_id,
        exception=exception,
        current_step=current_step,
        total_steps=total_steps,
        workflow_name=workflow_name,
        **kwargs
    )

# Convenience class for structured logging within a workflow context
class WorkflowLogger:
    """
    A context-aware logger for workflow operations.
    
    This class maintains context about the current workflow and provides
    convenient methods for logging at different stages.
    
    Example:
        logger = WorkflowLogger(
            user_id=str(user.id),
            analysis_id=str(analysis.id),
            task_id=task_id
        )
        
        logger.start_step("data_retrieval", "Data Retrieval Agent", 1)
        logger.info("Retrieved patient data successfully")
        logger.complete_step("data_retrieval", "Data Retrieval Agent", 1, elapsed_ms=5000)
    """
    
    def __init__(
        self,
        user_id: str,
        analysis_id: str,
        workflow_name: str = "Agentic AI Transcriptome Analysis",
        total_steps: int = 6,
        task_id: str = "",
        correlation_id: str = ""
    ):
        self.user_id = user_id
        self.analysis_id = analysis_id
        self.workflow_name = workflow_name
        self.total_steps = total_steps
        self.task_id = task_id
        self.correlation_id = correlation_id or str(uuid.uuid4())
        self.current_step = 0
        
    def start_workflow(self) -> Optional[str]:
        """Start the workflow logging."""
        return log_workflow_start(
            user_id=self.user_id,
            analysis_id=self.analysis_id,
            workflow_name=self.workflow_name,
            total_steps=self.total_steps,
            task_id=self.task_id,
            correlation_id=self.correlation_id
        )
    
    def start_step(self, step: str, agent_name: str, step_number: int) -> Optional[str]:
        """Start a workflow step."""
        self.current_step = step_number
        return log_step_start(
            agent_name=agent_name,
            step=step,
            user_id=self.user_id,
            analysis_id=self.analysis_id,
            step_index=step_number - 1,
            current_step=step_number,
            total_steps=self.total_steps,
            task_id=self.task_id,
            correlation_id=self.correlation_id
        )
    
    def complete_step(
        self, 
        step: str, 
        agent_name: str, 
        step_number: int, 
        elapsed_ms: int = 0
    ) -> Optional[str]:
        """Complete a workflow step."""
        return log_step_complete(
            agent_name=agent_name,
            step=step,
            user_id=self.user_id,
            analysis_id=self.analysis_id,
            step_index=step_number - 1,
            current_step=step_number,
            total_steps=self.total_steps,
            elapsed_time_ms=elapsed_ms,
            task_id=self.task_id,
            correlation_id=self.correlation_id
        )
    
    def info(self, agent_name: str, message: str, step: str = "", **kwargs) -> Optional[str]:
        """Log an info message."""
        return log_workflow_step(
            agent_name=agent_name,
            message=message,
            user_id=self.user_id,
            analysis_id=self.analysis_id,
            step=step,
            current_step=self.current_step,
            total_steps=self.total_steps,
            log_level="INFO",
            task_id=self.task_id,
            correlation_id=self.correlation_id,
            **kwargs
        )
    
    def warning(self, agent_name: str, message: str, step: str = "", **kwargs) -> Optional[str]:
        """Log a warning message."""
        return log_warning(
            agent_name=agent_name,
            message=message,
            user_id=self.user_id,
            analysis_id=self.analysis_id,
            step=step,
            task_id=self.task_id,
            correlation_id=self.correlation_id,
            **kwargs
        )
    
    def error(
        self, 
        agent_name: str, 
        message: str, 
        step: str = "", 
        error_code: str = "",
        exception: Exception = Exception(),
        **kwargs
    ) -> Optional[str]:
        """Log an error message."""
        return log_error(
            agent_name=agent_name,
            message=message,
            user_id=self.user_id,
            analysis_id=self.analysis_id,
            error_code=error_code,
            step=step,
            task_id=self.task_id,
            exception=exception,
            correlation_id=self.correlation_id,
            **kwargs
        )
    
    def complete_workflow(self, elapsed_ms: int = 0) -> Optional[str]:
        """Complete the workflow logging."""
        return log_workflow_complete(
            user_id=self.user_id,
            analysis_id=self.analysis_id,
            workflow_name=self.workflow_name,
            total_steps=self.total_steps,
            elapsed_time_ms=elapsed_ms,
            task_id=self.task_id,
            correlation_id=self.correlation_id
        )
    
    def fail_workflow(
        self, 
        error_message: str, 
        error_code: str = "", 
        exception: Exception = Exception()
    ) -> Optional[str]:
        """Fail the workflow logging."""
        return log_workflow_failed(
            user_id=self.user_id,
            analysis_id=self.analysis_id,
            error_message=error_message,
            workflow_name=self.workflow_name,
            error_code=error_code,
            current_step=self.current_step,
            total_steps=self.total_steps,
            task_id=self.task_id,
            exception=exception,
            correlation_id=self.correlation_id
        )

# Async version of WorkflowLogger for use in async contexts
class AsyncWorkflowLogger:
    """
    An async context-aware logger for workflow operations.
    
    This class maintains context about the current workflow and provides
    convenient async methods for logging at different stages.
    
    Example:
        logger = AsyncWorkflowLogger(
            user_id=str(user.id),
            analysis_id=str(analysis.id),
            task_id=task_id
        )
        
        await logger.start_step("data_retrieval", "Data Retrieval Agent", 1)
        await logger.info("Retrieved patient data successfully")
        await logger.complete_step("data_retrieval", "Data Retrieval Agent", 1, elapsed_ms=5000)
    """
    
    def __init__(
        self,
        user_id: str,
        analysis_id: str,
        workflow_name: str = "Agentic AI Transcriptome Analysis",
        total_steps: int = 6,
        task_id: str = "",
        correlation_id: str = ""
    ):
        self.user_id = user_id
        self.analysis_id = analysis_id
        self.workflow_name = workflow_name
        self.total_steps = total_steps
        self.task_id = task_id
        self.correlation_id = correlation_id or str(uuid.uuid4())
        self.current_step = 0
        
    async def start_workflow(self) -> Optional[str]:
        """Start the workflow logging."""
        return await _log_async(
            agent_name="Workflow Coordinator",
            message=f"Starting {self.workflow_name}",
            user_id=self.user_id,
            analysis_id=self.analysis_id,
            step="workflow_start",
            step_index=0,
            progress=0,
            current_step=0,
            total_steps=self.total_steps,
            log_level="INFO",
            task_id=self.task_id,
            correlation_id=self.correlation_id,
            workflow_name=self.workflow_name
        )
    
    async def start_step(self, step: str, agent_name: str, step_number: int) -> Optional[str]:
        """Start a workflow step."""
        self.current_step = step_number
        progress = int((step_number / self.total_steps) * 100) if self.total_steps else None
        
        return await _log_async(
            agent_name=agent_name,
            message=f"Starting {step.replace('_', ' ').title()}",
            user_id=self.user_id,
            analysis_id=self.analysis_id,
            step=step,
            step_index=step_number - 1,
            progress=int(progress) if progress else 0,
            current_step=step_number,
            total_steps=self.total_steps,
            log_level="INFO",
            task_id=self.task_id,
            correlation_id=self.correlation_id
        )
    
    async def complete_step(
        self, 
        step: str, 
        agent_name: str, 
        step_number: int, 
        elapsed_ms: int = 0
    ) -> Optional[str]:
        """Complete a workflow step."""
        progress = int((step_number / self.total_steps) * 100) if self.total_steps else None
        
        return await _log_async(
            agent_name=agent_name,
            message=f"Completed {step.replace('_', ' ').title()}",
            user_id=self.user_id,
            analysis_id=self.analysis_id,
            step=step,
            step_index=step_number - 1,
            progress=int(progress) if progress else 0,
            current_step=step_number,
            total_steps=self.total_steps,
            log_level="INFO",
            status="completed",
            elapsed_time_ms=elapsed_ms,
            task_id=self.task_id,
            correlation_id=self.correlation_id
        )
    
    async def info(self, agent_name: str, message: str, step: str = "", **kwargs) -> Optional[str]:
        """Log an info message."""
        return await _log_async(
            agent_name=agent_name,
            message=message,
            user_id=self.user_id,
            analysis_id=self.analysis_id,
            step=step,
            current_step=self.current_step,
            total_steps=self.total_steps,
            log_level="INFO",
            task_id=self.task_id,
            correlation_id=self.correlation_id,
            **kwargs
        )
    
    async def warning(self, agent_name: str, message: str, step: str = "", **kwargs) -> Optional[str]:
        """Log a warning message."""
        return await _log_async(
            agent_name=agent_name,
            message=message,
            user_id=self.user_id,
            analysis_id=self.analysis_id,
            step=step,
            log_level="WARNING",
            task_id=self.task_id,
            correlation_id=self.correlation_id,
            **kwargs
        )
    
    async def error(
        self, 
        agent_name: str, 
        message: str, 
        step: str = "", 
        error_code: str = "",
        exception: Exception = Exception(),
        **kwargs
    ) -> Optional[str]:
        """Log an error message."""
        # Extract traceback if exception is provided
        error_data = kwargs.copy()
        if exception:
            error_data["error_traceback"] = traceback.format_exc()
            if not kwargs.get("error_message"):
                error_data["error_message"] = str(exception)
        
        return await _log_async(
            agent_name=agent_name,
            message=message,
            user_id=self.user_id,
            analysis_id=self.analysis_id,
            step=step,
            log_level="ERROR",
            status="failed",
            task_id=self.task_id,
            error_code=error_code,
            correlation_id=self.correlation_id,
            **error_data
        )
    
    async def complete_workflow(self, elapsed_ms: int = 0) -> Optional[str]:
        """Complete the workflow logging and prevent further emissions for this instance."""
        # Mark finalization to short-circuit any further logs from this instance
        setattr(self, "_finalized", True)
        return await _log_async(
            agent_name="Workflow Coordinator",
            message=f"Completed {self.workflow_name}",
            user_id=self.user_id,
            analysis_id=self.analysis_id,
            step="workflow_complete",
            step_index=self.total_steps,
            progress=100,
            current_step=self.total_steps,
            total_steps=self.total_steps,
            log_level="INFO",
            status="completed",
            elapsed_time_ms=elapsed_ms,
            task_id=self.task_id,
            correlation_id=self.correlation_id,
            workflow_name=self.workflow_name
        )
    
    async def fail_workflow(
        self, 
        error_message: str, 
        error_code: str = "", 
        exception: Exception = Exception()
    ) -> Optional[str]:
        """Fail the workflow logging."""
        # Extract traceback if exception is provided
        error_data = {}
        if exception:
            error_data["error_traceback"] = traceback.format_exc()
            error_data["error_message"] = error_message
        
        return await _log_async(
            agent_name="Workflow Coordinator",
            message=f"Failed {self.workflow_name}: {error_message}",
            user_id=self.user_id,
            analysis_id=self.analysis_id,
            step="workflow_failed",
            log_level="ERROR",
            status="failed",
            task_id=self.task_id,
            error_code=error_code,
            current_step=self.current_step,
            total_steps=self.total_steps,
            correlation_id=self.correlation_id,
            workflow_name=self.workflow_name,
            **error_data
        )


# ============================================================================
# LLM LOG ENHANCEMENT
# ============================================================================

async def update_log_with_enhancement(
    log_id: str,
    enhanced_message: str
) -> bool:
    """
    Update a log entry with LLM-enhanced message and re-broadcast.
    
    Production-ready: Gracefully handles missing database fields by storing
    enhanced message in details JSON field or skipping DB update if field doesn't exist.
    
    Args:
        log_id: DB log ID to update
        enhanced_message: Enhanced MDX message
    
    Returns:
        True if successful, False otherwise
    """
    try:
        import json
        import redis.asyncio as aioredis
        from analysisapp.models import LogMessage
        from django.conf import settings
        from django.db import close_old_connections
        from agenticaib.db_pool import sync_to_async_with_cleanup
        
        # FIXED: Use sync_to_async_with_cleanup to prevent connection leaks
        # Get log entry
        log_entry = await sync_to_async_with_cleanup(LogMessage.objects.get)(id=log_id)
        
        # Try to update enhanced_log_message field if it exists
        # Otherwise, store in details JSON field as fallback
        try:
            # Check if field exists by trying to access it
            if hasattr(log_entry, 'enhanced_log_message'):
                log_entry.enhanced_log_message = enhanced_message
                # FIXED: Use sync_to_async_with_cleanup
                await sync_to_async_with_cleanup(log_entry.save)(update_fields=['enhanced_log_message'])
            else:
                # Field doesn't exist - store in details JSON field instead
                details = log_entry.details or {}
                details['enhanced_log_message'] = enhanced_message
                details['enhancement_timestamp'] = datetime.now(timezone.utc).isoformat()
                log_entry.details = details
                # FIXED: Use sync_to_async_with_cleanup
                await sync_to_async_with_cleanup(log_entry.save)(update_fields=['details'])
        except (AttributeError, ValueError) as db_error:
            # Field doesn't exist or can't be updated - skip DB update, just broadcast
            logger.debug(f"Enhanced log message field not available, storing in details or skipping DB update: {db_error}")
        
        # Build Redis channels (same pattern as original broadcast)
        user_id = str(log_entry.user_id.id) if log_entry.user_id else ""
        analysis_id = str(log_entry.analysis_id) if log_entry.analysis_id else ""
        
        channels = [
            f"logs:user:{user_id}:analysis:{analysis_id}",
            f"logs:user:{user_id}:all",
            f"logs:analysis:{analysis_id}",
            "logs:global"
        ]
        
        # Build updated broadcast data
        broadcast_data = {
            "db_log_id": log_id,
            "enhanced_log_message": enhanced_message,
            "enhancement_timestamp": datetime.now(timezone.utc).isoformat(),
            "user_id": user_id,
            "analysis_id": analysis_id,
            "agent_name": log_entry.agent_name,
            "original_message": getattr(log_entry, 'log_message', ''),
            "log_level": log_entry.log_level,
            "step": log_entry.step,
            "status": log_entry.status,
            "channels": channels
        }
        
        # Broadcast to Redis
        redis_url = getattr(settings, 'REDIS_URL', 'redis://localhost:6379/0')
        redis_client = await aioredis.from_url(redis_url)
        
        log_json = json.dumps(broadcast_data, default=str)
        
        for channel in channels:
            await redis_client.publish(channel, log_json)
        
        await redis_client.close()
        
        logger.debug(f"Enhanced log {log_id} re-broadcasted to {len(channels)} channels")
        return True
        
    except Exception as e:
        # Graceful degradation - log error but don't crash
        logger.debug(f"Failed to update enhanced log {log_id}: {e}")
        return False

# Convenience functions for easy import and use
async def log_async(
    agent_name: str,
    message: str,
    user_id: str,
    analysis_id: str,
    log_level: str = "INFO",
    step: str = "",
    **kwargs
) -> Optional[str]:
    """
    Simple async logging function for easy import and use from async contexts.
    This is the main function you should use from async contexts like coordinators.
    
    Args:
        agent_name: Name of the agent
        message: Log message
        user_id: User identifier
        analysis_id: Analysis identifier
        log_level: Log level (INFO, WARNING, ERROR, etc.)
        step: Current step name
        **kwargs: Additional fields
        
    Returns:
        Log message ID if successful, None otherwise
        
    Example:
        await log_async(
            agent_name="Data Retrieval Agent",
            message="Starting data retrieval",
            user_id=str(user.id),
            analysis_id=str(analysis.id),
            step="data_retrieval"
        )
    """
    return await _log_async(
        agent_name=agent_name,
        message=message,
        user_id=user_id,
        analysis_id=analysis_id,
        log_level=log_level,
        step=step,
        **kwargs
    )