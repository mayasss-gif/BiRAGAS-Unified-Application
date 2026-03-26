# Agentic AI Enhanced Logging System

This document provides comprehensive examples of how to use the enhanced logging system that automatically saves logs to the database and broadcasts them to real-time channels.

## Overview

The enhanced logging system provides:
- **Database Persistence**: All logs are saved to the `LogMessage` model in PostgreSQL
- **Real-time Broadcasting**: Logs are broadcasted to Redis channels for live streaming
- **Structured Logging**: Comprehensive metadata including progress tracking, performance metrics, and error details
- **Multiple Interfaces**: FastAPI endpoints, synchronous functions for Celery, and convenient utility functions

## Quick Start

### 1. From FastAPI Routes (send_log endpoint)

The FastAPI `/send-log/` endpoint automatically creates database records when broadcasting logs:

```python
# The send_log endpoint in fastapi_app/routes/logs.py automatically:
# 1. Validates the log data
# 2. Saves to LogMessage database table
# 3. Broadcasts to Redis channels
# 4. Streams to connected clients

# Example log data sent to the endpoint:
log_data = {
    "agent_name": "Data Retrieval Agent",
    "log_message": "Starting patient data retrieval",
    "step": "data_retrieval",
    "step_index": 1,
    "log_level": "INFO",
    "status": "running",
    "progress_percentage": 10,
    "current_step": 1,
    "total_steps": 6,
    "user_id": str(user.id),
    "analysis_id": str(analysis.id)
}
```

### 2. From Celery Tasks and Coordinators

Use the convenient utility functions from `agentic_ai_wf.helpers.logging_utils`:

```python
from agentic_ai_wf.helpers.logging_utils import (
    log_workflow_step, log_step_start, log_step_complete, 
    log_error, WorkflowLogger
)

# Simple workflow step logging
log_workflow_step(
    agent_name="Gene Prioritization Agent",
    message="Processing differential expression analysis",
    user_id=str(user.id),
    analysis_id=str(analysis.id),
    step="gene_prioritization",
    step_index=2,
    progress=33,
    current_step=2,
    total_steps=6,
    task_id=self.request.id  # Celery task ID
)

# Error logging with automatic traceback
try:
    # Some operation that might fail
    process_gene_data()
except Exception as e:
    log_error(
        agent_name="Gene Prioritization Agent",
        message="Failed to process gene prioritization",
        user_id=str(user.id),
        analysis_id=str(analysis.id),
        error_code="GENE_PROC_001",
        step="gene_prioritization",
        exception=e,  # Automatically extracts traceback
        task_id=self.request.id
    )
```

### 3. Using the WorkflowLogger Class

For structured logging throughout a workflow:

```python
from agentic_ai_wf.helpers.logging_utils import WorkflowLogger

# Initialize logger for the entire workflow
logger = WorkflowLogger(
    user_id=str(user.id),
    analysis_id=str(analysis.id),
    task_id=self.request.id,
    total_steps=6
)

# Start the workflow
logger.start_workflow()

# Log each step
logger.start_step("data_retrieval", "Data Retrieval Agent", 1)
logger.info("Data Retrieval Agent", "Retrieved patient transcriptome data")
logger.complete_step("data_retrieval", "Data Retrieval Agent", 1, elapsed_ms=5000)

# Handle errors
try:
    perform_analysis()
except Exception as e:
    logger.error(
        "Gene Prioritization Agent", 
        "Analysis failed", 
        step="gene_prioritization",
        error_code="ANALYSIS_001",
        exception=e
    )
    logger.fail_workflow("Analysis pipeline failed", exception=e)
    return

# Complete the workflow
logger.complete_workflow(elapsed_ms=30000)
```

## Detailed Examples

### Example 1: Enhanced Celery Task with Comprehensive Logging

```python
from celery import shared_task
from agentic_ai_wf.helpers.logging_utils import WorkflowLogger, log_error
import time

@shared_task(bind=True, max_retries=3)
def transcriptome_analysis_task(self, analysis_id: str, user_id: str):
    """Enhanced Celery task with comprehensive logging."""
    
    # Initialize workflow logger
    logger = WorkflowLogger(
        user_id=user_id,
        analysis_id=analysis_id,
        task_id=self.request.id,
        total_steps=6
    )
    
    start_time = time.time()
    
    try:
        # Start workflow
        logger.start_workflow()
        
        # Step 1: Authentication & Authorization
        logger.start_step("authentication", "Authentication Agent", 1)
        # ... authentication logic ...
        logger.complete_step("authentication", "Authentication Agent", 1, 
                           elapsed_ms=int((time.time() - start_time) * 1000))
        
        # Step 2: Data Retrieval
        step_start = time.time()
        logger.start_step("data_retrieval", "Data Retrieval Agent", 2)
        
        try:
            # Simulate data retrieval with progress updates
            logger.info("Data Retrieval Agent", "Connecting to data sources")
            time.sleep(2)
            
            logger.info("Data Retrieval Agent", "Downloading transcriptome data", 
                       details={"file_size": "2.3GB", "source": "GEO"})
            time.sleep(3)
            
            logger.complete_step("data_retrieval", "Data Retrieval Agent", 2,
                               elapsed_ms=int((time.time() - step_start) * 1000))
        
        except Exception as e:
            logger.error("Data Retrieval Agent", "Failed to retrieve data",
                        step="data_retrieval", error_code="DATA_001", exception=e)
            raise
        
        # Step 3: Gene Prioritization
        step_start = time.time()
        logger.start_step("gene_prioritization", "Gene Prioritization Agent", 3)
        
        # Simulate processing with performance metrics
        import psutil
        memory_usage = psutil.virtual_memory().used / (1024 * 1024)  # MB
        cpu_percent = psutil.cpu_percent()
        
        logger.info("Gene Prioritization Agent", 
                   "Processing differential expression analysis",
                   memory_usage_mb=memory_usage,
                   cpu_usage_percent=cpu_percent,
                   details={"genes_processed": 15000, "method": "DESeq2"})
        
        time.sleep(5)
        
        logger.complete_step("gene_prioritization", "Gene Prioritization Agent", 3,
                           elapsed_ms=int((time.time() - step_start) * 1000))
        
        # Continue with remaining steps...
        # Step 4: Pathway Analysis
        # Step 5: Drug Prioritization  
        # Step 6: Report Generation
        
        # Complete workflow
        total_elapsed = int((time.time() - start_time) * 1000)
        logger.complete_workflow(elapsed_ms=total_elapsed)
        
        return {"status": "completed", "analysis_id": analysis_id}
        
    except Exception as e:
        # Log workflow failure
        logger.fail_workflow(
            error_message=f"Workflow failed: {str(e)}",
            error_code="WORKFLOW_001",
            exception=e
        )
        raise
```

### Example 2: Coordinator Integration

```python
from agentic_ai_wf.helpers.logging_utils import log_workflow_start, log_step_start, log_error

async def run_workflow_coordinator(analysis_id: str, user_id: str):
    """Enhanced coordinator with integrated logging."""
    
    # Start workflow logging
    log_workflow_start(
        user_id=user_id,
        analysis_id=analysis_id,
        workflow_name="Agentic AI Transcriptome Analysis",
        total_steps=6
    )
    
    try:
        # Step 1: Authentication
        log_step_start("Authentication Agent", "authentication", 
                      user_id, analysis_id, step_index=0, current_step=1)
        
        auth_result = await authenticate_user(user_id, analysis_id)
        
        if not auth_result.success:
            log_error("Authentication Agent", "Authentication failed",
                     user_id, analysis_id, error_code="AUTH_001",
                     error_message=auth_result.error)
            return
        
        # Continue with other steps...
        
    except Exception as e:
        log_error("Workflow Coordinator", "Coordinator failed",
                 user_id, analysis_id, error_code="COORD_001", exception=e)
        raise
```

### Example 3: Broadcasting API Integration

```python
from agentic_ai_wf.broadcast_api import send_log_message

# The broadcast_api now automatically creates database records
async def send_enhanced_log(user_id: str, analysis_id: str, log_data: dict):
    """Enhanced broadcasting that saves to database."""
    
    # The broadcast_api send_log_message function now:
    # 1. Validates the user and analysis
    # 2. Creates LogMessage in database
    # 3. Broadcasts to Redis channels
    # 4. Handles errors gracefully
    
    result = await send_log_message(
        user_id=user_id,
        analysis_id=analysis_id,
        agent_name="Custom Agent",
        log_level="INFO",
        status="running",
        message="Custom operation completed",
        step="custom_step",
        progress_percentage=75,
        details={"custom_field": "custom_value"}
    )
    
    if result.success:
        print(f"Log created with ID: {result.log_id}")
    else:
        print(f"Failed to create log: {result.error}")
```

## Database Schema

The enhanced `LogMessage` model includes these key fields:

```python
class LogMessage(models.Model):
    # Identification
    id = models.UUIDField(primary_key=True)
    workflow_id = models.CharField(max_length=128)
    run_id = models.CharField(max_length=128)
    
    # Agent and step information
    agent_id = models.CharField(max_length=64)
    agent_name = models.CharField(max_length=255)
    step = models.CharField(max_length=128)
    step_index = models.IntegerField()
    
    # Log classification
    log_level = models.CharField(choices=LogLevel.choices)
    status = models.CharField(choices=WorkflowStatus.choices)
    source = models.CharField(choices=LogSource.choices)
    
    # Progress tracking
    progress_percentage = models.IntegerField(null=True)
    current_step = models.IntegerField(null=True)
    total_steps = models.IntegerField(null=True)
    
    # Performance metrics
    elapsed_time_ms = models.BigIntegerField(null=True)
    memory_usage_mb = models.FloatField(null=True)
    cpu_usage_percent = models.FloatField(null=True)
    
    # Relationships
    analysis = models.ForeignKey(Analysis, on_delete=models.CASCADE)
    user = models.ForeignKey(CoreUser, on_delete=models.CASCADE)
    
    # Celery task information
    task_id = models.CharField(max_length=128, null=True)
    task_name = models.CharField(max_length=255, null=True)
    
    # Error information
    error_code = models.CharField(max_length=64, null=True)
    error_message = models.TextField(null=True)
    error_traceback = models.TextField(null=True)
    
    # Flexible JSON fields
    details = models.JSONField(null=True)
    meta = models.JSONField(null=True)
    context = models.JSONField(null=True)
```

## Querying Logs

The enhanced model provides powerful querying capabilities:

```python
from analysisapp.models import LogMessage

# Get logs for a specific analysis
analysis_logs = LogMessage.objects.for_analysis(analysis_id)

# Get error logs only
error_logs = LogMessage.objects.errors_only()

# Get logs from the last 24 hours
recent_logs = LogMessage.objects.recent(hours=24)

# Get logs with progress information
progress_logs = LogMessage.objects.with_progress()

# Get performance summary
summary = LogMessage.objects.performance_summary()
print(f"Average elapsed time: {summary['avg_elapsed_time']}ms")
print(f"Total errors: {summary['error_count']}")

# Complex filtering
complex_query = (LogMessage.objects
                .for_analysis(analysis_id)
                .by_level("WARNING")
                .recent(hours=12)
                .order_by('-timestamp'))

# Get logs for a specific workflow run
workflow_logs = LogMessage.objects.for_run(run_id)
```

## Real-time Streaming

Logs are automatically broadcasted to multiple Redis channels:

- `logs:user:{user_id}:analysis:{analysis_id}` - User-specific analysis logs
- `logs:user:{user_id}:all` - All logs for a user
- `logs:analysis:{analysis_id}` - All logs for an analysis
- `logs:global` - Global log stream

Connect to these channels to receive real-time log updates in your frontend applications.

## Best Practices

1. **Use WorkflowLogger for structured workflows** - It maintains context and provides convenient methods
2. **Include performance metrics** - Memory and CPU usage help with optimization
3. **Use error codes** - Standardized error codes help with monitoring and debugging
4. **Include correlation IDs** - For tracking requests across services
5. **Add relevant tags** - For categorizing and filtering logs
6. **Use appropriate log levels** - DEBUG for development, INFO for normal operations, WARNING for issues, ERROR for failures
7. **Include progress tracking** - For long-running operations to show user progress
8. **Add detailed context** - Use the `details`, `meta`, and `context` JSON fields for additional information

## Migration

To use the enhanced logging system:

1. **Run migrations** to create the new LogMessage table:
   ```bash
   python manage.py makemigrations analysisapp
   python manage.py migrate
   ```

2. **Update your code** to use the new logging functions instead of manual log creation

3. **Configure Redis** for real-time broadcasting (if not already configured)

4. **Update frontend** to connect to the new streaming endpoints for real-time log display

The system is backward compatible and will gracefully handle missing dependencies.