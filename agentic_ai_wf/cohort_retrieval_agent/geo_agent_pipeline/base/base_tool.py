"""
Base tool class for the Cohort Retrieval Agent system.

This module provides the abstract base class for all tools, including
common functionality like retry logic, error handling, and validation.
"""

import asyncio
import random
import time
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Callable, TypeVar, Generic
from dataclasses import dataclass

# Project Imports
from   ..config import CohortRetrievalConfig, RetryConfig
from   ..exceptions import CohortRetrievalError, TimeoutError as CohortTimeoutError

T = TypeVar('T')

@dataclass
class ToolResult(Generic[T]):
    """Standard result container for tool operations."""
    success: bool
    data: Optional[T] = None
    error: Optional[str] = None
    details: Optional[Dict[str, Any]] = None
    execution_time: float = 0.0
    retry_count: int = 0


class BaseTool(ABC, Generic[T]):
    """
    Abstract base class for all tools in the Cohort Retrieval Agent system.
    
    Provides common functionality:
    - Retry logic with exponential backoff
    - Error handling and recovery
    - Validation of inputs and outputs
    - Progress tracking
    - Logging
    """
    
    def __init__(self, config: CohortRetrievalConfig, name: str = ""):
        self.config = config
        self.name = name or self.__class__.__name__
        self.logger = logging.getLogger(f"cohort_retrieval.{self.name}")
        self.retry_config = config.retry_config
        
        # Statistics
        self.total_operations = 0
        self.successful_operations = 0
        self.failed_operations = 0
        self.total_retry_attempts = 0
    
    async def run_with_retry(self, operation: Callable, *args, **kwargs) -> ToolResult[T]:
        """
        Execute an operation with retry logic and error handling.
        
        Args:
            operation: The async function to execute
            *args: Arguments to pass to the operation
            **kwargs: Keyword arguments to pass to the operation
            
        Returns:
            ToolResult containing the operation result
        """
        start_time = time.time()
        last_error = None
        
        for attempt in range(self.retry_config.max_retries + 1):
            try:
                self.logger.debug(f"Attempt {attempt + 1} for {self.name}")
                # Execute the operation
                result = await operation(*args, **kwargs)
                # Validate the result
                if self.validate_output(result):
                    execution_time = time.time() - start_time
                    self.successful_operations += 1
                    self.total_operations += 1
                    
                    self.logger.info(f"{self.name} completed successfully in {execution_time:.2f}s")
                    
                    return ToolResult(
                        success=True,
                        data=result,
                        execution_time=execution_time,
                        retry_count=attempt
                    )
                else:
                    raise CohortRetrievalError(f"Output validation failed for {self.name}")
                    
            except Exception as e:
                last_error = e
                self.logger.warning(f"Attempt {attempt + 1} failed for {self.name}: {e}")
                
                # If this is the last attempt, don't wait
                if attempt >= self.retry_config.max_retries:
                    break
                
                # Calculate delay with exponential backoff and jitter
                delay = self._calculate_delay(attempt)
                self.logger.info(f"Retrying {self.name} in {delay:.2f}s")
                await asyncio.sleep(delay)
                
                self.total_retry_attempts += 1
        
        # All attempts failed
        execution_time = time.time() - start_time
        self.failed_operations += 1
        self.total_operations += 1
        
        self.logger.error(f"{self.name} failed after {self.retry_config.max_retries + 1} attempts")
        
        return ToolResult(
            success=False,
            error=str(last_error),
            details={"last_error": str(last_error), "attempts": self.retry_config.max_retries + 1},
            execution_time=execution_time,
            retry_count=self.retry_config.max_retries
        )
    
    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay for exponential backoff with jitter."""
        base_delay = self.retry_config.base_delay
        exponential_base = self.retry_config.exponential_base
        max_delay = self.retry_config.max_delay
        
        # Calculate exponential delay
        delay = base_delay * (exponential_base ** attempt)
        
        # Cap at max_delay
        delay = min(delay, max_delay)
        
        # Add jitter if enabled
        if self.retry_config.jitter:
            jitter = random.uniform(0.5, 1.5)
            delay *= jitter
        
        return delay
    
    async def execute_with_timeout(self, operation: Callable, timeout: int, *args, **kwargs) -> ToolResult[T]:
        """
        Execute an operation with a timeout.
        
        Args:
            operation: The async function to execute
            timeout: Timeout in seconds
            *args: Arguments to pass to the operation
            **kwargs: Keyword arguments to pass to the operation
            
        Returns:
            ToolResult containing the operation result
        """
        try:
            result = await asyncio.wait_for(operation(*args, **kwargs), timeout=timeout)
            return ToolResult(success=True, data=result)
        except asyncio.TimeoutError:
            error_msg = f"{self.name} timed out after {timeout} seconds"
            self.logger.error(error_msg)
            return ToolResult(
                success=False,
                error=error_msg,
                details={"timeout": timeout}
            )
        except Exception as e:
            error_msg = f"{self.name} failed: {e}"
            self.logger.error(error_msg)
            return ToolResult(
                success=False,
                error=error_msg,
                details={"exception": str(e)}
            )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get tool execution statistics."""
        success_rate = (self.successful_operations / self.total_operations * 100) if self.total_operations > 0 else 0
        
        return {
            "tool_name": self.name,
            "total_operations": self.total_operations,
            "successful_operations": self.successful_operations,
            "failed_operations": self.failed_operations,
            "success_rate": round(success_rate, 2),
            "total_retry_attempts": self.total_retry_attempts,
            "average_retries_per_operation": round(self.total_retry_attempts / self.total_operations, 2) if self.total_operations > 0 else 0
        }
    
    def reset_statistics(self):
        """Reset tool execution statistics."""
        self.total_operations = 0
        self.successful_operations = 0
        self.failed_operations = 0
        self.total_retry_attempts = 0
        self.logger.info(f"Statistics reset for {self.name}")
    
    def log_performance(self):
        """Log performance statistics."""
        stats = self.get_statistics()
        self.logger.info(f"Performance stats for {self.name}: {stats}")


class AsyncContextTool(BaseTool[T]):
    """
    Base class for tools that need to manage async context (like HTTP sessions).
    """
    
    def __init__(self, config: CohortRetrievalConfig, name: str = ""):
        super().__init__(config, name)
        self._context = None
    
    @abstractmethod
    async def create_context(self) -> Any:
        """Create and return the async context (e.g., aiohttp.ClientSession)."""
        pass
    
    @abstractmethod
    async def close_context(self, context: Any):
        """Close the async context."""
        pass
    
    async def __aenter__(self):
        """Async context manager entry."""
        self._context = await self.create_context()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self._context:
            await self.close_context(self._context)
            self._context = None
    
    @property
    def context(self) -> Any:
        """Get the current context."""
        return self._context 