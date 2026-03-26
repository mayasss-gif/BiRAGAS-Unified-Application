"""
Base tool class for DEG Pipeline Agent tools.
"""

import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Callable, List
from functools import wraps
from pathlib import Path

from ..config import DEGPipelineConfig
from ..exceptions import DEGPipelineError, RecoverableError, NonRecoverableError


class BaseTool(ABC):
    """Base class for all DEG pipeline tools."""
    
    def __init__(self, config: DEGPipelineConfig, logger: Optional[logging.Logger] = None):
        self.config = config
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self.execution_history: List[Dict[str, Any]] = []
        self.last_error: Optional[Exception] = None
        self.success_count = 0
        self.error_count = 0
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Tool name."""
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """Tool description."""
        pass
    
    @abstractmethod
    def execute(self, *args, **kwargs) -> Any:
        """Execute the tool's main function."""
        pass
    
    def retry_with_backoff(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute a function with retry logic and exponential backoff.
        
        Args:
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
            
        Raises:
            Exception: If all retries are exhausted
        """
        max_retries = self.config.max_retries
        base_delay = self.config.retry_delay
        
        for attempt in range(max_retries + 1):
            try:
                result = func(*args, **kwargs)
                if attempt > 0:
                    self.logger.info(f"✅ Succeeded on attempt {attempt + 1}")
                return result
                
            except RecoverableError as e:
                if attempt < max_retries:
                    delay = base_delay * (2 ** attempt)
                    self.logger.warning(f"⚠️  Recoverable error on attempt {attempt + 1}: {e}")
                    self.logger.info(f"🔄 Retrying in {delay} seconds...")
                    time.sleep(delay)
                    
                    # Try to auto-fix if enabled
                    if self.config.enable_auto_fix and hasattr(e, 'fix_suggestion'):
                        self.logger.info(f"🔧 Attempting auto-fix: {e.fix_suggestion}")
                        try:
                            self._apply_fix(e.fix_suggestion)
                        except Exception as fix_error:
                            self.logger.warning(f"❌ Auto-fix failed: {fix_error}")
                else:
                    self.logger.error(f"❌ All {max_retries + 1} attempts failed")
                    raise
                    
            except NonRecoverableError as e:
                self.logger.error(f"❌ Non-recoverable error: {e}")
                raise
                
            except Exception as e:
                self.last_error = e
                self.error_count += 1
                
                if attempt < max_retries:
                    delay = base_delay * (2 ** attempt)
                    self.logger.warning(f"⚠️  Error on attempt {attempt + 1}: {e}")
                    self.logger.info(f"🔄 Retrying in {delay} seconds...")
                    time.sleep(delay)
                else:
                    self.logger.error(f"❌ All {max_retries + 1} attempts failed")
                    raise
    
    def _apply_fix(self, fix_suggestion: str) -> None:
        """
        Apply an automatic fix based on the suggestion.
        
        Args:
            fix_suggestion: Description of the fix to apply
        """
        # This is a placeholder - specific tools can override this
        self.logger.info(f"🔧 Applying fix: {fix_suggestion}")
    
    def validate_input(self, *args, **kwargs) -> None:
        """
        Validate input parameters.
        
        Args:
            *args: Input arguments
            **kwargs: Input keyword arguments
            
        Raises:
            DEGPipelineError: If validation fails
        """
        # Default validation - can be overridden by specific tools
        pass
    
    def validate_output(self, result: Any) -> Any:
        """
        Validate output result.
        
        Args:
            result: Output result to validate
            
        Returns:
            Validated result
            
        Raises:
            DEGPipelineError: If validation fails
        """
        # Default validation - can be overridden by specific tools
        return result
    
    def log_execution(self, func_name: str, args: tuple, kwargs: dict, result: Any, error: Optional[Exception] = None) -> None:
        """
        Log execution details.
        
        Args:
            func_name: Name of the executed function
            args: Function arguments
            kwargs: Function keyword arguments
            result: Function result
            error: Exception if one occurred
        """
        execution_record = {
            "timestamp": time.time(),
            "function": func_name,
            "args_count": len(args),
            "kwargs_keys": list(kwargs.keys()),
            "success": error is None,
            "error": str(error) if error else None,
            "result_type": type(result).__name__ if result is not None else None
        }
        
        self.execution_history.append(execution_record)
        
        if error:
            self.logger.error(f"❌ {func_name} failed: {error}")
        else:
            self.logger.info(f"✅ {func_name} completed successfully")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get execution statistics.
        
        Returns:
            Dictionary containing execution statistics
        """
        total_executions = len(self.execution_history)
        successful_executions = sum(1 for record in self.execution_history if record["success"])
        
        return {
            "tool_name": self.name,
            "total_executions": total_executions,
            "successful_executions": successful_executions,
            "failed_executions": total_executions - successful_executions,
            "success_rate": successful_executions / total_executions if total_executions > 0 else 0,
            "last_error": str(self.last_error) if self.last_error else None,
            "execution_history": self.execution_history[-5:]  # Last 5 executions
        }
    
    def reset_stats(self) -> None:
        """Reset execution statistics."""
        self.execution_history.clear()
        self.last_error = None
        self.success_count = 0
        self.error_count = 0
    
    def safe_execute(self, *args, **kwargs) -> Any:
        """
        Safely execute the tool with error handling and logging.
        
        Args:
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Tool execution result
        """
        start_time = time.time()
        result = None
        error = None
        
        try:
            # Validate input
            self.validate_input(*args, **kwargs)
            
            # Execute with retry logic
            result = self.retry_with_backoff(self.execute, *args, **kwargs)
            
            # Validate output
            result = self.validate_output(result)
            
            self.success_count += 1
            
        except Exception as e:
            error = e
            self.error_count += 1
            raise
            
        finally:
            # Log execution
            execution_time = time.time() - start_time
            self.logger.info(f"⏱️  {self.name} execution time: {execution_time:.2f}s")
            self.log_execution(self.execute.__name__, args, kwargs, result, error)
        
        return result
    
    def __str__(self) -> str:
        return f"{self.name}: {self.description}"
    
    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}: {self.name}>" 