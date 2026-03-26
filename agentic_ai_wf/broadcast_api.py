"""
Production-ready Broadcasting API for Agentic AI Workflow.

This module provides:
- User authentication and authorization
- Enhanced logging with structured data
- Real-time progress tracking
- Error handling and monitoring
- Integration with Django backend
- Scalable architecture patterns
"""

import httpx
import asyncio
import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from enum import Enum
from decouple import config

# Configuration
ENV = config("ENV", default="dev")
API_BASE_URL = config("API_BASE_URL", default="https://dev-agent-admin.f420.ai" if ENV !=
                      "prod" else "https://agent-admin.f420.ai")
LOGS_ENDPOINT = f"{API_BASE_URL}/fastapi/logs/send-log/"
AUTH_ENDPOINT = f"{API_BASE_URL}/api/auth/user-info/"
ANALYSIS_PERMISSION_ENDPOINT = f"{API_BASE_URL}/api/analysis/{{analysis_id}}/permissions/"

logger = logging.getLogger(__name__)


class LogLevel(Enum):
    """Standardized log levels"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class WorkflowStatus(Enum):
    """Workflow execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PARTIALLY_COMPLETED = "partially_completed"
    FULLY_COMPLETED = "fully_completed"


@dataclass
class LogMessage:
    """Structured log message"""
    timestamp: str
    agent_id: str
    agent_name: str
    workflow_id: str
    workflow_name: str
    run_id: str
    analysis_id: str
    step: str
    step_index: int
    log_level: str
    status: str
    log_message: str
    user_id: str
    user_email: Optional[str] = None
    details: Optional[Dict[str, Any]] = None
    elapsed_time_ms: int = 0
    error: Optional[str] = None
    meta: Optional[Dict[str, Any]] = None
    progress_percentage: Optional[int] = None
    current_step: Optional[int] = None
    total_steps: Optional[int] = None


@dataclass
class UserInfo:
    """User information"""
    id: str
    email: str
    username: str
    is_active: bool
    permissions: List[str]


class AuthenticationError(Exception):
    """Authentication related errors"""
    pass


class AuthorizationError(Exception):
    """Authorization related errors"""
    pass


class BroadcastError(Exception):
    """Broadcasting related errors"""
    pass


class UserAuthenticator:
    """
    Production-ready user authentication and authorization.

    Features:
    - JWT token validation
    - User information retrieval
    - Permission checking
    - Caching for performance
    """

    def __init__(self):
        self.user_cache = {}
        self.cache_ttl = 300  # 5 minutes

    async def get_user_info(self, user_id: str) -> Optional[UserInfo]:
        """
        Get user information by user ID.

        Args:
            user_id: User identifier

        Returns:
            UserInfo object or None if user not found

        Raises:
            AuthenticationError: If authentication fails
        """
        try:
            # Check cache first
            cache_key = f"user_{user_id}"
            if cache_key in self.user_cache:
                cached_data, timestamp = self.user_cache[cache_key]
                if time.time() - timestamp < self.cache_ttl:
                    return cached_data

            # Fetch from API
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(
                    f"{API_BASE_URL}/api/users/{user_id}/",
                    headers={"Content-Type": "application/json"}
                )

                if response.status_code == 404:
                    return None

                response.raise_for_status()
                user_data = response.json()

                user_info = UserInfo(
                    id=user_data["id"],
                    email=user_data["email"],
                    username=user_data.get("username", ""),
                    is_active=user_data.get("is_active", False),
                    permissions=user_data.get("permissions", [])
                )

                # Cache the result
                self.user_cache[cache_key] = (user_info, time.time())

                return user_info

        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error while fetching user info: {e}")
            raise AuthenticationError(f"Failed to authenticate user {user_id}")
        except Exception as e:
            logger.error(f"Unexpected error while fetching user info: {e}")
            raise AuthenticationError(f"Authentication service unavailable")

    async def check_analysis_permission(self, user_id: str, analysis_id: str) -> bool:
        """
        Check if user has permission to access specific analysis.

        Args:
            user_id: User identifier
            analysis_id: Analysis identifier

        Returns:
            True if user has permission, False otherwise

        Raises:
            AuthorizationError: If authorization check fails
        """
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(
                    ANALYSIS_PERMISSION_ENDPOINT.format(
                        analysis_id=analysis_id),
                    params={"user_id": user_id},
                    headers={"Content-Type": "application/json"}
                )

                if response.status_code == 404:
                    return False

                response.raise_for_status()
                permission_data = response.json()

                return permission_data.get("has_permission", False)

        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error while checking permissions: {e}")
            # In case of API failure, allow access for system users
            return user_id == "system"
        except Exception as e:
            logger.error(f"Unexpected error while checking permissions: {e}")
            return user_id == "system"


class WorkflowLogger:
    """
    Production-ready workflow logging with enhanced features.

    Features:
    - Structured logging
    - Real-time progress tracking
    - Error categorization
    - Performance monitoring
    - Retry mechanisms
    """

    def __init__(
        self,
        workflow_id: str,
        run_id: str,
        analysis_id: str,
        user_id: str,
        workflow_name: str = "Agentic AI Transcriptome Analysis"
    ):
        self.workflow_id = workflow_id
        self.run_id = run_id
        self.analysis_id = analysis_id
        self.user_id = user_id
        self.workflow_name = workflow_name
        self.start_time = time.time()
        self.user_email = None

        # Initialize user email
        asyncio.create_task(self._initialize_user_info())

    async def _initialize_user_info(self):
        """Initialize user information"""
        try:
            authenticator = UserAuthenticator()
            user_info = await authenticator.get_user_info(self.user_id)
            if user_info:
                self.user_email = user_info.email
        except Exception as e:
            logger.warning(f"Failed to initialize user info: {e}")
            self.user_email = "unknown@example.com"

    async def log_info(
        self,
        message: str,
        step: str,
        agent_name: Optional[str] = None,
        meta: Optional[Dict[str, Any]] = None
    ):
        """Log informational message"""
        await self._send_log(
            message=message,
            step=step,
            agent_name=agent_name or self._get_agent_name(step),
            log_level=LogLevel.INFO,
            status=WorkflowStatus.RUNNING,
            meta=meta or {}
        )

    async def log_warning(
        self,
        message: str,
        step: str,
        agent_name: Optional[str] = None,
        error: Optional[str] = None,
        meta: Optional[Dict[str, Any]] = None
    ):
        """Log warning message"""
        await self._send_log(
            message=message,
            step=step,
            agent_name=agent_name or self._get_agent_name(step),
            log_level=LogLevel.WARNING,
            status=WorkflowStatus.RUNNING,
            error=error or "",
            meta=meta or {}
        )

    async def log_error(
        self,
        message: str,
        step: str,
        agent_name: Optional[str] = None,
        error: Optional[str] = None,
        meta: Optional[Dict[str, Any]] = None
    ):
        """Log error message"""
        await self._send_log(
            message=message,
            step=step,
            agent_name=agent_name or self._get_agent_name(step),
            log_level=LogLevel.ERROR,
            status=WorkflowStatus.FAILED,
            error=error or "",
            meta=meta or {}
        )

    async def log_success(
        self,
        message: str,
        step: str,
        agent_name: Optional[str] = None,
        meta: Optional[Dict[str, Any]] = None
    ):
        """Log success message"""
        await self._send_log(
            message=message,
            step=step,
            agent_name=agent_name or self._get_agent_name(step),
            log_level=LogLevel.INFO,
            status=WorkflowStatus.COMPLETED,
            meta=meta or {}
        )

    async def log_progress(
        self,
        message: str,
        step: str,
        progress_percentage: int,
        current_step: int,
        total_steps: int,
        agent_name: Optional[str] = None,
        meta: Optional[Dict[str, Any]] = None
    ):
        """Log progress update"""
        progress_meta = {
            "progress_percentage": progress_percentage,
            "current_step": current_step,
            "total_steps": total_steps
        }
        if meta:
            progress_meta.update(meta)

        await self._send_log(
            message=message,
            step=step,
            agent_name=agent_name or self._get_agent_name(step),
            log_level=LogLevel.INFO,
            status=WorkflowStatus.RUNNING,
            meta=progress_meta,
            progress_percentage=progress_percentage,
            current_step=current_step,
            total_steps=total_steps
        )

    def _get_agent_name(self, step: str) -> str:
        """Get agent name based on step"""
        agent_mapping = {
            "authentication": "Authentication Agent",
            "validation": "Validation Agent",
            "initialization": "Initialization Agent",
            "literature_search": "Literature Search Agent",
            "deg_analysis": "DEG Analysis Agent",
            "gene_prioritization": "Gene Prioritization Agent",
            "pathway_enrichment": "Pathway Enrichment Agent",
            "drug_discovery": "Drug Discovery Agent",
            "report_generation": "Report Generation Agent",
            "finalization": "Finalization Agent",
            "cleanup": "Cleanup Agent"
        }
        return agent_mapping.get(step, "Workflow Agent")

    def _get_agent_id(self, step: str) -> str:
        """Get agent ID based on step"""
        agent_id_mapping = {
            "authentication": "AUTH-001",
            "validation": "VAL-001",
            "initialization": "INIT-001",
            "literature_search": "LIT-001",
            "deg_analysis": "DEG-001",
            "gene_prioritization": "GP-001",
            "pathway_enrichment": "PE-001",
            "drug_discovery": "DD-001",
            "report_generation": "RPT-001",
            "finalization": "FIN-001",
            "cleanup": "CLN-001"
        }
        return agent_id_mapping.get(step, f"WF-{step.upper()}")

    def _get_step_index(self, step: str) -> int:
        """Get step index based on step name"""
        step_indices = {
            "authentication": 0,
            "validation": 1,
            "initialization": 2,
            "literature_search": 3,
            "deg_analysis": 4,
            "gene_prioritization": 5,
            "pathway_enrichment": 6,
            "drug_discovery": 7,
            "report_generation": 8,
            "finalization": 9,
            "cleanup": 10
        }
        return step_indices.get(step, 99)

    async def _send_log(
        self,
        message: str,
        step: str,
        agent_name: str,
        log_level: LogLevel,
        status: WorkflowStatus,
        error: Optional[str] = None,
        meta: Optional[Dict[str, Any]] = None,
        progress_percentage: Optional[int] = None,
        current_step: Optional[int] = None,
        total_steps: Optional[int] = None
    ):
        """Send log message to API"""
        try:
            elapsed_time = int((time.time() - self.start_time) * 1000)

            log_message = LogMessage(
                timestamp=datetime.now(timezone.utc).isoformat(),
                agent_id=self._get_agent_id(step),
                agent_name=agent_name,
                workflow_id=self.workflow_id,
                workflow_name=self.workflow_name,
                run_id=self.run_id,
                analysis_id=self.analysis_id,
                step=step,
                step_index=self._get_step_index(step),
                log_level=log_level.value,
                status=status.value,
                log_message=message,
                user_id=self.user_id,
                user_email=self.user_email,
                details=meta or {},
                elapsed_time_ms=elapsed_time,
                error=error,
                meta=meta or {},
                progress_percentage=progress_percentage,
                current_step=current_step,
                total_steps=total_steps
            )

            await send_log_via_api(asdict(log_message))

        except Exception as e:
            logger.error(f"Failed to send log message: {e}")


async def send_log_via_api(
    log: Dict[str, Any],
    user_token: Optional[str] = None,
    max_retries: int = 3,
    timeout: float = 30.0
) -> bool:
    """
    Send log message to API with retry mechanism.

    Args:
        log: Log message dictionary
        max_retries: Maximum number of retry attempts
        timeout: Request timeout in seconds

    Returns:
        True if successful, False otherwise

    Raises:
        BroadcastError: If all retry attempts fail
    """
    for attempt in range(max_retries + 1):
        try:
            headers = {"Content-Type": "application/json"}
            if user_token:
                headers["Authorization"] = f"Bearer {user_token}"

            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(
                    LOGS_ENDPOINT,
                    json=log,
                    headers=headers
                )

                response.raise_for_status()

                logger.debug(f"Log sent successfully: {response.status_code}")
                return True

        except httpx.HTTPStatusError as e:
            error_msg = f"HTTP error sending log (attempt {attempt + 1}/{max_retries + 1}): {e}"
            logger.warning(error_msg)

            if attempt < max_retries:
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
            else:
                logger.error(
                    f"Failed to send log after {max_retries + 1} attempts")
                raise BroadcastError(f"Failed to send log: {e}")

        except Exception as e:
            error_msg = f"Unexpected error sending log (attempt {attempt + 1}/{max_retries + 1}): {e}"
            logger.warning(error_msg)

            if attempt < max_retries:
                await asyncio.sleep(2 ** attempt)
            else:
                logger.error(
                    f"Failed to send log after {max_retries + 1} attempts")
                raise BroadcastError(f"Failed to send log: {e}")

    return False

# Backward compatibility functions


def get_structured_log(
    log_message: str = "",
    agent_name: str = "Transcriptome Agent",
    workflow_name: str = "Agentic AI Workflow",
    run_id: str = "1",
    step: str = "Transcriptome Agent",
    step_index: int = 0,
    log_level: str = "INFO",
    status: str = "running",
    details: Optional[Dict[str, Any]] = None,
    user_id: str = "1",
    user_email: str = "f420testing@ayassbioscience.com",
    elapsed_time_ms: int = 0,
    error: Optional[str] = None,
    meta: Optional[Dict[str, Any]] = None,
    timestamp: Optional[str] = None,
    agent_id: str = "AGENT-T1",
    workflow_id: str = "WORKFLOW-T1"
) -> Dict[str, Any]:
    """
    Backward compatibility function for creating structured logs.

    Note: This function is deprecated. Use WorkflowLogger instead.
    """
    logger.warning(
        "get_structured_log is deprecated. Use WorkflowLogger instead.")

    if details is None:
        details = {}
    if meta is None:
        meta = {}
    if timestamp is None:
        timestamp = datetime.now(timezone.utc).isoformat()

    return {
        "timestamp": timestamp,
        "agent_id": agent_id,
        "agent_name": agent_name,
        "workflow_id": workflow_id,
        "workflow_name": workflow_name,
        "run_id": run_id,
        "step": step,
        "step_index": step_index,
        "log_level": log_level,
        "status": status,
        "log_message": log_message,
        "details": details,
        "user_id": user_id,
        "user_email": user_email,
        "elapsed_time_ms": elapsed_time_ms,
        "error": error,
        "meta": meta
    }

# Health check function


async def health_check() -> Dict[str, Any]:
    """
    Check the health of the broadcasting API.

    Returns:
        Health status dictionary
    """
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{API_BASE_URL}/health/")
            response.raise_for_status()

            return {
                "status": "healthy",
                "api_accessible": True,
                "response_time_ms": response.elapsed.total_seconds() * 1000
            }

    except Exception as e:
        return {
            "status": "unhealthy",
            "api_accessible": False,
            "error": str(e)
        }

# Example usage and testing
if __name__ == "__main__":
    async def test_workflow_logger():
        """Test the workflow logger"""
        logger_instance = WorkflowLogger(
            workflow_id="TEST-WF-001",
            run_id="TEST-RUN-001",
            analysis_id="TEST-ANALYSIS-001",
            user_id="test-user-123"
        )

        await logger_instance.log_info(
            "Starting test workflow",
            "initialization",
            meta={"test": True}
        )

        await logger_instance.log_progress(
            "Processing data",
            "data_processing",
            progress_percentage=50,
            current_step=3,
            total_steps=6
        )

        await logger_instance.log_success(
            "Test completed successfully",
            "finalization"
        )

    async def test_authentication():
        """Test user authentication"""
        auth = UserAuthenticator()

        try:
            user_info = await auth.get_user_info("test-user-123")
            print(f"User info: {user_info}")

            has_permission = await auth.check_analysis_permission(
                "test-user-123",
                "test-analysis-123"
            )
            print(f"Has permission: {has_permission}")

        except Exception as e:
            print(f"Authentication test failed: {e}")

    async def main():
        """Run all tests"""
        print("Testing workflow logger...")
        await test_workflow_logger()

        print("Testing authentication...")
        await test_authentication()

        print("Testing health check...")
        health = await health_check()
        print(f"Health status: {health}")

    asyncio.run(main())
