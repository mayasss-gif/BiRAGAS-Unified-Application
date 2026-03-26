"""
Configuration for LLM log enhancement feature.

This module provides centralized control over the log enhancement feature,
allowing it to be enabled/disabled globally or per-environment.
"""

import os
from typing import Optional

# Feature flags
ENABLE_LLM_LOG_ENHANCEMENT = os.getenv(
    "ENABLE_LLM_LOG_ENHANCEMENT", 
    "true"
).lower() in ("true", "1", "yes")

# LLM configuration
LLM_ENHANCEMENT_MODEL = os.getenv(
    "LLM_ENHANCEMENT_MODEL", 
    "gpt-4o-mini"
)

LLM_ENHANCEMENT_TIMEOUT = float(os.getenv(
    "LLM_ENHANCEMENT_TIMEOUT", 
    "3.0"
))

LLM_ENHANCEMENT_MAX_TOKENS = int(os.getenv(
    "LLM_ENHANCEMENT_MAX_TOKENS", 
    "300"
))

# Skip enhancement for certain log types (to save costs)
SKIP_ENHANCEMENT_FOR_STEPS = {
    "workflow_start",
    "workflow_complete",
}

SKIP_ENHANCEMENT_FOR_AGENTS = set()  # Add agent names to skip


def should_enhance_log(
    step: str = "",
    agent_name: str = "",
    log_level: str = "INFO"
) -> bool:
    """
    Determine if a log should be enhanced based on config.
    
    Args:
        step: Log step name
        agent_name: Agent name
        log_level: Log level
    
    Returns:
        True if log should be enhanced
    """
    # Global toggle
    if not ENABLE_LLM_LOG_ENHANCEMENT:
        return False
    
    # Skip certain steps
    if step in SKIP_ENHANCEMENT_FOR_STEPS:
        return False
    
    # Skip certain agents
    if agent_name in SKIP_ENHANCEMENT_FOR_AGENTS:
        return False
    
    # Only enhance INFO/WARNING/ERROR (skip DEBUG)
    if log_level not in ("INFO", "WARNING", "ERROR"):
        return False
    
    return True

