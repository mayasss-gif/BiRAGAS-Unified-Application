"""
Configuration Settings for Autonomous Pathway Analysis Agent

Pydantic-based configuration management for the LangGraph agent system
with environment variable support and validation.
"""

import os
from typing import Optional
from decouple import config
from pydantic_settings import BaseSettings, SettingsConfigDict

os.environ["OPENAI_API_KEY"] = config("OPENAI_API_KEY")
class AgentConfig(BaseSettings):
    """Configuration settings for the autonomous pathway analysis agent"""
    
    # Retry and error handling
    max_retries: int = 3
    retry_backoff_base: int = 2
    retry_delay_max: int = 60  # Maximum delay in seconds
    
    # Checkpointing and state management
    enable_checkpointing: bool = True
    checkpoint_interval: int = 5  # Save checkpoint every N stages
    
    # Logging configuration
    log_level: str = "INFO"
    log_file: str = "logs/agentic_workflow.log"
    enable_console_logging: bool = True
    
    # LLM configuration
    openai_api_key: Optional[str] = os.getenv("OPENAI_API_KEY")
    openai_model: str = "gpt-4o"
    openai_temperature: float = 0.1
    openai_max_tokens: int = 1000
    
    # Pipeline configuration
    default_batch_size: int = 10
    default_top_n_pathways: int = 100
    default_patient_prefix: str = "59d6bcdc-9333-4aae-894b"
    
    # Output configuration
    output_directory: str = "ALL_ANALYSIS"
    enable_progress_tracking: bool = True
    progress_update_interval: int = 10  # Update progress every N seconds
    
    # Performance configuration
    max_concurrent_workers: int = 10
    memory_optimization: bool = True
    enable_parallel_processing: bool = True
    
    # API configuration
    api_timeout: int = 30
    api_rate_limit: int = 60  # Requests per minute
    enable_api_caching: bool = True
    
    model_config = SettingsConfigDict(
        env_prefix="PATHWAY_AGENT_",
        case_sensitive=False,
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


# Global configuration instance
config = AgentConfig()


def get_config() -> AgentConfig:
    """Get the global configuration instance"""
    return config


def update_config(**kwargs) -> None:
    """Update configuration settings"""
    global config
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            raise ValueError(f"Unknown configuration parameter: {key}")


def validate_config() -> bool:
    """Validate configuration settings"""
    try:
        # Check required directories
        os.makedirs(config.output_directory, exist_ok=True)
        os.makedirs(os.path.dirname(config.log_file), exist_ok=True)
        
        # Check for required API key
        if not config.openai_api_key:
            print("Warning: OPENAI_API_KEY not found in environment variables")
            print("Some features may not work without a valid API key")
        
        # Validate numeric ranges
        assert 1 <= config.max_retries <= 10, "max_retries must be between 1 and 10"
        assert 1 <= config.retry_backoff_base <= 10, "retry_backoff_base must be between 1 and 10"
        assert 0.0 <= config.openai_temperature <= 2.0, "openai_temperature must be between 0.0 and 2.0"
        assert 1 <= config.max_concurrent_workers <= 50, "max_concurrent_workers must be between 1 and 50"
        
        return True
    except Exception as e:
        print(f"Configuration validation failed: {e}")
        return False
