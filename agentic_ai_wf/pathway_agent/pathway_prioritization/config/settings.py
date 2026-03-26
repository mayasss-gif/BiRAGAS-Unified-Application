# src/pathway_prioritization/config/settings.py
import os
import logging
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

class Settings:
    """Application settings and configuration"""
    
    def __init__(self):
        self.openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
        self.openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o")
        self.max_workers: int = int(os.getenv("MAX_WORKERS", "5"))
        self.batch_size: int = int(os.getenv("BATCH_SIZE", "10"))
        self.log_level: str = os.getenv("LOG_LEVEL", "INFO")
        
        # LLM Configuration
        self.max_tokens: int = 1000
        self.temperature: float = 0.2
        self.top_p: float = 0.9
        
        # Paths
        self.base_dir: Path = Path(__file__).parent.parent.parent.parent
        self.data_dir: Path = self.base_dir / "data"
        self.output_dir: Path = self.base_dir / "results"
        self.cache_dir: Path = self.data_dir / "cache"
        
    def validate(self) -> bool:
        """Validate settings"""
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        return True
    
    def setup_directories(self) -> None:
        """Create necessary directories with proper error handling"""
        directories = [self.data_dir, self.output_dir, self.cache_dir]
        
        for directory in directories:
            try:
                # If it exists as a file, remove it first
                if directory.exists() and directory.is_file():
                    logger.warning(f"Removing file {directory} to create directory")
                    directory.unlink()
                
                # Create directory (including parents)
                directory.mkdir(parents=True, exist_ok=True)
                logger.debug(f"Directory ensured: {directory}")
                
            except Exception as e:
                logger.error(f"Error creating directory {directory}: {e}")
                raise

# Global settings instance
settings = Settings()



 # src/pathway_prioritization/config/settings.py
# import os
# from pathlib import Path
# from typing import Optional
# from dotenv import load_dotenv

# # Load environment variables
# load_dotenv()

# class Settings:
#     """Application settings and configuration"""
    
#     def __init__(self):
#         self.openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
#         self.openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o")
#         self.max_workers: int = int(os.getenv("MAX_WORKERS", "5"))
#         self.batch_size: int = int(os.getenv("BATCH_SIZE", "10"))
#         self.log_level: str = os.getenv("LOG_LEVEL", "INFO")
        
#         # LLM Configuration
#         self.max_tokens: int = 1000
#         self.temperature: float = 0.2
#         self.top_p: float = 0.9
        
#         # Paths
#         self.base_dir: Path = Path(__file__).parent.parent.parent.parent
#         self.data_dir: Path = self.base_dir / "data"
#         self.output_dir: Path = self.base_dir / "results"
#         self.cache_dir: Path = self.data_dir / "cache"
        
#     def validate(self) -> bool:
#         """Validate settings"""
#         if not self.openai_api_key:
#             raise ValueError("OPENAI_API_KEY environment variable is required")
#         return True
    
#     def setup_directories(self) -> None:
#         """Create necessary directories"""
#         self.data_dir.mkdir(exist_ok=True)
#         self.output_dir.mkdir(exist_ok=True)
#         self.cache_dir.mkdir(exist_ok=True)

# # Global settings instance
# settings = Settings()
