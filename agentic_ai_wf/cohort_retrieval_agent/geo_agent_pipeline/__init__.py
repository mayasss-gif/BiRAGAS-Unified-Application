"""geo_agent_pipeline
Cohort Retrieval Agent - A robust system for retrieving biomedical datasets.

This package provides a comprehensive, scalable agent system for downloading
and organizing biomedical datasets from multiple sources including GEO, SRA,
TCGA, GTEx, and ArrayExpress.

Main Components:
- CohortRetrievalAgent: Main orchestrator for multi-source retrieval
- Data source agents: GEO, SRA, TCGA, GTEx, ArrayExpress
- Specialized tools: Query, Filter, Download, Metadata, Validation
- Configuration management with validation
- Progress tracking and monitoring

Example Usage:
    import asyncio
    from   import CohortRetrievalAgent
    
    async def main():
        agent = CohortRetrievalAgent()
        result = await agent.retrieve_cohort("pancreatic cancer", max_datasets_per_source=5)
        print(f"Downloaded {result.total_datasets_downloaded} datasets")
    
    asyncio.run(main())

CLI Usage:
    python -m  .cli "pancreatic cancer" --max-datasets 5
"""

# Main agent
from .agent import CohortRetrievalAgent, CohortResult

# Configuration
from .config import CohortRetrievalConfig

# Data source agents
from .agents import (
    GEORetrievalAgent
)

# Base classes
from .base.base_agent import BaseRetrievalAgent, AgentResult, DatasetInfo
from .base.base_tool import BaseTool

# Tools
from .tools import (
    QueryTool,
    FilterTool,
    DownloadTool,
    MetadataTool,
    ValidationTool
)

# Exceptions
from .exceptions import (
    CohortRetrievalError,
    AgentError,
)

# Version
__version__ = "1.0.0"

# Main exports
__all__ = [
    # Main classes
    "CohortRetrievalAgent",
    "CohortResult",
    "CohortRetrievalConfig",
    
    # Agents
    "GEORetrievalAgent",
   
    
    # Base classes
    "BaseRetrievalAgent",
    "AgentResult",
    "DatasetInfo",
    "BaseTool",
    
    # Tools
    "QueryTool",
    "FilterTool",
    "DownloadTool",
    "MetadataTool",
    "ValidationTool",
    
    # Exceptions
    "CohortRetrievalError",
    "AgentError",
  
    # CLI
    "geo_pipeline"
] 