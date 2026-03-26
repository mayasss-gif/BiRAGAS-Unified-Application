"""
Configuration for the DEG Pipeline Agent.
"""
import re
import os
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass, field


@dataclass
class DEGPipelineConfig:
    """Configuration for the DEG Pipeline Agent."""
    
    # Base paths
    base_project_dir: str = field(default_factory=lambda: os.getenv("AGENTIC_AI_BASE_DIR", "agentic_ai_wf"))
    shared_data_dir: str = field(default_factory=lambda: os.getenv("SHARED_DATA_DIR", "shared"))
    deg_data_dir: str = field(default_factory=lambda: os.getenv("DEG_DATA_DIR", "deg_data"))
    
    # Input/Output paths
    geo_dir: Optional[str] = None
    analysis_transcriptome_dir: Optional[str] = None
    output_dir: Optional[str] = None
    disease_name: Optional[str] = 'unknown_disease'
    user_query: Optional[str] = None  # For relevancy ranking
    
    # Analysis parameters
    analysis_id: Optional[str] = 'unknown_analysis'
    min_samples_per_gene: int = 2
    max_genes: int = 50000
    padj_threshold: float = 0.05
    log2fc_threshold: float = 0.0
    
    # Retry and error handling
    max_retries: int = 3
    retry_delay: float = 1.0
    enable_auto_fix: bool = True
    
    # File processing
    supported_formats: List[str] = field(default_factory=lambda: [".csv", ".tsv", ".txt", ".xlsx", ".xls"])
    compression_formats: List[str] = field(default_factory=lambda: [".gz", ".bz2"])
    
    # Metadata extraction
    metadata_keywords: Tuple[str, ...] = field(default_factory=lambda: (
        "genotype", "tissue", "disease", "disease state", "treatment", 
        "cell type", "etiology", "bclc stage", "condition", "group", "metadata", "series_matrix"
    ))
    
    # Gene mapping
    gene_scopes: List[str] = field(default_factory=lambda: ["ensembl.gene", "entrezgene"])
    gene_fields: str = "symbol"
    gene_species: str = "human"
    
    # Logging
    log_level: str = "INFO"
    log_to_file: bool = True
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Agent behavior
    agent_name: str = "DEGPipelineAgent"
    agent_description: str = "Robust DEG analysis pipeline with self-healing capabilities"
    enable_monitoring: bool = True
    
    # Resource limits
    max_memory_gb: Optional[float] = None
    max_processing_time_minutes: Optional[int] = None
    
    def __post_init__(self):
        """Validate and set default paths."""
        # Set default output directory if not provided
        if self.output_dir is None:
            disease_part = self._sanitize_name(self.disease_name) if self.disease_name else "unknown_disease"
            analysis_part = self.analysis_id if self.analysis_id else "unknown_analysis"
            self.output_dir = str(Path(self.base_project_dir) / self.shared_data_dir / self.deg_data_dir / disease_part / analysis_part)
        
        # Ensure paths are Path objects
        if self.geo_dir:
            self.geo_dir = str(Path(self.geo_dir).expanduser().resolve())
        if self.analysis_transcriptome_dir:
            self.analysis_transcriptome_dir = str(Path(self.analysis_transcriptome_dir).expanduser().resolve())
        if self.output_dir:
            self.output_dir = str(Path(self.output_dir).expanduser().resolve())
    
    def _sanitize_name(self, name: Optional[str]) -> str:
        """Sanitize a name for filesystem usage."""
        if name is None:
            return "unnamed"
        
        # Convert to lowercase and replace spaces with underscores
        sanitized = name.lower().replace(' ', '_')
        # Replace invalid characters with underscores
        sanitized = re.sub(r'[<>:"/\\|?*]', '_', sanitized)
        # Remove multiple consecutive underscores
        sanitized = re.sub(r'_+', '_', sanitized)
        # Remove leading/trailing underscores
        sanitized = sanitized.strip('_')
        # Ensure it's not empty
        if not sanitized:
            sanitized = "unnamed"
        return sanitized
    
    def _get_output_dir(self) -> Path:
        """Get the output directory as a Path object."""
        # Use the already constructed output_dir from __post_init__
        return Path(self.output_dir) if self.output_dir else Path(".")
    
    def get_disease_output_dir(self) -> Path:
        """Get the disease-specific output directory."""
        return self._get_output_dir()
    
    def get_shared_gse_dir(self) -> Path:
        """Get the shared GSE directory for the disease."""
        disease_part = self._sanitize_name(self.disease_name) if self.disease_name else "unknown_disease"
        return Path(self.base_project_dir) / self.shared_data_dir / self.deg_data_dir / disease_part / "shared_gse"
    
    def get_analysis_output_dir(self) -> Path:
        """Get the analysis-specific output directory (without GSE dirs)."""
        return self._get_output_dir()
    
    def should_use_shared_gse(self, gse_id: str) -> tuple[bool, Path]:
        """
        Check if GSE should use shared directory and return the path.
        
        Args:
            gse_id: GSE identifier (e.g., 'GSE123456')
            
        Returns:
            Tuple of (should_use_shared, shared_path)
        """
        if not gse_id or not gse_id.startswith('GSE'):
            return False, Path()
        
        shared_gse_dir = self.get_shared_gse_dir()
        shared_gse_path = shared_gse_dir / gse_id
        
        # Use shared directory if it exists (already processed) or if we should create it
        return True, shared_gse_path
    

    def _validate(self) -> None:
        """Validate the configuration."""
        errors = []
        
        # Check required fields
        if not self.geo_dir and not self.analysis_transcriptome_dir:
            errors.append("Either geo_dir or analysis_transcriptome_dir must be specified")
        
        # Check paths exist
        if self.geo_dir and not Path(self.geo_dir).exists():
            errors.append(f"GEO directory does not exist: {self.geo_dir}")
        
        if self.analysis_transcriptome_dir and not Path(self.analysis_transcriptome_dir).exists():
            errors.append(f"Input transcriptome directory does not exist: {self.analysis_transcriptome_dir}")
        
        if self.analysis_id is None:
            errors.append("analysis_id must be specified")
        
        if self.disease_name is None:
            errors.append("disease_name must be specified")
        
        # Check numeric parameters
        if self.min_samples_per_gene < 1:
            errors.append("min_samples_per_gene must be >= 1")
        
        if self.max_genes < 100:
            errors.append("max_genes must be >= 100")
        
        if self.padj_threshold < 0 or self.padj_threshold > 1:
            errors.append("padj_threshold must be between 0 and 1")
        
        if self.max_retries < 1:
            errors.append("max_retries must be >= 1")
        
        if self.retry_delay < 0:
            errors.append("retry_delay must be >= 0")
        
        if errors:
            raise ValueError(f"Configuration validation failed: {'; '.join(errors)}")
    
    def _to_dict(self) -> Dict:
        """Convert configuration to dictionary."""
        return {
            "geo_dir": self.geo_dir,
            "analysis_transcriptome_dir": self.analysis_transcriptome_dir,
            "output_dir": self.output_dir,
            "disease_name": self.disease_name,
            "analysis_id": self.analysis_id,
            "min_samples_per_gene": self.min_samples_per_gene,
            "max_genes": self.max_genes,
            "padj_threshold": self.padj_threshold,
            "log2fc_threshold": self.log2fc_threshold,
            "max_retries": self.max_retries,
            "retry_delay": self.retry_delay,
            "enable_auto_fix": self.enable_auto_fix,
            "supported_formats": self.supported_formats,
            "compression_formats": self.compression_formats,
            "metadata_keywords": self.metadata_keywords,
            "gene_scopes": self.gene_scopes,
            "gene_fields": self.gene_fields,
            "gene_species": self.gene_species,
            "log_level": self.log_level,
            "log_to_file": self.log_to_file,
            "agent_name": self.agent_name,
            "agent_description": self.agent_description,
            "enable_monitoring": self.enable_monitoring,
            "max_memory_gb": self.max_memory_gb,
            "max_processing_time_minutes": self.max_processing_time_minutes
        }
    
    @classmethod
    def _from_dict(cls, config_dict: Dict) -> "DEGPipelineConfig":
        """Create configuration from dictionary."""
        return cls(**config_dict)
    
    @classmethod
    def _default(cls) -> "DEGPipelineConfig":
        """Create default configuration."""
        return cls()
    
    def _create_work_directories(self) -> None:
        """Create necessary work directories."""
        if self.output_dir:
            Path(self.output_dir).mkdir(parents=True, exist_ok=True) 