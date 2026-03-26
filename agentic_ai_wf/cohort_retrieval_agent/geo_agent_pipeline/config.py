"""
Configuration management for the Cohort Retrieval Agent system.

This module provides centralized configuration management with validation,
environment variable support, and extensible settings for multiple data sources.
"""
import re
import os
import json
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from pathlib import Path

from .exceptions import ConfigurationError

class AgentName(Enum):
    GEO = "geo"
    SRA = "sra"
    TCGA = "tcga"
    GTEx = "gtex"
    ArrayExpress = "arrayexpress"

@dataclass
class DirectoryPathsConfig:
    """
    Comprehensive directory paths configuration for cohort retrieval agents.
    
    Provides scalable, customizable, manageable, and trackable directory structures
    following the pattern: agentic_ai_wf/shared/cohort_data/AgentName/DiseaseName/
    """
    
    # Base paths
    base_project_dir: str = field(default_factory=lambda: os.getenv("AGENTIC_AI_BASE_DIR", "./agentic_ai_wf"))
    shared_data_dir: str = field(default_factory=lambda: os.getenv("SHARED_DATA_DIR", "shared"))
    cohort_data_dir: str = field(default_factory=lambda: os.getenv("COHORT_DATA_DIR", "cohort_data"))
    ontology_dir : str = field(default_factory = lambda:os.getenv("COHORT_ONTO_DIR","./agentic_ai_wf/shared/cohort_data/GEO"))
    invalid_dir : str = field(default_factory = lambda:os.getenv("COHORT_INV_DIR","./agentic_ai_wf/cohort_retrieval_agent/invalid/GEO"))
    # Agent-specific directories
    geo_agent_dir: str = "GEO"
    sra_agent_dir: str = "SRA"
    tcga_agent_dir: str = "TCGA"
    gtex_agent_dir: str = "GTEx"
    arrayexpress_agent_dir: str = "ArrayExpress"
    
    # Analysis directories for all diseases (shared/analysis/disease_name)
    analysis_dir: str = "analysis"
    
    # Subdirectory structure for each agent/disease combination
    raw_data_subdir: str = "raw_data"
    processed_data_subdir: str = "processed_data"
    metadata_subdir: str = "metadata"
    logs_subdir: str = "logs"
    temp_subdir: str = "temp"
    cache_subdir: str = "cache"
    reports_subdir: str = "reports"
    validation_subdir: str = "validation"
    
    # File naming patterns
    dataset_summary_filename: str = "dataset_summary.json"
    download_log_filename: str = "download_log.txt"
    validation_report_filename: str = "validation_report.json"
    metadata_index_filename: str = "metadata_index.json"
    
    # Archive and backup settings
    enable_archiving: bool = True
    archive_subdir: str = "archive"
    backup_subdir: str = "backup"
    max_archive_age_days: int = 30
    
    # Tracking and monitoring
    enable_tracking: bool = True
    tracking_subdir: str = "tracking"
    usage_stats_filename: str = "usage_stats.json"
    performance_metrics_filename: str = "performance_metrics.json"
    
    def get_base_cohort_path(self) -> Path:
        """Get the base cohort data path."""
        return Path(self.base_project_dir) / self.shared_data_dir / self.cohort_data_dir
    
    def get_agent_base_path(self, agent_name: str) -> Path:
        """Get the base path for a specific agent."""
        agent_dir_map = {
            AgentName.GEO.value: self.geo_agent_dir,
            AgentName.SRA.value: self.sra_agent_dir,
            AgentName.TCGA.value: self.tcga_agent_dir,
            AgentName.GTEx.value: self.gtex_agent_dir,
            AgentName.ArrayExpress.value: self.arrayexpress_agent_dir
        }
        
        agent_dir = agent_dir_map.get(agent_name.lower(), agent_name.upper())
        return self.get_base_cohort_path() / agent_dir
 
    def get_disease_path(self, agent_name: str, disease_name: str, filter_str: Optional[str] = None) -> Path:
        """Get the path for a specific agent and disease combination, optionally adding a filter suffix."""
        if filter_str:
            name = filter_str
        else:
            name = disease_name

        # sanitized_disease = self._sanitize_name(name)
        sanitized_disease = name
        return self.get_agent_base_path(agent_name) / sanitized_disease
  
    def get_analysis_path(self,disease_name: str) -> Path:
        """Get the analysis path for agent/disease."""
        sanitized_disease = self._sanitize_name(disease_name)
        return Path(self.base_project_dir) / self.shared_data_dir / self.analysis_dir / sanitized_disease
    

    def get_raw_data_path(self, agent_name: str, disease_name: str, filter_str: Optional[str] = None) -> Path:
        return self.get_disease_path(agent_name, disease_name, filter_str) / self.raw_data_subdir
    
    def get_processed_data_path(self, agent_name: str, disease_name: str) -> Path:
        """Get the processed data path for agent/disease."""
        return self.get_disease_path(agent_name, disease_name) / self.processed_data_subdir
    
    def get_metadata_path(self, agent_name: str, disease_name: str) -> Path:
        """Get the metadata path for agent/disease."""
        return self.get_disease_path(agent_name, disease_name) / self.metadata_subdir
    
    def get_logs_path(self, agent_name: str, disease_name: str) -> Path:
        """Get the logs path for agent/disease."""
        return self.get_disease_path(agent_name, disease_name) / self.logs_subdir
    
    def get_temp_path(self, agent_name: str, disease_name: str) -> Path:
        """Get the temporary files path for agent/disease."""
        return self.get_disease_path(agent_name, disease_name) / self.temp_subdir
    
    def get_cache_path(self, agent_name: str, disease_name: str) -> Path:
        """Get the cache path for agent/disease."""
        return self.get_disease_path(agent_name, disease_name) / self.cache_subdir
    
    def get_reports_path(self, agent_name: str, disease_name: str) -> Path:
        """Get the reports path for agent/disease."""
        return self.get_disease_path(agent_name, disease_name) / self.reports_subdir
    
    def get_validation_path(self, agent_name: str, disease_name: str) -> Path:
        """Get the validation path for agent/disease."""
        return self.get_disease_path(agent_name, disease_name) / self.validation_subdir
    
    def get_archive_path(self, agent_name: str, disease_name: str) -> Path:
        """Get the archive path for agent/disease."""
        return self.get_disease_path(agent_name, disease_name) / self.archive_subdir
    
    def get_backup_path(self, agent_name: str, disease_name: str) -> Path:
        """Get the backup path for agent/disease."""
        return self.get_disease_path(agent_name, disease_name) / self.backup_subdir
    
    def get_tracking_path(self, agent_name: str, disease_name: str) -> Path:
        """Get the tracking path for agent/disease."""
        return self.get_disease_path(agent_name, disease_name) / self.tracking_subdir
    
    def get_dataset_path(self, agent_name: str, disease_name: str, dataset_id: str) -> Path:
        """Get the path for a specific dataset."""
        sanitized_dataset_id = self._sanitize_name(dataset_id)
        return self.get_raw_data_path(agent_name, disease_name) / sanitized_dataset_id
    
    def get_all_paths(self, agent_name: str, disease_name: str) -> Dict[str, Path]:
        """Get all paths for an agent/disease combination."""
        return {
            "base": self.get_disease_path(agent_name, disease_name),
            "raw_data": self.get_raw_data_path(agent_name, disease_name),
            "processed_data": self.get_processed_data_path(agent_name, disease_name),
            "metadata": self.get_metadata_path(agent_name, disease_name),
            "logs": self.get_logs_path(agent_name, disease_name),
            "temp": self.get_temp_path(agent_name, disease_name),
            "cache": self.get_cache_path(agent_name, disease_name),
            "reports": self.get_reports_path(agent_name, disease_name),
            "validation": self.get_validation_path(agent_name, disease_name),
            "archive": self.get_archive_path(agent_name, disease_name),
            "backup": self.get_backup_path(agent_name, disease_name),
            "tracking": self.get_tracking_path(agent_name, disease_name)
        }
    
    def create_all_directories(self, agent_name: str, disease_name: str) -> Dict[str, Path]:
        """Create all directories for an agent/disease combination."""
        paths = self.get_all_paths(agent_name, disease_name)
        
        created_paths = {}
        for path_type, path in paths.items():
            try:
                path.mkdir(parents=True, exist_ok=True)
                created_paths[path_type] = path
            except Exception as e:
                raise ConfigurationError(f"Failed to create {path_type} directory {path}: {e}")
        
        return created_paths
    
    def get_file_path(self, agent_name: str, disease_name: str, file_type: str, filename: str = None) -> Path:
        """Get the full path for a specific file type."""
        file_map = {
            "dataset_summary": (self.get_metadata_path(agent_name, disease_name), self.dataset_summary_filename),
            "download_log": (self.get_logs_path(agent_name, disease_name), self.download_log_filename),
            "validation_report": (self.get_validation_path(agent_name, disease_name), self.validation_report_filename),
            "metadata_index": (self.get_metadata_path(agent_name, disease_name), self.metadata_index_filename),
            "usage_stats": (self.get_tracking_path(agent_name, disease_name), self.usage_stats_filename),
            "performance_metrics": (self.get_tracking_path(agent_name, disease_name), self.performance_metrics_filename)
        }
        
        if file_type in file_map:
            directory, default_filename = file_map[file_type]
            return directory / (filename or default_filename)
        else:
            raise ConfigurationError(f"Unknown file type: {file_type}")
    
    def get_directory_structure(self, agent_name: str, disease_name: str) -> Dict[str, Any]:
        """Get a complete directory structure representation."""
        paths = self.get_all_paths(agent_name, disease_name)
        
        structure = {
            "agent_name": agent_name,
            "disease_name": disease_name,
            "base_path": str(paths["base"]),
            "directories": {
                path_type: {
                    "path": str(path),
                    "exists": path.exists(),
                    "size_mb": self._get_directory_size_mb(path) if path.exists() else 0
                }
                for path_type, path in paths.items()
            },
            "files": {
                "dataset_summary": str(self.get_file_path(agent_name, disease_name, "dataset_summary")),
                "download_log": str(self.get_file_path(agent_name, disease_name, "download_log")),
                "validation_report": str(self.get_file_path(agent_name, disease_name, "validation_report")),
                "metadata_index": str(self.get_file_path(agent_name, disease_name, "metadata_index")),
                "usage_stats": str(self.get_file_path(agent_name, disease_name, "usage_stats")),
                "performance_metrics": str(self.get_file_path(agent_name, disease_name, "performance_metrics"))
            }
        }
        
        return structure
    
    def list_all_agents(self) -> List[str]:
        """List all agents that have data directories."""
        base_path = self.get_base_cohort_path()
        if not base_path.exists():
            return []
        
        agents = []
        for item in base_path.iterdir():
            if item.is_dir():
                agents.append(item.name)
        
        return sorted(agents)
    
    def list_diseases_for_agent(self, agent_name: str) -> List[str]:
        """List all diseases for a specific agent."""
        agent_path = self.get_agent_base_path(agent_name)
        if not agent_path.exists():
            return []
        
        diseases = []
        for item in agent_path.iterdir():
            if item.is_dir():
                diseases.append(item.name)
        
        return sorted(diseases)
    
    def get_storage_summary(self) -> Dict[str, Any]:
        """Get a comprehensive storage summary."""
        summary = {
            "base_path": str(self.get_base_cohort_path()),
            "total_size_mb": 0,
            "agents": {}
        }
        
        for agent in self.list_all_agents():
            agent_summary = {
                "diseases": {},
                "total_size_mb": 0
            }
            
            for disease in self.list_diseases_for_agent(agent):
                disease_path = self.get_disease_path(agent, disease)
                disease_size = self._get_directory_size_mb(disease_path)
                agent_summary["diseases"][disease] = {
                    "size_mb": disease_size,
                    "path": str(disease_path)
                }
                agent_summary["total_size_mb"] += disease_size
            
            summary["agents"][agent] = agent_summary
            summary["total_size_mb"] += agent_summary["total_size_mb"]
        
        return summary
    
    def cleanup_old_archives(self, agent_name: str = None, disease_name: str = None):
        """Clean up old archive files based on max_archive_age_days."""
        if not self.enable_archiving:
            return
        
        import time
        cutoff_time = time.time() - (self.max_archive_age_days * 24 * 60 * 60)
        
        if agent_name and disease_name:
            # Clean specific agent/disease
            archive_path = self.get_archive_path(agent_name, disease_name)
            self._cleanup_directory(archive_path, cutoff_time)
        elif agent_name:
            # Clean all diseases for agent
            for disease in self.list_diseases_for_agent(agent_name):
                archive_path = self.get_archive_path(agent_name, disease)
                self._cleanup_directory(archive_path, cutoff_time)
        else:
            # Clean all agents
            for agent in self.list_all_agents():
                for disease in self.list_diseases_for_agent(agent):
                    archive_path = self.get_archive_path(agent, disease)
                    self._cleanup_directory(archive_path, cutoff_time)

    def _sanitize_name(self, name: str) -> str:
        """Sanitize a name for filesystem usage, converting to lowercase and replacing spaces with underscores."""
        
        # Convert to lowercase first
        sanitized = name.lower()
        
        # Replace spaces with underscores
        sanitized = re.sub(r'\s+', '_', sanitized)
        
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
    
    def _get_directory_size_mb(self, directory: Path) -> float:
        """Get the size of a directory in MB."""
        if not directory.exists():
            return 0.0
        
        total_size = 0
        try:
            for item in directory.rglob('*'):
                if item.is_file():
                    total_size += item.stat().st_size
        except (OSError, PermissionError):
            pass
        
        return total_size / (1024 * 1024)  # Convert to MB
    
    def _cleanup_directory(self, directory: Path, cutoff_time: float):
        """Clean up files older than cutoff_time in a directory."""
        if not directory.exists():
            return
        
        try:
            for item in directory.iterdir():
                if item.is_file() and item.stat().st_mtime < cutoff_time:
                    item.unlink()
                elif item.is_dir():
                    self._cleanup_directory(item, cutoff_time)
                    # Remove empty directories
                    try:
                        item.rmdir()
                    except OSError:
                        pass  # Directory not empty
        except (OSError, PermissionError):
            pass


@dataclass
class RetryConfig:
    """Configuration for retry mechanisms."""
    max_retries: int = 3
    base_delay: float = 1.0
    exponential_base: float = 2.0
    max_delay: float = 60.0
    jitter: bool = True


@dataclass
class NetworkConfig:
    """Configuration for network operations."""
    timeout: int = 30
    concurrent_downloads: int = 5
    chunk_size: int = 8192
    user_agent: str = "CohortRetrievalAgent/1.0"


@dataclass
class GEOConfig:
    """Configuration specific to GEO data retrieval."""
    base_url: str = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    ftp_host: str = "ftp.ncbi.nlm.nih.gov"
    db: str = "gds"
    retmode: str = "xml"
    max_datasets: int = 2
    
    # NCBI Entrez email (required by NCBI E-utilities)
    entrez_email: str = field(default_factory=lambda: os.getenv("ENTREZ_EMAIL", "f420testing@ayassbioscience.com"))
    
    # Keywords for filtering
    tissue_keywords: List[str] = field(default_factory=lambda: [
        "tissue", "organ", "biopsy", "surgical", "resection", "primary",
        "tumor", "cancer", "carcinoma", "adenocarcinoma", "metastatic"
    ])
    
    exclude_keywords: List[str] = field(default_factory=lambda: [
        "cell line", "cultured", "in vitro", "immortalized", "transformed",
        "serum", "plasma", "blood", "pbmc", "peripheral blood", "culture"
    ])
    
    # Enhanced supplementary file configuration with categories and priorities
    supplementary_config: Dict[str, Any] = field(default_factory=lambda: {
        # File type categories with priorities (higher = preferred)
        "file_categories": {
            "raw_counts": {
                "keywords": ["raw_counts", "readcounts", "featurecounts", "htseq_counts", "star_counts"],
                "priority": 10,
                "description": "Raw count matrices from RNA-seq quantification",
                "required": True
            },
            "normalized_counts": {
                "keywords": ["norm_counts", "normalized_counts", "vst_counts", "rlog_counts"],
                "priority": 9,
                "description": "Normalized count matrices",
                "required": False
            },
            "expression_measures": {
                "keywords": ["fpkm", "rpkm", "tpm", ".tar","cpm", "tmm"],
                "priority": 8,
                "description": "Expression level measurements",
                "required": False
            },
            "differential_expression": {
                "keywords": ["gene_exp.diff", "diff", "deseq2", "edger", "limma"],
                "priority": 7,
                "description": "Differential expression analysis results",
                "required": False
            },
            "general_counts": {
                "keywords": ["count", "counts", "matrix", "expression"],
                "priority": 6,
                "description": "General count or expression files",
                "required": False
            },
            "metadata": {
                "keywords": ["metadata", "annotation", "phenotype", "clinical"],
                "priority": 5,
                "description": "Sample metadata and annotations",
                "required": False
            }
        },
        
        # File format preferences
        "preferred_formats": [ ".tsv", ".csv", ".gz", ".tar", ".zip"],
        "excluded_formats": [".pdf", ".doc", ".docx", ".ppt", ".pptx", ".jpg", ".png"],
        
        # Validation rules
        "validation_rules": {
            "min_file_size_bytes": 1024,  # 1KB minimum
            "max_file_size_mb": 5000,     # 5GB maximum
            "require_at_least_one_category": True,
            "preferred_categories": ["raw_counts", "normalized_counts", "expression_measures"]
        },
        
        # Download behavior
        "download_behavior": {
            "max_concurrent_downloads": 3,
            "retry_attempts": 3,
            "retry_delay_seconds": 2,
            "timeout_seconds": 300,
            "require_supplementary_files": False,  # Make optional by default
            "fallback_to_series_matrix": True
        }
    })
    
    # Legacy support - maintain backward compatibility
    supplementary_keywords: List[str] = field(default_factory=lambda: [
        "count", "counts", "raw_counts", "readcounts", "featurecounts",
        "norm_counts", "normalized_counts", "FPKM", "RPKM", "TPM", "CPM",
        "TMM", "gene_exp.diff", "diff"
    ])
    
    def get_all_supplementary_keywords(self) -> List[str]:
        """Get all supplementary keywords from all categories."""
        all_keywords = []
        for category_info in self.supplementary_config["file_categories"].values():
            all_keywords.extend(category_info["keywords"])
        return list(set(all_keywords))  # Remove duplicates
    
    def get_keywords_by_priority(self) -> List[Tuple[str, int]]:
        """Get keywords sorted by priority (highest first)."""
        keyword_priority_pairs = []
        for category_info in self.supplementary_config["file_categories"].values():
            priority = category_info["priority"]
            for keyword in category_info["keywords"]:
                keyword_priority_pairs.append((keyword, priority))
        
        return sorted(keyword_priority_pairs, key=lambda x: x[1], reverse=True)
    
    def get_required_categories(self) -> List[str]:
        """Get list of required file categories."""
        return [
            category for category, info in self.supplementary_config["file_categories"].items()
            if info.get("required", False)
        ]
    
    def is_preferred_format(self, filename: str) -> bool:
        """Check if file format is preferred."""
        preferred_formats = self.supplementary_config["preferred_formats"]
        excluded_formats = self.supplementary_config["excluded_formats"]
        
        # Check if file has excluded format
        if any(filename.lower().endswith(ext) for ext in excluded_formats):
            return False
        
        # Check if file has preferred format
        return any(filename.lower().endswith(ext) for ext in preferred_formats)
    
    def categorize_file(self, filename: str) -> Optional[str]:
        """Categorize a file based on its name."""
        filename_lower = filename.lower()
        
        # Find the highest priority category that matches
        best_category = None
        best_priority = -1
        
        for category, info in self.supplementary_config["file_categories"].items():
            for keyword in info["keywords"]:
                if keyword.lower() in filename_lower:
                    if info["priority"] > best_priority:
                        best_category = category
                        best_priority = info["priority"]
        
        return best_category


@dataclass
class SRAConfig:
    """Configuration for SRA data retrieval (placeholder)."""
    base_url: str = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    db: str = "sra"
    enabled: bool = False  # Will be enabled when implemented


@dataclass
class TCGAConfig:
    """Configuration for TCGA data retrieval (placeholder)."""
    base_url: str = "https://api.gdc.cancer.gov"
    enabled: bool = False


@dataclass
class GTExConfig:
    """Configuration for GTEx data retrieval (placeholder)."""
    base_url: str = "https://www.gtexportal.org/api/v2"
    enabled: bool = False


@dataclass
class ArrayExpressConfig:
    """Configuration for ArrayExpress data retrieval (placeholder)."""
    base_url: str = "https://www.ebi.ac.uk/arrayexpress/json/v3"
    enabled: bool = False


@dataclass
class CohortRetrievalConfig:
    """Main configuration for the Cohort Retrieval Agent system."""
    
    # Directory paths configuration
    directory_paths: DirectoryPathsConfig = field(default_factory=DirectoryPathsConfig)
    
    # Legacy directory paths for backward compatibility
    output_dir: str = field(default_factory=lambda: os.getenv("COHORT_OUTPUT_DIR", "./temp"))
    temp_dir: str = field(default_factory=lambda: os.getenv("COHORT_TEMP_DIR", "./temp"))
    log_dir: str = field(default_factory=lambda: os.getenv("COHORT_LOG_DIR", "./logs"))
    # Agent configurations
    retry_config: RetryConfig = field(default_factory=RetryConfig)
    network_config: NetworkConfig = field(default_factory=NetworkConfig)
    geo_config: GEOConfig = field(default_factory=GEOConfig)
    sra_config: SRAConfig = field(default_factory=SRAConfig)
    tcga_config: TCGAConfig = field(default_factory=TCGAConfig)
    gtex_config: GTExConfig = field(default_factory=GTExConfig)
    arrayexpress_config: ArrayExpressConfig = field(default_factory=ArrayExpressConfig)
    
    # General settings
    max_concurrent_agents: int = 3
    progress_update_interval: int = 10  # seconds
    validate_downloads: bool = True
    save_metadata: bool = True
    cleanup_temp_files: bool = True
    
    # Logging
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        self._validate_config()
        self._ensure_directories()
    
    def _validate_config(self):
        """Validate configuration parameters."""
        # Validate retry config
        if self.retry_config.max_retries < 0:
            raise ConfigurationError("max_retries must be non-negative")
        
        if self.retry_config.base_delay <= 0:
            raise ConfigurationError("base_delay must be positive")
        
        # Validate network config
        if self.network_config.timeout <= 0:
            raise ConfigurationError("timeout must be positive")
        
        if self.network_config.concurrent_downloads <= 0:
            raise ConfigurationError("concurrent_downloads must be positive")
        
        # Validate general settings
        if self.max_concurrent_agents <= 0:
            raise ConfigurationError("max_concurrent_agents must be positive")
        
        if self.progress_update_interval <= 0:
            raise ConfigurationError("progress_update_interval must be positive")
        
        # Validate log level
        valid_log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if self.log_level not in valid_log_levels:
            raise ConfigurationError(f"log_level must be one of {valid_log_levels}")
        
        # Validate GEO config
        if not self.geo_config.entrez_email:
            raise ConfigurationError("ENTREZ_EMAIL is required for GEO queries. Please set the ENTREZ_EMAIL environment variable.")
    
    def _ensure_directories(self):
        """Ensure all required directories exist."""
        for dir_path in [self.output_dir, self.temp_dir, self.log_dir]:
            try:
                Path(dir_path).mkdir(parents=True, exist_ok=True)
            except Exception as e:
                raise ConfigurationError(f"Failed to create directory {dir_path}: {e}")
    
    @classmethod
    def from_file(cls, config_path: str) -> "CohortRetrievalConfig":
        """Load configuration from a JSON file."""
        try:
            with open(config_path, 'r') as f:
                config_data = json.load(f)
            
            # Create nested config objects
            if 'retry_config' in config_data:
                config_data['retry_config'] = RetryConfig(**config_data['retry_config'])
            
            if 'network_config' in config_data:
                config_data['network_config'] = NetworkConfig(**config_data['network_config'])
            
            if 'geo_config' in config_data:
                config_data['geo_config'] = GEOConfig(**config_data['geo_config'])
            
            if 'sra_config' in config_data:
                config_data['sra_config'] = SRAConfig(**config_data['sra_config'])
            
            if 'tcga_config' in config_data:
                config_data['tcga_config'] = TCGAConfig(**config_data['tcga_config'])
            
            if 'gtex_config' in config_data:
                config_data['gtex_config'] = GTExConfig(**config_data['gtex_config'])
            
            if 'arrayexpress_config' in config_data:
                config_data['arrayexpress_config'] = ArrayExpressConfig(**config_data['arrayexpress_config'])
            
            return cls(**config_data)
        
        except FileNotFoundError:
            raise ConfigurationError(f"Configuration file not found: {config_path}")
        except json.JSONDecodeError as e:
            raise ConfigurationError(f"Invalid JSON in configuration file: {e}")
        except Exception as e:
            raise ConfigurationError(f"Error loading configuration: {e}")
    
    def to_file(self, config_path: str):
        """Save configuration to a JSON file."""
        try:
            # Convert to dict for JSON serialization
            config_dict = self.to_dict()
            
            with open(config_path, 'w') as f:
                json.dump(config_dict, f, indent=2)
        
        except Exception as e:
            raise ConfigurationError(f"Error saving configuration: {e}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'output_dir': self.output_dir,
            'temp_dir': self.temp_dir,
            'log_dir': self.log_dir,
            'retry_config': {
                'max_retries': self.retry_config.max_retries,
                'base_delay': self.retry_config.base_delay,
                'exponential_base': self.retry_config.exponential_base,
                'max_delay': self.retry_config.max_delay,
                'jitter': self.retry_config.jitter
            },
            'network_config': {
                'timeout': self.network_config.timeout,
                'concurrent_downloads': self.network_config.concurrent_downloads,
                'chunk_size': self.network_config.chunk_size,
                'user_agent': self.network_config.user_agent
            },
            'geo_config': {
                'base_url': self.geo_config.base_url,
                'ftp_host': self.geo_config.ftp_host,
                'db': self.geo_config.db,
                'retmode': self.geo_config.retmode,
                'max_datasets': self.geo_config.max_datasets,
                'entrez_email': self.geo_config.entrez_email,
                'tissue_keywords': self.geo_config.tissue_keywords,
                'exclude_keywords': self.geo_config.exclude_keywords,
                'supplementary_config': self.geo_config.supplementary_config,
                'supplementary_keywords': self.geo_config.supplementary_keywords
            },
            'sra_config': {
                'base_url': self.sra_config.base_url,
                'db': self.sra_config.db,
                'enabled': self.sra_config.enabled
            },
            'tcga_config': {
                'base_url': self.tcga_config.base_url,
                'enabled': self.tcga_config.enabled
            },
            'gtex_config': {
                'base_url': self.gtex_config.base_url,
                'enabled': self.gtex_config.enabled
            },
            'arrayexpress_config': {
                'base_url': self.arrayexpress_config.base_url,
                'enabled': self.arrayexpress_config.enabled
            },
            'max_concurrent_agents': self.max_concurrent_agents,
            'progress_update_interval': self.progress_update_interval,
            'validate_downloads': self.validate_downloads,
            'save_metadata': self.save_metadata,
            'cleanup_temp_files': self.cleanup_temp_files,
            'log_level': self.log_level,
            'log_format': self.log_format
        }
    
    def get_enabled_agents(self) -> List[str]:
        """Get list of enabled agent names."""
        enabled_agents = [AgentName.GEO]  # GEO is always enabled
        
        if self.sra_config.enabled:
            enabled_agents.append(AgentName.SRA)
        if self.tcga_config.enabled:
            enabled_agents.append(AgentName.TCGA)
        if self.gtex_config.enabled:
            enabled_agents.append(AgentName.GTEx)
        if self.arrayexpress_config.enabled:
            enabled_agents.append(AgentName.ArrayExpress)
        
        return enabled_agents


def get_config() -> CohortRetrievalConfig:
    """Get the default configuration instance."""
    return CohortRetrievalConfig() 
