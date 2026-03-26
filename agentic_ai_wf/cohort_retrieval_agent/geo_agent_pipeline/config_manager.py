"""
Dynamic Configuration Manager for the Cohort Retrieval Agent system.

This module provides runtime configuration management with validation,
environment variable support, and customizable settings for keywords,
validation rules, and download behavior.
"""

import os
import json
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Type, Tuple
from dataclasses import dataclass, field, asdict
import logging
from copy import deepcopy

from .config import CohortRetrievalConfig, GEOConfig
from .exceptions import ConfigurationError


@dataclass
class KeywordConfig:
    """Configuration for keyword-based filtering and categorization."""
    name: str
    keywords: List[str]
    priority: int = 5
    description: str = ""
    required: bool = False
    case_sensitive: bool = False
    regex_patterns: List[str] = field(default_factory=list)
    exclude_patterns: List[str] = field(default_factory=list)


@dataclass 
class ValidationRuleConfig:
    """Configuration for validation rules."""
    name: str
    rule_type: str  # "size", "format", "content", "statistical"
    parameters: Dict[str, Any] = field(default_factory=dict)
    severity: str = "warning"  # "info", "warning", "error", "critical"
    enabled: bool = True
    description: str = ""


@dataclass
class DownloadBehaviorConfig:
    """Configuration for download behavior."""
    max_concurrent_downloads: int = 3
    retry_attempts: int = 3
    retry_delay_seconds: float = 2.0
    timeout_seconds: int = 300
    prefer_http_over_ftp: bool = True
    require_supplementary_files: bool = False
    fallback_to_series_matrix: bool = True
    max_file_size_mb: int = 5000
    min_file_size_bytes: int = 1024
    allowed_extensions: List[str] = field(default_factory=lambda: [
        ".txt", ".tsv", ".csv", ".gz", ".tar", ".zip", ".h5", ".rds"
    ])
    blocked_extensions: List[str] = field(default_factory=lambda: [
        ".pdf", ".doc", ".docx", ".ppt", ".pptx", ".jpg", ".png", ".exe"
    ])


class ConfigurationManager:
    """
    Manager for dynamic configuration with runtime customization support.
    
    Features:
    - Load/save configurations from multiple formats (JSON, YAML, Python)
    - Runtime modification of settings
    - Validation of configuration changes
    - Environment variable overrides
    - Configuration templates and presets
    - Backup and restore functionality
    """
    
    def __init__(self, base_config: Optional[CohortRetrievalConfig] = None):
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Base configuration
        self.base_config = base_config or CohortRetrievalConfig()
        
        # Dynamic configurations
        self.keyword_configs: Dict[str, KeywordConfig] = {}
        self.validation_rules: Dict[str, ValidationRuleConfig] = {}
        self.download_behavior = DownloadBehaviorConfig()
        
        # Configuration history for rollback
        self.config_history: List[Dict[str, Any]] = []
        self.max_history_size = 10
        
        # Initialize with default configurations
        self._initialize_default_configs()
    
    def _initialize_default_configs(self):
        """Initialize with default keyword and validation configurations."""
        # Initialize keyword configurations from base config
        if hasattr(self.base_config.geo_config, 'supplementary_config'):
            file_categories = self.base_config.geo_config.supplementary_config.get("file_categories", {})
            
            for category_name, category_info in file_categories.items():
                self.keyword_configs[category_name] = KeywordConfig(
                    name=category_name,
                    keywords=category_info.get("keywords", []),
                    priority=category_info.get("priority", 5),
                    description=category_info.get("description", ""),
                    required=category_info.get("required", False)
                )
        
        # Initialize validation rules
        self._initialize_default_validation_rules()
        
        # Initialize download behavior from base config
        if hasattr(self.base_config.geo_config, 'supplementary_config'):
            download_config = self.base_config.geo_config.supplementary_config.get("download_behavior", {})
            self.download_behavior = DownloadBehaviorConfig(**download_config)
    
    def _initialize_default_validation_rules(self):
        """Initialize default validation rules."""
        default_rules = [
            ValidationRuleConfig(
                name="file_size_check",
                rule_type="size",
                parameters={
                    "min_size_bytes": 1024,
                    "max_size_mb": 5000
                },
                severity="warning",
                description="Validate file size is within acceptable range"
            ),
            ValidationRuleConfig(
                name="matrix_dimensions",
                rule_type="content",
                parameters={
                    "min_rows": 1000,
                    "min_columns": 10,
                    "max_zero_fraction": 0.9
                },
                severity="warning",
                description="Validate count matrix dimensions and sparsity"
            ),
            ValidationRuleConfig(
                name="numeric_data_check",
                rule_type="content",
                parameters={
                    "allow_negative": False,
                    "check_data_types": True
                },
                severity="error",
                description="Validate numeric data integrity"
            ),
            ValidationRuleConfig(
                name="metadata_completeness",
                rule_type="content",
                parameters={
                    "required_columns": ["sample_id"],
                    "check_duplicates": True
                },
                severity="warning",
                description="Validate metadata file completeness"
            )
        ]
        
        for rule in default_rules:
            self.validation_rules[rule.name] = rule
    
    def add_keyword_config(self, config: KeywordConfig) -> bool:
        """Add or update a keyword configuration."""
        try:
            # Validate the configuration
            if not config.name or not config.keywords:
                raise ConfigurationError("Keyword config must have name and keywords")
            
            # Save current state for rollback
            self._save_state()
            
            # Add the configuration
            self.keyword_configs[config.name] = config
            
            self.logger.info(f"Added keyword config: {config.name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to add keyword config: {e}")
            return False
    
    def update_keyword_config(self, name: str, **updates) -> bool:
        """Update an existing keyword configuration."""
        try:
            if name not in self.keyword_configs:
                raise ConfigurationError(f"Keyword config '{name}' not found")
            
            # Save current state for rollback
            self._save_state()
            
            # Update the configuration
            config = self.keyword_configs[name]
            for key, value in updates.items():
                if hasattr(config, key):
                    setattr(config, key, value)
                else:
                    raise ConfigurationError(f"Invalid parameter: {key}")
            
            self.logger.info(f"Updated keyword config: {name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to update keyword config: {e}")
            return False
    
    def remove_keyword_config(self, name: str) -> bool:
        """Remove a keyword configuration."""
        try:
            if name not in self.keyword_configs:
                raise ConfigurationError(f"Keyword config '{name}' not found")
            
            # Save current state for rollback
            self._save_state()
            
            # Remove the configuration
            del self.keyword_configs[name]
            
            self.logger.info(f"Removed keyword config: {name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to remove keyword config: {e}")
            return False
    
    def add_validation_rule(self, rule: ValidationRuleConfig) -> bool:
        """Add or update a validation rule."""
        try:
            # Validate the rule
            if not rule.name or not rule.rule_type:
                raise ConfigurationError("Validation rule must have name and rule_type")
            
            valid_types = ["size", "format", "content", "statistical"]
            if rule.rule_type not in valid_types:
                raise ConfigurationError(f"Invalid rule_type. Must be one of: {valid_types}")
            
            valid_severities = ["info", "warning", "error", "critical"]
            if rule.severity not in valid_severities:
                raise ConfigurationError(f"Invalid severity. Must be one of: {valid_severities}")
            
            # Save current state for rollback
            self._save_state()
            
            # Add the rule
            self.validation_rules[rule.name] = rule
            
            self.logger.info(f"Added validation rule: {rule.name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to add validation rule: {e}")
            return False
    
    def update_validation_rule(self, name: str, **updates) -> bool:
        """Update an existing validation rule."""
        try:
            if name not in self.validation_rules:
                raise ConfigurationError(f"Validation rule '{name}' not found")
            
            # Save current state for rollback
            self._save_state()
            
            # Update the rule
            rule = self.validation_rules[name]
            for key, value in updates.items():
                if hasattr(rule, key):
                    setattr(rule, key, value)
                else:
                    raise ConfigurationError(f"Invalid parameter: {key}")
            
            self.logger.info(f"Updated validation rule: {name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to update validation rule: {e}")
            return False
    
    def update_download_behavior(self, **updates) -> bool:
        """Update download behavior configuration."""
        try:
            # Save current state for rollback
            self._save_state()
            
            # Update download behavior
            for key, value in updates.items():
                if hasattr(self.download_behavior, key):
                    setattr(self.download_behavior, key, value)
                else:
                    raise ConfigurationError(f"Invalid parameter: {key}")
            
            self.logger.info("Updated download behavior configuration")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to update download behavior: {e}")
            return False
    
    def get_keyword_config(self, name: str) -> Optional[KeywordConfig]:
        """Get a keyword configuration by name."""
        return self.keyword_configs.get(name)
    
    def get_validation_rule(self, name: str) -> Optional[ValidationRuleConfig]:
        """Get a validation rule by name."""
        return self.validation_rules.get(name)
    
    def get_all_keywords_by_category(self) -> Dict[str, List[str]]:
        """Get all keywords organized by category."""
        return {
            name: config.keywords 
            for name, config in self.keyword_configs.items()
        }
    
    def get_keywords_by_priority(self) -> List[Tuple[str, List[str], int]]:
        """Get keywords sorted by priority."""
        return sorted(
            [(name, config.keywords, config.priority) 
             for name, config in self.keyword_configs.items()],
            key=lambda x: x[2],
            reverse=True
        )
    
    def categorize_filename(self, filename: str) -> Optional[str]:
        """Categorize a filename using configured keywords."""
        filename_lower = filename.lower()
        
        # Find the highest priority category that matches
        best_category = None
        best_priority = -1
        
        for name, config in self.keyword_configs.items():
            for keyword in config.keywords:
                keyword_to_check = keyword if config.case_sensitive else keyword.lower()
                if keyword_to_check in filename_lower:
                    if config.priority > best_priority:
                        best_category = name
                        best_priority = config.priority
        
        return best_category
    
    def load_from_file(self, file_path: Union[str, Path]) -> bool:
        """Load configuration from a file (JSON or YAML)."""
        try:
            file_path = Path(file_path)
            
            if not file_path.exists():
                raise ConfigurationError(f"Configuration file not found: {file_path}")
            
            # Save current state for rollback
            self._save_state()
            
            with open(file_path, 'r') as f:
                if file_path.suffix.lower() in ['.yaml', '.yml']:
                    data = yaml.safe_load(f)
                else:
                    data = json.load(f)
            
            # Load keyword configurations
            if 'keyword_configs' in data:
                self.keyword_configs = {}
                for name, config_data in data['keyword_configs'].items():
                    self.keyword_configs[name] = KeywordConfig(**config_data)
            
            # Load validation rules
            if 'validation_rules' in data:
                self.validation_rules = {}
                for name, rule_data in data['validation_rules'].items():
                    self.validation_rules[name] = ValidationRuleConfig(**rule_data)
            
            # Load download behavior
            if 'download_behavior' in data:
                self.download_behavior = DownloadBehaviorConfig(**data['download_behavior'])
            
            self.logger.info(f"Loaded configuration from: {file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load configuration: {e}")
            # Rollback on error
            self.rollback()
            return False
    
    def save_to_file(self, file_path: Union[str, Path], format: str = "json") -> bool:
        """Save current configuration to a file."""
        try:
            file_path = Path(file_path)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Prepare data for serialization
            data = {
                "keyword_configs": {
                    name: asdict(config) 
                    for name, config in self.keyword_configs.items()
                },
                "validation_rules": {
                    name: asdict(rule) 
                    for name, rule in self.validation_rules.items()
                },
                "download_behavior": asdict(self.download_behavior)
            }
            
            with open(file_path, 'w') as f:
                if format.lower() == "yaml":
                    yaml.dump(data, f, default_flow_style=False, indent=2)
                else:
                    json.dump(data, f, indent=2)
            
            self.logger.info(f"Saved configuration to: {file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save configuration: {e}")
            return False
    
    def apply_environment_overrides(self):
        """Apply configuration overrides from environment variables."""
        try:
            # Download behavior overrides
            env_mappings = {
                "COHORT_MAX_CONCURRENT_DOWNLOADS": ("max_concurrent_downloads", int),
                "COHORT_RETRY_ATTEMPTS": ("retry_attempts", int),
                "COHORT_TIMEOUT_SECONDS": ("timeout_seconds", int),
                "COHORT_REQUIRE_SUPPLEMENTARY": ("require_supplementary_files", bool),
                "COHORT_MAX_FILE_SIZE_MB": ("max_file_size_mb", int)
            }
            
            for env_var, (attr_name, type_func) in env_mappings.items():
                env_value = os.getenv(env_var)
                if env_value is not None:
                    try:
                        converted_value = type_func(env_value.lower() in ['true', '1', 'yes'] if type_func == bool else env_value)
                        setattr(self.download_behavior, attr_name, converted_value)
                        self.logger.info(f"Applied environment override: {env_var}={converted_value}")
                    except ValueError:
                        self.logger.warning(f"Invalid environment value for {env_var}: {env_value}")
            
        except Exception as e:
            self.logger.error(f"Failed to apply environment overrides: {e}")
    
    def create_preset(self, name: str, description: str = "") -> bool:
        """Create a configuration preset."""
        try:
            preset_data = {
                "name": name,
                "description": description,
                "keyword_configs": {k: asdict(v) for k, v in self.keyword_configs.items()},
                "validation_rules": {k: asdict(v) for k, v in self.validation_rules.items()},
                "download_behavior": asdict(self.download_behavior)
            }
            
            # Save preset to file
            preset_dir = Path(self.base_config.output_dir) / "config_presets"
            preset_dir.mkdir(parents=True, exist_ok=True)
            preset_file = preset_dir / f"{name}.json"
            
            with open(preset_file, 'w') as f:
                json.dump(preset_data, f, indent=2)
            
            self.logger.info(f"Created configuration preset: {name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to create preset: {e}")
            return False
    
    def load_preset(self, name: str) -> bool:
        """Load a configuration preset."""
        try:
            preset_dir = Path(self.base_config.output_dir) / "config_presets"
            preset_file = preset_dir / f"{name}.json"
            
            if not preset_file.exists():
                raise ConfigurationError(f"Preset not found: {name}")
            
            return self.load_from_file(preset_file)
            
        except Exception as e:
            self.logger.error(f"Failed to load preset: {e}")
            return False
    
    def _save_state(self):
        """Save current configuration state for rollback."""
        try:
            state = {
                "keyword_configs": deepcopy(self.keyword_configs),
                "validation_rules": deepcopy(self.validation_rules),
                "download_behavior": deepcopy(self.download_behavior)
            }
            
            self.config_history.append(state)
            
            # Limit history size
            if len(self.config_history) > self.max_history_size:
                self.config_history.pop(0)
                
        except Exception as e:
            self.logger.warning(f"Failed to save configuration state: {e}")
    
    def rollback(self) -> bool:
        """Rollback to the previous configuration state."""
        try:
            if not self.config_history:
                self.logger.warning("No configuration history available for rollback")
                return False
            
            # Restore previous state
            previous_state = self.config_history.pop()
            self.keyword_configs = previous_state["keyword_configs"]
            self.validation_rules = previous_state["validation_rules"]
            self.download_behavior = previous_state["download_behavior"]
            
            self.logger.info("Configuration rolled back to previous state")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to rollback configuration: {e}")
            return False
    
    def validate_configuration(self) -> List[str]:
        """Validate the current configuration and return any issues."""
        issues = []
        
        # Validate keyword configurations
        for name, config in self.keyword_configs.items():
            if not config.keywords:
                issues.append(f"Keyword config '{name}' has no keywords")
            
            if config.priority < 0 or config.priority > 10:
                issues.append(f"Keyword config '{name}' has invalid priority: {config.priority}")
        
        # Validate validation rules
        for name, rule in self.validation_rules.items():
            if not rule.parameters and rule.rule_type in ["size", "content", "statistical"]:
                issues.append(f"Validation rule '{name}' missing required parameters")
        
        # Validate download behavior
        if self.download_behavior.max_concurrent_downloads <= 0:
            issues.append("max_concurrent_downloads must be positive")
        
        if self.download_behavior.retry_attempts < 0:
            issues.append("retry_attempts must be non-negative")
        
        return issues
    
    def get_configuration_summary(self) -> Dict[str, Any]:
        """Get a summary of the current configuration."""
        return {
            "keyword_configs": {
                "count": len(self.keyword_configs),
                "categories": list(self.keyword_configs.keys()),
                "total_keywords": sum(len(config.keywords) for config in self.keyword_configs.values())
            },
            "validation_rules": {
                "count": len(self.validation_rules),
                "enabled_rules": [name for name, rule in self.validation_rules.items() if rule.enabled],
                "rules_by_severity": {
                    severity: [name for name, rule in self.validation_rules.items() if rule.severity == severity]
                    for severity in ["info", "warning", "error", "critical"]
                }
            },
            "download_behavior": {
                "max_concurrent": self.download_behavior.max_concurrent_downloads,
                "retry_attempts": self.download_behavior.retry_attempts,
                "require_supplementary": self.download_behavior.require_supplementary_files,
                "timeout_seconds": self.download_behavior.timeout_seconds
            },
            "validation_issues": self.validate_configuration()
        } 