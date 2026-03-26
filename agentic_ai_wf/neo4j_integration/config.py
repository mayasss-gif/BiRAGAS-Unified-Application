"""
Neo4j Integration Configuration

Configuration settings for Neo4j database connection and data loading.
"""

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class Neo4jConfig:
    """Configuration class for Neo4j connection and operations."""

    # Connection settings
    uri: str = "bolt://localhost:7687"
    username: str = "neo4j"
    password: str = "password"
    database: str = "neo4j"

    # Performance settings
    batch_size: int = 1000
    timeout_seconds: int = 30
    max_retry_attempts: int = 3

    # Data loading settings
    create_indexes: bool = True
    clear_on_start: bool = False

    # Logging settings
    log_level: str = "INFO"

    @classmethod
    def from_environment(cls) -> 'Neo4jConfig':
        """Create configuration from environment variables."""
        return cls(
            uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            username=os.getenv("NEO4J_USERNAME", "neo4j"),
            password=os.getenv("NEO4J_PASSWORD", "password"),
            database=os.getenv("NEO4J_DATABASE", "neo4j"),
            batch_size=int(os.getenv("NEO4J_BATCH_SIZE", "1000")),
            timeout_seconds=int(os.getenv("NEO4J_TIMEOUT", "30")),
            max_retry_attempts=int(os.getenv("NEO4J_MAX_RETRIES", "3")),
            create_indexes=os.getenv(
                "NEO4J_CREATE_INDEXES", "true").lower() == "true",
            clear_on_start=os.getenv(
                "NEO4J_CLEAR_ON_START", "false").lower() == "true",
            log_level=os.getenv("NEO4J_LOG_LEVEL", "INFO")
        )

    def validate(self) -> bool:
        """Validate configuration settings."""
        if not self.uri:
            raise ValueError("URI cannot be empty")
        if not self.username:
            raise ValueError("Username cannot be empty")
        if not self.password:
            raise ValueError("Password cannot be empty")
        if self.batch_size <= 0:
            raise ValueError("Batch size must be positive")
        if self.timeout_seconds <= 0:
            raise ValueError("Timeout must be positive")
        if self.max_retry_attempts < 0:
            raise ValueError("Max retry attempts cannot be negative")

        return True


# Default configuration instance
DEFAULT_CONFIG = Neo4jConfig()

# Environment-based configuration instance
ENV_CONFIG = Neo4jConfig.from_environment()


# Query configuration
@dataclass
class QueryConfig:
    """Configuration for query execution."""

    default_limit: int = 100
    max_limit: int = 10000
    enable_explain: bool = False
    enable_profile: bool = False

    # Timeout for specific query types (in seconds)
    simple_query_timeout: int = 10
    complex_query_timeout: int = 60
    bulk_operation_timeout: int = 300


# Default query configuration
DEFAULT_QUERY_CONFIG = QueryConfig()


# Data loading configuration
@dataclass
class DataLoadingConfig:
    """Configuration for data loading operations."""

    # CSV loading settings
    csv_chunk_size: int = 10000
    skip_empty_values: bool = True
    normalize_strings: bool = True

    # Relationship creation settings
    create_missing_nodes: bool = True
    skip_duplicate_relationships: bool = True

    # Error handling
    continue_on_error: bool = True
    max_errors_per_batch: int = 100

    # Performance settings
    use_transactions: bool = True
    transaction_size: int = 1000


# Default data loading configuration
DEFAULT_DATA_LOADING_CONFIG = DataLoadingConfig()


# Schema configuration
BIOMEDICAL_SCHEMA = {
    "node_labels": [
        "Gene",
        "Pathway",
        "Drug",
        "Disease",
        "Protein",
        "Compound",
        "ClinicalTrial"
    ],
    "relationship_types": [
        "BELONGS_TO",
        "TARGETS",
        "TREATS",
        "ASSOCIATED_WITH",
        "INTERACTS_WITH",
        "REGULATES",
        "ENRICHED_IN",
        "PARTICIPATES_IN",
        "ENCODES",
        "BINDS_TO",
        "METABOLIZES"
    ],
    "required_properties": {
        "Gene": ["symbol"],
        "Pathway": ["pathway_id", "name", "source"],
        "Drug": ["name"],
        "Disease": ["name"]
    },
    "indexed_properties": {
        "Gene": ["symbol", "ensembl_id"],
        "Pathway": ["pathway_id"],
        "Drug": ["name", "drugbank_id"],
        "Disease": ["name", "disease_id"]
    }
}


def get_connection_string(config: Optional[Neo4jConfig] = None) -> str:
    """Get Neo4j connection string from configuration."""
    if config is None:
        config = ENV_CONFIG

    return f"{config.uri} (user: {config.username}, db: {config.database})"


def create_sample_env_file(file_path: str = ".env.neo4j") -> None:
    """Create a sample environment file for Neo4j configuration."""

    env_content = """# Neo4j Configuration
# Copy this file to .env and update with your actual values

# Connection settings
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password_here
NEO4J_DATABASE=neo4j

# Performance settings
NEO4J_BATCH_SIZE=1000
NEO4J_TIMEOUT=30
NEO4J_MAX_RETRIES=3

# Operational settings
NEO4J_CREATE_INDEXES=true
NEO4J_CLEAR_ON_START=false
NEO4J_LOG_LEVEL=INFO

# Example for remote Neo4j instance:
# NEO4J_URI=bolt://your-neo4j-server.com:7687
# NEO4J_USERNAME=your_username
# NEO4J_PASSWORD=your_secure_password
# NEO4J_DATABASE=biomedical_data

# Example for Neo4j Aura (cloud):
# NEO4J_URI=neo4j+s://your-instance.databases.neo4j.io
# NEO4J_USERNAME=neo4j
# NEO4J_PASSWORD=your_aura_password
"""

    with open(file_path, 'w') as f:
        f.write(env_content)

    print(f"Sample environment file created: {file_path}")
    print("Please update it with your actual Neo4j credentials.")


if __name__ == "__main__":
    # Create sample environment file if run directly
    create_sample_env_file()

    # Print current configuration
    print("Current Neo4j Configuration:")
    print(f"  URI: {ENV_CONFIG.uri}")
    print(f"  Username: {ENV_CONFIG.username}")
    print(f"  Database: {ENV_CONFIG.database}")
    print(f"  Batch Size: {ENV_CONFIG.batch_size}")
    print(f"  Create Indexes: {ENV_CONFIG.create_indexes}")

    # Validate configuration
    try:
        ENV_CONFIG.validate()
        print("✅ Configuration is valid")
    except ValueError as e:
        print(f"❌ Configuration error: {e}")
