# DEG Pipeline Agent

A robust, self-healing agent system for Differential Expression Gene (DEG) analysis using the OpenAI Agents SDK pattern.

## 🎯 Overview

The DEG Pipeline Agent replaces the original `run_deg_pipeline.py` script with a sophisticated agent-based system that provides:

- **Self-healing capabilities**: Automatically detects and fixes common errors
- **Robust error handling**: Comprehensive retry mechanisms and fallback strategies  
- **Modular design**: Specialized tools for each pipeline step
- **Production-ready**: Extensive validation, logging, and monitoring
- **Same output format**: Maintains compatibility with existing downstream processes

## 🏗️ Architecture

### Agent Components

```
DEGPipelineAgent
├── DataLoaderTool           # Load and sanitize count data
├── DatasetDetectorTool      # Find count/metadata pairs
├── MetadataExtractorTool    # Extract metadata from GEO/patient files
├── DESeq2AnalyzerTool       # Run differential expression analysis
├── GeneMapperTool           # Map gene IDs to symbols
├── FileValidatorTool        # Validate processed files
└── ErrorFixerTool           # Automatic error detection and fixing
```

### Key Features

- ✅ **Automatic Error Recovery**: Handles file format issues, missing data, and analysis failures
- ✅ **Flexible Input**: Supports GEO series matrix files and patient metadata
- ✅ **Robust Data Loading**: Multiple fallback strategies for different file formats
- ✅ **Comprehensive Validation**: Validates data integrity at every step
- ✅ **Detailed Logging**: Extensive logging with configurable levels
- ✅ **Progress Monitoring**: Real-time status updates and execution statistics
- ✅ **Configuration Management**: Centralized configuration with validation

## 🚀 Quick Start

### Basic Usage

```bash
# Run with both GEO and patient data
python run_deg_agent.py /path/to/geo /path/to/patients cancer_type

# Run with only patient data
python run_deg_agent.py --patient-dir /path/to/patients cancer_type

# Run with custom parameters
python run_deg_agent.py /path/to/geo /path/to/patients cancer_type \
  --max-genes 3000 --padj-threshold 0.01 --verbose
```

### Programmatic Usage

```python
from agentic_ai_wf.deg_pipeline_agent import DEGPipelineAgent, DEGPipelineConfig

# Create configuration
config = DEGPipelineConfig(
    geo_dir="/path/to/geo",
    patient_dir="/path/to/patients", 
    disease_name="cancer_type",
    max_retries=3,
    enable_auto_fix=True
)

# Initialize agent
agent = DEGPipelineAgent(config)

# Run pipeline
result = agent.run_pipeline()

# Check results
print(f"Success rate: {result['summary']['success_rate']:.1%}")
```

## 📋 Configuration Options

### Analysis Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `padj_threshold` | 0.05 | Adjusted p-value threshold for significance |
| `max_genes` | 5000 | Maximum number of genes to analyze |
| `min_samples_per_gene` | 2 | Minimum samples per gene for filtering |
| `log2fc_threshold` | 0.0 | Log2 fold change threshold |

### Agent Behavior

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_retries` | 3 | Maximum retry attempts for failed operations |
| `retry_delay` | 1.0 | Base delay between retries (seconds) |
| `enable_auto_fix` | True | Enable automatic error fixing |
| `enable_monitoring` | True | Enable execution monitoring |

### File Processing

| Parameter | Default | Description |
|-----------|---------|-------------|
| `supported_formats` | ['.csv', '.tsv', '.txt'] | Supported file formats |
| `compression_formats` | ['.gz', '.bz2'] | Supported compression formats |

## 🔧 Tools Documentation

### DataLoaderTool

Loads and sanitizes count data with robust error handling:
- Automatic format detection (CSV/TSV)
- Data sanitization (handle inf/NaN values)
- Gene filtering and subsampling
- Multiple fallback loading strategies

### MetadataExtractorTool

Extracts metadata from various sources:
- GEO series matrix files (with GEOparse + manual fallback)
- Patient metadata files
- Automatic condition column detection
- Sample alignment validation

### DESeq2AnalyzerTool

Runs differential expression analysis:
- PyDESeq2 integration
- Multiple comparison handling
- Reference condition auto-selection
- Gene ID to symbol mapping

### ErrorFixerTool

Provides self-healing capabilities:
- Permission error fixes
- Disk space cleanup
- File format issue detection
- Configuration validation
- Automatic recovery suggestions

## 📊 Output Structure

The agent maintains the same output structure as the original pipeline:

```
results/
└── {disease_name}/
    └── {sample_name}/
        ├── prep/
        │   ├── prep_counts.csv
        │   ├── prep_meta.csv
        │   └── {sample}_meta_log.json
        ├── {sample_name}_DEGs.csv
        ├── data_summary.log
        ├── decision.log
        └── deseq2_summary.log
```

## 🔍 Error Handling

### Automatic Fixes

The agent can automatically fix:
- File permission issues
- Directory creation problems
- File format inconsistencies
- Sample ID mismatches
- Empty or corrupted files

### Recovery Strategies

- **Retry with backoff**: Exponential backoff for transient errors
- **Fallback methods**: Alternative approaches for data loading
- **Graceful degradation**: Continue processing other datasets if one fails
- **Error context**: Detailed error information for debugging

## 📈 Monitoring

### Execution Statistics

```python
# Get pipeline status
status = agent.get_pipeline_status()
print(f"Current step: {status['current_step']}")
print(f"Success rate: {status['successful_datasets']}/{status['total_datasets']}")

# Get tool statistics  
for tool_name, stats in status['tool_stats'].items():
    print(f"{tool_name}: {stats['success_rate']:.1%} success rate")
```

### Logging

- **Console logging**: Real-time progress updates
- **File logging**: Detailed execution logs (optional)
- **Structured logs**: JSON logs for automated analysis
- **Configurable levels**: DEBUG, INFO, WARNING, ERROR

## 🧪 Testing

### Validation Mode

```bash
# Validate configuration without running
python run_deg_agent.py --validate-only --show-config
```

### Development Options

```bash
# Show detailed configuration
python run_deg_agent.py --show-config

# Enable verbose logging
python run_deg_agent.py --verbose

# Disable auto-fixing for debugging
python run_deg_agent.py --disable-auto-fix
```

## 🔄 Migration from Original Pipeline

### Backward Compatibility

The agent script (`run_deg_agent.py`) accepts the same arguments as the original:

```bash
# Old way
python run_deg_pipeline.py /path/to/geo /path/to/patients disease_name

# New way (same result)
python run_deg_agent.py /path/to/geo /path/to/patients disease_name
```

### Improvements Over Original

1. **Error Recovery**: Automatically handles and fixes common failures
2. **Better Validation**: Comprehensive input/output validation
3. **Modular Design**: Easier to test and maintain individual components
4. **Monitoring**: Real-time progress tracking and statistics
5. **Configuration**: Centralized, validated configuration management
6. **Logging**: Detailed, structured logging for debugging

## 📚 Dependencies

### Required

- pandas
- numpy
- pathlib
- pydeseq2 (for DESeq2 analysis)
- mygene (for gene ID mapping)
- GEOparse (for GEO metadata extraction)

### Optional

- logging (enhanced logging features)
- json (configuration and log serialization)

## 🤝 Contributing

### Adding New Tools

1. Create a new tool class inheriting from `BaseTool`
2. Implement required methods: `name`, `description`, `execute`
3. Add validation methods: `validate_input`, `validate_output`
4. Register the tool in the agent's `_initialize_tools` method

### Error Handling Patterns

```python
from ..exceptions import RecoverableError, NonRecoverableError

# For errors that can be automatically fixed
raise RecoverableError("File format issue", fix_suggestion="try_different_separator")

# For errors requiring manual intervention
raise NonRecoverableError("Critical system error", error_code="SYSTEM_001")
```

## 📄 License

This project is part of the agentic_ai_wf package and follows the same licensing terms.

---

## 🔗 Related Documentation

- [Original Pipeline Documentation](../README.md)
- [Configuration Reference](config.py)
- [Tool Development Guide](tools/README.md)
- [Error Handling Guide](exceptions.py) 