# Drug Prioritization Tool

## Overview

The Drug Prioritization Tool is a production-ready Python module that prioritizes drugs based on multiple weighted criteria including DEG (Differentially Expressed Gene) match status, condition matching status, and regulatory approval status.

## 🚀 Key Improvements Made

### Fixed Issues
- **Fixed main issue**: "local variable main_df is not accessible" error
- **Corrected file path structure**: Now saves to `outputdir/analysis_id/` as requested
- **Improved filename generation**: Better naming convention instead of `pathway_id_.csv`
- **Enhanced error handling**: Comprehensive exception handling and validation

### Production-Ready Features
- ✅ **Comprehensive error handling** with custom exceptions
- ✅ **Input validation** for all parameters
- ✅ **Proper logging** with informative messages
- ✅ **Type hints** for better code maintenance
- ✅ **Modular design** with separate functions for each concern
- ✅ **Flexible configuration** with optional parameters
- ✅ **Robust file handling** with proper directory creation
- ✅ **Statistics and reporting** capabilities
- ✅ **Graceful fallbacks** for missing configuration

## 📊 Priority Scoring System

The tool uses a weighted scoring system (0-6) where higher scores indicate higher priority:

| Score | DEG Match | Condition Match | Approved | Description |
|-------|-----------|----------------|----------|-------------|
| **6** | ✅ | ✅ | ✅ | **Most favorable** - All criteria met |
| **5** | ✅ | ✅ | ❌ | DEG + Condition match, not approved |
| **4** | ✅ | ❌ | ✅ | DEG match + approved, no condition match |
| **3** | ✅ | ❌ | ❌ | DEG match only |
| **2** | ❌ | ✅ | ✅ | Condition match + approved, no DEG match |
| **1** | ❌ | ✅ | ❌ | Condition match only |
| **0** | ❌ | ❌ | ❌/✅ | **Least favorable** - No matches |

## 🔧 Usage

### Basic Usage

```python
from agentic_ai_wf.processing.prioritization_tool import prioritize_drugs
from agentic_ai_wf.models.drugs_models import EnrichedDrugDegInput

# Assume you have your drugs data
drugs_input = EnrichedDrugDegInput(drugs=your_drugs_list)

# Basic prioritization (uses default settings)
result = prioritize_drugs(drugs_input)

# Access prioritized drugs
for drug in result.drugs:
    print(f"{drug.name}: Priority {drug.priority_status}")
```

### Advanced Usage with Custom Settings

```python
# Custom output directory and analysis ID
result = prioritize_drugs(
    drugs_input,
    output_dir="./custom_analysis_output",
    analysis_id="analysis_001"
)

# Files will be saved to: ./custom_analysis_output/analysis_001/
```

### Getting Statistics

```python
from agentic_ai_wf.processing.prioritization_tool import get_prioritization_statistics

# Get detailed statistics
stats = get_prioritization_statistics(result)
print(f"Total drugs: {stats['total_drugs']}")
print(f"High priority drugs: {stats['priority_levels']['high_priority']}")
```

## 📁 Output Structure

The tool saves prioritized drugs to:
```
{output_dir}/
└── {analysis_id}/
    └── prioritized_drugs_{pathway_id}_{timestamp}.csv
```

### Default Configuration
- **Output Directory**: `./agentic_ai_wf/processed_pathways/`
- **Analysis ID**: From `global_config.patient_dir` or auto-generated
- **Filename**: `prioritized_drugs_{pathway_id}_{timestamp}.csv`

## 🛠️ Functions

### Core Functions

#### `prioritize_drugs(drugs, output_dir=None, analysis_id=None)`
Main function that prioritizes drugs and saves results.

**Parameters:**
- `drugs`: EnrichedDrugDegInput object containing drugs to prioritize
- `output_dir`: Optional custom output directory
- `analysis_id`: Optional custom analysis identifier

**Returns:** `EnrichedDrugPriorityOutput` containing prioritized drugs

#### `get_prioritization_statistics(drugs)`
Generate detailed statistics about prioritization results.

**Parameters:**
- `drugs`: EnrichedDrugPriorityOutput object

**Returns:** Dictionary with detailed statistics

### Helper Functions

- `validate_prioritization_input()`: Validates input data
- `calculate_priority_score()`: Calculates priority score for individual drugs
- `create_output_directory()`: Creates output directory structure
- `generate_filename()`: Generates appropriate filenames
- `save_prioritized_drugs()`: Saves results to CSV
- `get_analysis_id()`: Gets analysis ID with fallback options

## 🔍 Error Handling

The tool includes comprehensive error handling:

```python
from agentic_ai_wf.processing.prioritization_tool import PrioritizationError

try:
    result = prioritize_drugs(drugs_input)
except PrioritizationError as e:
    print(f"Prioritization failed: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

## 📋 Example Output

The saved CSV file includes all drug information plus priority scores:

```csv
name,priority_status,approved,deg_match_status,matching_status,drug_id,pathway_id,...
Aspirin,6,True,True,True,D001,hsa04657,...
Experimental Drug X,5,False,True,True,D002,hsa04657,...
Metformin,0,True,False,False,D003,hsa04910,...
```

## 🧪 Testing

Run the example script to test the functionality:

```bash
python agentic_ai_wf/processing/prioritization_example.py
```

## 📊 Logging

The tool provides comprehensive logging:

```
2024-01-15 10:30:00 - prioritization_tool - INFO - Starting drug prioritization process
2024-01-15 10:30:00 - prioritization_tool - INFO - Validation passed: 10 drugs ready for prioritization
2024-01-15 10:30:01 - prioritization_tool - INFO - Created output directory: ./processed_pathways/analysis_001
2024-01-15 10:30:01 - prioritization_tool - INFO - Successfully saved 10 prioritized drugs to ./processed_pathways/analysis_001/prioritized_drugs_hsa04657_20240115_103001.csv
```

## 🔄 Migration from Old Version

The new version maintains backward compatibility but offers improved features:

**Old Usage:**
```python
# Old version had limited error handling and fixed paths
result = prioritize_drugs(drugs_input)
```

**New Usage:**
```python
# New version with flexible configuration and better error handling
result = prioritize_drugs(
    drugs_input,
    output_dir="./custom_output",  # Now configurable
    analysis_id="my_analysis"      # Now configurable
)
```

## 🚨 Important Notes

1. **Directory Structure**: Files are now saved to `outputdir/analysis_id/` as requested
2. **Filename Format**: Improved from `pathway_id_.csv` to `prioritized_drugs_{pathway_id}_{timestamp}.csv`
3. **Error Handling**: Comprehensive validation prevents runtime errors
4. **Flexible Configuration**: Output paths and analysis IDs are now configurable
5. **Statistics**: Additional reporting capabilities for better insights

## 🤝 Contributing

When modifying this tool:
1. Maintain the existing priority scoring logic
2. Add appropriate error handling for new features
3. Update logging messages for clarity
4. Include type hints for new functions
5. Add tests for new functionality

## 📈 Performance

The tool is optimized for:
- Large drug datasets (1000+ drugs)
- Multiple pathway analysis
- Concurrent processing support
- Memory-efficient DataFrame operations 