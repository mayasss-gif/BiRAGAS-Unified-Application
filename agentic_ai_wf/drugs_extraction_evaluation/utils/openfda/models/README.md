# FDA Models Package

This package contains structured data models for FDA drug information, organized into focused modules to maintain clean, readable code under 100 lines per file.

## Package Structure

```
models/
├── __init__.py          # Package exports (30 lines)
├── enums.py             # FDA approval status and route enums (30 lines)
├── drug_label.py        # Drug labeling information model (59 lines)
├── drug_names.py        # Drug names and route model (43 lines)
├── fda_approval.py      # Main FDA approval info model (89 lines)
├── factories.py         # Factory functions for creating instances (60 lines)
├── query_result.py      # Query result container model (43 lines)
└── README.md           # This documentation
```

## File Breakdown

### `enums.py` (25 lines)
- `FDAApprovalStatus` - FDA approval statuses
- `RouteOfAdministration` - Common drug administration routes

### `drug_label.py` (65 lines)
- `DrugLabelInfo` - FDA drug labeling information
- Methods: `to_dict()`, `to_json()`, `from_dict()`, `from_fda_response()`

### `drug_names.py` (45 lines)
- `DrugNamesAndRoute` - Drug names and administration route
- Methods: `to_dict()`, `from_dict()`, `from_fda_response()`

### `fda_approval.py` (89 lines)
- `FDAApprovalInfo` - Main comprehensive FDA drug information
- Utility methods: `is_approved()`, `has_label_info()`, `get_summary()`

### `factories.py` (60 lines)
- Factory functions: `create_approved_drug()`, `create_not_found_drug()`, `create_error_drug()`
- Serialization: `create_from_dict()`

### `query_result.py` (45 lines)
- `FDAQueryResult` - Container for FDA query results with metadata
- Methods: `add_result()`, `get_approved_drugs()`, `to_dict()`, `to_json()`

## Usage

```python
from agentic_ai_wf.drugs_extraction_evaluation.utils.openfda.models import (
    FDAApprovalInfo,
    DrugLabelInfo,
    FDAApprovalStatus
)

# Create an approved drug
approved_drug = create_approved_drug(
    drug_name="Lisinopril",
    brand_name="Zestril",
    generic_name="Lisinopril",
    route="Oral",
    indications="Hypertension treatment"
)

# Check status
if approved_drug.is_approved():
    print("Drug is FDA approved")

# Get summary
summary = approved_drug.get_summary()
```

## Benefits of This Structure

1. **Maintainability**: Each file has a single responsibility
2. **Readability**: All files are under 100 lines
3. **Modularity**: Easy to import only what you need
4. **Testability**: Each module can be tested independently
5. **Extensibility**: Easy to add new models or modify existing ones
6. **Organization**: Clear separation of concerns

## Testing

```bash
# Test all models
python -m agentic_ai_wf.drugs_extraction_evaluation.utils.openfda.test_models

# Test FDA functionality
python run_tests.py
``` 