# FDA Data Models

This document describes the structured data models for FDA drug information using Python dataclasses.

## Overview

The FDA data models provide a type-safe, validated, and serializable way to handle FDA drug information. They replace the previous dictionary-based approach with proper data structures that include validation, serialization, and utility methods.

## Models

### 1. FDAApprovalStatus (Enum)

Represents the FDA approval status of a drug.

```python
from openfda import FDAApprovalStatus

# Available statuses:
FDAApprovalStatus.APPROVED           # "Approved"
FDAApprovalStatus.NOT_FOUND          # "Not Found in FDA Database"
FDAApprovalStatus.ERROR              # "Error in Verification"
FDAApprovalStatus.PENDING            # "Pending"
FDAApprovalStatus.WITHDRAWN          # "Withdrawn"
```

### 2. RouteOfAdministration (Enum)

Common routes of drug administration.

```python
from openfda import RouteOfAdministration

# Available routes:
RouteOfAdministration.ORAL           # "Oral"
RouteOfAdministration.INTRAVENOUS    # "Intravenous"
RouteOfAdministration.INTRAMUSCULAR  # "Intramuscular"
RouteOfAdministration.SUBCUTANEOUS   # "Subcutaneous"
RouteOfAdministration.TOPICAL        # "Topical"
RouteOfAdministration.INHALATION     # "Inhalation"
# ... and more
```

### 3. DrugLabelInfo (Dataclass)

Structured representation of FDA drug labeling information.

```python
from openfda import DrugLabelInfo

label_info = DrugLabelInfo(
    indications_and_usage="For the treatment of hypertension",
    dosage_and_administration="Take 10mg once daily",
    mechanism_of_action="ACE inhibitor that blocks angiotensin II formation",
    warnings_and_precautions="May cause dizziness",
    contraindications="Pregnancy, severe renal impairment",
    boxed_warning="Not available",
    patient_counseling_info="Take with or without food"
)

# Convert to dictionary
label_dict = label_info.to_dict()

# Convert to JSON
label_json = label_info.to_json()

# Create from dictionary
reconstructed = DrugLabelInfo.from_dict(label_dict)
```

### 4. DrugNamesAndRoute (Dataclass)

Structured representation of drug names and administration route.

```python
from openfda import DrugNamesAndRoute

names_route = DrugNamesAndRoute(
    brand_name="Zestril",
    generic_name="Lisinopril",
    route_of_administration="Oral"
)

# Convert to dictionary
route_dict = names_route.to_dict()

# Create from dictionary
reconstructed = DrugNamesAndRoute.from_dict(route_dict)
```

### 5. FDAApprovalInfo (Dataclass)

The main data model that combines all FDA-related information about a drug.

```python
from openfda import FDAApprovalInfo, DrugLabelInfo

# Create using factory methods
approved_drug = FDAApprovalInfo.create_approved(
    drug_name="Lisinopril",
    brand_name="Zestril",
    generic_name="Lisinopril",
    route="Oral",
    indications="Treatment of hypertension",
    reactions="Dizziness, cough, headache",
    label_info=DrugLabelInfo(...)
)

# Check status
if approved_drug.is_approved():
    print("Drug is FDA approved")

# Get summary
summary = approved_drug.get_summary()
print(summary)
# Output: {
#     'drug_name': 'Lisinopril',
#     'status': 'Approved',
#     'brand_name': 'Zestril',
#     'generic_name': 'Lisinopril',
#     'route': 'Oral',
#     'has_detailed_info': 'True'
# }

# Serialize
drug_dict = approved_drug.to_dict()
drug_json = approved_drug.to_json()

# Deserialize
reconstructed = FDAApprovalInfo.from_dict(drug_dict)
```

### 6. FDAQueryResult (Dataclass)

Container for FDA query results with metadata.

```python
from openfda import FDAQueryResult, FDAApprovalInfo

# Create query result
query_result = FDAQueryResult(
    query_drug_name="Aspirin",
    query_duration_ms=1250.5
)

# Add results
drug1 = FDAApprovalInfo.create_approved(...)
drug2 = FDAApprovalInfo.create_approved(...)

query_result.add_result(drug1)
query_result.add_result(drug2)

# Get approved drugs only
approved_drugs = query_result.get_approved_drugs()

# Serialize
result_dict = query_result.to_dict()
result_json = query_result.to_json()
```

## Factory Methods

The `FDAApprovalInfo` class provides convenient factory methods:

### create_approved()
```python
FDAApprovalInfo.create_approved(
    drug_name="Drug Name",
    brand_name="Brand Name",
    generic_name="Generic Name",
    route="Oral",
    indications="Drug indications",
    reactions="Adverse reactions",  # optional
    label_info=DrugLabelInfo(...)  # optional
)
```

### create_not_found()
```python
FDAApprovalInfo.create_not_found("Unknown Drug")
```

### create_error()
```python
FDAApprovalInfo.create_error("Drug Name", "Error message")
```

## Usage Examples

### Basic FDA Query
```python
from openfda import verify_fda_approval

# Query FDA for a drug
fda_info = verify_fda_approval("Aspirin")

# Check if approved
if fda_info.is_approved():
    print(f"Drug: {fda_info.drug_name}")
    print(f"Brand: {fda_info.brand_name}")
    print(f"Generic: {fda_info.generic_name}")
    print(f"Route: {fda_info.route_of_administration}")
    print(f"Indications: {fda_info.fda_indications}")
else:
    print(f"Status: {fda_info.fda_approved_status.value}")
```

### Working with Label Information
```python
# Check if detailed label info is available
if fda_info.has_label_info():
    label = fda_info.label_info
    print(f"Mechanism: {label.mechanism_of_action}")
    print(f"Dosage: {label.dosage_and_administration}")
    print(f"Warnings: {label.warnings_and_precautions}")
```

### Serialization for Storage/API
```python
# Convert to JSON for API response
json_response = fda_info.to_json()

# Convert to dictionary for database storage
db_record = fda_info.to_dict()

# Reconstruct from stored data
restored_info = FDAApprovalInfo.from_dict(db_record)
```

### Batch Processing
```python
from openfda import FDAQueryResult

# Process multiple drugs
drug_names = ["Aspirin", "Ibuprofen", "Acetaminophen"]
query_result = FDAQueryResult(query_drug_name="Multiple Drugs")

for drug_name in drug_names:
    fda_info = verify_fda_approval(drug_name)
    query_result.add_result(fda_info)

# Get only approved drugs
approved_drugs = query_result.get_approved_drugs()
print(f"Found {len(approved_drugs)} approved drugs")
```

## Migration from Dictionary Format

If you have existing code using the dictionary format, you can use the legacy function:

```python
from openfda import verify_fda_approval_dict

# Returns dictionary format (legacy)
fda_dict = verify_fda_approval_dict("Aspirin")

# Or convert new format to dictionary
fda_info = verify_fda_approval("Aspirin")
fda_dict = fda_info.to_dict()
```

## Benefits

1. **Type Safety**: All fields are properly typed with type hints
2. **Validation**: Automatic data validation and cleaning
3. **Serialization**: Easy JSON/dict conversion for storage and APIs
4. **Factory Methods**: Convenient creation patterns for different scenarios
5. **Extensibility**: Easy to add new fields and methods
6. **Documentation**: Self-documenting with comprehensive docstrings
7. **IDE Support**: Better autocomplete and error detection
8. **Consistency**: Standardized data structure across the application

## Testing

Run the test suite to validate the models:

```bash
python -m agentic_ai_wf.drugs_extraction_evaluation.utils.openfda.test_models
```

Run the demo to see examples:

```bash
python -m agentic_ai_wf.drugs_extraction_evaluation.utils.openfda.demo_models
```

## File Structure

```
openfda/
├── models.py              # Main data models
├── fda_approval.py        # Updated to use models
├── label_info.py          # Updated to use models
├── text_processor.py      # Text processing utilities
├── test_models.py         # Model tests
├── demo_models.py         # Usage examples
├── README_MODELS.md       # This documentation
└── __init__.py           # Package exports
``` 