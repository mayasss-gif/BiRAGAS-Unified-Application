"""
Drug label information model.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, Any
import json


@dataclass
class DrugLabelInfo:
    """
    Structured representation of FDA drug labeling information.
    
    Contains all the key sections from FDA drug labels including
    indications, dosage, warnings, and other clinical information.
    """
    
    indications_and_usage: str = field(default="Not available")
    dosage_and_administration: str = field(default="Not available")
    mechanism_of_action: str = field(default="Not available")
    warnings_and_precautions: str = field(default="Not available")
    contraindications: str = field(default="Not available")
    boxed_warning: str = field(default="Not available")
    patient_counseling_info: str = field(default="Not available")
    
    def __post_init__(self):
        """Validate and clean data after initialization."""
        # Ensure all fields are strings and not None
        for field_name, field_value in self.__dict__.items():
            if field_value is None:
                setattr(self, field_name, "Not available")
            elif not isinstance(field_value, str):
                setattr(self, field_name, str(field_value))
    
    def to_dict(self) -> Dict[str, str]:
        """Convert to dictionary format."""
        return asdict(self)
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DrugLabelInfo':
        """Create instance from dictionary."""
        return cls(**data)
    
    @classmethod
    def from_fda_response(cls, fda_data: Dict[str, Any]) -> 'DrugLabelInfo':
        """Create instance from FDA API response."""
        return cls(
            indications_and_usage=fda_data.get('indications_and_usage', ['Not available'])[0],
            dosage_and_administration=fda_data.get('dosage_and_administration', ['Not available'])[0],
            mechanism_of_action=fda_data.get('mechanism_of_action', 'Not available'),
            warnings_and_precautions=fda_data.get('warnings_and_precautions', ['Not available'])[0],
            contraindications=fda_data.get('contraindications', ['Not available'])[0],
            boxed_warning=fda_data.get('boxed_warning', ['Not available'])[0],
            patient_counseling_info=fda_data.get('information_for_patients', ['Not available'])[0]
        ) 