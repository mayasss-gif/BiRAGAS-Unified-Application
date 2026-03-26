"""
Main FDA approval information model.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any
import json
from datetime import datetime

from .enums import FDAApprovalStatus
from .drug_label import DrugLabelInfo


@dataclass
class FDAApprovalInfo:
    """
    Comprehensive FDA approval information for a drug.
    
    This is the main data model that combines all FDA-related information
    about a drug including approval status, names, routes, and clinical data.
    """
    
    # Basic identification
    drug_name: str = field(default="")
    fda_approved_status: FDAApprovalStatus = field(default=FDAApprovalStatus.NOT_FOUND)
    
    # Drug names and administration
    brand_name: str = field(default="Not available")
    generic_name: str = field(default="Not available")
    route_of_administration: str = field(default="Not available")
    
    # Clinical information
    fda_indications: str = field(default="Not available")
    fda_adverse_reactions: str = field(default="Not available")
    
    # Detailed label information
    label_info: Optional[DrugLabelInfo] = field(default=None)
    
    # Metadata
    query_timestamp: datetime = field(default_factory=datetime.now)
    source_api: str = field(default="OpenFDA")
    
    def __post_init__(self):
        """Validate and clean data after initialization."""
        # Ensure drug_name is not None
        if self.drug_name is None:
            self.drug_name = ""
        
        # Ensure all string fields are not None
        for field_name, field_value in self.__dict__.items():
            if isinstance(field_value, str) and field_value is None:
                setattr(self, field_name, "Not available")
        
        # Initialize label_info if None
        if self.label_info is None:
            self.label_info = DrugLabelInfo()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        data = asdict(self)
        # Convert enum to string
        data['fda_approved_status'] = self.fda_approved_status.value
        # Convert datetime to string
        data['query_timestamp'] = self.query_timestamp.isoformat()
        return data
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2, default=str)
    
    def is_approved(self) -> bool:
        """Check if the drug is FDA approved."""
        return self.fda_approved_status == FDAApprovalStatus.APPROVED
    
    def has_label_info(self) -> bool:
        """Check if detailed label information is available."""
        return self.label_info is not None and any(
            value != "Not available" for value in self.label_info.to_dict().values()
        )
    
    def get_summary(self) -> Dict[str, str]:
        """Get a summary of key information."""
        return {
            'drug_name': self.drug_name,
            'status': self.fda_approved_status.value,
            'brand_name': self.brand_name,
            'generic_name': self.generic_name,
            'route': self.route_of_administration,
            'has_detailed_info': str(self.has_label_info())
        } 