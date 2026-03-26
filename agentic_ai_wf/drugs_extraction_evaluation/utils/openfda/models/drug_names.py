"""
Drug names and route model.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, Any


@dataclass
class DrugNamesAndRoute:
    """
    Structured representation of drug names and administration route.
    """
    
    brand_name: str = field(default="Not available")
    generic_name: str = field(default="Not available")
    route_of_administration: str = field(default="Not available")
    
    def __post_init__(self):
        """Validate and clean data after initialization."""
        if self.brand_name is None:
            self.brand_name = "Not available"
        if self.generic_name is None:
            self.generic_name = "Not available"
        if self.route_of_administration is None:
            self.route_of_administration = "Not available"
    
    def to_dict(self) -> Dict[str, str]:
        """Convert to dictionary format."""
        return asdict(self, dict_factory=dict)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DrugNamesAndRoute':
        """Create instance from dictionary."""
        return cls(**data)
    
    @classmethod
    def from_fda_response(cls, fda_data: Dict[str, Any]) -> 'DrugNamesAndRoute':
        """Create instance from FDA API response."""
        return cls(
            brand_name=fda_data.get('brand_name', 'Not available'),
            generic_name=fda_data.get('generic_name', 'Not available'),
            route_of_administration=', '.join(fda_data.get('route', ['Not available']))
        ) 