"""
FDA query result container model.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any
import json
from datetime import datetime

from .fda_approval import FDAApprovalInfo


@dataclass
class FDAQueryResult:
    """
    Container for FDA query results with metadata.
    """
    
    query_drug_name: str
    results: List[FDAApprovalInfo] = field(default_factory=list)
    total_results: int = field(default=0)
    query_timestamp: datetime = field(default_factory=datetime.now)
    query_duration_ms: Optional[float] = field(default=None)
    success: bool = field(default=True)
    error_message: Optional[str] = field(default=None)
    
    def add_result(self, result: FDAApprovalInfo) -> None:
        """Add a result to the query results."""
        self.results.append(result)
        self.total_results = len(self.results)
    
    def get_approved_drugs(self) -> List[FDAApprovalInfo]:
        """Get only approved drugs from results."""
        return [result for result in self.results if result.is_approved()]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        data = asdict(self)
        data['query_timestamp'] = self.query_timestamp.isoformat()
        return data
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2, default=str) 