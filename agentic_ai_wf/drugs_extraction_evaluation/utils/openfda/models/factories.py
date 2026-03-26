"""
Factory methods for creating FDA approval information instances.
"""

from typing import Optional, Dict, Any
from datetime import datetime

from .enums import FDAApprovalStatus
from .drug_label import DrugLabelInfo
from .fda_approval import FDAApprovalInfo


def create_approved_drug(drug_name: str, brand_name: str, generic_name: str, 
                        route: str, indications: str, reactions: str = "Not available",
                        label_info: Optional[DrugLabelInfo] = None) -> FDAApprovalInfo:
    """Factory function to create an approved drug instance."""
    return FDAApprovalInfo(
        drug_name=drug_name,
        fda_approved_status=FDAApprovalStatus.APPROVED,
        brand_name=brand_name,
        generic_name=generic_name,
        route_of_administration=route,
        fda_indications=indications,
        fda_adverse_reactions=reactions,
        label_info=label_info
    )


def create_not_found_drug(drug_name: str) -> FDAApprovalInfo:
    """Factory function to create a not found drug instance."""
    return FDAApprovalInfo(
        drug_name=drug_name,
        fda_approved_status=FDAApprovalStatus.NOT_FOUND
    )


def create_error_drug(drug_name: str, error_message: str = "Error in Verification") -> FDAApprovalInfo:
    """Factory function to create an error instance."""
    return FDAApprovalInfo(
        drug_name=drug_name,
        fda_approved_status=FDAApprovalStatus.ERROR,
        fda_indications=error_message,
        fda_adverse_reactions=error_message
    )


def create_from_dict(data: Dict[str, Any]) -> FDAApprovalInfo:
    """Create FDAApprovalInfo instance from dictionary."""
    # Handle enum conversion
    if 'fda_approved_status' in data and isinstance(data['fda_approved_status'], str):
        data['fda_approved_status'] = FDAApprovalStatus(data['fda_approved_status'])
    
    # Handle datetime conversion
    if 'query_timestamp' in data and isinstance(data['query_timestamp'], str):
        data['query_timestamp'] = datetime.fromisoformat(data['query_timestamp'])
    
    # Handle nested label_info
    if 'label_info' in data and isinstance(data['label_info'], dict):
        data['label_info'] = DrugLabelInfo.from_dict(data['label_info'])
    
    return FDAApprovalInfo(**data) 