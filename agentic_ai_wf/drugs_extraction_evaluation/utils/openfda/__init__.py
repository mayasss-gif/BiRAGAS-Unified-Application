"""
OpenFDA utilities for drug information extraction and processing.

This package provides utilities for interacting with the FDA API
and processing drug label information.
"""

from .text_processor import TextProcessor
from .label_info import get_drug_label_info
from .adverse_reactions import get_adverse_reactions
from .fda_approval import verify_fda_approval, verify_fda_approval_dict
from .name_and_route import get_drug_names_and_route
from .models import (
    FDAApprovalInfo,
    DrugLabelInfo,
    DrugNamesAndRoute,
    FDAQueryResult,
    FDAApprovalStatus,
    RouteOfAdministration
)

__all__ = [
    'TextProcessor',
    'get_drug_label_info',
    'get_adverse_reactions', 
    'verify_fda_approval',
    'verify_fda_approval_dict',
    'get_drug_names_and_route',
    'FDAApprovalInfo',
    'DrugLabelInfo',
    'DrugNamesAndRoute',
    'FDAQueryResult',
    'FDAApprovalStatus',
    'RouteOfAdministration',
]
