"""
FDA Data Models Package.

This package contains structured data models for FDA drug information
using Python dataclasses with proper type hints, validation, and serialization.
"""

from .enums import FDAApprovalStatus, RouteOfAdministration
from .drug_label import DrugLabelInfo
from .drug_names import DrugNamesAndRoute
from .fda_approval import FDAApprovalInfo
from .query_result import FDAQueryResult
from .factories import (
    create_approved_drug,
    create_not_found_drug,
    create_error_drug,
    create_from_dict
)

__all__ = [
    'FDAApprovalStatus',
    'RouteOfAdministration', 
    'DrugLabelInfo',
    'DrugNamesAndRoute',
    'FDAApprovalInfo',
    'FDAQueryResult',
    'create_approved_drug',
    'create_not_found_drug',
    'create_error_drug',
    'create_from_dict',
] 