"""
Enums for FDA drug information.
"""

from enum import Enum


class FDAApprovalStatus(str, Enum):
    """Enumeration for FDA approval status."""
    APPROVED = "Approved"
    NOT_FOUND = "not found"
    ERROR = "Error in Verification"
    PENDING = "Pending"
    WITHDRAWN = "Withdrawn"


class RouteOfAdministration(str, Enum):
    """Common routes of drug administration."""
    ORAL = "Oral"
    INTRAVENOUS = "Intravenous"
    INTRAMUSCULAR = "Intramuscular"
    SUBCUTANEOUS = "Subcutaneous"
    TOPICAL = "Topical"
    INHALATION = "Inhalation"
    RECTAL = "Rectal"
    VAGINAL = "Vaginal"
    OPHTHALMIC = "Ophthalmic"
    OTIC = "Otic"
    NASAL = "Nasal"
    TRANSDERMAL = "Transdermal"
    OTHER = "Other" 