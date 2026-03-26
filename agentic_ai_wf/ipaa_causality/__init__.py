"""
IPAA Causality Pipeline - Production API.

Public API: run_ipaa_pipeline, IPAAAgent, IPAAConfig, ItemSpec.
"""
from .Pathway_Causality_Main import run_ipaa_pipeline
from .agent import IPAAAgent
from .config.models import IPAAConfig, ItemSpec
from .services import run_ipaa, run_causality, run_aggregation, run_reports

__all__ = [
    "run_ipaa_pipeline",
    "IPAAAgent",
    "IPAAConfig",
    "ItemSpec",
    "run_ipaa",
    "run_causality",
    "run_aggregation",
    "run_reports",
]
