"""
multiomics package

Exposes all main pipeline steps and helper modules:

Steps:
    - step1_ingestion
    - step2_preprocessing
    - step3_integration
    - step4_ml_biomarkers
    - step5_crossomics
    - step6_literature

Helpers / CLI:
    - cli
    - config
    - utils
"""

from . import step1_ingestion
from . import step2_preprocessing
from . import step3_integration
from . import step4_ml_biomarkers
from . import step5_crossomics
from . import step6_literature
from . import cli
from . import config
from . import utils
from .pipeline import run_pipeline

__all__ = [
    "step1_ingestion",
    "step2_preprocessing",
    "step3_integration",
    "step4_ml_biomarkers",
    "step5_crossomics",
    "step6_literature",
    "cli",
    "config",
    "utils",
    "run_pipeline",
]
