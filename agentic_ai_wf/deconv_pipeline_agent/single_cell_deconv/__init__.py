"""
single_cell_deconv: top-level package init kept intentionally light.
Avoid importing subpackages here to prevent import-time crashes.
"""

from .run_analysis import run_pipeline

__version__ = "0.0.0"
__all__ = ["run_pipeline"]

