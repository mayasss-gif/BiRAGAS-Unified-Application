"""
H5AD preprocessing module for file size reduction.
"""

from .h5ad_reducer import (
    auto_reduce_h5ad,
    reduce_h5ad_file,
    determine_filtering_strategy,
    get_file_size_mb,
    get_file_size_gb,
)

__all__ = [
    "auto_reduce_h5ad",
    "reduce_h5ad_file",
    "determine_filtering_strategy",
    "get_file_size_mb",
    "get_file_size_gb",
]
