#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DEPRECATED — This standalone script is no longer used.

Dataset summary generation is now handled dynamically by
``build_header_dataset_stage0.py`` which fetches sample metadata
live from the NCBI GEO API via ``geo_fetch.py``.

See: crispr/reporting/geo_fetch.py
     crispr/reporting/build_header_dataset_stage0.py
"""

raise ImportError(
    "datasetLLM.py is deprecated. "
    "Dataset summaries are now generated dynamically via geo_fetch.py "
    "and build_header_dataset_stage0.py. "
    "See the reporting package documentation."
)
