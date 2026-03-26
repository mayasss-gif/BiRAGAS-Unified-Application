# # src/pathway_prioritization/__init__.py
# """
# Pathway Prioritization System
# A scalable system for scoring and prioritizing biological pathways based on disease relevance
# """

# from .core.prioritizer import PathwayPrioritizer
# from .core.processor import PathwayDataProcessor
# from .models import PathwayData, PathwayScore, ProcessingConfig

# __all__ = [
#     'PathwayPrioritizer',
#     'PathwayDataProcessor', 
#     'PathwayData',
#     'PathwayScore',
#     'ProcessingConfig'
# ]

# src/pathway_prioritization/__init__.py
"""
Pathway Prioritization System
A scalable system for scoring and prioritizing biological pathways based on disease relevance
"""


from .core.prioritizer import PathwayPrioritizer
from .core.processor import PathwayDataProcessor
from .models import PathwayData, PathwayScore, ProcessingConfig
from .cli import main

__all__ = [
    'PathwayPrioritizer',
    'PathwayDataProcessor', 
    'PathwayData',
    'PathwayScore',
    'ProcessingConfig',
    'main'
]