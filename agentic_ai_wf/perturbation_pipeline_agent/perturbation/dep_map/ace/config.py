"""
ACE Configuration Settings
"""

from dataclasses import dataclass


@dataclass
class ACEConfig:
    """
    Configuration for ACE (Average Causal Effect) computation.
    
    Attributes:
        center: Method for computing central tendency ("median" or "mean")
        n_boot: Number of bootstrap iterations for confidence intervals
        seed: Random seed for reproducibility
        min_models: Minimum number of models required for gene analysis
        stability_threshold: Threshold for classifying effects as "Robust" vs "Unstable"
    """
    center: str = "median"
    n_boot: int = 1000
    seed: int = 7
    min_models: int = 10
    stability_threshold: float = 0.70
