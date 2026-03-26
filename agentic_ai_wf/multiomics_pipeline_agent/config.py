from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class LayerPaths:
    genomics: Optional[str] = None
    transcriptomics: Optional[str] = None
    epigenomics: Optional[str] = None
    proteomics: Optional[str] = None
    metabolomics: Optional[str] = None


@dataclass
class PipelineConfig:
    base_dir: str
    layers: LayerPaths
    metadata: Optional[str]
    label_column: Optional[str]
    n_pcs_per_layer: int = 20
    integrated_dim: int = 50

    query_term: Optional[str] = None
    email_for_ncbi: Optional[str] = None
    top_n_results: int = 20


def load_config(path: str) -> PipelineConfig:
    """Load YAML config into a strongly-typed object."""
    data = yaml.safe_load(Path(path).read_text())

    layers = LayerPaths(**data.get("layers", {}))

    return PipelineConfig(
        base_dir=data["base_dir"],
        layers=layers,
        metadata=data.get("metadata"),
        label_column=data.get("label_column"),
        n_pcs_per_layer=int(data.get("n_pcs_per_layer", 20)),
        integrated_dim=int(data.get("integrated_dim", 50)),
        query_term=data.get("query_term"),
        email_for_ncbi=data.get("email_for_ncbi"),
        top_n_results=int(data.get("top_n_results", 20)),
    )
