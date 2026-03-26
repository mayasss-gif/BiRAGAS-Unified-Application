# src/pathway_prioritization/models/pathway_models.py
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from pathlib import Path

@dataclass
class PathwayData:
    """Data class representing a biological pathway"""
    db_id: str
    pathway_source: str
    pathway_id: str
    pathway_name: str
    number_of_genes: int
    number_of_genes_in_background: int
    input_genes: str
    pathway_genes: str
    p_value: float
    fdr: float
    regulation: str
    clinical_relevance: str
    functional_relevance: str
    hit_score: float
    ontology_source: str
    main_class: str
    sub_class: str
    disease_category: str = ""
    disease_subcategory: str = ""
    cellular_component: str = ""
    subcellular_element: str = ""
    references: str = ""
    audit_log: str = ""
    relation: str = ""

@dataclass
class PathwayScore:
    """Data class representing pathway scoring results"""
    pathway_data: PathwayData
    llm_score: float
    score_justification: str
    confidence_level: str
    priority_rank: int = 0

@dataclass 
class DiseaseContext:
    """Data class representing disease-specific context"""
    key_pathways: List[str]
    molecular_features: List[str]
    therapeutic_targets: List[str]
    disease_category: str
    source: str

@dataclass
class ProcessingConfig:
    """Configuration for pathway processing"""
    batch_size: int = 10
    top_n_pathways: int = 100
    max_workers: int = 5
    apply_llm_to_kegg_only: bool = False
    output_dir: Path = Path("results")