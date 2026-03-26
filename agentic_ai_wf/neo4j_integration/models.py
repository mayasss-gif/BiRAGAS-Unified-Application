"""
Neo4j Data Models for Biomedical Entities

Defines data structures and methods for creating nodes and relationships
in the Neo4j graph database.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)


@dataclass
class GeneNode:
    """Gene node model for Neo4j."""
    symbol: str
    ensembl_id: Optional[str] = None
    entrez_id: Optional[str] = None
    description: Optional[str] = None
    chromosome: Optional[str] = None
    start_position: Optional[int] = None
    end_position: Optional[int] = None
    gene_type: Optional[str] = None
    log2_fold_change: Optional[float] = None
    p_value: Optional[float] = None
    adjusted_p_value: Optional[float] = None

    def to_cypher_create(self) -> tuple[str, Dict[str, Any]]:
        """Generate Cypher CREATE statement for this gene."""
        properties = {
            "symbol": self.symbol,
            "ensembl_id": self.ensembl_id,
            "entrez_id": self.entrez_id,
            "description": self.description,
            "chromosome": self.chromosome,
            "start_position": self.start_position,
            "end_position": self.end_position,
            "gene_type": self.gene_type,
            "log2_fold_change": self.log2_fold_change,
            "p_value": self.p_value,
            "adjusted_p_value": self.adjusted_p_value
        }
        # Remove None values
        properties = {k: v for k, v in properties.items() if v is not None}

        query = "CREATE (g:Gene $properties) RETURN g"
        return query, {"properties": properties}

    def to_cypher_merge(self) -> tuple[str, Dict[str, Any]]:
        """Generate Cypher MERGE statement for this gene."""
        properties = {
            "symbol": self.symbol,
            "ensembl_id": self.ensembl_id,
            "entrez_id": self.entrez_id,
            "description": self.description,
            "chromosome": self.chromosome,
            "start_position": self.start_position,
            "end_position": self.end_position,
            "gene_type": self.gene_type,
            "log2_fold_change": self.log2_fold_change,
            "p_value": self.p_value,
            "adjusted_p_value": self.adjusted_p_value
        }
        # Remove None values
        properties = {k: v for k, v in properties.items() if v is not None}

        query = "MERGE (g:Gene {symbol: $symbol}) SET g += $properties RETURN g"
        return query, {"symbol": self.symbol, "properties": properties}


@dataclass
class PathwayNode:
    """Pathway node model for Neo4j."""
    pathway_id: str
    name: str
    source: str  # KEGG, Reactome, GO, etc.
    description: Optional[str] = None
    category: Optional[str] = None
    p_value: Optional[float] = None
    adjusted_p_value: Optional[float] = None
    enrichment_score: Optional[float] = None
    gene_count: Optional[int] = None

    def to_cypher_create(self) -> tuple[str, Dict[str, Any]]:
        """Generate Cypher CREATE statement for this pathway."""
        properties = {
            "pathway_id": self.pathway_id,
            "name": self.name,
            "source": self.source,
            "description": self.description,
            "category": self.category,
            "p_value": self.p_value,
            "adjusted_p_value": self.adjusted_p_value,
            "enrichment_score": self.enrichment_score,
            "gene_count": self.gene_count
        }
        # Remove None values
        properties = {k: v for k, v in properties.items() if v is not None}

        query = "CREATE (p:Pathway $properties) RETURN p"
        return query, {"properties": properties}

    def to_cypher_merge(self) -> tuple[str, Dict[str, Any]]:
        """Generate Cypher MERGE statement for this pathway."""
        properties = {
            "pathway_id": self.pathway_id,
            "name": self.name,
            "source": self.source,
            "description": self.description,
            "category": self.category,
            "p_value": self.p_value,
            "adjusted_p_value": self.adjusted_p_value,
            "enrichment_score": self.enrichment_score,
            "gene_count": self.gene_count
        }
        # Remove None values
        properties = {k: v for k, v in properties.items() if v is not None}

        query = "MERGE (p:Pathway {pathway_id: $pathway_id}) SET p += $properties RETURN p"
        return query, {"pathway_id": self.pathway_id, "properties": properties}


@dataclass
class DrugNode:
    """Drug node model for Neo4j."""
    name: str
    drug_id: Optional[str] = None
    drugbank_id: Optional[str] = None
    chembl_id: Optional[str] = None
    mechanism_of_action: Optional[str] = None
    drug_class: Optional[str] = None
    fda_approved: Optional[bool] = None
    clinical_trial_phase: Optional[str] = None
    brand_names: Optional[List[str]] = None
    target_genes: Optional[List[str]] = None

    def to_cypher_create(self) -> tuple[str, Dict[str, Any]]:
        """Generate Cypher CREATE statement for this drug."""
        properties = {
            "name": self.name,
            "drug_id": self.drug_id,
            "drugbank_id": self.drugbank_id,
            "chembl_id": self.chembl_id,
            "mechanism_of_action": self.mechanism_of_action,
            "drug_class": self.drug_class,
            "fda_approved": self.fda_approved,
            "clinical_trial_phase": self.clinical_trial_phase,
            "brand_names": self.brand_names,
            "target_genes": self.target_genes
        }
        # Remove None values
        properties = {k: v for k, v in properties.items() if v is not None}

        query = "CREATE (d:Drug $properties) RETURN d"
        return query, {"properties": properties}

    def to_cypher_merge(self) -> tuple[str, Dict[str, Any]]:
        """Generate Cypher MERGE statement for this drug."""
        properties = {
            "name": self.name,
            "drug_id": self.drug_id,
            "drugbank_id": self.drugbank_id,
            "chembl_id": self.chembl_id,
            "mechanism_of_action": self.mechanism_of_action,
            "drug_class": self.drug_class,
            "fda_approved": self.fda_approved,
            "clinical_trial_phase": self.clinical_trial_phase,
            "brand_names": self.brand_names,
            "target_genes": self.target_genes
        }
        # Remove None values
        properties = {k: v for k, v in properties.items() if v is not None}

        query = "MERGE (d:Drug {name: $name}) SET d += $properties RETURN d"
        return query, {"name": self.name, "properties": properties}


@dataclass
class DiseaseNode:
    """Disease node model for Neo4j."""
    name: str
    disease_id: Optional[str] = None
    description: Optional[str] = None
    category: Optional[str] = None
    icd10_code: Optional[str] = None
    mesh_id: Optional[str] = None

    def to_cypher_create(self) -> tuple[str, Dict[str, Any]]:
        """Generate Cypher CREATE statement for this disease."""
        properties = {
            "name": self.name,
            "disease_id": self.disease_id,
            "description": self.description,
            "category": self.category,
            "icd10_code": self.icd10_code,
            "mesh_id": self.mesh_id
        }
        # Remove None values
        properties = {k: v for k, v in properties.items() if v is not None}

        query = "CREATE (dis:Disease $properties) RETURN dis"
        return query, {"properties": properties}

    def to_cypher_merge(self) -> tuple[str, Dict[str, Any]]:
        """Generate Cypher MERGE statement for this disease."""
        properties = {
            "name": self.name,
            "disease_id": self.disease_id,
            "description": self.description,
            "category": self.category,
            "icd10_code": self.icd10_code,
            "mesh_id": self.mesh_id
        }
        # Remove None values
        properties = {k: v for k, v in properties.items() if v is not None}

        query = "MERGE (dis:Disease {name: $name}) SET dis += $properties RETURN dis"
        return query, {"name": self.name, "properties": properties}


def create_relationship(from_node_label: str, from_property: str, from_value: str,
                        to_node_label: str, to_property: str, to_value: str,
                        relationship_type: str, properties: Optional[Dict[str, Any]] = None) -> tuple[str, Dict[str, Any]]:
    """
    Create a relationship between two nodes.

    Args:
        from_node_label: Label of the source node (e.g., 'Gene')
        from_property: Property to match on source node (e.g., 'symbol')
        from_value: Value of the property on source node
        to_node_label: Label of the target node (e.g., 'Pathway')
        to_property: Property to match on target node (e.g., 'pathway_id')
        to_value: Value of the property on target node
        relationship_type: Type of relationship (e.g., 'BELONGS_TO')
        properties: Optional properties for the relationship

    Returns:
        Tuple of (query_string, parameters)
    """
    properties = properties or {}

    if properties:
        props_str = " $rel_properties"
        params = {
            "from_value": from_value,
            "to_value": to_value,
            "rel_properties": properties
        }
    else:
        props_str = ""
        params = {
            "from_value": from_value,
            "to_value": to_value
        }

    query = f"""
    MATCH (from:{from_node_label} {{{from_property}: $from_value}})
    MATCH (to:{to_node_label} {{{to_property}: $to_value}})
    MERGE (from)-[r:{relationship_type}{props_str}]->(to)
    RETURN from, r, to
    """

    return query, params


# Common relationship types
class RelationshipTypes:
    """Common relationship types used in biomedical graphs."""
    BELONGS_TO = "BELONGS_TO"  # Gene belongs to pathway
    TARGETS = "TARGETS"  # Drug targets gene
    TREATS = "TREATS"  # Drug treats disease
    ASSOCIATED_WITH = "ASSOCIATED_WITH"  # Gene associated with disease
    INTERACTS_WITH = "INTERACTS_WITH"  # Gene/protein interactions
    REGULATES = "REGULATES"  # Gene regulates another gene
    ENRICHED_IN = "ENRICHED_IN"  # Pathway enriched in condition
    PARTICIPATES_IN = "PARTICIPATES_IN"  # Gene participates in pathway
