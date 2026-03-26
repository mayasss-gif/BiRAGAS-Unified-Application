"""
Cypher Query Builder for Common Biomedical Queries

Provides high-level methods for building complex Cypher queries
for biomedical data analysis.
"""

from typing import Dict, List, Optional, Any, cast, Tuple
import logging

logger = logging.getLogger(__name__)


class CypherQueryBuilder:
    """
    High-level query builder for biomedical graph queries.
    """

    @staticmethod
    def find_genes_by_disease(disease_name: str, limit: int = 100) -> tuple[str, Dict[str, Any]]:
        """Find genes associated with a specific disease."""
        query = """
        MATCH (dis:Disease {name: $disease_name})-[:ASSOCIATED_WITH]-(g:Gene)
        RETURN g.symbol as gene_symbol, g.ensembl_id, g.log2_fold_change, g.p_value
        ORDER BY g.p_value ASC
        LIMIT $limit
        """
        return query, {"disease_name": disease_name, "limit": limit}

    @staticmethod
    def find_pathways_by_gene(gene_symbol: str, limit: int = 50) -> tuple[str, Dict[str, Any]]:
        """Find pathways containing a specific gene."""
        query = """
        MATCH (g:Gene {symbol: $gene_symbol})-[:BELONGS_TO]->(p:Pathway)
        RETURN p.pathway_id, p.name, p.source, p.p_value, p.enrichment_score
        ORDER BY p.p_value ASC
        LIMIT $limit
        """
        return query, {"gene_symbol": gene_symbol, "limit": limit}

    @staticmethod
    def find_drugs_by_gene(gene_symbol: str, limit: int = 30) -> tuple[str, Dict[str, Any]]:
        """Find drugs that target a specific gene."""
        query = """
        MATCH (g:Gene {symbol: $gene_symbol})<-[:TARGETS]-(d:Drug)
        RETURN d.name, d.mechanism_of_action, d.drug_class, d.fda_approved
        ORDER BY d.fda_approved DESC, d.name
        LIMIT $limit
        """
        return query, {"gene_symbol": gene_symbol, "limit": limit}

    @staticmethod
    def find_drugs_by_pathway(pathway_id: str, limit: int = 50) -> tuple[str, Dict[str, Any]]:
        """Find drugs that target genes in a specific pathway."""
        query = """
        MATCH (p:Pathway {pathway_id: $pathway_id})<-[:BELONGS_TO]-(g:Gene)<-[:TARGETS]-(d:Drug)
        RETURN DISTINCT d.name, d.mechanism_of_action, d.fda_approved, 
               count(g) as target_genes_in_pathway
        ORDER BY target_genes_in_pathway DESC, d.fda_approved DESC
        LIMIT $limit
        """
        return query, {"pathway_id": pathway_id, "limit": limit}

    @staticmethod
    def find_enriched_pathways(min_p_value: float = 0.05,
                               min_gene_count: int = 3,
                               limit: int = 100) -> tuple[str, Dict[str, Any]]:
        """Find significantly enriched pathways."""
        query = """
        MATCH (p:Pathway)
        WHERE p.p_value <= $min_p_value AND p.gene_count >= $min_gene_count
        RETURN p.pathway_id, p.name, p.source, p.p_value, p.enrichment_score, p.gene_count
        ORDER BY p.p_value ASC, p.enrichment_score DESC
        LIMIT $limit
        """
        return query, {
            "min_p_value": min_p_value,
            "min_gene_count": min_gene_count,
            "limit": limit
        }

    @staticmethod
    def find_gene_interactions(gene_symbol: str,
                               interaction_types: Optional[List[str]] = None,
                               limit: int = 50) -> tuple[str, Dict[str, Any]]:
        """Find genes that interact with a specific gene."""
        if interaction_types:
            rel_filter = f"WHERE type(r) IN {cast(str, interaction_types)}"
            params = {"gene_symbol": gene_symbol, "limit": limit}
        else:
            rel_filter = ""
            params = {"gene_symbol": gene_symbol, "limit": limit}

        query = f"""
        MATCH (g1:Gene {{symbol: $gene_symbol}})-[r]-(g2:Gene)
        {rel_filter}
        RETURN g2.symbol, type(r) as relationship_type, r
        LIMIT $limit
        """
        return query, params

    @staticmethod
    def find_drug_targets_in_pathways(drug_name: str, limit: int = 50) -> tuple[str, Dict[str, Any]]:
        """Find which pathways contain genes targeted by a specific drug."""
        query = """
        MATCH (d:Drug {name: $drug_name})-[:TARGETS]->(g:Gene)-[:BELONGS_TO]->(p:Pathway)
        RETURN p.pathway_id, p.name, p.source, 
               collect(DISTINCT g.symbol) as target_genes,
               count(DISTINCT g) as target_count
        ORDER BY target_count DESC
        LIMIT $limit
        """
        return query, {"drug_name": drug_name, "limit": limit}

    @staticmethod
    def find_shortest_path_gene_to_drug(gene_symbol: str,
                                        drug_name: str,
                                        max_hops: int = 4) -> tuple[str, Dict[str, Any]]:
        """Find shortest path from gene to drug through the graph."""
        query = """
        MATCH path = shortestPath(
            (g:Gene {symbol: $gene_symbol})-[*1..$max_hops]-(d:Drug {name: $drug_name})
        )
        RETURN path, length(path) as path_length
        ORDER BY path_length ASC
        LIMIT 1
        """
        return query, {
            "gene_symbol": gene_symbol,
            "drug_name": drug_name,
            "max_hops": max_hops
        }

    @staticmethod
    def get_node_degree(node_label: str,
                        property_name: str,
                        property_value: str) -> tuple[str, Dict[str, Any]]:
        """Get the degree (number of connections) for a specific node."""
        query = f"""
        MATCH (n:{node_label} {{{property_name}: $property_value}})
        OPTIONAL MATCH (n)-[r]-()
        RETURN n, count(r) as degree
        """
        return query, {"property_value": property_value}

    @staticmethod
    def find_common_pathways_for_genes(gene_symbols: List[str],
                                       min_genes: int = 2,
                                       limit: int = 50) -> tuple[str, Dict[str, Any]]:
        """Find pathways that contain multiple genes from the given list."""
        query = """
        MATCH (g:Gene)-[:BELONGS_TO]->(p:Pathway)
        WHERE g.symbol IN $gene_symbols
        WITH p, collect(DISTINCT g.symbol) as genes_in_pathway
        WHERE size(genes_in_pathway) >= $min_genes
        RETURN p.pathway_id, p.name, p.source, genes_in_pathway, 
               size(genes_in_pathway) as gene_count
        ORDER BY gene_count DESC, p.p_value ASC
        LIMIT $limit
        """
        return query, {
            "gene_symbols": gene_symbols,
            "min_genes": min_genes,
            "limit": limit
        }

    @staticmethod
    def find_hub_genes(min_degree: int = 10, limit: int = 100) -> tuple[str, Dict[str, Any]]:
        """Find highly connected genes (hub genes)."""
        query = """
        MATCH (g:Gene)
        OPTIONAL MATCH (g)-[r]-()
        WITH g, count(r) as degree
        WHERE degree >= $min_degree
        RETURN g.symbol, g.ensembl_id, g.description, degree
        ORDER BY degree DESC
        LIMIT $limit
        """
        return query, {"min_degree": min_degree, "limit": limit}

    @staticmethod
    def analyze_pathway_overlap(pathway_ids: List[str]) -> tuple[str, Dict[str, Any]]:
        """Analyze gene overlap between multiple pathways."""
        query = """
        MATCH (p:Pathway)-[:BELONGS_TO]-(g:Gene)
        WHERE p.pathway_id IN $pathway_ids
        WITH g, collect(DISTINCT p.pathway_id) as pathways
        WHERE size(pathways) > 1
        RETURN g.symbol, pathways, size(pathways) as pathway_count
        ORDER BY pathway_count DESC, g.symbol
        """
        return query, {"pathway_ids": pathway_ids}

    @staticmethod
    def find_potential_drug_targets(disease_name: str,
                                    min_fold_change: float = 1.5,
                                    max_p_value: float = 0.05,
                                    limit: int = 50) -> tuple[str, Dict[str, Any]]:
        """Find potential drug targets for a disease based on gene expression."""
        query = """
        MATCH (dis:Disease {name: $disease_name})-[:ASSOCIATED_WITH]-(g:Gene)
        WHERE abs(g.log2_fold_change) >= $min_fold_change 
          AND g.p_value <= $max_p_value
        OPTIONAL MATCH (g)<-[:TARGETS]-(d:Drug)
        RETURN g.symbol, g.log2_fold_change, g.p_value, 
               collect(DISTINCT d.name) as existing_drugs,
               count(DISTINCT d) as drug_count
        ORDER BY abs(g.log2_fold_change) DESC, g.p_value ASC
        LIMIT $limit
        """
        return query, {
            "disease_name": disease_name,
            "min_fold_change": min_fold_change,
            "max_p_value": max_p_value,
            "limit": limit
        }

    @staticmethod
    def find_genes_by_disease_with_scores(disease_name: str, limit: int = 10000) -> tuple[str, Dict[str, Any]]:
        """Find genes associated with a specific disease with GeneCards scores."""
        query = """
        MATCH (d:Disease)-[r:ASSOCIATED_WITH|INFERRED_ASSOCIATION]-(g:Gene)
            WHERE (
                toLower(d.name) CONTAINS toLower($disease_name) OR 
                toLower($disease_name) CONTAINS toLower(d.name) OR 
                (d.aliases IS NOT NULL AND any(alias IN d.aliases WHERE toLower(alias) CONTAINS toLower($disease_name)))
            ) 
            AND r.gene_score IS NOT NULL
            WITH g.symbol AS gene_symbol, 
                r.gene_score AS gene_score, 
                r.disorder_score AS disorder_score, 
                r.disorder_type AS disorder_type
            ORDER BY gene_score DESC
            WITH gene_symbol, collect({gene_score: gene_score, disorder_score: disorder_score, disorder_type: disorder_type})[0] AS top_hit
            RETURN 
                gene_symbol,
                top_hit.gene_score AS gene_score,
                top_hit.disorder_score AS disorder_score,
                top_hit.disorder_type AS disorder_type
            ORDER BY gene_score DESC
            LIMIT $limit
        """
        return query, {"disease_name": disease_name, "limit": limit}

    @staticmethod
    def find_drugs_by_disease(disease_name: str, limit: int = 100) -> tuple[str, Dict[str, Any]]:
        """Find drugs associated with a specific disease."""
        query = """
        MATCH (d:Disease {name: $disease_name})-[:ASSOCIATED_WITH]-(g:Gene)<-[:TARGETS]-(d:Drug)
        RETURN d.name, d.mechanism_of_action, d.drug_class, d.fda_approved
        ORDER BY d.fda_approved DESC, d.name
        LIMIT $limit"""

        return query, {"disease_name": disease_name, "limit": limit}

    @staticmethod
    def find_pathways_by_disease(disease_name: str, limit: int = 100) -> tuple[str, Dict[str, Any]]:
        """Find pathways associated with a specific disease."""
        query = """
            MATCH (d:Disease)
            WHERE d.name CONTAINS $disease_name OR $disease_name CONTAINS d.name
            MATCH (g:Gene)-[r:ASSOCIATED_WITH|INFERRED_ASSOCIATION]->(d)
            MATCH (g)-[pr:BELONGS_TO_PATHWAY|BELONGS_TO_SUPER_PATHWAY]->(p:Pathway)
            RETURN DISTINCT p.name as pathway_name, p.pathway_id as pathway_id, 
                COLLECT(DISTINCT g.symbol) as disease_genes,
                SIZE(COLLECT(DISTINCT g.symbol)) as gene_count
            ORDER BY gene_count DESC
            LIMIT $limit
            """
        return query, {"disease_name": disease_name, "limit": limit}

    @staticmethod
    def get_pathway_drugs(pathway_id: str, limit: int = 5000) -> tuple[str, Dict[str, Any]]:
        """
        Get drugs associated with a KEGG pathway from Neo4j.
        
        Args:
            pathway_id: KEGG pathway ID (e.g., 'hsa05206')
            limit: Maximum number of drugs to return
            
        Returns:
            Tuple of (query, parameters)
        """
        query = """
        MATCH (p:KeggPathway {pathway_id: $pathway_id})
        OPTIONAL MATCH (p)-[:CONTAINS_DRUG]->(d:KeggDrug)
        WHERE d IS NOT NULL AND toLower(d.fda_approved_status) = 'approved'

        WITH p, d
        OPTIONAL MATCH (d)-[:TARGETS_KEGG_GENE]->(kg:KeggDrugGene)
        WITH p, d, collect(DISTINCT kg.symbol) AS kegg_symbols

        OPTIONAL MATCH (d)-[:TARGETS]->(g:Gene)
        WITH p, d, kegg_symbols, collect(DISTINCT g.symbol) AS gene_symbols

        OPTIONAL MATCH (d)-[:BELONGS_TO_CLASS]->(dc:DrugClass)
        WITH p, d, kegg_symbols, gene_symbols, collect(DISTINCT dc.name) AS drug_classes

        RETURN DISTINCT
            d.drug_name AS drug_name,
            d.kegg_id AS drug_id,
            d.mechanism_of_action AS mechanism_of_action,
            d.generic_name AS generic_name,
            d.brand_name AS brand_name,
            d.route_of_administration AS route_of_administration,
            d.fda_approved_status AS fda_approved_status,
            d.efficacy AS efficacy,
            d.target_genes_updated AS target_genes_property,
            kegg_symbols + gene_symbols AS target_genes,
            drug_classes AS drug_classes,
            d.drugbank_id AS drugbank_id,
            d.chembl_id AS chembl_id,
            p.pathway_name AS pathway_name,
            p.pathway_id AS pathway_id
        ORDER BY d.drug_name
        LIMIT $limit
        """
        return query, {"pathway_id": pathway_id, "limit": limit}

    @staticmethod
    def get_drugs_by_gene_symbol(gene_symbol: str, limit: int = 5000) -> tuple[str, Dict[str, Any]]:
        """
        Get drugs that target a specific gene by symbol from Neo4j.
        
        Args:
            gene_symbol: Gene symbol to search for (e.g., 'SCN3A')
            limit: Maximum number of drugs to return
            
        Returns:
            Tuple of (query, parameters)
        """
        query = """
        WITH toLower($gene_symbol) AS sym

        // 1) via Gene (preferred)
        OPTIONAL MATCH (g:Gene)
        WHERE toLower(g.symbol) = sym
           OR any(a IN coalesce(g.aliases, []) WHERE toLower(a) = sym)
        OPTIONAL MATCH (d1:KeggDrug)-[r1:TARGETS]->(g)
        WITH sym, collect(DISTINCT {d:d1, rel:r1, source:'Gene', hit_symbol:g.symbol}) AS gene_hits

        // 2) via KeggDrugGene (fallback)
        OPTIONAL MATCH (kg:KeggDrugGene)
        WHERE toLower(kg.symbol) = sym
        OPTIONAL MATCH (d2:KeggDrug)-[r2:TARGETS_KEGG_GENE]->(kg)
        WITH sym, gene_hits,
             collect(DISTINCT {d:d2, rel:r2, source:'KeggDrugGene', hit_symbol:kg.symbol}) AS kegg_hits

        // Combine AFTER both aggregates are materialized
        WITH gene_hits + kegg_hits AS hits
        UNWIND hits AS h
        WITH h WHERE h.d IS NOT NULL
        WITH h.d AS d,
             collect(DISTINCT h.source)      AS evidence_sources,
             collect(DISTINCT h.hit_symbol)  AS matched_symbols
        RETURN DISTINCT
          d.kegg_id                AS drug_id,
          d.drug_name              AS drug_name,
          d.fda_approved_status    AS fda_approved_status,
          d.mechanism_of_action    AS mechanism_of_action,
          d.generic_name           AS generic_name,
          d.brand_name             AS brand_name,
          d.route_of_administration AS route_of_administration,
          d.efficacy               AS efficacy,
          d.target_genes_updated   AS target_genes_property,
          d.drugbank_id            AS drugbank_id,
          d.chembl_id              AS chembl_id,
          evidence_sources,
          matched_symbols
        ORDER BY drug_name
        LIMIT $limit
        """
        return query, {"gene_symbol": gene_symbol, "limit": limit}
        
    @staticmethod
    def get_reactome_pathway_drugs(pathway_id: str, limit: int = 5000) -> tuple[str, Dict[str, Any]]:
        """
        Get drugs associated with a KEGG pathway from Neo4j.
        
        Args:
            pathway_id: KEGG pathway ID (e.g., 'hsa05206')
            limit: Maximum number of drugs to return
            
        Returns:
            Tuple of (query, parameters)
        """
        query = """
        MATCH (p:Pathway {pathway_id: $pathway_id})
        OPTIONAL MATCH (p)-[:CONTAINS_DRUG]->(d:KeggDrug)
        WHERE d IS NOT NULL AND toLower(d.fda_approved_status) = 'approved'

        WITH p, d
        OPTIONAL MATCH (d)-[:TARGETS_KEGG_GENE]->(kg:KeggDrugGene)
        WITH p, d, collect(DISTINCT kg.symbol) AS kegg_symbols

        OPTIONAL MATCH (d)-[:TARGETS]->(g:Gene)
        WITH p, d, kegg_symbols, collect(DISTINCT g.symbol) AS gene_symbols

        OPTIONAL MATCH (d)-[:BELONGS_TO_CLASS]->(dc:DrugClass)
        WITH p, d, kegg_symbols, gene_symbols, collect(DISTINCT dc.name) AS drug_classes

        RETURN DISTINCT
            d.drug_name AS drug_name,
            d.kegg_id AS drug_id,
            d.mechanism_of_action AS mechanism_of_action,
            d.generic_name AS generic_name,
            d.brand_name AS brand_name,
            d.route_of_administration AS route_of_administration,
            d.fda_approved_status AS fda_approved_status,
            d.efficacy AS efficacy,
            d.target_genes_updated AS target_genes_property,
            kegg_symbols + gene_symbols AS target_genes,
            drug_classes AS drug_classes,
            d.drugbank_id AS drugbank_id,
            d.chembl_id AS chembl_id,
            p.pathway_name AS pathway_name,
            p.pathway_id AS pathway_id
        ORDER BY d.drug_name
        LIMIT $limit
        """
        return query, {"pathway_id": pathway_id, "limit": limit}

    @staticmethod
    def get_pathway_drugs_by_name(pathway_name: str, limit: int = 5000) -> tuple[str, Dict[str, Any]]:
        """
        Get drugs associated with a KEGG pathway from Neo4j.
        
        Args:
            pathway_name: KEGG pathway name (e.g., 'PI3K-Akt signaling pathway')
            limit: Maximum number of drugs to return
            
        Returns:
            Tuple of (query, parameters)
        """
        query = """
        WITH toLower($pathway_name) AS query
        MATCH (p:KeggPathway)
        WHERE toLower(p.pathway_name) CONTAINS query
        OR toLower(p.pathway_name) STARTS WITH query
        OR apoc.text.levenshteinSimilarity(toLower(p.pathway_name), query) > 0.5

        OPTIONAL MATCH (p)-[:CONTAINS_DRUG]->(d:KeggDrug)
        WHERE d IS NOT NULL

        WITH p, d
        OPTIONAL MATCH (d)-[:TARGETS_KEGG_GENE]->(kg:KeggDrugGene)
        WITH p, d, collect(DISTINCT kg.symbol) AS kegg_symbols

        OPTIONAL MATCH (d)-[:TARGETS]->(g:Gene)
        WITH p, d, kegg_symbols, collect(DISTINCT g.symbol) AS gene_symbols

        OPTIONAL MATCH (d)-[:BELONGS_TO_CLASS]->(dc:DrugClass)
        WITH p, d, kegg_symbols, gene_symbols, collect(DISTINCT dc.name) AS drug_classes

        RETURN DISTINCT
            d.drug_name AS drug_name,
            d.kegg_id AS drug_id,
            d.mechanism_of_action AS mechanism_of_action,
            d.generic_name AS generic_name,
            d.brand_name AS brand_name,
            d.route_of_administration AS route_of_administration,
            d.fda_approved_status AS fda_approved_status,
            d.efficacy AS efficacy,
            d.target_genes_updated AS target_genes_property,
            kegg_symbols + gene_symbols AS target_genes,
            drug_classes AS drug_classes,
            d.drugbank_id AS drugbank_id,
            d.chembl_id AS chembl_id,
            p.pathway_name AS pathway_name,
            p.pathway_id AS pathway_id
        ORDER BY d.drug_name
        LIMIT $limit
        """
        return query, {"pathway_name": pathway_name, "limit": limit}

    @staticmethod
    def get_pathway_drugs_with_gene_overlap(pathway_id: str, pathway_genes: List[str], limit: int = 5000) -> tuple[str, Dict[str, Any]]:
        """
        Get drugs associated with a KEGG pathway that have gene overlap with provided genes.
        
        Args:
            pathway_id: KEGG pathway ID
            pathway_genes: List of gene symbols to check for overlap
            limit: Maximum number of drugs to return
            
        Returns:
            Tuple of (query, parameters)
        """
        query = """
        MATCH (p:KeggPathway {pathway_id: $pathway_id})
        OPTIONAL MATCH (p)-[:CONTAINS_DRUG]-(d:KeggDrug)
        WITH p, collect(DISTINCT d) as all_drugs
        UNWIND all_drugs as drug
        WITH drug, p
        WHERE drug IS NOT NULL AND drug.target_genes_updated IS NOT NULL
        OPTIONAL MATCH (drug)-[:TARGETS_KEGG_GENE]-(kg:KeggDrugGene)
        OPTIONAL MATCH (drug)-[:TARGETS]-(g:Gene)
        OPTIONAL MATCH (drug)-[:BELONGS_TO_CLASS]-(dc:DrugClass)
        WITH drug, p, 
             collect(DISTINCT kg.symbol) + collect(DISTINCT g.symbol) as drug_target_genes,
             collect(DISTINCT dc.name) as drug_classes,
             [gene IN $pathway_genes WHERE gene IN drug.target_genes_updated] as overlapping_genes
        WHERE size(overlapping_genes) > 0
        RETURN DISTINCT 
            drug.drug_name as drug_name,
            drug.kegg_id as drug_id,
            drug.mechanism_of_action as mechanism_of_action,
            drug.generic_name as generic_name,
            drug.brand_name as brand_name,
            drug.route_of_administration as route_of_administration,
            drug.fda_approved_status as fda_approved_status,
            drug.efficacy as efficacy,
            drug.target_genes_updated as target_genes_property,
            drug_target_genes as target_genes,
            drug_classes,
            drug.drugbank_id as drugbank_id,
            drug.chembl_id as chembl_id,
            overlapping_genes,
            size(overlapping_genes) as overlap_count,
            p.pathway_name as pathway_name,
            p.pathway_id as pathway_id
        ORDER BY overlap_count DESC, drug.fda_approved_status DESC, drug.drug_name
        LIMIT $limit
        """
        return query, {"pathway_id": pathway_id, "pathway_genes": pathway_genes, "limit": limit}

    @staticmethod
    def get_kegg_pathway_info(pathway_id: str) -> tuple[str, Dict[str, Any]]:
        """
        Get detailed information about a KEGG pathway.
        
        Args:
            pathway_id: KEGG pathway ID
            
        Returns:
            Tuple of (query, parameters)
        """
        query = """
        MATCH (p:KeggPathway {pathway_id: $pathway_id})
        OPTIONAL MATCH (p)-[:CONTAINS_GENE]-(g:Gene)
        OPTIONAL MATCH (p)-[:CONTAINS_KEGG_GENE]-(kg:KeggDrugGene)
        OPTIONAL MATCH (p)-[:CONTAINS_PATHWAY_GENE]-(pg:KeggPathwayGene)
        OPTIONAL MATCH (p)-[:CONTAINS_DRUG]-(d:KeggDrug)
        RETURN 
            p.pathway_id,
            p.pathway_name,
            p.description,
            p.gene_count,
            p.drug_count,
            collect(DISTINCT g.symbol) + collect(DISTINCT kg.symbol) + collect(DISTINCT pg.symbol) as genes,
            collect(DISTINCT d.drug_name) as drugs,
            size(collect(DISTINCT g)) + size(collect(DISTINCT kg)) + size(collect(DISTINCT pg)) as actual_gene_count,
            size(collect(DISTINCT d)) as actual_drug_count
        """
        return query, {"pathway_id": pathway_id}

    @staticmethod
    def search_kegg_pathways_by_genes(gene_symbols: List[str], min_gene_overlap: int = 1, limit: int = 5000) -> tuple[str, Dict[str, Any]]:
        """
        Find KEGG pathways that contain the specified genes.
        
        Args:
            gene_symbols: List of gene symbols to search for
            min_gene_overlap: Minimum number of genes that must overlap
            limit: Maximum number of pathways to return
            
        Returns:
            Tuple of (query, parameters)
        """
        query = """
        MATCH (p:KeggPathway)
        OPTIONAL MATCH (p)-[:CONTAINS_GENE]-(g:Gene)
        OPTIONAL MATCH (p)-[:CONTAINS_KEGG_GENE]-(kg:KeggDrugGene)
        OPTIONAL MATCH (p)-[:CONTAINS_PATHWAY_GENE]-(pg:KeggPathwayGene)
        WITH p, 
             collect(DISTINCT g.symbol) + collect(DISTINCT kg.symbol) + collect(DISTINCT pg.symbol) as all_genes
        WITH p, [gene IN all_genes WHERE gene IN $gene_symbols] as pathway_genes
        WHERE size(pathway_genes) >= $min_gene_overlap
        OPTIONAL MATCH (p)-[:CONTAINS_DRUG]-(d:KeggDrug)
        RETURN 
            p.pathway_id,
            p.pathway_name,
            p.description,
            pathway_genes,
            size(pathway_genes) as gene_overlap_count,
            collect(DISTINCT d.drug_name) as pathway_drugs,
            size(collect(DISTINCT d)) as drug_count
        ORDER BY gene_overlap_count DESC, drug_count DESC
        LIMIT $limit
        """
        return query, {
            "gene_symbols": gene_symbols,
            "min_gene_overlap": min_gene_overlap,
            "limit": limit
        }

    @staticmethod
    def get_drugs_targeting_pathway_genes(pathway_id: str, limit: int = 5000) -> tuple[str, Dict[str, Any]]:
        """
        Get all drugs that target genes within a specific KEGG pathway.
        
        Args:
            pathway_id: KEGG pathway ID
            limit: Maximum number of drugs to return
            
        Returns:
            Tuple of (query, parameters)
        """
        query = """
        MATCH (p:KeggPathway {pathway_id: $pathway_id})
        OPTIONAL MATCH (p)-[:CONTAINS_GENE]-(g:Gene)
        OPTIONAL MATCH (p)-[:CONTAINS_KEGG_GENE]-(kg:KeggDrugGene)
        OPTIONAL MATCH (p)-[:CONTAINS_PATHWAY_GENE]-(pg:KeggPathwayGene)
        WITH p, collect(DISTINCT g.symbol) + collect(DISTINCT kg.symbol) + collect(DISTINCT pg.symbol) as pathway_gene_symbols
        MATCH (d:KeggDrug)
        WHERE d.target_genes_updated IS NOT NULL AND 
              any(gene IN pathway_gene_symbols WHERE gene IN d.target_genes_updated)
        OPTIONAL MATCH (d)-[:TARGETS_KEGG_GENE]-(kg2:KeggDrugGene)
        OPTIONAL MATCH (d)-[:TARGETS]-(g2:Gene)
        OPTIONAL MATCH (d)-[:BELONGS_TO_CLASS]-(dc:DrugClass)
        WITH d, p, pathway_gene_symbols,
             [gene IN pathway_gene_symbols WHERE gene IN d.target_genes_updated] as targeted_pathway_genes,
             collect(DISTINCT kg2.symbol) + collect(DISTINCT g2.symbol) as drug_target_genes,
             collect(DISTINCT dc.name) as drug_classes
        RETURN DISTINCT
            d.drug_name as drug_name,
            d.kegg_id as drug_id,
            d.mechanism_of_action as mechanism_of_action,
            d.generic_name as generic_name,
            d.brand_name as brand_name,
            d.route_of_administration as route_of_administration,
            d.fda_approved_status as fda_approved_status,
            d.efficacy as efficacy,
            d.target_genes_updated as target_genes_property,
            drug_target_genes as target_genes,
            drug_classes,
            d.drugbank_id as drugbank_id,
            d.chembl_id as chembl_id,
            targeted_pathway_genes,
            size(targeted_pathway_genes) as targeted_gene_count,
            p.pathway_name as pathway_name,
            p.pathway_id as pathway_id
        ORDER BY targeted_gene_count DESC, d.fda_approved_status DESC, d.drug_name
        LIMIT $limit
        """
        return query, {"pathway_id": pathway_id, "limit": limit}

    @staticmethod
    def find_kegg_drug_pathway_connections(drug_name: str, limit: int = 5000) -> tuple[str, Dict[str, Any]]:
        """
        Find KEGG pathways connected to a specific drug.
        
        Args:
            drug_name: Name of the drug
            limit: Maximum number of pathways to return
            
        Returns:
            Tuple of (query, parameters)
        """
        query = """
        MATCH (d:KeggDrug {drug_name: $drug_name})
        OPTIONAL MATCH (d)-[:AFFECTS_PATHWAY]-(p:KeggPathway)
        OPTIONAL MATCH (p2:KeggPathway)-[:CONTAINS_DRUG]-(d)
        WITH d, collect(DISTINCT p) + collect(DISTINCT p2) as all_pathways
        UNWIND all_pathways as pathway
        WITH pathway, d
        WHERE pathway IS NOT NULL
        OPTIONAL MATCH (pathway)-[:CONTAINS_GENE|CONTAINS_KEGG_GENE|CONTAINS_PATHWAY_GENE]-(g)
        WITH pathway, d,
             [gene_symbol IN collect(DISTINCT g.symbol) WHERE gene_symbol IN d.target_genes_updated] as targeted_genes_in_pathway
        RETURN DISTINCT
            pathway.pathway_id,
            pathway.pathway_name,
            pathway.description,
            targeted_genes_in_pathway,
            size(targeted_genes_in_pathway) as targeted_gene_count
        ORDER BY targeted_gene_count DESC, pathway.pathway_name
        LIMIT $limit
        """
        return query, {"drug_name": drug_name, "limit": limit}

    @staticmethod
    def get_kegg_pathway_statistics() -> tuple[str, Dict[str, Any]]:
        """
        Get overall statistics about KEGG pathways in the database.
        
        Returns:
            Tuple of (query, parameters)
        """
        query = """
        MATCH (p:KeggPathway)
        OPTIONAL MATCH (p)-[:CONTAINS_GENE]-(g:Gene)
        OPTIONAL MATCH (p)-[:CONTAINS_KEGG_GENE]-(kg:KeggDrugGene)
        OPTIONAL MATCH (p)-[:CONTAINS_PATHWAY_GENE]-(pg:KeggPathwayGene)
        OPTIONAL MATCH (p)-[:CONTAINS_DRUG]-(d:KeggDrug)
        WITH p, 
             size(collect(DISTINCT g)) + size(collect(DISTINCT kg)) + size(collect(DISTINCT pg)) as gene_count,
             size(collect(DISTINCT d)) as drug_count
        RETURN 
            count(p) as total_pathways,
            avg(gene_count) as avg_genes_per_pathway,
            avg(drug_count) as avg_drugs_per_pathway,
            max(gene_count) as max_genes_in_pathway,
            max(drug_count) as max_drugs_in_pathway,
            sum(gene_count) as total_pathway_gene_connections,
            sum(drug_count) as total_pathway_drug_connections
        """
        return query, {}

    @staticmethod
    def custom_query(query_template: str, **kwargs) -> tuple[str, Dict[str, Any]]:
        """Execute a custom query with parameter substitution."""
        return query_template, kwargs

    @staticmethod
    def update_kegg_drug_moa(kegg_id: str, mechanism_of_action: str):
        """
        Build query to update a KeggDrug node's mechanism_of_action (and optionally drugbank_id).
        """
        query = """
        MATCH (kd:KeggDrug {kegg_id: $kegg_id})
        SET kd.mechanism_of_action = $mechanism_of_action,
            kd.updated_at = datetime()
        """

        params = {
            "kegg_id": kegg_id,
            "mechanism_of_action": mechanism_of_action,
        }
        return query, params

    @staticmethod
    def find_drugs_by_genes_and_disease(genes: List[str], disease_name: str, limit: int = 5000) -> Tuple[str, Dict[str, Any]]:
        """
        Find drugs that interact with a list of genes AND are associated with a specific disease
        via the POTENTIAL_TREATMENT relationship.

        Args:
            genes (List[str]): List of gene symbols.
            disease_name (str): Name (or partial name) of the disease to filter.
            limit (int): Max number of rows to return.

        Returns:
            Tuple[str, Dict[str, Any]]: Cypher query string and parameter dict.
        """
        query = """
            MATCH (g:Gene)-[:INTERACTS_WITH_DRUG]->(d:Drug)
            WHERE g.symbol IN $genes
            AND toLower(d.status) CONTAINS "approved"
            MATCH (d)-[:POTENTIAL_TREATMENT]->(dis:Disease)
            WHERE toLower(dis.name) CONTAINS toLower($disease_name) 
            OR toLower(dis.aliases) CONTAINS toLower($disease_name)
            RETURN DISTINCT
                d.name        AS name,
                d.action      AS action,
                d.cas_numbers AS cas_numbers,
                d.group       AS drug_group,
                d.role        AS role,
                d.sources     AS sources,
                d.status      AS status,
                g.symbol      AS target_gene,
            collect(DISTINCT dis.name) AS diseases
            ORDER BY d.name
            LIMIT $limit
            """
        params = {
            "genes": genes,
            "disease_name": disease_name,
            "limit": limit
        }
        return query, params



