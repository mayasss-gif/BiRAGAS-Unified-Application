"""
Knowledge Graph-based Disease Search System

Leverages Neo4j relationships and medical ontologies for intelligent semantic search
without external dependencies like Cognee.
"""

import logging
import json
from typing import List, Dict, Any, Set, Tuple, Optional
from collections import defaultdict
import re
from fuzzywuzzy import fuzz
from .connection import Neo4jConnection

logger = logging.getLogger(__name__)


class KnowledgeGraphSearch:
    """
    Advanced knowledge graph search using Neo4j relationships and medical reasoning.
    """

    def __init__(self, connection: Neo4jConnection):
        self.db = connection

        # Medical ontology patterns
        self.medical_patterns = {
            # Cancer patterns
            'cancer_patterns': [
                r'(.+)\s+cancer',
                r'(.+)\s+carcinoma',
                r'(.+)\s+neoplasm',
                r'(.+)\s+tumor',
                r'(.+)\s+malignancy'
            ],

            # Anatomical mappings
            'anatomical_synonyms': {
                'stomach': ['gastric', 'gastro'],
                'lung': ['pulmonary', 'respiratory', 'bronchial'],
                'breast': ['mammary'],
                'colon': ['colorectal', 'bowel', 'intestinal'],
                'kidney': ['renal'],
                'liver': ['hepatic'],
                'heart': ['cardiac', 'coronary'],
                'brain': ['cerebral', 'neurological'],
                'bone': ['skeletal', 'osseous'],
                'blood': ['hematologic', 'hematopoietic'],
                'skin': ['cutaneous', 'dermal'],
                'eye': ['ocular', 'ophthalmic'],
                'ear': ['otic', 'auditory']
            },

            # Disease type mappings
            'disease_types': {
                'cancer': ['carcinoma', 'sarcoma', 'lymphoma', 'leukemia', 'neoplasm', 'tumor', 'malignancy'],
                'infection': ['bacterial', 'viral', 'fungal', 'parasitic', 'sepsis', 'inflammatory'],
                'genetic': ['hereditary', 'familial', 'congenital', 'inherited'],
                'autoimmune': ['inflammatory', 'immune', 'rheumatic'],
                'metabolic': ['diabetes', 'thyroid', 'hormonal', 'endocrine']
            }
        }

        # Load disease relationships from graph
        self.disease_relationships = self._build_disease_relationships()

    def _build_disease_relationships(self) -> Dict[str, Dict[str, Any]]:
        """Build a comprehensive disease relationship map from the Neo4j graph."""

        relationships = defaultdict(lambda: {
            'aliases': [],
            'related_diseases': [],
            'genes': [],
            'pathways': [],
            'anatomical_location': None,
            'disease_type': None
        })

        try:
            # Get disease relationships
            queries = [
                # Basic disease info with aliases
                """
                MATCH (d:Disease)
                RETURN d.name as disease_name, d.aliases as aliases
                """,

                # Disease-gene relationships
                """
                MATCH (d:Disease)-[r:ASSOCIATED_WITH|INFERRED_ASSOCIATION]-(g:Gene)
                RETURN d.name as disease_name, collect(DISTINCT g.symbol) as genes
                """,

                # Disease-pathway relationships (if exists)
                """
                MATCH (d:Disease)-[:RELATED_TO]-(p:Pathway)
                RETURN d.name as disease_name, collect(DISTINCT p.name) as pathways
                """,

                # Gene-pathway relationships for disease context
                """
                MATCH (d:Disease)-[:ASSOCIATED_WITH]-(g:Gene)-[:BELONGS_TO]-(p:Pathway)
                RETURN d.name as disease_name, collect(DISTINCT p.name) as related_pathways
                """
            ]

            # Execute basic disease info query
            basic_info = self.db.execute_query(queries[0])
            for result in basic_info:
                disease_name = result['disease_name']
                aliases = result.get('aliases', [])

                # Clean aliases
                clean_aliases = []
                if aliases:
                    if isinstance(aliases, list):
                        clean_aliases = [
                            str(alias).strip() for alias in aliases if alias and str(alias).strip()]
                    elif isinstance(aliases, str):
                        clean_aliases = [
                            alias.strip() for alias in aliases.split(',') if alias.strip()]

                relationships[disease_name]['aliases'] = clean_aliases

                # Extract anatomical location and disease type
                self._extract_medical_metadata(
                    disease_name, relationships[disease_name])

            # Execute gene relationships query
            try:
                gene_info = self.db.execute_query(queries[1])
                for result in gene_info:
                    disease_name = result['disease_name']
                    genes = result.get('genes', [])
                    if genes:
                        relationships[disease_name]['genes'] = genes
            except Exception as e:
                logger.warning(f"Could not load gene relationships: {e}")

            # Execute pathway relationships queries
            for pathway_query in queries[2:]:
                try:
                    pathway_info = self.db.execute_query(pathway_query)
                    for result in pathway_info:
                        disease_name = result['disease_name']
                        pathways = result.get('pathways', []) or result.get(
                            'related_pathways', [])
                        if pathways:
                            relationships[disease_name]['pathways'] = pathways
                except Exception as e:
                    logger.warning(
                        f"Could not load pathway relationships: {e}")

            logger.info(
                f"Built relationships for {len(relationships)} diseases")
            return dict(relationships)

        except Exception as e:
            logger.error(f"Failed to build disease relationships: {e}")
            return {}

    def _extract_medical_metadata(self, disease_name: str, disease_info: Dict[str, Any]):
        """Extract anatomical location and disease type from disease name and aliases."""

        disease_lower = disease_name.lower()
        all_text = [disease_lower] + [alias.lower()
                                      for alias in disease_info['aliases']]
        combined_text = ' '.join(all_text)

        # Extract anatomical location
        for anatomy, synonyms in self.medical_patterns['anatomical_synonyms'].items():
            if anatomy in combined_text or any(syn in combined_text for syn in synonyms):
                disease_info['anatomical_location'] = anatomy
                break

        # Extract disease type
        for disease_type, keywords in self.medical_patterns['disease_types'].items():
            if any(keyword in combined_text for keyword in keywords):
                disease_info['disease_type'] = disease_type
                break

    def _expand_query_semantically(self, query: str) -> Set[str]:
        """Expand query using medical knowledge and patterns."""

        expanded = {query.lower()}
        query_lower = query.lower()

        # Extract anatomical references
        for anatomy, synonyms in self.medical_patterns['anatomical_synonyms'].items():
            if anatomy in query_lower:
                for synonym in synonyms:
                    expanded.add(query_lower.replace(anatomy, synonym))

            for synonym in synonyms:
                if synonym in query_lower:
                    expanded.add(query_lower.replace(synonym, anatomy))
                    for other_synonym in synonyms:
                        if other_synonym != synonym:
                            expanded.add(query_lower.replace(
                                synonym, other_synonym))

        # Extract disease type patterns
        for disease_type, variations in self.medical_patterns['disease_types'].items():
            for variation in variations:
                if variation in query_lower:
                    for other_variation in variations:
                        if other_variation != variation:
                            expanded.add(query_lower.replace(
                                variation, other_variation))

        # Handle cancer patterns specifically
        for pattern in self.medical_patterns['cancer_patterns']:
            match = re.search(pattern, query_lower)
            if match:
                organ = match.group(1)
                # Generate variations
                cancer_terms = ['cancer', 'carcinoma', 'neoplasm', 'tumor']
                for term in cancer_terms:
                    expanded.add(f"{organ} {term}")

        return expanded

    def _calculate_graph_similarity(self, query: str, disease_name: str, disease_info: Dict[str, Any]) -> float:
        """Calculate similarity using graph relationships and medical knowledge."""

        query_lower = query.lower()
        disease_lower = disease_name.lower()

        # Direct match
        if query_lower == disease_lower:
            return 1.0

        # Alias match
        for alias in disease_info['aliases']:
            if alias and query_lower == alias.lower():
                return 0.95

        # Semantic expansion match
        expanded_queries = self._expand_query_semantically(query)

        # Check expanded queries
        for expanded_query in expanded_queries:
            if expanded_query == disease_lower:
                return 0.9

            for alias in disease_info['aliases']:
                if alias and expanded_query == alias.lower():
                    return 0.85

        # Graph-based similarity using shared genes/pathways
        graph_score = self._calculate_contextual_similarity(
            query, disease_name, disease_info)

        # Fuzzy matching with medical patterns
        fuzzy_score = self._calculate_medical_fuzzy_score(
            query, disease_name, disease_info)

        # Combine scores with weights
        final_score = max(graph_score, fuzzy_score)

        return final_score

    def _calculate_contextual_similarity(self, query: str, disease_name: str, disease_info: Dict[str, Any]) -> float:
        """Calculate similarity based on graph context (genes, pathways, etc.)."""

        # This is a placeholder for more sophisticated graph-based similarity
        # You could expand this to:
        # 1. Find diseases with similar gene sets
        # 2. Use pathway overlap
        # 3. Use anatomical proximity

        base_score = 0.0

        # Anatomical location bonus
        if disease_info['anatomical_location']:
            for anatomy, synonyms in self.medical_patterns['anatomical_synonyms'].items():
                if anatomy == disease_info['anatomical_location']:
                    if anatomy in query.lower() or any(syn in query.lower() for syn in synonyms):
                        base_score += 0.2
                        break

        # Disease type bonus
        if disease_info['disease_type']:
            disease_keywords = self.medical_patterns['disease_types'].get(
                disease_info['disease_type'], [])
            if any(keyword in query.lower() for keyword in disease_keywords):
                base_score += 0.2

        # Cap at 0.8 to leave room for exact matches
        return min(base_score, 0.8)

    def _calculate_medical_fuzzy_score(self, query: str, disease_name: str, disease_info: Dict[str, Any]) -> float:
        """Calculate fuzzy similarity with medical awareness."""

        max_score = 0.0

        # Score against disease name
        disease_score = fuzz.ratio(query.lower(), disease_name.lower()) / 100.0
        max_score = max(max_score, disease_score)

        # Score against aliases
        for alias in disease_info['aliases']:
            if alias:
                alias_score = fuzz.ratio(query.lower(), alias.lower()) / 100.0
                max_score = max(max_score, alias_score)

        # Score against expanded queries
        expanded_queries = self._expand_query_semantically(query)
        for expanded_query in expanded_queries:
            expanded_score = fuzz.ratio(
                expanded_query, disease_name.lower()) / 100.0
            max_score = max(max_score, expanded_score)

            for alias in disease_info['aliases']:
                if alias:
                    alias_expanded_score = fuzz.ratio(
                        expanded_query, alias.lower()) / 100.0
                    max_score = max(max_score, alias_expanded_score)

        return max_score

    def search_diseases(self, query: str, top_k: int = 10, min_score: float = 0.4) -> List[Dict[str, Any]]:
        """
        Search diseases using knowledge graph and medical reasoning.

        Args:
            query: Disease name to search for
            top_k: Number of results to return
            min_score: Minimum similarity score

        Returns:
            List of matching diseases with enhanced metadata
        """
        if not self.disease_relationships:
            logger.warning("No disease relationships loaded")
            return []

        results = []

        for disease_name, disease_info in self.disease_relationships.items():
            similarity = self._calculate_graph_similarity(
                query, disease_name, disease_info)

            if similarity >= min_score:
                results.append({
                    'disease_name': disease_name,
                    'similarity_score': similarity,
                    'aliases': disease_info['aliases'],
                    'anatomical_location': disease_info['anatomical_location'],
                    'disease_type': disease_info['disease_type'],
                    'gene_count': len(disease_info['genes']),
                    'pathway_count': len(disease_info['pathways']),
                    'match_type': 'knowledge_graph'
                })

        # Sort by similarity score
        results.sort(key=lambda x: x['similarity_score'], reverse=True)

        return results[:top_k]

    def get_disease_context(self, disease_name: str) -> Dict[str, Any]:
        """Get comprehensive context for a disease from the knowledge graph."""

        if disease_name not in self.disease_relationships:
            return {}

        disease_info = self.disease_relationships[disease_name]

        # Get top genes with scores
        gene_query = """
        MATCH (d:Disease {name: $disease_name})-[r:ASSOCIATED_WITH|INFERRED_ASSOCIATION]-(g:Gene)
        RETURN g.symbol as gene_symbol, r.gene_score as gene_score
        ORDER BY r.gene_score DESC
        LIMIT 10
        """

        try:
            gene_results = self.db.execute_query(
                gene_query, {"disease_name": disease_name})
            top_genes = [
                {"gene": result['gene_symbol'],
                    "score": result.get('gene_score', 0)}
                for result in gene_results
            ]
        except Exception as e:
            logger.warning(f"Could not get gene context: {e}")
            top_genes = []

        return {
            'disease_name': disease_name,
            'aliases': disease_info['aliases'],
            'anatomical_location': disease_info['anatomical_location'],
            'disease_type': disease_info['disease_type'],
            'top_genes': top_genes,
            'pathways': disease_info['pathways'],
            'gene_count': len(disease_info['genes']),
            'pathway_count': len(disease_info['pathways'])
        }


def get_knowledge_graph_suggestions(query: str,
                                    db_connection: Neo4jConnection,
                                    top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Get disease suggestions using knowledge graph reasoning.

    Args:
        query: Disease name query
        db_connection: Neo4j connection
        top_k: Number of suggestions to return

    Returns:
        List of disease suggestions with rich context
    """
    try:
        search_engine = KnowledgeGraphSearch(db_connection)
        return search_engine.search_diseases(query, top_k=top_k)
    except Exception as e:
        logger.error(f"Failed to get knowledge graph suggestions: {e}")
        return []

# simple solution to get diseases list from neo4j
def get_diseases_list(db_connection: Neo4jConnection, query: str = "lupus", top_k: int = 100) -> List[Dict[str, Any]]:
    """Get diseases list from neo4j."""
    try:
        # Use the specific query for disease search
        cypher_query = f"""
        MATCH (n:Disease)
        WHERE toLower(n.name) CONTAINS '{query.lower()}' 
        OR ANY(alias IN n.aliases WHERE toLower(alias) CONTAINS '{query.lower()}')
        RETURN n.name
        ORDER BY n.name
        LIMIT {top_k}
        """
        
        if not db_connection.driver:
            logger.error("Neo4j driver not initialized")
            return []
        
        with db_connection.driver.session(database=db_connection.database) as session:
            result = session.run(cypher_query, query_param=query.lower(), top_k=top_k)  # type: ignore
            diseases = [record["n.name"] for record in result]
            return diseases
            
    except Exception as e:
        logger.error(f"Failed to get diseases list: {e}")
        return []

if __name__ == "__main__":
    db_connection = Neo4jConnection(
        # host="localhost",
        # port=7687,
        # user="neo4j",
        # password="neo4j"
    )
    
    diseases = get_diseases_list(db_connection, query='lupus')
    print(diseases)