"""
Medical-Aware Disease Search System

Uses medical synonyms, ontologies, and domain knowledge for better disease matching.
"""

import logging
from typing import List, Dict, Any, Set
from fuzzywuzzy import fuzz
from .connection import Neo4jConnection

logger = logging.getLogger(__name__)


class MedicalDiseaseSearch:
    """
    Medical-aware disease search using domain knowledge and synonyms.
    """

    def __init__(self, connection: Neo4jConnection):
        self.db = connection

        # Medical synonym mappings (expandable)
        self.medical_synonyms = {
            'gastric cancer': ['stomach cancer', 'gastric carcinoma', 'stomach carcinoma'],
            'stomach cancer': ['gastric cancer', 'gastric carcinoma', 'stomach carcinoma'],
            'breast cancer': ['breast carcinoma', 'mammary cancer', 'breast neoplasm'],
            'lung cancer': ['pulmonary cancer', 'lung carcinoma', 'bronchogenic carcinoma'],
            'colon cancer': ['colorectal cancer', 'bowel cancer', 'colonic carcinoma'],
            'alzheimer': ['alzheimer disease', 'alzheimers disease', 'dementia alzheimer type'],
            'diabetes': ['diabetes mellitus', 'diabetic disorder'],
            'heart disease': ['cardiac disease', 'cardiovascular disease', 'coronary disease'],
            'hypertension': ['high blood pressure', 'arterial hypertension'],
            'otitis media': ['middle ear infection', 'ear infection'],
        }

        # Common medical terms and variations
        self.medical_terms = {
            'cancer': ['carcinoma', 'neoplasm', 'tumor', 'malignancy'],
            'disease': ['disorder', 'syndrome', 'condition'],
            'infection': ['inflammatory', 'sepsis'],
        }

        # Load diseases from database
        self.diseases = self._load_diseases()

    def _load_diseases(self) -> List[Dict[str, Any]]:
        """Load diseases with proper alias handling."""
        query = """
        MATCH (d:Disease)
        RETURN d.name as disease_name, d.aliases as aliases
        """

        try:
            results = self.db.execute_query(query)

            # Process and clean aliases
            processed_diseases = []
            for result in results:
                disease_name = result['disease_name']
                aliases = result.get('aliases', [])

                # Clean aliases - handle various formats
                clean_aliases = []
                if aliases:
                    if isinstance(aliases, list):
                        clean_aliases = [
                            str(alias).strip() for alias in aliases if alias and str(alias).strip()]
                    elif isinstance(aliases, str):
                        # Handle string representation of list
                        clean_aliases = [
                            alias.strip() for alias in aliases.split(',') if alias.strip()]

                processed_diseases.append({
                    'disease_name': disease_name,
                    'aliases': clean_aliases
                })

            logger.info(f"Loaded {len(processed_diseases)} diseases")
            return processed_diseases

        except Exception as e:
            logger.error(f"Failed to load diseases: {e}")
            return []

    def _expand_query_with_synonyms(self, query: str) -> Set[str]:
        """Expand query with medical synonyms."""
        expanded = {query.lower()}

        # Add direct synonyms
        for term, synonyms in self.medical_synonyms.items():
            if term.lower() in query.lower():
                expanded.update([s.lower() for s in synonyms])

            for synonym in synonyms:
                if synonym.lower() in query.lower():
                    expanded.add(term.lower())
                    expanded.update([s.lower() for s in synonyms])

        # Add medical term variations
        for base_term, variations in self.medical_terms.items():
            if base_term.lower() in query.lower():
                for variation in variations:
                    expanded.add(query.lower().replace(
                        base_term.lower(), variation))

        return expanded

    def _calculate_medical_similarity(self, query: str, disease_name: str, aliases: List[str]) -> float:
        """Calculate similarity with medical knowledge."""
        query_lower = query.lower()
        disease_lower = disease_name.lower()

        # Exact match
        if query_lower == disease_lower:
            return 1.0

        # Check aliases
        for alias in aliases:
            if alias and query_lower == alias.lower():
                return 0.95

        # Expand query with synonyms
        expanded_queries = self._expand_query_with_synonyms(query)

        # Check expanded queries against disease name
        for expanded_query in expanded_queries:
            if expanded_query == disease_lower:
                return 0.9

            # Check against aliases
            for alias in aliases:
                if alias and expanded_query == alias.lower():
                    return 0.85

        # Fuzzy matching with medical awareness
        max_score = 0.0

        # Check main disease name
        fuzzy_score = fuzz.ratio(query_lower, disease_lower) / 100.0
        max_score = max(max_score, fuzzy_score)

        # Check aliases
        for alias in aliases:
            if alias:
                alias_score = fuzz.ratio(query_lower, alias.lower()) / 100.0
                max_score = max(max_score, alias_score)

        # Check expanded queries
        for expanded_query in expanded_queries:
            expanded_score = fuzz.ratio(expanded_query, disease_lower) / 100.0
            max_score = max(max_score, expanded_score)

            for alias in aliases:
                if alias:
                    alias_expanded_score = fuzz.ratio(
                        expanded_query, alias.lower()) / 100.0
                    max_score = max(max_score, alias_expanded_score)

        return max_score

    def search_diseases(self, query: str, top_k: int = 10, min_score: float = 0.4) -> List[Dict[str, Any]]:
        """
        Search diseases with medical knowledge.

        Args:
            query: Disease name to search for
            top_k: Number of results to return
            min_score: Minimum similarity score

        Returns:
            List of matching diseases with scores
        """
        if not self.diseases:
            return []

        results = []

        for disease_info in self.diseases:
            disease_name = disease_info['disease_name']
            aliases = disease_info['aliases']

            similarity = self._calculate_medical_similarity(
                query, disease_name, aliases)

            if similarity >= min_score:
                results.append({
                    'disease_name': disease_name,
                    'similarity_score': similarity,
                    'aliases': aliases,
                    'match_type': 'medical_aware'
                })

        # Sort by similarity score
        results.sort(key=lambda x: x['similarity_score'], reverse=True)

        return results[:top_k]


def get_medical_disease_suggestions(query: str,
                                    db_connection: Neo4jConnection,
                                    top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Get disease suggestions using medical knowledge.

    Args:
        query: Disease name query
        db_connection: Neo4j connection
        top_k: Number of suggestions to return

    Returns:
        List of disease suggestions
    """
    try:
        search_engine = MedicalDiseaseSearch(db_connection)
        return search_engine.search_diseases(query, top_k=top_k)
    except Exception as e:
        logger.error(f"Failed to get medical disease suggestions: {e}")
        return []
