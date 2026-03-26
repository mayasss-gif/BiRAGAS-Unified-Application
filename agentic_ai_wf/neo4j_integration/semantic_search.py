"""
Semantic Search for Disease Names using Vector Embeddings

Provides fuzzy matching and semantic similarity search for disease names
to help users find the right disease even with typos or variations.
"""

import json
import pickle
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import fuzz, process
from .connection import Neo4jConnection

logger = logging.getLogger(__name__)


class DiseaseSemanticSearch:
    """
    Semantic search for disease names using embeddings and fuzzy matching.
    """

    def __init__(self,
                 connection: Neo4jConnection,
                 model_name: str = "all-MiniLM-L6-v2",
                 embeddings_cache_path: str = "disease_embeddings.pkl"):
        """
        Initialize semantic search system.

        Args:
            connection: Neo4j database connection
            model_name: Sentence transformer model name
            embeddings_cache_path: Path to cache embeddings
        """
        self.db = connection
        self.model_name = model_name
        self.embeddings_cache_path = embeddings_cache_path

        # Load sentence transformer model
        try:
            self.model = SentenceTransformer(model_name)
            logger.info(f"Loaded sentence transformer model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to load sentence transformer: {e}")
            raise

        # Disease data and embeddings
        self.diseases = []
        self.disease_embeddings = None
        self.disease_aliases = {}

        # Load or create embeddings
        self._load_or_create_embeddings()

    def _get_all_diseases_with_aliases(self) -> List[Dict[str, Any]]:
        """Get all diseases with their aliases from Neo4j."""
        query = """
        MATCH (d:Disease)
        RETURN d.name as disease_name, d.aliases as aliases
        """

        try:
            results = self.db.execute_query(query)
            logger.info(f"Retrieved {len(results)} diseases from database")
            return results
        except Exception as e:
            logger.error(f"Failed to retrieve diseases: {e}")
            return []

    def _create_disease_text_variants(self, disease_name: str, aliases: Optional[List[str]] = None) -> List[str]:
        """Create text variants for a disease including aliases."""
        variants = [disease_name]

        if aliases:
            variants.extend(aliases)

        # Add common variations
        variants.extend([
            disease_name.lower(),
            disease_name.upper(),
            disease_name.replace(" ", ""),
            disease_name.replace("-", " "),
            disease_name.replace("_", " ")
        ])

        # Remove duplicates while preserving order
        seen = set()
        unique_variants = []
        for variant in variants:
            if variant and variant not in seen:
                seen.add(variant)
                unique_variants.append(variant)

        return unique_variants

    def _load_or_create_embeddings(self):
        """Load existing embeddings or create new ones."""
        cache_path = Path(self.embeddings_cache_path)

        if cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    cached_data = pickle.load(f)
                    self.diseases = cached_data['diseases']
                    self.disease_embeddings = cached_data['embeddings']
                    self.disease_aliases = cached_data['aliases']
                logger.info(
                    f"Loaded {len(self.diseases)} disease embeddings from cache")
                return
            except Exception as e:
                logger.warning(f"Failed to load cached embeddings: {e}")

        # Create new embeddings
        self._create_embeddings()

    def _create_embeddings(self):
        """Create embeddings for all diseases."""
        logger.info("Creating disease embeddings...")

        # Get diseases from database
        disease_data = self._get_all_diseases_with_aliases()

        if not disease_data:
            logger.warning("No diseases found in database")
            return

        # Prepare disease texts and create embeddings
        disease_texts = []
        self.diseases = []
        self.disease_aliases = {}

        for disease_info in disease_data:
            disease_name = disease_info['disease_name']
            aliases = disease_info.get('aliases', [])

            # Store aliases mapping
            self.disease_aliases[disease_name] = aliases or []

            # Create text variants for embedding
            text_variants = self._create_disease_text_variants(
                disease_name, aliases)

            # Use the primary name for embedding (could be enhanced to use all variants)
            embedding_text = f"{disease_name} {' '.join(aliases or [])}"
            disease_texts.append(embedding_text)
            self.diseases.append(disease_name)

        # Create embeddings
        try:
            self.disease_embeddings = self.model.encode(disease_texts)
            logger.info(
                f"Created embeddings for {len(self.diseases)} diseases")

            # Cache embeddings
            self._cache_embeddings()

        except Exception as e:
            logger.error(f"Failed to create embeddings: {e}")
            raise

    def _cache_embeddings(self):
        """Cache embeddings to disk."""
        try:
            cache_data = {
                'diseases': self.diseases,
                'embeddings': self.disease_embeddings,
                'aliases': self.disease_aliases
            }

            with open(self.embeddings_cache_path, 'wb') as f:
                pickle.dump(cache_data, f)

            logger.info(f"Cached embeddings to {self.embeddings_cache_path}")

        except Exception as e:
            logger.warning(f"Failed to cache embeddings: {e}")

    def find_similar_diseases(self,
                              query: str,
                              top_k: int = 10,
                              similarity_threshold: float = 0.3) -> List[Dict[str, Any]]:
        """
        Find diseases similar to the query using semantic search.

        Args:
            query: Disease name to search for
            top_k: Number of results to return
            similarity_threshold: Minimum similarity score

        Returns:
            List of similar diseases with scores
        """
        if not self.diseases or self.disease_embeddings is None:
            logger.warning("No disease embeddings available")
            return []

        try:
            # Create embedding for query
            query_embedding = self.model.encode([query])

            # Calculate cosine similarity
            similarities = cosine_similarity(
                query_embedding, self.disease_embeddings)[0]

            # Get top similar diseases
            top_indices = np.argsort(similarities)[::-1][:top_k]

            results = []
            for idx in top_indices:
                similarity_score = similarities[idx]

                if similarity_score >= similarity_threshold:
                    disease_name = self.diseases[idx]
                    aliases = self.disease_aliases.get(disease_name, [])

                    results.append({
                        'disease_name': disease_name,
                        'similarity_score': float(similarity_score),
                        'aliases': aliases,
                        'match_type': 'semantic'
                    })

            logger.info(f"Found {len(results)} similar diseases for '{query}'")
            return results

        except Exception as e:
            logger.error(f"Error in semantic search: {e}")
            return []

    def find_diseases_fuzzy(self,
                            query: str,
                            top_k: int = 10,
                            score_threshold: int = 60) -> List[Dict[str, Any]]:
        """
        Find diseases using fuzzy string matching.

        Args:
            query: Disease name to search for
            top_k: Number of results to return
            score_threshold: Minimum fuzzy score (0-100)

        Returns:
            List of matching diseases with scores
        """
        if not self.diseases:
            return []

        try:
            # Create search list including aliases
            search_list = []
            disease_mapping = {}

            for disease in self.diseases:
                search_list.append(disease)
                disease_mapping[disease] = disease

                # Add aliases
                for alias in self.disease_aliases.get(disease, []):
                    search_list.append(alias)
                    disease_mapping[alias] = disease

            # Perform fuzzy matching
            matches = process.extract(query, search_list, limit=top_k * 2)

            # Process results
            results = []
            seen_diseases = set()

            for match_result in matches:
                match_text, score = match_result[0], match_result[1]
                if score >= score_threshold:
                    original_disease = disease_mapping[match_text]

                    if original_disease not in seen_diseases:
                        seen_diseases.add(original_disease)

                        results.append({
                            'disease_name': original_disease,
                            'similarity_score': score / 100.0,  # Normalize to 0-1
                            'matched_text': match_text,
                            'aliases': self.disease_aliases.get(original_disease, []),
                            'match_type': 'fuzzy'
                        })

                        if len(results) >= top_k:
                            break

            logger.info(f"Found {len(results)} fuzzy matches for '{query}'")
            return results

        except Exception as e:
            logger.error(f"Error in fuzzy search: {e}")
            return []

    def search_diseases(self,
                        query: str,
                        top_k: int = 10,
                        use_semantic: bool = True,
                        use_fuzzy: bool = True) -> List[Dict[str, Any]]:
        """
        Combined search using both semantic and fuzzy matching.

        Args:
            query: Disease name to search for
            top_k: Number of results to return
            use_semantic: Whether to use semantic search
            use_fuzzy: Whether to use fuzzy matching

        Returns:
            Ranked list of matching diseases
        """
        all_results = []

        # Semantic search
        if use_semantic:
            semantic_results = self.find_similar_diseases(query, top_k=top_k)
            all_results.extend(semantic_results)

        # Fuzzy search
        if use_fuzzy:
            fuzzy_results = self.find_diseases_fuzzy(query, top_k=top_k)
            all_results.extend(fuzzy_results)

        # Combine and deduplicate results
        disease_scores = {}

        for result in all_results:
            disease_name = result['disease_name']
            score = result['similarity_score']
            match_type = result['match_type']

            if disease_name not in disease_scores:
                disease_scores[disease_name] = {
                    'disease_name': disease_name,
                    'best_score': score,
                    'match_types': [match_type],
                    'aliases': result.get('aliases', [])
                }
            else:
                # Keep best score and combine match types
                if score > disease_scores[disease_name]['best_score']:
                    disease_scores[disease_name]['best_score'] = score

                if match_type not in disease_scores[disease_name]['match_types']:
                    disease_scores[disease_name]['match_types'].append(
                        match_type)

        # Sort by score and return top results
        final_results = sorted(
            disease_scores.values(),
            key=lambda x: x['best_score'],
            reverse=True
        )[:top_k]

        return final_results

    def refresh_embeddings(self):
        """Refresh embeddings from database."""
        logger.info("Refreshing disease embeddings...")

        # Remove cached embeddings
        cache_path = Path(self.embeddings_cache_path)
        if cache_path.exists():
            cache_path.unlink()

        # Recreate embeddings
        self._create_embeddings()

        logger.info("Disease embeddings refreshed")


def get_disease_suggestions(query: str,
                            db_connection: Neo4jConnection,
                            top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Convenience function to get disease suggestions.

    Args:
        query: Disease name query
        db_connection: Neo4j connection
        top_k: Number of suggestions to return

    Returns:
        List of disease suggestions
    """
    try:
        search_engine = DiseaseSemanticSearch(db_connection)
        return search_engine.search_diseases(query, top_k=top_k)
    except Exception as e:
        logger.error(f"Failed to get disease suggestions: {e}")
        return []
