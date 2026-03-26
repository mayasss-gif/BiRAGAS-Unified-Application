# agentic_ai_wf/neo4j_integration/genecards_scorer.py

from agentic_ai_wf.neo4j_integration.query_builder import CypherQueryBuilder

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Use shared connection pool
from .connection_pool import get_shared_connection as _get_connection


def genecards_scorer(disease_name: str) -> dict[str, list[Any]]:
    """
    Fetch GeneCards relevance scores for a given disease from Neo4j database.
    Uses a shared connection pool to avoid connection churn.

    Args:
        disease_name (str): Disease name to search for.

    Returns:
        dict[str, list[Any]]: Dictionary mapping gene symbol to relevance score.
    """

    logger.info(f"Fetching GeneCards scores for {disease_name}")

    gene_score_dict = {}

    try:
        # Use shared connection pool (reuses connection across requests)
        conn = _get_connection()

        query, params = CypherQueryBuilder.find_genes_by_disease_with_scores(
            disease_name)

        results = conn.execute_query(query, params)

        gene_score_dict = {"gene_symbol": [record.get("gene_symbol") for record in results], "gene_score": [record.get("gene_score") for record in results], "disorder_score": [record.get("disorder_score") for record in results], "disorder_type": [record.get("disorder_type") for record in results]}

        # If no exact match found, try partial matching
        if not gene_score_dict.get("gene_symbol"):
            query = """
            MATCH (d:Disease)-[r:ASSOCIATED_WITH|INFERRED_ASSOCIATION]-(g:Gene)
            WHERE (
                toLower(d.name) CONTAINS toLower($disease_name) OR 
                toLower($disease_name) CONTAINS toLower(d.name) OR 
                (d.aliases IS NOT NULL AND any(alias IN d.aliases WHERE toLower(alias) CONTAINS toLower($disease_name)))
            ) 
            AND r.gene_score IS NOT NULL
            RETURN 
                g.symbol AS gene_symbol, 
                r.gene_score AS gene_score, 
                r.disorder_score AS disorder_score, 
                r.disorder_type AS disorder_type
            ORDER BY r.gene_score DESC
            LIMIT 100
            """

            results = conn.execute_query(query, {"disease_name": disease_name})

            gene_score_dict = {"gene_symbol": [record.get("gene_symbol") for record in results], "gene_score": [record.get("gene_score") for record in results], "disorder_score": [record.get("disorder_score") for record in results], "disorder_type": [record.get("disorder_type") for record in results]}

        # Don't close connection - keep it alive for reuse

    except Exception as e:
        logger.error(f"Error fetching GeneCards scores: {e}")
        # Same keys as success path so API clients never see a bare {} (avoids KeyError on gene_symbol).
        gene_score_dict = {
            "gene_symbol": [],
            "gene_score": [],
            "disorder_score": [],
            "disorder_type": [],
        }

    logger.info(
        f"Fetched {len(gene_score_dict.get('gene_symbol', []))} GeneCards scores for {disease_name}")

    return gene_score_dict


if __name__ == "__main__":
    gene_score_dict = genecards_scorer("Gastric Cancer")
    print(gene_score_dict)
