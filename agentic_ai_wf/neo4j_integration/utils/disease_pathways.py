from agentic_ai_wf.neo4j_integration.query_builder import CypherQueryBuilder
from .connection_pool import get_shared_connection

def get_disease_pathways(disease_name: str) -> list[dict]:
    """Get disease pathways using shared connection pool"""
    conn = get_shared_connection()
    query, params = CypherQueryBuilder.find_pathways_by_disease(disease_name)
    results = conn.execute_query(query, params)
    # Don't close connection - keep it alive for reuse
    return results