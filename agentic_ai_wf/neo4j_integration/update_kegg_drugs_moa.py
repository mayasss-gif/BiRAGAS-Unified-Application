import logging
import pandas as pd
from typing import Dict, Any, List
from .query_builder import CypherQueryBuilder
from .connection import Neo4jConnection

logger = logging.getLogger(__name__)

def update_drug_mechanism(neo4j_connection: Neo4jConnection, kegg_id: str, mechanism_of_action: str) -> Dict[str, Any]:
    """
    Update mechanism_of_action (and optionally drugbank_id) for a KeggDrug node.
    """
    logger.info(f"Updating KeggDrug {kegg_id} with new MOA: {mechanism_of_action}")
    
    query, params = CypherQueryBuilder.update_kegg_drug_moa(kegg_id, mechanism_of_action)
    result = neo4j_connection.execute_query(query, params)
    
    logger.info(f"Updated KeggDrug {kegg_id}")
    return {"kegg_id": kegg_id, "mechanism_of_action": mechanism_of_action}


def update_drugs_from_csv(neo4j_connection: Neo4jConnection, csv_path: str) -> List[Dict[str, Any]]:
    """
    Bulk update KeggDrug nodes from a CSV file.
    
    CSV must contain: kegg_id, mechanism_of_action
    """
    logger.info(f"Loading updates from CSV: {csv_path}")
    df = pd.read_csv(csv_path)

    updates = []
    for _, row in df.iterrows():
        result = update_drug_mechanism(
            neo4j_connection=neo4j_connection,
            kegg_id=row["kegg_id"],
            mechanism_of_action=row["mechanism_of_action"],
        )
        updates.append(result)

    logger.info(f"Applied {len(updates)} updates from {csv_path}")
    return updates

if __name__ == "__main__":
    neo4j_connection = Neo4jConnection()
    update_drugs_from_csv(neo4j_connection, "agentic_ai_wf/neo4j_integration/data/kegg_drugs_moa.csv")