#!/usr/bin/env python3
"""
Quick check for genes with NULL aliases
"""

import logging
from .connection import Neo4jConnection

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def check_null_aliases():
    """Check current state of genes with NULL aliases."""
    try:
        db = Neo4jConnection()
        logger.info("✅ Connected to Neo4j successfully!")

        # Count genes with NULL aliases
        query = """
        MATCH (g:Gene)
        WHERE g.aliases IS NULL
        RETURN COUNT(g) as count
        """

        result = db.execute_query(query)
        null_count = result[0]['count'] if result else 0

        # Get total gene count
        total_query = "MATCH (g:Gene) RETURN COUNT(g) as count"
        total_result = db.execute_query(total_query)
        total_count = total_result[0]['count'] if total_result else 0

        logger.info(f"Genes with NULL aliases: {null_count:,}")
        logger.info(f"Total genes: {total_count:,}")
        logger.info(
            f"Percentage with NULL aliases: {(null_count/total_count*100):.2f}%")

        # Sample some genes with NULL aliases
        sample_query = """
        MATCH (g:Gene)
        WHERE g.aliases IS NULL
        RETURN g.symbol as symbol
        ORDER BY g.symbol
        LIMIT 10
        """

        sample_result = db.execute_query(sample_query)
        logger.info("Sample genes with NULL aliases:")
        for record in sample_result:
            logger.info(f"  - {record['symbol']}")

        db.close()

    except Exception as e:
        logger.error(f"Error: {e}")


if __name__ == "__main__":
    check_null_aliases()
