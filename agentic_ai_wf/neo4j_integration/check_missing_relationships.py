#!/usr/bin/env python3
"""
Check Missing Relationships

Quick verification of genes missing diseases, drugs, or pathways relationships.
"""

import logging
from .connection import Neo4jConnection

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def check_missing_relationships():
    """Check current state of missing relationships."""
    try:
        db = Neo4jConnection()
        logger.info("✅ Connected to Neo4j successfully!")

        # Total gene count
        total_genes_query = "MATCH (g:Gene) RETURN COUNT(g) as count"
        total_genes = db.execute_query(total_genes_query)[0]['count']

        # Genes with diseases
        genes_with_diseases_query = """
        MATCH (g:Gene)-[:ASSOCIATED_WITH|INFERRED_ASSOCIATION]->(:Disease)
        RETURN COUNT(DISTINCT g) as count
        """
        genes_with_diseases = db.execute_query(
            genes_with_diseases_query)[0]['count']

        # Genes with drugs
        genes_with_drugs_query = """
        MATCH (g:Gene)-[:INTERACTS_WITH_DRUG]->(:Drug)
        RETURN COUNT(DISTINCT g) as count
        """
        genes_with_drugs = db.execute_query(genes_with_drugs_query)[0]['count']

        # Genes with pathways
        genes_with_pathways_query = """
        MATCH (g:Gene)-[:BELONGS_TO_PATHWAY|BELONGS_TO_SUPER_PATHWAY]->(:Pathway)
        RETURN COUNT(DISTINCT g) as count
        """
        genes_with_pathways = db.execute_query(
            genes_with_pathways_query)[0]['count']

        # Entity counts
        disease_count = db.execute_query(
            "MATCH (d:Disease) RETURN COUNT(d) as count")[0]['count']
        drug_count = db.execute_query(
            "MATCH (d:Drug) RETURN COUNT(d) as count")[0]['count']
        pathway_count = db.execute_query(
            "MATCH (p:Pathway) RETURN COUNT(p) as count")[0]['count']

        logger.info("=" * 60)
        logger.info("RELATIONSHIP STATUS ANALYSIS")
        logger.info("=" * 60)
        logger.info(f"Total genes: {total_genes:,}")
        logger.info(
            f"Genes with diseases: {genes_with_diseases:,} ({(genes_with_diseases/total_genes*100):.1f}%)")
        logger.info(
            f"Genes with drugs: {genes_with_drugs:,} ({(genes_with_drugs/total_genes*100):.1f}%)")
        logger.info(
            f"Genes with pathways: {genes_with_pathways:,} ({(genes_with_pathways/total_genes*100):.1f}%)")

        logger.info("\nMissing Relationships:")
        logger.info(
            f"Genes missing diseases: {total_genes - genes_with_diseases:,}")
        logger.info(f"Genes missing drugs: {total_genes - genes_with_drugs:,}")
        logger.info(
            f"Genes missing pathways: {total_genes - genes_with_pathways:,}")

        logger.info("\nEntity Counts:")
        logger.info(f"Total diseases: {disease_count:,}")
        logger.info(f"Total drugs: {drug_count:,}")
        logger.info(f"Total pathways: {pathway_count:,}")

        # Sample genes missing relationships
        logger.info("\nSample genes missing diseases (first 5):")
        missing_diseases_sample = db.execute_query("""
        MATCH (g:Gene)
        WHERE NOT (g)-[:ASSOCIATED_WITH|INFERRED_ASSOCIATION]->(:Disease)
        RETURN g.symbol as symbol
        LIMIT 5
        """)
        for record in missing_diseases_sample:
            logger.info(f"  - {record['symbol']}")

        logger.info("\nSample genes missing drugs (first 5):")
        missing_drugs_sample = db.execute_query("""
        MATCH (g:Gene)
        WHERE NOT (g)-[:INTERACTS_WITH_DRUG]->(:Drug)
        RETURN g.symbol as symbol
        LIMIT 5
        """)
        for record in missing_drugs_sample:
            logger.info(f"  - {record['symbol']}")

        logger.info("\nSample genes missing pathways (first 5):")
        missing_pathways_sample = db.execute_query("""
        MATCH (g:Gene)
        WHERE NOT (g)-[:BELONGS_TO_PATHWAY|BELONGS_TO_SUPER_PATHWAY]->(:Pathway)
        RETURN g.symbol as symbol
        LIMIT 5
        """)
        for record in missing_pathways_sample:
            logger.info(f"  - {record['symbol']}")

        db.close()

    except Exception as e:
        logger.error(f"Error: {e}")


if __name__ == "__main__":
    check_missing_relationships()
