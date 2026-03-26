"""
Neo4j KEGG Drug ↔ Reactome Pathway Linker
Reads a CSV (pathway_id, pathway_name, drug_ids) and creates relationships between existing nodes:
(Pathway {pathway_id}) -[:CONTAINS_DRUG]-> (KeggDrug {kegg_id})

Assumptions:
- All nodes already exist; we do NOT create nodes here.
- drug_ids are KEGG Drug IDs like "D02596".
- pathway_id uses Reactome IDs like "R-HSA-1059683".
"""

import pandas as pd
import logging
from typing import List, Dict, Any
from pathlib import Path
from connection import Neo4jConnection  # same interface as in your example

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class KEGGDrugPathwayLinker:
    """Creates (Pathway)-[:CONTAINS_DRUG]->(KeggDrug) relationships for existing nodes."""

    def __init__(self, neo4j_connection: Neo4jConnection):
        self.neo4j = neo4j_connection

    def create_supporting_indexes(self):
        """Optional: add helpful indexes/constraints (no node creation)."""
        queries = [
            "CREATE CONSTRAINT IF NOT EXISTS FOR (p:Pathway) REQUIRE p.pathway_id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (d:KeggDrug) REQUIRE d.kegg_id IS UNIQUE",
            "CREATE INDEX IF NOT EXISTS FOR (p:Pathway) ON (p.name)"
        ]
        for q in queries:
            try:
                self.neo4j.execute_query(q)
                logger.info(f"✅ Created: {q}")
            except Exception as e:
                logger.warning(f"⚠️ Index/constraint creation failed: {e}")

    @staticmethod
    def _parse_drug_ids(drug_ids_cell: Any) -> List[str]:
        """Parse 'D02596, D10161, ...' → ['D02596','D10161', ...]"""
        if pd.isna(drug_ids_cell):
            return []
        if isinstance(drug_ids_cell, str):
            return [tok.strip().strip('"').strip("'") for tok in drug_ids_cell.split(",") if tok.strip()]
        if isinstance(drug_ids_cell, list):
            return [str(x).strip() for x in drug_ids_cell if str(x).strip()]
        return []

    def link_one(self, pathway_id: str, kegg_id: str, source: str = "ReactomeCSV") -> Dict[str, Any]:
        """
        Create (Pathway {pathway_id})-[:CONTAINS_DRUG]->(KeggDrug {kegg_id})
        Only if BOTH nodes exist. Returns a small status dict.
        """
        cypher = """
        MATCH (p:Pathway {pathway_id: $pathway_id})
        MATCH (d:KeggDrug {kegg_id: $kegg_id})
        MERGE (p)-[r:CONTAINS_DRUG]->(d)
        ON CREATE SET
            r.created_at = datetime(),
            r.source = coalesce($source, 'unknown')
        SET r.last_seen = datetime()
        RETURN p.pathway_id AS pathway_id, d.kegg_id AS kegg_id
        """
        try:
            res = self.neo4j.execute_query(cypher, {
                "pathway_id": pathway_id,
                "kegg_id": kegg_id,
                "source": source
            })
            if res and len(res) > 0:
                return {"ok": True, "pathway_id": pathway_id, "kegg_id": kegg_id}
            else:
                return {"ok": False, "pathway_id": pathway_id, "kegg_id": kegg_id, "reason": "MATCH returned 0 rows"}
        except Exception as e:
            return {"ok": False, "pathway_id": pathway_id, "kegg_id": kegg_id, "reason": str(e)}

    def link_bulk(self, pathway_id: str, drug_ids: List[str], source: str = "ReactomeCSV") -> Dict[str, int]:
        """Link a list of KEGG IDs to one pathway_id; returns counters."""
        created = 0
        missing = 0
        for dk in drug_ids:
            if not dk:
                continue
            status = self.link_one(pathway_id, dk, source=source)
            if status["ok"]:
                created += 1
            else:
                missing += 1
                logger.warning(f"Missing/mismatch for pathway_id={pathway_id}, kegg_id={dk}: {status.get('reason')}")
        return {"created_or_merged": created, "failed_or_missing": missing}

    def import_from_csv(self, csv_file_path: str) -> None:
        """Read CSV (pathway_id, pathway_name, drug_ids) and create relationships."""
        logger.info(f"Starting link import from: {csv_file_path}")
        df = pd.read_csv(csv_file_path)

        # Basic validation
        required_cols = {"pathway_id", "drug_ids"}
        if not required_cols.issubset(df.columns):
            raise ValueError(f"CSV must contain columns: {required_cols}. Found: {list(df.columns)}")

        total_rows = len(df)
        total_links = 0
        total_missing = 0

        for i, row in df.iterrows():
            if i % 100 == 0:
                logger.info(f"Progress: {i}/{total_rows}")

            pathway_id = str(row["pathway_id"]).strip()
            drug_ids = self._parse_drug_ids(row["drug_ids"])

            if not pathway_id:
                logger.warning(f"Row {i}: missing pathway_id, skipping.")
                continue

            if not drug_ids:
                logger.info(f"Row {i}: no drug_ids for pathway {pathway_id}, skipping.")
                continue

            counts = self.link_bulk(pathway_id, drug_ids)
            total_links += counts["created_or_merged"]
            total_missing += counts["failed_or_missing"]

        logger.info(f"🎉 Link import completed! Rows processed: {total_rows}")
        logger.info(f"  Relationships created/merged: {total_links}")
        logger.info(f"  Missing/mismatched pairs:     {total_missing}")


def main():
    neo4j_conn = Neo4jConnection()
    try:
        linker = KEGGDrugPathwayLinker(neo4j_conn)
        logger.info("=== Reactome Pathway ↔ KeggDrug CONTAINS_DRUG Linking ===")

        # optional but recommended
        linker.create_supporting_indexes()

        # Update the path to your CSV
        csv_file = "data/R-HSA-GK-Pathways.csv"
        if Path(csv_file).exists():
            linker.import_from_csv(csv_file)
        else:
            logger.error(f"CSV not found: {csv_file}")

        logger.info("✅ Linking complete.")
    except Exception as e:
        logger.error(f"❌ Linking failed: {e}")
        raise
    finally:
        neo4j_conn.close()


if __name__ == "__main__":
    main()
