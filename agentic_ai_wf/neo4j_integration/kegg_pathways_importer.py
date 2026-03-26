"""
Neo4j KEGG Pathways Data Import Script
Handles KEGG pathway data with relationships to drugs and genes.
"""

import pandas as pd
import ast
import logging
from typing import List, Dict, Any
from pathlib import Path
from connection import Neo4jConnection

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class KEGGPathwaysNeo4jImporter:
    """Imports KEGG pathway data into Neo4j with intelligent relationship management."""
    
    def __init__(self, neo4j_connection: Neo4jConnection):
        self.neo4j = neo4j_connection
        
    def create_constraints_and_indexes(self):
        """Create necessary constraints and indexes for pathways."""
        queries = [
            "CREATE CONSTRAINT IF NOT EXISTS FOR (p:KeggPathway) REQUIRE p.pathway_id IS UNIQUE",
            "CREATE INDEX IF NOT EXISTS FOR (p:KeggPathway) ON (p.pathway_name)",
            "CREATE INDEX IF NOT EXISTS FOR (p:KeggPathway) ON (p.gene_count)",
            "CREATE INDEX IF NOT EXISTS FOR (p:KeggPathway) ON (p.drug_count)"
        ]
        
        for query in queries:
            try:
                self.neo4j.execute_query(query)
                logger.info(f"✅ Created: {query}")
            except Exception as e:
                logger.warning(f"❌ Failed: {e}")
    
    def parse_list_string(self, list_str: str) -> List[str]:
        """Safely parse string representation of list."""
        if pd.isna(list_str) or list_str in ['', 'Not available', 'Not_available', '[]']:
            return []
        
        try:
            if isinstance(list_str, str) and list_str.startswith('['):
                return ast.literal_eval(list_str)
            elif isinstance(list_str, str):
                return [item.strip().strip("'\"") for item in list_str.split(',') if item.strip()]
            return []
        except Exception as e:
            logger.warning(f"Parse error for {list_str}: {e}")
            return []
    
    def check_kegg_drug_exists(self, drug_id: str) -> bool:
        """Check if KeggDrug exists."""
        query = "MATCH (d:KeggDrug {kegg_id: $drug_id}) RETURN count(d) as count"
        result = self.neo4j.execute_query(query, {"drug_id": drug_id})
        return result[0]['count'] > 0 if result else False
    
    def check_gene_exists(self, gene_symbol: str) -> bool:
        """Check if Gene exists (from GeneCards)."""
        query = "MATCH (g:Gene {symbol: $symbol}) RETURN count(g) as count"
        result = self.neo4j.execute_query(query, {"symbol": gene_symbol})
        return result[0]['count'] > 0 if result else False
    
    def check_kegg_drug_gene_exists(self, gene_symbol: str) -> bool:
        """Check if KeggDrugGene exists."""
        query = "MATCH (kg:KeggDrugGene {symbol: $symbol}) RETURN count(kg) as count"
        result = self.neo4j.execute_query(query, {"symbol": gene_symbol})
        return result[0]['count'] > 0 if result else False
    
    def create_kegg_pathway_node(self, pathway_data: Dict[str, Any]) -> None:
        """Create KeggPathway node with all properties."""
        query = """
        MERGE (p:KeggPathway {pathway_id: $pathway_id})
        SET p.pathway_name = $pathway_name,
            p.gene_count = $gene_count,
            p.drug_count = $drug_count,
            p.updated_at = datetime()
        """
        
        params = {
            'pathway_id': pathway_data.get('pathway_id', ''),
            'pathway_name': pathway_data.get('pathway_name', ''),
            'gene_count': int(pathway_data.get('gene_count', 0)),
            'drug_count': int(pathway_data.get('drug_count', 0))
        }
        
        self.neo4j.execute_query(query, params)
        logger.debug(f"Created KeggPathway: {params['pathway_id']}")
    
    def create_pathway_drug_relationships(self, pathway_id: str, drug_ids: List[str]) -> None:
        """Create relationships between KeggPathway and KeggDrug."""
        for drug_id in drug_ids:
            if not drug_id or not drug_id.strip():
                continue
                
            drug_id = drug_id.strip()
            
            if self.check_kegg_drug_exists(drug_id):
                query = """
                MATCH (p:KeggPathway {pathway_id: $pathway_id})
                MATCH (d:KeggDrug {kegg_id: $drug_id})
                MERGE (p)-[:CONTAINS_DRUG]->(d)
                MERGE (d)-[:AFFECTS_PATHWAY]->(p)
                """
                self.neo4j.execute_query(query, {
                    'pathway_id': pathway_id,
                    'drug_id': drug_id
                })
                logger.debug(f"Linked pathway {pathway_id} to drug {drug_id}")
            else:
                logger.debug(f"Drug {drug_id} not found in KeggDrug nodes")
    
    def create_pathway_gene_relationships(self, pathway_id: str, gene_symbols: List[str]) -> None:
        """Create relationships between KeggPathway and genes (Gene or KeggDrugGene)."""
        for gene_symbol in gene_symbols:
            if not gene_symbol or not gene_symbol.strip():
                continue
                
            gene_symbol = gene_symbol.strip()
            
            # Check if gene exists in Gene node (from GeneCards)
            if self.check_gene_exists(gene_symbol):
                query = """
                MATCH (p:KeggPathway {pathway_id: $pathway_id})
                MATCH (g:Gene {symbol: $gene_symbol})
                MERGE (p)-[:CONTAINS_GENE]->(g)
                MERGE (g)-[:PARTICIPATES_IN_PATHWAY]->(p)
                """
                self.neo4j.execute_query(query, {
                    'pathway_id': pathway_id,
                    'gene_symbol': gene_symbol
                })
                logger.debug(f"Linked pathway {pathway_id} to Gene {gene_symbol}")
                
            # Check if gene exists in KeggDrugGene node
            elif self.check_kegg_drug_gene_exists(gene_symbol):
                query = """
                MATCH (p:KeggPathway {pathway_id: $pathway_id})
                MATCH (kg:KeggDrugGene {symbol: $gene_symbol})
                MERGE (p)-[:CONTAINS_KEGG_GENE]->(kg)
                MERGE (kg)-[:PARTICIPATES_IN_PATHWAY]->(p)
                """
                self.neo4j.execute_query(query, {
                    'pathway_id': pathway_id,
                    'gene_symbol': gene_symbol
                })
                logger.debug(f"Linked pathway {pathway_id} to KeggDrugGene {gene_symbol}")
                
            else:
                # Create new KeggPathwayGene node for genes not in either existing node type
                query = """
                MATCH (p:KeggPathway {pathway_id: $pathway_id})
                MERGE (pg:KeggPathwayGene {symbol: $gene_symbol})
                SET pg.updated_at = datetime()
                MERGE (p)-[:CONTAINS_PATHWAY_GENE]->(pg)
                MERGE (pg)-[:PARTICIPATES_IN_PATHWAY]->(p)
                """
                self.neo4j.execute_query(query, {
                    'pathway_id': pathway_id,
                    'gene_symbol': gene_symbol
                })
                logger.debug(f"Created KeggPathwayGene and linked to pathway {pathway_id}: {gene_symbol}")
    
    def import_pathway_data(self, pathway_data: Dict[str, Any]) -> None:
        """Import single pathway record with all relationships."""
        pathway_id = pathway_data.get('pathway_id', '')
        if not pathway_id:
            logger.warning("Skipping pathway with no pathway_id")
            return
        
        try:
            # Create KeggPathway node
            self.create_kegg_pathway_node(pathway_data)
            
            # Create drug relationships
            drug_ids = self.parse_list_string(pathway_data.get('drugs', ''))
            if drug_ids:
                self.create_pathway_drug_relationships(pathway_id, drug_ids)
            
            # Create gene relationships
            gene_symbols = self.parse_list_string(pathway_data.get('genes', ''))
            if gene_symbols:
                self.create_pathway_gene_relationships(pathway_id, gene_symbols)
            
            logger.info(f"✅ Imported pathway: {pathway_id} - {pathway_data.get('pathway_name', 'Unknown')}")
            logger.info(f"   Drugs: {len(drug_ids)}, Genes: {len(gene_symbols)}")
            
        except Exception as e:
            logger.error(f"❌ Failed to import pathway {pathway_id}: {e}")
    
    def import_from_csv(self, csv_file_path: str) -> None:
        """Import KEGG pathway data from CSV file."""
        logger.info(f"Starting pathway import from: {csv_file_path}")
        
        # Setup database
        self.create_constraints_and_indexes()
        
        # Load and process data
        df = pd.read_csv(csv_file_path)
        logger.info(f"Loaded {len(df)} pathway records")
        
        for index, row in df.iterrows():
            if index % 50 == 0:
                logger.info(f"Progress: {index}/{len(df)}")
            
            self.import_pathway_data(row.to_dict())
        
        logger.info(f"🎉 Pathway import completed! Processed {len(df)} pathways")
    
    def get_pathway_stats(self) -> Dict[str, int]:
        """Get statistics about imported pathway data."""
        queries = {
            'kegg_pathways': "MATCH (p:KeggPathway) RETURN count(p) as count",
            'kegg_pathway_genes': "MATCH (pg:KeggPathwayGene) RETURN count(pg) as count",
            'pathway_drug_relationships': "MATCH (:KeggPathway)-[:CONTAINS_DRUG]->(:KeggDrug) RETURN count(*) as count",
            'pathway_gene_relationships': "MATCH (:KeggPathway)-[:CONTAINS_GENE]->(:Gene) RETURN count(*) as count",
            'pathway_kegg_gene_relationships': "MATCH (:KeggPathway)-[:CONTAINS_KEGG_GENE]->(:KeggDrugGene) RETURN count(*) as count",
            'pathway_pathway_gene_relationships': "MATCH (:KeggPathway)-[:CONTAINS_PATHWAY_GENE]->(:KeggPathwayGene) RETURN count(*) as count"
        }
        
        stats = {}
        for name, query in queries.items():
            try:
                result = self.neo4j.execute_query(query)
                stats[name] = result[0]['count'] if result else 0
            except Exception as e:
                logger.warning(f"Failed to get stats for {name}: {e}")
                stats[name] = 0
        
        return stats
    
    def create_additional_constraints(self):
        """Create additional constraints for pathway-specific nodes."""
        queries = [
            "CREATE CONSTRAINT IF NOT EXISTS FOR (pg:KeggPathwayGene) REQUIRE pg.symbol IS UNIQUE"
        ]
        
        for query in queries:
            try:
                self.neo4j.execute_query(query)
                logger.info(f"✅ Created additional constraint: {query}")
            except Exception as e:
                logger.warning(f"❌ Failed to create constraint: {e}")


def main():
    """Main function to run the KEGG pathway import."""
    # Initialize Neo4j connection
    neo4j_conn = Neo4jConnection()
    
    try:
        # Initialize importer
        importer = KEGGPathwaysNeo4jImporter(neo4j_conn)
        
        logger.info("=== KEGG Pathways Neo4j Import ===")
        
        # Setup database
        importer.create_constraints_and_indexes()
        importer.create_additional_constraints()
        
        # Import from CSV
        csv_file = "./agentic_ai_wf/drugs_extraction_evaluation/r_codes/kegg_pathways_final.csv"
        
        if Path(csv_file).exists():
            logger.info(f"Importing pathways from CSV: {csv_file}")
            importer.import_from_csv(csv_file)
            
            # Get and display statistics
            stats = importer.get_pathway_stats()
            logger.info("📊 Import Statistics:")
            for key, value in stats.items():
                logger.info(f"   {key}: {value}")
                
        else:
            logger.error(f"CSV file not found: {csv_file}")
        
        logger.info("🎉 KEGG Pathways import completed!")
        
    except Exception as e:
        logger.error(f"❌ Import failed: {e}")
        raise
    finally:
        neo4j_conn.close()


if __name__ == "__main__":
    main()