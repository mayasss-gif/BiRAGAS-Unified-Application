"""
Neo4j KEGG Drug Data Import Script
Handles KEGG drug data with proper node management and relationships.
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


class KEGGDrugNeo4jImporter:
    """Imports KEGG drug data into Neo4j with intelligent node management."""
    
    def __init__(self, neo4j_connection: Neo4jConnection):
        self.neo4j = neo4j_connection
        
    def create_constraints_and_indexes(self):
        """Create necessary constraints and indexes."""
        queries = [
            "CREATE CONSTRAINT IF NOT EXISTS FOR (d:KeggDrug) REQUIRE d.kegg_id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (g:Gene) REQUIRE g.symbol IS UNIQUE", 
            "CREATE CONSTRAINT IF NOT EXISTS FOR (kg:KeggDrugGene) REQUIRE kg.symbol IS UNIQUE",
            "CREATE INDEX IF NOT EXISTS FOR (d:KeggDrug) ON (d.drug_name)",
            "CREATE INDEX IF NOT EXISTS FOR (d:KeggDrug) ON (d.fda_approved_status)"
        ]
        
        for query in queries:
            try:
                self.neo4j.execute_query(query)
                logger.info(f"✅ Created: {query}")
            except Exception as e:
                logger.warning(f"❌ Failed: {e}")
    
    def parse_list_string(self, list_str: str) -> List[str]:
        """Safely parse string representation of list."""
        if pd.isna(list_str) or list_str in ['', 'Not available', 'Not_available']:
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
    
    def check_gene_exists(self, gene_symbol: str) -> bool:
        """Check if gene exists in Gene node (from GeneCards)."""
        query = "MATCH (g:Gene {symbol: $symbol}) RETURN count(g) as count"
        result = self.neo4j.execute_query(query, {"symbol": gene_symbol})
        return result[0]['count'] > 0 if result else False
    
    def create_kegg_drug_node(self, drug_data: Dict[str, Any]) -> None:
        """Create KeggDrug node with all properties."""
        query = """
        MERGE (d:KeggDrug {kegg_id: $kegg_id})
        SET d.drug_name = $drug_name,
            d.target_genes_updated = $target_genes_updated,
            d.efficacy = $efficacy,
            d.drugbank_id = $drugbank_id,
            d.chembl_id = $chembl_id,
            d.fda_approved_status = $fda_approved_status,
            d.brand_name = $brand_name,
            d.generic_name = $generic_name,
            d.route_of_administration = $route_of_administration,
            d.fda_adverse_reactions = $fda_adverse_reactions,
            d.mechanism_of_action = $mechanism_of_action,
            d.updated_at = datetime()
        """
        
        params = {
            'kegg_id': drug_data.get('KeggDrug_id', ''),
            'drug_name': drug_data.get('Drug_name', ''),
            'target_genes_updated': drug_data.get('TargetGenes_updated', ''),
            'efficacy': drug_data.get('Efficacy', ''),
            'drugbank_id': drug_data.get('Drugbank_id', ''),
            'chembl_id': drug_data.get('Chembl_id', ''),
            'fda_approved_status': drug_data.get('fdaAproved_Status', ''),
            'brand_name': drug_data.get('Brand_name', ''),
            'generic_name': drug_data.get('Generic_name', ''),
            'route_of_administration': drug_data.get('Route_of_Administration', ''),
            'fda_adverse_reactions': drug_data.get('fda_adverse_reactions', ''),
            'mechanism_of_action': drug_data.get('Mechanism_of_Action', '')
        }
        
        self.neo4j.execute_query(query, params)
    
    def create_gene_relationships(self, kegg_id: str, target_genes: List[str]) -> None:
        """Create relationships with genes - use existing Gene or create KeggDrugGene."""
        for gene_symbol in target_genes:
            if not gene_symbol or not gene_symbol.strip():
                continue
                
            gene_symbol = gene_symbol.strip()
            
            if self.check_gene_exists(gene_symbol):
                # Link to existing Gene node (from GeneCards)
                query = """
                MATCH (d:KeggDrug {kegg_id: $kegg_id})
                MATCH (g:Gene {symbol: $gene_symbol})
                MERGE (d)-[:TARGETS]->(g)
                """
                self.neo4j.execute_query(query, {
                    'kegg_id': kegg_id, 
                    'gene_symbol': gene_symbol
                })
                logger.debug(f"Linked to existing Gene: {gene_symbol}")
            else:
                # Create new KeggDrugGene node
                query = """
                MATCH (d:KeggDrug {kegg_id: $kegg_id})
                MERGE (kg:KeggDrugGene {symbol: $gene_symbol})
                SET kg.updated_at = datetime()
                MERGE (d)-[:TARGETS_KEGG_GENE]->(kg)
                """
                self.neo4j.execute_query(query, {
                    'kegg_id': kegg_id,
                    'gene_symbol': gene_symbol
                })
                logger.debug(f"Created KeggDrugGene: {gene_symbol}")
    
    def create_pathway_relationships(self, kegg_id: str, pathways: List[str]) -> None:
        """Create relationships with pathways."""
        for pathway_id in pathways:
            if not pathway_id or not pathway_id.strip():
                continue
                
            query = """
            MATCH (d:KeggDrug {kegg_id: $kegg_id})
            MERGE (p:Pathway {pathway_id: $pathway_id})
            SET p.updated_at = datetime()
            MERGE (d)-[:AFFECTS_PATHWAY]->(p)
            """
            self.neo4j.execute_query(query, {
                'kegg_id': kegg_id,
                'pathway_id': pathway_id.strip()
            })
    
    def create_drug_class_relationships(self, kegg_id: str, drug_classes: List[str]) -> None:
        """Create relationships with drug classes."""
        for drug_class in drug_classes:
            if not drug_class or not drug_class.strip():
                continue
                
            query = """
            MATCH (d:KeggDrug {kegg_id: $kegg_id})
            MERGE (dc:DrugClass {name: $drug_class})
            SET dc.updated_at = datetime()
            MERGE (d)-[:BELONGS_TO_CLASS]->(dc)
            """
            self.neo4j.execute_query(query, {
                'kegg_id': kegg_id,
                'drug_class': drug_class.strip()
            })
    
    def import_drug_data(self, drug_data: Dict[str, Any]) -> None:
        """Import single drug record with all relationships."""
        kegg_id = drug_data.get('KeggDrug_id', '')
        if not kegg_id:
            logger.warning("Skipping drug with no KEGG ID")
            return
        
        try:
            # Create KeggDrug node
            self.create_kegg_drug_node(drug_data)
            
            # Create gene relationships
            target_genes = self.parse_list_string(drug_data.get('TargetGenes', ''))
            if target_genes:
                self.create_gene_relationships(kegg_id, target_genes)
            else:
                target_genes = self.parse_list_string(drug_data.get('TargetGenes_updated', ''))
                if target_genes:
                    self.create_gene_relationships(kegg_id, target_genes)
            
            # # Create pathway relationships
            # pathways = self.parse_list_string(drug_data.get('KeggTarget_Pathways', ''))
            # if pathways:
            #     self.create_pathway_relationships(kegg_id, pathways)
            
            # Create drug class relationships
            drug_classes = self.parse_list_string(drug_data.get('DrugClasses', ''))
            if drug_classes:
                self.create_drug_class_relationships(kegg_id, drug_classes)
            
            logger.info(f"✅ Imported: {kegg_id} - {drug_data.get('Drug_name', 'Unknown')}")
            
        except Exception as e:
            logger.error(f"❌ Failed to import {kegg_id}: {e}")
    
    def import_from_csv(self, csv_file_path: str) -> None:
        """Import KEGG drug data from CSV file."""
        logger.info(f"Starting import from: {csv_file_path}")
        
        # Setup database
        self.create_constraints_and_indexes()
        
        # Load and process data
        df = pd.read_csv(csv_file_path)
        logger.info(f"Loaded {len(df)} drug records")
        
        for index, row in df.iterrows():
            if index % 100 == 0:
                logger.info(f"Progress: {index}/{len(df)}")
            
            self.import_drug_data(row.to_dict())
        
        logger.info(f"🎉 Import completed! Processed {len(df)} drugs")


def main():
    """Main function to run the KEGG drug import."""
    
    # Initialize Neo4j connection
    neo4j_conn = Neo4jConnection()
    
    try:
        # Initialize importer
        importer = KEGGDrugNeo4jImporter(neo4j_conn)
        
        logger.info("=== KEGG Drug Neo4j Import ===")
        
        # Setup database
        importer.create_constraints_and_indexes()
        
        
        # For CSV import - example:
        csv_file = "./agentic_ai_wf/drugs_extraction_evaluation/r_codes/drug_details_final.csv"
        if Path(csv_file).exists():
            logger.info(f"Importing from CSV: {csv_file}")
            importer.import_from_csv(csv_file)
        else:
            logger.info("CSV file not found, using sample data only")
        
        logger.info("🎉 KEGG Drug import completed!")
        
    except Exception as e:
        logger.error(f"❌ Import failed: {e}")
        raise
    finally:
        neo4j_conn.close()


if __name__ == "__main__":
    main()