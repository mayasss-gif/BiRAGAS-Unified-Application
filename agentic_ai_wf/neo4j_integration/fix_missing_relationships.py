#!/usr/bin/env python3
"""
Fix Missing Relationships and Related Entities

Identifies genes with missing diseases, drugs, and pathways relationships
and re-creates them from the original JSON files without touching gene nodes.
"""

import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Set, Optional, Tuple
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from .connection import Neo4jConnection

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RelationshipFixer:
    """Fix missing relationships and related entities."""

    def __init__(self, neo4j_connection: Neo4jConnection):
        self.db = neo4j_connection
        self.lock = threading.Lock()
        self.processed_count = 0
        self.diseases_created = 0
        self.drugs_created = 0
        self.pathways_created = 0
        self.relationships_created = 0
        self.errors = 0

    def get_genes_missing_relationships(self) -> Dict[str, Dict[str, bool]]:
        """Identify genes that are missing diseases, drugs, or pathways."""
        logger.info("Analyzing genes for missing relationships...")

        # Get genes with missing diseases
        missing_diseases_query = """
        MATCH (g:Gene)
        WHERE NOT (g)-[:ASSOCIATED_WITH|INFERRED_ASSOCIATION]->(:Disease)
        RETURN g.symbol as symbol
        """

        # Get genes with missing drugs
        missing_drugs_query = """
        MATCH (g:Gene)
        WHERE NOT (g)-[:INTERACTS_WITH_DRUG]->(:Drug)
        RETURN g.symbol as symbol
        """

        # Get genes with missing pathways
        missing_pathways_query = """
        MATCH (g:Gene)
        WHERE NOT (g)-[:BELONGS_TO_PATHWAY|BELONGS_TO_SUPER_PATHWAY]->(:Pathway)
        RETURN g.symbol as symbol
        """

        try:
            missing_diseases = {r['symbol']
                                for r in self.db.execute_query(missing_diseases_query)}
            missing_drugs = {r['symbol']
                             for r in self.db.execute_query(missing_drugs_query)}
            missing_pathways = {r['symbol']
                                for r in self.db.execute_query(missing_pathways_query)}

            # Combine all genes that need processing
            all_genes_needing_fix = missing_diseases | missing_drugs | missing_pathways

            genes_status = {}
            for gene in all_genes_needing_fix:
                genes_status[gene] = {
                    'missing_diseases': gene in missing_diseases,
                    'missing_drugs': gene in missing_drugs,
                    'missing_pathways': gene in missing_pathways
                }

            logger.info(f"Genes missing diseases: {len(missing_diseases):,}")
            logger.info(f"Genes missing drugs: {len(missing_drugs):,}")
            logger.info(f"Genes missing pathways: {len(missing_pathways):,}")
            logger.info(
                f"Total genes needing relationship fixes: {len(genes_status):,}")

            return genes_status

        except Exception as e:
            logger.error(f"Error analyzing missing relationships: {e}")
            return {}

    def find_json_files_for_genes(self, data_directory: str, gene_symbols: List[str]) -> Dict[str, Optional[Path]]:
        """Find JSON files for specific genes."""
        logger.info(f"Finding JSON files for {len(gene_symbols):,} genes...")

        data_path = Path(data_directory)
        if not data_path.exists():
            raise FileNotFoundError(
                f"Data directory not found: {data_directory}")

        target_genes = set(gene_symbols)
        gene_to_file = {gene: None for gene in gene_symbols}
        files_found = 0

        for folder in sorted(data_path.iterdir()):
            if folder.is_dir() and len(folder.name) == 2:
                for json_file in folder.glob("*.json"):
                    gene_symbol = json_file.stem
                    if gene_symbol in target_genes:
                        gene_to_file[gene_symbol] = json_file  # type: ignore
                        files_found += 1

                        if files_found % 1000 == 0:
                            logger.info(
                                f"Found {files_found:,} files so far...")

        files_missing = len(gene_symbols) - files_found
        logger.info(
            f"Found files for {files_found:,} genes, {files_missing:,} missing")

        return gene_to_file  # type: ignore

    def extract_relationship_data(self, json_file: Path, gene_symbol: str) -> Optional[Dict]:
        """Extract only relationship data from JSON file."""
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                json_data = json.load(f)

            relationship_data = {
                "gene_symbol": gene_symbol,
                "diseases": [],
                "drugs": [],
                "pathways": []
            }

            # Extract disease associations
            if "MalaCardsDisorders" in json_data and json_data["MalaCardsDisorders"]:
                for disorder in json_data["MalaCardsDisorders"]:
                    if isinstance(disorder, dict):
                        relationship_data["diseases"].append({
                            "name": disorder.get("Name", ""),
                            "malacards_id": disorder.get("Accession", ""),
                            "aliases": disorder.get("Aliases", []),
                            "sources": disorder.get("Sources", []),
                            "gene_score": disorder.get("GeneScore"),
                            "disorder_score": disorder.get("DisorderScore"),
                            "is_elite": disorder.get("IsElite", False),
                            "is_cancer_census": disorder.get("IsCancerCensus", False),
                            "disorder_type": "direct"
                        })

            # Extract inferred disease associations
            if "MalaCardsInferredDisorders" in json_data and json_data["MalaCardsInferredDisorders"]:
                for disorder in json_data["MalaCardsInferredDisorders"]:
                    if isinstance(disorder, dict):
                        relationship_data["diseases"].append({
                            "name": disorder.get("Name", ""),
                            "malacards_id": disorder.get("Accession", ""),
                            "aliases": disorder.get("Aliases", []),
                            "sources": disorder.get("Sources", []),
                            "gene_score": disorder.get("GeneScore"),
                            "disorder_score": disorder.get("DisorderScore"),
                            "is_elite": False,
                            "is_cancer_census": disorder.get("IsCancerCensus", False),
                            "disorder_type": "inferred"
                        })

            # Extract drug associations
            if "UnifiedDrugs" in json_data and json_data["UnifiedDrugs"]:
                drug_data = json_data["UnifiedDrugs"]
                if isinstance(drug_data, list):
                    for drug in drug_data:
                        if isinstance(drug, dict):
                            relationship_data["drugs"].append({
                                "name": drug.get("Name", ""),
                                "sources": drug.get("Sources", []),
                                "group": drug.get("Group", ""),
                                "role": drug.get("Role", ""),
                                "action": drug.get("Action", ""),
                                "synonyms": drug.get("Synonyms", []),
                                "pubchem_ids": drug.get("PubChemIds", []),
                                "cas_numbers": drug.get("CasNumbers", ""),
                                "pubmed_ids": drug.get("PubMedIds", []),
                                "status": drug.get("Status", "")
                            })

            # Extract pathway associations from "Pathways"
            if "Pathways" in json_data and json_data["Pathways"]:
                for pathway in json_data["Pathways"]:
                    if isinstance(pathway, dict):
                        relationship_data["pathways"].append({
                            "name": pathway.get("Name", ""),
                            "pathway_id": pathway.get("Accession", ""),
                            "source": pathway.get("Source", ""),
                            "category": "",
                            "evidence_score": None,
                            "super_pathway_name": ""
                        })

            # Extract pathway associations from "SuperPathway"
            if "SuperPathway" in json_data and json_data["SuperPathway"]:
                for super_pathway in json_data["SuperPathway"]:
                    if isinstance(super_pathway, dict):
                        super_pathway_name = super_pathway.get("Name", "")
                        members = super_pathway.get("Members", [])

                        for member in members:
                            if isinstance(member, dict):
                                # Parse score safely
                                score_str = member.get("Score", "")
                                evidence_score = None

                                if score_str and score_str != "-":
                                    try:
                                        if score_str.startswith('.'):
                                            score_str = '0' + score_str
                                        evidence_score = float(score_str)
                                    except (ValueError, TypeError):
                                        evidence_score = None

                                relationship_data["pathways"].append({
                                    "name": member.get("Pathway", ""),
                                    "pathway_id": member.get("SourceAccession", ""),
                                    "source": member.get("Source", ""),
                                    "category": "SuperPathway",
                                    "evidence_score": evidence_score,
                                    "super_pathway_name": super_pathway_name
                                })

            return relationship_data

        except Exception as e:
            logger.error(
                f"Error extracting relationships from {json_file}: {e}")
            return None

    def create_disease_relationships(self, gene_symbol: str, diseases: List[Dict]) -> int:
        """Create disease nodes and relationships."""
        created_count = 0

        for disease in diseases:
            disease_name = disease.get('name', '')
            if not disease_name:
                continue

            try:
                # Create disease node
                disease_properties = {
                    'name': disease_name,
                    'malacards_id': disease.get('malacards_id', ''),
                    'aliases': json.dumps(disease.get('aliases', [])),
                    'sources': json.dumps(disease.get('sources', [])),
                    'is_elite': disease.get('is_elite', False),
                    'is_cancer_census': disease.get('is_cancer_census', False),
                    'is_inferred': disease.get('disorder_type') == 'inferred',
                    'created_at': datetime.now().isoformat()
                }

                # Remove empty values
                disease_properties = {k: v for k, v in disease_properties.items()
                                      if v is not None and v != '' and v != '[]'}

                # Determine relationship type
                relationship_type = "INFERRED_ASSOCIATION" if disease.get(
                    'disorder_type') == 'inferred' else "ASSOCIATED_WITH"

                query = f"""
                MERGE (d:Disease {{name: $disease_name}})
                SET d += $disease_properties
                WITH d
                MATCH (g:Gene {{symbol: $gene_symbol}})
                MERGE (g)-[r:{relationship_type}]->(d)
                SET r.gene_score = $gene_score,
                    r.disorder_score = $disorder_score,
                    r.is_elite = $is_elite,
                    r.is_cancer_census = $is_cancer_census,
                    r.disorder_type = $disorder_type,
                    r.sources = $sources,
                    r.created_at = $created_at
                """

                self.db.execute_write(query, {
                    'disease_name': disease_name,
                    'disease_properties': disease_properties,
                    'gene_symbol': gene_symbol,
                    'gene_score': disease.get('gene_score'),
                    'disorder_score': disease.get('disorder_score'),
                    'is_elite': disease.get('is_elite', False),
                    'is_cancer_census': disease.get('is_cancer_census', False),
                    'disorder_type': disease.get('disorder_type', 'direct'),
                    'sources': json.dumps(disease.get('sources', [])),
                    'created_at': datetime.now().isoformat()
                })

                created_count += 1

            except Exception as e:
                logger.error(
                    f"Error creating disease {disease_name} for {gene_symbol}: {e}")

        return created_count

    def create_drug_relationships(self, gene_symbol: str, drugs: List[Dict]) -> int:
        """Create drug nodes and relationships."""
        created_count = 0

        for drug in drugs:
            drug_name = drug.get('name', '')
            if not drug_name:
                continue

            try:
                # Create drug node
                drug_properties = {
                    'name': drug_name,
                    'sources': json.dumps(drug.get('sources', [])),
                    'group': drug.get('group', ''),
                    'role': drug.get('role', ''),
                    'action': drug.get('action', ''),
                    'synonyms': json.dumps(drug.get('synonyms', [])),
                    'pubchem_ids': json.dumps(drug.get('pubchem_ids', [])),
                    'cas_numbers': drug.get('cas_numbers', ''),
                    'pubmed_ids': json.dumps(drug.get('pubmed_ids', [])),
                    'status': drug.get('status', ''),
                    'created_at': datetime.now().isoformat()
                }

                # Remove empty values
                drug_properties = {k: v for k, v in drug_properties.items()
                                   if v is not None and v != '' and v != '[]'}

                query = """
                MERGE (d:Drug {name: $drug_name})
                SET d += $drug_properties
                WITH d
                MATCH (g:Gene {symbol: $gene_symbol})
                MERGE (g)-[r:INTERACTS_WITH_DRUG]->(d)
                SET r.group = $group,
                    r.role = $role,
                    r.action = $action,
                    r.sources = $sources,
                    r.status = $status,
                    r.created_at = $created_at
                """

                self.db.execute_write(query, {
                    'drug_name': drug_name,
                    'drug_properties': drug_properties,
                    'gene_symbol': gene_symbol,
                    'group': drug.get('group', ''),
                    'role': drug.get('role', ''),
                    'action': drug.get('action', ''),
                    'sources': json.dumps(drug.get('sources', [])),
                    'status': drug.get('status', ''),
                    'created_at': datetime.now().isoformat()
                })

                created_count += 1

            except Exception as e:
                logger.error(
                    f"Error creating drug {drug_name} for {gene_symbol}: {e}")

        return created_count

    def create_pathway_relationships(self, gene_symbol: str, pathways: List[Dict]) -> int:
        """Create pathway nodes and relationships."""
        created_count = 0

        for pathway in pathways:
            pathway_name = pathway.get('name', '')
            if not pathway_name:
                continue

            try:
                # Create pathway node
                pathway_properties = {
                    'name': pathway_name,
                    'pathway_id': pathway.get('pathway_id', ''),
                    'source': pathway.get('source', ''),
                    'category': pathway.get('category', ''),
                    'evidence_score': pathway.get('evidence_score'),
                    'super_pathway_name': pathway.get('super_pathway_name', ''),
                    'created_at': datetime.now().isoformat()
                }

                # Remove empty values
                pathway_properties = {k: v for k, v in pathway_properties.items()
                                      if v is not None and v != ''}

                # Determine relationship type
                relationship_type = "BELONGS_TO_SUPER_PATHWAY" if pathway.get(
                    'category') == 'SuperPathway' else "BELONGS_TO_PATHWAY"

                query = f"""
                MERGE (p:Pathway {{name: $pathway_name}})
                SET p += $pathway_properties
                WITH p
                MATCH (g:Gene {{symbol: $gene_symbol}})
                MERGE (g)-[r:{relationship_type}]->(p)
                SET r.source = $source,
                    r.evidence_score = $evidence_score,
                    r.category = $category,
                    r.super_pathway_name = $super_pathway_name,
                    r.created_at = $created_at
                """

                self.db.execute_write(query, {
                    'pathway_name': pathway_name,
                    'pathway_properties': pathway_properties,
                    'gene_symbol': gene_symbol,
                    'source': pathway.get('source', ''),
                    'evidence_score': pathway.get('evidence_score'),
                    'category': pathway.get('category', ''),
                    'super_pathway_name': pathway.get('super_pathway_name', ''),
                    'created_at': datetime.now().isoformat()
                })

                created_count += 1

            except Exception as e:
                logger.error(
                    f"Error creating pathway {pathway_name} for {gene_symbol}: {e}")

        return created_count

    def process_gene_relationships(self, gene_symbol: str, json_file: Path, missing_status: Dict[str, bool]) -> bool:
        """Process relationships for a single gene."""
        try:
            # Extract relationship data
            rel_data = self.extract_relationship_data(json_file, gene_symbol)
            if not rel_data:
                return False

            diseases_created = 0
            drugs_created = 0
            pathways_created = 0

            # Create missing diseases
            if missing_status.get('missing_diseases', False):
                diseases_created = self.create_disease_relationships(
                    gene_symbol, rel_data['diseases'])

            # Create missing drugs
            if missing_status.get('missing_drugs', False):
                drugs_created = self.create_drug_relationships(
                    gene_symbol, rel_data['drugs'])

            # Create missing pathways
            if missing_status.get('missing_pathways', False):
                pathways_created = self.create_pathway_relationships(
                    gene_symbol, rel_data['pathways'])

            # Update counters
            with self.lock:
                self.diseases_created += diseases_created
                self.drugs_created += drugs_created
                self.pathways_created += pathways_created
                self.relationships_created += diseases_created + drugs_created + pathways_created
                self.processed_count += 1

            return True

        except Exception as e:
            logger.error(
                f"Error processing relationships for {gene_symbol}: {e}")
            with self.lock:
                self.errors += 1
                self.processed_count += 1
            return False

    def fix_missing_relationships(self, data_directory: str, max_workers: int = 2, batch_size: int = 1000):
        """Fix all missing relationships."""
        logger.info("=" * 60)
        logger.info("FIXING MISSING RELATIONSHIPS")
        logger.info("=" * 60)

        # Step 1: Identify genes with missing relationships
        genes_status = self.get_genes_missing_relationships()
        if not genes_status:
            logger.info("No genes with missing relationships found.")
            return

        # Step 2: Find corresponding JSON files
        gene_symbols = list(genes_status.keys())
        gene_to_file = self.find_json_files_for_genes(
            data_directory, gene_symbols)

        # Step 3: Filter genes with available files
        genes_with_files = {
            gene: (file_path, genes_status[gene])
            for gene, file_path in gene_to_file.items()
            if file_path is not None
        }

        logger.info(f"Genes to process: {len(genes_with_files):,}")

        if not genes_with_files:
            logger.warning("No files found for genes needing fixes.")
            return

        # Step 4: Process in batches with single thread (to avoid deadlocks)
        genes_list = list(genes_with_files.items())
        total_genes = len(genes_list)

        logger.info(
            f"Processing {total_genes:,} genes (single-threaded for stability)...")

        for i, (gene_symbol, (json_file, missing_status)) in enumerate(genes_list):
            try:
                success = self.process_gene_relationships(
                    gene_symbol, json_file, missing_status)

                if i % 100 == 0 and i > 0:
                    self.print_progress()

            except Exception as e:
                logger.error(f"Error processing {gene_symbol}: {e}")
                with self.lock:
                    self.errors += 1
                    self.processed_count += 1

        # Final results
        logger.info("=" * 60)
        logger.info("RELATIONSHIP FIXING COMPLETED!")
        logger.info("=" * 60)
        self.print_final_results()

    def print_progress(self):
        """Print current progress."""
        logger.info(f"""Progress: {self.processed_count:,} processed
        Diseases created: {self.diseases_created:,}
        Drugs created: {self.drugs_created:,} 
        Pathways created: {self.pathways_created:,}
        Errors: {self.errors:,}""")

    def print_final_results(self):
        """Print final results."""
        logger.info(f"""
        Final Results:
        ==============
        Genes processed: {self.processed_count:,}
        Diseases created: {self.diseases_created:,}
        Drugs created: {self.drugs_created:,}
        Pathways created: {self.pathways_created:,}
        Total relationships: {self.relationships_created:,}
        Errors: {self.errors:,}
        """)


def main():
    """Main function."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Fix missing relationships and entities')
    parser.add_argument(
        'data_directory', help='Path to GeneCards data directory')
    parser.add_argument('--batch-size', type=int, default=1000,
                        help='Batch size for processing (default: 1000)')

    args = parser.parse_args()

    # Connect to Neo4j
    try:
        db = Neo4jConnection()
        logger.info("✅ Connected to Neo4j successfully!")
    except Exception as e:
        logger.error(f"❌ Failed to connect to Neo4j: {e}")
        sys.exit(1)

    # Create fixer and run
    fixer = RelationshipFixer(db)

    try:
        fixer.fix_missing_relationships(
            data_directory=args.data_directory,
            batch_size=args.batch_size
        )

    except KeyboardInterrupt:
        logger.info("Fix interrupted by user")
    except Exception as e:
        logger.error(f"Fix failed: {e}")
        sys.exit(1)
    finally:
        db.close()


if __name__ == "__main__":
    main()
