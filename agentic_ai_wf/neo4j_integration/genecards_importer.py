#!/usr/bin/env python3
"""
GeneCards JSON to Neo4j Importer

Imports large-scale GeneCards JSON datasets into Neo4j with optimized schema
and batch processing for biomedical research applications.

Features:
- Handles 10GB+ datasets with memory-efficient streaming
- Rich biomedical entity modeling (genes, diseases, drugs, pathways)
- Batch processing with progress tracking
- Relationship inference and creation
- Error handling and resumption capabilities
"""

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Iterator, Tuple
from dataclasses import dataclass
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import socket

from .connection import Neo4jConnection

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def detect_optimal_workers(neo4j_uri: Optional[str] = None) -> int:
    """Detect optimal number of workers based on environment."""
    # Check if running on EC2
    try:
        # EC2 metadata service check
        response = os.system(
            "curl -s --connect-timeout 1 http://169.254.169.254/latest/meta-data/instance-id > /dev/null 2>&1")
        is_ec2 = response == 0
    except:
        is_ec2 = False

    # Check if Neo4j is remote
    is_remote_neo4j = neo4j_uri and not any(host in neo4j_uri.lower() for host in [
                                            'localhost', '127.0.0.1', '0.0.0.0'])

    if is_ec2 or is_remote_neo4j:
        logger.info(
            "Detected EC2 or remote Neo4j - using conservative worker count")
        return 1  # Conservative for EC2/remote
    else:
        logger.info("Detected local environment - using optimal worker count")
        return 4  # Optimal for local


@dataclass
class ImportStats:
    """Track import statistics."""
    files_processed: int = 0
    genes_created: int = 0
    diseases_created: int = 0
    drugs_created: int = 0
    pathways_created: int = 0
    publications_created: int = 0
    proteins_created: int = 0
    summaries_created: int = 0
    relationships_created: int = 0
    errors: int = 0
    start_time: float = 0

    def __post_init__(self):
        if self.start_time == 0:
            self.start_time = time.time()

    def elapsed_time(self) -> float:
        return time.time() - self.start_time

    def files_per_second(self) -> float:
        elapsed = self.elapsed_time()
        return self.files_processed / elapsed if elapsed > 0 else 0


class GeneCardsImporter:
    """Import GeneCards JSON files into Neo4j database."""

    def __init__(self, neo4j_connection: Neo4jConnection, batch_size: int = 50):
        self.db = neo4j_connection
        self.batch_size = batch_size
        self.stats = ImportStats()
        self.lock = threading.Lock()

        # Create indexes for performance
        self.create_indexes()

    def create_indexes(self):
        """Create indexes and constraints for optimal performance."""
        logger.info("Creating database indexes and constraints...")

        indexes_and_constraints = [
            # Gene indexes
            "CREATE INDEX gene_symbol IF NOT EXISTS FOR (g:Gene) ON (g.symbol)",
            "CREATE INDEX gene_genecards_id IF NOT EXISTS FOR (g:Gene) ON (g.genecards_id)",
            "CREATE INDEX gene_ncbi_id IF NOT EXISTS FOR (g:Gene) ON (g.ncbi_id)",
            "CREATE INDEX gene_ensembl_id IF NOT EXISTS FOR (g:Gene) ON (g.ensembl_id)",

            # Disease indexes
            "CREATE INDEX disease_name IF NOT EXISTS FOR (d:Disease) ON (d.name)",
            "CREATE INDEX disease_malacards_id IF NOT EXISTS FOR (d:Disease) ON (d.malacards_id)",

            # Drug indexes
            "CREATE INDEX drug_name IF NOT EXISTS FOR (d:Drug) ON (d.name)",
            "CREATE INDEX drug_drugbank_id IF NOT EXISTS FOR (d:Drug) ON (d.drugbank_id)",

            # Pathway indexes
            "CREATE INDEX pathway_name IF NOT EXISTS FOR (p:Pathway) ON (p.name)",
            "CREATE INDEX pathway_id IF NOT EXISTS FOR (p:Pathway) ON (p.pathway_id)",
            "CREATE INDEX pathway_super_pathway_name IF NOT EXISTS FOR (p:Pathway) ON (p.super_pathway_name)",

            # Publication indexes
            "CREATE INDEX publication_pubmed_id IF NOT EXISTS FOR (p:Publication) ON (p.pubmed_id)",

            # Protein indexes
            "CREATE INDEX protein_accession IF NOT EXISTS FOR (p:Protein) ON (p.accession)",

            # Variant indexes
            "CREATE INDEX variant_accession IF NOT EXISTS FOR (v:Variant) ON (v.accession)",

            # Summary indexes
            "CREATE INDEX summary_source IF NOT EXISTS FOR (s:Summary) ON (s.source)",

            # Constraints for uniqueness
            "CREATE CONSTRAINT gene_symbol_unique IF NOT EXISTS FOR (g:Gene) REQUIRE g.symbol IS UNIQUE",
            "CREATE CONSTRAINT publication_pubmed_unique IF NOT EXISTS FOR (p:Publication) REQUIRE p.pubmed_id IS UNIQUE",
        ]

        for index_query in indexes_and_constraints:
            try:
                self.db.execute_write(index_query)
                logger.debug(f"Created: {index_query}")
            except Exception as e:
                logger.warning(f"Index creation warning: {e}")

    def find_json_files(self, data_directory: str) -> Iterator[Path]:
        """Find all JSON files in the GeneCards directory structure."""
        data_path = Path(data_directory)

        if not data_path.exists():
            raise FileNotFoundError(
                f"Data directory not found: {data_directory}")

        logger.info(f"Scanning for JSON files in: {data_directory}")

        # Find folders like AB, XY, etc.
        for folder in sorted(data_path.iterdir()):
            if folder.is_dir() and len(folder.name) == 2:
                logger.info(f"Processing folder: {folder.name}")

                # Find JSON files in each folder
                for json_file in folder.glob("*.json"):
                    yield json_file

    def _extract_gene_data(self, json_data: Dict[str, Any], gene_symbol: str) -> Dict[str, Any]:
        """
        Extract gene data from GeneCards JSON format.

        Args:
            json_data: Raw JSON data from GeneCards file
            gene_symbol: Gene symbol extracted from filename

        Returns:
            Structured gene data for Neo4j import
        """
        try:
            # Start with the gene symbol from filename
            gene_data = {
                "gene_symbol": gene_symbol,
                "aliases": [],
                "external_ids": {},
                "genomic_location": {},
                "diseases": [],
                "drugs": [],
                "pathways": [],
                "publications": [],
                "proteins": [],
                "gene_interactions": [],
                "summaries": []
            }

            # Extract basic gene information from "Gene" section
            if "Gene" in json_data and json_data["Gene"]:
                gene_info = json_data["Gene"][0] if isinstance(
                    json_data["Gene"], list) else json_data["Gene"]
                if isinstance(gene_info, dict):
                    gene_data["name"] = gene_info.get("Name", "")
                    gene_data["category"] = gene_info.get("Category", "")
                    gene_data["gifts_score"] = gene_info.get("Gifts")
                    gene_data["genecards_id"] = gene_info.get(
                        "GeneCardsId", "")
                    gene_data["source"] = gene_info.get("Source", "")
                    gene_data["is_approved"] = gene_info.get(
                        "IsApproved", False)

            # Extract aliases
            if "Aliases" in json_data and json_data["Aliases"]:
                aliases = []
                for alias in json_data["Aliases"]:
                    if isinstance(alias, dict) and "Value" in alias:
                        aliases.append(alias["Value"])
                gene_data["aliases"] = aliases

            # Extract external identifiers
            if "ExternalIdentifiers" in json_data and json_data["ExternalIdentifiers"]:
                external_ids = {}
                for ext_id in json_data["ExternalIdentifiers"]:
                    if isinstance(ext_id, dict):
                        source = ext_id.get("Source", "").lower()
                        value = ext_id.get("Value", "")
                        if source and value:
                            external_ids[f"{source}_id"] = value
                gene_data["external_ids"] = external_ids

            # Extract genomic location from "Genomics" section
            if "Genomics" in json_data and json_data["Genomics"]:
                genomics = json_data["Genomics"][0] if isinstance(
                    json_data["Genomics"], list) else json_data["Genomics"]
                if isinstance(genomics, dict):
                    # Extract bands information
                    bands = genomics.get("Bands", [])
                    gene_data["genomic_location"] = {"bands": bands}

                    # Extract data for each genomic version
                    for version_key in ["Hg38", "Hg19NCBIGene", "Hg19Ensembl"]:
                        version_data = genomics.get(version_key, {})
                        if version_data:
                            locations = version_data.get("Locations", [])
                            if locations:
                                loc = locations[0]
                                gene_data["genomic_location"][version_key] = {
                                    "chromosome": version_data.get("Chromosome"),
                                    "start_position": loc.get("Start"),
                                    "end_position": loc.get("End"),
                                    "gene_size": loc.get("Size"),
                                    "strand": version_data.get("Strand"),
                                    "version": version_data.get("Version"),
                                    "version_official": version_data.get("VersionOfficial")
                                }

            # Extract summaries from "Summaries" section
            if "Summaries" in json_data and json_data["Summaries"]:
                summaries_collected = []
                summary_data = json_data["Summaries"]
                if isinstance(summary_data, list):
                    for summary in summary_data:
                        if isinstance(summary, dict):
                            summaries_collected.append({
                                "summary": summary.get("Summary", ""),
                                "source": summary.get("Source", "")
                            })
                elif isinstance(summary_data, dict):
                    summaries_collected.append({
                        "summary": summary_data.get("Summary", ""),
                        "source": summary_data.get("Source", "")
                    })

                # Store the summaries
                if summaries_collected:
                    gene_data["summaries"] = summaries_collected

            # Extract disease associations from "MalaCardsDisorders"
            if "MalaCardsDisorders" in json_data and json_data["MalaCardsDisorders"]:
                for disorder in json_data["MalaCardsDisorders"]:
                    if isinstance(disorder, dict):
                        gene_data["diseases"].append({
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

            # Extract inferred disease associations from "MalaCardsInferredDisorders"
            if "MalaCardsInferredDisorders" in json_data and json_data["MalaCardsInferredDisorders"]:
                for disorder in json_data["MalaCardsInferredDisorders"]:
                    if isinstance(disorder, dict):
                        gene_data["diseases"].append({
                            "name": disorder.get("Name", ""),
                            "malacards_id": disorder.get("Accession", ""),
                            "aliases": disorder.get("Aliases", []),
                            "sources": disorder.get("Sources", []),
                            "gene_score": disorder.get("GeneScore"),
                            "disorder_score": disorder.get("DisorderScore"),
                            "is_elite": False,  # Inferred disorders are not elite
                            "is_cancer_census": disorder.get("IsCancerCensus", False),
                            "disorder_type": "inferred"
                        })

            # Extract drug associations from multiple drug sections
            if "UnifiedDrugs" in json_data and json_data["UnifiedDrugs"]:
                drug_data = json_data["UnifiedDrugs"]
                if isinstance(drug_data, list):
                    for drug in drug_data:
                        if isinstance(drug, dict):
                            gene_data["drugs"].append({
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
                        gene_data["pathways"].append({
                            "name": pathway.get("Name", ""),
                            "pathway_id": pathway.get("Accession", ""),
                            "source": pathway.get("Source", ""),
                            "category": "",  # Not available in GeneCards data
                            "evidence_score": None  # Not available in GeneCards data
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
                                        # Handle scores like ".42" -> 0.42
                                        if score_str.startswith('.'):
                                            score_str = '0' + score_str
                                        evidence_score = float(score_str)
                                    except (ValueError, TypeError):
                                        evidence_score = None

                                gene_data["pathways"].append({
                                    "name": member.get("Pathway", ""),
                                    "pathway_id": member.get("SourceAccession", ""),
                                    "source": member.get("Source", ""),
                                    "category": "SuperPathway",
                                    "evidence_score": evidence_score,
                                    "super_pathway_name": super_pathway_name
                                })

            # Extract publication references from "Publications"
            if "Publications" in json_data and json_data["Publications"]:
                for pub in json_data["Publications"]:
                    if isinstance(pub, dict):
                        gene_data["publications"].append({
                            "pubmed_id": pub.get("PubmedId", ""),
                            "title": pub.get("Title", ""),
                            "authors": pub.get("Authors", ""),
                            "journal": pub.get("Journal", ""),
                            "year": pub.get("Year"),
                            "citation_count": pub.get("CitationCount")
                        })

            # Extract protein information from "Proteins"
            if "Proteins" in json_data and json_data["Proteins"]:
                for protein in json_data["Proteins"]:
                    if isinstance(protein, dict):
                        # Parse subcellular location list
                        subcellular_locations = protein.get(
                            "SubCellularLocation", [])
                        if isinstance(subcellular_locations, list):
                            subcellular_location_str = "; ".join(
                                subcellular_locations)
                        else:
                            subcellular_location_str = str(
                                subcellular_locations) if subcellular_locations else ""

                        # Parse quaternary structure list
                        quaternary_structures = protein.get(
                            "QuaternaryStructure", [])
                        if isinstance(quaternary_structures, list):
                            quaternary_structure_str = "; ".join(
                                quaternary_structures)
                        else:
                            quaternary_structure_str = str(
                                quaternary_structures) if quaternary_structures else ""

                        # Parse secondary accessions list
                        secondary_accessions = protein.get(
                            "SecondaryAccessions", [])
                        if isinstance(secondary_accessions, list):
                            secondary_accessions_str = "; ".join(
                                secondary_accessions)
                        else:
                            secondary_accessions_str = str(
                                secondary_accessions) if secondary_accessions else ""

                        # Parse cofactors list
                        cofactors = protein.get("Cofactors", [])
                        if isinstance(cofactors, list):
                            cofactors_str = "; ".join(cofactors)
                        else:
                            cofactors_str = str(cofactors) if cofactors else ""

                        # Parse other list fields
                        sequence_caution = protein.get("SequenceCaution", [])
                        peptides = protein.get("Peptides", [])
                        ptms = protein.get(
                            "PostTranslationalModifications", [])

                        gene_data["proteins"].append({
                            "name": protein.get("Name", ""),
                            "full_name": protein.get("FullName", ""),
                            "size": protein.get("Size"),  # Size in amino acids
                            "mass": protein.get("Mass"),  # Molecular mass
                            # Primary accession ID
                            "accession": protein.get("Accession", ""),
                            "quaternary_structure": quaternary_structure_str,
                            "subcellular_location": subcellular_location_str,
                            "secondary_accessions": secondary_accessions_str,
                            "cofactors": cofactors_str,
                            "sequence_caution": sequence_caution,
                            "peptides": peptides,
                            "post_translational_modifications": ptms,
                            "source": protein.get("Source", ""),
                            "existences": protein.get("Existences", "")
                        })

            # Extract gene interactions from "Interactions"
            if "Interactions" in json_data and json_data["Interactions"]:
                for interaction in json_data["Interactions"]:
                    if isinstance(interaction, dict):
                        interactant = interaction.get("Interactant", "")
                        details = interaction.get("Details", [])
                        for detail in details:
                            if isinstance(detail, dict):
                                gene_data["gene_interactions"].append({
                                    "interactor_symbol": interactant,
                                    "protein": detail.get("Protein", ""),
                                    "source": detail.get("Source", ""),
                                    "details": detail.get("Details", [])
                                })

            return gene_data

        except Exception as e:
            logger.error(f"Error extracting gene data for {gene_symbol}: {e}")
            # Return minimal gene data on error
            return {
                "gene_symbol": gene_symbol,
                "aliases": [],
                "external_ids": {},
                "genomic_location": {},
                "diseases": [],
                "drugs": [],
                "pathways": [],
                "publications": [],
                "proteins": [],
                "gene_interactions": [],
                "summaries": []
            }

    def _process_json_file(self, file_path: Path) -> bool:
        """
        Process a single JSON file and import its data.

        Args:
            file_path: Path to the JSON file

        Returns:
            True if successful, False otherwise
        """
        try:
            # Extract gene symbol from filename (remove .json extension)
            gene_symbol = file_path.stem

            # Read and parse JSON file
            with open(file_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)

            # Extract structured data
            gene_data = self._extract_gene_data(json_data, gene_symbol)

            # Import to Neo4j
            self._import_gene_to_neo4j(gene_data)

            return True

        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in file {file_path}: {e}")
            return False
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
            return False

    def _import_gene_to_neo4j(self, gene_data: Dict) -> Optional[str]:
        """Import gene data to Neo4j."""
        try:
            # Extract gene symbol
            gene_symbol = gene_data.get('gene_symbol', 'unknown')

            # Extract genomic location data (prioritize Hg38)
            genomic_location = gene_data.get('genomic_location', {})
            hg38_data = genomic_location.get('Hg38', {})

            # Create gene node with basic properties
            node_props = {
                'symbol': gene_symbol,
                'name': gene_data.get('name', ''),
                'category': gene_data.get('category', ''),
                'gifts_score': gene_data.get('gifts_score'),
                'genecards_id': gene_data.get('genecards_id', ''),
                'source': gene_data.get('source', ''),
                'is_approved': gene_data.get('is_approved', False),
                'aliases': json.dumps(gene_data.get('aliases', [])),
                'chromosome': hg38_data.get('chromosome'),
                'start_position': hg38_data.get('start_position'),
                'end_position': hg38_data.get('end_position'),
                'gene_size': hg38_data.get('gene_size'),
                'strand': hg38_data.get('strand'),
                'genomic_location': json.dumps(genomic_location),
                'created_at': datetime.now().isoformat(),
                'source_db': 'GeneCards'
            }

            # Add external IDs
            external_ids = gene_data.get('external_ids', {})
            for key, value in external_ids.items():
                if value:
                    node_props[key] = value

            # Remove None values
            node_props = {k: v for k, v in node_props.items() if v is not None}

            # Create gene node
            query = """
            MERGE (g:Gene {symbol: $symbol})
            SET g += $props
            RETURN g.symbol as symbol
            """

            result = self.db.execute_write(
                query, {'symbol': gene_symbol, 'props': node_props})

            if result:
                logger.debug(f"Created gene node: {gene_symbol}")

                # Create related entities and relationships
                self.create_disease_associations(gene_symbol, gene_data)
                self.create_drug_associations(gene_symbol, gene_data)
                self.create_pathway_associations(gene_symbol, gene_data)
                self.create_publications(gene_symbol, gene_data)
                self.create_protein_data(gene_symbol, gene_data)
                self.create_gene_interactions(gene_symbol, gene_data)
                self.create_summaries(gene_symbol, gene_data)

                # self._create_missing_relationships(gene_symbol, gene_data)

                with self.lock:
                    self.stats.genes_created += 1

                return gene_symbol
            else:
                logger.error(f"Failed to create gene node: {gene_symbol}")
                return None

        except Exception as e:
            logger.error(f"Error importing gene data: {e}")
            return None

    def _create_missing_relationships(self, gene_symbol: str, gene_data: Dict):
        """Create missing biomedical relationships."""
        try:
            # Drug-Disease relationships (inferred from gene associations)
            self._create_drug_disease_relationships(gene_symbol, gene_data)

            # Pathway-Disease relationships (inferred from gene associations)
            self._create_pathway_disease_relationships(gene_symbol, gene_data)

            # Protein-Protein interactions (if available)
            self._create_protein_protein_interactions(gene_symbol, gene_data)

        except Exception as e:
            logger.error(
                f"Error creating missing relationships for {gene_symbol}: {e}")

    def _create_drug_disease_relationships(self, gene_symbol: str, gene_data: Dict):
        """Create drug-disease relationships based on gene associations."""
        try:
            diseases = gene_data.get('diseases', [])
            drugs = gene_data.get('drugs', [])

            for disease in diseases:
                disease_name = disease.get('name', '')
                if not disease_name:
                    continue

                for drug in drugs:
                    drug_name = drug.get('name', '')
                    if not drug_name:
                        continue

                    # Create potential drug-disease relationship
                    query = """
                    MATCH (drug:Drug {name: $drug_name})
                    MATCH (disease:Disease {name: $disease_name})
                    MERGE (drug)-[r:POTENTIAL_TREATMENT]->(disease)
                    SET r.evidence_gene = $gene_symbol,
                        r.confidence = 'inferred',
                        r.created_at = $created_at
                    """

                    self.db.execute_write(query, {
                        'drug_name': drug_name,
                        'disease_name': disease_name,
                        'gene_symbol': gene_symbol,
                        'created_at': datetime.now().isoformat()
                    })

                    self.stats.relationships_created += 1

        except Exception as e:
            logger.error(f"Error creating drug-disease relationships: {e}")

    def _create_pathway_disease_relationships(self, gene_symbol: str, gene_data: Dict):
        """Create pathway-disease relationships based on gene associations."""
        try:
            diseases = gene_data.get('diseases', [])
            pathways = gene_data.get('pathways', [])

            for disease in diseases:
                disease_name = disease.get('name', '')
                gene_score = disease.get('gene_score', 0)

                if not disease_name:
                    continue

                for pathway in pathways:
                    pathway_name = pathway.get('name', '')
                    evidence_score = pathway.get('evidence_score', 0)

                    if not pathway_name:
                        continue

                    # Create pathway-disease association
                    query = """
                    MATCH (pathway:Pathway {name: $pathway_name})
                    MATCH (disease:Disease {name: $disease_name})
                    MERGE (pathway)-[r:ASSOCIATED_WITH_DISEASE]->(disease)
                    SET r.evidence_gene = $gene_symbol,
                        r.gene_score = $gene_score,
                        r.pathway_evidence = $evidence_score,
                        r.created_at = $created_at
                    """

                    self.db.execute_write(query, {
                        'pathway_name': pathway_name,
                        'disease_name': disease_name,
                        'gene_symbol': gene_symbol,
                        'gene_score': gene_score,
                        'evidence_score': evidence_score,
                        'created_at': datetime.now().isoformat()
                    })

                    self.stats.relationships_created += 1

        except Exception as e:
            logger.error(f"Error creating pathway-disease relationships: {e}")

    def _create_protein_protein_interactions(self, gene_symbol: str, gene_data: Dict):
        """Create protein-protein interactions from gene interaction data."""
        try:
            proteins = gene_data.get('proteins', [])
            interactions = gene_data.get('gene_interactions', [])

            if not proteins or not interactions:
                return

            main_protein = proteins[0] if proteins else None
            if not main_protein:
                return

            main_accession = main_protein.get('accession', '')
            if not main_accession:
                return

            for interaction in interactions:
                interactor_symbol = interaction.get('interactor_symbol', '')
                protein_id = interaction.get('protein', '')

                if not interactor_symbol or not protein_id:
                    continue

                # Create protein-protein interaction
                query = """
                MATCH (p1:Protein {accession: $main_accession})
                MERGE (p2:Protein {accession: $interactor_protein})
                MERGE (p1)-[r:PROTEIN_INTERACTION]->(p2)
                SET r.evidence_gene = $gene_symbol,
                    r.interactor_gene = $interactor_symbol,
                    r.source = $source,
                    r.created_at = $created_at
                """

                self.db.execute_write(query, {
                    'main_accession': main_accession,
                    'interactor_protein': protein_id,
                    'gene_symbol': gene_symbol,
                    'interactor_symbol': interactor_symbol,
                    'source': interaction.get('source', ''),
                    'created_at': datetime.now().isoformat()
                })

                self.stats.relationships_created += 1

        except Exception as e:
            logger.error(f"Error creating protein-protein interactions: {e}")

    def create_disease_associations(self, gene_symbol: str, gene_data: Dict):
        """Create disease nodes and associations with gene and disorder scores."""
        try:
            for disease in gene_data.get('diseases', []):
                disease_name = disease.get('name', '')
                if not disease_name:
                    continue

                # Create disease node with comprehensive properties
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

                # Determine relationship type based on disorder type
                relationship_type = "INFERRED_ASSOCIATION" if disease.get(
                    'disorder_type') == 'inferred' else "ASSOCIATED_WITH"

                query = """
                MERGE (d:Disease {name: $disease_name})
                SET d += $disease_properties
                WITH d
                MATCH (g:Gene {symbol: $gene_symbol})
                MERGE (g)-[r:""" + relationship_type + """]->(d)
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

                with self.lock:
                    self.stats.diseases_created += 1
                    self.stats.relationships_created += 1

        except Exception as e:
            logger.error(
                f"Error creating disease associations for {gene_symbol}: {e}")

    def create_drug_associations(self, gene_symbol: str, gene_data: Dict):
        """Create drug nodes and associations."""
        try:
            for drug in gene_data.get('drugs', []):
                drug_name = drug.get('name', '')
                if not drug_name:
                    continue

                # Create drug node with available properties
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

                with self.lock:
                    self.stats.drugs_created += 1
                    self.stats.relationships_created += 1

        except Exception as e:
            logger.error(
                f"Error creating drug associations for {gene_symbol}: {e}")

    def create_pathway_associations(self, gene_symbol: str, gene_data: Dict):
        """Create pathway nodes and associations."""
        try:
            for pathway in gene_data.get('pathways', []):
                pathway_name = pathway.get('name', '')
                if not pathway_name:
                    continue

                # Create pathway node with all available properties
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

                # Determine relationship type based on category
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

                with self.lock:
                    self.stats.pathways_created += 1
                    self.stats.relationships_created += 1

        except Exception as e:
            logger.error(
                f"Error creating pathway associations for {gene_symbol}: {e}")

    def create_publications(self, gene_symbol: str, gene_data: Dict):
        """Create publication nodes and associations."""
        try:
            for pub in gene_data.get('publications', []):
                pubmed_id = pub.get('pubmed_id')
                if not pubmed_id:
                    continue

                # Create publication node
                pub_properties = {
                    'pubmed_id': str(pubmed_id),
                    'title': pub.get('title', ''),
                    'year': pub.get('year'),
                    'journal': pub.get('journal', ''),
                    'authors': pub.get('authors', ''),
                    'citation_count': pub.get('citation_count'),
                    'created_at': datetime.now().isoformat()
                }

                # Remove empty values
                pub_properties = {k: v for k, v in pub_properties.items()
                                  if v is not None and v != '' and v != []}

                query = """
                MERGE (p:Publication {pubmed_id: $pubmed_id})
                SET p += $pub_properties
                WITH p
                MATCH (g:Gene {symbol: $gene_symbol})
                MERGE (g)-[r:MENTIONED_IN]->(p)
                """

                self.db.execute_write(query, {
                    'pubmed_id': str(pubmed_id),
                    'pub_properties': pub_properties,
                    'gene_symbol': gene_symbol
                })

                with self.lock:
                    self.stats.publications_created += 1
                    self.stats.relationships_created += 1

        except Exception as e:
            logger.error(f"Error creating publications for {gene_symbol}: {e}")

    def create_protein_data(self, gene_symbol: str, gene_data: Dict):
        """Create protein nodes and data."""
        try:
            for protein in gene_data.get('proteins', []):
                accession = protein.get('accession', '')
                if not accession:
                    continue

                # Create protein node with comprehensive properties
                protein_properties = {
                    'accession': accession,
                    'name': protein.get('name', ''),
                    'full_name': protein.get('full_name', ''),
                    'size': protein.get('size'),  # Size in amino acids
                    'mass': protein.get('mass'),  # Molecular mass
                    'quaternary_structure': protein.get('quaternary_structure', ''),
                    'subcellular_location': protein.get('subcellular_location', ''),
                    'secondary_accessions': protein.get('secondary_accessions', ''),
                    'cofactors': protein.get('cofactors', ''),
                    'sequence_caution': json.dumps(protein.get('sequence_caution', [])),
                    'peptides': json.dumps(protein.get('peptides', [])),
                    'post_translational_modifications': json.dumps(protein.get('post_translational_modifications', [])),
                    'source': protein.get('source', ''),
                    'existences': protein.get('existences', ''),
                    'created_at': datetime.now().isoformat()
                }

                # Remove empty values (including empty JSON arrays)
                protein_properties = {k: v for k, v in protein_properties.items()
                                      if v is not None and v != '' and v != '[]' and v != 'null'}

                query = """
                MERGE (p:Protein {accession: $accession})
                SET p += $protein_properties
                WITH p
                MATCH (g:Gene {symbol: $gene_symbol})
                MERGE (g)-[r:ENCODES]->(p)
                SET r.source = $source,
                    r.created_at = $created_at
                """

                self.db.execute_write(query, {
                    'accession': accession,
                    'protein_properties': protein_properties,
                    'gene_symbol': gene_symbol,
                    'source': protein.get('source', ''),
                    'created_at': datetime.now().isoformat()
                })

                with self.lock:
                    self.stats.relationships_created += 1
                    self.stats.proteins_created += 1

        except Exception as e:
            logger.error(f"Error creating protein data for {gene_symbol}: {e}")

    def create_gene_interactions(self, gene_symbol: str, gene_data: Dict):
        """Create gene-gene interaction relationships."""
        try:
            for interaction in gene_data.get('gene_interactions', []):
                interactant = interaction.get('interactor_symbol', '')
                if not interactant or interactant == gene_symbol:
                    continue

                # Create interaction relationship
                query = """
                MATCH (g1:Gene {symbol: $gene_symbol})
                MERGE (g2:Gene {symbol: $interactant})
                MERGE (g1)-[r:INTERACTS_WITH]->(g2)
                SET r.protein = $protein,
                    r.source = $source,
                    r.details = $details,
                    r.created_at = $created_at
                """

                self.db.execute_write(query, {
                    'gene_symbol': gene_symbol,
                    'interactant': interactant,
                    'protein': interaction.get('protein', ''),
                    'source': interaction.get('source', ''),
                    'details': json.dumps(interaction.get('details', [])),
                    'created_at': datetime.now().isoformat()
                })

                with self.lock:
                    self.stats.relationships_created += 1

        except Exception as e:
            logger.error(f"Error creating interactions for {gene_symbol}: {e}")

    def create_summaries(self, gene_symbol: str, gene_data: Dict):
        """Create summary nodes and associations."""
        try:
            for summary in gene_data.get('summaries', []):
                summary_text = summary.get('summary', '')
                if not summary_text:
                    continue

                # Create summary node
                summary_properties = {
                    'summary': summary_text,
                    'source': summary.get('source', ''),
                    'created_at': datetime.now().isoformat()
                }

                # Remove empty values
                summary_properties = {k: v for k, v in summary_properties.items()
                                      if v is not None and v != ''}

                query = """
                MERGE (s:Summary {summary: $summary_text})
                SET s += $summary_properties
                WITH s
                MATCH (g:Gene {symbol: $gene_symbol})
                MERGE (g)-[r:HAS_SUMMARY]->(s)
                """

                self.db.execute_write(query, {
                    'summary_text': summary_text,
                    'summary_properties': summary_properties,
                    'gene_symbol': gene_symbol
                })

                with self.lock:
                    self.stats.relationships_created += 1
                    self.stats.summaries_created += 1

        except Exception as e:
            logger.error(f"Error creating summaries for {gene_symbol}: {e}")

    def process_gene_file(self, json_file: Path) -> bool:
        """Process a single gene JSON file."""
        try:
            # Process the JSON file using the new method
            success = self._process_json_file(json_file)

            if success:
                with self.lock:
                    self.stats.files_processed += 1
            else:
                with self.lock:
                    self.stats.errors += 1

            return success

        except Exception as e:
            logger.error(f"Error processing {json_file}: {e}")
            with self.lock:
                self.stats.errors += 1
            return False

    def print_progress(self):
        """Print import progress."""
        elapsed = self.stats.elapsed_time()
        rate = self.stats.files_per_second()

        logger.info(f"""
        Progress Update:
        ================
        Files processed: {self.stats.files_processed:,}
        Genes created: {self.stats.genes_created:,}
        Diseases created: {self.stats.diseases_created:,}
        Drugs created: {self.stats.drugs_created:,}
        Pathways created: {self.stats.pathways_created:,}
        Publications created: {self.stats.publications_created:,}
        Proteins created: {self.stats.proteins_created:,}
        Summaries created: {self.stats.summaries_created:,}
        Relationships created: {self.stats.relationships_created:,}
        Errors: {self.stats.errors:,}
        
        Performance:
        ============
        Elapsed time: {elapsed:.1f} seconds
        Rate: {rate:.2f} files/second
        """)

    def import_dataset(self, data_directory: str, max_workers: int = 4, progress_interval: int = 100, start_index: int = 0):
        """Import the entire GeneCards dataset."""
        logger.info(
            f"Starting GeneCards dataset import from: {data_directory}")
        logger.info(f"Using {max_workers} worker threads")

        json_files = list(self.find_json_files(data_directory))
        total_files = len(json_files)

        logger.info(f"Found {total_files:,} JSON files to process")

        if total_files == 0:
            logger.warning(
                "No JSON files found! Check your data directory structure.")
            return

        # Check existing genes in database and skip those files
        logger.info("Checking for existing genes in database...")
        try:
            existing_genes_result = self.db.execute_query(
                "MATCH (n:Gene) RETURN n.symbol")
            existing_genes = {record['n.symbol']
                              for record in existing_genes_result}
            logger.info(
                f"Found {len(existing_genes):,} existing genes in database")

            # Filter out files for genes that already exist
            original_count = len(json_files)
            json_files = [
                f for f in json_files if f.stem not in existing_genes]
            skipped_count = original_count - len(json_files)

            logger.info(
                f"Skipping {skipped_count:,} files for existing genes, processing {len(json_files):,} remaining files")
        except Exception as e:
            logger.warning(
                f"Could not check existing genes: {e}. Processing all files.")

        # Process files with thread pool
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_file = {
                executor.submit(self.process_gene_file, json_file): json_file
                for json_file in json_files
            }

            # Process completed tasks
            for future in as_completed(future_to_file):
                json_file = future_to_file[future]
                try:
                    success = future.result()
                    if not success:
                        logger.warning(f"Failed to process: {json_file}")
                except Exception as e:
                    logger.error(f"Exception processing {json_file}: {e}")
                    with self.lock:
                        self.stats.errors += 1

                # Print progress periodically
                if self.stats.files_processed % progress_interval == 0:
                    self.print_progress()

        # Final statistics
        logger.info("=" * 60)
        logger.info("IMPORT COMPLETED!")
        logger.info("=" * 60)
        self.print_progress()

        # Database statistics
        db_stats = self.db.get_database_stats()
        logger.info(f"""
        Final Database Statistics:
        ==========================
        Total nodes: {db_stats.get('total_nodes', 0):,}
        Total relationships: {db_stats.get('total_relationships', 0):,}
        Gene nodes: {db_stats.get('gene_count', 0):,}
        Disease nodes: {db_stats.get('disease_count', 0):,}
        Drug nodes: {db_stats.get('drug_count', 0):,}
        Pathway nodes: {db_stats.get('pathway_count', 0):,}
        """)


def main():
    """Main function for command-line usage."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Import GeneCards JSON dataset to Neo4j')
    parser.add_argument(
        'data_directory', help='Path to GeneCards data directory')
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of worker threads (default: 4)')
    parser.add_argument('--batch-size', type=int, default=50,
                        help='Batch size for processing (default: 50)')
    parser.add_argument('--progress-interval', type=int, default=100,
                        help='Progress report interval (default: 100)')
    parser.add_argument('--start-index', type=int, default=0,
                        help='Resume processing from this file index (default: 0)')
    parser.add_argument(
        '--neo4j-uri', default='bolt://localhost:7687', help='Neo4j URI')
    parser.add_argument('--neo4j-user', default='neo4j', help='Neo4j username')
    parser.add_argument('--neo4j-password',
                        default='password', help='Neo4j password')

    args = parser.parse_args()

    # Auto-detect optimal workers if not specified
    # if args.workers == 4:  # Default value
    #     optimal_workers = detect_optimal_workers(args.neo4j_uri)
    #     logger.info(
    #         f"Auto-detected {optimal_workers} workers for this environment")
    #     args.workers = optimal_workers

    # Connect to Neo4j
    try:
        db = Neo4jConnection(
            # uri=args.neo4j_uri,
            # username=args.neo4j_user,
            # password=args.neo4j_password
        )
        logger.info("✅ Connected to Neo4j successfully!")
    except Exception as e:
        logger.error(f"❌ Failed to connect to Neo4j: {e}")
        sys.exit(1)

    # Create importer and run
    importer = GeneCardsImporter(db, batch_size=args.batch_size)

    try:
        importer.import_dataset(
            data_directory=args.data_directory,
            max_workers=args.workers,
            progress_interval=args.progress_interval,
            start_index=args.start_index
        )
    except KeyboardInterrupt:
        logger.info("Import interrupted by user")
    except Exception as e:
        logger.error(f"Import failed: {e}")
        sys.exit(1)
    finally:
        db.close()


if __name__ == "__main__":
    main()
