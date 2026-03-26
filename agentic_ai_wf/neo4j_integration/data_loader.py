"""
Data Loader for Neo4j Integration

Handles loading biomedical data from various sources (CSV, JSON, etc.)
into the Neo4j graph database.
"""

import pandas as pd
import json
import logging
from pathlib import Path
from typing import Dict, Optional, Any, Union, cast
from agentic_ai_wf.neo4j_integration.connection import Neo4jConnection
from agentic_ai_wf.neo4j_integration.models import GeneNode, PathwayNode, DrugNode, DiseaseNode, create_relationship, RelationshipTypes

logger = logging.getLogger(__name__)


class DataLoader:
    """
    Load biomedical data into Neo4j from various sources.
    """

    def __init__(self, connection: Neo4jConnection):
        self.db = connection

    def load_genes_from_csv(self, csv_path: Union[str, Path],
                            batch_size: int = 1000) -> Dict[str, int]:
        """
        Load gene data from CSV file.

        Expected columns:
        - symbol (required)
        - ensembl_id, entrez_id, description, chromosome, etc. (optional)
        - log2_fold_change, p_value, adjusted_p_value (optional for expression data)

        Args:
            csv_path: Path to CSV file
            batch_size: Number of records to process at once

        Returns:
            Dictionary with loading statistics
        """
        try:
            df = pd.read_csv(csv_path)
            logger.info(f"Loading {len(df)} genes from {csv_path}")

            if 'symbol' not in df.columns:
                raise ValueError("CSV must contain 'symbol' column")

            # Process in batches
            total_created = 0
            total_errors = 0

            for i in range(0, len(df), batch_size):
                batch = df.iloc[i:i+batch_size]
                logger.info(
                    f"Processing batch {i//batch_size + 1}/{(len(df)-1)//batch_size + 1}")

                for _, row in batch.iterrows():
                    try:
                        gene = GeneNode(
                            symbol=row['symbol'],
                            ensembl_id=row.get('ensembl_id'),
                            entrez_id=row.get('entrez_id'),
                            description=row.get('description'),
                            chromosome=row.get('chromosome'),
                            start_position=row.get('start_position'),
                            end_position=row.get('end_position'),
                            gene_type=row.get('gene_type'),
                            log2_fold_change=row.get('log2_fold_change'),
                            p_value=row.get('p_value'),
                            adjusted_p_value=row.get('adjusted_p_value')
                        )

                        query, params = gene.to_cypher_merge()
                        self.db.execute_write(query, params)
                        total_created += 1

                    except Exception as e:
                        logger.warning(
                            f"Error loading gene {row.get('symbol', 'unknown')}: {e}")
                        total_errors += 1

            logger.info(
                f"Gene loading complete: {total_created} created, {total_errors} errors")
            return {"created": total_created, "errors": total_errors}

        except Exception as e:
            logger.error(f"Failed to load genes from {csv_path}: {e}")
            raise

    def load_pathways_from_csv(self, csv_path: Union[str, Path],
                               batch_size: int = 500) -> Dict[str, int]:
        """
        Load pathway data from CSV file.

        Expected columns:
        - pathway_id (required)
        - name, source (required)
        - description, category, p_value, enrichment_score, etc. (optional)

        Args:
            csv_path: Path to CSV file
            batch_size: Number of records to process at once

        Returns:
            Dictionary with loading statistics
        """
        try:
            df = pd.read_csv(csv_path)
            logger.info(f"Loading {len(df)} pathways from {csv_path}")

            required_cols = ['pathway_id', 'name', 'source']
            missing_cols = [
                col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"CSV must contain columns: {missing_cols}")

            total_created = 0
            total_errors = 0

            for i in range(0, len(df), batch_size):
                batch = df.iloc[i:i+batch_size]

                for _, row in batch.iterrows():
                    try:
                        pathway = PathwayNode(
                            pathway_id=row['pathway_id'],
                            name=row['name'],
                            source=row['source'],
                            description=row.get('description'),
                            category=row.get('category'),
                            p_value=row.get('p_value'),
                            adjusted_p_value=row.get('adjusted_p_value'),
                            enrichment_score=row.get('enrichment_score'),
                            gene_count=row.get('gene_count')
                        )

                        query, params = pathway.to_cypher_merge()
                        self.db.execute_write(query, params)
                        total_created += 1

                    except Exception as e:
                        logger.warning(
                            f"Error loading pathway {row.get('pathway_id', 'unknown')}: {e}")
                        total_errors += 1

            logger.info(
                f"Pathway loading complete: {total_created} created, {total_errors} errors")
            return {"created": total_created, "errors": total_errors}

        except Exception as e:
            logger.error(f"Failed to load pathways from {csv_path}: {e}")
            raise

    def load_drugs_from_csv(self, csv_path: Union[str, Path],
                            batch_size: int = 500) -> Dict[str, int]:
        """
        Load drug data from CSV file.

        Expected columns:
        - name (required)
        - drug_id, drugbank_id, mechanism_of_action, etc. (optional)

        Args:
            csv_path: Path to CSV file
            batch_size: Number of records to process at once

        Returns:
            Dictionary with loading statistics
        """
        try:
            df = pd.read_csv(csv_path)
            logger.info(f"Loading {len(df)} drugs from {csv_path}")

            if 'name' not in df.columns:
                raise ValueError("CSV must contain 'name' column")

            total_created = 0
            total_errors = 0

            for i in range(0, len(df), batch_size):
                batch = df.iloc[i:i+batch_size]

                for _, row in batch.iterrows():
                    try:
                        # Handle list fields
                        brand_names = None
                        if 'brand_names' in row and pd.notna(row['brand_names']):
                            brand_names = str(row['brand_names']).split(',')

                        target_genes = None
                        if 'target_genes' in row and pd.notna(row['target_genes']):
                            target_genes = str(row['target_genes']).split(',')

                        drug = DrugNode(
                            name=row['name'],
                            drug_id=row.get('drug_id'),
                            drugbank_id=row.get('drugbank_id'),
                            chembl_id=row.get('chembl_id'),
                            mechanism_of_action=row.get('mechanism_of_action'),
                            drug_class=row.get('drug_class'),
                            fda_approved=row.get('fda_approved'),
                            clinical_trial_phase=row.get(
                                'clinical_trial_phase'),
                            brand_names=brand_names,
                            target_genes=target_genes
                        )

                        query, params = drug.to_cypher_merge()
                        self.db.execute_write(query, params)
                        total_created += 1

                    except Exception as e:
                        logger.warning(
                            f"Error loading drug {row.get('name', 'unknown')}: {e}")
                        total_errors += 1

            logger.info(
                f"Drug loading complete: {total_created} created, {total_errors} errors")
            return {"created": total_created, "errors": total_errors}

        except Exception as e:
            logger.error(f"Failed to load drugs from {csv_path}: {e}")
            raise

    def create_gene_pathway_relationships(self, csv_path: Union[str, Path]) -> Dict[str, int]:
        """
        Create relationships between genes and pathways from CSV.

        Expected columns:
        - gene_symbol
        - pathway_id
        - relationship_properties (optional JSON string)

        Args:
            csv_path: Path to CSV file with gene-pathway mappings

        Returns:
            Dictionary with relationship creation statistics
        """
        try:
            df = pd.read_csv(csv_path)
            logger.info(
                f"Creating {len(df)} gene-pathway relationships from {csv_path}")

            required_cols = ['gene_symbol', 'pathway_id']
            missing_cols = [
                col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"CSV must contain columns: {missing_cols}")

            total_created = 0
            total_errors = 0

            for _, row in df.iterrows():
                try:
                    properties = {}
                    if 'relationship_properties' in row and pd.notna(row['relationship_properties']).any():
                        properties = json.loads(
                            cast(str, row['relationship_properties']))

                    query, params = create_relationship(
                        from_node_label="Gene",
                        from_property="symbol",
                        from_value=str(row['gene_symbol']),
                        to_node_label="Pathway",
                        to_property="pathway_id",
                        to_value=str(row['pathway_id']),
                        relationship_type=RelationshipTypes.BELONGS_TO,
                        properties=properties
                    )

                    self.db.execute_write(query, params)
                    total_created += 1

                except Exception as e:
                    logger.warning(
                        f"Error creating relationship {row.get('gene_symbol', 'unknown')} -> {row.get('pathway_id', 'unknown')}: {e}")
                    total_errors += 1

            logger.info(
                f"Gene-pathway relationship creation complete: {total_created} created, {total_errors} errors")
            return {"created": total_created, "errors": total_errors}

        except Exception as e:
            logger.error(
                f"Failed to create gene-pathway relationships from {csv_path}: {e}")
            raise

    def create_drug_gene_relationships(self, csv_path: Union[str, Path]) -> Dict[str, int]:
        """
        Create relationships between drugs and genes from CSV.

        Expected columns:
        - drug_name
        - gene_symbol
        - relationship_properties (optional JSON string)

        Args:
            csv_path: Path to CSV file with drug-gene mappings

        Returns:
            Dictionary with relationship creation statistics
        """
        try:
            df = pd.read_csv(csv_path)
            logger.info(
                f"Creating {len(df)} drug-gene relationships from {csv_path}")

            required_cols = ['drug_name', 'gene_symbol']
            missing_cols = [
                col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"CSV must contain columns: {missing_cols}")

            total_created = 0
            total_errors = 0

            for _, row in df.iterrows():
                try:
                    properties = {}
                    if 'relationship_properties' in row and pd.notna(row['relationship_properties']).any():
                        properties = json.loads(
                            cast(str, row['relationship_properties']))

                    query, params = create_relationship(
                        from_node_label="Drug",
                        from_property="name",
                        from_value=cast(str, row['drug_name']),
                        to_node_label="Gene",
                        to_property="symbol",
                        to_value=cast(str, row['gene_symbol']),
                        relationship_type=RelationshipTypes.TARGETS,
                        properties=properties
                    )

                    self.db.execute_write(query, params)
                    total_created += 1

                except Exception as e:
                    logger.warning(
                        f"Error creating relationship {row.get('drug_name', 'unknown')} -> {row.get('gene_symbol', 'unknown')}: {e}")
                    total_errors += 1

            logger.info(
                f"Drug-gene relationship creation complete: {total_created} created, {total_errors} errors")
            return {"created": total_created, "errors": total_errors}

        except Exception as e:
            logger.error(
                f"Failed to create drug-gene relationships from {csv_path}: {e}")
            raise

    def load_from_workflow_results(self,
                                   genes_csv: Optional[str] = None,
                                   pathways_csv: Optional[str] = None,
                                   drugs_csv: Optional[str] = None,
                                   disease_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Load data from agentic AI workflow results.

        Args:
            genes_csv: Path to DEG analysis results
            pathways_csv: Path to pathway enrichment results  
            drugs_csv: Path to drug discovery results
            disease_name: Name of the disease being analyzed

        Returns:
            Dictionary with loading statistics for all data types
        """
        results = {}

        # Create disease node if provided
        if disease_name:
            try:
                disease = DiseaseNode(name=disease_name)
                query, params = disease.to_cypher_merge()
                self.db.execute_write(query, params)
                results['disease'] = {"created": 1, "errors": 0}
                logger.info(f"Created disease node: {disease_name}")
            except Exception as e:
                logger.error(f"Failed to create disease node: {e}")
                results['disease'] = {"created": 0, "errors": 1}

        # Load genes
        if genes_csv:
            try:
                results['genes'] = self.load_genes_from_csv(genes_csv)

                # Create gene-disease relationships if disease was provided
                if disease_name:
                    self._create_gene_disease_relationships(
                        genes_csv, disease_name)

            except Exception as e:
                logger.error(f"Failed to load genes: {e}")
                results['genes'] = {"created": 0, "errors": 1}

        # Load pathways
        if pathways_csv:
            try:
                results['pathways'] = self.load_pathways_from_csv(pathways_csv)
            except Exception as e:
                logger.error(f"Failed to load pathways: {e}")
                results['pathways'] = {"created": 0, "errors": 1}

        # Load drugs
        if drugs_csv:
            try:
                results['drugs'] = self.load_drugs_from_csv(drugs_csv)
            except Exception as e:
                logger.error(f"Failed to load drugs: {e}")
                results['drugs'] = {"created": 0, "errors": 1}

        return results

    def _create_gene_disease_relationships(self, genes_csv: str, disease_name: str):
        """Create relationships between genes and disease."""
        try:
            df = pd.read_csv(genes_csv)
            total_created = 0

            for _, row in df.iterrows():
                if 'symbol' in row:
                    properties = {}
                    if 'log2_fold_change' in row:
                        properties['log2_fold_change'] = row['log2_fold_change']
                    if 'p_value' in row:
                        properties['p_value'] = row['p_value']

                    query, params = create_relationship(
                        from_node_label="Gene",
                        from_property="symbol",
                        from_value=str(row['symbol']),
                        to_node_label="Disease",
                        to_property="name",
                        to_value=disease_name,
                        relationship_type=RelationshipTypes.ASSOCIATED_WITH,
                        properties=properties
                    )

                    self.db.execute_write(query, params)
                    total_created += 1

            logger.info(f"Created {total_created} gene-disease relationships")

        except Exception as e:
            logger.error(f"Failed to create gene-disease relationships: {e}")
            raise
