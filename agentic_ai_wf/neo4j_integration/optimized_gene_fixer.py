#!/usr/bin/env python3
"""
Optimized Gene Fixer for NULL Aliases

Streamlined version that fixes genes with NULL aliases without
index creation conflicts and deadlock issues.
"""

import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

from .connection import Neo4jConnection

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class OptimizedGeneFixer:
    """Optimized fixer for genes with NULL aliases."""

    def __init__(self, neo4j_connection: Neo4jConnection):
        self.db = neo4j_connection
        self.processed_count = 0
        self.success_count = 0
        self.error_count = 0
        self.missing_files_count = 0

    def get_genes_with_null_aliases(self) -> List[str]:
        """Get gene symbols that have NULL aliases."""
        logger.info("Querying genes with NULL aliases...")

        query = """
        MATCH (g:Gene)
        WHERE g.aliases IS NULL
        RETURN g.symbol as symbol
        ORDER BY g.symbol
        """

        try:
            result = self.db.execute_query(query)
            gene_symbols = [record['symbol'] for record in result]
            logger.info(f"Found {len(gene_symbols):,} genes with NULL aliases")
            return gene_symbols
        except Exception as e:
            logger.error(f"Error querying genes with NULL aliases: {e}")
            return []

    def find_json_files_for_genes(self, data_directory: str, gene_symbols: List[str]) -> Dict[str, Optional[Path]]:
        """Find JSON files corresponding to specific gene symbols."""
        logger.info(
            f"Searching for JSON files for {len(gene_symbols):,} genes...")

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

        files_missing = len(gene_symbols) - files_found
        logger.info(
            f"Found files for {files_found:,} genes, {files_missing:,} genes have no corresponding files")

        return gene_to_file  # type: ignore

    def extract_gene_data_from_file(self, json_file: Path, gene_symbol: str) -> Optional[Dict]:
        """Extract gene data from JSON file."""
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                json_data = json.load(f)

            # Extract basic gene information
            gene_data = {
                "gene_symbol": gene_symbol,
                "aliases": [],
                "external_ids": {},
                "name": "",
                "category": "",
                "gifts_score": None,
                "genecards_id": "",
                "source": "",
                "is_approved": False
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

            return gene_data

        except Exception as e:
            logger.error(f"Error extracting data from {json_file}: {e}")
            return None

    def update_gene_properties(self, gene_symbol: str, gene_data: Dict) -> bool:
        """Update gene properties without recreating the entire node."""
        try:
            # Prepare update properties
            update_props = {
                'name': gene_data.get('name', ''),
                'category': gene_data.get('category', ''),
                'gifts_score': gene_data.get('gifts_score'),
                'genecards_id': gene_data.get('genecards_id', ''),
                'source': gene_data.get('source', ''),
                'is_approved': gene_data.get('is_approved', False),
                'aliases': json.dumps(gene_data.get('aliases', [])),
                'updated_at': datetime.now().isoformat()
            }

            # Add external IDs
            external_ids = gene_data.get('external_ids', {})
            for key, value in external_ids.items():
                if value:
                    update_props[key] = value

            # Remove None values
            update_props = {k: v for k,
                            v in update_props.items() if v is not None}

            # Update the gene node
            query = """
            MATCH (g:Gene {symbol: $gene_symbol})
            SET g += $props
            RETURN g.symbol as symbol
            """

            result = self.db.execute_write(query, {
                'gene_symbol': gene_symbol,
                'props': update_props
            })

            if result:
                logger.debug(f"Updated gene: {gene_symbol}")
                return True
            else:
                logger.error(f"Failed to update gene: {gene_symbol}")
                return False

        except Exception as e:
            logger.error(f"Error updating gene {gene_symbol}: {e}")
            return False

    def fix_genes_batch(self, genes_with_files: Dict[str, Path], batch_size: int = 10):
        """Process genes in batches to avoid memory issues."""
        genes_list = list(genes_with_files.items())
        total_batches = (len(genes_list) + batch_size - 1) // batch_size

        logger.info(
            f"Processing {len(genes_list):,} genes in {total_batches} batches of {batch_size}")

        for i in range(0, len(genes_list), batch_size):
            batch = genes_list[i:i + batch_size]
            batch_num = (i // batch_size) + 1

            logger.info(
                f"Processing batch {batch_num}/{total_batches} ({len(batch)} genes)")

            for gene_symbol, json_file in batch:
                try:
                    # Extract gene data
                    gene_data = self.extract_gene_data_from_file(
                        json_file, gene_symbol)
                    if not gene_data:
                        self.error_count += 1
                        continue

                    # Update gene properties
                    success = self.update_gene_properties(
                        gene_symbol, gene_data)

                    if success:
                        self.success_count += 1
                    else:
                        self.error_count += 1

                    self.processed_count += 1

                    # Progress update
                    if self.processed_count % 50 == 0:
                        logger.info(
                            f"Progress: {self.processed_count:,} processed, {self.success_count:,} success, {self.error_count:,} errors")

                except Exception as e:
                    logger.error(f"Error processing {gene_symbol}: {e}")
                    self.error_count += 1
                    self.processed_count += 1

            # Small delay between batches to reduce database pressure
            time.sleep(0.1)

    def fix_genes_with_null_aliases(self, data_directory: str, batch_size: int = 10):
        """Fix all genes with NULL aliases using batch processing."""
        logger.info("=" * 60)
        logger.info("OPTIMIZED GENE FIXING - NULL ALIASES")
        logger.info("=" * 60)

        # Step 1: Get genes with NULL aliases
        target_genes = self.get_genes_with_null_aliases()
        if not target_genes:
            logger.info("No genes with NULL aliases found. Nothing to fix.")
            return

        # Step 2: Find corresponding JSON files
        gene_to_file = self.find_json_files_for_genes(
            data_directory, target_genes)

        # Step 3: Separate genes with and without files
        genes_with_files = {gene: file_path for gene,
                            file_path in gene_to_file.items() if file_path is not None}
        genes_without_files = {
            gene for gene, file_path in gene_to_file.items() if file_path is None}

        logger.info(f"Genes with files to process: {len(genes_with_files):,}")
        logger.info(
            f"Genes without files (will be skipped): {len(genes_without_files):,}")

        if genes_without_files:
            logger.info("Sample genes without files (first 5):")
            for gene in list(genes_without_files)[:5]:
                logger.info(f"  - {gene}")

        if not genes_with_files:
            logger.warning("No files found for any genes. Nothing to process.")
            return

        # Step 4: Process genes in batches
        self.missing_files_count = len(genes_without_files)
        self.fix_genes_batch(genes_with_files, batch_size)

        # Final results
        logger.info("=" * 60)
        logger.info("FIXING COMPLETED!")
        logger.info("=" * 60)
        self.print_final_results()

    def print_final_results(self):
        """Print final results."""
        total_target_genes = self.processed_count + self.missing_files_count

        logger.info(f"""
        Final Results:
        ==============
        Total target genes: {total_target_genes:,}
        Genes processed: {self.processed_count:,}
        Successfully fixed: {self.success_count:,}
        Processing errors: {self.error_count:,}
        Missing files (skipped): {self.missing_files_count:,}
        
        Success rate: {(self.success_count/self.processed_count*100):.1f}% (of processed)
        """)

    def verify_fix(self) -> Dict[str, int]:
        """Verify that the fix worked."""
        logger.info("Verifying fix...")

        null_count_query = "MATCH (g:Gene) WHERE g.aliases IS NULL RETURN COUNT(g) as count"
        non_null_count_query = "MATCH (g:Gene) WHERE g.aliases IS NOT NULL RETURN COUNT(g) as count"

        try:
            null_result = self.db.execute_query(null_count_query)
            non_null_result = self.db.execute_query(non_null_count_query)

            null_count = null_result[0]['count'] if null_result else 0
            non_null_count = non_null_result[0]['count'] if non_null_result else 0

            logger.info(f"Verification Results:")
            logger.info(f"  Genes with NULL aliases: {null_count:,}")
            logger.info(f"  Genes with non-NULL aliases: {non_null_count:,}")
            logger.info(f"  Total genes: {null_count + non_null_count:,}")

            return {
                'null_aliases': null_count,
                'non_null_aliases': non_null_count,
                'total_genes': null_count + non_null_count
            }

        except Exception as e:
            logger.error(f"Error verifying fix: {e}")
            return {}


def main():
    """Main function."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Fix genes with NULL aliases (optimized)')
    parser.add_argument(
        'data_directory', help='Path to GeneCards data directory')
    parser.add_argument('--batch-size', type=int, default=10,
                        help='Batch size for processing (default: 10)')
    parser.add_argument('--verify-only', action='store_true',
                        help='Only verify current state, do not fix')

    args = parser.parse_args()

    # Connect to Neo4j
    try:
        db = Neo4jConnection()
        logger.info("✅ Connected to Neo4j successfully!")
    except Exception as e:
        logger.error(f"❌ Failed to connect to Neo4j: {e}")
        sys.exit(1)

    # Create fixer and run
    fixer = OptimizedGeneFixer(db)

    try:
        if args.verify_only:
            fixer.verify_fix()
        else:
            fixer.fix_genes_with_null_aliases(
                data_directory=args.data_directory,
                batch_size=args.batch_size
            )
            fixer.verify_fix()

    except KeyboardInterrupt:
        logger.info("Fix interrupted by user")
    except Exception as e:
        logger.error(f"Fix failed: {e}")
        sys.exit(1)
    finally:
        db.close()


if __name__ == "__main__":
    main()
