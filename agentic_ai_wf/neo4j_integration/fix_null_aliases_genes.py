#!/usr/bin/env python3
"""
Fix Genes with NULL Aliases

Identifies genes with NULL aliases, finds their corresponding JSON files,
and re-imports them with complete property overwrite to fix missing data.
"""

import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Set, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from .connection import Neo4jConnection
from .genecards_importer import GeneCardsImporter

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class NullAliasesGenesFixer:
    """Fix genes with NULL aliases by re-importing their data."""

    def __init__(self, neo4j_connection: Neo4jConnection):
        self.db = neo4j_connection
        self.lock = threading.Lock()
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

        # Create a set for faster lookup
        target_genes = set(gene_symbols)
        gene_to_file = {}

        # Initialize all genes as not found
        for gene in gene_symbols:
            gene_to_file[gene] = None

        files_found = 0

        # Search through directory structure
        for folder in sorted(data_path.iterdir()):
            if folder.is_dir() and len(folder.name) == 2:
                for json_file in folder.glob("*.json"):
                    gene_symbol = json_file.stem
                    if gene_symbol in target_genes:
                        gene_to_file[gene_symbol] = json_file
                        files_found += 1

                        if files_found % 100 == 0:
                            logger.info(f"Found {files_found} files so far...")

        files_missing = len(gene_symbols) - files_found
        logger.info(
            f"Found files for {files_found:,} genes, {files_missing:,} genes have no corresponding files")

        return gene_to_file

    def delete_gene_completely(self, gene_symbol: str) -> bool:
        """Delete a gene and all its relationships completely."""
        try:
            query = """
            MATCH (g:Gene {symbol: $gene_symbol})
            DETACH DELETE g
            """

            self.db.execute_write(query, {'gene_symbol': gene_symbol})
            logger.debug(f"Deleted gene: {gene_symbol}")
            return True

        except Exception as e:
            logger.error(f"Error deleting gene {gene_symbol}: {e}")
            return False

    def process_single_gene_file(self, gene_symbol: str, json_file: Path, importer: GeneCardsImporter) -> bool:
        """Process a single gene file with complete overwrite."""
        try:
            logger.debug(f"Processing {gene_symbol} from {json_file}")

            # Step 1: Delete existing gene completely
            if not self.delete_gene_completely(gene_symbol):
                logger.error(f"Failed to delete existing gene: {gene_symbol}")
                return False

            # Step 2: Re-import the gene from file
            success = importer.process_gene_file(json_file)

            if success:
                logger.debug(f"Successfully re-imported: {gene_symbol}")
                with self.lock:
                    self.success_count += 1
            else:
                logger.error(f"Failed to re-import: {gene_symbol}")
                with self.lock:
                    self.error_count += 1

            with self.lock:
                self.processed_count += 1

            return success

        except Exception as e:
            logger.error(f"Error processing {gene_symbol}: {e}")
            with self.lock:
                self.error_count += 1
                self.processed_count += 1
            return False

    def fix_genes_with_null_aliases(self, data_directory: str, max_workers: int = 2, progress_interval: int = 50):
        """Fix all genes with NULL aliases by re-importing their data."""
        logger.info("=" * 60)
        logger.info("FIXING GENES WITH NULL ALIASES")
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
            logger.info("Genes without corresponding files (first 10):")
            for gene in list(genes_without_files)[:10]:
                logger.info(f"  - {gene}")

        if not genes_with_files:
            logger.warning("No files found for any genes. Nothing to process.")
            return

        # Step 4: Create importer for processing (skip index creation)
        importer = GeneCardsImporter(self.db, batch_size=50)
        # Skip index creation since they already exist
        importer.create_indexes = lambda: None

        # Step 5: Process files with threading (reduced workers to prevent deadlocks)
        # Force single worker to prevent deadlocks
        effective_workers = min(max_workers, 1)
        logger.info(
            f"Starting re-import with {effective_workers} worker threads (single-threaded to prevent deadlocks)...")

        with ThreadPoolExecutor(max_workers=effective_workers) as executor:
            # Submit all tasks
            future_to_gene = {
                executor.submit(self.process_single_gene_file, gene, file_path, importer): gene
                for gene, file_path in genes_with_files.items()
            }

            # Process completed tasks
            for future in as_completed(future_to_gene):
                gene = future_to_gene[future]
                try:
                    success = future.result()
                    if not success:
                        logger.warning(f"Failed to process: {gene}")
                except Exception as e:
                    logger.error(f"Exception processing {gene}: {e}")

                # Print progress periodically
                if self.processed_count % progress_interval == 0:
                    self.print_progress()

        # Final results
        self.missing_files_count = len(genes_without_files)
        logger.info("=" * 60)
        logger.info("FIXING COMPLETED!")
        logger.info("=" * 60)
        self.print_final_results()

    def print_progress(self):
        """Print current progress."""
        logger.info(
            f"Progress: {self.processed_count:,} processed, {self.success_count:,} success, {self.error_count:,} errors")

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
        """Verify that the fix worked by checking NULL aliases again."""
        logger.info("Verifying fix...")

        # Count remaining NULL aliases
        null_count_query = """
        MATCH (g:Gene)
        WHERE g.aliases IS NULL
        RETURN COUNT(g) as count
        """

        # Count non-NULL aliases
        non_null_count_query = """
        MATCH (g:Gene)
        WHERE g.aliases IS NOT NULL
        RETURN COUNT(g) as count
        """

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
    """Main function for fixing genes with NULL aliases."""
    import argparse

    parser = argparse.ArgumentParser(description='Fix genes with NULL aliases')
    parser.add_argument(
        'data_directory', help='Path to GeneCards data directory')
    parser.add_argument('--workers', type=int, default=1,
                        help='Number of worker threads (default: 1, forced to prevent deadlocks)')
    parser.add_argument('--progress-interval', type=int, default=50,
                        help='Progress report interval (default: 50)')
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
    fixer = NullAliasesGenesFixer(db)

    try:
        if args.verify_only:
            # Only verify current state
            fixer.verify_fix()
        else:
            # Run the fix
            fixer.fix_genes_with_null_aliases(
                data_directory=args.data_directory,
                max_workers=args.workers,
                progress_interval=args.progress_interval
            )

            # Verify the fix worked
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
