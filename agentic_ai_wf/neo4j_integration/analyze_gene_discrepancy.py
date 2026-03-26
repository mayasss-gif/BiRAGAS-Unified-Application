#!/usr/bin/env python3
"""
Gene Data Discrepancy Analysis

Analyzes discrepancies between file count and database gene count.
Identifies duplicates, missing files, and data inconsistencies.
"""

import json
import logging
from pathlib import Path
from typing import Set, Dict, List, Tuple
from collections import Counter, defaultdict
from .connection import Neo4jConnection

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GeneDiscrepancyAnalyzer:
    """Analyze discrepancies between gene files and database."""

    def __init__(self, neo4j_connection: Neo4jConnection):
        self.db = neo4j_connection

    def find_json_files(self, data_directory: str) -> List[Path]:
        """Find all JSON files and return their paths."""
        data_path = Path(data_directory)
        json_files = []

        for folder in sorted(data_path.iterdir()):
            if folder.is_dir() and len(folder.name) == 2:
                for json_file in folder.glob("*.json"):
                    json_files.append(json_file)

        return json_files

    def analyze_file_gene_symbols(self, json_files: List[Path]) -> Dict[str, List[Path]]:
        """Extract gene symbols from files and detect duplicates."""
        gene_to_files = defaultdict(list)

        logger.info(f"Analyzing {len(json_files):,} JSON files...")

        for i, json_file in enumerate(json_files):
            if i % 10000 == 0:
                logger.info(f"Processed {i:,} files...")

            # Gene symbol from filename
            file_gene_symbol = json_file.stem
            gene_to_files[file_gene_symbol].append(json_file)

            # Also check gene symbol inside the file
            # try:
            #     with open(json_file, 'r', encoding='utf-8') as f:
            #         json_data = json.load(f)

            #     # Extract gene symbol from Gene section
            #     if "Gene" in json_data and json_data["Gene"]:
            #         gene_info = json_data["Gene"][0] if isinstance(
            #             json_data["Gene"], list) else json_data["Gene"]
            #         if isinstance(gene_info, dict):
            #             internal_symbol = gene_info.get("Name", "")
            #             if internal_symbol and internal_symbol != file_gene_symbol:
            #                 logger.warning(
            #                     f"Symbol mismatch in {json_file}: "
            #                     f"filename='{file_gene_symbol}' vs internal='{internal_symbol}'"
            #                 )
            # except Exception as e:
            #     logger.error(f"Error reading {json_file}: {e}")

        return dict(gene_to_files)

    def get_database_genes(self) -> Set[str]:
        """Get all gene symbols from database."""
        logger.info("Querying database for all gene symbols...")

        try:
            result = self.db.execute_query("MATCH (n:Gene) RETURN n.symbol")
            genes = {record['n.symbol'] for record in result}
            logger.info(f"Found {len(genes):,} genes in database")
            return genes
        except Exception as e:
            logger.error(f"Error querying database: {e}")
            return set()

    def get_database_gene_details(self) -> List[Dict]:
        """Get detailed gene information from database."""
        logger.info("Querying database for detailed gene information...")

        query = """
        MATCH (g:Gene)
        RETURN g.symbol as symbol, 
               g.name as name,
               g.genecards_id as genecards_id,
               g.source as source,
               g.created_at as created_at
        ORDER BY g.symbol
        """

        try:
            result = self.db.execute_query(query)
            logger.info(f"Retrieved details for {len(result):,} genes")
            return result
        except Exception as e:
            logger.error(f"Error querying database details: {e}")
            return []

    def find_duplicate_genes_in_db(self) -> List[Dict]:
        """Find duplicate gene symbols in database."""
        logger.info("Checking for duplicate genes in database...")

        query = """
        MATCH (g:Gene)
        WITH g.symbol as symbol, count(*) as count
        WHERE count > 1
        RETURN symbol, count
        ORDER BY count DESC, symbol
        """

        try:
            result = self.db.execute_query(query)
            if result:
                logger.warning(
                    f"Found {len(result)} duplicate gene symbols in database!")
                for record in result:
                    logger.warning(
                        f"  {record['symbol']}: {record['count']} instances")
            else:
                logger.info("No duplicate gene symbols found in database")
            return result
        except Exception as e:
            logger.error(f"Error checking duplicates: {e}")
            return []

    def analyze_discrepancies(self, data_directory: str) -> Dict:
        """Perform comprehensive discrepancy analysis."""
        logger.info("=" * 60)
        logger.info("GENE DISCREPANCY ANALYSIS")
        logger.info("=" * 60)

        # 1. Get file information
        json_files = self.find_json_files(data_directory)
        file_count = len(json_files)
        logger.info(f"Total JSON files found: {file_count:,}")

        # 2. Analyze file gene symbols
        gene_to_files = self.analyze_file_gene_symbols(json_files)
        unique_file_genes = len(gene_to_files)
        logger.info(f"Unique gene symbols from files: {unique_file_genes:,}")

        # 3. Find duplicate files
        duplicate_files = {gene: files for gene,
                           files in gene_to_files.items() if len(files) > 1}
        if duplicate_files:
            logger.warning(
                f"Found {len(duplicate_files)} genes with multiple files:")
            # Show first 10
            for gene, files in list(duplicate_files.items())[:10]:
                logger.warning(f"  {gene}: {len(files)} files")
                for file_path in files:
                    logger.warning(f"    - {file_path}")

        # 4. Get database information
        db_genes = self.get_database_genes()
        db_gene_count = len(db_genes)
        logger.info(f"Genes in database: {db_gene_count:,}")

        # 5. Find database duplicates
        db_duplicates = self.find_duplicate_genes_in_db()

        # 6. Compare file genes vs database genes
        file_genes = set(gene_to_files.keys())

        # Genes in files but not in database
        missing_from_db = file_genes - db_genes
        logger.info(
            f"Genes in files but NOT in database: {len(missing_from_db):,}")
        if missing_from_db:
            logger.info("Sample missing genes (first 10):")
            for gene in list(missing_from_db)[:10]:
                logger.info(f"  - {gene}")

        # Genes in database but not in files
        missing_from_files = db_genes - file_genes
        logger.info(
            f"Genes in database but NOT in files: {len(missing_from_files):,}")
        if missing_from_files:
            logger.info("Sample extra genes (first 10):")
            for gene in list(missing_from_files)[:10]:
                logger.info(f"  - {gene}")

        # 7. Get detailed gene info from database
        db_gene_details = self.get_database_gene_details()

        # 8. Analyze gene sources and creation times
        source_counts = Counter()
        creation_dates = defaultdict(int)

        for gene_detail in db_gene_details:
            source = gene_detail.get('source', 'unknown')
            source_counts[source] += 1

            created_at = gene_detail.get('created_at', '')
            if created_at:
                creation_date = created_at.split('T')[0]  # Extract date part
                creation_dates[creation_date] += 1

        logger.info(f"\nGene sources in database:")
        for source, count in source_counts.most_common():
            logger.info(f"  {source}: {count:,}")

        logger.info(f"\nGene creation dates (top 10):")
        for date, count in sorted(creation_dates.items(), key=lambda x: x[1], reverse=True)[:10]:
            logger.info(f"  {date}: {count:,}")

        # 9. Summary
        logger.info("=" * 60)
        logger.info("SUMMARY")
        logger.info("=" * 60)
        logger.info(f"JSON files: {file_count:,}")
        logger.info(f"Unique genes from files: {unique_file_genes:,}")
        logger.info(f"Genes in database: {db_gene_count:,}")
        logger.info(f"Duplicate file genes: {len(duplicate_files):,}")
        logger.info(f"Duplicate DB genes: {len(db_duplicates):,}")
        logger.info(f"Missing from DB: {len(missing_from_db):,}")
        logger.info(f"Extra in DB: {len(missing_from_files):,}")

        return {
            'file_count': file_count,
            'unique_file_genes': unique_file_genes,
            'db_gene_count': db_gene_count,
            'duplicate_files': duplicate_files,
            'db_duplicates': db_duplicates,
            'missing_from_db': missing_from_db,
            'missing_from_files': missing_from_files,
            'source_counts': source_counts,
            'creation_dates': creation_dates
        }


def main():
    """Main function for analysis."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Analyze gene data discrepancies')
    parser.add_argument(
        'data_directory', help='Path to GeneCards data directory')
    parser.add_argument(
        '--neo4j-uri', default='bolt://localhost:7687', help='Neo4j URI')
    parser.add_argument('--neo4j-user', default='neo4j', help='Neo4j username')
    parser.add_argument('--neo4j-password',
                        default='password', help='Neo4j password')

    args = parser.parse_args()

    # Connect to Neo4j
    try:
        db = Neo4jConnection()
        logger.info("✅ Connected to Neo4j successfully!")
    except Exception as e:
        logger.error(f"❌ Failed to connect to Neo4j: {e}")
        return

    # Run analysis
    analyzer = GeneDiscrepancyAnalyzer(db)
    try:
        results = analyzer.analyze_discrepancies(args.data_directory)

        # Save results to file
        output_file = "gene_discrepancy_analysis.json"
        with open(output_file, 'w') as f:
            # Convert sets to lists for JSON serialization
            json_results = {
                'file_count': results['file_count'],
                'unique_file_genes': results['unique_file_genes'],
                'db_gene_count': results['db_gene_count'],
                'duplicate_files_count': len(results['duplicate_files']),
                'db_duplicates_count': len(results['db_duplicates']),
                'missing_from_db_count': len(results['missing_from_db']),
                'missing_from_files_count': len(results['missing_from_files']),
                'missing_from_db': list(results['missing_from_db']),
                'missing_from_files': list(results['missing_from_files']),
                'source_counts': dict(results['source_counts']),
                'creation_dates': dict(results['creation_dates'])
            }
            json.dump(json_results, f, indent=2)

        logger.info(f"Detailed results saved to: {output_file}")

    except Exception as e:
        logger.error(f"Analysis failed: {e}")
    finally:
        db.close()


if __name__ == "__main__":
    main()
