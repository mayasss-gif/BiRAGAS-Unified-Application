"""
KEGG Data Processor for Neo4j Database Queries

This module provides high-level functions to query KEGG pathway data
from Neo4j database using the CypherQueryBuilder methods.
"""

import logging
from typing import List, Dict, Any, Optional
from .connection import Neo4jConnection
from .query_builder import CypherQueryBuilder

logger = logging.getLogger(__name__)


class KEGGDataProcessor:
    """
    High-level processor for KEGG pathway data queries.
    
    Provides convenient methods to execute KEGG-related queries
    and process results from Neo4j database.
    """
    
    def __init__(self, neo4j_connection: Optional[Neo4jConnection] = None):
        """Initialize with Neo4j connection."""
        self.neo4j = neo4j_connection or Neo4jConnection()
    
    def get_pathway_drugs(self, pathway_id: str, limit: int = 5000) -> List[Dict[str, Any]]:
        """
        Get drugs associated with a KEGG pathway.
        
        Args:
            pathway_id: KEGG pathway ID (e.g., 'hsa05206')
            limit: Maximum number of drugs to return
            
        Returns:
            List of drug dictionaries with drug information
        """
        logger.info(f"Fetching drugs for pathway: {pathway_id}")
        
        query, params = CypherQueryBuilder.get_pathway_drugs(pathway_id, limit)
        results = self.neo4j.execute_query(query, params)
        
        logger.info(f"Found {len(results)} drugs for pathway {pathway_id}")
        return results

    def get_reactome_pathway_drugs(self, pathway_id: str, limit: int = 5000) -> List[Dict[str, Any]]:
        """
        Get drugs associated with a Reactome pathway.
        
        Args:
            pathway_id: Reactome pathway ID
            limit: Maximum number of drugs to return
            
        Returns:
            List of drug dictionaries with drug information
        """
        logger.info(f"Fetching drugs for Reactome pathway: {pathway_id}")
        
        query, params = CypherQueryBuilder.get_reactome_pathway_drugs(pathway_id, limit)
        results = self.neo4j.execute_query(query, params)
        
        logger.info(f"Found {len(results)} drugs for Reactome pathway {pathway_id}")
        return results
        
    def get_drugs_by_gene_symbol(self, gene_symbol: str, limit: int = 5000) -> List[Dict[str, Any]]:
        """
        Get drugs that target a specific gene by symbol.
        
        Args:
            gene_symbol: Gene symbol to search for (e.g., 'SCN3A')
            limit: Maximum number of drugs to return
            
        Returns:
            List of drug dictionaries with drug information
        """
        logger.info(f"Fetching drugs targeting gene: {gene_symbol}")
        
        query, params = CypherQueryBuilder.get_drugs_by_gene_symbol(gene_symbol, limit)
        results = self.neo4j.execute_query(query, params)
        
        # Add target_genes list for compatibility with other drug retrieval methods
        for drug in results:
            # Add matched symbol to target_genes list
            matched_symbols = drug.get('matched_symbols', [])
            drug['target_genes'] = matched_symbols
            
            # Add pathway information placeholders for compatibility
            drug['pathway_name'] = f"Gene Target: {gene_symbol}"
            drug['pathway_id'] = f"gene_{gene_symbol}"
            
        logger.info(f"Found {len(results)} drugs targeting gene {gene_symbol}")
        return results
    
    def get_pathway_drugs_by_name(self, pathway_name: str, limit: int = 5000) -> List[Dict[str, Any]]:
        """
        Get drugs associated with a KEGG pathway by name.
        
        Args:
            pathway_name: KEGG pathway name (e.g., 'PI3K-Akt signaling pathway')
            limit: Maximum number of drugs to return
            
        Returns:
            List of drug dictionaries with drug information
        """
        logger.info(f"Fetching drugs for pathway: {pathway_name}")
        
        query, params = CypherQueryBuilder.get_pathway_drugs_by_name(pathway_name, limit)
        results = self.neo4j.execute_query(query, params)
        
        logger.info(f"Found {len(results)} drugs for pathway {pathway_name}")
        return results
    
    def get_pathway_drugs_with_gene_overlap(self, pathway_id: str, pathway_genes: List[str], 
                                          limit: int = 5000) -> List[Dict[str, Any]]:
        """
        Get drugs associated with a KEGG pathway that have gene overlap.
        
        Args:
            pathway_id: KEGG pathway ID
            pathway_genes: List of gene symbols to check for overlap
            limit: Maximum number of drugs to return
            
        Returns:
            List of drug dictionaries with overlap information
        """
        logger.info(f"Fetching drugs with gene overlap for pathway: {pathway_id}")
        logger.info(f"Checking overlap with {len(pathway_genes)} genes")
        
        query, params = CypherQueryBuilder.get_pathway_drugs_with_gene_overlap(
            pathway_id, pathway_genes, limit
        )
        results = self.neo4j.execute_query(query, params)
        
        logger.info(f"Found {len(results)} drugs with gene overlap for pathway {pathway_id}")
        return results
    
    def get_pathway_info(self, pathway_id: str) -> Dict[str, Any]:
        """
        Get detailed information about a KEGG pathway.
        
        Args:
            pathway_id: KEGG pathway ID
            
        Returns:
            Dictionary with pathway information
        """
        logger.info(f"Fetching pathway info for: {pathway_id}")
        
        query, params = CypherQueryBuilder.get_kegg_pathway_info(pathway_id)
        results = self.neo4j.execute_query(query, params)
        
        if results:
            logger.info(f"Retrieved info for pathway {pathway_id}")
            return results[0]
        else:
            logger.warning(f"No pathway found with ID: {pathway_id}")
            return {}
    
    def search_pathways_by_genes(self, gene_symbols: List[str], min_gene_overlap: int = 1, 
                                limit: int = 5000) -> List[Dict[str, Any]]:
        """
        Find KEGG pathways that contain the specified genes.
        
        Args:
            gene_symbols: List of gene symbols to search for
            min_gene_overlap: Minimum number of genes that must overlap
            limit: Maximum number of pathways to return
            
        Returns:
            List of pathway dictionaries with gene overlap info
        """
        logger.info(f"Searching pathways for {len(gene_symbols)} genes")
        logger.info(f"Minimum overlap required: {min_gene_overlap}")
        
        query, params = CypherQueryBuilder.search_kegg_pathways_by_genes(
            gene_symbols, min_gene_overlap, limit
        )
        results = self.neo4j.execute_query(query, params)
        
        logger.info(f"Found {len(results)} pathways with gene overlap")
        return results
    
    def get_drugs_targeting_pathway_genes(self, pathway_id: str, 
                                        limit: int = 5000) -> List[Dict[str, Any]]:
        """
        Get all drugs that target genes within a specific KEGG pathway.
        
        Args:
            pathway_id: KEGG pathway ID
            limit: Maximum number of drugs to return
            
        Returns:
            List of drug dictionaries with targeting information
        """
        logger.info(f"Fetching drugs targeting genes in pathway: {pathway_id}")
        
        query, params = CypherQueryBuilder.get_drugs_targeting_pathway_genes(
            pathway_id, limit
        )
        results = self.neo4j.execute_query(query, params)
        
        logger.info(f"Found {len(results)} drugs targeting pathway genes")
        return results
    
    def find_drug_pathway_connections(self, drug_name: str, 
                                    limit: int = 5000) -> List[Dict[str, Any]]:
        """
        Find KEGG pathways connected to a specific drug.
        
        Args:
            drug_name: Name of the drug
            limit: Maximum number of pathways to return
            
        Returns:
            List of pathway dictionaries with connection info
        """
        logger.info(f"Finding pathway connections for drug: {drug_name}")
        
        query, params = CypherQueryBuilder.find_kegg_drug_pathway_connections(
            drug_name, limit
        )
        results = self.neo4j.execute_query(query, params)
        
        logger.info(f"Found {len(results)} pathway connections for {drug_name}")
        return results
    
    def get_pathway_statistics(self) -> Dict[str, Any]:
        """
        Get overall statistics about KEGG pathways in the database.
        
        Returns:
            Dictionary with pathway statistics
        """
        logger.info("Fetching KEGG pathway database statistics")
        
        query, params = CypherQueryBuilder.get_kegg_pathway_statistics()
        results = self.neo4j.execute_query(query, params)
        
        if results:
            stats = results[0]
            logger.info("Retrieved pathway statistics:")
            logger.info(f"  Total pathways: {stats.get('total_pathways', 'N/A')}")
            logger.info(f"  Avg genes per pathway: {stats.get('avg_genes_per_pathway', 'N/A'):.2f}")
            logger.info(f"  Avg drugs per pathway: {stats.get('avg_drugs_per_pathway', 'N/A'):.2f}")
            return stats
        else:
            logger.warning("No statistics available")
            return {}
    
    def analyze_gene_drug_connections(self, gene_symbols: List[str]) -> Dict[str, Any]:
        """
        Comprehensive analysis of gene-drug connections through pathways.
        
        Args:
            gene_symbols: List of gene symbols to analyze
            
        Returns:
            Dictionary with comprehensive analysis results
        """
        logger.info(f"Performing comprehensive analysis for {len(gene_symbols)} genes")
        
        analysis_results = {
            "input_genes": gene_symbols,
            "relevant_pathways": [],
            "potential_drugs": [],
            "gene_pathway_overlap": {},
            "drug_gene_overlap": {}
        }
        
        # Find pathways containing these genes
        pathways = self.search_pathways_by_genes(gene_symbols, min_gene_overlap=1)
        analysis_results["relevant_pathways"] = pathways
        
        # For each pathway, get drugs with gene overlap
        all_drugs = []
        for pathway in pathways:
            pathway_id = pathway.get("pathway_id")
            if pathway_id:
                drugs = self.get_pathway_drugs_with_gene_overlap(
                    pathway_id, gene_symbols
                )
                for drug in drugs:
                    drug["source_pathway"] = pathway_id
                    drug["source_pathway_name"] = pathway.get("pathway_name")
                all_drugs.extend(drugs)
        
        # Remove duplicates and sort by overlap
        unique_drugs = {}
        for drug in all_drugs:
            drug_name = drug.get("drug_name")
            if drug_name not in unique_drugs:
                unique_drugs[drug_name] = drug
            else:
                # Keep the one with higher overlap
                existing_overlap = unique_drugs[drug_name].get("overlap_count", 0)
                current_overlap = drug.get("overlap_count", 0)
                if current_overlap > existing_overlap:
                    unique_drugs[drug_name] = drug
        
        analysis_results["potential_drugs"] = sorted(
            unique_drugs.values(), 
            key=lambda x: x.get("overlap_count", 0), 
            reverse=True
        )
        
        logger.info(f"Analysis complete:")
        logger.info(f"  Found {len(pathways)} relevant pathways")
        logger.info(f"  Found {len(unique_drugs)} unique potential drugs")
        
        return analysis_results
    
    def get_pathway_drug_summary(self, pathway_id: str) -> Dict[str, Any]:
        """
        Get a comprehensive summary of a pathway including drugs and genes.
        
        Args:
            pathway_id: KEGG pathway ID
            
        Returns:
            Dictionary with comprehensive pathway summary
        """
        logger.info(f"Getting comprehensive summary for pathway: {pathway_id}")
        
        summary = {
            "pathway_info": {},
            "direct_drugs": [],
            "targeting_drugs": [],
            "total_unique_drugs": 0
        }
        
        # Get pathway info
        pathway_info = self.get_pathway_info(pathway_id)
        summary["pathway_info"] = pathway_info
        
        # Get direct pathway drugs
        direct_drugs = self.get_pathway_drugs(pathway_id)
        summary["direct_drugs"] = direct_drugs
        
        # Get drugs targeting pathway genes
        targeting_drugs = self.get_drugs_targeting_pathway_genes(pathway_id)
        summary["targeting_drugs"] = targeting_drugs
        
        # Calculate unique drugs
        all_drug_names = set()
        for drug in direct_drugs:
            all_drug_names.add(drug.get("drug_name"))
        for drug in targeting_drugs:
            all_drug_names.add(drug.get("drug_name"))
        
        summary["total_unique_drugs"] = len(all_drug_names)
        
        logger.info(f"Pathway summary complete:")
        logger.info(f"  Pathway: {pathway_info.get('pathway_name', 'Unknown')}")
        logger.info(f"  Direct drugs: {len(direct_drugs)}")
        logger.info(f"  Targeting drugs: {len(targeting_drugs)}")
        logger.info(f"  Total unique drugs: {summary['total_unique_drugs']}")
        
        return summary

    def find_drugs_by_genes_and_disease(self, genes: List[str], disease_name: str, limit: int = 5000) -> List[Dict[str, Any]]:
        """
        Find drugs that interact with a list of genes AND are associated with a specific disease.
        """
        logger.info(f"Finding drugs for {len(genes)} genes and disease: {disease_name}")
        query, params = CypherQueryBuilder.find_drugs_by_genes_and_disease(genes, disease_name, limit)
        results = self.neo4j.execute_query(query, params)
        logger.info(f"Found {len(results)} drugs for {len(genes)} genes and disease {disease_name}")
        return results

    def close(self):
        """Close the Neo4j connection."""
        if self.neo4j:
            self.neo4j.close()

def find_drugs_by_genes_and_disease(genes: List[str], disease_name: str, limit: int = 5000, 
                                    neo4j_conn: Optional[Neo4jConnection] = None) -> List[Dict[str, Any]]:
    """Convenience function to find drugs by genes and disease."""
    processor = KEGGDataProcessor(neo4j_conn)
    try:
        return processor.find_drugs_by_genes_and_disease(genes, disease_name, limit)
    finally:
        if neo4j_conn is None:  # Only close if we created the connection
            processor.close()


# Convenience functions for direct usage
def get_pathway_drugs(pathway_id: str, limit: int = 5000, 
                     neo4j_conn: Optional[Neo4jConnection] = None) -> List[Dict[str, Any]]:
    """Convenience function to get pathway drugs."""
    processor = KEGGDataProcessor(neo4j_conn)
    try:
        return processor.get_pathway_drugs(pathway_id, limit)
    finally:
        if neo4j_conn is None:  # Only close if we created the connection
            processor.close()

def get_reactome_pathway_drugs(pathway_id: str, limit: int = 5000, 
                     neo4j_conn: Optional[Neo4jConnection] = None) -> List[Dict[str, Any]]:
    """Convenience function to get Reactome pathway drugs."""
    processor = KEGGDataProcessor(neo4j_conn)
    try:
        return processor.get_reactome_pathway_drugs(pathway_id, limit)
    finally:
        if neo4j_conn is None:  # Only close if we created the connection
            processor.close()

def get_drugs_by_gene_symbol(gene_symbol: str, limit: int = 5000,
                          neo4j_conn: Optional[Neo4jConnection] = None) -> List[Dict[str, Any]]:
    """Convenience function to get drugs targeting a specific gene."""
    processor = KEGGDataProcessor(neo4j_conn)
    try:
        return processor.get_drugs_by_gene_symbol(gene_symbol, limit)
    finally:
        if neo4j_conn is None:  # Only close if we created the connection
            processor.close()


def get_pathway_drugs_by_name(pathway_name: str, limit: int = 5000) -> List[Dict[str, Any]]:
        """
        Get drugs associated with a KEGG pathway by name.
        
        Args:
            pathway_name: KEGG pathway name (e.g., 'PI3K-Akt signaling pathway')
            limit: Maximum number of drugs to return
            
        Returns:
            List of drug dictionaries with drug information
        """
        logger.info(f"Fetching drugs for pathway: {pathway_name}")
        
        processor = KEGGDataProcessor()
        results = processor.get_pathway_drugs_by_name(pathway_name, limit)
        
        logger.info(f"Found {len(results)} drugs for pathway {pathway_name}")
        return results

def get_pathway_drugs_with_overlap(pathway_id: str, pathway_genes: List[str], 
                                 limit: int = 5000, 
                                 neo4j_conn: Optional[Neo4jConnection] = None) -> List[Dict[str, Any]]:
    """Convenience function to get pathway drugs with gene overlap."""
    processor = KEGGDataProcessor(neo4j_conn)
    try:
        return processor.get_pathway_drugs_with_gene_overlap(pathway_id, pathway_genes, limit)
    finally:
        if neo4j_conn is None:
            processor.close()


def search_pathways_by_genes(gene_symbols: List[str], min_overlap: int = 1, 
                           limit: int = 5000,
                           neo4j_conn: Optional[Neo4jConnection] = None) -> List[Dict[str, Any]]:
    """Convenience function to search pathways by genes."""
    processor = KEGGDataProcessor(neo4j_conn)
    try:
        return processor.search_pathways_by_genes(gene_symbols, min_overlap, limit)
    finally:
        if neo4j_conn is None:
            processor.close()


def analyze_genes_for_drugs(gene_symbols: List[str],
                          neo4j_conn: Optional[Neo4jConnection] = None) -> Dict[str, Any]:
    """Convenience function for comprehensive gene-drug analysis."""
    processor = KEGGDataProcessor(neo4j_conn)
    try:
        return processor.analyze_gene_drug_connections(gene_symbols)
    finally:
        if neo4j_conn is None:
            processor.close()


def get_pathway_summary(pathway_id: str,
                       neo4j_conn: Optional[Neo4jConnection] = None) -> Dict[str, Any]:
    """Convenience function to get comprehensive pathway summary."""
    processor = KEGGDataProcessor(neo4j_conn)
    try:
        return processor.get_pathway_drug_summary(pathway_id)
    finally:
        if neo4j_conn is None:
            processor.close()


# Example usage and testing functions
def demo_kegg_queries():
    """Demonstrate all KEGG query functionality."""
    processor = KEGGDataProcessor()
    
    try:
        # Get database statistics
        print("=== KEGG Database Statistics ===")
        stats = processor.get_pathway_statistics()
        for key, value in stats.items():
            print(f"{key}: {value}")
        
        # Example pathway analysis
        example_pathway = "hsa05206"  # Pathways in cancer
        print(f"\n=== Pathway Analysis: {example_pathway} ===")
        
        pathway_info = processor.get_pathway_info(example_pathway)
        print(f"Pathway: {pathway_info.get('pathway_name', 'Unknown')}")
        print(f"Genes: {pathway_info.get('actual_gene_count', 0)}")
        print(f"Drugs: {pathway_info.get('actual_drug_count', 0)}")
        
        # Get pathway drugs
        drugs = processor.get_pathway_drugs(example_pathway, limit=10)
        print(f"\nFound {len(drugs)} drugs for pathway")
        for drug in drugs[:3]:  # Show first 3
            print(f"  - {drug.get('drug_name')} (FDA: {drug.get('fda_approved')})")
        
        # Example gene analysis
        example_genes = ["TP53", "BRCA1", "EGFR"]
        print(f"\n=== Gene Analysis: {example_genes} ===")
        
        pathways = processor.search_pathways_by_genes(example_genes, limit=5)
        print(f"Found {len(pathways)} pathways containing these genes")
        for pathway in pathways[:3]:
            print(f"  - {pathway.get('pathway_name')} (overlap: {pathway.get('gene_overlap_count')})")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
    finally:
        processor.close()


if __name__ == "__main__":
    # Run demo
    demo_kegg_queries()
