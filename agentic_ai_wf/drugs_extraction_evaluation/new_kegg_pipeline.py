import pandas as pd
import numpy as np
import time
from typing import Dict, List, Optional, Any
import re
import logging
from pathlib import Path
from .universal_prioritization_tool import run_universal_prioritization

# FDA VERIFICATION FUNCTIONS
from .utils.drugs_filtering import filter_log2fc_data
from agentic_ai_wf.neo4j_integration.kegg_data_processor import (
    get_pathway_drugs, get_reactome_pathway_drugs, 
    get_pathway_drugs_by_name, get_drugs_by_gene_symbol, find_drugs_by_genes_and_disease
)
from .column_config import DrugColumnConfig

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class KEGGDrugPipeline:
    """
    A comprehensive pipeline for KEGG pathway enrichment and drug discovery.

    This pipeline reads pathway data, filters for KEGG pathways, finds associated drugs,
    and matches them against pathway genes to identify potential therapeutic targets.
    """

    def __init__(self):
        self.max_drugs_per_pathway = 500  # Limit for performance

        # Patient prioritized genes data
        self.patient_genes_df = None
        self.patient_log2fc_columns = []
        self.synonym_to_gene_map = {}  # Maps synonyms to primary gene names

        logger.info("Initialized KEGG Drug Pipeline")

    def load_patient_genes(self, patient_genes_file: str, patient_prefix: str = 'patient') -> bool:
        """
        Load patient prioritized genes with log2FC data.

        Args:
            patient_genes_file: Path to CSV file with patient gene prioritization
            patient_prefix: Prefix for patient columns
            filter_positive_only: If True, only keep genes with positive log2FC values

        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            self.patient_genes_df = pd.read_csv(patient_genes_file)

            # Find log2FC columns (patient_prefix*_log2FC pattern)
            log2fc_cols = [
                col for col in self.patient_genes_df.columns if col.startswith(f"{patient_prefix}") and col.endswith('_log2FC')]
            self.patient_log2fc_columns = log2fc_cols

            logger.info(
                f"Loaded {len(self.patient_genes_df)} patient prioritized genes from {patient_genes_file}")
            logger.info(f"Found log2FC columns: {log2fc_cols}")

            # Convert Gene column to uppercase for matching
            if 'Gene' in self.patient_genes_df.columns:
                self.patient_genes_df['Gene_Upper'] = self.patient_genes_df['Gene'].str.upper()

                # Process HGNC_Synonyms column for synonym mapping
                self._process_synonyms()

                # Initialize results
                results = {"upregulated": [], "downregulated": []}

                if log2fc_cols:
                    # Iterate rows
                    for _, row in self.patient_genes_df.iterrows():
                        gene_symbol = row['Gene']
                        synonyms = []
                        if 'HGNC_Synonyms' in row and isinstance(row['HGNC_Synonyms'], str):
                            synonyms = [s.strip() for s in row['HGNC_Synonyms'].split(';') if s.strip()]

                        # Collect all log2FC values across patient columns
                        log2fc_values = []
                        for col in log2fc_cols:
                            try:
                                val = float(row[col])
                                if not pd.isna(val):
                                    log2fc_values.append(val)
                            except Exception:
                                continue

                        if not log2fc_values:
                            continue

                        # Representative value (mean across columns)
                        mean_log2fc = sum(log2fc_values) / len(log2fc_values)

                        gene_entry = {
                            "gene": gene_symbol,
                            "log2fc": mean_log2fc,
                            "synonyms": synonyms,
                        }

                        # Apply thresholds
                        if any(val > 1 for val in log2fc_values):
                            results["upregulated"].append(gene_entry)
                        elif any(val < -1 for val in log2fc_values):
                            results["downregulated"].append(gene_entry)

                    logger.info(
                        f"Extracted {len(results['upregulated'])} upregulated and "
                        f"{len(results['downregulated'])} downregulated genes"
                    )

                    # Store dict for downstream access
                    self.patient_gene_signatures = results
                else:
                    logger.warning("No log2FC columns found - cannot classify up/down genes")
                    self.patient_gene_signatures = {"upregulated": [], "downregulated": []}
        except Exception as e:
            logger.error(f"Error loading patient genes: {e}")
            return False
        return True
    
    def _process_synonyms(self):
        """
        Process HGNC_Synonyms column to create a mapping from synonyms to primary gene names.
        """
        self.synonym_to_gene_map = {}
        
        if 'HGNC_Synonyms' not in self.patient_genes_df.columns:
            logger.warning("HGNC_Synonyms column not found - synonym matching will be disabled")
            return
        
        synonym_count = 0
        for _, row in self.patient_genes_df.iterrows():
            gene = row['Gene']
            synonyms_str = row['HGNC_Synonyms']
            
            if pd.notna(synonyms_str) and synonyms_str.strip():
                # Split by semicolon and clean up each synonym
                synonyms = [syn.strip().upper() for syn in str(synonyms_str).split(';') if syn.strip()]
                
                for synonym in synonyms:
                    if synonym:  # Ensure non-empty
                        self.synonym_to_gene_map[synonym] = gene
                        synonym_count += 1
        
        logger.info(f"Processed {synonym_count} gene synonyms from HGNC_Synonyms column")
        logger.info(f"Created synonym mapping for {len(set(self.synonym_to_gene_map.values()))} unique genes")

    def get_patient_gene_log2fc(self, gene: str) -> dict:
        """
        Get log2FC values for a gene from patient data, including synonym lookup and override status.

        Args:
            gene: Gene symbol to look up

        Returns:
            Dictionary with log2FC values for each patient column and override information
        """
        if self.patient_genes_df is None:
            return {}

        gene_upper = gene.upper()
        
        # First try direct gene match
        gene_row = self.patient_genes_df[self.patient_genes_df['Gene_Upper'] == gene_upper]
        
        # If no direct match, try synonym lookup
        if len(gene_row) == 0 and gene_upper in self.synonym_to_gene_map:
            primary_gene = self.synonym_to_gene_map[gene_upper]
            primary_gene_upper = primary_gene.upper()
            gene_row = self.patient_genes_df[self.patient_genes_df['Gene_Upper'] == primary_gene_upper]
            logger.debug(f"Found gene via synonym: {gene} -> {primary_gene}")

        if len(gene_row) == 0:
            return {}

        log2fc_data = {}
        for col in self.patient_log2fc_columns:
            value = gene_row.iloc[0][col]
            if pd.notna(value):
                log2fc_data[col] = value

        return log2fc_data

    def check_gene_overlap_with_patient_genes(self, pathway_genes: List[str], drug_targets: List[str]) -> tuple[bool, List[str], str]:
        """
        Check for gene overlap, first with pathway genes, then with patient prioritized genes.
        Includes synonym matching for enhanced gene identification and considers ACTIVITY_OVERRIDE flags.

        Args:
            pathway_genes: List of pathway-associated genes
            drug_targets: List of drug target genes

        Returns:
            Tuple of (has_overlap, overlapping_genes, match_type)
        """
        if not drug_targets:
            return False, [], "No overlap"

        # Convert to uppercase for comparison
        target_set = set(gene.upper() for gene in drug_targets)

        # First try pathway genes (if provided)
        if pathway_genes:
            pathway_set = set(gene.upper() for gene in pathway_genes)
            pathway_overlap = pathway_set.intersection(target_set)
            if pathway_overlap:
                logger.info(
                    f"✓ Direct pathway gene overlap: {list(pathway_overlap)}")
                return True, list(pathway_overlap), "Direct Pathway Gene Match"

        # If no pathway overlap, try patient prioritized genes
        if self.patient_genes_df is not None:
            # Collect all matches (direct, synonym, and activity override)
            all_matches = []
            match_details = []
            
            # Get sets of genes with different match types
            patient_genes_set = set(self.patient_genes_df['Gene_Upper'].dropna())
            
            # Create a set of genes with ACTIVITY_OVERRIDE flag if the column exists
            override_genes_set = set()
            if 'Override_Reason' in self.patient_genes_df.columns:
                override_genes = self.patient_genes_df[self.patient_genes_df['Override_Reason'] == 'ACTIVITY_OVERRIDE']
                if not override_genes.empty:
                    override_genes_set = set(override_genes['Gene_Upper'].dropna())
            
            # Direct patient gene match
            patient_overlap = patient_genes_set.intersection(target_set)
            if patient_overlap:
                all_matches.extend(list(patient_overlap))
                match_details.append(f"Direct: {list(patient_overlap)}")
                logger.info(
                    f"✓ Patient prioritized gene overlap (direct): {list(patient_overlap)}")

            # Activity override match
            if override_genes_set:
                override_overlap = override_genes_set.intersection(target_set)
                if override_overlap:
                    # Filter out genes already in direct match
                    new_override_matches = [gene for gene in override_overlap if gene not in all_matches]
                    if new_override_matches:
                        all_matches.extend(new_override_matches)
                        match_details.append(f"Activity Override: {new_override_matches}")
                        logger.info(
                            f"✓ Patient gene activity override match: {new_override_matches}")

            # Synonym matching for remaining targets
            synonym_matches = []
            primary_genes_matched = []
            
            for target_gene in target_set:
                # Skip if already found as direct match
                if target_gene in patient_overlap:
                    continue
                    
                if target_gene in self.synonym_to_gene_map:
                    primary_gene = self.synonym_to_gene_map[target_gene]
                    primary_gene_upper = primary_gene.upper()
                    
                    # Check if the primary gene is in our patient genes or override genes
                    if primary_gene_upper in patient_genes_set or primary_gene_upper in override_genes_set:
                        # Only add if not already in all_matches
                        if primary_gene_upper not in all_matches:
                            synonym_matches.append(target_gene)
                            primary_genes_matched.append(primary_gene)
                            all_matches.append(primary_gene_upper)
            
            if synonym_matches:
                match_details.append(f"Synonym: {synonym_matches} -> {primary_genes_matched}")
                logger.info(
                    f"✓ Patient gene synonym overlap: {synonym_matches} -> {primary_genes_matched}")
            
            # Return combined results
            if all_matches:
                match_type = "Patient Gene Match (" + "; ".join(match_details) + ")"
                logger.info(f"✓ Combined gene overlap found: {all_matches}")
                return True, all_matches, match_type

        return False, [], "No overlap"

    def filter_pathways_file(self, csv_file_path: str, max_rows: Optional[int] = None) -> pd.DataFrame:
        """
        Read CSV and filter for upregulated pathways with high/medium confidence from all databases, keeping only specified columns.

        Args:
            csv_file_path: Path to the input CSV file
            max_rows: Maximum number of rows to process (None for all rows)

        Returns:
            Filtered DataFrame with upregulated pathways having high or medium confidence levels from all databases
        """
        # Define columns to keep
        columns_to_keep = [
            'Priority_Rank', 'LLM_Score', 'Confidence_Level', 'Score_Justification',
            'DB_ID', 'Pathway_Source', 'Pathway_ID', 'Pathway_Name',
            'Regulation', 'Clinical_Relevance', 'Functional_Relevance',
            'Pathway_Associated_Genes'
        ]

        try:
            # Read the CSV file
            df = pd.read_csv(csv_file_path)
            logger.info(
                f"Loaded {len(df)} total pathways from {csv_file_path}")

            # Filter for only upregulated pathways (all databases)
            if 'Regulation' in df.columns:
                before_regulation_filter = len(df)
                combined_df = df[df['Regulation'].str.lower() == 'up'].copy()
                after_regulation_filter = len(combined_df)
                logger.info(f"Filtered to upregulated pathways only: {before_regulation_filter} → {after_regulation_filter} pathways")
            else:
                logger.warning("'Regulation' column not found - using all pathways")
                combined_df = df.copy()
            
            # Filter for high and medium confidence levels only
            if 'Confidence_Level' in combined_df.columns:
                before_confidence_filter = len(combined_df)
                combined_df = combined_df[combined_df['Confidence_Level'].str.lower().isin(['high', 'medium'])].copy()
                after_confidence_filter = len(combined_df)
                logger.info(f"Filtered to high/medium confidence only: {before_confidence_filter} → {after_confidence_filter} pathways")
            else:
                logger.warning("'Confidence_Level' column not found - using all pathways")
            
            # Keep only specified columns that exist in the dataframe
            existing_columns = [
                col for col in columns_to_keep if col in combined_df.columns]
            combined_df = combined_df[existing_columns]
            
            # Log the counts by database type (after upregulation filter)
            if 'DB_ID' in combined_df.columns:
                db_counts = combined_df['DB_ID'].str.extract('([A-Z]+)', expand=False).value_counts()
                logger.info(f"Upregulated pathways by database: {dict(db_counts)}")
            
            logger.info(f"Total upregulated pathways processed: {len(combined_df)}")

            # Apply max_rows limit if specified
            if max_rows is not None:
                combined_df = combined_df[:max_rows]
                logger.info(
                    f"Processing {len(combined_df)} pathways (limited to {max_rows})")
            else:
                logger.info(
                    f"Processing all {len(combined_df)} pathways")

            return pd.DataFrame(combined_df)

        except Exception as e:
            logger.error(f"Error reading CSV: {str(e)}")
            return pd.DataFrame()

    def create_target_mechanism(self, drug_data: Dict) -> str:
        """
        Create a combined target-mechanism column from target and efficacy data.

        Args:
            drug_data: Dictionary containing drug information with target-updated and efficacy-updated

        Returns:
            Combined target-mechanism string
        """
        target = drug_data.get('target_updated', '')
        efficacy = drug_data.get('efficacy_updated', '')

        # Clean up None values
        if target is None:
            target = ''
        if efficacy is None:
            efficacy = ''

        # Convert to string and strip
        target = str(target).strip()
        efficacy = str(efficacy).strip()

        if target and efficacy:
            return f"{target} / {efficacy}"
        elif target:
            return target
        elif efficacy:
            return efficacy
        else:
            return "Not available"

    # _extract_target_genes method removed - no longer needed

    def parse_pathway_genes(self, gene_string: str) -> List[str]:
        """
        Parse pathway associated genes from the CSV.

        Args:
            gene_string: Comma-separated gene string

        Returns:
            List of cleaned gene symbols
        """
        if pd.isna(gene_string):
            return []

        # Split by comma and clean up
        genes = [gene.strip() for gene in str(gene_string).split(',')]

        # Remove any prefixes and keep only gene symbols
        clean_genes = []
        for gene in genes:
            # Look for gene symbols (typically uppercase letters/numbers)
            gene_match = re.search(r'[A-Z][A-Z0-9]+', gene)
            if gene_match:
                clean_genes.append(gene_match.group())

        return list(set(clean_genes))

    def find_gene_overlap(self, pathway_genes: List[str], drug_targets: List[str]) -> List[str]:
        """
        Find overlapping genes between pathway and drug targets.

        Args:
            pathway_genes: Genes associated with the pathway
            drug_targets: Target genes of the drug

        Returns:
            List of overlapping genes
        """
        pathway_set = set(g.upper() for g in pathway_genes)
        drug_set = set(g.upper() for g in drug_targets)
        overlap = list(pathway_set.intersection(drug_set))
        return overlap

    def check_gene_overlap(self, pathway_genes: List[str], drug_targets: List[str]) -> bool:
        """
        Check if there's any overlap between pathway genes and drug targets.

        Args:
            pathway_genes: List of pathway-associated genes
            drug_targets: List of drug target genes

        Returns:
            True if there's overlap, False otherwise
        """
        if not pathway_genes or not drug_targets:
            return False

        # Convert to uppercase for comparison
        pathway_set = set(gene.upper() for gene in pathway_genes)
        target_set = set(gene.upper() for gene in drug_targets)

        # Find intersection
        overlap = pathway_set.intersection(target_set)

        # Debug logging
        if overlap:
            logger.info(f"Gene overlap found: {list(overlap)}")
        else:
            logger.debug(f"No gene overlap found")

        return len(overlap) > 0
    
    def process_pathways(self, kegg_pathways: pd.DataFrame) -> pd.DataFrame:
        """
        Process KEGG pathways to find drugs, including both matching and non-matching.

        Args:
            kegg_pathways: DataFrame with KEGG pathway information

        Returns:
            DataFrame with drug-pathway information (both matches and non-matches)
        """
        results = []

        for i, (idx, row) in enumerate(kegg_pathways.iterrows()):
            try:
                pathway_id = row['Pathway_ID']
                pathway_name = row['Pathway_Name']
                pathway_db_id = row['DB_ID']

                logger.info(
                    f"Processing pathway {i+1}/{len(kegg_pathways)}: {pathway_name}")

                # Parse pathway genes
                pathway_genes = self.parse_pathway_genes(
                    str(row['Pathway_Associated_Genes']))
                logger.info(
                    f"Found {len(pathway_genes)} genes for pathway: {', '.join(pathway_genes[:5])}...")

                # Get drugs for this pathway
                if str(pathway_db_id).lower() == 'kegg':
                    drugs = get_pathway_drugs(str(pathway_id))
                elif str(pathway_db_id).lower() == 'reactome':
                    drugs = get_reactome_pathway_drugs(str(pathway_id))
                elif str(pathway_db_id).lower() == 'gene':
                    # If the pathway is actually a gene, get drugs targeting that gene
                    gene_symbol = str(pathway_name).split(':')[-1].strip()
                    logger.info(f"Fetching drugs targeting gene: {gene_symbol}")
                    drugs = get_drugs_by_gene_symbol(gene_symbol)
                else:
                    logger.info(f"Fetching drugs for pathway: {pathway_name} by name for DB {pathway_db_id}")
                    drugs = get_pathway_drugs_by_name(str(pathway_name))

                    # check if 5+ drugs with fda approved status
                    fda_approved_drugs = [drug for drug in drugs if drug['fda_approved_status'] == 'Approved']
                    if len(fda_approved_drugs) >= 5:
                        logger.info(f"Found {len(fda_approved_drugs)} FDA approved drugs for pathway: {pathway_name} - using only FDA approved drugs")
                        drugs = fda_approved_drugs
                    else:
                        logger.info(f"Only {len(fda_approved_drugs)} FDA approved drugs found for pathway: {pathway_name} - using all drugs")
                        drugs = drugs

                # Process each drug - include ALL drugs, not just matching ones
                for drug in drugs:
                    drug_targets = drug['target_genes']
                    logger.info(f"Drug targets: {drug_targets}")

                    if isinstance(drug_targets, list) and len(drug_targets) > 0:
                        logger.info(f"Checking gene overlap for drug: {drug['drug_name']} with {len(drug_targets)} targets: {drug_targets}")
                        # Check for gene overlap using new method
                        has_overlap, overlapping_genes, match_type = self.check_gene_overlap_with_patient_genes(
                            pathway_genes, drug_targets)

                        # Calculate overlap genes and get log2FC data
                        if has_overlap:
                            overlap_str = ', '.join(overlapping_genes)
                            match_status = match_type

                            # Get log2FC data for overlapping genes
                            log2fc_data = {}
                            for gene in overlapping_genes:
                                gene_log2fc = self.get_patient_gene_log2fc(gene)
                                log2fc_data.update(gene_log2fc)

                            # Format log2FC data as string
                            log2fc_items = []
                            
                            # Add override reason if available
                            if 'override_reason' in log2fc_data:
                                log2fc_items.append(f"Override: {log2fc_data['override_reason']}")
                                # Remove from dict so it's not included in the numeric values
                                del log2fc_data['override_reason']
                            
                            # Add expression data if available
                            expression_items = []
                            for key in list(log2fc_data.keys()):
                                if key.startswith('expression_'):
                                    expression_items.append(f"{key.replace('expression_', '')}: {log2fc_data[key]:.2f}")
                                    del log2fc_data[key]
                            
                            if expression_items:
                                log2fc_items.append("Expression: " + ", ".join(expression_items))
                            
                            # Add log2FC values
                            log2fc_items.extend([f"{col}: {value:.2f}" for col, value in log2fc_data.items()])
                            
                            log2fc_str = '; '.join(log2fc_items)
                            if not log2fc_str:
                                log2fc_str = "No log2FC data available"

                            result = {
                            'pathway_id': drug['pathway_id'],
                            'pathway_name': drug['pathway_name'],
                            'pathway_genes': ', '.join(pathway_genes),
                            'drug_id': drug['drug_id'],
                            'drug_name': drug['drug_name'],
                            'target_genes': drug['target_genes'],
                            'gene_overlap': overlap_str,
                            'match_status': match_status,
                            'patient_log2fc': log2fc_str,
                            'drug_classes': drug['drug_classes'],
                            'mechanism_of_action': drug['mechanism_of_action'],
                            'drugbank_id': drug['drugbank_id'],
                            'chembl_id': drug['chembl_id'],
                            'target_genes_lfc': 'Not available',
                            'fda_approved_status': drug['fda_approved_status'],
                            'brand_name': drug['brand_name'],
                            'generic_name': drug['generic_name'],
                            'route_of_administration': drug['route_of_administration'],
                            'efficacy': drug['efficacy'],
                            'target_genes_property': drug['target_genes_property'],
                            'clinical_relevance': row.get('Clinical_Relevance', ''),
                            'functional_relevance': row.get('Functional_Relevance', ''),
                            'regulation': row.get('Regulation', ''),
                            'p_value': row.get('P_Value', ''),
                            'fdr': row.get('FDR', ''),
                            'priority_rank': row.get('Priority_Rank', ''),
                            'llm_score': row.get('LLM_Score', ''),
                            'score_justification': row.get('Score_Justification', ''),
                            'target_mechanism': self.create_target_mechanism(drug)
                            }
                            results.append(result)
                        else:
                            logger.info(f"No gene overlap found for drug: {drug['drug_name']}")

            except Exception as e:
                logger.error(
                    f"Error processing pathway {row.get('Pathway_ID', 'unknown')}: {str(e)}")
                continue

        return pd.DataFrame(results)

    def save_results(self, df: pd.DataFrame, output_file: str):
        """
        Save results to CSV.

        Args:
            df: Results DataFrame
            output_file: Output file path
        """
        # Ensure output directory exists
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if df.empty:
            logger.warning(f"Saving empty DataFrame to: {output_file}")
            
            df = pd.DataFrame(columns=DrugColumnConfig.get_all_expected_columns())
        else:
            logger.info(f"Saving {df.shape[0]} drugs results to: {output_file}")
            logger.info(f"DataFrame shape and columns: {df.shape} with {len(df.columns)} columns")
            logger.info(f"Key columns present: final_rank={df.get('final_rank') is not None}, recommendation={df.get('recommendation') is not None}")
            
            # Ensure all required columns are present using centralized configuration
            
            df = DrugColumnConfig.add_missing_columns(df, preserve_existing=True)
            logger.info("Column validation completed using centralized configuration")
        
        df.to_csv(output_file, index=False)
        logger.info(f"Results saved to: {output_file}")

def filter_and_deduplicate_results(results_df):
    """
    Filter results to keep only up-regulated pathways and remove duplicates.
    
    Args:
        results_df: DataFrame with drug evaluation results
        
    Returns:
        Filtered and deduplicated DataFrame
    """
    print("Sorting dataframe...")
    results_df = results_df.sort_values(
        by=['llm_score', 'priority_rank'], 
        ascending=[False, True]
    ).reset_index(drop=True)
    
    # getting only up regulated pathways (case-insensitive)
    print("Filtering for up-regulated pathways...")
    results_df = results_df[results_df['regulation'].str.lower() == 'up']
    
    # Keep only unique combinations of pathway_id + drug_id, keeping the first row for each combination
    print("Removing duplicates...")
    results_df = results_df.drop_duplicates(subset=['pathway_id', 'drug_id'], keep='first').reset_index(drop=True)
    
    print(f"Final dataframe shape: {results_df.shape}")
    return results_df

def expand_genes_with_synonyms(genes_with_synonyms: List[Dict[str, Any]]) -> List[str]:
    """
    Expand a list of gene entries with synonyms into a flat, unique list of symbols.

    Args:
        genes_with_synonyms: List of dicts with keys 'gene' and 'synonyms'.

    Returns:
        List of unique gene symbols (uppercase).
    """
    expanded: set[str] = set()

    for entry in genes_with_synonyms:
        gene = str(entry.get("gene") or "").strip()
        if gene:
            expanded.add(gene.upper())

        synonyms = entry.get("synonyms", [])
        if isinstance(synonyms, str):
            synonyms = [s.strip() for s in synonyms.split(";") if s.strip()]
        if isinstance(synonyms, list):
            for s in synonyms:
                if s:
                    expanded.add(s.upper())

    return sorted(expanded)


def main(consolidated_pathways_file: Optional[Path] = None, prioritized_genes_file: Optional[Path] = None, output_file: Optional[Path] = None, analysis_id: str = "kegg_drug_discovery", disease_name: str = "Unknown Disease") -> Path: 
    """
    Main function to run the KEGG drug discovery pipeline.
    
    Args:
        consolidated_pathways_file: Path to the consolidated pathways CSV file
        prioritized_genes_file: Path to the prioritized genes CSV file
        output_file: Path to save the results
        analysis_id: Identifier for the analysis
        disease_name: Name of the disease being analyzed
        
    Returns:
        Path to the output file with drug discovery results
    """
    # Initialize pipelinez
    pipeline = KEGGDrugPipeline()

    # Step 1: Read and filter CSV - process both KEGG and REACTOME pathways
    if consolidated_pathways_file is None:
        input_file = "test_files/small_test_pathways.csv"
    else:
        input_file = consolidated_pathways_file
    if prioritized_genes_file is None:
        pg_input_file = "test_files/Alzheimer_DEGs_prioritized.csv"
    else:
        pg_input_file = prioritized_genes_file
    
    if disease_name == 'Unknown Disease':
        disease_name = "Alzheimer"

    # Set default output file
    if output_file is None:
        output_file = Path(
            f"./agentic_ai_wf/shared/drugs_discovery/{analysis_id}/drug_discovery_results.csv")

    # Step 2: Load patient prioritized genes
    logger.info(f"Loading patient prioritized genes from {pg_input_file}")
    patient_genes_loaded = pipeline.load_patient_genes(
        str(pg_input_file), patient_prefix=analysis_id)

    # Define output file path - ALWAYS create this file
    up_drugs_file = Path(output_file).parent / "up_targeting_drugs_disease_associated.csv"
    
    # Ensure parent directory exists
    up_drugs_file.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Ensured directory exists: {up_drugs_file.parent}")

    # Always recreate up_drugs_file to ensure fresh data for current analysis
    logger.info(f"Checking up_drugs_file: exists={up_drugs_file.exists()}, size={up_drugs_file.stat().st_size if up_drugs_file.exists() else 'N/A'}")
    
    # ALWAYS create/recreate the file to ensure current analysis data
    if patient_genes_loaded:
        logger.info("✓ Patient prioritized genes loaded successfully")

        # Get expanded upregulated gene list
        up_genes = pipeline.patient_gene_signatures.get("upregulated", [])
        expanded_up_genes = expand_genes_with_synonyms(up_genes)

        logger.info(
            f"Expanded {len(up_genes)} upregulated genes into "
            f"{len(expanded_up_genes)} unique gene symbols (incl. synonyms)"
        )

        # Query drugs for disease
        logger.info(f"Querying drugs for {len(expanded_up_genes)} genes and disease: {disease_name}")
        up_targeting_drugs_disease_associated = find_drugs_by_genes_and_disease(
            expanded_up_genes,
            disease_name
        )

        logger.info(
            f"Retrieved {len(up_targeting_drugs_disease_associated)} candidate "
            f"drugs for {disease_name}"
        )

        # Save results
        up_targeting_drugs_disease_associated_df = pd.DataFrame(
            up_targeting_drugs_disease_associated
        )
        logger.info(f"Created DataFrame with shape: {up_targeting_drugs_disease_associated_df.shape}")
    else:
        logger.warning(
            "⚠ Patient prioritized genes file not found or invalid - creating empty up_drugs file"
        )
        # Create empty DataFrame with proper schema when patient genes fail
        up_targeting_drugs_disease_associated_df = pd.DataFrame(
            columns=DrugColumnConfig.get_all_expected_columns()
        )
        logger.info(f"Created empty DataFrame with schema columns: {len(up_targeting_drugs_disease_associated_df.columns)}")

    # Always save the file (populated or empty with schema) - FORCE OVERWRITE
    try:
        up_targeting_drugs_disease_associated_df.to_csv(up_drugs_file, index=False)
        logger.info(f"✓ Successfully wrote up_drugs file: {up_drugs_file} (size: {up_drugs_file.stat().st_size} bytes)")
    except Exception as save_error:
        logger.error(f"Failed to save up_drugs file: {save_error}")
        # Emergency fallback
        pd.DataFrame(columns=DrugColumnConfig.get_all_expected_columns()).to_csv(up_drugs_file, index=False)
        logger.warning(f"Created emergency up_drugs file: {up_drugs_file}")

    # Step 3: Process pathways to find drugs (both matching and non-matching)

    # Check if output file already exists and has valid schema
    if output_file.exists():
        try:
            existing_df = pd.read_csv(output_file)
            is_valid, missing_cols = DrugColumnConfig.validate_critical_columns(existing_df)
            if not existing_df.empty and is_valid:
                logger.info(f"Output file {output_file} already exists with valid data. Skipping pipeline execution.")
                return Path(output_file).resolve()
            else:
                logger.warning(f"Existing output file is empty or missing critical columns: {missing_cols}. Re-running pipeline.")
        except Exception as e:
            logger.warning(f"Error reading existing output file: {str(e)}. Will proceed with pipeline execution.")

    logger.info(f"Loading pathways from {input_file}")
    pathways_df = pipeline.filter_pathways_file(
        str(input_file), max_rows=None)  # Process all pathways

    if pathways_df.empty:
        logger.error("No pathways found in the data")
        # Create empty results file and return path
        empty_df = pd.DataFrame()
        pipeline.save_results(empty_df, str(output_file))
        return Path(output_file).resolve()

    logger.info(f"Starting to process {len(pathways_df)} pathways...")

    if patient_genes_loaded:
        logger.info(
            "Note: Will check pathway genes first, then patient prioritized genes for matches")
    else:
        logger.info(
            "Note: All drugs associated with pathways will be included, regardless of gene overlap")

    start_time = time.time()
    pathway_results_df = pipeline.process_pathways(pathways_df)
    logger.info(f"PATHWAY DRUG DISCOVERY PIPELINE - PATHWAY RESULTS: {pathway_results_df.shape}")
    
    # Step 3b: Get drugs by patient prioritized genes (direct targeting)
    if patient_genes_loaded:
        logger.info("Finding drugs that directly target patient prioritized genes...")
        # Use only pathway results for the main pipeline
        results_df = pathway_results_df
    else:
        results_df = pathway_results_df
        logger.warning("Patient genes not loaded - skipping direct gene targeting drug search")
    
    logger.info(f"PATHWAY DRUG DISCOVERY PIPELINE - FINAL RESULTS: {results_df.columns}, {results_df.shape}")

    end_time = time.time()

    logger.info(f"Processing completed in {end_time - start_time:.1f} seconds")

    # Step 4: Filter log2FC data
    logger.info("Filtering log2FC data...")
    results_df = filter_log2fc_data(results_df)
    logger.info(f"After filtering log2FC result data: {results_df.shape}")

    end_time = time.time()
    logger.info(f"Filtering completed in {end_time - start_time:.1f} seconds")  
   
    start_time = time.time()
    # Step 5: Apply LLM-based drug evaluation
    logger.info("Applying LLM-based drug evaluation...")
    evaluated_results_df = None  # Initialize to prevent undefined variable errors
    
    try:
        logger.info(f"Result df: {results_df.shape}, Columns: {results_df.columns}")

        # Check if results_df is empty
        if results_df.empty or results_df.shape[0] == 0:
            logger.warning("No drug discovery results found. Skipping filtering and evaluation steps.")
            evaluated_results_df = results_df  # Return empty dataframe
        else:
            results_df = filter_and_deduplicate_results(results_df)

            # Apply Smart Parallel Prioritization (auto-detects Celery environments)
            try:
                # Use the enhanced parallel_drug_prioritizer which auto-detects environment
                from .parallel_drug_prioritizer import run_parallel_drug_prioritization
                logger.info("Using smart parallel prioritization (auto-detects Celery)")
                
                evaluated_results_df = run_parallel_drug_prioritization(
                    df=results_df,
                    disease_name=disease_name,
                    model="gpt-5-mini-2025-08-07",
                    max_workers=10,
                    use_threads=None  # Auto-detect based on environment
                )
            except Exception as e:
                logger.warning(f"Parallel prioritization failed: {e}")
                # Fallback to original sequential implementation
                logger.info("Falling back to sequential prioritization")
                try:
                    evaluated_results_df = run_universal_prioritization(
                        data_input=results_df,
                        disease_name=disease_name,
                        model="gpt-5-mini-2025-08-07"
                    )
                except Exception as fallback_e:
                    logger.error(f"Sequential prioritization also failed: {fallback_e}")
                    evaluated_results_df = results_df  # Use original results as final fallback
        logger.info("✓ LLM drug evaluation completed successfully")

        logger.info(f"Evaluated results: {evaluated_results_df.shape} Columns: {evaluated_results_df.columns}")
        

    except Exception as e:
        logger.warning(f"LLM drug evaluation failed: {e}")
        logger.warning("Continuing with original results...")
        evaluated_results_df = results_df  # Ensure variable is always set

    end_time = time.time()

    # Final safety check - ensure evaluated_results_df is never None
    if evaluated_results_df is None:
        logger.warning("evaluated_results_df is None - using original results as fallback")
        evaluated_results_df = results_df

    # Handle empty DataFrame case - create with required columns
    if evaluated_results_df.empty or evaluated_results_df.shape[0] == 0:
        logger.warning("No drugs found in results. Creating empty DataFrame with required columns.")
        
        evaluated_results_df = pd.DataFrame(columns=DrugColumnConfig.get_all_expected_columns())
    else:
        # Use centralized column configuration
        
        
        # Add any missing columns using centralized configuration
        evaluated_results_df = DrugColumnConfig.add_missing_columns(evaluated_results_df, preserve_existing=True)
        
        # Log results
        final_columns = list(evaluated_results_df.columns)
        logger.info(f"Final results preserving all columns: {evaluated_results_df.shape} with columns: {len(final_columns)}")
        logger.info(f"Preserved critical columns: recommendation={evaluated_results_df.get('recommendation') is not None}")

    logger.info(f"Drugs Final Results Saving to File Shape:: {evaluated_results_df.shape}")
    logger.info(f"LLM evaluation completed in {end_time - start_time:.1f} seconds")

    # Step 6: Save results (including LLM evaluation if successful) - ALWAYS guarantee file creation
    try:
        pipeline.save_results(evaluated_results_df, str(output_file))
        logger.info(f"✓ Guaranteed output file created: {output_file}")
    except Exception as save_error:
        logger.error(f"Failed to save results: {save_error}")
        # Create emergency fallback file with proper schema
        emergency_df = pd.DataFrame(columns=DrugColumnConfig.get_all_expected_columns())
        emergency_df.to_csv(str(output_file), index=False)
        logger.warning(f"Created emergency empty file with proper schema: {output_file}")

    # Step 7: Print summary
    logger.info(f"Processing completed in {end_time - start_time:.1f} seconds")

    # Final validation - ensure file exists before returning
    if not output_file.exists():
        logger.error(f"Critical error: Output file {output_file} does not exist after pipeline completion")
        # Create final emergency file
        pd.DataFrame(columns=DrugColumnConfig.get_all_expected_columns()).to_csv(str(output_file), index=False)
        logger.warning(f"Created final emergency file: {output_file}")

    return Path(output_file).resolve()


if __name__ == "__main__":
    output_file = main(analysis_id="4dd3c5c2-f5af-4b62-8335-0ff1901619cf")
    print(f"Output file: {output_file}")
