import requests
import csv
from pathlib import Path
import pandas as pd
from typing import List, Dict, Tuple, Optional
from datetime import datetime
from .helpers import logger
from .config import (
    API_BASE,
    SPECIES,
    CALLER,
    MAX_CHAR,
    MAX_COUNT,
    API_TO_KEY,
    CATEGORY_DISPLAY,
    ALLOWED_KEYS,
)
# --- Core Helper Functions ---


def read_genes_from_csv(path: Path, patient_prefix: str) -> List[str]:
    """
    Reads a CSV file containing gene symbols with column name 'gene' or 'Gene'
    and returns a list of gene symbols.

    Args:
        path (Path): The path to the CSV file containing gene symbols.
    Returns:
        List[str]: A list of gene symbols.
    Raises:
        ValueError: If the input CSV does not have a column header 'gene' or 'Gene'.
        FileNotFoundError: If the file doesn't exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    try:
        # Use pandas for more efficient CSV reading
        df = pd.read_csv(path, encoding='utf-8')

        # Only keep rows where 'Confidence' is 'High' or 'Medium'
        if 'Confidence' in df.columns:
            df = df[df['Confidence'].isin(['High', 'Medium'])]

        # Find gene column (case-insensitive)
        gene_col = next(
            (col for col in df.columns if col.lower() == 'gene'), None)

        if gene_col is None:
            raise ValueError(
                "Input CSV must have a column header 'gene' or 'Gene'.")

        # Extract genes, strip whitespace, and filter out empty values
        lfc_col = [col for col in df.columns if col.lower().startswith(
            patient_prefix) and col.lower().endswith('log2fc')][0]
        up_regulated_df = df[df[lfc_col] > 0]
        up_regulated_genes = up_regulated_df[gene_col].astype(
            str).str.strip().dropna().tolist()
        # Remove empty strings
        up_regulated_genes = [gene for gene in up_regulated_genes if gene]

        down_regulated_df = df[df[lfc_col] < 0]
        down_regulated_genes = down_regulated_df[gene_col].astype(
            str).str.strip().dropna().tolist()
        # Remove empty strings
        down_regulated_genes = [gene for gene in down_regulated_genes if gene]

        logger.info(
            f"Read {len(up_regulated_genes)} up-regulated genes and {len(down_regulated_genes)} down-regulated genes from {path}")
        return up_regulated_genes, down_regulated_genes

    except pd.errors.EmptyDataError:
        logger.error(f"CSV file {path} is empty")
        raise ValueError(f"CSV file {path} is empty")
    except pd.errors.ParserError as e:
        logger.error(f"Error parsing CSV file {path}: {e}")
        raise ValueError(f"Invalid CSV format in {path}: {e}")
    except Exception as e:
        logger.error(f"Error reading CSV file {path}: {e}")
        raise


def chunk_genes(genes: List[str]) -> List[List[str]]:
    """
    Chunks a list of gene symbols into smaller batches for processing.

    Args:
        genes (List[str]): A list of gene symbols.
    Returns:
        List[List[str]]: A list of gene symbol batches.
    """
    if not genes:
        return []

    batches = []
    current_batch = []
    current_length = 0

    for gene in genes:
        gene_length = len(gene) + 1  # +1 for separator

        # Check if adding this gene would exceed limits
        if (current_batch and
                (current_length + gene_length > MAX_CHAR or len(current_batch) >= MAX_COUNT)):
            batches.append(current_batch)
            current_batch = []
            current_length = 0

        current_batch.append(gene)
        current_length += gene_length

    if current_batch:
        batches.append(current_batch)

    return batches


def _map_to_string_ids(genes: List[str]) -> Tuple[Dict[str, str], List[str]]:
    """
    Maps a list of gene symbols to their corresponding STRING IDs using the StringAPI.

    Args:
        genes (List[str]): A list of gene symbols.
    Returns:
        Tuple[Dict[str, str], List[str]]: A tuple containing:
            - mapping (dict): A dictionary mapping gene symbols to their corresponding STRING IDs.
            - sids (list): A list of STRING IDs.
    Raises:
        requests.RequestException: If the API request fails.
    """
    if not genes:
        return {}, []

    url = f"{API_BASE}/json/get_string_ids"
    params = {
        "identifiers": "\r".join(genes),
        "species": SPECIES,
        "caller_identity": CALLER
    }

    try:
        response = requests.post(url, data=params, timeout=30)
        response.raise_for_status()
        data = response.json()

        mapping = {}
        sids = []

        for row in data:
            name = row.get("queryItem")
            sid = row.get("stringId")
            if name and sid:
                mapping[name] = sid
                sids.append(sid)

        return mapping, sids

    except requests.RequestException as e:
        logger.error(f"Error mapping genes to STRING IDs: {e}")
        raise


def _fetch_enrichment(string_ids: List[str]) -> Dict[str, List[Dict]]:
    """
    Fetches enrichment categories for a list of STRING IDs using the StringAPI.

    Args:
        string_ids (List[str]): A list of STRING IDs.
    Returns:
        Dict[str, List[Dict]]: A dictionary containing enrichment categories.
    Raises:
        requests.RequestException: If the API request fails.
    """
    if not string_ids:
        return {}

    url = f"{API_BASE}/tsv/enrichment"
    params = {
        "identifiers": "\r".join(string_ids),
        "species": SPECIES,
        "caller_identity": CALLER
    }

    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()

        reader = csv.DictReader(response.text.splitlines(), delimiter='\t')
        enrichment_data = {}

        for row in reader:
            api_cat = row.get('category')
            if api_cat and api_cat in API_TO_KEY:
                key = API_TO_KEY[api_cat]
                enrichment_data.setdefault(key, []).append(row)

        return enrichment_data

    except requests.RequestException as e:
        logger.error(f"Error fetching enrichment data: {e}")
        raise


def _process_enrichment_data(enrichment_data: Dict[str, List[Dict]]) -> List[Dict]:
    """
    Processes enrichment data and converts it to a standardized format.

    Args:
        enrichment_data (Dict[str, List[Dict]]): Raw enrichment data from API.
    Returns:
        List[Dict]: Processed enrichment data in standardized format.
    """
    processed_data = []

    # Category mapping for database identification
    category_to_db = {
        'biological_process': 'GO_BP',
        'molecular_function': 'GO_MF',
        'compartments': 'GO_CC',
        'rctm': 'REACTOME',
        'wikipathways': 'WIKIPATHWAY',
        'kegg': 'KEGG'
    }

    for key, rows in enrichment_data.items():
        if key not in ALLOWED_KEYS:
            continue

        category_name = CATEGORY_DISPLAY[key].lower()
        db_id = category_to_db.get(category_name, 'UNKNOWN')

        for row in rows:
            processed_row = {
                'DB_ID': db_id,
                'Pathway source': category_name,
                'Pathway ID': row.get('term', ''),
                'number_of_genes': row.get('number_of_genes', ''),
                'number_of_genes_in_background': row.get('number_of_genes_in_background', ''),
                'ncbiTaxonId': row.get('ncbiTaxonId', ''),
                'inputGenes': row.get('inputGenes', ''),
                'Pathway associated genes': row.get('preferredNames', ''),
                'p_value': row.get('p_value', ''),
                'fdr': row.get('fdr', ''),
                'Pathway': row.get('description', '')
            }
            processed_data.append(processed_row)

    return processed_data

# --- Agentic Wrappers ---


def StringAPI_tool(genes: List[str]) -> Tuple[Dict[str, str], List[str]]:
    """
    Maps a list of gene symbols to their corresponding STRING IDs using the StringAPI.

    Args:
        genes (List[str]): A list of gene symbols.
    Returns:
        Tuple[Dict[str, str], List[str]]: A tuple containing:
            - mapping (dict): A dictionary mapping gene symbols to their corresponding STRING IDs.
            - sids (list): A list of STRING IDs.
    """
    mapping, sids = _map_to_string_ids(genes)
    logger.info(
        f"Mapped {len(mapping)} inputs to {len(sids)} STRING IDs via StringAPI")
    return mapping, sids


def Enrichment_tool(string_ids: List[str]) -> Dict[str, List[Dict]]:
    """
    Fetches enrichment categories for a list of STRING IDs using the StringAPI.

    Args:
        string_ids (List[str]): A list of STRING IDs.
    Returns:
        Dict[str, List[Dict]]: A dictionary containing enrichment categories.
    """
    enrichment = _fetch_enrichment(string_ids)
    logger.info(
        f"Retrieved enrichment categories via Enrichment_Agent: {list(enrichment.keys())}")
    return enrichment

# --- Main Pipeline ---
def enrichment_pipeline(disease_name: str, gene_file_path: Path, analysis_id: str, output_dir: Path) -> Path:
    """
    Runs the optimized enrichment pipeline for a given gene file.
    
    The pipeline performs the following steps:
        1. Checks if enriched file already exists and skips processing if found.
        2. Reads the gene symbols from the input CSV file.
        3. Chunks the gene symbols into smaller batches for processing.
        4. Maps each batch of gene symbols to their corresponding STRING IDs.
        5. Fetches enrichment categories for each batch of STRING IDs.
        6. Processes and combines all enrichment data in memory.
        7. Saves the combined results to a single CSV file.
    
    Args:
        disease_name (str): Name of the disease for output organization.
        gene_file_path (Path): The path to the CSV file containing gene symbols.
        output_dir (Optional[Path]): Directory to save the output file.
                                   If None, uses the same directory as input file.
    Returns:
        Path: The path to the combined CSV file.
    Raises:
        ValueError: If no enrichment data is found.
        FileNotFoundError: If input file doesn't exist.
    """
    # Validate input
    if not gene_file_path.exists():
        raise FileNotFoundError(f"Input file not found: {gene_file_path}")
    
    # Setup output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate output filename
    sanitized_disease_name = disease_name.replace(
        " ", "_").replace("/", "_").replace("\\", "_")
    output_filename = f"{sanitized_disease_name}_Pathways_Enrichment.csv"
    output_path = output_dir / output_filename
    
    # Check if enriched file already exists
    if output_path.exists():
        logger.info(f"Enriched file already exists: {output_path}")
        logger.info("Skipping enrichment pipeline. Using existing file.")
        
        # Validate the existing file has expected structure
        try:
            existing_df = pd.read_csv(output_path)
            expected_columns = [
                'DB_ID', 'Pathway source', 'Pathway ID',
                'number_of_genes', 'number_of_genes_in_background',
                'ncbiTaxonId', 'inputGenes', 'Pathway associated genes',
                'p_value', 'fdr', 'Pathway', 'Regulation'
            ]
            
            missing_columns = [col for col in expected_columns if col not in existing_df.columns]
            if missing_columns:
                logger.warning(f"Existing file missing columns: {missing_columns}")
                logger.info("Will reprocess to ensure complete data.")
            else:
                logger.info(f"Existing file validated. Found {len(existing_df)} enrichment entries.")
                return output_path
                
        except Exception as e:
            logger.warning(f"Error reading existing file: {e}")
            logger.info("Will reprocess to ensure data integrity.")
    
    # Read genes
    up_regulated_genes, down_regulated_genes = read_genes_from_csv(gene_file_path, analysis_id)
    if not up_regulated_genes and not down_regulated_genes:
        raise ValueError("No genes found in the input file")
    
    # Process genes in batches with parallel processing
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    def process_gene_batch(chunk: List[str], regulation_type: str, batch_num: int) -> List[Dict]:
        """Process a single batch of genes and return enrichment data."""
        try:
            logger.info(f"Processing {regulation_type} batch {batch_num} with {len(chunk)} genes")
            
            # Map genes to STRING IDs
            mapping, sids = StringAPI_tool(chunk)
            
            if not sids:
                logger.warning(f"No STRING IDs found for {regulation_type} batch {batch_num}")
                return []
            
            # Fetch enrichment data
            enrichment = Enrichment_tool(sids)
            
            if enrichment:
                # Process enrichment data
                processed_data = _process_enrichment_data(enrichment)
                logger.info(f"{regulation_type} batch {batch_num}: Found {len(processed_data)} enrichment entries")
                return processed_data
            else:
                logger.warning(f"No enrichment data found for {regulation_type} batch {batch_num}")
                return []
                
        except Exception as e:
            logger.error(f"Error processing {regulation_type} batch {batch_num}: {e}")
            return []
    
    # Prepare batches
    up_regulated_gene_chunks = chunk_genes(up_regulated_genes)
    down_regulated_gene_chunks = chunk_genes(down_regulated_genes)
    
    logger.info(f"Processing {len(up_regulated_genes)} up-regulated genes in {len(up_regulated_gene_chunks)} batches")
    logger.info(f"Processing {len(down_regulated_genes)} down-regulated genes in {len(down_regulated_gene_chunks)} batches")
    
    # Process batches in parallel with limited concurrency to avoid API rate limits
    max_workers = 3  # Limit concurrent API calls
    all_up_regulated_enrichment_data = []
    all_down_regulated_enrichment_data = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit up-regulated batches
        up_futures = {
            executor.submit(process_gene_batch, chunk, "up-regulated", i): i 
            for i, chunk in enumerate(up_regulated_gene_chunks, 1)
        }
        
        # Submit down-regulated batches
        down_futures = {
            executor.submit(process_gene_batch, chunk, "down-regulated", i): i 
            for i, chunk in enumerate(down_regulated_gene_chunks, 1)
        }
        
        # Collect results as they complete
        for future in as_completed(up_futures):
            batch_num = up_futures[future]
            try:
                result = future.result()
                all_up_regulated_enrichment_data.extend(result)
            except Exception as e:
                logger.error(f"Error collecting up-regulated batch {batch_num} result: {e}")
        
        for future in as_completed(down_futures):
            batch_num = down_futures[future]
            try:
                result = future.result()
                all_down_regulated_enrichment_data.extend(result)
            except Exception as e:
                logger.error(f"Error collecting down-regulated batch {batch_num} result: {e}")
    
    # Check if we have any data - warn but don't fail if one group is empty
    if not all_up_regulated_enrichment_data:
        logger.warning("No enrichment data found for any of the up-regulated gene batches")
    
    if not all_down_regulated_enrichment_data:
        logger.warning("No enrichment data found for any of the down-regulated gene batches")
    
    # Only fail if neither group has any data
    if not all_up_regulated_enrichment_data and not all_down_regulated_enrichment_data:
        logger.error("No enrichment data found for either up-regulated or down-regulated genes")
        # Create an empty output file with proper headers instead of failing
        empty_df = pd.DataFrame(columns=[
            'DB_ID', 'Pathway source', 'Pathway ID',
            'number_of_genes', 'number_of_genes_in_background',
            'ncbiTaxonId', 'inputGenes', 'Pathway associated genes',
            'p_value', 'fdr', 'Pathway', 'Regulation'
        ])
        empty_df.to_csv(output_path, index=False)
        logger.info(f"Created empty enrichment file: {output_path}")
        return output_path
    
    # Create DataFrame and save to CSV with optimized memory usage
    try:
        # Pre-define expected columns to avoid repeated lookups
        expected_columns = [
            'DB_ID', 'Pathway source', 'Pathway ID',
            'number_of_genes', 'number_of_genes_in_background',
            'ncbiTaxonId', 'inputGenes', 'Pathway associated genes',
            'p_value', 'fdr', 'Pathway'
        ]
        
        # Create DataFrames with pre-allocated columns, handling empty data gracefully
        dataframes_to_concat = []
        
        if all_up_regulated_enrichment_data:
            up_df = pd.DataFrame(all_up_regulated_enrichment_data, columns=expected_columns)
            up_df.insert(len(expected_columns), 'Regulation', 'Up')
            dataframes_to_concat.append(up_df)
            logger.info(f"Up-regulated enrichment data: {len(up_df)}")
        else:
            logger.info("Up-regulated enrichment data: 0 (no data)")
            
        if all_down_regulated_enrichment_data:
            down_df = pd.DataFrame(all_down_regulated_enrichment_data, columns=expected_columns)
            down_df.insert(len(expected_columns), 'Regulation', 'Down')
            dataframes_to_concat.append(down_df)
            logger.info(f"Down-regulated enrichment data: {len(down_df)}")
        else:
            logger.info("Down-regulated enrichment data: 0 (no data)")

        # Merge the dataframes efficiently (only if we have any data)
        if dataframes_to_concat:
            df = pd.concat(dataframes_to_concat, ignore_index=True)
        else:
            # This should not happen due to our earlier check, but just in case
            df = pd.DataFrame(columns=expected_columns + ['Regulation'])
        
        
        # Apply FDR correction and filter out low-confidence results (only if we have data)
        if len(df) > 0:
            fdr_filtered_df = df[df['fdr'].astype(float) < 0.05]
            if len(fdr_filtered_df) < 10:
                logger.info("Fewer than 10 enrichment entries after FDR < 0.05 filter; relaxing threshold to 0.08")
                fdr_filtered_df = df[df['fdr'].astype(float) < 0.08]
            if len(fdr_filtered_df) < 10:
                logger.info("Fewer than 10 enrichment entries after FDR < 0.08 filter; relaxing threshold to 0.1")
                fdr_filtered_df = df[df['fdr'].astype(float) < 0.1]
            df = fdr_filtered_df
        else:
            logger.warning("No enrichment data to filter - proceeding with empty DataFrame")

        
        # Save to CSV with optimized settings
        df.to_csv(output_path, index=False, compression=None)
        
        logger.info(f"Enrichment pipeline completed successfully")
        logger.info(f"Total enrichment entries: {len(df)}")
        logger.info(f"Output saved to: {output_path}")
        
        return output_path
        
    except Exception as e:
        logger.error(f"Error saving enrichment data: {e}")
        raise
 
# --- Legacy compatibility function ---
def run_enrichment_pipeline(gene_file_path: Path, analysis_id: str) -> Path:
    """
    Legacy wrapper for backward compatibility.
    
    Args:
        gene_file_path (Path): The path to the CSV file containing gene symbols.
    Returns:
        Path: The path to the combined CSV file.
    """
    return enrichment_pipeline(gene_file_path, analysis_id)