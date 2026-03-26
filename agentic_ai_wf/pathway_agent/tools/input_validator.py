import logging
import requests
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

from pydantic import BaseModel
import pandas as pd

from agents import Agent, Runner, function_tool
from agents import output_guardrail, OutputGuardrailTripwireTriggered, GuardrailFunctionOutput
from agents import RunContextWrapper

from ..helpers import setup_logger, logger

# logger = setup_logger("InputProcessingAgent")


class InputContext(BaseModel):
    raw_genes: list[str] 
    valid_genes: list[str] 
    invalid_genes: list[str] 



@function_tool
def read_deg_csv(path: str) -> list[str]:
    """
    Reads a CSV file containing differential expression analysis (DEA) results 
    and extracts a list of gene symbols from the 'Gene' column.

    Args:
        path (str): Path to the input CSV file. The file must contain a column named 'Gene'.

    Returns:
        list[str]: A list of gene symbols as strings, with missing values removed.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        pd.errors.EmptyDataError: If the file is empty.
        KeyError: If the 'Gene' column is missing from the file.

    Notes:
        - Converts gene symbols to string format and drops any NaN values.
        - Logs the number of gene symbols successfully loaded.

    Example:
        genes = read_deg_csv("results/DEG_output.csv")
        print(genes[:5])  # ['TP53', 'BRCA1', 'EGFR', ...]
    """
    try:
        df = pd.read_csv(path)
        genes_column = df['Gene']
        genes = genes_column.dropna().astype(str).tolist()
        logger.info(f"Loaded {len(genes)} genes from CSV")
        return genes
    except Exception as e:
        logger.error(f"Error reading CSV file: {e}")
        raise e


@function_tool
def validate_genes(
    genes: list[str]
) -> InputContext:
    """
    Validates a list of gene symbols using the NCBI Gene database with threading for Celery compatibility.

    This function checks whether each gene symbol in the input list exists in the 
    NCBI Gene database via the Clinical Tables API. It performs the validation using 
    ThreadPoolExecutor to optimize for performance and API responsiveness while being 
    compatible with Celery workers.

    Args:
        genes (list[str]): A list of gene symbols to validate.

    Returns:
        InputContext: An object containing:
            - raw_genes (list[str]): The original list of input gene symbols.
            - valid_genes (list[str]): Gene symbols found in the NCBI database.
            - invalid_genes (list[str]): Gene symbols not found in the NCBI database.

    Notes:
        - Uses requests for blocking HTTP requests with ThreadPoolExecutor for concurrency.
        - Processes genes in batches of 500 with a short delay between batches to avoid overloading the API.
        - Each gene is checked against the NCBI Clinical Tables API for presence.
        - A gene is considered valid if it returns at least one match.
        - Thread-safe for use in Celery workers.

    Example:
        result = validate_genes(["TP53", "BRCA1", "FAKEGENE"])
        print(result.valid_genes)   # ["TP53", "BRCA1"]
        print(result.invalid_genes) # ["FAKEGENE"]
    """
    base_url = "https://clinicaltables.nlm.nih.gov/api/ncbi_genes/v3/search"
    valid = []
    invalid = []
    count: int = 1
    batch_size = 500
    max_workers = 10  # Limit concurrent requests
    
    # Thread-safe locks for result lists
    valid_lock = Lock()
    invalid_lock = Lock()
    
    def check_gene(gene):
        """Check a single gene using synchronous requests."""
        try:
            # Query for up to `count` matches, requesting only the 'Symbol' display field
            params = {
                "terms": gene,
                "count": count,    
            }
            
            response = requests.get(base_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            total_matches = data[0]                           

            # If the exact gene appears among returned symbols, mark as valid
            if total_matches > 0:
                with valid_lock:
                    valid.append(gene)
            else:
                with invalid_lock:
                    invalid.append(gene)
                    
        except Exception as e:
            logger.warning(f"Error validating gene {gene}: {e}")
            # Treat errors as invalid genes
            with invalid_lock:
                invalid.append(gene)
    
    logger.info(f"Validating {len(genes)} genes using threading")
    
    # Process genes in batches using ThreadPoolExecutor
    for i in range(0, len(genes), batch_size):
        batch = genes[i:i + batch_size]
        batch_num = (i // batch_size) + 1
        total_batches = (len(genes) + batch_size - 1) // batch_size
        
        logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch)} genes)")
        
        # Use ThreadPoolExecutor for concurrent validation
        with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="GeneValidator") as executor:
            # Submit all genes in this batch
            futures = {executor.submit(check_gene, gene): gene for gene in batch}
            
            # Wait for all to complete
            for future in as_completed(futures):
                gene = futures[future]
                try:
                    future.result()  # This will raise an exception if check_gene failed
                except Exception as e:
                    logger.error(f"Unexpected error in gene validation for {gene}: {e}")
        
        # Add a small delay between batches to be extra safe with API rate limits
        if i + batch_size < len(genes):  # Don't sleep after the last batch
            time.sleep(0.1)
    
    logger.info(f"Validation done: {len(valid)} valid / {len(invalid)} invalid")
    
    return InputContext(raw_genes=genes, valid_genes=valid, invalid_genes=invalid)

