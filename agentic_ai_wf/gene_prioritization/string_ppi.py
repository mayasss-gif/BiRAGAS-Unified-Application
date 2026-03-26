import requests
import pandas as pd
from io import StringIO
from functools import lru_cache
import traceback
import networkx as nx
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import time

from .helpers import logger

_STRING_STATS = {
    "requests_total": 0,
    "requests_failed": 0,
    "requests_retried": 0,
    "rate_limit_sleeps": 0,
}
_RATE_LOCK = Lock()
_LAST_REQUEST_TS = 0.0
_MIN_REQUEST_INTERVAL = 0.25  # 4 req/sec per process


def _rate_limit() -> None:
    global _LAST_REQUEST_TS
    with _RATE_LOCK:
        now = time.monotonic()
        wait = _MIN_REQUEST_INTERVAL - (now - _LAST_REQUEST_TS)
        if wait > 0:
            _STRING_STATS["rate_limit_sleeps"] += 1
            time.sleep(wait)
        _LAST_REQUEST_TS = time.monotonic()




def _post_with_retries(url, params, max_retries=3, timeout=(5, 30)):
    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            _rate_limit()
            _STRING_STATS["requests_total"] += 1
            response = requests.post(url, data=params, timeout=timeout)
            if response.status_code in {429, 500, 502, 503, 504, 524}:
                raise requests.HTTPError(
                    f"HTTP {response.status_code} from STRING API",
                    response=response,
                )
            response.raise_for_status()
            return response
        except Exception as exc:
            last_error = exc
            _STRING_STATS["requests_failed"] += 1
            _STRING_STATS["requests_retried"] += 1
            sleep_time = min(2 ** attempt, 10)
            logger.warning(
                "STRING API request failed (attempt %s/%s): %s",
                attempt,
                max_retries,
                exc,
            )
            time.sleep(sleep_time)
    logger.error("STRING API request failed after retries: %s", last_error)
    return None


# Fetch STRING IDs with exception handling
@lru_cache(maxsize=128)
def fetch_string_ids_cached(gene_key, species):
    url = "https://string-db.org/api/tsv/get_string_ids"
    params = {
        "identifiers": gene_key,
        "species": species,
        "limit": 1,
        "echo_query": 1
    }
    response = _post_with_retries(url, params)
    if response is None:
        return ""
    return response.text

def get_string_ids(gene_list, species=9606):
    try:
        
        gene_key = "%0d".join(gene_list)
        response_text = fetch_string_ids_cached(gene_key, species)
        if not response_text.strip():
            logger.warning("STRING ID mapping unavailable; skipping PPI.")
            return {}
        df = pd.read_csv(StringIO(response_text), sep="\t")
        id_map = dict(zip(df["queryItem"], df["stringId"]))
        
        logger.info(f"Mapped {len(id_map)} genes to STRING IDs.")
        return id_map
    except Exception as e:
        logger.error(f"Error fetching STRING IDs: {e}\n{traceback.format_exc()}")
        return {}

# Retrieve interactions
@lru_cache(maxsize=128)
def fetch_interactions_cached(string_key, species):
    url = "https://string-db.org/api/tsv/network"
    params = {"identifiers": string_key, "species": species}
    response = _post_with_retries(url, params)
    if response is None:
        return ""
    return response.text

def get_interactions_batch(batch_data):
    """Process a single batch of STRING IDs"""
    batch, species, batch_num = batch_data
    try:
        start_time = time.time()
        string_key = "%0d".join(batch)
        response_text = fetch_interactions_cached(string_key, species)
        if not response_text.strip():
            logger.warning(f"Batch {batch_num}: empty STRING response")
            return pd.DataFrame()
        batch_df = pd.read_csv(StringIO(response_text), sep="\t")
        
        processing_time = time.time() - start_time
        logger.info(f"Batch {batch_num}: Retrieved {len(batch_df)} interactions in {processing_time:.2f}s")
        return batch_df
    except Exception as e:
        logger.error(f"Error retrieving interactions for batch {batch_num}: {e}")
        return pd.DataFrame()

def get_interactions(string_ids, species=9606, max_workers=5):
    """Retrieve interactions using parallel processing for better performance"""
    try:
        if not string_ids:
            logger.warning("No STRING IDs provided; skipping interactions.")
            return pd.DataFrame()
        batch_size = 200
        total_batches = (len(string_ids) + batch_size - 1) // batch_size
        logger.info(f"Retrieving STRING PPI interactions for {len(string_ids)} IDs in {total_batches} batches using {max_workers} workers")
        
        # Prepare batch data for parallel processing
        batch_data = []
        for i in range(0, len(string_ids), batch_size):
            batch = string_ids[i:i+batch_size]
            batch_num = (i // batch_size) + 1
            batch_data.append((batch, species, batch_num))
        
        # Use ThreadPoolExecutor for parallel batch processing
        all_interactions = []
        successful_batches = 0
        
        with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="StringAPI") as executor:
            # Submit all batch jobs
            future_to_batch = {executor.submit(get_interactions_batch, data): data[2] for data in batch_data}
            
            # Collect results as they complete
            for future in as_completed(future_to_batch):
                batch_num = future_to_batch[future]
                try:
                    batch_df = future.result()
                    if not batch_df.empty:
                        all_interactions.append(batch_df)
                        successful_batches += 1
                except Exception as e:
                    logger.error(f"Exception in batch {batch_num}: {e}")
        
        # Combine all batch results
        if all_interactions:
            interactions_df = pd.concat(all_interactions, ignore_index=True)
        else:
            interactions_df = pd.DataFrame()
            
        logger.info(f"Retrieved {len(interactions_df)} total interactions from {successful_batches}/{total_batches} successful batches")
        return interactions_df
    except Exception as e:
        logger.error(f"Error retrieving interactions: {e}\n{traceback.format_exc()}")
        return pd.DataFrame()

# Build graph from interactions
def build_graph(interactions_df):
    try:
        logger.info("Building graph...")
        if interactions_df.empty:
            logger.warning("No interactions available; returning empty graph.")
            return nx.Graph()
        G = nx.Graph()
        for _, row in interactions_df.iterrows():
            G.add_edge(row["preferredName_A"], row["preferredName_B"], weight=row["score"])
        logger.info(f"Graph built with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
        return G
    except Exception as e:
        logger.error(f"Error building graph: {e}\n{traceback.format_exc()}")
        return nx.Graph()



def compute_degrees(G):
    try:
        degrees = dict(G.degree())
        if not degrees:
            logger.warning("Empty graph; no degrees computed.")
            return pd.DataFrame(columns=["Gene", "Degree", "Category"])
        degree_df = pd.DataFrame(degrees.items(), columns=["Gene", "Degree"]).sort_values(by="Degree", ascending=False)
        quantiles = degree_df["Degree"].quantile([0.33, 0.66])

        def categorize(degree):
            if degree >= quantiles[0.66]: return "HIGH"
            elif degree >= quantiles[0.33]: return "Medium"
            else: return "LOW"

        degree_df["Category"] = degree_df["Degree"].apply(categorize)
        logger.info("Node degrees computed and categorized.")
        return degree_df
    except Exception as e:
        logger.error(f"Error computing degrees: {e}\n{traceback.format_exc()}")
        return pd.DataFrame(columns=["Gene", "Degree", "Category"])


def compute_gene_ppi_metrics(gene_batch_data):
    """Compute PPI metrics for a batch of genes"""
    gene_batch, filtered_df, batch_num = gene_batch_data
    try:
        start_time = time.time()
        batch_results = []
        
        for gene in gene_batch:
            # Find all interactions where gene is A or B
            relevant = filtered_df[
                (filtered_df["preferredName_A"] == gene) |
                (filtered_df["preferredName_B"] == gene)
            ]
            degree = len(relevant)
            avg_score = relevant["score"].mean() if not relevant.empty else 0
            ppi_score = degree * avg_score
            batch_results.append((gene, degree, avg_score, ppi_score))
        
        processing_time = time.time() - start_time
        logger.info(f"PPI Batch {batch_num}: Processed {len(gene_batch)} genes in {processing_time:.2f}s")
        return batch_results
    except Exception as e:
        logger.error(f"Error processing PPI batch {batch_num}: {e}")
        return []

def compute_ppi_metrics(interactions_df, degs, max_workers=8):
    """Compute PPI metrics using parallel processing for better performance"""
    try:
        if interactions_df.empty or not degs:
            logger.warning("No interactions or genes; skipping PPI metrics.")
            return pd.DataFrame(
                columns=["Gene", "PPI_Degree", "PPI_Avg_Score", "PPI_Score"]
            )
        logger.info(f"Computing STRING PPI metrics for {len(degs)} genes using {max_workers} workers...")

        # Filter interactions where both nodes are in DEGs (symmetric filtering)
        filtered_df = interactions_df[
            interactions_df["preferredName_A"].isin(degs) &
            interactions_df["preferredName_B"].isin(degs)
        ]

        logger.info(f"Filtered interactions: {len(filtered_df)} (down from {len(interactions_df)})")

        # Prepare gene batches for parallel processing
        batch_size = max(50, len(degs) // max_workers)  # Ensure reasonable batch size
        gene_batches = []
        for i in range(0, len(degs), batch_size):
            gene_batch = degs[i:i+batch_size]
            batch_num = (i // batch_size) + 1
            gene_batches.append((gene_batch, filtered_df, batch_num))

        total_batches = len(gene_batches)
        logger.info(f"Processing {len(degs)} genes in {total_batches} batches of ~{batch_size} genes each")

        # Use ThreadPoolExecutor for parallel gene processing
        all_results = []
        successful_batches = 0

        with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="PPICompute") as executor:
            # Submit all batch jobs
            future_to_batch = {executor.submit(compute_gene_ppi_metrics, data): data[2] for data in gene_batches}
            
            # Collect results as they complete
            for future in as_completed(future_to_batch):
                batch_num = future_to_batch[future]
                try:
                    batch_results = future.result()
                    if batch_results:
                        all_results.extend(batch_results)
                        successful_batches += 1
                except Exception as e:
                    logger.error(f"Exception in PPI batch {batch_num}: {e}")

        logger.info(f"Successfully processed {successful_batches}/{total_batches} PPI computation batches")

        # Create DataFrame from all results
        ppi_df = pd.DataFrame(all_results, columns=["Gene", "PPI_Degree", "PPI_Avg_Score", "PPI_Score"])
        return ppi_df.sort_values(by="PPI_Score", ascending=False)

    except Exception as e:
        logger.error(f"Error computing PPI metrics: {e}\n{traceback.format_exc()}")
        return pd.DataFrame(
            columns=["Gene", "PPI_Degree", "PPI_Avg_Score", "PPI_Score"]
        )


def string_ppi_lookup(df):
    string_ids = get_string_ids(df['Gene'].tolist())
    if not string_ids:
        logger.warning("STRING ID mapping failed; returning input without PPI.")
        logger.info("STRING API stats: %s", _STRING_STATS)
        return _with_empty_ppi(df)
    interactions_df = get_interactions(list(string_ids.values()))
    if interactions_df.empty:
        logger.warning("STRING interactions unavailable; returning input without PPI.")
        logger.info("STRING API stats: %s", _STRING_STATS)
        return _with_empty_ppi(df)
    G = build_graph(interactions_df)
    degree_df = compute_degrees(G)
    if degree_df.empty:
        logger.warning("PPI degree computation empty; returning input without PPI.")
        logger.info("STRING API stats: %s", _STRING_STATS)
        return _with_empty_ppi(df)
    ppi_df = compute_ppi_metrics(interactions_df, degree_df['Gene'].tolist())
    # print(ppi_df.head())

    combined_df = pd.merge(df, ppi_df, on='Gene', how='left')
    logger.info("STRING API stats: %s", _STRING_STATS)
    return combined_df


def _with_empty_ppi(df):
    result = df.copy()
    for col in ["PPI_Degree", "PPI_Avg_Score", "PPI_Score"]:
        if col not in result.columns:
            result[col] = None
    return result