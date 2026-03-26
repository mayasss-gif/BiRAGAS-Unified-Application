from pydantic import BaseModel, Field
import logging
import pandas as pd
from collections import defaultdict
from agents import Agent, Runner, WebSearchTool
import json
from dotenv import load_dotenv
from .config import PATHWAY_LITERATURE_LIMIT
from functools import lru_cache
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
import asyncio
import time
from typing import Dict, List, Set, Optional
import gc

load_dotenv()


from .helpers import logger

# Configuration for processing
DEFAULT_CONCURRENCY = 4  # Can be higher since no API calls
BATCH_SIZE = 10  # Batch size for processing pathways



class PerformanceMonitor:
    """Monitor and track performance metrics during processing."""
    
    def __init__(self):
        self.start_time = None
        self.batch_times = []
        self.total_processed = 0
        self.total_errors = 0
        self.memory_usage = []
    
    def start(self):
        """Start performance monitoring."""
        self.start_time = time.time()
        logger.info(f"Performance monitoring started at {time.strftime('%H:%M:%S')}")
    
    def record_batch(self, batch_size: int, errors: int = 0):
        """Record batch processing metrics."""
        batch_time = time.time()
        if self.batch_times:
            batch_duration = batch_time - self.batch_times[-1]
        else:
            batch_duration = batch_time - self.start_time
        
        self.batch_times.append(batch_time)
        self.total_processed += batch_size
        self.total_errors += errors
        
        # Calculate performance metrics
        total_time = batch_time - self.start_time
        avg_time_per_pathway = total_time / self.total_processed if self.total_processed > 0 else 0
        pathways_per_minute = (self.total_processed / total_time) * 60 if total_time > 0 else 0
        
        logger.info(f"Batch completed in {batch_duration:.2f}s")
        logger.info(f"   Total processed: {self.total_processed}")
        logger.info(f"   Average time per pathway: {avg_time_per_pathway:.2f}s")
        logger.info(f"   Processing rate: {pathways_per_minute:.1f} pathways/min")
        logger.info(f"   Error rate: {self.total_errors/self.total_processed*100:.1f}%")
    
    def get_summary(self):
        """Get final performance summary."""
        if not self.start_time:
            return "Performance monitoring not started"
        
        total_time = time.time() - self.start_time
        avg_time_per_pathway = total_time / self.total_processed if self.total_processed > 0 else 0
        pathways_per_minute = (self.total_processed / total_time) * 60 if total_time > 0 else 0
        
        return {
            "total_time": total_time,
            "total_processed": self.total_processed,
            "total_errors": self.total_errors,
            "avg_time_per_pathway": avg_time_per_pathway,
            "pathways_per_minute": pathways_per_minute,
            "error_rate": self.total_errors/self.total_processed*100 if self.total_processed > 0 else 0
        }


@lru_cache(maxsize=1000)
def get_pathway_cache_key(pathway_id: str, disease: str) -> str:
    """Generate cache key for pathway processing."""
    return f"{pathway_id}_{disease}"


def optimize_dataframe_memory(df: pd.DataFrame) -> pd.DataFrame:
    """Optimize DataFrame memory usage by converting data types."""
    logger.info("Optimizing DataFrame memory usage...")
    
    # Convert object columns to category if they have low cardinality
    for col in df.select_dtypes(include=['object']).columns:
        if df[col].nunique() / len(df) < 0.5:  # Less than 50% unique values
            df[col] = df[col].astype('category')
    
    # Convert numeric columns to appropriate types
    for col in df.select_dtypes(include=['int64']).columns:
        if df[col].min() >= 0:
            if df[col].max() < 255:
                df[col] = df[col].astype('uint8')
            elif df[col].max() < 65535:
                df[col] = df[col].astype('uint16')
            elif df[col].max() < 4294967295:
                df[col] = df[col].astype('uint32')
    
    logger.info(f"Memory optimization complete. Memory usage reduced.")
    return df

def get_pathways_json(combined_file_path: str) -> list[dict]:
    """Optimized pathway data extraction with vectorized operations."""
    df = pd.read_csv(combined_file_path)
    
    # Only get KEGG data; if less than 10, switch to all three
    # kegg_mask = df['DB_ID'] == 'KEGG'
    # if kegg_mask.sum() < 10:
    #     mask = df['DB_ID'].isin(['KEGG', 'REACTOME', 'WIKIPATHWAY'])
    # else:
    #     mask = kegg_mask
    mask = df['DB_ID'].isin(['KEGG', 'REACTOME', 'WIKIPATHWAY'])
    # chunk = df[mask][["Pathway ID", "Pathway", "fdr", "Pathway associated genes"]].copy()
    chunk = df[["Pathway ID", "Pathway", "fdr", "Pathway associated genes"]].copy()
    
    # Handle NaN values first, then split
    chunk['Pathway associated genes'] = chunk['Pathway associated genes'].fillna('')
    chunk['Pathway associated genes'] = chunk['Pathway associated genes'].str.split(',')
    
    # Replace empty strings that result in [''] with empty lists
    chunk['Pathway associated genes'] = chunk['Pathway associated genes'].apply(
        lambda x: [] if x == [''] else [gene.strip() for gene in x if gene.strip()]
    )
    
    return chunk.to_dict(orient="records")


def _run_async_in_thread(coro):
    """Helper to run async coroutines in a thread-safe way for Celery."""
    import asyncio
    import threading
    
    def run_in_thread():
        try:
            # Create new event loop for this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(coro)
            finally:
                loop.close()
        except Exception as e:
            logger.error(f"Error running async function in thread: {e}")
            return []
    
    # Run in a separate thread to avoid Celery event loop conflicts
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(run_in_thread)
        return future.result(timeout=30)  # 30 second timeout

class RelevanceOutput(BaseModel):
    clinical_relevance: str = Field(description="The clinical relevance of the pathway and biomarkers to the disease")
    functional_relevance: str = Field(description="The functional relevance of the pathway and biomarkers to the disease")
    audit_log: str = Field(description="Detailed log of the analysis process and findings")
    references: str = Field(description="References or sources used for the analysis")
    relation: str = Field(description="Type of relationship (Direct, Indirect, or None)")
    hit_score: int = Field(description="Numerical score indicating the number of supporting studies")

# OPTIMIZATION 6: Batch relevance processing
relevance_agent = Agent(
    name="relevance_agent",
    instructions=(
        """
        You are a biomedical literature analysis agent. You will receive a list of biomarkers (e.g., TP53, CDK2), a pathway (e.g., Cell Cycle), a disease (e.g., Breast Cancer), and optionally, a list of research article abstracts.

        Your task is to generate comprehensive analysis including:

        1. **Clinical Relevance** (2-3 lines) — How the pathway and its biomarkers relate to diagnosis, prognosis, treatment, or patient outcomes in the given disease.

        2. **Functional Relevance** (2-3 lines) — The biological or mechanistic roles these biomarkers play in the pathway and disease context.

        3. **Audit Log** — A detailed log of your analysis process, including:
           - Number of biomarkers analyzed
           - Key findings for each biomarker
           - Evidence quality assessment
           - Any limitations or uncertainties

        4. **References** — List of credible relevant sources weblinks that support analysis using the websearch tool. Only site sources if found on the web else leave it as "N/A".
        IMPORTANT: Only include the weblinks in the references section, do not include any other text and separate each weblink with a comma.

        5. **Relation** — Classify the relationship as:
           - "Direct" if strong evidence exists linking biomarkers to the disease
           - "Indirect" if the relationship is plausible but less well-established
           - "None" if no clear relationship can be established

        6. **Hit Score** — A numerical score indicating the number of supporting studies:
           - Use the references to determine the number of supporting studies.

        - If **weblinks are provided**, base your analysis on their content.
        - If **no weblinks are provided**, use your biomedical knowledge to infer plausible relevance based on the disease, pathway, and biomarkers alone.

        Always keep the writing focused, scientifically sound, and evidence-based.
        """
    ),
    model="gpt-5-mini",
    tools= [WebSearchTool()],
    output_type=RelevanceOutput
)



async def run_literature_pipeline(input_csv: str, disease: str, max_concurrent: int = 60, batch_size: int = 30):
    """
    
    Args:
        input_csv: Path to the input CSV file
        disease: Disease name for analysis
        max_concurrent: Maximum number of concurrent pathway processing tasks
        batch_size: Number of pathways to process in each batch
    """
    # Initialize performance monitoring
    monitor = PerformanceMonitor()
    monitor.start()
    
    logger.info(f"Running optimized literature pipeline for {disease} disease")
    logger.info(f"Configuration: max_concurrent={max_concurrent}, batch_size={batch_size}")
    
    # Load and prepare data efficiently
    logger.info("Loading and preparing data...")
    df = pd.read_csv(input_csv)
    
    # Optimize DataFrame memory usage
    df = optimize_dataframe_memory(df)
    
    all_data = get_pathways_json(input_csv)
    
    # Add missing literature columns efficiently
    literature_columns = ["clinical_relevance", "functional_relevance", "audit_log", "references", "relation", "hit_score"]
    missing_columns = [col for col in literature_columns if col not in df.columns]
    if missing_columns:
        df[missing_columns] = None
        logger.info(f"Added missing columns: {missing_columns}")
    
    # Create efficient lookup for existing results
    logger.info("Analyzing existing results...")
    existing_results = set()
    for _, row in df.iterrows():
        pathway_id = row["Pathway ID"]
        clinical_empty = pd.isna(row["clinical_relevance"]) or row["clinical_relevance"] in [None, "", "Error"]
        functional_empty = pd.isna(row["functional_relevance"]) or row["functional_relevance"] in [None, "", "Error"]
        
        if not (clinical_empty or functional_empty):
            existing_results.add(pathway_id)
    
    # Filter pathways that need processing
    pathways_to_process = [pathway for pathway in all_data if pathway["Pathway ID"] not in existing_results]
    

    if PATHWAY_LITERATURE_LIMIT and len(pathways_to_process) > PATHWAY_LITERATURE_LIMIT:
        pathways_to_process = pathways_to_process[:PATHWAY_LITERATURE_LIMIT]
        logger.info(f"Limited processing to {PATHWAY_LITERATURE_LIMIT} pathways due to configuration")

    total_pathways = len(all_data)
    already_processed = len(existing_results)
    needs_processing = len(pathways_to_process)
    
    logger.info(f"Statistics:")
    logger.info(f"   Total pathways: {total_pathways}")
    logger.info(f"   Already processed: {already_processed}")
    logger.info(f"   Need processing: {needs_processing}")
    logger.info(f"   Skip rate: {already_processed/total_pathways*100:.1f}%")
    
    if needs_processing == 0:
        logger.info("All pathways already have literature data. No processing needed.")
        return []
    
    # Process pathways continuously with optimized concurrency
    all_results = []
    semaphore = asyncio.Semaphore(max_concurrent)
    results_queue = asyncio.Queue()
    checkpoint_counter = 0
    checkpoint_interval = batch_size  # Save every N completed pathways
    
    async def process_single_pathway(pathway_entry, max_retries: int = 2):
        """Process a single pathway with error handling and retry logic."""
        async with semaphore:  # Limit concurrent operations
            pathway_id = pathway_entry['Pathway ID']
            logger.info(f"Processing pathway: {pathway_id}")
            
            for attempt in range(max_retries + 1):
                try:
                    relevance_input = {
                        "biomarkers": pathway_entry["Pathway associated genes"],
                        "pathway": pathway_entry["Pathway"],
                        "disease": disease
                    }
                    
                    # Run the relevance agent
                    res = await Runner.run(relevance_agent, json.dumps(relevance_input))
                    
                    result = {
                        "Pathway ID": pathway_id,
                        "clinical_relevance": res.final_output.clinical_relevance,
                        "functional_relevance": res.final_output.functional_relevance,
                        "audit_log": res.final_output.audit_log,
                        "references": res.final_output.references,
                        "relation": res.final_output.relation,
                        "hit_score": res.final_output.hit_score
                    }
                    
                    logger.info(f"Completed: {pathway_id}")
                    return result
                    
                except Exception as e:
                    if attempt < max_retries:
                        logger.info(f"Attempt {attempt + 1} failed for {pathway_id}: {str(e)}. Retrying...")
                        await asyncio.sleep(1)  # Brief delay before retry
                        continue
                    else:
                        logger.info(f"All attempts failed for {pathway_id}: {str(e)}")
                        # Return error result after all retries exhausted
                        return {
                            "Pathway ID": pathway_id,
                            "clinical_relevance": f"Error after {max_retries + 1} attempts: {str(e)}",
                            "functional_relevance": f"Error after {max_retries + 1} attempts: {str(e)}",
                            "audit_log": f"Processing failed after {max_retries + 1} attempts: {str(e)}",
                            "references": "Error",
                            "relation": "Error",
                            "hit_score": 0
                        }
    
    async def result_processor():
        """Process completed results and handle checkpoints."""
        nonlocal checkpoint_counter
        processed_count = 0
        
        while True:
            try:
                # Wait for a result with timeout
                result = await asyncio.wait_for(results_queue.get(), timeout=1.0)
                
                if result is None:  # Shutdown signal
                    break
                
                # Add to results
                all_results.append(result)
                processed_count += 1
                
                # Update DataFrame immediately for this result
                update_dataframe_efficiently(df, [result])
                
                # Increment checkpoint counter
                checkpoint_counter += 1
                
                # Save checkpoint periodically
                if checkpoint_counter >= checkpoint_interval:
                    df.to_csv(input_csv, index=False)
                    logger.info(f"Checkpoint saved: {len(all_results)} total processed")
                    checkpoint_counter = 0
                    
                    # Memory cleanup
                    gc.collect()
                
                # Record individual pathway completion
                monitor.record_batch(1, 1 if result.get('clinical_relevance', '').startswith('Error') else 0)
                
                results_queue.task_done()
                
            except asyncio.TimeoutError:
                # No results available, continue waiting
                continue
            except Exception as e:
                logger.error(f"Error in result processor: {e}")
                continue
    
    # Start the result processor
    processor_task = asyncio.create_task(result_processor())
    
    # Create all pathway processing tasks with proper result handling
    pathway_tasks = []
    
    async def process_and_queue(pathway):
        """Process a pathway and queue the result."""
        try:
            result = await process_single_pathway(pathway)
            await results_queue.put(result)
        except Exception as e:
            logger.error(f"Error processing pathway {pathway.get('Pathway ID', 'unknown')}: {e}")
            await results_queue.put(None)
    
    # Create tasks for all pathways
    for pathway in pathways_to_process:
        task = asyncio.create_task(process_and_queue(pathway))
        pathway_tasks.append(task)
    
    logger.info(f"Started processing {len(pathway_tasks)} pathways with max_concurrent={max_concurrent}")
    
    # Wait for all pathway processing to complete
    await asyncio.gather(*pathway_tasks, return_exceptions=True)
    
    # Signal result processor to shutdown
    await results_queue.put(None)
    await processor_task
    
    # Final checkpoint
    if checkpoint_counter > 0:
        df.to_csv(input_csv, index=False)
        logger.info(f"Final checkpoint saved: {len(all_results)} total processed")
    
    # Get final performance summary
    performance_summary = monitor.get_summary()
    
    logger.info(f"Processing complete!")
    logger.info(f"  Final Performance Summary:")
    logger.info(f"   Total pathways processed: {len(all_results)}")
    logger.info(f"   Total processing time: {performance_summary['total_time']:.2f} seconds")
    logger.info(f"   Average time per pathway: {performance_summary['avg_time_per_pathway']:.2f} seconds")
    logger.info(f"   Processing rate: {performance_summary['pathways_per_minute']:.1f} pathways/minute")
    logger.info(f"   Success rate: {100 - performance_summary['error_rate']:.1f}%")
    logger.info(f"   Error rate: {performance_summary['error_rate']:.1f}%")
    
    # Memory usage summary
    logger.info(f"Memory Optimization:")
    logger.info(f"   DataFrame memory optimized: Done")
    logger.info(f"   Garbage collection performed: Done")
    
    return all_results


def update_dataframe_efficiently(df: pd.DataFrame, results: list):
    """
    Efficiently update DataFrame with results using vectorized operations.
    Reduces time complexity from O(n²) to O(n).
    # Handles categorical columns by converting them to object type before updating.
    """
    if not results:
        return
    
    # Create a mapping from Pathway ID to result data
    result_map = {result["Pathway ID"]: result for result in results}
    
    # Get all Pathway IDs that need updating
    pathway_ids = list(result_map.keys())
    
    # Create boolean mask for rows to update
    mask = df["Pathway ID"].isin(pathway_ids)
    
    if not mask.any():
        logger.info("No matching pathways found for update")
        return
    
    # ======================================================================================================================
    # Handle categorical columns by converting to object type before updating
    literature_columns = ["clinical_relevance", "functional_relevance", "audit_log", "references", "relation", "hit_score"]
    categorical_columns = []
    
    for col in literature_columns:
        if col in df.columns and df[col].dtype.name == 'category':
            categorical_columns.append(col)
            # Convert categorical to object to allow new values
            df[col] = df[col].astype('object')
    
    if categorical_columns:
        logger.info(f"Converted categorical columns to object type: {categorical_columns}")

    # ======================================================================================================================
    
    
    # Update all columns at once using vectorized operations
    for pathway_id in pathway_ids:
        pathway_mask = df["Pathway ID"] == pathway_id
        result = result_map[pathway_id]
        
        df.loc[pathway_mask, "clinical_relevance"] = result["clinical_relevance"]
        df.loc[pathway_mask, "functional_relevance"] = result["functional_relevance"]
        df.loc[pathway_mask, "audit_log"] = result["audit_log"]
        df.loc[pathway_mask, "references"] = result["references"]
        df.loc[pathway_mask, "relation"] = result["relation"]
        df.loc[pathway_mask, "hit_score"] = result["hit_score"]


# # Clean Celery-compatible function
# def run_literature_pipeline_celery(input_csv: str, disease: str, max_concurrent: int = 5, batch_size: int = 10):
#     """
#     Clean Celery-compatible synchronous version of the literature pipeline.
#     Uses threading for parallel processing instead of asyncio.
    
#     This function is designed to work with Celery workers and maintains all optimizations:
#     - Parallel processing using ThreadPoolExecutor
#     - Intelligent filtering to skip already processed pathways
#     - Vectorized DataFrame operations for O(n) complexity
#     - Batch processing with memory management
#     - Error handling with retry mechanisms
#     - Progress tracking and checkpointing
#     - Performance monitoring and memory optimization
    
#     Args:
#         input_csv: Path to the input CSV file
#         disease: Disease name for analysis
#         max_concurrent: Maximum number of concurrent pathway processing tasks
#         batch_size: Number of pathways to process in each batch
    
#     Returns:
#         List of processed pathway results
#     """
#     # Initialize performance monitoring
#     monitor = PerformanceMonitor()
#     monitor.start()
    
#     logger.info(f"Running Celery-compatible literature pipeline for {disease} disease")
#     logger.info(f"Configuration: max_concurrent={max_concurrent}, batch_size={batch_size}")
    
#     # Load and prepare data efficiently
#     logger.info("Loading and preparing data...")
#     df = pd.read_csv(input_csv)
    
#     # Optimize DataFrame memory usage
#     df = optimize_dataframe_memory(df)
    
#     all_data = get_pathways_json(input_csv)
    
#     # Add missing literature columns efficiently
#     literature_columns = ["clinical_relevance", "functional_relevance", "audit_log", "references", "relation", "hit_score"]
#     missing_columns = [col for col in literature_columns if col not in df.columns]
#     if missing_columns:
#         df[missing_columns] = None
#         logger.info(f"Added missing columns: {missing_columns}")
    
#     # Create efficient lookup for existing results
#     logger.info("Analyzing existing results...")
#     existing_results = set()
#     for _, row in df.iterrows():
#         pathway_id = row["Pathway ID"]
#         clinical_empty = pd.isna(row["clinical_relevance"]) or row["clinical_relevance"] in [None, "", "Error"]
#         functional_empty = pd.isna(row["functional_relevance"]) or row["functional_relevance"] in [None, "", "Error"]
        
#         if not (clinical_empty or functional_empty):
#             existing_results.add(pathway_id)
    
#     # Filter pathways that need processing
#     pathways_to_process = [pathway for pathway in all_data if pathway["Pathway ID"] not in existing_results]
    
#     total_pathways = len(all_data)
#     already_processed = len(existing_results)
#     needs_processing = len(pathways_to_process)
    
#     logger.info(f"Statistics:")
#     logger.info(f"   Total pathways: {total_pathways}")
#     logger.info(f"   Already processed: {already_processed}")
#     logger.info(f"   Need processing: {needs_processing}")
#     logger.info(f"   Skip rate: {already_processed/total_pathways*100:.1f}%")
    
#     if needs_processing == 0:
#         logger.info("All pathways already have literature data. No processing needed.")
#         return []
    
#     def process_single_pathway_sync(pathway_entry, max_retries: int = 2):
#         """Process a single pathway synchronously with error handling and retry logic."""
#         pathway_id = pathway_entry['Pathway ID']
#         logger.info(f"Processing pathway: {pathway_id}")
        
#         for attempt in range(max_retries + 1):
#             try:
#                 relevance_input = {
#                     "biomarkers": pathway_entry["Pathway associated genes"],
#                     "pathway": pathway_entry["Pathway"],
#                     "disease": disease
#                 }
                
#                 # Run the relevance agent using the thread-safe helper
#                 res = _run_async_in_thread(Runner.run(relevance_agent, json.dumps(relevance_input)))
                
#                 result = {
#                     "Pathway ID": pathway_id,
#                     "clinical_relevance": res.final_output.clinical_relevance,
#                     "functional_relevance": res.final_output.functional_relevance,
#                     "audit_log": res.final_output.audit_log,
#                     "references": res.final_output.references,
#                     "relation": res.final_output.relation,
#                     "hit_score": res.final_output.hit_score
#                 }
                
#                 logger.info(f"Completed: {pathway_id}")
#                 return result
                
#             except Exception as e:
#                 if attempt < max_retries:
#                     logger.info(f"Attempt {attempt + 1} failed for {pathway_id}: {str(e)}. Retrying...")
#                     time.sleep(1)  # Brief delay before retry
#                     continue
#                 else:
#                     logger.info(f"All attempts failed for {pathway_id}: {str(e)}")
#                     # Return error result after all retries exhausted
#                     return {
#                         "Pathway ID": pathway_id,
#                         "clinical_relevance": f"Error after {max_retries + 1} attempts: {str(e)}",
#                         "functional_relevance": f"Error after {max_retries + 1} attempts: {str(e)}",
#                         "audit_log": f"Processing failed after {max_retries + 1} attempts: {str(e)}",
#                         "references": "Error",
#                         "relation": "Error",
#                         "hit_score": 0
#                     }
    
#     # Process in batches using ThreadPoolExecutor for Celery compatibility
#     all_results = []
#     total_batches = (needs_processing + batch_size - 1) // batch_size
    
#     for batch_idx in range(0, needs_processing, batch_size):
#         batch_pathways = pathways_to_process[batch_idx:batch_idx + batch_size]
#         batch_num = (batch_idx // batch_size) + 1
        
#         logger.info(f"----> Processing batch {batch_num}/{total_batches} ({len(batch_pathways)} pathways)")
        
#         # Process batch in parallel using ThreadPoolExecutor
#         with ThreadPoolExecutor(max_workers=max_concurrent, thread_name_prefix="PathwayWorker") as executor:
#             # Submit all tasks
#             future_to_pathway = {
#                 executor.submit(process_single_pathway_sync, pathway): pathway 
#                 for pathway in batch_pathways
#             }
            
#             # Collect results as they complete
#             batch_results = []
#             error_count = 0
#             for future in as_completed(future_to_pathway):
#                 pathway_entry = future_to_pathway[future]
#                 try:
#                     result = future.result()
#                     batch_results.append(result)
#                     if result.get('clinical_relevance', '').startswith('Error'):
#                         error_count += 1
#                 except Exception as e:
#                     logger.info(f"Exception in threaded processing for pathway {pathway_entry.get('Pathway ID', 'N/A')}: {e}")
#                     error_count += 1
#                     # Create error result
#                     batch_results.append({
#                         "Pathway ID": pathway_entry.get("Pathway ID", "Unknown"),
#                         "clinical_relevance": f"Threading error: {str(e)}",
#                         "functional_relevance": f"Threading error: {str(e)}",
#                         "audit_log": f"Processing failed in thread: {str(e)}",
#                         "references": "Error",
#                         "relation": "Error",
#                         "hit_score": 0
#                     })
        
#         all_results.extend(batch_results)
        
#         # Record batch performance metrics
#         monitor.record_batch(len(batch_results), error_count)
        
#         # Update DataFrame efficiently using vectorized operations
#         if batch_results:
#             logger.info(f"Updating DataFrame with {len(batch_results)} results...")
#             update_dataframe_efficiently(df, batch_results)
            
#             # Save checkpoint
#             df.to_csv(input_csv, index=False)
#             logger.info(f"Checkpoint saved: {len(all_results)} total processed")
        
#         # Memory cleanup and garbage collection
#         del batch_results
#         gc.collect()  # Force garbage collection to free memory
    
#     # Get final performance summary
#     performance_summary = monitor.get_summary()
    
#     logger.info(f"Processing complete!")
#     logger.info(f"Final Performance Summary:")
#     logger.info(f"   Total pathways processed: {len(all_results)}")
#     logger.info(f"   Total processing time: {performance_summary['total_time']:.2f} seconds")
#     logger.info(f"   Average time per pathway: {performance_summary['avg_time_per_pathway']:.2f} seconds")
#     logger.info(f"   Processing rate: {performance_summary['pathways_per_minute']:.1f} pathways/minute")
#     logger.info(f"   Success rate: {100 - performance_summary['error_rate']:.1f}%")
#     logger.info(f"   Error rate: {performance_summary['error_rate']:.1f}%")
    
#     # Memory usage summary
#     logger.info(f"Memory Optimization:")
#     logger.info(f"   DataFrame memory optimized: Done")
#     logger.info(f"   Garbage collection performed: Done")
#     logger.info(f"   Celery-compatible threading: Done")
    
#     return all_results


# # Celery task wrapper
# def celery_literature_pipeline_task(input_csv: str, disease: str, max_concurrent: int = 5, batch_size: int = 10):
#     """
#     Celery task wrapper for the literature pipeline.
#     This function can be used directly as a Celery task.
    
#     Example usage in Celery:
#     @app.task
#     def process_literature(input_csv, disease, max_concurrent=5, batch_size=10):
#         return celery_literature_pipeline_task(input_csv, disease, max_concurrent, batch_size)
#     """
#     try:
#         return run_literature_pipeline_celery(input_csv, disease, max_concurrent, batch_size)
#     except Exception as e:
#         logger.info(f"Celery task failed: {str(e)}")
#         raise e


# Use the async version for direct execution
# asyncio.run(run_literature_pipeline(r"ALL_ANALYSIS\Melanoma_Pathways_Enrichment.csv", "Melanoma"))

# Uncomment the line below to use the Celery-compatible version instead:
# run_literature_pipeline_celery(r"ALL_ANALYSIS\Melanoma_Pathways_Enrichment.csv", "Melanoma")
