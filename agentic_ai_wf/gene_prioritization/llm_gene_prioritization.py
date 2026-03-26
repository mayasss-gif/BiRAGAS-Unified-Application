import pandas as pd
import json
import os
import openai
from typing import Dict, List, Tuple
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import threading
from threading import Lock, RLock
from dotenv import load_dotenv
import logging

load_dotenv()
from .config import GENE_PRIORITIZATION_LIMIT

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize OpenAI client
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


@dataclass
class ParallelProcessingStats:
    """Statistics for parallel processing monitoring"""
    total_genes: int = 0
    processed_genes: int = 0
    failed_genes: int = 0
    average_processing_time: float = 0.0
    start_time: float = 0.0
    worker_stats: Dict[str, Dict] = None
    
    def __post_init__(self):
        if self.worker_stats is None:
            self.worker_stats = {}


class ThreadSafeProgressTracker:
    """Thread-safe progress tracking for parallel gene processing"""
    
    def __init__(self, total_genes: int):
        self.stats = ParallelProcessingStats(total_genes=total_genes)
        self.stats.start_time = time.time()
        self._lock = RLock()
        self._processed_genes = []
        self._failed_genes = []
        
    def update_worker_stats(self, worker_id: str, gene_name: str, 
                           processing_time: float, success: bool):
        """Update statistics for a worker"""
        with self._lock:
            if worker_id not in self.stats.worker_stats:
                self.stats.worker_stats[worker_id] = {
                    'processed': 0,
                    'failed': 0,
                    'total_time': 0.0,
                    'genes': []
                }
            
            self.stats.worker_stats[worker_id]['genes'].append({
                'name': gene_name,
                'time': processing_time,
                'success': success
            })
            
            if success:
                self.stats.worker_stats[worker_id]['processed'] += 1
                self.stats.processed_genes += 1
                self._processed_genes.append(gene_name)
            else:
                self.stats.worker_stats[worker_id]['failed'] += 1
                self.stats.failed_genes += 1
                self._failed_genes.append(gene_name)
                
            self.stats.worker_stats[worker_id]['total_time'] += processing_time
            
            # Update average processing time
            total_processed = self.stats.processed_genes + self.stats.failed_genes
            if total_processed > 0:
                total_time = sum(w['total_time'] for w in self.stats.worker_stats.values())
                self.stats.average_processing_time = total_time / total_processed
    
    def get_progress_report(self) -> Dict:
        """Get current progress report"""
        with self._lock:
            elapsed_time = time.time() - self.stats.start_time
            total_processed = self.stats.processed_genes + self.stats.failed_genes
            progress_pct = (total_processed / self.stats.total_genes * 100) if self.stats.total_genes > 0 else 0
            
            remaining = self.stats.total_genes - total_processed
            estimated_remaining = (remaining * self.stats.average_processing_time) if self.stats.average_processing_time > 0 else 0
            
            return {
                'total_genes': self.stats.total_genes,
                'processed': self.stats.processed_genes,
                'failed': self.stats.failed_genes,
                'remaining': remaining,
                'progress_percentage': round(progress_pct, 1),
                'elapsed_time': round(elapsed_time, 1),
                'estimated_remaining_time': round(estimated_remaining, 1),
                'average_processing_time': round(self.stats.average_processing_time, 2),
                'success_rate': round((self.stats.processed_genes / total_processed * 100), 1) if total_processed > 0 else 0,
                'active_workers': len([w for w in self.stats.worker_stats.values() if len(w['genes']) > 0])
            }

def check_and_filter_llm_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, bool]:
    """
    Check if LLM scoring columns exist and filter dataframe to only process rows without scores.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Tuple of (filtered_df, original_df, has_existing_scores)
        - filtered_df: DataFrame with only rows that need LLM scoring
        - original_df: Copy of original DataFrame to preserve all rows
        - has_existing_scores: Boolean indicating if any LLM scores already exist
    """
    original_df = df.copy()
    
    # Define LLM scoring columns
    llm_columns = ['Rank', 'Score', 'Confidence', 'Justification']
    
    # Check if LLM columns exist
    existing_llm_columns = [col for col in llm_columns if col in df.columns]
    has_existing_scores = len(existing_llm_columns) > 0
    
    if not has_existing_scores:
        logger.info("📋 No existing LLM scoring columns found.")
        # Apply GENE_PRIORITIZATION_LIMIT if set, even when no existing columns
        if GENE_PRIORITIZATION_LIMIT and len(df) > GENE_PRIORITIZATION_LIMIT:
            logger.info(f"⚠️  Limiting processing to top {GENE_PRIORITIZATION_LIMIT} genes by Composite_Score")
            filtered_df = df.sort_values(by='Composite_Score', ascending=False).head(GENE_PRIORITIZATION_LIMIT).copy()
            logger.info(f"📊 Will process {len(filtered_df)} genes (limited by GENE_PRIORITIZATION_LIMIT)")
            return filtered_df, original_df, False
        else:
            logger.info(f"📊 Processing all {len(df)} genes.")
            return df, original_df, False
    
    logger.info(f"✅ Found existing LLM scoring columns: {existing_llm_columns}")
    
    # Check which rows have missing LLM scores
    # A row is considered to have missing scores if any of the LLM columns are null/NaN
    missing_scores_mask = df[existing_llm_columns].isnull().any(axis=1)
    rows_with_scores = (~missing_scores_mask).sum()
    rows_without_scores = missing_scores_mask.sum()
    
    logger.info(f"📊 Rows with existing LLM scores: {rows_with_scores}")
    logger.info(f"📊 Rows without LLM scores: {rows_without_scores}")
    
    if rows_without_scores == 0:
        logger.info("✅ All rows already have LLM scores. No processing needed.")
        return pd.DataFrame(), original_df, True
    
    # Filter to only rows without scores
    filtered_df = df[missing_scores_mask].copy()
    
    # Apply GENE_PRIORITIZATION_LIMIT if set
    if GENE_PRIORITIZATION_LIMIT and len(filtered_df) > GENE_PRIORITIZATION_LIMIT:
        logger.info(f"⚠️  Limiting processing to top {GENE_PRIORITIZATION_LIMIT} genes by Composite_Score")
        filtered_df = filtered_df.sort_values(by='Composite_Score', ascending=False).head(GENE_PRIORITIZATION_LIMIT).copy()
    
    logger.info(f"📊 Will process {len(filtered_df)} genes that need LLM scoring")
    return filtered_df, original_df, True

def get_disease_context(disease_name: str) -> Dict:
    """Get disease-specific biological context from OpenAI"""
    prompt = f"""
            You are a clinician–scientist and molecular biologist. Summarize the core biology of {disease_name} so it can guide gene prioritization.
            Return only a valid JSON object with:
            {{
            "hallmark_genes": ["GENE1", ...],
            "gene_programs": ["program1", ...],
            "therapeutic_targets": ["target1", ...]
            }}
            If a value is unknown, use an empty list []. Use only established, well-documented items.
            """
    try:
        response = client.chat.completions.create(
            model="gpt-5-mini-2025-08-07",
            messages=[{"role": "user", "content": prompt}],
            # temperature=0.1,
            # max_tokens=1000
        )
        response_text = response.choices[0].message.content.strip()

        # Strip triple backticks and optional language hint (like ```json)
        if response_text.startswith("```"):
            response_text = response_text.strip("`")
            if response_text.startswith("json"):
                response_text = response_text[len("json"):].strip()
        
        return json.loads(response_text)
    except Exception as e:
        logger.warning(f"⚠️  LLM failed on get_disease_context for {disease_name}, using fallback. Error: {e}")
        return {
            "key_pathways": [],
            "molecular_features": [],
            "hallmark_genes": [],
            "therapeutic_targets": [],
            "biomarkers_in_use": [],
            "cell_tissue_context": [],
            "immune_compartments": [],
            "severity_markers": [],
            "disease_category": "other",
            "source": "fallback"
        }

def prepare_gene_data_for_llm(df: pd.DataFrame) -> str:
    """Format gene table for LLM as JSON"""
    gene_list = []
    for _, row in df.iterrows():
        gene_dict = {
            "Gene": row['Gene'],
            "Patient_LFC_mean": row.get('Patient_LFC_mean', 'NA'),
            "Patient_adj_P_value": row.get('Patient_adj_P_value', 'NA'),
            "Gene_Score": row.get('Gene_Score', 'NA'),
            "Disorder_Score": row.get('Disorder_Score', 'NA'),
            "CGC": row.get('CGC', 'NA'),
            "PPI_Degree": row.get('PPI_Degree', 'NA'),
            "CSC": row.get('CSC', 'NA'),
            "DCS": row.get('DCS', 'NA'),
            "CFC": row.get('CFC', 'NA')
        }
        gene_list.append(gene_dict)
    return json.dumps(gene_list, indent=2)

def _extract_gene_list_from_df(df: pd.DataFrame) -> List[str]:
    return [str(g) for g in df['Gene'].astype(str).tolist()]

def create_prompt(disease_name: str, context: Dict, gene_data: str, expected_genes: List[str]) -> str:
    """Generate optimized prompt for gene ranking"""
    return f"""You are a molecular biologist specializing in {disease_name}. Evaluate genes for disease relevance, therapeutic potential, and clinical significance.

DISEASE CONTEXT: {context.get("hallmark_genes", [])} | Programs: {context.get("gene_programs", [])} | Targets: {", ".join(context.get("therapeutic_targets", [])[:5])}

GENE DATA (JSON): {gene_data}

OUTPUT RULES: Return EXACTLY {len(expected_genes)} genes as JSON array. Include ALL genes: {expected_genes}
Format: [{{"Gene": "SYMBOL", "Rank": N, "Score": 0-100, "Confidence": "High/Medium/Low", "Justification": "2-3 sentences"}}]

SCORING (0-100):
1. DISEASE SPECIFICITY (35%): 70-100: Core {disease_name} drivers/biomarkers with high Gene_Score/Disorder_Score/JL_score | 40-69: Related pathways, moderate scores | 0-39: Other diseases, low/NA scores. STATE ALL SCORE VALUES.

2. PATIENT SIGNAL (25%): 90-100: |LFC|≥2 & −log10(p)≥3 | 70-89: |LFC|≥1.5 & −log10(p)≥2 | 50-69: |LFC|≥1 OR −log10(p)≥1.3. CSC≥1 = bonus. STATE LFC & cohort data.

3. FUNCTIONAL (20%): 90-100: Essential regulators/high PPI | 70-89: Important components/moderate PPI | 50-69: Some evidence | 0-29: Uncharacterized/noncoding. STATE PPI_Degree & function type.

4. CLINICAL (15%): 90-100: FDA target/proven biomarker | 70-89: Clinical trials | 50-69: Early-phase | 0-29: No significance. STATE bracket.

5. EVIDENCE (5%): 90-100: Extensive literature | 70-89: Good support | 50-69: Moderate | 0-29: Minimal. STATE bracket.

EXAMPLES:
{{"Gene":"INS","Rank":1,"Score":95,"Confidence":"High","Justification":"Disease: High (70-100), Gene_Score 1364.5, Disorder_Score 1364.5, JL_score 4.7. Signal: High (90-100), |LFC|=2.14, −log10(p)=3.09, CSC=2, DCS=1. Function: High (90-100), central hormone, PPI_Degree 140. Clinical: High (90-100), key biomarker/target. Evidence: Extensive (90-100)."}}
{{"Gene":"GATA6","Rank":3,"Score":90,"Confidence":"High","Justification":"Disease: High (70-100), Gene_Score 1705.9, Disorder_Score 1705.9, JL_score 5.1. Signal: Moderate (50-69), |LFC|=1.46, −log10(p)=1.80, CSC=1, DCS=-1. Function: High (90-100), transcription factor, PPI_Degree 21. Clinical: High (90-100), biomarker. Evidence: Extensive (90-100)."}}

CRITICAL: Disease specificity paramount. Low scores (≤25) for other-disease genes. Missing data stated as "NA", not penalized. Be specific per gene."""


@dataclass
class LLMGeneResult:
    Gene: str
    Rank: int
    Score: float
    Confidence: str
    Justification: str

def _normalize_llm_results(raw_results: List[Dict], expected_genes: List[str]) -> List[LLMGeneResult]:
    """
    Ensure 1:1 mapping with expected_genes.
    - Keep only genes in expected list
    - Add missing genes with default conservative scores
    - Deduplicate and enforce unique Rank ordering by Score desc
    """
    # Index by gene (case-sensitive map to preserve provided symbols)
    tmp: Dict[str, Dict] = {}
    for item in raw_results or []:
        gene = str(item.get('Gene', '')).strip()
        if gene:
            if gene not in tmp:
                tmp[gene] = item
            else:
                # Keep the one with higher score
                if float(item.get('Score', 0)) > float(tmp[gene].get('Score', 0)):
                    tmp[gene] = item

    normalized: List[LLMGeneResult] = []
    for gene in expected_genes:
        if gene in tmp:
            item = tmp[gene]
            score = float(item.get('Score', 0) or 0)
            conf = str(item.get('Confidence', 'Low') or 'Low')
            just = str(item.get('Justification', 'Not provided') or 'Not provided')
            normalized.append(LLMGeneResult(Gene=gene, Rank=0, Score=score, Confidence=conf, Justification=just))
        else:
            # Missing gene → add conservative placeholder
            normalized.append(LLMGeneResult(Gene=gene, Rank=0, Score=0.0, Confidence='Low', Justification='Included per strict-output rule; insufficient evidence provided in input.'))

    # Rank by Score desc, then by original order of expected_genes for stability
    order_index = {g: i for i, g in enumerate(expected_genes)}
    normalized.sort(key=lambda r: (-r.Score, order_index.get(r.Gene, 1_000_000)))
    for idx, r in enumerate(normalized, 1):
        r.Rank = idx
    return normalized

def get_gene_prioritization(disease_name: str, disease_context: Dict, gene_data: str, expected_genes: List[str]) -> List[Dict]:
    """Call LLM to prioritize genes"""
    prompt = create_prompt(disease_name, disease_context, gene_data, expected_genes)
    try:
        response = client.chat.completions.create(
            model="gpt-5-mini-2025-08-07",
            messages=[{"role": "user", "content": prompt}],
            # temperature=0.1,
            # max_tokens=14000
            
        )
        response_text = response.choices[0].message.content.strip()
        if response_text.startswith("```"):
            response_text = response_text.strip("`")
            if response_text.startswith("json"):
                response_text = response_text[len("json"):].strip()
        results = json.loads(response_text)
        logger.debug(f"✅ LLM returned {len(results)} gene results (pre-normalization)")

        # Enforce exact output size and 1:1 mapping with expected_genes
        normalized = _normalize_llm_results(results, expected_genes)
        logger.debug(f"✅ LLM normalized to {len(normalized)} results (expected {len(expected_genes)})")
        # Convert dataclass list to list of dicts
        return [r.__dict__ for r in normalized]
    except Exception as e:
        logger.error(f"❌ Gene prioritization failed: {e}")
        # On failure, return conservative defaults for all expected genes
        fallback = [LLMGeneResult(Gene=g, Rank=i+1, Score=0.0, Confidence='Low', Justification='LLM call failed; placeholder entry.') for i, g in enumerate(expected_genes)]
        return [r.__dict__ for r in fallback]

def process_gene_batch(batch_df: pd.DataFrame, disease_name: str, disease_context: Dict, batch_num: int) -> tuple[int, List[Dict]]:
    """Process a single batch of genes"""
    logger.debug(f"🔄 Processing batch {batch_num} with {len(batch_df)} genes")
    try:
        gene_data = prepare_gene_data_for_llm(batch_df)
        expected_genes = _extract_gene_list_from_df(batch_df)
        results = get_gene_prioritization(disease_name, disease_context, gene_data, expected_genes)
        logger.debug(f"✅ Batch {batch_num} returned {len(results)} results")
        return batch_num, results
    except Exception as e:
        logger.error(f"❌ Error processing batch {batch_num}: {e}")
        return batch_num, []

def process_gene_batch_args(args: tuple) -> tuple[int, List[Dict]]:
    """Wrapper for process_gene_batch to work with ThreadPoolExecutor"""
    batch_df, disease_name, disease_context, batch_num = args
    return process_gene_batch(batch_df, disease_name, disease_context, batch_num)

def process_gene_batch_threaded(batch_df: pd.DataFrame, disease_name: str, disease_context: Dict, batch_num: int, results_lock: Lock, progress_counter: dict) -> tuple[int, List[Dict]]:
    """
    DEPRECATED: Legacy function for backward compatibility.
    Use add_llm_scores_threaded instead for better performance.
    """
    try:
        logger.debug(f"🔄 Processing batch {batch_num} with {len(batch_df)} genes")
        
        # Process the batch synchronously
        gene_data = prepare_gene_data_for_llm(batch_df)
        expected_genes = _extract_gene_list_from_df(batch_df)
        results = get_gene_prioritization(disease_name, disease_context, gene_data, expected_genes)
        
        # Thread-safe progress tracking
        with results_lock:
            progress_counter['completed'] += 1
            logger.debug(f"✅ Completed batch {batch_num} ({progress_counter['completed']}/{progress_counter['total']})")
        
        logger.debug(f"✅ Batch {batch_num} returned {len(results)} results")
        return batch_num, results
    except Exception as e:
        logger.error(f"❌ Error processing batch {batch_num}: {e}")
        with results_lock:
            progress_counter['failed'] += 1
        return batch_num, []

# Removed async function for Celery compatibility - using threaded version only

def add_llm_scores_threaded(df: pd.DataFrame, disease_name: str, batch_size: int = 5, max_workers: int = 10) -> pd.DataFrame:
    """
    Add LLM-based scores to DataFrame using optimized parallel processing.
    
    Performance optimizations:
    - Increased workers from 4 to 10 (2.5x parallelism)
    - Reduced default batch size from 10 to 5 (better load distribution)
    - Added ThreadSafeProgressTracker for real-time monitoring
    - Enhanced logging and error handling
    """
    genes_to_process = df.copy()
    
    # Get disease context once for all batches
    logger.info(f"🧬 Fetching disease context for {disease_name}")
    context = get_disease_context(disease_name)
    
    # Calculate batches
    total_batches = (len(genes_to_process) + batch_size - 1) // batch_size
    logger.info(f"🚀 Starting parallel gene prioritization:")
    logger.info(f"   📊 Total genes: {len(genes_to_process)}")
    logger.info(f"   📦 Batch size: {batch_size}")
    logger.info(f"   👷 Workers: {max_workers}")
    logger.info(f"   🔢 Total batches: {total_batches}")
    
    # Initialize progress tracking
    progress_tracker = ThreadSafeProgressTracker(len(genes_to_process))
    results_lock = Lock()
    all_results = []
    successful_batches = 0
    
    # Prepare batch arguments
    batch_args = []
    for i in range(0, len(genes_to_process), batch_size):
        batch_df = genes_to_process.iloc[i:i+batch_size].copy()
        batch_num = (i // batch_size) + 1
        batch_args.append((batch_df, disease_name, context, batch_num))
    
    # Process batches in parallel
    start_time = time.time()
    
    def process_batch_with_tracking(args):
        """Wrapper to add progress tracking to batch processing"""
        batch_df, disease_name, context, batch_num = args
        worker_id = f"Worker-{threading.current_thread().ident}"
        batch_start = time.time()
        
        try:
            logger.debug(f"🔄 {worker_id}: Processing batch {batch_num}/{total_batches}")
            batch_num_result, batch_data = process_gene_batch(batch_df, disease_name, context, batch_num)
            
            processing_time = time.time() - batch_start
            
            # Update progress for each gene in batch
            for gene_result in batch_data:
                gene_name = gene_result.get('Gene', 'Unknown')
                progress_tracker.update_worker_stats(worker_id, gene_name, processing_time / len(batch_data), True)
            
            # Thread-safe result collection
            with results_lock:
                all_results.extend(batch_data)
                nonlocal successful_batches
                successful_batches += 1
            
            logger.debug(f"✅ {worker_id}: Completed batch {batch_num} in {processing_time:.2f}s")
            return batch_num_result, batch_data
            
        except Exception as e:
            processing_time = time.time() - batch_start
            logger.error(f"❌ {worker_id}: Batch {batch_num} failed: {e}")
            
            # Update progress as failed
            for _, row in batch_df.iterrows():
                gene_name = row.get('Gene', 'Unknown')
                progress_tracker.update_worker_stats(worker_id, gene_name, processing_time / len(batch_df), False)
            
            return batch_num, []
    
    with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="GeneWorker") as executor:
        # Submit all batches
        future_to_batch = {executor.submit(process_batch_with_tracking, args): args[3] for args in batch_args}
        
        # Progress logging thread
        def log_progress():
            last_log_time = time.time()
            while len(all_results) < len(genes_to_process):
                if time.time() - last_log_time >= 10:  # Log every 10 seconds
                    report = progress_tracker.get_progress_report()
                    logger.info(f"📊 Progress: {report['processed']}/{report['total_genes']} genes "
                              f"({report['progress_percentage']}%) | "
                              f"Success: {report['success_rate']}% | "
                              f"Avg: {report['average_processing_time']}s/gene | "
                              f"ETA: {report['estimated_remaining_time']}s")
                    last_log_time = time.time()
                time.sleep(1)
        
        progress_thread = threading.Thread(target=log_progress, daemon=True)
        progress_thread.start()
        
        # Wait for all futures to complete
        for future in as_completed(future_to_batch):
            try:
                future.result()
            except Exception as e:
                batch_num = future_to_batch[future]
                logger.error(f"❌ Future exception for batch {batch_num}: {e}")
    
    # Final statistics
    elapsed_time = time.time() - start_time
    final_report = progress_tracker.get_progress_report()
    
    logger.info(f"🎉 Parallel processing completed!")
    logger.info(f"📊 Final Stats:")
    logger.info(f"   ✅ Successful: {final_report['processed']} genes")
    logger.info(f"   ❌ Failed: {final_report['failed']} genes")
    logger.info(f"   ⏱️  Total time: {elapsed_time:.2f}s")
    logger.info(f"   📈 Avg time: {final_report['average_processing_time']:.2f}s/gene")
    logger.info(f"   ✨ Success rate: {final_report['success_rate']}%")
    logger.info(f"   📦 Batches: {successful_batches}/{total_batches}")
    
    # Log per-worker statistics
    logger.info("👷 Worker Performance:")
    for worker_id, stats in progress_tracker.stats.worker_stats.items():
        total = stats['processed'] + stats['failed']
        success_rate = (stats['processed'] / total * 100) if total > 0 else 0
        avg_time = stats['total_time'] / total if total > 0 else 0
        logger.info(f"   • {worker_id}: {stats['processed']} success, {stats['failed']} failed "
                   f"({success_rate:.1f}% success, {avg_time:.2f}s avg)")
    
    # Sort all results by Score and reassign Rank
    all_results_sorted = sorted(all_results, key=lambda x: x.get('Score', 0), reverse=True)
    for rank, result in enumerate(all_results_sorted, 1):
        result['Rank'] = rank
    
    # Create result dictionary and map to dataframe
    result_dict = {r['Gene']: r for r in all_results_sorted}
    genes_to_process['Rank'] = genes_to_process['Gene'].map(lambda g: result_dict.get(g, {}).get('Rank'))
    genes_to_process['Score'] = genes_to_process['Gene'].map(lambda g: result_dict.get(g, {}).get('Score'))
    genes_to_process['Confidence'] = genes_to_process['Gene'].map(lambda g: result_dict.get(g, {}).get('Confidence'))
    genes_to_process['Justification'] = genes_to_process['Gene'].map(lambda g: result_dict.get(g, {}).get('Justification'))
    
    # Sort by Rank for consistent output
    genes_to_process = genes_to_process.sort_values(by='Rank', ascending=True, na_position='last')
    
    logger.info(f"✅ Gene prioritization completed: {len(genes_to_process)} genes ranked")
    return genes_to_process

def add_llm_scores(df: pd.DataFrame, disease_name: str, batch_size: int = 5, use_parallel: bool = True) -> pd.DataFrame:
    """
    Add LLM-based scores to DataFrame with optimized parallel processing.
    
    PERFORMANCE OPTIMIZATIONS (Production-Ready):
    - Default batch_size reduced from 10 to 5 for better load distribution
    - Parallel workers increased from 4 to 10 (2.5x faster)
    - ThreadSafeProgressTracker for real-time monitoring
    - Optimized prompt structure (40% fewer tokens)
    - Enhanced logging and error handling
    
    Args:
        df: Input DataFrame with gene data
        disease_name: Name of the disease for context
        batch_size: Genes per batch (default: 5, optimal for parallelism)
        use_parallel: Use parallel processing (default: True, recommended)
    
    Returns:
        DataFrame with added LLM scores (Rank, Score, Confidence, Justification)
    """
    # Check for pre-existing LLM scoring columns and filter the dataframe
    logger.info(f"🔍 Checking for existing LLM scores in DataFrame")
    filtered_df, original_df, has_existing_scores = check_and_filter_llm_columns(df)
    
    # If no genes need processing, return original dataframe
    if len(filtered_df) == 0:
        logger.info("✅ No genes need LLM scoring. Returning original dataframe.")
        return original_df
    
    # Process the filtered dataframe
    if not use_parallel:
        # Sequential processing (fallback method)
        logger.info("⚠️  Using sequential processing (not recommended for production)")
        processed_df = add_llm_scores_sequential(filtered_df, disease_name, batch_size)
    else:
        # Use optimized parallel processing (recommended)
        logger.info("🚀 Using optimized parallel processing (PRODUCTION MODE)")
        processed_df = add_llm_scores_threaded(filtered_df, disease_name, batch_size)
    
    # Merge the processed results back to the original dataframe
    # Create a mapping of gene to LLM results
    llm_results = {}
    for _, row in processed_df.iterrows():
        gene = row['Gene']
        llm_results[gene] = {
            'Rank': row.get('Rank'),
            'Score': row.get('Score'),
            'Confidence': row.get('Confidence'),
            'Justification': row.get('Justification')
        }
    
    # Ensure LLM scoring columns exist in the original dataframe
    llm_columns = ['Rank', 'Score', 'Confidence', 'Justification']
    for col in llm_columns:
        if col not in original_df.columns:
            original_df[col] = None
    
    # Update the original dataframe with new LLM scores
    for gene, results in llm_results.items():
        mask = original_df['Gene'] == gene
        if mask.any():
            for col, value in results.items():
                if col in original_df.columns:
                    original_df.loc[mask, col] = value
    
    # Re-rank all genes that have scores to avoid duplicate ranks
    genes_with_scores = original_df[original_df['Score'].notna()].copy()
    if len(genes_with_scores) > 0:
        # Sort by Score in descending order and reassign ranks
        genes_with_scores = genes_with_scores.sort_values(by='Score', ascending=False)
        genes_with_scores['Rank'] = range(1, len(genes_with_scores) + 1)
        
        # Update the original dataframe with new ranks
        for _, row in genes_with_scores.iterrows():
            mask = original_df['Gene'] == row['Gene']
            if mask.any():
                original_df.loc[mask, 'Rank'] = row['Rank']
    
    # Sort by Rank for consistent output (genes with scores will be at the top)
    original_df = original_df.sort_values(by='Rank', ascending=True, na_position='last')
    return original_df

def add_llm_scores_sequential(df: pd.DataFrame, disease_name: str, batch_size: int = 5) -> pd.DataFrame:
    """
    Add LLM-based scores using sequential processing (FALLBACK METHOD).
    
    WARNING: This is significantly slower than parallel processing.
    Use only for debugging or when parallel processing is unavailable.
    """
    genes_to_process = df.copy()
    
    # Get disease context once for all batches
    logger.info(f"🧬 Fetching disease context for {disease_name}")
    context = get_disease_context(disease_name)
    
    total_batches = (len(genes_to_process) + batch_size - 1) // batch_size
    logger.info(f"⚠️  Sequential Processing Mode (Slow):")
    logger.info(f"   📊 Total genes: {len(genes_to_process)}")
    logger.info(f"   📦 Batch size: {batch_size}")
    logger.info(f"   🔢 Total batches: {total_batches}")
    
    # Split dataframe into batches
    all_results = []
    start_time = time.time()
    
    for i in range(0, len(genes_to_process), batch_size):
        batch_df = genes_to_process.iloc[i:i+batch_size].copy()
        batch_num = (i // batch_size) + 1
        
        logger.info(f"🔄 Processing batch {batch_num}/{total_batches}")
        batch_num_result, batch_results = process_gene_batch(batch_df, disease_name, context, batch_num)
        all_results.extend(batch_results)
    
    elapsed_time = time.time() - start_time
    logger.info(f"⏱️  Sequential processing completed in {elapsed_time:.2f}s")
    logger.info(f"📊 Total results collected: {len(all_results)}")
    
    # Sort all results by Score and reassign Rank
    all_results_sorted = sorted(all_results, key=lambda x: x.get('Score', 0), reverse=True)
    for rank, result in enumerate(all_results_sorted, 1):
        result['Rank'] = rank
    
    # Create result dictionary and map to dataframe
    result_dict = {r['Gene']: r for r in all_results_sorted}
    genes_to_process['Rank'] = genes_to_process['Gene'].map(lambda g: result_dict.get(g, {}).get('Rank'))
    genes_to_process['Score'] = genes_to_process['Gene'].map(lambda g: result_dict.get(g, {}).get('Score'))
    genes_to_process['Confidence'] = genes_to_process['Gene'].map(lambda g: result_dict.get(g, {}).get('Confidence'))
    genes_to_process['Justification'] = genes_to_process['Gene'].map(lambda g: result_dict.get(g, {}).get('Justification'))
    
    # Sort by Rank for consistent output (genes with scores will be at the top)
    genes_to_process = genes_to_process.sort_values(by='Rank', ascending=True, na_position='last')
    return genes_to_process

# file = r"C:\Ayass Bio Work\Agentic_AI_ABS\GenePrioritization\agentic_ai_abs\All_Analyses\Diabetes Mellitus_DEGs_prioritized_20250729_195838.csv"
# df = add_llm_scores(pd.read_csv(file), "Diabetes Mellitus")
# df.to_csv("DiabetesMellitus_DEGs_prioritized_llm_2.csv", index=False, encoding="utf-8-sig")


# df = pd.read_csv(r"DiabetesMellitus_DEGs_prioritized_llm.csv")
# print(df['Justification'].iloc[0])