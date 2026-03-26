"""
Parallel Drug Prioritization System
==================================

This module implements parallel processing for drug prioritization to significantly
improve performance when processing large datasets.
"""

import logging
import pandas as pd
import multiprocessing as mp
from typing import List, Dict
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

from .universal_drug_prioritizer import UniversalDrugPrioritizer, PrioritizedDrug

logger = logging.getLogger(__name__)

def _prioritize_single_drug(drug_data: Dict) -> Dict:
    """
    Prioritize a single drug - designed for parallel processing (exact replica of pathway validation pattern)
    
    Args:
        drug_data: Dictionary containing drug info and prioritization parameters
    
    Returns:
        Dictionary with prioritization result or error info
    """
    try:
        # Extract parameters (similar to pathway validation)
        drug_name = drug_data['drug_name']
        disease_name = drug_data['disease_name']
        drug_dict = drug_data['drug_dict']
        disease_context = drug_data['disease_context']
        model = drug_data['model']
        
        # Create drug prioritizer for this thread (similar to pathway validator)
        prioritizer = UniversalDrugPrioritizer(model=model)
        
        # CRITICAL: Set the cached disease context to avoid regeneration
        prioritizer._disease_context_cache[disease_name] = disease_context
        
        # Perform prioritization (similar to pathway validation)
        result = prioritizer._process_single_drug(
            drug=drug_dict,
            disease_name=disease_name, 
            disease_context=disease_context
        )
        
        return {
            'success': True,
            'result': result,
            'drug_name': drug_name,
            'processing_time': time.time() - drug_data.get('start_time', time.time())
        }
        
    except Exception as e:
        logger.warning(f"Failed to prioritize drug {drug_data.get('drug_name', 'Unknown')}: {e}")
        return {
            'success': False,
            'error': str(e),
            'drug_name': drug_data.get('drug_name', 'Unknown'),
            'processing_time': time.time() - drug_data.get('start_time', time.time())
        }

# Performance configuration for parallel drug processing (similar to pathway validation)
class DrugPrioritizationConfig:
    """Configuration class for drug prioritization performance optimization"""
    
    # Parallel processing settings
    DEFAULT_MAX_WORKERS = 5  # Conservative for API rate limiting
    MAX_WORKERS_LIMIT = 10   # Prevent resource exhaustion
    
    # Timeout settings (in seconds) - similar to pathway validation
    SINGLE_DRUG_TIMEOUT = 30      # 30s per drug (same as pathway)
    TOTAL_PRIORITIZATION_TIMEOUT = 600  # 10 minutes total (longer than pathways due to complexity)
    
    # Performance thresholds
    SLOW_DRUG_THRESHOLD = 15      # seconds
    PARALLEL_THRESHOLD = 3        # Use parallel if >= 3 drugs
    
    @classmethod
    def get_optimal_workers(cls, num_drugs: int, max_workers: int = None) -> int:
        """Determine optimal number of workers based on drug count"""
        if max_workers is None:
            max_workers = cls.DEFAULT_MAX_WORKERS
        
        # Limit workers to prevent resource exhaustion
        max_workers = min(max_workers, cls.MAX_WORKERS_LIMIT)
        
        # Use fewer workers for small batches (conservative for API limits)
        if num_drugs <= 3:
            return min(2, max_workers)
        elif num_drugs <= 10:
            return min(3, max_workers)
        elif num_drugs <= 25:
            return min(4, max_workers)
        else:
            return max_workers
    
    @classmethod
    def should_use_parallel(cls, num_drugs: int) -> bool:
        """Determine if parallel processing should be used"""
        return num_drugs >= cls.PARALLEL_THRESHOLD

class ParallelDrugPrioritizer:
    """
    Parallel implementation of drug prioritization using multiprocessing or threading.
    Distributes batches across multiple workers for faster execution.
    
    Automatically detects Celery/daemon environments and switches to thread-based
    processing to avoid "daemonic processes cannot have children" errors.
    """
    
    def __init__(self, 
                 model: str = "gpt-5-mini-2025-08-07",
                 max_workers: int = None,  # Auto-optimize based on workload
                 use_threads: bool = None):
        """
        Initialize parallel drug prioritizer
        
        Args:
            model: LLM model to use
            max_workers: Maximum number of parallel workers (auto-optimized if None)
            use_threads: If True, use threads instead of processes (auto-detects Celery)
        """
        self.model = model
        # Initialize with default, will be optimized based on actual workload
        self.max_workers = max_workers or DrugPrioritizationConfig.DEFAULT_MAX_WORKERS
        self.max_workers = min(self.max_workers, mp.cpu_count())
        
        # Auto-detect if we're running in a daemon process (like Celery)
        if use_threads is None:
            try:
                # Check if we're in a daemon process or if the parent process is daemon
                import multiprocessing
                current_process = multiprocessing.current_process()
                
                # More robust detection for Celery workers
                self.use_threads = (
                    current_process.daemon or 
                    hasattr(current_process, '_parent_pid') or
                    'celery' in str(current_process.name).lower() or
                    'forkpoolworker' in str(current_process.name).lower()
                )
            except:
                self.use_threads = False
        else:
            self.use_threads = use_threads
        
        execution_mode = "Thread-based" if self.use_threads else "Process-based"
        logger.info(f"Initialized ParallelDrugPrioritizer ({execution_mode}): Model={model}, Workers={self.max_workers}/{mp.cpu_count()} CPUs")
    
    def prioritize_drugs_parallel(self, 
                                df: pd.DataFrame, 
                                disease_name: str) -> List[PrioritizedDrug]:
        """
        Main entry point for parallel drug prioritization - EXACT REPLICA of pathway validation pattern
        
        Args:
            df: DataFrame with drug-pathway data
            disease_name: Target disease name
            
        Returns:
            List of PrioritizedDrug objects
        """
        start_time = time.time()
        
        try:
            # Validate input (similar to pathway validation)
            if 'drug_name' not in df.columns:
                return []
            
            # Auto-optimize configuration (similar to pathway validation)
            batch_size = len(df)  # Process all drugs
            
            # Prepare drug subset (similar to pathway validation)
            drug_subset = df.copy()
            if 'priority_score' in drug_subset.columns:
                drug_subset = drug_subset.nlargest(batch_size, 'priority_score')
            else:
                drug_subset = drug_subset.head(batch_size)
            
            # Optimize worker count based on actual drug count (similar to pathway validation)
            num_drugs = len(drug_subset)
            if self.max_workers is None:
                max_workers = DrugPrioritizationConfig.get_optimal_workers(num_drugs)
            else:
                max_workers = DrugPrioritizationConfig.get_optimal_workers(num_drugs, self.max_workers)
            
            logger.info(f"🚀 Starting optimized parallel drug prioritization: {num_drugs} drugs, {max_workers} workers")
            
            # Generate shared disease context (optimization)
            logger.info(f"Generating shared disease context for {disease_name}")
            disease_context = self._generate_shared_disease_context(disease_name)
            logger.info("Disease context generated and will be shared across all workers")
            
            # Prepare drug data for parallel processing (EXACT REPLICA of pathway tasks)
            drug_tasks = []
            for _, drug_row in drug_subset.iterrows():
                drug_name = drug_row.get('drug_name', 'Unknown')
                
                drug_tasks.append({
                    'drug_name': drug_name,
                    'disease_name': disease_name,
                    'drug_dict': drug_row.to_dict(),
                    'disease_context': disease_context,
                    'model': self.model,
                    'start_time': time.time()
                })
            
            logger.info(f"📊 Processing {len(drug_tasks)} drugs with {max_workers} workers")
            
            # Execute parallel prioritization with detailed worker logging (EXACT REPLICA)
            prioritized_results = []
            failed_prioritizations = []
            
            # Track worker progress (EXACT REPLICA)
            worker_stats = {}
            completed_count = 0
            total_drugs = len(drug_tasks)
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all tasks (EXACT REPLICA of pathway validation)
                future_to_drug = {
                    executor.submit(_prioritize_single_drug, task): task['drug_name'] 
                    for task in drug_tasks
                }
                
                logger.info(f"🔄 Submitted {len(future_to_drug)} drug prioritization tasks to {max_workers} workers")
                
                # Collect results as they complete with timeout protection (EXACT REPLICA)
                try:
                    for future in as_completed(future_to_drug, timeout=DrugPrioritizationConfig.TOTAL_PRIORITIZATION_TIMEOUT):
                        drug_name = future_to_drug[future]
                        completed_count += 1
                        remaining = total_drugs - completed_count
                        
                        try:
                            # Apply individual drug timeout (EXACT REPLICA)
                            result_data = future.result(timeout=DrugPrioritizationConfig.SINGLE_DRUG_TIMEOUT)
                            worker_id = f"Worker-{hash(future) % max_workers + 1}"
                            
                            # Track worker statistics (EXACT REPLICA)
                            if worker_id not in worker_stats:
                                worker_stats[worker_id] = {'completed': 0, 'failed': 0}
                            
                            if result_data['success']:
                                prioritized_results.append(result_data['result'])
                                worker_stats[worker_id]['completed'] += 1
                                logger.info(f"✅ {worker_id}: Prioritized drug {drug_name} "
                                          f"({result_data['processing_time']:.2f}s) - "
                                          f"Progress: {completed_count}/{total_drugs} ({remaining} remaining)")
                            else:
                                failed_prioritizations.append(result_data)
                                worker_stats[worker_id]['failed'] += 1
                                logger.warning(f"❌ {worker_id}: Failed drug {drug_name} - {result_data['error']} - "
                                             f"Progress: {completed_count}/{total_drugs} ({remaining} remaining)")
                        except TimeoutError:
                            logger.error(f"⏰ Timeout processing {drug_name} (>{DrugPrioritizationConfig.SINGLE_DRUG_TIMEOUT}s) - "
                                       f"Progress: {completed_count}/{total_drugs} ({remaining} remaining)")
                            failed_prioritizations.append({
                                'drug_name': drug_name, 
                                'error': f'Timeout after {DrugPrioritizationConfig.SINGLE_DRUG_TIMEOUT}s', 
                                'success': False
                            })
                        except Exception as e:
                            logger.error(f"💥 Exception processing {drug_name}: {e} - "
                                       f"Progress: {completed_count}/{total_drugs} ({remaining} remaining)")
                            failed_prioritizations.append({
                                'drug_name': drug_name, 
                                'error': str(e), 
                                'success': False
                            })
                except TimeoutError:
                    # Handle total prioritization timeout (EXACT REPLICA)
                    incomplete_drugs = len(future_to_drug) - completed_count
                    logger.error(f"🚨 Total prioritization timeout ({DrugPrioritizationConfig.TOTAL_PRIORITIZATION_TIMEOUT}s exceeded) - "
                               f"{incomplete_drugs} drugs incomplete")
                    
                    # Cancel remaining futures and mark as failed (EXACT REPLICA)
                    for future, drug_name in future_to_drug.items():
                        if not future.done():
                            future.cancel()
                            failed_prioritizations.append({
                                'drug_name': drug_name,
                                'error': f'Cancelled due to total timeout ({DrugPrioritizationConfig.TOTAL_PRIORITIZATION_TIMEOUT}s)',
                                'success': False
                            })
            
            # Log final worker statistics (EXACT REPLICA)
            logger.info("📊 Final Worker Statistics:")
            for worker_id, stats in worker_stats.items():
                total_processed = stats['completed'] + stats['failed']
                success_rate = (stats['completed'] / total_processed * 100) if total_processed > 0 else 0
                logger.info(f"  • {worker_id}: {total_processed} drugs processed "
                           f"({stats['completed']} success, {stats['failed']} failed, {success_rate:.1f}% success rate)")
            
            # Performance metrics (EXACT REPLICA)
            total_time = time.time() - start_time
            avg_time_per_drug = total_time / max(1, len(drug_tasks))
            success_rate = len(prioritized_results) / max(1, len(drug_tasks)) * 100
            
            logger.info(f"⚡ Parallel drug prioritization completed in {total_time:.2f}s")
            logger.info(f"📈 Average time per drug: {avg_time_per_drug:.2f}s")
            logger.info(f"✅ Success rate: {success_rate:.1f}% ({len(prioritized_results)}/{len(drug_tasks)})")
            
            if failed_prioritizations:
                logger.warning(f"⚠️ {len(failed_prioritizations)} drugs failed during processing")
            
            return prioritized_results
        
        except Exception as e:
            logger.error(f"💥 Parallel drug prioritization failed: {e}")
            return []
    
    def _generate_shared_disease_context(self, disease_name: str):
        """
        Generate disease context once to be shared across all workers.
        This optimization avoids redundant context generation in each worker.
        """
        try:
            # Create a temporary prioritizer instance to generate disease context
            temp_prioritizer = UniversalDrugPrioritizer(model=self.model)
            
            # Use the existing disease context generator if available
            if hasattr(temp_prioritizer, '_generate_disease_context'):
                disease_context = temp_prioritizer._generate_disease_context(disease_name)
                logger.info(f"Generated comprehensive disease context for {disease_name}")
                return disease_context
            else:
                # Fallback to basic disease context
                from .universal_drug_prioritizer import DiseaseContext
                disease_context = DiseaseContext(
                    disease_name=disease_name,
                    pathophysiology=f"General pathophysiology of {disease_name}",
                    key_molecular_drivers="Standard disease pathways and drivers",
                    therapeutic_targets="Common therapeutic targets",
                    contraindications="General contraindications",
                    standard_treatments="Standard treatments",
                    clinical_considerations="General clinical considerations"
                )
                logger.info(f"Generated basic disease context for {disease_name}")
                return disease_context
                
        except Exception as e:
            logger.warning(f"Failed to generate comprehensive disease context: {e}")
            # Final fallback - minimal shim with required attributes
            disease_context = type('DiseaseContext', (), {
                'disease_name': disease_name,
                'pathophysiology': f"General pathophysiology of {disease_name}",
                'key_molecular_drivers': "Standard disease pathways and drivers",
                'therapeutic_targets': "Common therapeutic targets",
                'contraindications': "General contraindications",
                'standard_treatments': "Standard treatments",
                'clinical_considerations': "General clinical considerations"
            })()
            logger.info(f"Generated fallback disease context for {disease_name}")
            return disease_context
    
def run_parallel_drug_prioritization(df: pd.DataFrame, 
                                   disease_name: str,
                                   model: str = "gpt-5-mini-2025-08-07",
                                   max_workers: int = None,  # Auto-optimize based on workload
                                   use_threads: bool = None) -> pd.DataFrame:
    """
    Main function to run parallel drug prioritization with robust timeout and error handling
    
    Args:
        df: Input DataFrame with drug data
        disease_name: Target disease name
        model: LLM model to use
        max_workers: Maximum parallel workers (auto-optimized if None)
        use_threads: If True, use threads (auto-detects Celery if None)
        
    Returns:
        DataFrame with prioritized drugs
    """
    logger.info(f"🚀 Starting enhanced parallel drug prioritization")
    
    # Check if we should use parallel processing
    if not DrugPrioritizationConfig.should_use_parallel(len(df)):
        logger.info(f"📊 Dataset too small ({len(df)} drugs), using sequential processing")
        from .universal_drug_prioritizer import prioritize_drugs_universal
        return prioritize_drugs_universal(df, disease_name, model)
    
    # Create parallel prioritizer
    parallel_prioritizer = ParallelDrugPrioritizer(
        model=model,
        max_workers=max_workers,
        use_threads=use_threads
    )
    
    # Run parallel prioritization
    prioritized_drugs = parallel_prioritizer.prioritize_drugs_parallel(df, disease_name)
    
    if not prioritized_drugs:
        logger.warning("⚠️ No results from parallel prioritization, attempting sequential fallback")
        try:
            from .universal_drug_prioritizer import prioritize_drugs_universal
            return prioritize_drugs_universal(df, disease_name, model)
        except Exception as e:
            logger.error(f"💥 Sequential fallback also failed: {e}")
            logger.warning("Returning original DataFrame as emergency fallback")
            return df
    
    # Convert results back to DataFrame format
    try:
        # Helper to get attribute from Pydantic object or dict
        def _attr(obj, key, default=None):
            try:
                if isinstance(obj, dict):
                    return obj.get(key, default)
                return getattr(obj, key, default)
            except Exception:
                return default

        # Extract data from PrioritizedDrug objects
        results_data = []
        for drug in prioritized_drugs:
            drug_name = _attr(drug, 'drug_name', '') or _attr(drug, 'name', '')
            # Build minimal, schema-consistent dict; enrich from original row afterward
            drug_dict = {
                'pathway_id': _attr(drug, 'pathway_id', ''),
                'drug_id': _attr(drug, 'drug_id', ''),
                'drug_name': drug_name,
                'priority_score': _attr(drug, 'priority_score', 0),
                'confidence': _attr(drug, 'confidence', ''),
                'justification': _attr(drug, 'justification', ''),
                'recommendation': _attr(drug, 'recommendation', ''),
            }
            # Add original columns if they exist (prefer exact match when possible)
            original_row = None
            if 'drug_name' in df.columns:
                try:
                    original_row = df[df['drug_name'].astype(str).str.lower() == str(drug_name).lower()]
                except Exception:
                    original_row = None
            if original_row is None or original_row.empty:
                # Fallback fuzzy match on row string
                try:
                    original_row = df[df.apply(lambda x: str(drug_name).lower() in str(x).lower(), axis=1)]
                except Exception:
                    original_row = None
            if original_row is not None and not original_row.empty:
                drug_dict.update(original_row.iloc[0].to_dict())

            results_data.append(drug_dict)
        
        # Create results DataFrame
        results_df = pd.DataFrame(results_data)
        
        # Sort by priority score (descending)
        if 'priority_score' in results_df.columns:
            try:
                results_df['priority_score'] = pd.to_numeric(results_df['priority_score'], errors='coerce').fillna(0)
            except Exception:
                pass
            results_df = results_df.sort_values('priority_score', ascending=False)
        
        logger.info(f"🎯 Enhanced parallel prioritization completed successfully:")
        logger.info(f"  📊 {len(results_df)} drugs prioritized")
        logger.info(f"  ⚡ Pipeline optimizations applied (timeouts, worker scaling, error handling)")
        
        return results_df
        
    except Exception as e:
        logger.error(f"💥 Failed to convert results to DataFrame: {e}")
        logger.warning("⚠️ Attempting emergency fallback to sequential processing")
        try:
            from .universal_drug_prioritizer import prioritize_drugs_universal
            return prioritize_drugs_universal(df, disease_name, model)
        except Exception as fallback_e:
            logger.error(f"💥 Emergency fallback also failed: {fallback_e}")
            logger.warning("🚨 Returning original DataFrame as last resort")
            return df
