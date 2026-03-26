"""
Validation Integration for Clinical Reports
==========================================

This module integrates the validation layer with the existing combined_stats functions
to provide validated outputs for clinical transcriptome reports.
"""

import numpy as np
import pandas as pd
import logging
import time
from typing import Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
try:
    from .disease_focused_validation import (
        DiseaseContextValidator)
    # Import disease context optimization utilities
except ImportError:
    # Handle case when imported as standalone module
    from disease_focused_validation import (
        DiseaseContextValidator)

logger = logging.getLogger(__name__)

# Performance configuration for parallel processing
class ValidationConfig:
    """Configuration class for pathway validation performance optimization"""
    
    # Parallel processing settings
    DEFAULT_MAX_WORKERS = 10
    MAX_WORKERS_LIMIT = 20  # Prevent resource exhaustion
    DEFAULT_BATCH_SIZE = 1000  # Increased for ALL genes processing
    MAX_BATCH_SIZE = 5000      # Allow large gene sets
    
    # Timeout settings (in seconds)
    SINGLE_PATHWAY_TIMEOUT = 30
    TOTAL_VALIDATION_TIMEOUT = 300  # 5 minutes total
    
    # Performance thresholds
    SLOW_PATHWAY_THRESHOLD = 10  # seconds
    PARALLEL_THRESHOLD = 5       # Use parallel if >= 5 pathways
    
    @classmethod
    def get_optimal_workers(cls, num_pathways: int, max_workers: int = None) -> int:
        """Determine optimal number of workers based on pathway count"""
        if max_workers is None:
            max_workers = cls.DEFAULT_MAX_WORKERS
        
        # Limit workers to prevent resource exhaustion
        max_workers = min(max_workers, cls.MAX_WORKERS_LIMIT)
        
        # Use fewer workers for small batches
        if num_pathways <= 5:
            return min(3, max_workers)
        elif num_pathways <= 10:
            return min(5, max_workers)
        elif num_pathways <= 20:
            return min(8, max_workers)
        else:
            return max_workers
    
    @classmethod
    def should_use_parallel(cls, num_pathways: int) -> bool:
        """Determine if parallel processing should be used"""
        return num_pathways >= cls.PARALLEL_THRESHOLD


def _validate_single_pathway(pathway_data: Dict) -> Dict:
    """
    Validate a single pathway - designed for parallel processing
    
    Args:
        pathway_data: Dictionary containing pathway info and validation parameters
    
    Returns:
        Dictionary with validation result or error info
    """
    try:
        # Extract parameters
        pathway_name = pathway_data['pathway_name']
        direction = pathway_data['direction']
        disease_name = pathway_data['disease_name']
        evidence_text = pathway_data['evidence_text']
        regulation = pathway_data['regulation']
        
        # Create disease validator for this thread
        disease_validator = DiseaseContextValidator()
        
        # Perform validation
        result = disease_validator.validate_pathway(
            pathway_name=pathway_name,
            direction=direction,
            disease=disease_name,
            evidence_text=evidence_text
        )
        
        return {
            'success': True,
            'result': result,
            'pathway_name': pathway_name,
            'regulation': regulation,
            'processing_time': time.time() - pathway_data.get('start_time', time.time())
        }
        
    except Exception as e:
        logger.warning(f"Failed to validate pathway {pathway_data.get('pathway_name', 'Unknown')}: {e}")
        return {
            'success': False,
            'error': str(e),
            'pathway_name': pathway_data.get('pathway_name', 'Unknown'),
            'processing_time': time.time() - pathway_data.get('start_time', time.time())
        }

def validate_pathways_disease_focused_parallel(pathways_df: pd.DataFrame, disease_name: str, 
                                             max_workers: int = None, batch_size: int = None) -> Dict:
    """
    Validate pathways using disease-focused structured prompts with optimized parallel processing.
    
    Args:
        pathways_df: DataFrame with pathway results
        disease_name: Target disease for validation
        max_workers: Maximum number of threads (auto-optimized if None)
        batch_size: Number of pathways to process (auto-optimized if None)
        
    Returns:
        Dict containing validated pathways with Pathogenic/Protective/Uncertain classification
    """
    start_time = time.time()
    
    try:
        # Validate input
        if 'Pathway_Name' not in pathways_df.columns:
            return {'error': 'Pathway_Name column not found'}
        
        # Auto-optimize configuration
        if batch_size is None:
            batch_size = min(ValidationConfig.DEFAULT_BATCH_SIZE, len(pathways_df))
        batch_size = min(batch_size, ValidationConfig.MAX_BATCH_SIZE)
        
        # Prepare pathway subset
        pathway_subset = pathways_df.copy()
        if 'LLM_Score' in pathway_subset.columns:
            pathway_subset = pathway_subset.nlargest(batch_size, 'LLM_Score')
        else:
            pathway_subset = pathway_subset.head(batch_size)
        
        # Optimize worker count based on actual pathway count
        num_pathways = len(pathway_subset)
        if max_workers is None:
            max_workers = ValidationConfig.get_optimal_workers(num_pathways)
        else:
            max_workers = ValidationConfig.get_optimal_workers(num_pathways, max_workers)
        
        logger.info(f"🚀 Starting optimized parallel validation: {num_pathways} pathways, {max_workers} workers")
        
        # Prepare pathway data for parallel processing
        pathway_tasks = []
        for _, pathway_row in pathway_subset.iterrows():
            pathway_name = pathway_row['Pathway_Name']
            regulation = pathway_row.get('Regulation', 'Mixed')
            
            # Determine direction from multiple sources - include both up and down
            if regulation.lower() in ['up', 'upregulated']:
                direction = "Upregulated"
            elif regulation.lower() in ['down', 'downregulated']:
                direction = "Downregulated"
            else:
                # Auto-detect direction from associated genes or default to mixed
                direction = "Mixed"
            
            # Build evidence text
            evidence_parts = []
            if 'Clinical_Relevance' in pathway_row:
                evidence_parts.append(f"Clinical relevance: {pathway_row['Clinical_Relevance']}")
            if 'Functional_Relevance' in pathway_row:
                evidence_parts.append(f"Functional relevance: {pathway_row['Functional_Relevance']}")
            if 'LLM_Score' in pathway_row:
                evidence_parts.append(f"LLM Score: {pathway_row['LLM_Score']}")
            
            evidence_text = "; ".join(evidence_parts)
            
            pathway_tasks.append({
                'pathway_name': pathway_name,
                'direction': direction,
                'disease_name': disease_name,
                'evidence_text': evidence_text,
                'regulation': regulation,
                'start_time': time.time()
            })
        
        logger.info(f"📊 Processing {len(pathway_tasks)} pathways with {max_workers} workers")
        
        # Execute parallel validation with detailed worker logging
        validated_results = []
        failed_validations = []
        
        # Track worker progress
        worker_stats = {}
        completed_count = 0
        total_pathways = len(pathway_tasks)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_pathway = {
                executor.submit(_validate_single_pathway, task): task['pathway_name'] 
                for task in pathway_tasks
            }
            
            logger.info(f"🔄 Submitted {len(future_to_pathway)} pathway validation tasks to {max_workers} workers")
            
            # Collect results as they complete with timeout protection
            try:
                for future in as_completed(future_to_pathway, timeout=ValidationConfig.TOTAL_VALIDATION_TIMEOUT):
                    pathway_name = future_to_pathway[future]
                    completed_count += 1
                    remaining = total_pathways - completed_count
                    
                    try:
                        # Apply individual pathway timeout
                        result_data = future.result(timeout=ValidationConfig.SINGLE_PATHWAY_TIMEOUT)
                        worker_id = f"Worker-{hash(future) % max_workers + 1}"
                        
                        # Track worker statistics
                        if worker_id not in worker_stats:
                            worker_stats[worker_id] = {'completed': 0, 'failed': 0}
                        
                        if result_data['success']:
                            validated_results.append(result_data['result'])
                            worker_stats[worker_id]['completed'] += 1
                            logger.info(f"✅ {worker_id}: Validated pathway {pathway_name} "
                                      f"({result_data['processing_time']:.2f}s) - "
                                      f"Progress: {completed_count}/{total_pathways} ({remaining} remaining)")
                        else:
                            failed_validations.append(result_data)
                            worker_stats[worker_id]['failed'] += 1
                            logger.warning(f"❌ {worker_id}: Failed pathway {pathway_name} - {result_data['error']} - "
                                         f"Progress: {completed_count}/{total_pathways} ({remaining} remaining)")
                    except TimeoutError:
                        logger.error(f"⏰ Timeout processing {pathway_name} (>{ValidationConfig.SINGLE_PATHWAY_TIMEOUT}s) - "
                                   f"Progress: {completed_count}/{total_pathways} ({remaining} remaining)")
                        failed_validations.append({
                            'pathway_name': pathway_name, 
                            'error': f'Timeout after {ValidationConfig.SINGLE_PATHWAY_TIMEOUT}s', 
                            'success': False
                        })
                    except Exception as e:
                        logger.error(f"💥 Exception processing {pathway_name}: {e} - "
                                   f"Progress: {completed_count}/{total_pathways} ({remaining} remaining)")
                        failed_validations.append({
                            'pathway_name': pathway_name, 
                            'error': str(e), 
                            'success': False
                        })
            except TimeoutError:
                # Handle total validation timeout
                incomplete_pathways = len(future_to_pathway) - completed_count
                logger.error(f"🚨 Total validation timeout ({ValidationConfig.TOTAL_VALIDATION_TIMEOUT}s exceeded) - "
                           f"{incomplete_pathways} pathways incomplete")
                
                # Cancel remaining futures and mark as failed
                for future, pathway_name in future_to_pathway.items():
                    if not future.done():
                        future.cancel()
                        failed_validations.append({
                            'pathway_name': pathway_name,
                            'error': f'Cancelled due to total timeout ({ValidationConfig.TOTAL_VALIDATION_TIMEOUT}s)',
                            'success': False
                        })
        
        # Log final worker statistics
        logger.info("📊 Final Worker Statistics:")
        for worker_id, stats in worker_stats.items():
            total_processed = stats['completed'] + stats['failed']
            success_rate = (stats['completed'] / total_processed * 100) if total_processed > 0 else 0
            logger.info(f"  • {worker_id}: {total_processed} pathways processed "
                       f"({stats['completed']} success, {stats['failed']} failed, {success_rate:.1f}% success rate)")
        
        # Performance metrics
        total_time = time.time() - start_time
        avg_time_per_pathway = total_time / max(1, len(pathway_tasks))
        success_rate = len(validated_results) / max(1, len(pathway_tasks)) * 100
        
        logger.info(f"⚡ Parallel validation completed in {total_time:.2f}s")
        logger.info(f"📈 Average time per pathway: {avg_time_per_pathway:.2f}s")
        logger.info(f"✅ Success rate: {success_rate:.1f}% ({len(validated_results)}/{len(pathway_tasks)})")
        
        # Categorize results by status
        pathogenic = [r for r in validated_results if r.status == "Pathogenic"]
        protective = [r for r in validated_results if r.status == "Protective"]
        uncertain = [r for r in validated_results if r.status == "Uncertain"]
        
        # Build optimized response
        validated_pathways = {
            'pathogenic_pathways': [
                {
                    'pathway_name': r.pathway,
                    'status': r.status,
                    'regulation': _get_regulation_fast(r.pathway, pathways_df),
                    'confidence': r.confidence,
                    'justification': r.justification,
                    'clinical_impact': _get_clinical_impact(r.confidence),
                    'evidence_sources': r.evidence_sources
                }
                for r in pathogenic
            ],
            'protective_pathways': [
                {
                    'pathway_name': r.pathway,
                    'status': r.status,
                    'regulation': _get_regulation_fast(r.pathway, pathways_df),
                    'confidence': r.confidence,
                    'justification': r.justification,
                    'clinical_impact': _get_clinical_impact(r.confidence),
                    'evidence_sources': r.evidence_sources
                }
                for r in protective
            ],
            'uncertain_pathways': [
                {
                    'pathway_name': r.pathway,
                    'status': r.status,
                    'justification': r.justification,
                    'confidence': r.confidence
                }
                for r in uncertain
            ],
            'validation_summary': {
                'total_pathways_analyzed': len(validated_results),
                'pathogenic_count': len(pathogenic),
                'protective_count': len(protective),
                'uncertain_count': len(uncertain),
                'failed_count': len(failed_validations),
                'average_confidence': np.mean([r.confidence for r in validated_results]) if validated_results else 0.0,
                'processing_time_seconds': total_time,
                'average_time_per_pathway': avg_time_per_pathway,
                'success_rate_percent': success_rate,
                'parallel_workers_used': max_workers
            }
        }
        
        return validated_pathways
    
    except Exception as e:
        logger.error(f"💥 Parallel pathway validation failed: {e}")
        return {
            'pathogenic_pathways': [],
            'protective_pathways': [],
            'uncertain_pathways': [],
            'validation_summary': {
                'total_pathways_analyzed': 0, 
                'average_confidence': 0.0,
                'processing_time_seconds': time.time() - start_time,
                'error': str(e)
            }
        }

def _get_regulation_fast(pathway_name: str, pathways_df: pd.DataFrame) -> str:
    """Fast regulation lookup with caching"""
    try:
        match = pathways_df[pathways_df['Pathway_Name'] == pathway_name]
        return match['Regulation'].iloc[0] if not match.empty else 'Unknown'
    except:
        return 'Unknown'

def _get_clinical_impact(confidence: float) -> str:
    """Determine clinical impact based on confidence"""
    if confidence > 0.7:
        return 'High'
    elif confidence > 0.4:
        return 'Medium'
    else:
        return 'Low'

def validate_pathways_disease_focused(pathways_df: pd.DataFrame, disease_name: str, 
                                    use_parallel: bool = None, max_workers: int = None, 
                                    batch_size: int = None) -> Dict:
    """
    Validate pathways using disease-focused structured prompts with intelligent optimization.
    
    Args:
        pathways_df: DataFrame with pathway results
        disease_name: Target disease for validation
        use_parallel: Whether to use parallel processing (auto-detected if None)
        max_workers: Maximum number of threads (auto-optimized if None)
        batch_size: Number of pathways to process (auto-optimized if None)
        
    Returns:
        Dict containing validated pathways with Pathogenic/Protective/Uncertain classification
    """
    # Auto-detect whether to use parallel processing
    if use_parallel is None:
        use_parallel = ValidationConfig.should_use_parallel(len(pathways_df))
    
    # Use parallel processing for better performance
    if use_parallel:
        return validate_pathways_disease_focused_parallel(
            pathways_df=pathways_df,
            disease_name=disease_name,
            max_workers=max_workers,
            batch_size=batch_size
        )
    
    # Legacy sequential processing (kept for compatibility)
    logger.warning("🐌 Using legacy sequential validation - consider enabling parallel processing")
    start_time = time.time()
    disease_validator = DiseaseContextValidator()
    
    try:
        # Get pathway information
        if 'Pathway_Name' not in pathways_df.columns:
            return {'error': 'Pathway_Name column not found'}
        
        # Use configurable batch size  
        if batch_size is None:
            batch_size = min(ValidationConfig.DEFAULT_BATCH_SIZE, len(pathways_df))
        batch_size = min(batch_size, ValidationConfig.MAX_BATCH_SIZE)
        
        pathway_subset = pathways_df.copy()
        if 'LLM_Score' in pathway_subset.columns:
            pathway_subset = pathway_subset.nlargest(batch_size, 'LLM_Score')
        else:
            pathway_subset = pathway_subset.head(batch_size)
        
        # Validate pathways sequentially
        validated_results = []
        for _, pathway_row in pathway_subset.iterrows():
            pathway_name = pathway_row['Pathway_Name']
            regulation = pathway_row.get('Regulation', 'Mixed')
            
            # Determine direction from multiple sources - include both up and down
            if regulation.lower() in ['up', 'upregulated']:
                direction = "Upregulated"
            elif regulation.lower() in ['down', 'downregulated']:
                direction = "Downregulated"
            else:
                # Auto-detect direction from associated genes or default to mixed
                direction = "Mixed"
            
            # Build evidence text from available columns
            evidence_parts = []
            if 'Clinical_Relevance' in pathway_row:
                evidence_parts.append(f"Clinical relevance: {pathway_row['Clinical_Relevance']}")
            if 'Functional_Relevance' in pathway_row:
                evidence_parts.append(f"Functional relevance: {pathway_row['Functional_Relevance']}")
            if 'LLM_Score' in pathway_row:
                evidence_parts.append(f"LLM Score: {pathway_row['LLM_Score']}")
            
            evidence_text = "; ".join(evidence_parts)
            
            try:
                result = disease_validator.validate_pathway(
                    pathway_name=pathway_name,
                    direction=direction,
                    disease=disease_name,
                    evidence_text=evidence_text
                )
                validated_results.append(result)
            except Exception as e:
                logger.warning(f"Failed to validate pathway {pathway_name}: {e}")
                continue
        
        # Categorize results by status
        pathogenic = [r for r in validated_results if r.status == "Pathogenic"]
        protective = [r for r in validated_results if r.status == "Protective"]
        uncertain = [r for r in validated_results if r.status == "Uncertain"]
        
        # Performance metrics
        total_time = time.time() - start_time
        
        # Build response
        validated_pathways = {
            'pathogenic_pathways': [
                {
                    'pathway_name': r.pathway,
                    'status': r.status,
                    'regulation': _get_regulation_fast(r.pathway, pathways_df),
                    'confidence': r.confidence,
                    'justification': r.justification,
                    'clinical_impact': _get_clinical_impact(r.confidence),
                    'evidence_sources': r.evidence_sources
                }
                for r in pathogenic
            ],
            'protective_pathways': [
                {
                    'pathway_name': r.pathway,
                    'status': r.status,
                    'regulation': _get_regulation_fast(r.pathway, pathways_df),
                    'confidence': r.confidence,
                    'justification': r.justification,
                    'clinical_impact': _get_clinical_impact(r.confidence),
                    'evidence_sources': r.evidence_sources
                }
                for r in protective
            ],
            'uncertain_pathways': [
                {
                    'pathway_name': r.pathway,
                    'status': r.status,
                    'justification': r.justification,
                    'confidence': r.confidence
                }
                for r in uncertain
            ],
            'validation_summary': {
                'total_pathways_analyzed': len(validated_results),
                'pathogenic_count': len(pathogenic),
                'protective_count': len(protective),
                'uncertain_count': len(uncertain),
                'average_confidence': np.mean([r.confidence for r in validated_results]) if validated_results else 0.0,
                'processing_time_seconds': total_time,
                'processing_mode': 'sequential'
            }
        }
        
        return validated_pathways
    
    except Exception as e:
        logger.error(f"Sequential pathway validation failed: {e}")
        return {
            'pathogenic_pathways': [],
            'protective_pathways': [],
            'uncertain_pathways': [],
            'validation_summary': {
                'total_pathways_analyzed': 0, 
                'average_confidence': 0.0,
                'processing_time_seconds': time.time() - start_time,
                'error': str(e)
            }
        }

# Keep the original function for backward compatibility
def validate_pathways(pathways_df: pd.DataFrame, disease_name: str) -> Dict:
    """Backward compatibility wrapper"""
    return validate_pathways_disease_focused(pathways_df, disease_name)

