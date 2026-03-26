# -*- coding: utf-8 -*-
"""
Universal Drug Prioritization Tool
Integration layer for existing codebase with the new prioritization strategy.
"""

import os
import logging
import pandas as pd
from typing import Optional, Dict, Any
from pathlib import Path
from .column_config import DrugColumnConfig
from .universal_drug_prioritizer import UniversalDrugPrioritizer
from .data_enrichment_processor import enrich_drug_data
# from ..config.config import DEFAULT_MODEL
DEFAULT_MODEL = "gpt-5-mini-2025-08-07"

logger = logging.getLogger(__name__)

class UniversalPrioritizationTool:
    """
    Main integration tool for universal drug prioritization.
    Works with existing codebase data formats and workflows.
    """
    
    def __init__(self, model: str = DEFAULT_MODEL):
        self.model = model
        self.prioritizer = None
        logger.info(f"Initialized UniversalPrioritizationTool with model: {model}")
    
    def prioritize_from_dataframe(
        self, 
        df: pd.DataFrame, 
        disease_name: str,
        output_path: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Main entry point for dataframe-based prioritization.
        
        Args:
            df: Input DataFrame with drug-pathway data
            disease_name: Target disease name  
            output_path: Optional path to save results
            
        Returns:
            Prioritized drugs DataFrame
        """
        try:
            logger.info(f"Starting prioritization for {disease_name}")
            logger.info(f"Input data shape: {df.shape}")
            
            # Step 1: Data enrichment
            enriched_df = self._enrich_input_data(df)
            
            # Step 2: Run prioritization
            results_df = self._run_prioritization(enriched_df, disease_name)

            # Step 2.1: Remove drugs with less than 30 priority score
            results_df = results_df[results_df['priority_score'] > 0]
            
            # Step 3: Post-process results
            final_results = self._post_process_results(results_df, enriched_df)

            logger.info(f"Final results DF shape: {final_results.shape}")
            
            # Step 4: Save if output path provided
            if output_path:
                self._save_results(final_results, output_path)
            
            logger.info(f"Prioritization completed: {len(final_results)} drugs prioritized")
            return final_results
            
        except Exception as e:
            logger.error(f"Prioritization failed: {e}")
            return self._create_emergency_fallback(df)
    
    def prioritize_from_csv(
        self,
        csv_path: str,
        disease_name: str,
        output_path: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Prioritize drugs from CSV file.
        
        Args:
            csv_path: Path to input CSV file
            disease_name: Target disease name
            output_path: Optional path to save results
            
        Returns:
            Prioritized drugs DataFrame
        """
        try:
            df = pd.read_csv(csv_path)
            logger.info(f"Loaded data from {csv_path}: {df.shape}")
            return self.prioritize_from_dataframe(df, disease_name, output_path)
        except Exception as e:
            logger.error(f"Failed to load CSV {csv_path}: {e}")
            raise
    
    def _enrich_input_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enrich input data with required columns"""
        try:
            enriched_df = enrich_drug_data(df)
            logger.info(f"Data enrichment completed: {enriched_df.shape}")
            return enriched_df
        except Exception as e:
            logger.error(f"Data enrichment failed: {e}")
            return df  # Return original if enrichment fails
    
    def _run_prioritization(self, df: pd.DataFrame, disease_name: str) -> pd.DataFrame:
        """Run the core prioritization algorithm"""
        if self.prioritizer is None:
            self.prioritizer = UniversalDrugPrioritizer(model=self.model)
        
        return self.prioritizer.prioritize_drugs(df, disease_name)
    
    def _post_process_results(self, results_df: pd.DataFrame, original_df: pd.DataFrame) -> pd.DataFrame:
        """Post-process results to add additional context"""
        if results_df.empty:
            return results_df
        
        # Merge back with original data for additional context
        try:
            # Create lookup for additional details
            original_lookup = original_df.set_index(['pathway_id', 'drug_id']).to_dict('index')
            
            # Add context columns
            enhanced_results = []
            for _, row in results_df.iterrows():
                result_dict = row.to_dict()
                
                # Look up additional details
                key = (row['pathway_id'], row['drug_id'])
                if key in original_lookup:
                    additional_data = original_lookup[key]
                    logger.info(f"Key: {key}")
                    logger.info(f"Appending additional data: {additional_data}")

                    result_dict.update({
                        'pathway_name': additional_data.get('pathway_name', ''),
                        'pathway_associated_genes': additional_data.get('pathway_genes', ''),
                        'target_genes': additional_data.get('target_genes', ''),
                        'gene_overlap': additional_data.get('gene_overlap', ''),
                        'patient_log2fc': additional_data.get('patient_log2fc', ''),
                        'llm_score': additional_data.get('llm_score', ''),
                        'score_justification': additional_data.get('score_justification', ''),
                        'target_mechanism': additional_data.get('target_mechanism', ''),
                        'mechanism_of_action': additional_data.get('mechanism_of_action', ''),
                        'fda_approved_status': additional_data.get('fda_approved_status', ''),
                        'route_of_administration': additional_data.get('route_of_administration', ''),
                        'molecular_evidence_score': additional_data.get('molecular_evidence_score', 0.0),
                        'pathway_regulation': additional_data.get('regulation', ''),
                        # Add missing required columns with safe defaults
                        'p_value': additional_data.get('p_value', ''),
                        'fdr': additional_data.get('fdr', ''),
                        'gene_score': additional_data.get('gene_score', 0.0),
                        'disorder_score': additional_data.get('disorder_score', 0.0)
                    })
                
                enhanced_results.append(result_dict)
            
            final_df = pd.DataFrame(enhanced_results)
            
            
            
            # Add any missing columns using centralized configuration
            final_df = DrugColumnConfig.add_missing_columns(final_df, preserve_existing=True)
            
            # Validate critical columns are present
            is_valid, missing_critical = DrugColumnConfig.validate_critical_columns(final_df)
            if not is_valid:
                logger.warning(f"Missing critical columns: {missing_critical}")
            
            logger.info(f"Post-processing completed: {final_df.shape} preserving all {len(final_df.columns)} columns")
            logger.info(f"Key columns preserved: recommendation={final_df.get('recommendation') is not None}")
            return final_df
            
        except Exception as e:
            logger.warning(f"Post-processing failed: {e}")
            return results_df
    
    def _save_results(self, results_df: pd.DataFrame, output_path: str) -> None:
        """Save results to file"""
        try:
            # Ensure output directory exists
            output_dir = Path(output_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save to CSV
            results_df.to_csv(output_path, index=False)
            logger.info(f"Results saved to {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
    
    def _create_emergency_fallback(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create emergency fallback results"""
        logger.warning("Creating emergency fallback results")
        
        fallback_data = []
        for i, (_, row) in enumerate(df.head(3).iterrows(), 1):
            fallback_data.append({
                'final_rank': i,
                'drug_id': row.get('drug_id', ''),
                'drug_name': row.get('drug_name', ''),
                'priority_score': 25,  # Very low score
                'confidence': 'LOW',
                'justification': 'Emergency fallback - manual review required',
                'pathway_name': row.get('pathway_name', ''),
                'target_genes': row.get('target_genes', ''),
                'mechanism_of_action': row.get('mechanism_of_action', '')
            })
        
        return pd.DataFrame(fallback_data)
    
    def get_prioritization_summary(self, results_df: pd.DataFrame) -> Dict[str, Any]:
        """Generate summary statistics for prioritization results"""
        if results_df.empty:
            return {'total_drugs': 0, 'error': 'No results'}
        
        summary = {
            'total_drugs': len(results_df),
            'high_confidence': len(results_df[results_df['confidence'] == 'HIGH']),
            'medium_confidence': len(results_df[results_df['confidence'] == 'MEDIUM']),
            'low_confidence': len(results_df[results_df['confidence'] == 'LOW']),
            'avg_priority_score': results_df['priority_score'].mean(),
            'top_drug': results_df.iloc[0]['drug_name'] if len(results_df) > 0 else None,
            'score_distribution': {
                'high_score_80_plus': len(results_df[results_df['priority_score'] >= 80]),
                'medium_score_60_79': len(results_df[(results_df['priority_score'] >= 60) & (results_df['priority_score'] < 80)]),
                'low_score_below_60': len(results_df[results_df['priority_score'] < 60])
            }
        }
        
        return summary

# ===== Integration Functions =====

def run_universal_prioritization(
    data_input,  # Can be DataFrame, CSV path, or dict
    disease_name: str,
    output_path: Optional[str] = None,
    model: str = DEFAULT_MODEL
) -> pd.DataFrame:
    """
    Universal prioritization function that accepts multiple input types.
    
    Args:
        data_input: DataFrame, CSV path, or dictionary with drug data
        disease_name: Target disease name
        output_path: Optional output file path
        model: OpenAI model to use
        
    Returns:
        Prioritized drugs DataFrame
    """
    tool = UniversalPrioritizationTool(model=model)
    
    # Handle different input types
    if isinstance(data_input, pd.DataFrame):
        return tool.prioritize_from_dataframe(data_input, disease_name, output_path)
    elif isinstance(data_input, str) and data_input.endswith('.csv'):
        return tool.prioritize_from_csv(data_input, disease_name, output_path)
    elif isinstance(data_input, dict):
        df = pd.DataFrame(data_input)
        return tool.prioritize_from_dataframe(df, disease_name, output_path)
    else:
        raise ValueError("data_input must be DataFrame, CSV path, or dictionary")

def prioritize_with_existing_data(
    pathway_data: pd.DataFrame,
    drug_data: pd.DataFrame,
    patient_expression: pd.DataFrame,
    disease_name: str
) -> pd.DataFrame:
    """
    Prioritize drugs using separate pathway, drug, and expression datasets.
    Integrates with existing data structures.
    
    Args:
        pathway_data: Pathway enrichment data
        drug_data: Drug information data
        patient_expression: Patient expression data
        disease_name: Target disease name
        
    Returns:
        Prioritized drugs DataFrame
    """
    try:
        # Merge datasets
        merged_data = _merge_datasets(pathway_data, drug_data, patient_expression)
        
        # Run prioritization
        tool = UniversalPrioritizationTool()
        results = tool.prioritize_from_dataframe(merged_data, disease_name)
        
        return results
        
    except Exception as e:
        logger.error(f"Multi-dataset prioritization failed: {e}")
        raise

def _merge_datasets(pathway_data: pd.DataFrame, drug_data: pd.DataFrame, expression_data: pd.DataFrame) -> pd.DataFrame:
    """Helper function to merge multiple datasets"""
    try:
        # Start with pathway data as base
        merged = pathway_data.copy()
        
        # Add drug information if available
        if not drug_data.empty and 'drug_id' in drug_data.columns:
            merged = merged.merge(drug_data, on='drug_id', how='left', suffixes=('', '_drug'))
        
        # Add expression data if available
        if not expression_data.empty:
            # This would need to be customized based on expression data format
            # For now, assume it has gene-level log2FC data
            pass
        
        return merged
        
    except Exception as e:
        logger.error(f"Dataset merging failed: {e}")
        return pathway_data  # Return original if merge fails

# ===== Compatibility Functions =====

def prioritize_drugs(df: pd.DataFrame, disease_name: str, **kwargs) -> pd.DataFrame:
    """
    Drop-in replacement for existing prioritize_drugs function.
    Maintains API compatibility while using new prioritization strategy.
    """
    return run_universal_prioritization(df, disease_name, **kwargs)

if __name__ == "__main__":
    # Example usage
    sample_data = {
        'drug_id': ['D00001', 'D00002'],
        'drug_name': ['Aspirin', 'Metformin'],
        'pathway_id': ['hsa04611', 'hsa04152'],
        'pathway_name': ['Platelet activation', 'AMPK signaling'],
        'target_genes': ['PTGS1,PTGS2', 'PRKAA1,PRKAA2'],
        'gene_overlap': ['PTGS1', 'PRKAA1'],
        'patient_log2fc': ['1.2', '0.8'],
        'mechanism_of_action': ['COX inhibitor', 'AMPK activator'],
        'fda_approved_status': ['APPROVED', 'APPROVED']
    }
    
    result = run_universal_prioritization(sample_data, "Type 2 Diabetes")
    print(f"Prioritization completed: {len(result)} drugs")
    print(result.head())
