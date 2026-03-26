# -*- coding: utf-8 -*-
"""
Data Enrichment Processor for Universal Drug Prioritization
Handles missing column generation and data validation for the pipeline.
"""

import pandas as pd
import numpy as np
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)

class DataEnrichmentProcessor:
    """
    Processes input data to ensure all required columns are present
    with intelligent inference and validation.
    """
    
    REQUIRED_COLUMNS = [
        'pathway_id', 'pathway_name', 'pathway_genes', 'drug_id', 'drug_name',
        'target_genes', 'gene_overlap', 'match_status', 'patient_log2fc',
        'drug_classes', 'mechanism_of_action', 'drugbank_id', 'chembl_id',
        'target_genes_lfc', 'fda_approved_status', 'brand_name', 'generic_name',
        'route_of_administration', 'efficacy', 'target_genes_property',
        'clinical_relevance', 'functional_relevance', 'regulation', 'p_value',
        'fdr', 'priority_rank', 'llm_score', 'score_justification',
        'target_mechanism',
        # New strategy columns
        'molecular_evidence_score'
    ]
    
    def __init__(self):
        self.column_processors = {
            'molecular_evidence_score': self._calculate_molecular_evidence
        }
    
    def enrich_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Main enrichment function that adds missing columns with intelligent defaults.
        """
        enriched_df = df.copy()
        added_columns = []
        
        for col in self.REQUIRED_COLUMNS:
            if col not in enriched_df.columns:
                if col in self.column_processors:
                    enriched_df[col] = enriched_df.apply(self.column_processors[col], axis=1)
                    added_columns.append(col)
                else:
                    # Default empty string for missing columns
                    enriched_df[col] = ""
                    added_columns.append(col)
        
        logger.info(f"Added {len(added_columns)} missing columns: {added_columns}")
        return enriched_df
    
    def _calculate_molecular_evidence(self, row: pd.Series) -> float:
        """Calculate molecular evidence score (0-100)"""
        score = 0.0
        
        # Gene overlap scoring (40 points max)
        try:
            overlaps = str(row.get('gene_overlap', '')).split(',')
            overlap_count = len([g for g in overlaps if g.strip()])
            score += min(40, overlap_count * 10)
        except:
            pass
        
        # Expression data scoring (30 points max)
        try:
            log2fc_str = str(row.get('patient_log2fc', ''))
            if log2fc_str and log2fc_str != 'nan':
                # Parse log2fc values
                values = []
                if ',' in log2fc_str:
                    values = [float(x.strip()) for x in log2fc_str.split(',') if x.strip()]
                else:
                    values = [float(log2fc_str)]
                
                if values:
                    max_abs_fc = max(abs(v) for v in values)
                    score += min(30, max_abs_fc * 10)
        except:
            pass
        
        # FDA approval scoring (20 points max)
        fda_status = str(row.get('fda_approved_status', '')).upper()
        if fda_status == 'APPROVED':
            score += 20
        elif fda_status in ['YES', 'TRUE']:
            score += 20
        
        # Mechanism detail scoring (10 points max)
        mechanism = str(row.get('mechanism_of_action', ''))
        if len(mechanism) > 50:  # Detailed mechanism
            score += 10
        elif len(mechanism) > 20:
            score += 5
        
        return min(100.0, max(0.0, score))
    
    def _assess_pathway_strength(self, row: pd.Series) -> str:
        """Assess pathway association strength"""
        try:
            # Count gene overlaps
            overlaps = str(row.get('gene_overlap', '')).split(',')
            overlap_count = len([g for g in overlaps if g.strip()])
            
            # Count total pathway genes
            pathway_genes = str(row.get('pathway_genes', '')).split(',')
            total_genes = len([g for g in pathway_genes if g.strip()])
            
            if overlap_count >= 3:
                return "STRONG"
            elif overlap_count >= 1 and total_genes > 0:
                overlap_ratio = overlap_count / total_genes
                if overlap_ratio > 0.3:
                    return "STRONG"
                elif overlap_ratio > 0.1:
                    return "MODERATE"
            
            return "WEAK" if overlap_count == 0 else "MODERATE"
        except:
            return "MODERATE"
    
    def _check_expression_match(self, row: pd.Series) -> str:
        """Check expression pattern consistency"""
        try:
            # Get patient expression
            log2fc_str = str(row.get('patient_log2fc', ''))
            if not log2fc_str or log2fc_str == 'nan':
                return "UNKNOWN"
            
            # Parse expression values
            values = []
            if ',' in log2fc_str:
                values = [float(x.strip()) for x in log2fc_str.split(',') if x.strip()]
            else:
                values = [float(log2fc_str)]
            
            if not values:
                return "UNKNOWN"
            
            # Determine if targets are generally upregulated
            avg_fc = np.mean(values)
            is_upregulated = avg_fc >= 0.58  # ~1.5x threshold
            
            # Check drug mechanism
            mechanism = str(row.get('mechanism_of_action', '')).lower()
            is_inhibitory = any(term in mechanism for term in ['inhibitor', 'antagonist', 'blocker'])
            
            # Consistent if upregulated targets with inhibitory drug
            if is_upregulated and is_inhibitory:
                return "CONSISTENT"
            elif not is_upregulated and not is_inhibitory:
                return "CONSISTENT"
            else:
                return "INCONSISTENT"
        except:
            return "UNKNOWN"
    
    def _infer_pathway_role(self, row: pd.Series) -> str:
        """Infer pathway role in disease context"""
        pathway_name = str(row.get('pathway_name', '')).lower()
        
        # Disease-promoting indicators
        promoting_terms = [
            'cancer', 'tumor', 'oncogene', 'metastasis', 'carcinogenesis',
            'cell proliferation', 'angiogenesis', 'invasion', 'migration',
            'inflammation', 'inflammatory', 'oxidative stress',
            'apoptosis resistance', 'drug resistance'
        ]
        
        if any(term in pathway_name for term in promoting_terms):
            return "DISEASE-PROMOTING"
        
        # Disease-protective indicators
        protective_terms = [
            'dna repair', 'dna damage response', 'tumor suppressor',
            'cell cycle checkpoint', 'apoptosis induction',
            'immune surveillance', 'antioxidant', 'detoxification',
            'genome stability', 'homologous recombination'
        ]
        
        if any(term in pathway_name for term in protective_terms):
            return "DISEASE-PROTECTIVE"
        
        # Immune-modulatory indicators
        immune_terms = [
            'immune', 'immunity', 'cytokine', 'interferon', 'interleukin',
            'lymphocyte', 't cell', 'b cell', 'macrophage', 'dendritic',
            'complement', 'antibody', 'antigen processing'
        ]
        
        if any(term in pathway_name for term in immune_terms):
            return "IMMUNE-MODULATORY"
        
        return "CONTEXT-DEPENDENT"
    
    def _infer_target_role(self, row: pd.Series) -> str:
        """Infer target role in pathway"""
        mechanism = str(row.get('mechanism_of_action', '')).lower()
        target_genes = str(row.get('target_genes', '')).upper()
        
        # Known negative regulators
        negative_regulators = [
            'TP53', 'RB1', 'CDKN1A', 'CDKN1B', 'CDKN2A',
            'BRCA1', 'BRCA2', 'ATM', 'ATR', 'CHEK1', 'CHEK2'
        ]
        
        # Check if any targets are known tumor suppressors/negative regulators
        target_list = [g.strip() for g in target_genes.split(',')]
        if any(gene in negative_regulators for gene in target_list):
            return "NEGATIVE_REGULATOR"
        
        # Check mechanism for regulatory hints
        if any(term in mechanism for term in ['suppressor', 'inhibitor', 'repressor']):
            return "NEGATIVE_REGULATOR"
        
        return "POSITIVE_REGULATOR"
    
    def _infer_drug_effect(self, row: pd.Series) -> str:
        """Infer drug effect on target"""
        mechanism = str(row.get('mechanism_of_action', '')).lower()
        
        # Direct mappings from mechanism keywords
        effect_mappings = {
            'inhibitor': 'INHIBITOR',
            'antagonist': 'ANTAGONIST',
            'blocker': 'INHIBITOR',
            'suppressor': 'INHIBITOR',
            'degrader': 'DEGRADER',
            'depleter': 'DEGRADER',
            'activator': 'ACTIVATOR',
            'agonist': 'AGONIST',
            'enhancer': 'ACTIVATOR',
            'inducer': 'ACTIVATOR',
            'stimulator': 'ACTIVATOR'
        }
        
        for keyword, effect in effect_mappings.items():
            if keyword in mechanism:
                return effect
        
        return "MODULATOR"
    
    def _assess_clinical_relevance(self, row: pd.Series) -> str:
        """Assess clinical relevance"""
        fda_status = str(row.get('fda_approved_status', '')).upper()
        drug_classes = str(row.get('drug_classes', '')).lower()
        route = str(row.get('route_of_administration', '')).lower()
        
        # High relevance for FDA approved systemic drugs
        if fda_status in ['APPROVED', 'YES', 'TRUE']:
            if not any(term in f"{drug_classes} {route}" for term in ['topical', 'cream', 'ointment']):
                return "HIGH"
        
        # Check for investigational drugs
        if any(term in drug_classes for term in ['investigational', 'clinical trial', 'experimental']):
            return "MODERATE"
        
        return "LOW"
    
    def _assess_functional_relevance(self, row: pd.Series) -> str:
        """Assess functional relevance"""
        pathway_name = str(row.get('pathway_name', '')).lower()
        mechanism = str(row.get('mechanism_of_action', ''))
        
        # High functional relevance for key pathways
        high_relevance_pathways = [
            'cancer', 'apoptosis', 'cell cycle', 'dna repair',
            'signal transduction', 'metabolism', 'immune'
        ]
        
        if any(term in pathway_name for term in high_relevance_pathways):
            if len(mechanism) > 30:  # Detailed mechanism
                return "HIGH"
            else:
                return "MODERATE"
        
        return "MODERATE"
    
    def validate_enriched_data(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """Validate enriched dataframe and return validation report"""
        validation_report = {
            'missing_columns': [],
            'empty_critical_columns': [],
            'invalid_values': []
        }
        
        # Check for missing required columns
        for col in self.REQUIRED_COLUMNS:
            if col not in df.columns:
                validation_report['missing_columns'].append(col)
        
        # Check for empty critical columns
        critical_columns = ['drug_id', 'drug_name', 'pathway_id', 'pathway_name']
        for col in critical_columns:
            if col in df.columns and df[col].isna().any():
                validation_report['empty_critical_columns'].append(col)
        
        try:
            # Check for invalid molecular evidence scores
            if 'molecular_evidence_score' in df.columns:
                molecular_evidence_score = int(df['molecular_evidence_score'])
                invalid_scores = df[
                    (molecular_evidence_score < 0) | 
                    (molecular_evidence_score > 100)
                ]
                if not invalid_scores.empty:
                    validation_report['invalid_values'].append(
                        f"molecular_evidence_score out of range: {len(invalid_scores)} rows"
                    )
        except:
            pass
        return validation_report

def enrich_drug_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convenience function for data enrichment.
    
    Args:
        df: Input DataFrame with drug-pathway data
        
    Returns:
        Enriched DataFrame with all required columns
    """
    processor = DataEnrichmentProcessor()
    enriched_df = processor.enrich_dataframe(df)
    
    # Validate results
    validation_report = processor.validate_enriched_data(enriched_df)
    if any(validation_report.values()):
        logger.warning(f"Data validation issues: {validation_report}")
    
    return enriched_df
