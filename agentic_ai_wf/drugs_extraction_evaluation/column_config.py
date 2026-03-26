"""
Centralized column configuration for drug processing pipeline
Ensures consistent column handling across all modules
"""

class DrugColumnConfig:
    """Centralized configuration for drug processing column management"""
    
    # Core LLM-generated columns from universal_drug_prioritizer
    LLM_CORE_COLUMNS = [
        'final_rank',
        'pathway_id', 
        'drug_id', 
        'drug_name', 
        'priority_score', 
        'confidence', 
        'justification', 
        'recommendation'
    ]
    
    # Pathway analysis columns (from input data)
    PATHWAY_ANALYSIS_COLUMNS = [
        'pathway_name',
        'pathway_genes', 
        'target_genes', 
        'gene_overlap', 
        'match_status', 
        'patient_log2fc',
        'regulation', 
        'p_value', 
        'fdr'
    ]
    
    # Drug metadata columns (from KEGG/DrugBank)
    DRUG_METADATA_COLUMNS = [
        'drug_classes', 
        'mechanism_of_action', 
        'drugbank_id', 
        'chembl_id', 
        'target_genes_lfc', 
        'fda_approved_status', 
        'brand_name', 
        'generic_name', 
        'route_of_administration', 
        'efficacy'
    ]
    
    # Additional analysis columns
    ANALYSIS_COLUMNS = [
        'target_genes_property', 
        'clinical_relevance', 
        'functional_relevance', 
        'priority_rank', 
        'llm_score', 
        'score_justification', 
        'target_mechanism'
    ]
    
    # Optional/fallback columns (may be missing)
    OPTIONAL_COLUMNS = [
        'molecular_evidence_score', 
        'pathway_regulation', 
        'gene_score', 
        'disorder_score',
        'pathway_associated_genes'  # Legacy column name
    ]
    
    @classmethod
    def get_all_expected_columns(cls):
        """Get complete list of all expected columns in order"""
        return (
            cls.LLM_CORE_COLUMNS + 
            cls.PATHWAY_ANALYSIS_COLUMNS + 
            cls.DRUG_METADATA_COLUMNS + 
            cls.ANALYSIS_COLUMNS + 
            cls.OPTIONAL_COLUMNS
        )
    
    @classmethod
    def get_critical_columns(cls):
        """Get columns that must be present for pipeline to work"""
        return cls.LLM_CORE_COLUMNS + ['pathway_name', 'target_genes']
    
    @classmethod
    def get_default_value(cls, column_name):
        """Get appropriate default value for a column"""
        if column_name in ['priority_score', 'molecular_evidence_score', 'gene_score', 'disorder_score', 'llm_score']:
            return 0.0
        elif column_name == 'final_rank':
            return 1
        elif column_name == 'confidence':
            return 'MEDIUM'
        elif column_name == 'recommendation':
            return 'Consider'
        elif column_name in ['p_value', 'fdr']:
            return 1.0
        else:
            return ''
    
    @classmethod
    def add_missing_columns(cls, df, preserve_existing=True):
        """
        Add any missing columns to DataFrame with appropriate defaults
        
        Args:
            df: Input DataFrame
            preserve_existing: If True, keep all existing columns
        
        Returns:
            DataFrame with all columns present
        """
        import pandas as pd
        
        if preserve_existing:
            # Keep all existing columns and add missing ones
            expected_columns = cls.get_all_expected_columns()
            existing_columns = list(df.columns)
            
            # Find columns that are expected but missing
            missing_columns = [col for col in expected_columns if col not in existing_columns]
            
            # Add missing columns with defaults
            for col in missing_columns:
                df[col] = cls.get_default_value(col)
            
            # Special handling for final_rank if missing
            if 'final_rank' in missing_columns and len(df) > 0:
                if 'priority_score' in df.columns:
                    df = df.sort_values('priority_score', ascending=False).reset_index(drop=True)
                df['final_rank'] = range(1, len(df) + 1)
        
        return df
    
    @classmethod
    def validate_critical_columns(cls, df):
        """
        Validate that critical columns are present
        
        Returns:
            Tuple of (is_valid, missing_columns)
        """
        critical_columns = cls.get_critical_columns()
        missing = [col for col in critical_columns if col not in df.columns]
        return len(missing) == 0, missing
    
    @classmethod
    def get_column_order(cls, df_columns):
        """
        Get optimal column order based on available columns
        
        Args:
            df_columns: List of columns in the DataFrame
            
        Returns:
            List of columns in optimal order
        """
        all_expected = cls.get_all_expected_columns()
        
        # Start with expected columns that exist
        ordered_columns = [col for col in all_expected if col in df_columns]
        
        # Add any extra columns not in our config
        extra_columns = [col for col in df_columns if col not in all_expected]
        
        return ordered_columns + extra_columns
