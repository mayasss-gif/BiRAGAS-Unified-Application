"""
Gene mapper tool for DEG Pipeline Agent.
"""

import pandas as pd
import re
from typing import Union, Optional, Dict, List
from pathlib import Path

try:
    import mygene
    MYGENE_AVAILABLE = True
except ImportError:
    MYGENE_AVAILABLE = False

from .base_tool import BaseTool
from ..exceptions import GeneMapperError, ValidationError


class GeneMapperTool(BaseTool):
    """Tool for mapping gene IDs to HGNC symbols."""
    
    @property
    def name(self) -> str:
        return "GeneMapper"
    
    @property
    def description(self) -> str:
        return "Map gene IDs to HGNC symbols using MyGene.info"
    
    def execute(self, df: pd.DataFrame, id_column: str = "Gene", **kwargs) -> pd.DataFrame:
        """
        Map gene IDs to symbols with intelligent ID type detection.
        
        Args:
            df: DataFrame containing gene IDs
            id_column: Column name containing gene IDs
            **kwargs: Additional parameters
            
        Returns:
            DataFrame with mapped gene symbols
            
        Raises:
            GeneMapperError: If mapping fails
        """
        if not MYGENE_AVAILABLE:
            self.logger.warning("⚠️ MyGene not available, skipping gene mapping")
            return self._add_original_id_column(df, id_column)
        
        try:
            # Create a copy to avoid modifying original
            result_df = df.copy()
            
            # Extract gene IDs
            if id_column in result_df.columns:
                gene_ids = result_df[id_column].astype(str).tolist()
            elif result_df.index.name == id_column:
                gene_ids = result_df.index.astype(str).tolist()
            else:
                raise GeneMapperError(f"Column '{id_column}' not found in DataFrame")
            
            # Check if mapping is needed
            mapping_needed, id_types = self._analyze_gene_ids(gene_ids)
            
            if not mapping_needed:
                self.logger.info("✅ Gene IDs are already symbols, skipping mapping")
                return self._add_original_id_column(result_df, id_column)
            
            self.logger.info(f"🔍 Detected gene ID types: {id_types}")
            
            # Map IDs to symbols
            id_to_symbol = self._map_ids_to_symbols(gene_ids)
            
            # Apply mapping
            if id_column in result_df.columns:
                result_df["Original_ID"] = result_df[id_column]
                result_df[id_column] = result_df[id_column].map(id_to_symbol)
                result_df = result_df.set_index(id_column)
            else:
                result_df["Original_ID"] = result_df.index
                new_index = [id_to_symbol.get(str(gene_id), str(gene_id)) for gene_id in result_df.index]
                result_df.index = new_index
                result_df.index.name = id_column
            
            mapped_count = sum(1 for old_id, new_id in id_to_symbol.items() if old_id != new_id)
            self.logger.info(f"✅ Mapped {mapped_count}/{len(gene_ids)} gene IDs to symbols")
            
            return result_df
            
        except Exception as e:
            raise GeneMapperError(f"Gene mapping failed: {e}")
    
    def _analyze_gene_ids(self, gene_ids: List[str]) -> tuple:
        """
        Analyze gene IDs to determine if mapping is needed and identify ID types.
        
        Args:
            gene_ids: List of gene IDs to analyze
            
        Returns:
            Tuple of (mapping_needed: bool, id_types: dict with counts)
        """
        id_types = {
            'symbols': 0,
            'ensembl_gene': 0,
            'ensembl_transcript': 0,
            'entrez': 0,
            'other': 0
        }
        
        # Sample up to 100 IDs for analysis to avoid performance issues
        sample_ids = gene_ids[:100] if len(gene_ids) > 100 else gene_ids
        
        for gene_id in sample_ids:
            gene_id = str(gene_id).strip()
            
            if re.match(r'^ENSG\d{11}$', gene_id):
                id_types['ensembl_gene'] += 1
            elif re.match(r'^ENST\d{11}$', gene_id):
                id_types['ensembl_transcript'] += 1
            elif re.match(r'^\d+$', gene_id):
                id_types['entrez'] += 1
            elif re.match(r'^[A-Z][A-Z0-9-]*$', gene_id) and len(gene_id) <= 20:
                # Likely gene symbol (uppercase, reasonable length)
                id_types['symbols'] += 1
            else:
                id_types['other'] += 1
        
        # Determine if mapping is needed
        total_sampled = len(sample_ids)
        symbol_ratio = id_types['symbols'] / total_sampled if total_sampled > 0 else 0
        
        # If >80% are already symbols, skip mapping
        mapping_needed = symbol_ratio < 0.8
        
        return mapping_needed, id_types
    
    def _map_ids_to_symbols(self, gene_ids: list) -> dict:
        """
        Map gene IDs to symbols using MyGene.info with enhanced transcript handling.
        
        Args:
            gene_ids: List of gene IDs
            
        Returns:
            Dictionary mapping IDs to symbols
        """
        try:
            mg = mygene.MyGeneInfo()
            id_to_symbol = {}
            
            # Separate transcript IDs from other IDs
            transcript_ids = []
            other_ids = []
            
            for gene_id in gene_ids:
                gene_id_str = str(gene_id).strip()
                if re.match(r'^ENST\d{11}$', gene_id_str):
                    transcript_ids.append(gene_id_str)
                else:
                    other_ids.append(gene_id_str)
            
            # Handle transcript IDs specially
            if transcript_ids:
                self.logger.info(f"🧬 Handling {len(transcript_ids)} Ensembl transcript IDs")
                transcript_mapping = self._map_transcript_ids(transcript_ids, mg)
                id_to_symbol.update(transcript_mapping)
            
            # Handle other IDs with standard method
            if other_ids:
                self.logger.info(f"🔗 Handling {len(other_ids)} other gene IDs")
                other_mapping = self._map_standard_ids(other_ids, mg)
                id_to_symbol.update(other_mapping)
            
            # Ensure all IDs have a mapping (use original ID if no mapping found)
            for gene_id in gene_ids:
                if str(gene_id) not in id_to_symbol:
                    id_to_symbol[str(gene_id)] = str(gene_id)
            
            return id_to_symbol
            
        except Exception as e:
            self.logger.warning(f"⚠️ MyGene mapping failed: {e}")
            # Return identity mapping
            return {str(gene_id): str(gene_id) for gene_id in gene_ids}
    
    def _map_transcript_ids(self, transcript_ids: List[str], mg) -> Dict[str, str]:
        """
        Map Ensembl transcript IDs to gene symbols using batch processing.
        
        Args:
            transcript_ids: List of Ensembl transcript IDs
            mg: MyGeneInfo instance
            
        Returns:
            Dictionary mapping transcript IDs to symbols
        """
        id_to_symbol = {}
        
        try:
            # Process in batches of 1000 (MyGene.info recommended batch size)
            batch_size = 1000
            total_transcripts = len(transcript_ids)
            self.logger.info(f"Mapping {total_transcripts} transcript IDs in batches of {batch_size}")
            
            # Process all transcript IDs in batches
            for i in range(0, total_transcripts, batch_size):
                batch = transcript_ids[i:i+batch_size]
                self.logger.info(f"Processing batch {i//batch_size + 1}/{(total_transcripts-1)//batch_size + 1} ({len(batch)} transcripts)")
                
                # Convert transcript IDs to proper query format for batch processing
                # Using querymany with a custom scope for transcript IDs
                query_results = mg.querymany(
                    batch,
                    scopes="ensembl.transcript",
                    fields="ensembl.gene,symbol,name",
                    species=self.config.gene_species
                )
                
                # Process results
                for result in query_results:
                    query_id = str(result["query"])
                    symbol = result.get("symbol")
                    
                    if symbol:
                        id_to_symbol[query_id] = symbol
                        self.logger.debug(f"📍 Mapped transcript {query_id} -> {symbol}")
                    else:
                        id_to_symbol[query_id] = query_id
                        self.logger.debug(f"❌ No symbol found for transcript {query_id}")
                
                # Add any missing IDs (those that didn't return results)
                for transcript_id in batch:
                    if transcript_id not in id_to_symbol:
                        id_to_symbol[transcript_id] = transcript_id
                        self.logger.debug(f"❌ No result for transcript {transcript_id}")
                
        except Exception as e:
            self.logger.warning(f"⚠️ Batch transcript mapping failed: {e}")
            # Fall back to returning the original IDs
            id_to_symbol = {str(transcript_id): str(transcript_id) for transcript_id in transcript_ids}
        
        return id_to_symbol
    
    def _map_standard_ids(self, gene_ids: List[str], mg) -> Dict[str, str]:
        """
        Map standard gene IDs using the regular MyGene querymany approach.
        
        Args:
            gene_ids: List of gene IDs
            mg: MyGeneInfo instance
            
        Returns:
            Dictionary mapping IDs to symbols
        """
        try:
            # Query MyGene.info
            query_results = mg.querymany(
                gene_ids,
                scopes=self.config.gene_scopes,
                fields=self.config.gene_fields,
                species=self.config.gene_species
            )
            
            # Build mapping dictionary
            id_to_symbol = {}
            for result in query_results:
                query_id = str(result["query"])
                symbol = result.get("symbol", query_id)
                id_to_symbol[query_id] = symbol
            
            return id_to_symbol
            
        except Exception as e:
            self.logger.warning(f"⚠️ Standard mapping failed: {e}")
            return {str(gene_id): str(gene_id) for gene_id in gene_ids}
    
    def _add_original_id_column(self, df: pd.DataFrame, id_column: str) -> pd.DataFrame:
        """
        Add original ID column when mapping is not available.
        
        Args:
            df: Input DataFrame
            id_column: ID column name
            
        Returns:
            DataFrame with Original_ID column
        """
        result_df = df.copy()
        
        if id_column in result_df.columns:
            result_df["Original_ID"] = result_df[id_column]
            result_df = result_df.set_index(id_column)
        else:
            result_df["Original_ID"] = result_df.index
            result_df.index.name = id_column
        
        return result_df
    
    def validate_input(self, df: pd.DataFrame, id_column: str = "Gene", **kwargs) -> None:
        """
        Validate input parameters.
        
        Args:
            df: Input DataFrame
            id_column: ID column name
            **kwargs: Additional parameters
            
        Raises:
            ValidationError: If validation fails
        """
        if df.empty:
            raise ValidationError("Input DataFrame is empty")
        
        if id_column not in df.columns and df.index.name != id_column:
            raise ValidationError(f"Column '{id_column}' not found in DataFrame")
    
    def validate_output(self, result: pd.DataFrame) -> pd.DataFrame:
        """
        Validate output DataFrame.
        
        Args:
            result: Output DataFrame
            
        Returns:
            Validated DataFrame
            
        Raises:
            ValidationError: If validation fails
        """
        if result.empty:
            raise ValidationError("Output DataFrame is empty")
        
        if "Original_ID" not in result.columns:
            raise ValidationError("Output DataFrame missing Original_ID column")
        
        return result 