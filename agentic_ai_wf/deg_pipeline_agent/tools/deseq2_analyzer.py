"""
Unified DEG analyzer tool for DEG Pipeline Agent.
Automatically selects and runs DESeq2, edgeR, or limma-voom based on data characteristics.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union

try:
    from pydeseq2.dds import DeseqDataSet
    from pydeseq2.ds import DeseqStats
    PYDESEQ2_AVAILABLE = True
except ImportError:
    PYDESEQ2_AVAILABLE = False

from .base_tool import BaseTool
from .deg_tool_selector import choose_deg_tool, normalize_deg
from .edger_analyzer import EdgeRAnalyzerTool
from .limma_analyzer import LimmaAnalyzerTool
from ..exceptions import AnalysisError, RecoverableError, ValidationError


class DESeq2AnalyzerTool(BaseTool):
    """Unified tool for running DEG analysis with automatic tool selection (DESeq2, edgeR, or limma-voom)."""
    
    @property
    def name(self) -> str:
        return "DEGAnalyzer"
    
    @property
    def description(self) -> str:
        return "Run differential expression analysis with automatic tool selection (DESeq2/edgeR/limma-voom)"
    
    def execute(self, counts_file: Union[str, Path], metadata_file: Union[str, Path], 
                output_file: Union[str, Path], **kwargs) -> Dict:
        """
        Run DEG analysis with automatic tool selection.
        
        Args:
            counts_file: Path to counts file
            metadata_file: Path to metadata file
            output_file: Path to output file
            **kwargs: Additional parameters (harmonized: bool for Rule A)
            
        Returns:
            Analysis results
            
        Raises:
            AnalysisError: If analysis fails
        """
        counts_file = Path(counts_file)
        metadata_file = Path(metadata_file)
        output_file = Path(output_file)
        
        # Load and validate data
        counts_df, metadata_df = self._load_and_validate_data(counts_file, metadata_file)
        
        # Determine which tool to use
        harmonized = kwargs.get('harmonized', False)
        selected_tool = choose_deg_tool(counts_df, metadata_df, harmonized=harmonized)
        
        self.logger.info(f"🔍 Selected tool: {selected_tool}")
        
        # Run the selected tool
        if selected_tool == "deseq2":
            if not PYDESEQ2_AVAILABLE:
                self.logger.warning("⚠️ PyDESeq2 not available, falling back to edgeR")
                selected_tool = "edger"
                try:
                    results_df = self._run_edger(counts_file, metadata_file, output_file)
                    tool_used = "edger"
                except AnalysisError as e:
                    if "Rscript not found" in str(e):
                        self.logger.error("❌ R not available. Cannot run edgeR. Please install R or PyDESeq2.")
                        raise AnalysisError("Neither PyDESeq2 nor R is available. Please install one of them.")
                    raise
            else:
                results_df = self._run_deseq2(counts_df, metadata_df)
                tool_used = "deseq2"
        elif selected_tool == "edger":
            try:
                results_df = self._run_edger(counts_file, metadata_file, output_file)
                tool_used = "edger"
            except AnalysisError as e:
                if "Rscript not found" in str(e):
                    self.logger.warning("⚠️ R not available, falling back to DESeq2")
                    if PYDESEQ2_AVAILABLE:
                        results_df = self._run_deseq2(counts_df, metadata_df)
                        tool_used = "deseq2"
                    else:
                        raise AnalysisError("R not available and PyDESeq2 not installed. Please install R or PyDESeq2.")
                else:
                    raise
        elif selected_tool == "limma-voom":
            try:
                results_df = self._run_limma(counts_file, metadata_file, output_file)
                # Determine which limma method was used from the results
                if "tool" in results_df.columns:
                    tool_used = results_df["tool"].iloc[0] if len(results_df) > 0 else "limma-voom"
                else:
                    tool_used = "limma-voom"
            except AnalysisError as e:
                if "Rscript not found" in str(e):
                    self.logger.warning("⚠️ R not available, falling back to DESeq2")
                    if PYDESEQ2_AVAILABLE:
                        results_df = self._run_deseq2(counts_df, metadata_df)
                        tool_used = "deseq2"
                    else:
                        raise AnalysisError("R not available and PyDESeq2 not installed. Please install R or PyDESeq2.")
                else:
                    raise
        else:
            raise AnalysisError(f"Unknown tool selected: {selected_tool}")
        
        # Standardize output format
        if not results_df.empty:
            results_df = normalize_deg(results_df, tool_used)
            
            # Map gene IDs to symbols if needed
            results_df = self._map_gene_symbols(results_df)
            
            # Save standardized results
            self._save_results(results_df, output_file)
            
            # Generate summary
            summary = self._generate_summary(results_df)
        else:
            summary = {
                "n_genes": 0,
                "n_significant": 0,
                "comparisons": [],
                "summary": "No results generated"
            }
        
        summary["tool_used"] = tool_used
        self.logger.info(f"✅ {tool_used.upper()} analysis completed: {summary['n_significant']} significant genes")
        
        return summary
    
    def _load_and_validate_data(self, counts_file: Path, metadata_file: Path) -> tuple:
        """
        Load and validate counts and metadata data.
        
        Args:
            counts_file: Path to counts file
            metadata_file: Path to metadata file
            
        Returns:
            Tuple of (counts_df, metadata_df)
            
        Raises:
            AnalysisError: If data loading or validation fails
        """
        try:
            # Load counts
            counts_df = self._load_counts_data(counts_file)
            
            # Load metadata
            metadata_df = self._load_metadata_data(metadata_file)
            
            # Validate and align samples
            counts_df, metadata_df = self._align_samples(counts_df, metadata_df)
            
            return counts_df, metadata_df
            
        except Exception as e:
            raise AnalysisError(f"Failed to load and validate data: {e}")
    
    def _run_edger(self, counts_file: Path, metadata_file: Path, output_file: Path) -> pd.DataFrame:
        """Run edgeR analysis using EdgeRAnalyzerTool."""
        edger_tool = EdgeRAnalyzerTool(self.config, self.logger)
        try:
            summary = edger_tool.safe_execute(
                counts_file=str(counts_file),
                metadata_file=str(metadata_file),
                output_file=str(output_file)
            )
        except Exception as e:
            raise AnalysisError(f"edgeR execution failed: {e}")
        
        # Load results
        if output_file.exists():
            return pd.read_csv(output_file)
        else:
            raise AnalysisError("edgeR execution succeeded but output file not found")
    
    def _run_limma(self, counts_file: Path, metadata_file: Path, output_file: Path) -> pd.DataFrame:
        """Run limma analysis using LimmaAnalyzerTool."""
        limma_tool = LimmaAnalyzerTool(self.config, self.logger)
        try:
            summary = limma_tool.safe_execute(
                counts_file=str(counts_file),
                metadata_file=str(metadata_file),
                output_file=str(output_file)
            )
        except Exception as e:
            raise AnalysisError(f"limma execution failed: {e}")
        
        # Load results from file
        if output_file.exists():
            return pd.read_csv(output_file)
        else:
            raise AnalysisError("limma execution succeeded but output file not found")
    
    def _load_counts_data(self, counts_file: Path) -> pd.DataFrame:
        """
        Load counts data with error handling.
        
        Args:
            counts_file: Path to counts file
            
        Returns:
            Counts DataFrame
            
        Raises:
            AnalysisError: If loading fails
        """
        try:
            sep = "," if counts_file.suffix.lower() == ".csv" else "\t"
            counts_df = pd.read_csv(counts_file, sep=sep, index_col=0)
            
            if counts_df.empty:
                raise AnalysisError("Counts file is empty")
            
            # Ensure all values are numeric and non-negative
            numeric_counts = counts_df.select_dtypes(include=[np.number])
            if numeric_counts.shape[1] != counts_df.shape[1]:
                non_numeric_cols = set(counts_df.columns) - set(numeric_counts.columns)
                self.logger.warning(f"⚠️ Removing non-numeric columns: {non_numeric_cols}")
                counts_df = numeric_counts
            
            # Check for negative values
            if (counts_df < 0).any().any():
                self.logger.warning("⚠️ Found negative values, setting to 0")
                counts_df = counts_df.clip(lower=0)
            
            # Don't force to integers here - let tool selection decide
            # Only round if values are very close to integers
            if np.allclose(counts_df.values, np.round(counts_df.values), atol=1e-6):
                counts_df = counts_df.round().astype(int)
            
            self.logger.info(f"📊 Loaded counts: {counts_df.shape[0]} genes × {counts_df.shape[1]} samples")
            return counts_df
            
        except Exception as e:
            raise AnalysisError(f"Failed to load counts data: {e}")
    
    def _load_metadata_data(self, metadata_file: Path) -> pd.DataFrame:
        """
        Load metadata with error handling.
        
        Args:
            metadata_file: Path to metadata file
            
        Returns:
            Metadata DataFrame
            
        Raises:
            AnalysisError: If loading fails
        """
        try:
            sep = "," if metadata_file.suffix.lower() == ".csv" else "\t"
            metadata_df = pd.read_csv(metadata_file, sep=sep)
            
            if metadata_df.empty:
                raise AnalysisError("Metadata file is empty")
            
            # Ensure required columns exist
            if 'sample' not in metadata_df.columns:
                if len(metadata_df.columns) >= 1:
                    metadata_df = metadata_df.rename(columns={metadata_df.columns[0]: 'sample'})
                else:
                    raise AnalysisError("Metadata must have at least one column for sample IDs")
            
            if 'condition' not in metadata_df.columns:
                if len(metadata_df.columns) >= 2:
                    metadata_df = metadata_df.rename(columns={metadata_df.columns[1]: 'condition'})
                else:
                    raise AnalysisError("Metadata must have a condition column")
            
            # Set sample as index
            metadata_df = metadata_df.set_index('sample')
            
            self.logger.info(f"📋 Loaded metadata: {len(metadata_df)} samples")
            return metadata_df
            
        except Exception as e:
            raise AnalysisError(f"Failed to load metadata: {e}")
    
    def _align_samples(self, counts_df: pd.DataFrame, metadata_df: pd.DataFrame) -> tuple:
        """
        Align samples between counts and metadata.
        
        Args:
            counts_df: Counts DataFrame
            metadata_df: Metadata DataFrame
            
        Returns:
            Tuple of aligned DataFrames
            
        Raises:
            AnalysisError: If alignment fails
            NonRecoverableError: If alignment is impossible (non-recoverable)
        """
        try:
            from ..exceptions import NonRecoverableError
            
            # Handle metadata with 'sample' column vs index
            if 'sample' in metadata_df.columns:
                metadata_samples = set(metadata_df['sample'].astype(str))
                metadata_indexed = metadata_df.set_index('sample')
            else:
                metadata_samples = set(metadata_df.index.astype(str))
                metadata_indexed = metadata_df
            
            counts_samples = set(counts_df.columns.astype(str))
            common_samples = counts_samples.intersection(metadata_samples)
            
            if not common_samples:
                raise NonRecoverableError(
                    f"No common samples found between counts ({len(counts_samples)} samples) "
                    f"and metadata ({len(metadata_samples)} samples). "
                    f"Counts samples: {list(counts_samples)[:5]}... "
                    f"Metadata samples: {list(metadata_samples)[:5]}..."
                )
            
            if len(common_samples) < 2:
                raise NonRecoverableError(
                    f"Insufficient common samples ({len(common_samples)}) for DEG analysis. "
                    f"Need at least 2 samples."
                )
            
            # Check for critical mismatch (>50% missing)
            missing_in_metadata = counts_samples - metadata_samples
            if len(missing_in_metadata) > len(counts_samples) * 0.5:
                raise NonRecoverableError(
                    f"Critical sample mismatch: {len(missing_in_metadata)}/{len(counts_samples)} "
                    f"({len(missing_in_metadata)/len(counts_samples)*100:.1f}%) samples in counts missing from metadata. "
                    f"This indicates a data alignment issue that cannot be automatically fixed."
                )
            
            # Subset to common samples
            aligned_counts = counts_df[list(common_samples)]
            aligned_metadata = metadata_indexed.loc[list(common_samples)]
            
            # Reorder to match
            sample_order = list(common_samples)
            aligned_counts = aligned_counts[sample_order]
            aligned_metadata = aligned_metadata.loc[sample_order]
            
            missing_counts = len(counts_samples) - len(common_samples)
            missing_metadata = len(metadata_samples) - len(common_samples)
            
            if missing_counts > 0:
                self.logger.warning(f"⚠️ {missing_counts} samples in counts missing from metadata")
            if missing_metadata > 0:
                self.logger.warning(f"⚠️ {missing_metadata} samples in metadata missing from counts")
            
            self.logger.info(f"✅ Aligned {len(common_samples)} common samples")
            
            return aligned_counts, aligned_metadata
            
        except NonRecoverableError:
            raise
        except Exception as e:
            raise AnalysisError(f"Failed to align samples: {e}")
    
    def _run_deseq2(self, counts_df: pd.DataFrame, metadata_df: pd.DataFrame) -> pd.DataFrame:
        """
        Run DESeq2 analysis.
        
        Args:
            counts_df: Counts DataFrame (genes × samples)
            metadata_df: Metadata DataFrame (samples × conditions)
            
        Returns:
            Results DataFrame
            
        Raises:
            AnalysisError: If analysis fails
        """
        try:
            # Transpose counts for PyDESeq2 (samples × genes)
            counts_for_deseq = counts_df.T
            
            # Get unique conditions
            conditions = metadata_df['condition'].unique()
            conditions = [str(c) for c in conditions]  # Ensure strings
            
            if len(conditions) < 2:
                raise AnalysisError(f"Need at least 2 conditions for differential analysis, found: {conditions}")
            
            # Determine reference condition
            ref_condition = self._select_reference_condition(conditions)
            
            self.logger.info(f"🧬 Running DESeq2 with {len(conditions)} conditions, reference: {ref_condition}")
            
            # Create DESeq2 dataset
            dds = DeseqDataSet(
                counts=counts_for_deseq,
                metadata=metadata_df,
                design_factors=["condition"],
                ref_level={"condition": ref_condition}
            )
            
            # Make names unique
            dds.obs_names_make_unique()
            dds.var_names_make_unique()
            
            # Run DESeq2
            dds.deseq2()
            
            # Collect results for all comparisons
            all_results = []
            
            for condition in conditions:
                if condition == ref_condition:
                    continue
                
                try:
                    self.logger.info(f"📊 Running comparison: {condition} vs {ref_condition}")
                    
                    # Get statistics
                    stat_res = DeseqStats(dds, contrast=["condition", condition, ref_condition],
                    cooks_filter=False  # keep genes even if flagged as outliers
                    )
                    stat_res.summary()
                    
                    # Get results (keep ALL rows, even if padj/log2FC are NA)
                    results_df = stat_res.results_df
                    
                    if not results_df.empty:
                        # Ensure Gene column exists (PyDESeq2 uses index for gene IDs)
                        if "Gene" not in results_df.columns:
                            results_df = results_df.reset_index()
                            # The index column should be gene IDs
                            if results_df.index.name or len(results_df.columns) > 0:
                                first_col = results_df.columns[0]
                                results_df.rename(columns={first_col: "Gene"}, inplace=True)
                        
                        results_df["Comparison"] = f"{condition}_vs_{ref_condition}"
                        all_results.append(results_df)
                    
                except Exception as e:
                    self.logger.warning(f"⚠️ Failed comparison {condition} vs {ref_condition}: {e}")
                    continue
            
            # Combine all results
            if all_results:
                combined_results = pd.concat(all_results, ignore_index=False)
                self.logger.info(f"✅ DESeq2 completed: {len(combined_results)} results")
                return combined_results
            else:
                self.logger.warning("⚠️ No successful comparisons")
                return pd.DataFrame()
                
        except Exception as e:
            error_msg = str(e)
            # Check if this is an index mismatch error (non-recoverable)
            if "Index are different" in error_msg or "index does not match" in error_msg.lower():
                from ..exceptions import NonRecoverableError
                raise NonRecoverableError(
                    f"DESeq2 analysis failed due to sample index mismatch: {error_msg}. "
                    f"This indicates a data alignment issue that cannot be automatically fixed. "
                    f"Please check that all samples in counts data have corresponding entries in metadata."
                )
            raise AnalysisError(f"DESeq2 analysis failed: {e}")
    
    def _select_reference_condition(self, conditions: List[str]) -> str:
        """
        Select reference condition for DESeq2.
        
        Args:
            conditions: List of condition names
            
        Returns:
            Reference condition name
        """
        # Preferred reference conditions
        preferred_refs = ["control", "normal", "healthy", "untreated", "wild-type", "wt"]
        
        for ref in preferred_refs:
            for condition in conditions:
                if ref.lower() in condition.lower():
                    return condition
        
        # Default to first condition alphabetically
        return sorted(conditions)[0]
    
    def _map_gene_symbols(self, results_df: pd.DataFrame) -> pd.DataFrame:
        """
        Map gene IDs to symbols using the gene mapper tool with intelligent detection.
        
        Args:
            results_df: Results DataFrame (standardized format)
            
        Returns:
            Results DataFrame with gene symbols
        """
        try:
            # Ensure Gene column exists
            if "Gene" not in results_df.columns:
                if results_df.index.name == "Gene" or results_df.index.name is None:
                    results_df = results_df.reset_index()
                    if "Gene" not in results_df.columns:
                        results_df.rename(columns={results_df.columns[0]: "Gene"}, inplace=True)
            
            # Quick check if genes are already symbols to avoid unnecessary API calls
            gene_ids = results_df["Gene"].astype(str).tolist()
            if self._genes_appear_to_be_symbols(gene_ids):
                self.logger.info("✅ Gene IDs appear to be symbols already, skipping mapping")
                if "Original_ID" not in results_df.columns:
                    results_df["Original_ID"] = results_df["Gene"]
                return results_df
            
            # Use the gene mapper tool
            from .gene_mapper import GeneMapperTool
            gene_mapper = GeneMapperTool(self.config, self.logger)
            
            mapped_df = gene_mapper.safe_execute(results_df, id_column="Gene")
            
            return mapped_df
            
        except Exception as e:
            self.logger.warning(f"⚠️ Gene symbol mapping failed: {e}")
            # Ensure Original_ID exists
            if "Original_ID" not in results_df.columns and "Gene" in results_df.columns:
                results_df["Original_ID"] = results_df["Gene"]
            return results_df
    
    def _genes_appear_to_be_symbols(self, gene_ids: list) -> bool:
        """
        Quick heuristic check to see if genes are already symbols.
        
        Args:
            gene_ids: List of gene IDs
            
        Returns:
            True if genes appear to be symbols already
        """
        import re
        
        # Sample up to 50 genes for quick analysis
        sample_size = min(50, len(gene_ids))
        sample_ids = gene_ids[:sample_size]
        
        symbol_count = 0
        for gene_id in sample_ids:
            gene_id_str = str(gene_id).strip()
            
            # Check if it looks like a gene symbol:
            # - Starts with letter
            # - Contains only letters, numbers, hyphens
            # - Reasonable length (2-20 characters)
            # - Not obviously an Ensembl ID or Entrez ID
            if (re.match(r'^[A-Za-z][A-Za-z0-9-]*$', gene_id_str) and 
                2 <= len(gene_id_str) <= 20 and
                not re.match(r'^ENS[GT]\d{11}$', gene_id_str) and
                not re.match(r'^\d+$', gene_id_str)):
                symbol_count += 1
        
        # If >70% look like symbols, assume they are
        symbol_ratio = symbol_count / sample_size if sample_size > 0 else 0
        return symbol_ratio > 0.7
    
    def _save_results(self, results_df: pd.DataFrame, output_file: Path) -> None:
        """
        Save results to file.
        
        Args:
            results_df: Results DataFrame
            output_file: Output file path
            
        Raises:
            AnalysisError: If saving fails
        """
        try:
            # Ensure output directory exists
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Ensure Gene column exists and is not the index
            if "Gene" not in results_df.columns:
                # If Gene is the index, reset it to a column
                if results_df.index.name == "Gene" or results_df.index.name is None:
                    results_df = results_df.reset_index()
                    if "Gene" not in results_df.columns and len(results_df.columns) > 0:
                        # Rename first column to Gene if it contains gene IDs
                        results_df.rename(columns={results_df.columns[0]: "Gene"}, inplace=True)
            
            # Save results without index (Gene should be a regular column)
            results_df.to_csv(output_file, index=False)
            
            self.logger.info(f"💾 Saved results to {output_file}")
            
        except Exception as e:
            raise AnalysisError(f"Failed to save results: {e}")
    
    def _generate_summary(self, results_df: pd.DataFrame) -> Dict:
        """
        Generate analysis summary.
        
        Args:
            results_df: Results DataFrame
            
        Returns:
            Summary dictionary
        """
        if results_df.empty:
            return {
                "n_genes": 0,
                "n_significant": 0,
                "comparisons": [],
                "summary": "No results generated"
            }
        
        # Count significant results
        if "padj" in results_df.columns:
            significant = results_df["padj"] < self.config.padj_threshold
            n_significant = significant.sum()
        else:
            n_significant = 0
        
        # Get comparisons
        comparisons = []
        if "Comparison" in results_df.columns:
            comparisons = sorted(results_df["Comparison"].unique())
        
        return {
            "n_genes": len(results_df),
            "n_significant": int(n_significant),
            "comparisons": comparisons,
            "padj_threshold": self.config.padj_threshold,
            "summary": f"Found {n_significant} significant genes out of {len(results_df)} total"
        }
    
    def validate_input(self, counts_file: Union[str, Path], metadata_file: Union[str, Path], 
                      output_file: Union[str, Path], **kwargs) -> None:
        """
        Validate input parameters.
        
        Args:
            counts_file: Path to counts file
            metadata_file: Path to metadata file
            output_file: Path to output file
            **kwargs: Additional parameters
            
        Raises:
            ValidationError: If validation fails
        """
        # Validate input files
        if not Path(counts_file).exists():
            raise ValidationError(f"Counts file does not exist: {counts_file}")
        
        if not Path(metadata_file).exists():
            raise ValidationError(f"Metadata file does not exist: {metadata_file}")
        
        # Validate output directory can be created
        try:
            Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            raise ValidationError(f"Cannot create output directory: {e}")
    
    def validate_output(self, result: Dict) -> Dict:
        """
        Validate output result.
        
        Args:
            result: Output result
            
        Returns:
            Validated result
            
        Raises:
            ValidationError: If validation fails
        """
        required_fields = ["n_genes", "n_significant", "comparisons", "summary"]
        
        for field in required_fields:
            if field not in result:
                raise ValidationError(f"Missing required field in result: {field}")
        
        if not isinstance(result["n_genes"], int) or result["n_genes"] < 0:
            raise ValidationError("n_genes must be a non-negative integer")
        
        if not isinstance(result["n_significant"], int) or result["n_significant"] < 0:
            raise ValidationError("n_significant must be a non-negative integer")
        
        if not isinstance(result["comparisons"], list):
            raise ValidationError("comparisons must be a list")
        
        return result 