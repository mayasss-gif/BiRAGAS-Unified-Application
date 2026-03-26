"""
Metadata extractor tool for DEG Pipeline Agent.
"""

import gzip
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple

try:
    import GEOparse
    GEO_AVAILABLE = True
except ImportError:
    GEO_AVAILABLE = False

from .base_tool import BaseTool
from .metadata_llm_extractor import parse_geo_and_create_metadata
from .metadata_llm_extractor_enhanced import parse_geo_and_create_metadata_enhanced
from ..exceptions import MetadataError, RecoverableError, ValidationError
from ..helpers.metadata_extractor import analyze_file
from .preprocess_counts import process_geo_file

class MetadataExtractorTool(BaseTool):
    """Tool for extracting metadata from various sources."""
    
    @property
    def name(self) -> str:
        return "MetadataExtractor"
    
    @property
    def description(self) -> str:
        return "Extract metadata from GEO series matrix files and patient metadata"
    
    def execute(self, counts_file: Union[str, Path], metadata_file: Union[str, Path], 
                output_dir: Union[str, Path], **kwargs) -> Dict:
        """
        Extract metadata and create standardized output files.
        
        Args:
            counts_file: Path to counts file
            metadata_file: Path to metadata file
            output_dir: Output directory for processed files
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with extraction results
            
        Raises:
            MetadataError: If metadata extraction fails
        """
        counts_file = Path(counts_file)
        metadata_file = Path(metadata_file)
        output_dir = Path(output_dir)
        counts_df = None
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Load counts to get sample IDs and clean the data using new simplified API
            counts_df, sample_columns, gene_col = process_geo_file(counts_file)

            if counts_df is not None:
                sample_ids = sample_columns
                gene_col = gene_col
                print(f"\n\n++++ Cleaned counts: Columns: {counts_df.columns}")
            else:
                raise ValueError(f"Processing failed")
        except Exception as processing_error:
            print(f"Simplified processing failed: {processing_error}, falling back to legacy method")
            try:
                analyze_result = analyze_file(counts_file)

                if analyze_result["status"] == "success":
                    sample_ids = analyze_result["numeric_cols"]
                    counts_df = analyze_result["dataframe"]
                else:
                    sample_ids = self._load_sample_ids(counts_file)
                    counts_df = None
            except Exception as fallback_error:
                print(f"Legacy fallback also failed: {fallback_error}")
                try:
                    sample_ids = self._load_sample_ids(counts_file)
                    counts_df = None
                except Exception as final_error:
                    raise MetadataError(f"Failed to analyze counts file: {final_error}")
        
        # Determine extraction method based on file type
        if self._is_geo_series_matrix(metadata_file):
            result = self._extract_geo_metadata(counts_file, metadata_file, output_dir, sample_ids, counts_df=counts_df)
        else:
            result = self._extract_patient_metadata(counts_file, metadata_file, output_dir, sample_ids, counts_df=counts_df)
        
        return result
    
    def _load_sample_ids(self, counts_file: Path) -> List[str]:
        """
        Load sample IDs from counts file.
        Robustly excludes the first column regardless of its name.
        
        Args:
            counts_file: Path to counts file
            
        Returns:
            List of sample IDs
            
        Raises:
            MetadataError: If sample IDs cannot be loaded
        """
        try:
            # Determine separator
            sep = "," if counts_file.suffix.lower() == ".csv" else "\t"
            
            # Read only the header without setting index_col
            df_header = pd.read_csv(counts_file, sep=sep, nrows=0)
            all_columns = list(df_header.columns)
            
            if len(all_columns) < 2:
                raise MetadataError("Counts file must have at least 2 columns (gene column + sample columns)")
            
            # Always exclude the first column (gene identifiers) regardless of its name
            first_col_name = all_columns[0]
            sample_ids = all_columns[1:]  # Everything except the first column
            
            # Validate we have sample IDs
            if not sample_ids:
                raise MetadataError("No sample IDs found in counts file")
            
            # Log for debugging
            self.logger.info(f"🧬 First column '{first_col_name}' excluded (gene identifiers)")
            self.logger.info(f"✅ Loaded {len(sample_ids)} sample IDs: {sample_ids}")
            
            return sample_ids
            
        except Exception as e:
            raise MetadataError(f"Failed to load sample IDs from counts file: {e}")
    
    def _is_geo_series_matrix(self, metadata_file: Path) -> bool:
        """
        Check if file is a GEO series matrix file.
        
        Args:
            metadata_file: Path to metadata file
            
        Returns:
            True if GEO series matrix file
        """
        return "series_matrix" in metadata_file.name.lower()
    
    def _extract_geo_metadata(self, counts_file: Path, series_matrix_file: Path, 
                             output_dir: Path, sample_ids: List[str], counts_df: pd.DataFrame = None) -> Dict:
        """
        Extract metadata from GEO series matrix file.
        
        Args:
            counts_file: Path to counts file
            series_matrix_file: Path to series matrix file
            output_dir: Output directory
            sample_ids: List of sample IDs
            counts_df: Counts DataFrame
            
        Returns:
            Extraction results
            
        Raises:
            MetadataError: If extraction fails
        """
        log = []
        
        # Validate input
        if series_matrix_file.suffix.lower() in ['.fastq', '.fq', '.fastq.gz', '.fq.gz']:
            raise MetadataError("Invalid input: sequencing file, not metadata")
        
        # Try GEOparse first
        if GEO_AVAILABLE:
            try:
                metadata_df = self._extract_with_geoparse(series_matrix_file, sample_ids, log)
                print(f"✅ Metadata From GEOparse DataFrame: {metadata_df}")
            except Exception as e:
                log.append(f"⚠️ GEOparse failed: {e}")
                metadata_df = None
        else:
            log.append("⚠️ GEOparse not available, using manual extraction")
            metadata_df = None
        
        # Fallback to manual extraction if GEOparse failed
        if metadata_df is None:
            metadata_df = self._extract_manually(series_matrix_file, sample_ids, log)
            print(f"✅ Metadata From Manual Extraction DataFrame: {metadata_df}")
        if metadata_df is None:
            raise MetadataError("Failed to extract metadata using all available methods")
        
        # Find best condition column
        # condition_col, condition_vals = self._find_best_condition_column(metadata_df, sample_ids, log)
        
        # if condition_col is None:
        #     raise MetadataError("No suitable condition column found in metadata")
        
        # # Create initial metadata DataFrame
        # initial_metadata = pd.DataFrame({
        #     "sample": sample_ids,
        #     "condition": condition_vals
        # })

        # print(f"✅ Initial Metadata DataFrame: {initial_metadata}")
        
        # Filter to top 2 most frequent conditions if needed
        # final_metadata = self._filter_to_top2_conditions(initial_metadata, log)
        final_metadata = metadata_df

        print(f"✅ Final Metadata DataFrame: {final_metadata}")
        
        # Save output files
        print(f"Passing counts_df to save_processed_files: {counts_df.head(3)}")

        self._save_processed_files(counts_file, final_metadata, output_dir, counts_df=counts_df)
        
        # Save extraction log with top 2 conditions info
        return {
            "status": "success",
            "method": "geo_extraction",
            "metadata_df": final_metadata,
            "log": log
        }
        
    
    def _extract_with_geoparse(self, series_matrix_file: Path, sample_ids: List[str], 
                              log: List[str]) -> Optional[pd.DataFrame]:
        """
        Extract metadata using GEOparse.
        
        Args:
            series_matrix_file: Path to series matrix file
            sample_ids: List of sample IDs
            log: Log list to append to
            
        Returns:
            Metadata DataFrame or None if failed
        """
        try:
            gse = GEOparse.get_GEO(filepath=str(series_matrix_file), silent=True)
            metadata_df = gse.phenotype_data.copy()
            
            if metadata_df.empty:
                raise ValueError("Empty phenotype data")
            
            log.append(f"✅ GEOparse successful: {metadata_df.shape[0]}×{metadata_df.shape[1]}")
            
            # Align with sample IDs
            metadata_df = self._align_metadata_with_samples(metadata_df, sample_ids, log)
            
            return metadata_df
            
        except Exception as e:
            log.append(f"❌ GEOparse failed: {e}")
            return None
    
    def _extract_manually(self, series_matrix_file: Path, sample_ids: List[str], 
                         log: List[str]) -> Optional[pd.DataFrame]:
        """
        Extract metadata manually from series matrix file.
        
        Args:
            series_matrix_file: Path to series matrix file
            sample_ids: List of sample IDs
            log: Log list to append to
            
        Returns:
            Metadata DataFrame or None if failed
        """
        try:
            characteristics_data = self._parse_characteristics_manually(series_matrix_file, sample_ids, log)
            
            if characteristics_data is None:
                return None
            
            log.append("✅ Manual extraction successful")
            return characteristics_data
            
        except Exception as e:
            log.append(f"❌ Manual extraction failed: {e}")
            return None
    
    def _parse_characteristics_manually(self, series_matrix_file: Path, sample_ids: List[str], 
                                       log: List[str]) -> Optional[pd.DataFrame]:
        """
        Parse characteristics lines manually from series matrix file.
        Enhanced version with improved parsing logic.
        
        Args:
            series_matrix_file: Path to series matrix file
            sample_ids: List of sample IDs
            log: Log list to append to
            
        Returns:
            Metadata DataFrame or None if failed
        """
        raw_rows = []
        headers = []
        
        # Handle both gzipped and plain text files
        opener = gzip.open if series_matrix_file.name.endswith(".gz") else open
        
        geo_lines = []
        try:
            with opener(series_matrix_file, "rt") as f:
                check_multiple_groups = False
                for line in f:
                    if any(line.startswith(prefix) for prefix in ("!Series_summary", "!Series_overall_design", "!Sample_characteristics_ch1")):
                        if line.startswith("!Sample_characteristics_ch1"):
                            if self._has_multiple_groups(line):
                                geo_lines.append(line.strip())
                                check_multiple_groups = True
                            else:
                                print(f"❌ Found !Sample_characteristics_ch1 line with less than 2 groups.")
                        else:
                            geo_lines.append(line.strip())
            if not geo_lines or not check_multiple_groups:
                log.append("❌ No !Sample_characteristics_ch1 line with multiple groups found.")
                return None
            # Join relevant GEO lines into a single text block for LLM parsing
            geo_text = '''\n'''.join(geo_lines).strip()

            # print(f"✅ GEO Text: {geo_text}")
            
            print(f"✅ calling LLM extractor ...")
            # metadata_df = parse_geo_and_create_metadata(geo_text, sample_ids, disease="pancreatic cancer", model="gpt-4.1-mini-2025-04-14")

            # Test the enhanced parsing function
            metadata_df = parse_geo_and_create_metadata_enhanced(
                geo_text=geo_text,
                sample_ids=sample_ids,
                disease="pancreatic cancer",
                model="gpt-4.1-mini-2025-04-14"
            )
        
            print("\n📊 Generated Metadata DataFrame:")
            print(metadata_df)
            print(f"\nDataFrame shape: {metadata_df.shape}")
            print(f"Columns: {list(metadata_df.columns)}")

            print(f"✅ Metadata From LLM Extraction DataFrame: {metadata_df}")
            
            return metadata_df
            
        except Exception as e:
            log.append(f"❌ Manual parsing failed: {e}")
            return None

    def _has_multiple_groups(self, line: str) -> bool:
        """
        Check if a !Sample_characteristics_ch1 line has two or more unique values (excluding the prefix).
        """
        parts = line.strip().split("\t")[1:]  # Exclude the "!Sample_characteristics_ch1" prefix
        cleaned = [p.strip().strip('"') for p in parts if p.strip()]
        return len(set(cleaned)) >= 2
    
    def _extract_patient_metadata(self, counts_file: Path, metadata_file: Path, 
                                 output_dir: Path, sample_ids: List[str], counts_df: pd.DataFrame = None) -> Dict:
        """
        Extract metadata from patient metadata file.
        
        Args:
            counts_file: Path to counts file
            metadata_file: Path to metadata file
            output_dir: Output directory
            sample_ids: List of sample IDs
            
        Returns:
            Extraction results
        """
        try:
            # Load patient metadata
            sep = "," if metadata_file.suffix.lower() == ".csv" else "\t"
            metadata_df = pd.read_csv(metadata_file, sep=sep)
            
            # Standardize column names
            print(f"✅ Metadata DataFrame Columns: {metadata_df.columns} Checking for condition column")
            if "condition" not in metadata_df.columns:
                if len(metadata_df.columns) >= 2:
                    metadata_df.columns = ["sample", "condition"]
                else:
                    raise MetadataError("Patient metadata must have at least 2 columns")
            
            # Save processed files
            self._save_processed_files(counts_file, metadata_df, output_dir, counts_df=counts_df)
            
            # Create log
            log = ["Patient metadata processed successfully"]
            conditions = metadata_df["condition"].unique().tolist()
            # self._save_extraction_log(counts_file, log, "condition", output_dir, conditions)
            
            return {
                "status": "success",
                "method": "patient_metadata",
                "condition_column": "condition",
                "n_samples": len(metadata_df),
                "n_conditions": metadata_df["condition"].nunique(),
                "conditions": metadata_df["condition"].unique().tolist(),
                "log": log
            }
            
        except Exception as e:
            raise MetadataError(f"Failed to extract patient metadata: {e}")
    
    def _align_metadata_with_samples(self, metadata_df: pd.DataFrame, sample_ids: List[str], 
                                   log: List[str]) -> pd.DataFrame:
        """
        Align metadata with sample IDs.
        
        Args:
            metadata_df: Metadata DataFrame
            sample_ids: List of sample IDs
            log: Log list to append to
            
        Returns:
            Aligned metadata DataFrame
        """
        if set(sample_ids).issubset(metadata_df.index):
            # Perfect match
            aligned_df = metadata_df.loc[sample_ids]
            log.append("✅ Perfect sample ID match")
        else:
            # Positional alignment
            aligned_df = metadata_df.iloc[:len(sample_ids)]
            log.append("⚠️ Using positional alignment")
        
        return aligned_df
    
    def _find_best_condition_column(self, metadata_df: pd.DataFrame, sample_ids: List[str], 
                                   log: List[str]) -> Tuple[Optional[str], Optional[List[str]]]:
        """
        Find the best condition column in metadata with enhanced scoring.
        Scores each metadata column and picks the top scorer with >=2 unique values.
        
        Args:
            metadata_df: Metadata DataFrame
            sample_ids: List of sample IDs
            log: Log list to append to
            
        Returns:
            Tuple of (column_name, values) or (None, None) if not found
        """
        # Enhanced priority keywords
        priority_keywords = (
            "genotype", "tissue", "disease", "treatment", "cell type", "phenotype", 
            "disease state", "type", "etiology", "bclc stage", "sex", "condition"
        )
        
        scored_columns = []
        total_samples = len(sample_ids)
        
        for col in metadata_df.columns:
            if col == "sample":
                continue
            
            # Use shared utility methods for value extraction and analysis
            raw_values = metadata_df[col].fillna("").astype(str).tolist()
            cleaned_values, _ = self._extract_group_values(raw_values)
            stats = self._analyze_group_statistics(cleaned_values, min_samples_per_group=1)
            
            n_unique = stats["n_groups"]
            unique_values = stats["unique_groups"]
            
            # Skip if wrong number of samples
            if abs(len(cleaned_values) - total_samples) > 1:
                continue
            
            # Enhanced scoring algorithm
            keyword_score = sum(1 for k in priority_keywords if k.lower() in col.lower())
            value_score = sum(1 for v in cleaned_values for k in priority_keywords if k.lower() in v.lower())
            group_score = 5 if n_unique == 2 else (3 if n_unique > 2 else -n_unique)
            
            total_score = keyword_score * 3 + value_score + group_score
            
            scored_columns.append((total_score, col, cleaned_values, n_unique))
            unique_display = unique_values[:5] + ['...'] if n_unique > 5 else unique_values
            log.append(f"🧪 Scored '{col}': {total_score}, groups={unique_display}")
        
        if not scored_columns:
            return None, None
        
        # Sort by score (highest first)
        scored_columns.sort(key=lambda x: x[0], reverse=True)
        best_score, best_col, best_values, n_unique = scored_columns[0]
        
        if n_unique > 2:
            log.append(f"⚠️ Column '{best_col}' has {n_unique} levels; will pick top 2")
        else:
            log.append(f"✅ Column '{best_col}' has exactly 2 levels")
        
        return best_col, best_values
    
    def _extract_group_values(self, values: List[str]) -> Tuple[List[str], str]:
        """
        Extract group values and header from a list of values.
        
        Args:
            values: List of raw values (may contain "key: value" format)
            
        Returns:
            Tuple of (extracted_groups, header_name)
        """
        groups = []
        header = "unknown"
        
        for value in values:
            if ":" in value:
                # Extract header from first value if not set
                if header == "unknown":
                    header = value.split(":", 1)[0].strip().lower()
                
                # Extract group value (part after ":")
                group_value = value.split(":", 1)[1].strip()
                groups.append(group_value)
            else:
                # Values without ":" are treated as-is
                groups.append(value.strip())
        
        return groups, header
    
    def _get_top2_frequencies(self, items: Union[List[str], pd.Series], min_samples_per_group: int = 2) -> Dict:
        """
        Get top 2 most frequent items with comprehensive statistics.
        
        Args:
            items: List of items or pandas Series to analyze
            min_samples_per_group: Minimum samples required per group
            
        Returns:
            Dictionary with frequency analysis and top 2 selection
        """
        from collections import Counter
        
        # Handle both list and pandas Series input
        if isinstance(items, pd.Series):
            freq_counts = items.value_counts()
            item_counts = dict(freq_counts)
        else:
            item_counts = Counter(items)
            freq_counts = pd.Series(item_counts)
        
        unique_items = list(item_counts.keys())
        n_unique = len(unique_items)
        
        # Get top 2 most frequent items
        if isinstance(items, pd.Series):
            top2_items = freq_counts.nlargest(2).index.tolist()
        else:
            top2_items = [item[0] for item in Counter(items).most_common(2)]
        
        top2_counts = {item: item_counts[item] for item in top2_items if item in item_counts}
        
        # Validate sample sufficiency
        has_multiple_groups = n_unique >= 2
        sufficient_samples = all(count >= min_samples_per_group for count in item_counts.values()) if has_multiple_groups else False
        
        return {
            "n_unique": n_unique,
            "unique_items": unique_items,
            "item_counts": item_counts,
            "top2_items": top2_items,
            "top2_counts": top2_counts,
            "has_multiple_groups": has_multiple_groups,
            "sufficient_samples": sufficient_samples
        }
    
    def _analyze_group_statistics(self, groups: List[str], min_samples_per_group: int = 1) -> Dict:
        """
        Analyze group statistics for DEG suitability.
        
        Args:
            groups: List of group values
            min_samples_per_group: Minimum samples required per group
            
        Returns:
            Dictionary with group statistics
        """
        # Use shared frequency analysis utility
        freq_stats = self._get_top2_frequencies(groups, min_samples_per_group)
        
        return {
            "n_groups": freq_stats["n_unique"],
            "unique_groups": freq_stats["unique_items"],
            "group_counts": freq_stats["item_counts"],
            "has_multiple_groups": freq_stats["has_multiple_groups"],
            "sufficient_samples": freq_stats["sufficient_samples"],
            "top2_groups": freq_stats["top2_items"],
            "top2_counts": freq_stats["top2_counts"],
            "deg_suitable": freq_stats["has_multiple_groups"] and freq_stats["sufficient_samples"]
        }

    def _analyze_groups_in_line(self, values: List[str], line_index: int, min_samples_per_group: int = 1) -> Dict:
        """
        Analyze if a line contains multiple groups suitable for DEG analysis.
        
        Args:
            values: List of values from the characteristics line
            line_index: Index of the line for identification
            min_samples_per_group: Minimum samples required per group
            
        Returns:
            Dictionary with group analysis results
        """
        # Use shared utility methods
        groups, header = self._extract_group_values(values)
        stats = self._analyze_group_statistics(groups, min_samples_per_group)
        
        # Add line-specific information
        return {
            **stats,
            "header": header,
            "line_index": line_index
        }
    
    def _filter_to_top2_conditions(self, metadata_df: pd.DataFrame, log: List[str]) -> pd.DataFrame:
        """
        Filter metadata to top 2 most frequent conditions.
        
        Args:
            metadata_df: Metadata DataFrame with sample and condition columns
            log: Log list to append to
            
        Returns:
            Filtered metadata DataFrame
        """
        # Use shared frequency analysis utility
        freq_stats = self._get_top2_frequencies(metadata_df["condition"], min_samples_per_group=2)
        
        n_unique = freq_stats["n_unique"]
        top2_conditions = freq_stats["top2_items"]
        
        if n_unique <= 2:
            log.append(f"✅ Using all {n_unique} conditions: {freq_stats['unique_items']}")
            return metadata_df
        
        log.append(f"✅ Using top 2 conditions from {n_unique}: {top2_conditions}")
        
        # Filter metadata to only include top 2 conditions
        filtered_metadata = metadata_df[metadata_df["condition"].isin(top2_conditions)].copy()
        
        # Validate we have enough samples after filtering
        if len(filtered_metadata) < 4:  # Need at least 2 samples per condition
            log.append(f"⚠️ Warning: Only {len(filtered_metadata)} samples after filtering")
        
        return filtered_metadata
    

    
    def _save_processed_files(
        self,
        counts_file: Path,
        metadata_df: pd.DataFrame,
        output_dir: Path,
        counts_df: pd.DataFrame = None
    ) -> None:
        """
        Save processed counts and metadata files, filtering counts to match metadata samples.
        Dynamically uses the first column of metadata as the sample identifier.
        """
        print(f"Saving processed files to {output_dir}")

        # Load counts if not passed
        if counts_df is None:
            counts_df = pd.read_csv(counts_file, sep=None, engine="python", index_col=0)
        else:
            print(f"Counts DataFrame already provided: {counts_df.head(3)}")
            
        # Always ensure Gene column is set as index
        # If Gene column exists in columns, set it as index
        if 'Gene' in counts_df.columns:
            counts_df = counts_df.set_index('Gene')
        # If index is numeric range, assume first column contains gene names
        elif counts_df.index.equals(pd.RangeIndex(len(counts_df))):
            counts_df = counts_df.set_index(counts_df.columns[0])
        # If index already exists but not named 'Gene', rename it
        elif counts_df.index.name != 'Gene':
            counts_df.index.name = 'Gene'
        
        # Get sample column from metadata (typically 'sample' or similar)
        sample_column = metadata_df.columns[0]

        # Match metadata samples to counts columns
        aligned_samples = [s for s in metadata_df[sample_column] if s in counts_df.columns]

        if len(aligned_samples) < 2:
            raise MetadataError(
                f"Too few aligned samples after metadata filtering. "
                f"Using '{sample_column}' column for sample matching."
            )

        # Clean Ensembl gene IDs (remove version numbers)
        if counts_df.index.str.startswith('ENS').any():
            counts_df.index = counts_df.index.str.split('.').str[0]
            print("✅ Cleaned Ensembl gene IDs (removed version numbers)")

        # Filter and save counts
        filtered_counts = counts_df[aligned_samples]
        print(f"\n\n++++ Filtered counts: \n\n {filtered_counts.head(3)}")

        counts_output = output_dir / "prep_counts.csv"
        filtered_counts.to_csv(counts_output, index=True)

        # Filter and save metadata
        aligned_metadata = metadata_df[metadata_df[sample_column].isin(aligned_samples)]
        metadata_output = output_dir / "prep_meta.csv"
        aligned_metadata.to_csv(metadata_output, index=False)

        self.logger.info(
            f"✅ Saved processed files to {output_dir} ({len(aligned_samples)} samples) "
            f"using '{sample_column}' as sample identifier"
        )

    
    # def _save_extraction_log(self, counts_file: Path, log: List[str], condition_col: str, 
    #                         output_dir: Path, final_conditions: Optional[List[str]] = None) -> None:
    #     """
    #     Save extraction log as JSON file with enhanced information.
        
    #     Args:
    #         counts_file: Original counts file
    #         log: Log entries
    #         condition_col: Selected condition column
    #         output_dir: Output directory
    #         final_conditions: Final conditions used after filtering
    #     """
    #     base_name = counts_file.stem
    #     log_data = {
    #         "log": log,
    #         "condition_column": condition_col,
    #         "extraction_method": self.name,
    #         "final_conditions": final_conditions or []
    #     }
        
    #     log_file = output_dir / f"{base_name}_meta_log.json"
    #     with open(log_file, 'w') as f:
    #         json.dump(log_data, f, indent=2)
        
    #     self.logger.info(f"📝 Metadata extraction log saved: {log_file}")
    
    def validate_input(self, counts_file: Union[str, Path], metadata_file: Union[str, Path], 
                      output_dir: Union[str, Path], **kwargs) -> None:
        """
        Validate input parameters.
        
        Args:
            counts_file: Path to counts file
            metadata_file: Path to metadata file
            output_dir: Output directory
            **kwargs: Additional parameters
            
        Raises:
            ValidationError: If validation fails
        """
        # Validate counts file
        if not counts_file or not Path(counts_file).exists():
            raise ValidationError(f"Counts file does not exist: {counts_file}")
        
        # Validate metadata file
        if not metadata_file or not Path(metadata_file).exists():
            raise ValidationError(f"Metadata file does not exist: {metadata_file}")
        
        # Validate output directory can be created
        try:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
        except Exception as e:
            raise ValidationError(f"Cannot create output directory: {e}")
    
    # def validate_output(self, result: Dict) -> Dict:
    #     """
    #     Validate output result.
        
    #     Args:
    #         result: Output result
            
    #     Returns:
    #         Validated result
            
    #     Raises:
    #         ValidationError: If validation fails
    #     """
    #     required_fields = ["status", "method", "condition_column", "n_samples", "n_conditions", "conditions"]
        
    #     for field in required_fields:
    #         if field not in result:
    #             raise ValidationError(f"Missing required field in result: {field}")
        
    #     if result["status"] != "success":
    #         raise ValidationError(f"Extraction failed: {result.get('error', 'Unknown error')}")
        
    #     if result["n_samples"] < 2:
    #         raise ValidationError("Need at least 2 samples for analysis")
        
    #     if result["n_conditions"] < 2:
    #         raise ValidationError("Need at least 2 conditions for differential analysis")
        
    #     return result 