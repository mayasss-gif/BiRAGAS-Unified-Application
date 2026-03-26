from dataclasses import dataclass
from datetime import datetime
import re
import time
from typing import Optional
import pandas as pd
import numpy as np
from pathlib import Path
from agents import function_tool, RunContextWrapper, Agent
from ..helper_tools.file_utils import resolve_and_validate_file
from ..helper_tools.data_utils import validate_numeric_column, drop_na_in_columns
from ..configuration import FIXED_THRESHOLDS, DEFAULT_THRESHOLD, LARGE_FILE_ROW_WARNING

from ..helpers import logger


@dataclass
class FilterContext:
    df: Optional[pd.DataFrame] = None
    degs: Optional[pd.DataFrame] = None
    current_file: str = ""
    threshold: Optional[float] = None
    current_folder: str = ""
    causal: bool = False
    # Set by process_single_filter so tools recover from LLM path typos / hallucinations.
    canonical_input_path: str = ""
    canonical_output_dir: str = ""


def _resolve_filter_input_path(
    ctx: RunContextWrapper[FilterContext],
    file_path: str,
    allowed: set,
) -> Path:
    """
    Resolve the DEG file path: prefer LLM-provided path, fall back to canonical,
    then a single *_DEGs.csv in the same directory (self-heal UUID transcription errors).
    """
    canonical = (getattr(ctx.context, "canonical_input_path", None) or "").strip()
    raw = (file_path or "").strip()
    if not raw and canonical:
        raw = canonical

    def _try(candidate: str) -> Optional[Path]:
        if not candidate:
            return None
        try:
            return resolve_and_validate_file(candidate, allowed_extensions=allowed)
        except (FileNotFoundError, ValueError, OSError, TypeError) as e:
            logger.debug("resolve_filter_input: failed for %r: %s", candidate, e)
            return None

    resolved = _try(raw)
    if resolved is None and canonical:
        if raw and raw.strip() != canonical:
            logger.warning(
                "LLM path failed; using canonical_input_path (LLM snippet): %s",
                raw[:200],
            )
        resolved = _try(canonical)
    if resolved is not None:
        return resolved

    if canonical:
        cp = Path(canonical).expanduser()
        parent = cp.parent
        if parent.is_dir():
            matches = sorted(parent.glob("*_DEGs.csv"))
            if len(matches) == 1:
                logger.warning(
                    "Recovered DEG input via lone *_DEGs.csv in %s: %s",
                    parent,
                    matches[0].name,
                )
                return resolve_and_validate_file(matches[0], allowed_extensions=allowed)
            want_name = cp.name
            for m in matches:
                if m.name == want_name:
                    logger.warning("Recovered DEG input by exact filename match: %s", m)
                    return resolve_and_validate_file(m, allowed_extensions=allowed)
            if matches:
                logger.warning(
                    "Multiple *_DEGs.csv in %s; using first: %s",
                    parent,
                    matches[0].name,
                )
                return resolve_and_validate_file(matches[0], allowed_extensions=allowed)

    raise FileNotFoundError(
        f"Could not resolve DEG input file (tried LLM path and canonical). "
        f"raw={raw!r} canonical={canonical!r}"
    )


@function_tool
def load_expression_file(
    ctx: RunContextWrapper[FilterContext], file_path: str
) -> str:
    """
    Load a raw expression file (CSV/TSV/XLS/XLSX) into a DataFrame.
    Enhanced with improved error handling, validation, and performance optimizations.
    """
    start_time = time.time()
    
    try:
        if not isinstance(file_path, str):
            return "Error: Invalid file path provided"
        if not (file_path or "").strip() and not (
            getattr(ctx.context, "canonical_input_path", None) or ""
        ).strip():
            return "Error: Invalid file path provided"

        logger.debug("load_expression_file received path: %s", file_path)

        allowed = {".csv", ".tsv", ".xls", ".xlsx"}
        path_obj = _resolve_filter_input_path(ctx, file_path, allowed)
        
        # Log the resolved path
        logger.debug(f"Resolved path: {path_obj}")
        #ext = path_obj.suffix.lower()
        # remember both filename *and* its parent folder
        ctx.context.current_file = path_obj.name
        # Uppercase folder segment matches get_files_to_process / process_single_filter convention
        ctx.context.current_folder = path_obj.parent.name.upper()
        ext = path_obj.suffix.lower()
        
        # Check file size before loading
        file_size_mb = path_obj.stat().st_size / (1024 * 1024)
        if file_size_mb > 500:  # 500MB threshold
            logger.warning("Loading very large file (%.1f MB): %s", file_size_mb, path_obj.name)
        
        logger.info("Loading file: %s (%.1f MB)", path_obj.name, file_size_mb)
        
        # Enhanced file loading with optimized parameters
        df = None
        load_kwargs = {"low_memory": False}
        
        if ext in {".xls", ".xlsx"}:
            # Enhanced Excel loading with better error handling
            try:
                with pd.ExcelFile(path_obj) as xls:
                    # Check if multiple sheets exist and log info
                    sheet_names = xls.sheet_names
                    if len(sheet_names) > 1:
                        logger.info("Excel file has %d sheets. Loading first sheet: '%s'", 
                                  len(sheet_names), sheet_names[0])
                    
                    df = pd.read_excel(xls, engine="openpyxl", na_values=['', 'NA', 'N/A', 'null', 'NULL'])
            except Exception as excel_exc:
                # Fallback to different engines if openpyxl fails
                logger.warning("openpyxl failed, trying xlrd engine: %s", excel_exc)
                try:
                    df = pd.read_excel(path_obj, engine="xlrd", na_values=['', 'NA', 'N/A', 'null', 'NULL'])
                except Exception:
                    raise excel_exc  # Re-raise original exception
                    
        elif ext == ".tsv":
            # Enhanced TSV loading with encoding detection
            load_kwargs.update({
                "sep": "\t",
                "na_values": ['', 'NA', 'N/A', 'null', 'NULL'],
                "encoding": "utf-8"
            })
            try:
                df = pd.read_csv(path_obj, **load_kwargs)
            except UnicodeDecodeError:
                logger.warning("UTF-8 encoding failed, trying latin-1 for: %s", path_obj.name)
                load_kwargs["encoding"] = "latin-1"
                df = pd.read_csv(path_obj, **load_kwargs)
                
        else:  # CSV
            # Enhanced CSV loading with encoding detection and delimiter inference
            load_kwargs.update({
                "na_values": ['', 'NA', 'N/A', 'null', 'NULL'],
                "encoding": "utf-8"
            })
            try:
                df = pd.read_csv(path_obj, **load_kwargs)
            except UnicodeDecodeError:
                logger.warning("UTF-8 encoding failed, trying latin-1 for: %s", path_obj.name)
                load_kwargs["encoding"] = "latin-1"
                df = pd.read_csv(path_obj, **load_kwargs)
            except pd.errors.ParserError as parse_err:
                # Try with different separator if default fails
                if "," in str(parse_err) or "delimiter" in str(parse_err).lower():
                    logger.warning("Parser error with comma delimiter, trying semicolon: %s", parse_err)
                    load_kwargs["sep"] = ";"
                    df = pd.read_csv(path_obj, **load_kwargs)
                else:
                    raise
        
        # Validate loaded DataFrame
        if df is None:
            return f"Error: Failed to load data from {file_path}"
        
        if df.empty:
            logger.warning("Loaded empty DataFrame from: %s", path_obj.name)
            return f"Warning: Loaded empty file {path_obj.name}"
        
        rows, cols = df.shape
        load_time = time.time() - start_time
        
        # Enhanced logging with more details
        memory_usage_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
        logger.info("Successfully loaded %s: %d rows, %d columns, %.1f MB memory, %.2fs", 
                   path_obj.name, rows, cols, memory_usage_mb, load_time)
        
        # Check for potential issues in the data
        null_counts = df.isnull().sum().sum()
        if null_counts > 0:
            null_percentage = (null_counts / (rows * cols)) * 100
            logger.info("Data contains %d null values (%.1f%% of total)", null_counts, null_percentage)
        
        # Check for duplicate rows
        duplicate_rows = df.duplicated().sum()
        if duplicate_rows > 0:
            logger.warning("Data contains %d duplicate rows", duplicate_rows)
        
        # Large file warning with enhanced threshold logic
        if rows > LARGE_FILE_ROW_WARNING:
            logger.warning("Loaded large file (%d rows): %s. Consider data filtering for better performance.", 
                         rows, path_obj.name)
        
        # Update context
        #ctx.context.df = df   removed
        #ctx.context.current_file = path_obj.name   removed

        ctx.context.df = df
        
        # Enhanced return message with more information
        return (f"Loaded {path_obj.name} ({rows:,} rows, {cols} columns) "
                f"in {load_time:.2f}s. Memory usage: {memory_usage_mb:.1f}MB")
        
    except FileNotFoundError:
        error_msg = f"Error: File not found: {file_path}"
        logger.error(error_msg)
        return error_msg
    except PermissionError:
        error_msg = f"Error: Permission denied accessing: {file_path}"
        logger.error(error_msg)
        return error_msg
    except pd.errors.EmptyDataError:
        error_msg = f"Error: File is empty or contains no data: {file_path}"
        logger.error(error_msg)
        return error_msg
    except pd.errors.ParserError as parse_err:
        error_msg = f"Error: Failed to parse file {file_path}: {parse_err}"
        logger.error(error_msg)
        return error_msg
    except MemoryError:
        error_msg = f"Error: Insufficient memory to load file: {file_path}"
        logger.error(error_msg)
        return error_msg
    except Exception as exc:
        logger.exception("Unexpected error loading '%s': %s", file_path, exc)
        return f"Error loading {file_path}: {exc}"


@function_tool
def dynamic_filter(ctx: RunContextWrapper[FilterContext]) -> str:
    """
    Apply piecewise |log2FC| thresholding and drop invalid rows.
    Enhanced with better validation, statistics, and robust filtering logic.
    
    """
    start_time = time.time()
    df = ctx.context.df
    current = ctx.context.current_file
    
    
    # Enhanced input validation
    if df is None:
        logger.warning("dynamic_filter called but no data loaded")
        return "No data loaded."
    
    if df.empty:
        logger.warning("dynamic_filter called on empty DataFrame")
        return "No data available for filtering."
    
    alias_map = {
    r'(?i)^log2foldchange$': 'log2FC',
    r'(?i)^logfc$':          'log2FC',
    r'(?i)^log2\.?foldchange$': 'log2FC',
}
    for pattern, canonical in alias_map.items():
        for col in list(df.columns):
            if re.match(pattern, col):
                logger.info("Normalizing column '%s' → '%s'", col, canonical)
                df = df.rename(columns={col: canonical})
                
    if "log2FC" not in df.columns:
        available_cols = ", ".join(df.columns.tolist()[:10])  # Show first 10 columns
        if len(df.columns) > 10:
            available_cols += f"... (+{len(df.columns)-10} more)"
        logger.error("Missing 'log2FC' column in %s. Available columns: %s", current, available_cols)
        return "Missing 'log2FC' column."

    try:
        initial_rows = len(df)
        logger.info("Starting dynamic filtering on %s (%d rows)", current, initial_rows)
        
        # Enhanced numeric validation with detailed logging
        log2fc_valid_count = 0
        pval_valid_count = 0
        
        try:
            log2fc = validate_numeric_column(df["log2FC"], "log2FC", require_all_finite=False)
            log2fc_valid_count = len(log2fc)
            logger.debug("log2FC column: %d valid numeric values out of %d total", 
                        log2fc_valid_count, initial_rows)
        except Exception as log2fc_exc:
            logger.error("Failed to validate log2FC column: %s", log2fc_exc)
            return f"Error: Invalid log2FC column - {log2fc_exc}"
        
        # Enhanced p-value handling
        has_pvalue = "p-value" in df.columns
        if has_pvalue:
            try:
                pval = validate_numeric_column(df["p-value"], "p-value", require_all_finite=False)
                pval_valid_count = len(pval)
                logger.debug("p-value column: %d valid numeric values out of %d total", 
                            pval_valid_count, initial_rows)
                # Use intersection of valid indices
                valid_indices = log2fc.index.intersection(pval.index)
                df_clean = df.loc[valid_indices].copy()
            except Exception as pval_exc:
                logger.warning("p-value column validation failed, ignoring p-value: %s", pval_exc)
                df_clean = df.copy()
                has_pvalue = False
        else:
            logger.info("No p-value column found, filtering based on log2FC only")
            df_clean = df.copy()

        # Enhanced data cleaning with detailed statistics
        pre_clean_rows = len(df_clean)
        
        # Create comprehensive filtering mask
        log2fc_finite_mask = np.isfinite(df_clean["log2FC"])
        log2fc_finite_count = log2fc_finite_mask.sum()
        
        if has_pvalue:
            pval_valid_mask = ~df_clean["p-value"].isna()
            pval_valid_count_final = pval_valid_mask.sum()
            combined_mask = log2fc_finite_mask & pval_valid_mask
        else:
            combined_mask = log2fc_finite_mask
            pval_valid_count_final = pre_clean_rows  # All rows considered valid for p-value
        
        df_clean = df_clean[combined_mask].copy()
        post_clean_rows = len(df_clean)
        
        # Log cleaning statistics
        rows_removed = initial_rows - post_clean_rows
        if rows_removed > 0:
            removal_percentage = (rows_removed / initial_rows) * 100
            logger.info("Data cleaning removed %d rows (%.1f%%) - finite log2FC: %d, valid p-value: %d", 
                       rows_removed, removal_percentage, log2fc_finite_count, pval_valid_count_final)
        
        if df_clean.empty:
            logger.warning("All rows removed during cleaning - no valid data remaining")
            return "No valid data remaining after cleaning."

        # If causal mode, skip DEG thresholding entirely
        if ctx.context.causal:
            degs = df_clean.copy()
            deg_count = len(degs)
            logger.info("Causal flag set; skipping |log2FC| thresholding. Retaining %d cleaned rows.", deg_count)
            ctx.context.threshold = 0.0
            ctx.context.degs = degs
            processing_time = time.time() - start_time
            if deg_count:
                memory_usage_mb = degs.memory_usage(deep=True).sum() / (1024 * 1024)
                logger.info("Dynamic filtering (causal) completed in %.3fs, dataset memory: %.1f MB",
                            processing_time, memory_usage_mb)
            return (f"{deg_count:,} genes retained (causal mode; no |log2FC| threshold) "
                    f"from {post_clean_rows:,} cleaned rows.")

        # Enhanced statistical analysis
        abs_vals = df_clean["log2FC"].abs()
        log2fc_stats = {
            'count': len(abs_vals),
            'mean': float(abs_vals.mean()),
            'median': float(abs_vals.median()),
            'std': float(abs_vals.std()),
            'min': float(abs_vals.min()),
            'max': float(abs_vals.max()),
            'q25': float(abs_vals.quantile(0.25)),
            'q75': float(abs_vals.quantile(0.75))
        }
        
        logger.info("log2FC statistics - Mean: %.3f, Median: %.3f, Max: %.3f, Std: %.3f", 
                   log2fc_stats['mean'], log2fc_stats['median'], 
                   log2fc_stats['max'], log2fc_stats['std'])

        max_abs = log2fc_stats['max']

        # Enhanced threshold selection with detailed logging
        threshold = DEFAULT_THRESHOLD
        selected_rule = None
        
        for i, rule in enumerate(FIXED_THRESHOLDS):
            if max_abs >= rule["min_abs_log2fc"]:
                threshold = rule["threshold"]
                selected_rule = rule
                logger.debug("Threshold rule %d applied: max_abs=%.3f >= min_abs=%.3f -> threshold=%.3f", 
                           i, max_abs, rule["min_abs_log2fc"], threshold)
                break
        
        if selected_rule is None:
            logger.info("Using default threshold %.3f (max_abs=%.3f below all rule minimums)", 
                       threshold, max_abs)
        else:
            logger.info("Selected threshold %.3f based on max |log2FC|=%.3f", threshold, max_abs)

        # Enhanced DEG filtering with statistics
        deg_mask = abs_vals >= threshold
        degs = df_clean[deg_mask].copy()
        deg_count = len(degs)
        
        # Calculate additional DEG statistics
        if deg_count > 0:
            upregulated = (degs["log2FC"] > 0).sum()
            downregulated = (degs["log2FC"] < 0).sum()
            deg_percentage = (deg_count / post_clean_rows) * 100
            
            deg_log2fc_range = (float(degs["log2FC"].min()), float(degs["log2FC"].max()))
            
            logger.info("DEG filtering results: %d DEGs (%.1f%%) - %d upregulated, %d downregulated", 
                       deg_count, deg_percentage, upregulated, downregulated)
            logger.info("DEG log2FC range: [%.3f, %.3f]", deg_log2fc_range[0], deg_log2fc_range[1])
            
            # Log threshold efficiency
            if threshold > 0:
                potential_degs_at_half_threshold = (abs_vals >= threshold/2).sum()
                if potential_degs_at_half_threshold > deg_count:
                    logger.debug("Threshold sensitivity: %d genes at %.3f vs %d at %.3f", 
                               potential_degs_at_half_threshold, threshold/2, deg_count, threshold)
        else:
            logger.warning("No DEGs found with threshold %.3f (max |log2FC|=%.3f)", threshold, max_abs)

        # Update context
        ctx.context.threshold = threshold
        ctx.context.degs = degs
        
        # Enhanced timing and memory reporting
        processing_time = time.time() - start_time
        if degs is not None and not degs.empty:
            memory_usage_mb = degs.memory_usage(deep=True).sum() / (1024 * 1024)
            logger.info("Dynamic filtering completed in %.3fs, DEG dataset memory: %.1f MB", 
                       processing_time, memory_usage_mb)
        
        # Enhanced return message with comprehensive statistics
        if deg_count > 0:
            upregulated = (degs["log2FC"] > 0).sum()
            downregulated = (degs["log2FC"] < 0).sum()
            return (f"{deg_count:,} DEGs with |log2FC| ≥ {threshold:.2f} "
                   f"({upregulated:,} up, {downregulated:,} down) "
                   f"from {post_clean_rows:,} valid rows.")
        else:
            return f"0 DEGs found with |log2FC| ≥ {threshold:.2f} from {post_clean_rows:,} valid rows."
        
    except KeyError as key_err:
        error_msg = f"Error: Missing required column - {key_err}"
        logger.error("Column error in dynamic_filter for %s: %s", current, key_err)
        return error_msg
    except ValueError as val_err:
        error_msg = f"Error: Invalid data values - {val_err}"
        logger.error("Value error in dynamic_filter for %s: %s", current, val_err)
        return error_msg
    except MemoryError:
        error_msg = "Error: Insufficient memory for filtering operation"
        logger.error("Memory error in dynamic_filter for %s", current)
        return error_msg
    except Exception as exc:
        logger.exception("Unexpected error in dynamic_filter for %s: %s", current, exc)
        return f"Error in dynamic_filter: {exc}"


@function_tool
def save_filtered_degs(
    ctx: RunContextWrapper[FilterContext], output_dir: str
) -> str:
    """
    Save filtered DEGs as CSV, rename columns, drop invalid rows.
    Enhanced with robust validation, flexible column mapping, and comprehensive reporting.
    """
    start_time = time.time()
    degs = ctx.context.degs
    current = ctx.context.current_file
    threshold = ctx.context.threshold
    
    # Enhanced input validation
    if degs is None:
        logger.warning("save_filtered_degs called but no DEGs available for %s", current)
        return f"No DEGs to save for {current}."
    
    if degs.empty:
        logger.warning("save_filtered_degs called with empty DEGs DataFrame for %s", current)
        return f"No DEGs to save for {current} (empty dataset)."
    
    if not isinstance(output_dir, str):
        logger.error("Invalid output directory provided: %s", output_dir)
        return "Error: Invalid output directory specified."

    canon_out = (getattr(ctx.context, "canonical_output_dir", None) or "").strip()
    effective_out = (output_dir or "").strip()
    if canon_out:
        try:
            p_agent = Path(effective_out).expanduser().resolve()
            if not effective_out or not p_agent.exists():
                effective_out = canon_out
                logger.warning(
                    "Using canonical_output_dir %s (agent gave %r)",
                    canon_out,
                    output_dir,
                )
        except OSError:
            effective_out = canon_out
            logger.warning(
                "Using canonical_output_dir %s (agent path not resolvable: %r)",
                canon_out,
                output_dir,
            )

    if not effective_out:
        logger.error("Invalid output directory after resolution: %s", output_dir)
        return "Error: Invalid output directory specified."

    try:
        #initial_deg_count = len(degs)          removed part
        #logger.info("Saving %d DEGs from %s to %s", initial_deg_count, current, output_dir)            removed part
        

        initial_deg_count = len(degs)
        logger.info("Saving %d DEGs from %s/%s to %s",
                    initial_deg_count,
                    ctx.context.current_folder,
                    current,
                    effective_out)
        

        # Enhanced column mapping with more flexible patterns
        rename_map = {}
        original_columns = list(degs.columns)
        
        # Define comprehensive column mapping patterns
        column_patterns = {
            "Gene": [
                r"gene.*name", r"gene.*id", r"gene.*symbol", r"symbol", 
                r"gene", r"geneid", r"id", r"identifier", r"ensembl", r"genes",r"ensembl_id"
            ],
            "log2FC": [
                r"log2.*fc", r"log.*fc", r"logfc", r"log2.*fold", 
                r"fold.*change", r"fc", r"lfc", r"log2foldchange"
            ],
            "p-value": [
                r"p.*value", r"pvalue", r"p\.value", r"pval", 
                 r"prob", r"p_value"
            ],
            "adj-p-value": [
                r"adj.*p.*value", r"padj", r"p\.adj", r"fdr", 
                r"q.*value", r"qvalue", r"adjusted.*p", r"benjamini"
            ]
        }
        for col in degs.columns:
            low = col.lower()
            if col not in rename_map and \
            "log" in low and "fold" in low and "change" in low:
                rename_map[col] = "log2FC"
                logger.info("Dynamically mapped column '%s' -> 'log2FC'", col)
                break  # only map the first match

        # now apply
        df2 = degs.rename(columns=rename_map)

        # Apply pattern matching for column renaming
        for col in degs.columns:
            col_lower = col.lower().replace(" ", "").replace("-", "").replace("_", "")
            
            for canonical_name, patterns in column_patterns.items():
                for pattern in patterns:
                    if re.search(pattern, col_lower):
                        if canonical_name not in rename_map.values():
                            rename_map[col] = canonical_name
                            logger.debug("Mapped column '%s' -> '%s'", col, canonical_name)
                            break
                if col in rename_map:
                    break

        # **Re-apply EVERY mapping** (including your dynamic fallback) to the DataFrame:
        df2 = df2.rename(columns=rename_map)


        column_aliases = {
            # DESeq2
            "log2FoldChange": "log2FC",
            "pvalue":         "p-value",
            "padj":           "adj-p-value",
            # edgeR
            "logFC":          "log2FC",
            "PValue":         "p-value",
            "FDR":            "adj-p-value",
            # Limma
            "logFC":          "log2FC",    # repeats are fine
            "P.Value":        "p-value",
            "adj.P.Val":      "adj-p-value",
        }

        # 2) build a rename dict only for those columns actually present
        simple_map = {
            orig: new for orig, new in column_aliases.items()
                if orig in df2.columns
        }

        if simple_map:
            logger.info("Applying tool‐specific renames: %s", simple_map)
            df2 = df2.rename(columns=simple_map)


        # Now continue with your gene_pattern filtering, MyGeneInfo remapping, etc.
        gene_pattern = re.compile(r'^[A-Za-z][A-Za-z0-9\-_.]*$', re.IGNORECASE)
        mask = df2['Gene'].astype(str).str.match(gene_pattern)
        df2 = df2[mask]
        # Log column mapping results
        if rename_map:
            logger.info("Column mapping applied: %s", 
                       {k: v for k, v in rename_map.items()})
        else:
            logger.warning("No column patterns matched for standard renaming")


        # Enhanced requirement checking with flexible alternatives
        primary_req = ["Gene", "log2FC"]
        alternative_gene_cols = [col for col in df2.columns 
                               if any(pattern in col.lower() 
                                     for pattern in ["gene", "symbol", "id", "name"])]
        alternative_fc_cols = [col for col in df2.columns 
                              if any(pattern in col.lower() 
                                    for pattern in ["log2fc", "logfc", "fold", "fc"])]
        
        # Check for required columns with alternatives
        missing_cols = []
        final_columns = []
        
        # Gene column handling
        if "Gene" in df2.columns:
            final_columns.append("Gene")
        elif alternative_gene_cols:
            gene_col = alternative_gene_cols[0]
            df2 = df2.rename(columns={gene_col: "Gene"})
            final_columns.append("Gene")
            logger.info("Using '%s' as Gene column", gene_col)
        else:
            missing_cols.append("Gene")
        
        # log2FC column handling
        if "log2FC" in df2.columns:
            final_columns.append("log2FC")
        elif alternative_fc_cols:
            fc_col = alternative_fc_cols[0]
            df2 = df2.rename(columns={fc_col: "log2FC"})
            final_columns.append("log2FC")
            logger.info("Using '%s' as log2FC column", fc_col)
        else:
            missing_cols.append("log2FC")
        
        if missing_cols:
            available_cols = ", ".join(df2.columns.tolist()[:10])
            if len(df2.columns) > 10:
                available_cols += f"... (+{len(df2.columns)-10} more)"
            logger.error("Missing required columns %s in %s. Available: %s", 
                        missing_cols, current, available_cols)
            return f"Cannot save {current}: missing {missing_cols}"
        
        # Enhanced data cleaning with detailed tracking
        cleaning_stats = {
            'initial_rows': len(df2),
            'after_na_drop': 0,
            'after_gene_clean': 0,
            'after_log2fc_clean': 0,
            'after_pvalue_clean': 0,
            'final_rows': 0
        }
        
        # Drop NA values in required columns
        df2 = drop_na_in_columns(df2, final_columns)
        cleaning_stats['after_na_drop'] = len(df2)
        
        # Enhanced data validation and cleaning
        # Clean Gene column - remove empty strings and whitespace
        if "Gene" in df2.columns:
            initial_genes = len(df2)
            df2 = df2[df2["Gene"].astype(str).str.strip() != ""]
            df2 = df2[df2["Gene"].astype(str) != "nan"]
            cleaning_stats['after_gene_clean'] = len(df2)
            if len(df2) < initial_genes:
                logger.info("Removed %d rows with invalid gene identifiers", 
                           initial_genes - len(df2))
        
        # Enhanced log2FC validation
        if "log2FC" in df2.columns:
            initial_fc = len(df2)
            # More robust finite check
            try:
                df2["log2FC"] = pd.to_numeric(df2["log2FC"], errors='coerce')
                df2 = df2[df2["log2FC"].notna() & np.isfinite(df2["log2FC"])]
                cleaning_stats['after_log2fc_clean'] = len(df2)
                
                if len(df2) < initial_fc:
                    logger.info("Removed %d rows with invalid log2FC values", 
                               initial_fc - len(df2))
                
                # Log log2FC statistics
                if not df2.empty:
                    fc_stats = df2["log2FC"].describe()
                    logger.info("Final log2FC range: [%.3f, %.3f], mean: %.3f", 
                               fc_stats['min'], fc_stats['max'], fc_stats['mean'])
                
            except Exception as fc_exc:
                logger.error("Error processing log2FC column: %s", fc_exc)
                return f"Error processing log2FC values: {fc_exc}"
        
        # Enhanced p-value handling
        optional_columns = []
        if "p-value" in df2.columns:
            initial_pval = len(df2)
            try:
                df2["p-value"] = pd.to_numeric(df2["p-value"], errors='coerce')
                df2 = df2[df2["p-value"].notna() & np.isfinite(df2["p-value"])]
                cleaning_stats['after_pvalue_clean'] = len(df2)
                
                if len(df2) < initial_pval:
                    logger.info("Removed %d rows with invalid p-values", 
                               initial_pval - len(df2))
                
                # Validate p-value range
                if not df2.empty:
                    pval_min, pval_max = df2["p-value"].min(), df2["p-value"].max()
                    if pval_min < 0 or pval_max > 1:
                        logger.warning("p-values outside [0,1] range: [%.6f, %.6f]", 
                                     pval_min, pval_max)
                
                optional_columns.append("p-value")
            except Exception as pval_exc:
                logger.warning("Error processing p-value column, excluding: %s", pval_exc)
        
        # Check for adjusted p-values (do NOT drop rows for NaN/inf; keep as-is)
        # Also handle common alias 'padj'
        if "padj" in df2.columns and "adj-p-value" not in df2.columns:
            df2 = df2.rename(columns={"padj": "adj-p-value"})
        if "adj-p-value" in df2.columns:
            try:
                df2["adj-p-value"] = pd.to_numeric(df2["adj-p-value"], errors='coerce')
                # Do not filter on adj-p-value; retain rows even if NaN/inf
                optional_columns.append("adj-p-value")
                logger.info("Including adj-p-value column without dropping NaN/inf rows")
            except Exception:
                logger.warning("Error processing adj-p-value column, excluding")
        
        cleaning_stats['final_rows'] = len(df2)
        
        # Report cleaning statistics
        total_removed = cleaning_stats['initial_rows'] - cleaning_stats['final_rows']
        if total_removed > 0:
            removal_percentage = (total_removed / cleaning_stats['initial_rows']) * 100
            logger.info("Data cleaning removed %d rows (%.1f%%) - Final: %d valid DEGs", 
                       total_removed, removal_percentage, cleaning_stats['final_rows'])
        
        if df2.empty:
            logger.warning("All DEGs removed during cleaning for %s", current)
            return f"No valid DEGs remaining after data cleaning for {current}"
        
        # Enhanced output file handling
        try:
            output_path = Path(effective_out)
            if not output_path.exists():
                output_path.mkdir(parents=True, exist_ok=True)
                logger.info("Created output directory: %s", output_path)
        except Exception as dir_exc:
            logger.error(
                "Failed to create output directory %s: %s", effective_out, dir_exc
            )
            return f"Error: Cannot create output directory - {dir_exc}"
        
        # Generate enhanced filename with metadata
        #basename = Path(current).stem          removed part
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        basename = Path(current).stem
        folder   = ctx.context.current_folder
        
        # Include threshold and count in filename for better organization
        """if threshold is not None:
            filename = f"filtered_{basename}_t{threshold:.2f}_n{len(df2)}.csv"
        else:
            filename = f"filtered_{basename}_n{len(df2)}.csv"
        
        output_file = output_path / filename

        removed part
        """
        

        # include folder in the filename
        if threshold is not None:
            filename = f"filtered_{folder}_{basename}_t{threshold:.2f}_n{len(df2)}.csv"
        else:
            filename = f"filtered_{folder}_{basename}_n{len(df2)}.csv"
        # Prepare final column order
        output_columns = final_columns + optional_columns
        
        output_file = output_path / filename
        # Enhanced CSV writing with error handling
        try:
            df2.to_csv(output_file, index=False, columns=output_columns, 
                      encoding='utf-8', float_format='%.6g')
            
            file_size_mb = output_file.stat().st_size / (1024 * 1024)
            processing_time = time.time() - start_time
            
            logger.info("Successfully saved %d DEGs to %s (%.2f MB) in %.3fs", 
                       len(df2), output_file.name, file_size_mb, processing_time)
            
            # Generate summary statistics for the saved file
            if len(df2) > 0:
                upregulated = (df2["log2FC"] > 0).sum()
                downregulated = (df2["log2FC"] < 0).sum()
                
                summary_msg = (f"Filtered DEGs saved to {output_file.name}: "
                             f"{len(df2):,} total ({upregulated:,} up, {downregulated:,} down)")
                
                if threshold is not None:
                    summary_msg += f" with |log2FC| ≥ {threshold:.2f}"
                
                return summary_msg
            else:
                return f"Empty DEG file saved to {output_file.name}"
                
        except PermissionError:
            error_msg = f"Error: Permission denied writing to {output_file}"
            logger.error(error_msg)
            return error_msg
        except Exception as save_exc:
            error_msg = f"Error writing CSV file: {save_exc}"
            logger.error("Failed to save CSV for %s: %s", current, save_exc)
            return error_msg
            
    except Exception as exc:
        logger.exception("Unexpected error saving DEGs for %s: %s", current, exc)
        return f"Error saving filtered DEGs: {exc}"


def build_filter_agent() -> Agent[FilterContext]:
    """
    Build an enhanced FilterAgent with comprehensive instructions and error handling.
    """
    enhanced_instructions = (
        "You are a specialized DEG (Differentially Expressed Genes) filtering agent. "
        "Your workflow consists of three critical steps:\n\n"
        "1) LOAD: Load and validate the expression data file (CSV/TSV/XLS/XLSX)\n"
        "   - Support multiple file formats with robust encoding detection\n"
        "   - Validate data structure and report loading statistics\n"
        "   - Handle large files efficiently with memory monitoring\n\n"
        "2) FILTER: Apply dynamic |log2FC| threshold filtering\n"
        "   - Use piecewise thresholding based on data characteristics\n"
        "   - Clean invalid rows and handle missing values appropriately\n"
        "   - Generate comprehensive filtering statistics and DEG counts\n\n"
        "3) SAVE: Export filtered DEGs to standardized CSV format\n"
        "   - Apply intelligent column mapping to standard names\n"
        "   - Validate data integrity before saving\n"
        "   - Generate detailed save confirmation with statistics\n\n"
        "IMPORTANT GUIDELINES:\n"
        "- When calling load_expression_file, pass the file_path string EXACTLY as given in the "
        "user message — character-for-character — UUIDs are sensitive to typos.\n"
        "- If a canonical path is repeated in the prompt, prefer that exact string.\n"
        "- Always validate input data quality and report any issues\n"
        "- Provide detailed statistics at each step (row counts, filtering results)\n"
        "- Handle errors gracefully with clear, actionable error messages\n"
        "- Log all significant operations for debugging and audit purposes\n"
        "- Return concise but informative confirmations after each operation\n\n"
        "Your final response should be a brief summary of the complete workflow results, "
        "including the number of DEGs processed and saved successfully."
    )
    
    return Agent[FilterContext](
        name="EnhancedFilterAgent",
        instructions=enhanced_instructions,
        tools=[load_expression_file, dynamic_filter, save_filtered_degs],
    )