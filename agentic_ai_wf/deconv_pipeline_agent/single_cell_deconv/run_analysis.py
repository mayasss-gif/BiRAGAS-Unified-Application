#!/usr/bin/env python3
"""
Simple Python script to run the Bisque deconvolution pipeline.

Edit the configuration section below with your paths and parameters.
"""

import os
import glob
import json
import re
import asyncio
import logging
from pathlib import Path
from typing import List, Optional, Any

from .utils import abspath_any
from .drivers import run_s1, run_s2
from .s3.run import run_s3
# NOTE: Do NOT import Cibersort module directly - use subprocess for fork-safety
# from .Cibersort import run_cibersort_pipeline
from .xcell import run_xcell_pipeline
from .orchestrator import mode_inhouse_flow, SC_DISEASE, SC_ORGAN
from openai import OpenAI
from decouple import config
import subprocess
import sys

os.environ["OPENAI_API_KEY"] = config("OPENAI_API_KEY")


logger = logging.getLogger(__name__)

AGENT_NAME = "Deconvolution Agent"
STEP = "deconvolution"


def _emit_deconv_log(
    workflow_logger: Any,
    event_loop: Optional[asyncio.AbstractEventLoop],
    level: str,
    message: str,
) -> None:
    """Emit log to UI from sync pipeline (thread-safe via run_coroutine_threadsafe)."""
    if not workflow_logger or not event_loop:
        return
    try:
        async def _do_log() -> None:
            try:
                if level == "info":
                    await workflow_logger.info(agent_name=AGENT_NAME, message=message, step=STEP)
                elif level == "warning":
                    await workflow_logger.warning(agent_name=AGENT_NAME, message=message, step=STEP)
            except Exception:
                pass
        asyncio.run_coroutine_threadsafe(_do_log(), event_loop)
    except Exception:
        pass


# ============================================================================
# Data validation/preprocessing for deconvolution
# ============================================================================
def validate_and_preprocess_counts(counts_file: str) -> str:
    """
    Validate and preprocess counts file for deconvolution.
    
    Removes rows with missing/empty gene IDs to prevent R errors.
    Creates cleaned temporary file if needed.
    
    Args:
        counts_file: Path to counts file
        
    Returns:
        Path to validated/cleaned counts file (original if no changes needed)
        
    Raises:
        ValueError: If file is invalid or has no valid genes
    """
    import pandas as pd
    import tempfile
    from pathlib import Path
    
    counts_path = Path(counts_file)
    if not counts_path.exists():
        raise FileNotFoundError(f"Counts file not found: {counts_file}")
    
    # Read counts file
    try:
        sep = "," if counts_path.suffix.lower() == ".csv" else "\t"
        df = pd.read_csv(counts_file, sep=sep, index_col=None)
    except Exception as e:
        raise ValueError(f"Failed to read counts file: {e}") from e
    
    if df.empty:
        raise ValueError("Counts file is empty")
    
    # Find gene column (first column or column with gene-related name)
    gene_col_candidates = ["Gene", "gene", "GeneSymbol", "Gene_Symbol", "Symbol", 
                           "SYMBOL", "GeneSymbolID", "Name", "id", "ID"]
    gene_col = None
    for col in gene_col_candidates:
        if col in df.columns:
            gene_col = col
            break
    
    if gene_col is None:
        gene_col = df.columns[0]
        logger.info(f"Using first column as gene column: {gene_col}")
    
    # Validate and clean gene column
    initial_rows = len(df)
    
    # Remove rows with missing/empty gene IDs
    df_clean = df.dropna(subset=[gene_col])
    df_clean = df_clean[df_clean[gene_col].astype(str).str.strip() != ""]
    
    removed = initial_rows - len(df_clean)
    if removed > 0:
        logger.warning(f"Removed {removed} rows with missing/empty gene IDs")
    
    if len(df_clean) == 0:
        raise ValueError("No valid genes found after cleaning (all rows have missing/empty gene IDs)")
    
    # Check for duplicate gene IDs
    duplicates = df_clean[gene_col].duplicated().sum()
    if duplicates > 0:
        logger.warning(f"Found {duplicates} duplicate gene IDs, keeping first occurrence")
        df_clean = df_clean.drop_duplicates(subset=[gene_col], keep='first')
    
    # If no changes needed, return original file
    if removed == 0 and duplicates == 0:
        logger.info(f"Counts file validation passed: {len(df_clean)} genes")
        return counts_file
    
    # Create cleaned temporary file
    temp_dir = Path(counts_path.parent) / ".cleaned"
    temp_dir.mkdir(exist_ok=True)
    cleaned_file = temp_dir / f"{counts_path.stem}_cleaned{counts_path.suffix}"
    
    # Save cleaned file
    df_clean.to_csv(cleaned_file, index=False, sep=sep)
    logger.info(f"Created cleaned counts file: {cleaned_file} ({len(df_clean)} genes)")
    
    return str(cleaned_file)


# ============================================================================
# Subprocess wrapper for CIBERSORT (fork-safe for Celery)
# ============================================================================
def run_cibersort_via_subprocess(
    counts: str,
    out: str,
    lm22: Optional[str] = None,
    meta: Optional[str] = None,
    perm: int = 100,
    qn: bool = False,
    chunk_size: int = 60,
) -> None:
    """
    Run CIBERSORT in an isolated subprocess for Celery fork-safety.
    
    CRITICAL: This avoids rpy2/R fork-safety issues by running R in a
    completely separate process that never inherits forked state.
    
    Args:
        counts: Path to counts file
        out: Output directory
        lm22: Path to LM22 signature matrix (optional, uses default if None)
        meta: Optional metadata file
        perm: CIBERSORT permutations (default: 100)
        qn: Enable quantile normalization (default: False)
        chunk_size: Samples per page for plots (default: 60)
        
    Raises:
        subprocess.CalledProcessError: If CIBERSORT fails
    """
    # Use default LM22 path if not provided
    if lm22 is None:
        # Default LM22 signature matrix bundled with CIBERSORT module
        # __file__ is in: agentic_ai_wf/deconv_pipeline_agent/single_cell_deconv/run_analysis.py
        # LM22 is at: agentic_ai_wf/deconv_pipeline_agent/single_cell_deconv/inst/extdata/LM22.txt
        lm22_default = Path("./agentic_ai_wf/deconv_pipeline_agent/single_cell_deconv/inst/extdata/LM22.txt").resolve()
        if not lm22_default.exists():
            raise FileNotFoundError(f"Default LM22 file not found: {lm22_default}")
        lm22 = str(lm22_default)  # Use absolute path for subprocess
    
    # Build command to run CIBERSORT as subprocess
    cmd = [
        sys.executable,  # Use same Python interpreter
        "-m",
        "agentic_ai_wf.deconv_pipeline_agent.single_cell_deconv.Cibersort",
        "--counts", str(counts),
        "--lm22", str(lm22),  # Always required
        "--out", str(out),
        "--perm", str(perm),
        "--qn", "true" if qn else "false",
        "--chunk-size", str(chunk_size),
    ]
    
    # Add optional metadata
    if meta:
        cmd.extend(["--meta", str(meta)])
    
    # Validate and preprocess counts file before running CIBERSORTx
    try:
        validated_counts = validate_and_preprocess_counts(counts)
        if validated_counts != counts:
            logger.info(f"Using cleaned counts file: {validated_counts}")
            counts = validated_counts
    except Exception as e:
        logger.error(f"❌ Counts file validation failed: {e}")
        raise RuntimeError(f"Counts file validation failed: {e}") from e
    
    # Update command with validated counts file
    cmd[cmd.index("--counts") + 1] = str(counts)
    
    logger.info(f"Running CIBERSORT via subprocess: {' '.join(cmd)}")
    
    try:
        # Run in isolated subprocess (fork-safe)
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            timeout=3600,  # 1 hour timeout
        )
        
        # Log output
        if result.stdout:
            logger.info(f"CIBERSORT stdout:\n{result.stdout}")
        if result.stderr:
            logger.warning(f"CIBERSORT stderr:\n{result.stderr}")
            
        logger.info(f"✅ CIBERSORT completed successfully. Output: {out}")
        
    except subprocess.TimeoutExpired as e:
        logger.error(f"❌ CIBERSORT timed out after 1 hour")
        raise RuntimeError("CIBERSORT execution timed out") from e
        
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ CIBERSORT failed with exit code {e.returncode}")
        if e.stdout:
            logger.error(f"CIBERSORT stdout:\n{e.stdout}")
        if e.stderr:
            logger.error(f"CIBERSORT stderr:\n{e.stderr}")
        raise RuntimeError(f"CIBERSORT execution failed: {e.stderr}") from e


# ============================================================================
# CONFIGURATION - Edit these paths and parameters
# ============================================================================

# Input files (required)
# REF_H5AD = r"trachea.h5ad"  # or use ROOT instead
# ROOT = None  # Alternative to REF_H5AD: path to folder containing reference files
# BULK = r"Lupus.csv"
# METADATA = r"Lupus_meta.csv"  # Optional

# Output directories (optional - auto-generated if not specified)
OUTDIR = r"Bisque-Preprocessing"
S2_OUTDIR = r"Bisque_deconvolution_results"
ENH_OUTDIR = r"Enhanced-Deconv-Reports"

# Analysis parameters
BULK_GENE_COL = "Gene"  # Column name in bulk CSV containing gene identifiers
SPECIES = "human"  # "human" or "mouse"

# Tissue filtering (optional)
TISSUE_INCLUDE = []  # e.g., ["thymus", "spleen"]
TISSUE_KEYS = ["tissue"]  # Column names in reference to check for tissue
TISSUE_MATCH_MODE = "substring"  # "exact" or "substring"

# S2 tuning parameters
DROP_MITO = "true"
ALLOW_DUPLICATE_GENES = "true"
WARN_MIN_OVERLAP = "200"
MAX_PROP_DEVIATION = "0.05"
TOP_MARKERS_PER_CT = "25"
TOP_GENES_PER_SAMPLE = "40"
AUTO_INSTALL = "true"

# S3 labels - Use None to enable intelligent auto-detection
# Compatible with all metadata files validated by deg_file_validator_tools.py
SAMPLE_COL = None  # Auto-detects: 'sample', 'sample_id', 'sampleID', etc.
CONDITION_COL = None  # Auto-detects: 'condition', 'group', 'treatment', etc.
CONTROL_LABEL = None  # Auto-detects: 'Control', 'Normal', 'Healthy', etc.
PATIENT_LABEL = None  # Auto-detects: 'Disease', 'Patient', 'Treated', etc.

# Stage toggles (set to True to skip a stage)
SKIP_S1 = False
SKIP_S2 = False
SKIP_S3 = False

# ============================================================================
# RUN PIPELINE
# ============================================================================

def run_bisque_pipeline(bulk_file, metadata, h5ad_file, output_dir, auto_reduce_h5ad=True, workflow_logger: Any = None, event_loop: Optional[asyncio.AbstractEventLoop] = None):
    """Run the Bisque deconvolution pipeline.
    
    Args:
        bulk_file: Path to the bulk expression CSV file
        metadata: Path to the metadata CSV file (optional, can be None)
        h5ad_file: Path to the reference h5ad file
        output_dir: Main output directory where all results will be generated
        auto_reduce_h5ad: If True, automatically reduce large h5ad files before processing
    """
    
    # Resolve paths
    ref_h5ad_abs = abspath_any(h5ad_file) if h5ad_file else None
    bulk_abs = abspath_any(bulk_file) if bulk_file else None
    meta_abs = abspath_any(metadata) if metadata else None
    output_dir_abs = abspath_any(output_dir) if output_dir else None
    
    # Validate inputs
    if not bulk_abs:
        raise ValueError("Bulk file required. Please provide bulk_file parameter.")
    if not ref_h5ad_abs:
        raise ValueError("H5AD file required. Please provide h5ad_file parameter.")
    if not output_dir_abs:
        raise ValueError("Output directory required. Please provide output_dir parameter.")
    
    # Create main output directory if it doesn't exist
    Path(output_dir_abs).mkdir(parents=True, exist_ok=True)
    
    # Auto-reduce h5ad file if enabled and file is large
    if auto_reduce_h5ad and ref_h5ad_abs:
        try:
            from .preprocessing import auto_reduce_h5ad, get_file_size_mb
            
            file_size_mb = get_file_size_mb(ref_h5ad_abs)
            # Auto-reduce if file is larger than 500 MB
            if file_size_mb > 500:
                _emit_deconv_log(workflow_logger, event_loop, "info", f"Large h5ad ({file_size_mb:.1f} MB) — applying size reduction")
                logger.info(f"[Bisque Pipeline] Large h5ad file detected ({file_size_mb:.2f} MB). "
                           "Applying automatic size reduction...")
                
                # Create reduced file in preprocessing subdirectory
                preprocessing_dir = os.path.join(output_dir_abs, "preprocessing")
                Path(preprocessing_dir).mkdir(parents=True, exist_ok=True)
                
                reduced_h5ad_path, reduction_summary = auto_reduce_h5ad(
                    input_path=ref_h5ad_abs,
                    output_path=os.path.join(preprocessing_dir, "reference_reduced.h5ad"),
                )
                
                reduction_msg = (
                    f"H5AD reduced: {reduction_summary['reduction_ratio']:.0f}% smaller "
                    f"({reduction_summary['original_size_mb']:.1f} MB → {reduction_summary['final_size_mb']:.1f} MB)"
                )
                _emit_deconv_log(workflow_logger, event_loop, "info", reduction_msg)
                logger.info(f"[Bisque Pipeline] {reduction_msg}")
                logger.info(f"[Bisque Pipeline] Using reduced reference: {reduced_h5ad_path}")
                
                # Use reduced file for pipeline
                ref_h5ad_abs = reduced_h5ad_path
            else:
                msg = f"H5AD file size ({file_size_mb:.2f} MB) is acceptable. Skipping reduction."
                logger.info(f"[Bisque Pipeline] {msg}")
        except ImportError as e:
            warning_msg = f"H5AD reduction module not available: {e}. Continuing with original file."
            _emit_deconv_log(workflow_logger, event_loop, "warning", "H5AD reduction unavailable — using original file")
            logger.warning(f"[Bisque Pipeline] {warning_msg}")
        except Exception as e:
            warning_msg = f"H5AD reduction failed: {e}. Continuing with original file."
            _emit_deconv_log(workflow_logger, event_loop, "warning", "H5AD reduction failed — using original file")
            logger.warning(f"[Bisque Pipeline] {warning_msg}")
            # Continue with original file if reduction fails
    
    # Generate subdirectories inside the main output directory
    out_s1 = os.path.join(output_dir_abs, OUTDIR)
    out_s2 = os.path.join(output_dir_abs, S2_OUTDIR)
    out_s3 = os.path.join(output_dir_abs, ENH_OUTDIR)
    
    # Create subdirectories
    Path(out_s1).mkdir(parents=True, exist_ok=True)
    Path(out_s2).mkdir(parents=True, exist_ok=True)
    Path(out_s3).mkdir(parents=True, exist_ok=True)
    
    _emit_deconv_log(workflow_logger, event_loop, "info", f"Bisque pipeline: reference {os.path.basename(ref_h5ad_abs)}, bulk {os.path.basename(bulk_abs)}")

    if not SKIP_S1:
        _emit_deconv_log(workflow_logger, event_loop, "info", "Stage 1/3: QC and Bisque preparation...")
        run_s1(
            root_for_s1=ref_h5ad_abs,
            out_s1=out_s1,
            bulk_abs=bulk_abs,
            bulk_gene_col=BULK_GENE_COL,
            species=SPECIES,
            tissue_include=TISSUE_INCLUDE,
            tissue_keys=TISSUE_KEYS,
            tissue_match_mode=TISSUE_MATCH_MODE,
        )
        _emit_deconv_log(workflow_logger, event_loop, "info", "Stage 1/3 complete: QC and preparation done")
    if not SKIP_S2:
        _emit_deconv_log(workflow_logger, event_loop, "info", "Stage 2/3: Bisque deconvolution...")
        run_s2(
            out_s1=out_s1,
            out_s2=out_s2,
            drop_mito=DROP_MITO,
            allow_duplicate_genes=ALLOW_DUPLICATE_GENES,
            warn_min_overlap=WARN_MIN_OVERLAP,
            max_prop_deviation=MAX_PROP_DEVIATION,
            top_markers_per_ct=TOP_MARKERS_PER_CT,
            top_genes_per_sample=TOP_GENES_PER_SAMPLE,
            auto_install=AUTO_INSTALL,
        )
        _emit_deconv_log(workflow_logger, event_loop, "info", "Stage 2/3 complete: cell proportions estimated")
    if not SKIP_S3:
        _emit_deconv_log(workflow_logger, event_loop, "info", "Stage 3/3: Enhanced analysis and reporting...")
        run_s3(
            out_s2=out_s2,
            metadata_candidate=meta_abs or "",
            sample_col=SAMPLE_COL,
            condition_col=CONDITION_COL,
            control_label=CONTROL_LABEL,
            patient_label=PATIENT_LABEL,
            out_s3=out_s3,
        )
        _emit_deconv_log(workflow_logger, event_loop, "info", "Stage 3/3 complete: Bisque pipeline finished")



def _find_all_h5ad_files(sc_base_dir: Optional[str] = None) -> List[str]:
    """Find all .h5ad files in both SC_Disease_Dataset and SC_Normal_Organ directories."""
    if sc_base_dir:
        sc_disease_dir = os.path.join(sc_base_dir, "SC_Disease_Dataset")
        sc_organ_dir = os.path.join(sc_base_dir, "SC_Normal_Organ")
    else:
        sc_disease_dir = SC_DISEASE
        sc_organ_dir = SC_ORGAN
    
    all_files = []
    for directory in [sc_disease_dir, sc_organ_dir]:
        if os.path.isdir(directory):
            patt = os.path.join(directory, "**/*.h5ad")
            all_files.extend(glob.glob(patt, recursive=True))
    return list(dict.fromkeys(all_files))

def _llm_select_best_reference(disease_name: Optional[str], aliases: List[str], available_files: List[str]) -> Optional[str]:
    """
    Use LLM to select the most suitable .h5ad file from available files based on disease name.
    
    Parameters
    ----------
    disease_name : Optional[str]
        Disease name to match against.
    aliases : List[str]
        List of disease aliases.
    available_files : List[str]
        List of all available .h5ad file paths.
    
    Returns
    -------
    Optional[str]
        Path to the most suitable file, or None if LLM call fails or no files available.
    """
    if not available_files:
        return None
    
    key = os.getenv("OPENAI_API_KEY")
    if not key or OpenAI is None:
        logger.warning("No OpenAI API key available for LLM-based reference selection")
        return None
    
    try:
        # Extract just the filenames (basenames) for cleaner LLM input
        file_basenames = [os.path.basename(f) for f in available_files]
        
        # Prepare disease context
        disease_context = disease_name or ""
        if aliases:
            disease_context += f" (also known as: {', '.join(aliases[:5])})"
        
        client = OpenAI(api_key=key)
        sys_msg = (
            "You are a biomedical expert selecting the most appropriate single-cell reference dataset. "
            "Given a disease name and a list of available .h5ad file names, select the ONE file that "
            "best matches the disease. Consider disease-organ relationships, tissue types, and biological relevance. "
            "Return ONLY the exact filename (basename only, e.g., 'lupus_blood.h5ad'), nothing else."
        )
        
        user_prompt = f"""Disease: {disease_context}

Available .h5ad files:
{chr(10).join(f"- {fname}" for fname in file_basenames)}

Select the ONE filename that best matches this disease. Return only the filename."""
        
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,  # Low temperature for more deterministic selection
        )
        
        content = (resp.choices[0].message.content or "").strip()
        
        # Extract filename from response (handle cases where LLM adds extra text)
        # Try to find the filename in the response
        selected_filename = None
        for fname in file_basenames:
            if fname.lower() in content.lower():
                selected_filename = fname
                break
        
        # If exact match not found, try to extract just the filename part
        if not selected_filename:
            # Remove common prefixes/suffixes and extract potential filename
            lines = content.split('\n')
            for line in lines:
                line = line.strip().strip('"').strip("'").strip('`')
                if line.endswith('.h5ad'):
                    # Check if this matches any available file
                    for fname in file_basenames:
                        if line.lower() == fname.lower() or line.lower() in fname.lower():
                            selected_filename = fname
                            break
                    if selected_filename:
                        break
        
        if selected_filename:
            # Find the full path
            for full_path in available_files:
                if os.path.basename(full_path) == selected_filename:
                    logger.info(f"LLM selected reference: {selected_filename} for disease: {disease_context}")
                    return full_path
        
        logger.warning(f"LLM response did not match any available file. Response: {content[:100]}")
        return None
        
    except Exception as e:
        logger.error(f"Error in LLM-based reference selection: {e}")
        return None

def run_pipeline(
    bulk_file,
    metadata,
    output_dir,
    h5ad_file=None,
    technique=None,
    sample_type="blood",
    disease_name="lupus",
    sc_base_dir="SC_BASE",
    auto_reduce_h5ad=True,
    workflow_logger: Any = None,
    event_loop: Optional[asyncio.AbstractEventLoop] = None,
) -> None:
    """Run the pipeline.
    
    Args:
        bulk_file: Path to the bulk expression CSV file
        metadata: Path to the metadata CSV file (optional, can be None)
        output_dir: Main output directory where all results will be generated
        technique: The technique to use for deconvolution (bisque or cibersortx or xcell)
        sample_type: The sample type (blood or pbmc or plasma or tissue or biopsy or ffpe)
        disease_name: The disease name (e.g. "systemic lupus erythematosus", "lupus", "SLE", "pancreatic cancer", "PDAC")
        sc_base_dir: The base directory of the single-cell references which contains the SC_Disease_Dataset and SC_Normal_Organ subdirectories
    """

    # Log parameters before calling orchestrator
    
    logger.info(
        f"run_pipeline calling mode_inhouse_flow with:\n"
        f"  count_path: {bulk_file}\n"
        f"  sample_type: {repr(sample_type)} (type: {type(sample_type)})\n"
        f"  disease_name: {repr(disease_name)} (type: {type(disease_name)})\n"
        f"  meta_path: {metadata}\n"
        f"  sc_base_dir: {sc_base_dir}"
    )
    
    # Validate disease_name before passing to orchestrator
    if not disease_name or (isinstance(disease_name, str) and not disease_name.strip()):
        logger.error(
            f"disease_name validation failed in run_pipeline:\n"
            f"  Received: {repr(disease_name)}\n"
            f"  Type: {type(disease_name)}\n"
            f"  Function signature default: 'lupus'"
        )
        raise ValueError(
            f"disease_name is required but received invalid value: {repr(disease_name)}"
        )
    
    _emit_deconv_log(workflow_logger, event_loop, "info", "Running orchestrator to select technique and reference...")
    orchestrator_result = mode_inhouse_flow(
        count_path=bulk_file,
        sample_type=sample_type,
        disease_name=disease_name,
        meta_path=metadata,
        sc_base_dir=sc_base_dir
    )
    if technique is None:
        technique = orchestrator_result["tool_selected"].lower()
        _emit_deconv_log(workflow_logger, event_loop, "info", f"Orchestrator selected: {technique}")
    else:
        
        technique = technique.lower()
        if technique not in ["bisque", "cibersortx", "xcell"]:
            raise ValueError(f"Invalid technique: {technique}. Must be one of: bisque, cibersortx, xcell. If you want to use the orchestrator, set technique to None.")

    # Prioritize orchestrator's found references over manually provided h5ad_file
    sc_refs = orchestrator_result["decision_trace"].get('sc_refs', [])
    if h5ad_file is None:
        if len(sc_refs) > 0:
            h5ad_file = sc_refs[0]
            logger.info(f"Using orchestrator-found reference: {h5ad_file}")
            _emit_deconv_log(workflow_logger, event_loop, "info", f"Using reference: {os.path.basename(h5ad_file)}")
        else:
            if technique == "bisque":
                # Fallback: If no references found and disease_name is provided, use LLM to select best match
                if disease_name and sc_base_dir:
                    logger.info(f"No direct matches found. Using LLM fallback to select best reference for disease: {disease_name}")
                    all_available_files = _find_all_h5ad_files(sc_base_dir)
                    if all_available_files:
                        # Get aliases from orchestrator result if available (from llm_norm, not decision_trace)
                        # We need to get it from the mode_inhouse_flow call - but it's not directly exposed
                        # So we'll just use an empty list for aliases
                        aliases = []
                        llm_selected = _llm_select_best_reference(disease_name, aliases, all_available_files)
                        if llm_selected:
                            h5ad_file = llm_selected
                            logger.info(f"LLM fallback selected: {os.path.basename(llm_selected)}")
                            _emit_deconv_log(workflow_logger, event_loop, "info", f"LLM selected reference: {os.path.basename(llm_selected)}")
                        else:
                            logger.warning(f"LLM fallback failed to select a reference from {len(all_available_files)} available files")
                            # Keep the provided h5ad_file if LLM fails
                    else:
                        logger.warning("No .h5ad files found in sc_base_dir for LLM fallback")

            else:
                h5ad_file = None
                    # Keep the provided h5ad_file if no files found
            # If orchestrator didn't find any references and LLM fallback didn't work, 
            # fall back to provided h5ad_file (h5ad_file remains as provided, or None if not provided)

    
    # Create a folder inside the output dir by the name of the technique
    technique_out_dir = os.path.join(output_dir, technique)
    os.makedirs(technique_out_dir, exist_ok=True)

    if technique == "bisque":
        if h5ad_file is None:
            raise ValueError("For Bisque, a reference .h5ad file must be provided.")
        _emit_deconv_log(workflow_logger, event_loop, "info", "Starting Bisque deconvolution (QC → Bisque → enhanced analysis)")
        run_bisque_pipeline(bulk_file, metadata, h5ad_file, technique_out_dir, auto_reduce_h5ad=auto_reduce_h5ad, workflow_logger=workflow_logger, event_loop=event_loop)

    elif technique == "cibersortx":
        _emit_deconv_log(workflow_logger, event_loop, "info", "Starting CIBERSORTx deconvolution (LM22 signature)")
        # Use subprocess wrapper for Celery fork-safety (avoids rpy2/R deadlocks)
        run_cibersort_via_subprocess(
            counts=bulk_file,
            out=technique_out_dir,
            meta=metadata,
            perm=100,
            qn=False,
        )
        _emit_deconv_log(workflow_logger, event_loop, "info", "CIBERSORTx complete")
    elif technique == "xcell":
        _emit_deconv_log(workflow_logger, event_loop, "info", "Starting xCell deconvolution")
        run_xcell_pipeline(
            expr_file=bulk_file,
            meta_file=metadata,
            out_dir=technique_out_dir,
        )
        _emit_deconv_log(workflow_logger, event_loop, "info", "xCell complete")
    else:
        raise ValueError(f"Invalid technique: {technique}")


# if __name__ == "__main__":
#     try:
#         run_pipeline(
#             bulk_file=BULK,
#             metadata=METADATA,
#             h5ad_file=REF_H5AD
#         )
#     except KeyboardInterrupt:
#         print("\n[INFO] Interrupted by user")
#         exit(130)
#     except Exception as e:
#         print(f"\n[ERROR] {e}")
#         raise
