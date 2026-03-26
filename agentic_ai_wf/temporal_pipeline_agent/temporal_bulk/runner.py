# src/temporal_bulk_v4/runner.py
import asyncio
import os
import json
import warnings
from pathlib import Path
from typing import Optional, Union, List, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import dotenv
# Load environment variables from a .env file if present
dotenv.load_dotenv()


try:
    from statsmodels.stats.multitest import multipletests
except Exception:
    multipletests = None

try:
    from joblib import Parallel, delayed
    import multiprocessing as mp
    # Set multiprocessing start method for Linux compatibility (Python 3.8+)
    try:
        if hasattr(mp, 'set_start_method'):
            # Try 'spawn' first (most compatible), fall back to 'fork' if not available
            try:
                mp.set_start_method('spawn', force=False)
            except RuntimeError:
                # Already set or 'spawn' not available, try 'fork'
                try:
                    mp.set_start_method('fork', force=False)
                except RuntimeError:
                    pass  # Use default
    except Exception:
        pass  # Ignore errors, use system default
except Exception:
    Parallel = delayed = None
    mp = None

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

# ===== explicit package imports (deterministic) =====
from .core import *  # e.g., R_AVAILABLE if defined in core
from .io import (
    log,
    ensure_dir,
    read_counts,
    read_metadata,
    harmonize,
    read_table_auto,
)
from .transforms import cpm_log1p
from .pseudotime import pseudotime_pca, try_phenopath
from .impulse import impulse_fn, fit_impulse_single
from .ssgsea import (
    _resolve_gene_sets_for_gseapy,
    run_ssgsea as run_ssgsea_func,
    pathway_temporal_summary,
    ssgsea_report_to_matrix,
    find_ssgsea_report_in,
)
from .plotting import (
    save_boxplot_pseudotime,
    save_bar_top_pathways,
    grid_plot_trajectories,
    per_entity_pngs,               # NOTE: expects (long_df, out_dir, prefix)
    _rank_genes_for_gallery,
    _smooth_trend,
    plot_gene_trajectory,
    de_and_interaction_scatter,
)
from .export import export_trajectory_tables, build_causal_pack
from .reporting_html import write_detailed_html_report
from .parser import build_parser


def detect_gene_column_with_llm(csv_path: Union[str, Path], input_dir: Optional[Union[str, Path]] = None) -> Optional[str]:
    """
    Use OpenAI GPT-4o mini to detect which column in a CSV file contains gene names.
    
    Parameters
    ----------
    csv_path : str or Path
        Path to the CSV file
    input_dir : str or Path, optional
        Base directory for resolving relative file paths
        
    Returns
    -------
    str or None
        The name of the column containing genes, or None if detection fails
    """
    if OpenAI is None:
        warnings.warn("OpenAI package not available. Install with: pip install openai")
        return None
    
    # Resolve file path
    csv_file = Path(csv_path)
    if not csv_file.exists() and input_dir is not None:
        csv_file = Path(input_dir) / csv_path
    
    if not csv_file.exists():
        warnings.warn(f"CSV file not found: {csv_file}")
        return None
    
    try:
        # Read CSV to get column names and sample data
        df = pd.read_csv(csv_file, nrows=10)  # Read first 10 rows for context
        columns = df.columns.tolist()
        
        # Create a preview of the data
        preview_data = {}
        for col in columns:
            sample_values = df[col].head(5).astype(str).tolist()
            preview_data[col] = sample_values
        
        # Prepare prompt for GPT-4o mini
        prompt = f"""You are analyzing a CSV file to identify which column contains gene names/identifiers.

Column names in the file:
{', '.join(columns)}

Sample data from each column (first 5 values):
{json.dumps(preview_data, indent=2)}

Please identify which column contains gene names or gene identifiers. Gene names are typically:
- Short alphanumeric identifiers (e.g., "GAPDH", "ACTB", "TP53")
- May include underscores or hyphens
- Usually uppercase or mixed case
- Not numeric-only values
- Not dates or sample IDs

Respond with ONLY the column name, nothing else. If no column clearly contains genes, respond with "None"."""
        
        # Get API key from environment
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            warnings.warn("OPENAI_API_KEY environment variable not set. Cannot detect gene column.")
            return None
        
        # Call OpenAI API
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that identifies gene columns in CSV files. Always respond with only the column name."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=50
        )
        
        detected_col = response.choices[0].message.content.strip()
        
        # Validate that the detected column exists in the CSV
        if detected_col and detected_col != "None" and detected_col in columns:
            return detected_col
        else:
            warnings.warn(f"LLM detected column '{detected_col}' which is not in the CSV columns. Available columns: {columns}")
            return None
            
    except Exception as e:
        warnings.warn(f"Failed to detect gene column using LLM: {e}")
        return None


def detect_time_column_with_llm(csv_path: Union[str, Path], input_dir: Optional[Union[str, Path]] = None) -> Optional[str]:
    """
    Use OpenAI GPT-4o mini to detect which column in a CSV file contains time/temporal information.
    
    Parameters
    ----------
    csv_path : str or Path
        Path to the CSV file (metadata file)
    input_dir : str or Path, optional
        Base directory for resolving relative file paths
        
    Returns
    -------
    str or None
        The name of the column containing time information, or None if detection fails
    """
    if OpenAI is None:
        warnings.warn("OpenAI package not available. Install with: pip install openai")
        return None
    
    # Resolve file path
    csv_file = Path(csv_path)
    if not csv_file.exists() and input_dir is not None:
        csv_file = Path(input_dir) / csv_path
    
    if not csv_file.exists():
        warnings.warn(f"CSV file not found: {csv_file}")
        return None
    
    try:
        # Read CSV to get column names and sample data
        df = pd.read_csv(csv_file, nrows=20)  # Read more rows to better assess time progression
        columns = df.columns.tolist()
        
        # Filter to only numeric columns (time should be numeric)
        numeric_columns = []
        for col in columns:
            try:
                # Try to convert to numeric
                numeric_series = pd.to_numeric(df[col], errors='coerce')
                if numeric_series.notna().sum() > 0:  # Has at least some numeric values
                    numeric_columns.append(col)
            except Exception:
                continue
        
        if not numeric_columns:
            warnings.warn("No numeric columns found in metadata for time detection.")
            return None
        
        # Create a preview of the data for numeric columns
        preview_data = {}
        for col in numeric_columns:
            sample_values = df[col].head(10).astype(str).tolist()
            preview_data[col] = sample_values
        
        # Prepare prompt for GPT-4o mini
        prompt = f"""You are analyzing a metadata CSV file to identify which column contains time or temporal information.

Column names in the file:
{', '.join(columns)}

Numeric columns (candidates for time):
{', '.join(numeric_columns)}

Sample data from numeric columns (first 10 values):
{json.dumps(preview_data, indent=2)}

Please identify which column contains time or temporal progression information. Time columns are typically:
- Numeric values representing progression (e.g., 0, 6, 12, 24, 48 for hours; 0, 1, 2, 3, 7 for days)
- Values that increase or decrease monotonically across samples
- May be named "time", "day", "hour", "day_post", "timepoint", "age", "duration", etc.
- NOT sample IDs, gene counts, or other non-temporal measurements
- Should represent a temporal axis for biological progression

Respond with ONLY the column name, nothing else. If no column clearly represents time, respond with "None"."""
        
        # Get API key from environment
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            warnings.warn("OPENAI_API_KEY environment variable not set. Cannot detect time column.")
            return None
        
        # Call OpenAI API
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that identifies time/temporal columns in CSV files. Always respond with only the column name."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=50
        )
        
        detected_col = response.choices[0].message.content.strip()
        
        # Validate that the detected column exists and is numeric
        if detected_col and detected_col != "None" and detected_col in columns:
            # Double-check it's numeric
            try:
                pd.to_numeric(df[detected_col], errors='coerce')
                return detected_col
            except Exception:
                warnings.warn(f"LLM detected column '{detected_col}' but it is not numeric. Skipping.")
                return None
        else:
            warnings.warn(f"LLM detected column '{detected_col}' which is not in the CSV columns. Available columns: {columns}")
            return None
            
    except Exception as e:
        warnings.warn(f"Failed to detect time column using LLM: {e}")
        return None


def _emit_temporal_log(
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
                    await workflow_logger.info(
                        agent_name="Temporal Analysis Agent",
                        message=message,
                        step="temporal_analysis",
                    )
                elif level == "warning":
                    await workflow_logger.warning(
                        agent_name="Temporal Analysis Agent",
                        message=message,
                        step="temporal_analysis",
                    )
            except Exception:
                pass
        asyncio.run_coroutine_threadsafe(_do_log(), event_loop)
    except Exception:
        pass


def run_temporal_analysis(
    output_dir: Union[str, Path],
    counts: Union[str, Path],
    metadata: Union[str, Path],
    input_dir: Optional[Union[str, Path]] = None,
    treatment_level: str = '',
    genes_list: Union[str, List[str]] = '',
    deconv_csv: str = '',
    workflow_logger: Optional[Any] = None,
    event_loop: Optional[asyncio.AbstractEventLoop] = None,
) -> dict:
    """
    Run a complete temporal bulk RNA-seq analysis pipeline.
    
    This function performs end-to-end analysis of bulk RNA-seq data along a temporal axis,
    including pseudotime estimation, impulse model fitting, pathway enrichment analysis,
    and comprehensive visualization. It's designed for analyzing time-course or progression
    datasets where samples represent different stages of a biological process.
    
    The pipeline includes:
    - Data normalization (CPM + log1p transformation)
    - Temporal axis estimation (PCA-based pseudotime or PhenoPath)
    - Per-gene impulse model fitting with statistical testing
    - Pathway enrichment analysis (ssGSEA)
    - Cell-type deconvolution trajectory analysis (optional)
    - Publication-ready figures and HTML reports
    
    Parameters
    ----------
    output_dir : str or Path
        Directory where all results will be saved. Will be created if it doesn't exist.
    
    counts : str or Path
        Path to counts matrix file (CSV/TSV). Format: rows=genes, columns=samples.
        First column should contain gene IDs (or specify with gene_col parameter).
    
    metadata : str or Path
        Path to sample metadata file (CSV/TSV). Must include a 'sample_id' column
        that matches the column names in the counts file.
    
    input_dir : str or Path, optional
        Base directory for resolving relative file paths. If None, files are resolved
        from the current working directory or treated as absolute paths.
        Useful when all input files are organized in a subdirectory.
    
    treatment_level : str, default ''
        Name of the treatment/condition level to use for differential expression
        and interaction analysis. Should match a value in the covariate column.
        Example: 'Treatment', 'Disease', 'Lupus'
    
    genes_list : str or List[str], default ''
        Optional gene filtering. Can be:
        - File path (TXT/CSV): Path to file with gene IDs (one per line or in a column)
        - List of strings: Direct list of gene names, e.g., ["GENE1", "GENE2", ...]
        - Empty string/list: Use all genes (default)
        If provided, only these genes will be analyzed for impulse fitting.
    
    deconv_csv : str, default ''
        Optional path to cell-type deconvolution proportions file.
        Format: rows=samples, columns=cell types. If not provided, will attempt
        auto-detection in input_dir if available.
    
    gene_col : str, default 'gene'
        Name of the gene ID column in the counts file. The function will attempt
        auto-detection if this column name doesn't exist.
    
    Additional Parameters (with defaults)
    -------------------------------------
    The following parameters are available with sensible defaults. See function
    implementation for full parameter list:
    
    - time_col: Use provided time column instead of estimating pseudotime
    - covariate: Primary grouping variable (e.g., 'condition', 'treatment')
    - run_ssgsea: Enable pathway enrichment analysis (default: True)
    - gene_sets: Gene set library for ssGSEA (default: 'Hallmark')
    - make_figures: Generate visualization plots (default: True)
    - render_report: Generate HTML report (default: True)
    - n_jobs: Number of parallel workers for gene fitting (default: 1)
    
    Returns
    -------
    dict
        Dictionary containing:
        - 'status': Analysis completion status ('ok' or 'precheck_only')
        - 'output_dir': Path to output directory
        - 'pseudotime_csv': Path to pseudotime results
        - 'gene_fits_tsv': Path to gene-level impulse fit results
        - 'pathway_summary_csv': Path to pathway temporal summary (if ssGSEA ran)
        - 'pathway_profiles_csv': Path to pathway trajectory table (if ssGSEA ran)
        - 'cellmix_profiles_csv': Path to cell-type trajectory table (if deconv provided)
        - 'report_html': Path to detailed HTML report (if render_report=True)
        - 'n_samples': Number of samples analyzed
        - 'n_genes_fitted': Number of genes with impulse fits
        - 'n_significant': Number of genes with FDR < 0.05
    
    Examples
    --------
    Basic usage with minimal parameters:
    
    >>> result = run_temporal_analysis(
    ...     output_dir="results",
    ...     counts="counts.csv",
    ...     metadata="metadata.csv"
    ... )
    
    With gene filtering and deconvolution:
    
    >>> result = run_temporal_analysis(
    ...     output_dir="results",
    ...     counts="counts.csv",
    ...     metadata="metadata.csv",
    ...     treatment_level="Treatment",
    ...     genes_list=["GENE1", "GENE2", "GENE3"],
    ...     deconv_csv="cell_proportions.csv"
    ... )
    
    Using a gene list file:
    
    >>> result = run_temporal_analysis(
    ...     output_dir="results",
    ...     counts="counts.csv",
    ...     metadata="metadata.csv",
    ...     genes_list="priority_genes.txt"
    ... )
    
    Notes
    -----
    - All file paths can be absolute or relative to the current working directory
    - If input_dir is provided, relative paths are resolved relative to input_dir
    - The function creates checkpoints during long-running analyses
    - Requires gseapy package for pathway analysis (install with: pip install gseapy)
    - For PhenoPath pseudotime estimation, R and rpy2 are required
    
    See Also
    --------
    The output directory will contain:
    - pseudotime.csv: Sample-level temporal coordinates
    - temporal_gene_fits.tsv: Per-gene impulse model results with statistics
    - temporal_qc.tsv: Quality control metrics per gene
    - pathway_temporal_summary.csv: Pathway-pseudotime correlations
    - report_detailed.html: Comprehensive HTML report with all results
    - Multiple PNG figures for visualization
    """

    gene_col: str = 'gene'
    time_col: str = ''
    covariate: str = 'condition'
    extra_covariates: str = ''
    assume_log: bool = False
    use_phenopath: bool = True
    phenopath_thin: int = 5
    phenopath_maxiter: int = 500
    max_iter_nonlin: int = 800
    prefilter_rho: float = 0.2
    prefilter_min_var: float = 0.0
    max_genes: int = 20000
    n_jobs: int = 12
    checkpoint_every: int = 50
    progress_every: int = 100
    precheck_only: bool = False
    verbose: bool = False
    genes_list_col: str = ''
    signature_from_gene_list: bool = False
    restrict_pathways_by_gene_list: bool = False
    run_ssgsea: bool = True
    gene_sets: str = 'Hallmark'
    organism: str = 'Human'
    ssgsea_report: str = ''
    make_figures: bool = True
    top_n: int = 12
    render_report: bool = True,
    plot_genes_top_n: int = 24
    plot_genes_rank_by: str = 'fdr'
    save_gene_plots: bool = True
    seed: int = 42
    # Prevent BLAS oversubscription
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(var, "1")
    
    # Auto-detect CPU count and cap n_jobs appropriately for Linux servers
    try:
        import multiprocessing as mp
        cpu_count = mp.cpu_count()
        if n_jobs > cpu_count:
            warnings.warn(f"n_jobs={n_jobs} exceeds CPU count ({cpu_count}). Capping to {cpu_count}.")
            n_jobs = min(n_jobs, cpu_count)
        elif n_jobs == -1:
            n_jobs = cpu_count
    except Exception:
        pass  # Use provided n_jobs if detection fails

    np.random.seed(seed)

    out_dir = Path(output_dir)
    ensure_dir(out_dir)
    (out_dir / "RUNNING").write_text("started")
    _emit_temporal_log(workflow_logger, event_loop, "info", "Starting temporal analysis pipeline")

    # Log CPU info for debugging parallel processing issues
    if n_jobs > 1:
        try:
            import multiprocessing as mp
            cpu_count = mp.cpu_count()
            log(f"Parallel processing enabled: n_jobs={n_jobs}, CPU count={cpu_count}")
        except Exception:
            pass

    # ---- Resolve input file paths (allow relative to input_dir if provided) ----
    counts_path = Path(counts)
    if not counts_path.exists() and input_dir is not None:
        in_dir = Path(input_dir)
        counts_path = in_dir / counts
    meta_path = Path(metadata)
    if not meta_path.exists() and input_dir is not None:
        if 'in_dir' not in locals():
            in_dir = Path(input_dir)
        meta_path = in_dir / metadata

    # ---- Auto-detect gene column from genes_list CSV if applicable ----
    if genes_list and isinstance(genes_list, str) and genes_list.lower().endswith('.csv'):
        # Resolve genes_list path
        log(f"Resolving genes_list path: {genes_list}")
        gl_path_check = Path(genes_list)
        if not gl_path_check.exists() and input_dir is not None:
            gl_path_check = Path(input_dir) / genes_list
        
        if gl_path_check.exists():
            log("Detecting gene column from genes_list CSV using LLM...")
            detected_gene_col = detect_gene_column_with_llm(genes_list, input_dir)
            if detected_gene_col:
                gene_col = detected_gene_col
                log(f"Detected gene column: {gene_col}")
            else:
                log(f"Could not detect gene column automatically, using default: {gene_col}")

    # ---- Load & harmonize ----
    log("Loading counts & metadata…")
    _emit_temporal_log(workflow_logger, event_loop, "info", "Loading counts and metadata")
    expr_counts = read_counts(counts_path, gene_col)
    meta = read_metadata(meta_path)
    expr_counts, meta = harmonize(expr_counts, meta)
    log(f"Counts shape: genes={expr_counts.shape[0]}, samples={expr_counts.shape[1]}")
    _emit_temporal_log(
        workflow_logger, event_loop, "info",
        f"Loaded {expr_counts.shape[0]} genes × {expr_counts.shape[1]} samples"
    )

    # ---- Auto-detect time column from metadata CSV if not provided ----
    if not time_col:
        log("Detecting time column from metadata CSV using LLM...")
        _emit_temporal_log(workflow_logger, event_loop, "info", "Detecting time column (LLM)")
        detected_time_col = detect_time_column_with_llm(metadata, input_dir)
        if detected_time_col:
            time_col = detected_time_col
            log(f"Detected time column: {time_col}")
            _emit_temporal_log(workflow_logger, event_loop, "info", f"Using time column: {time_col}")
        else:
            log("Could not detect time column automatically. Will estimate pseudotime instead.")
            _emit_temporal_log(workflow_logger, event_loop, "info", "No time column — estimating pseudotime")

    # ---- Transform ----
    log("Normalizing (CPM→log1p)" if not assume_log else "Assuming log-scale input")
    expr_log = expr_counts if assume_log else cpm_log1p(expr_counts)

    # ---- Pseudotime or provided time ----
    extras = [s.strip() for s in extra_covariates.split(",") if s.strip()]
    if time_col and time_col in meta.columns and pd.api.types.is_numeric_dtype(meta[time_col]):
        log(f"Using provided time: {time_col}")
        _emit_temporal_log(workflow_logger, event_loop, "info", f"Using provided time axis: {time_col}")
        z = meta[time_col].astype(float)
        z = (z - z.min()) / (z.max() - z.min() + 1e-12)
        elbo = None
    else:
        if use_phenopath:
            log("Estimating pseudotime via phenopath…")
            _emit_temporal_log(workflow_logger, event_loop, "info", "Estimating pseudotime (PhenoPath)")
            z, elbo = try_phenopath(expr_log, meta, covariate, phenopath_thin, phenopath_maxiter)
        else:
            log("Estimating pseudotime via PCA PC1…")
            _emit_temporal_log(workflow_logger, event_loop, "info", "Estimating pseudotime (PCA)")
            z, elbo = pseudotime_pca(expr_log, seed=seed), None

    z.to_frame("pseudotime").to_csv(out_dir / "pseudotime.csv")
    log("Wrote pseudotime.csv")

    # ---- ELBO plot (if phenopath) ----
    if elbo is not None and len(elbo) > 0 and make_figures:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(np.arange(1, len(elbo) + 1), elbo, marker="o")
        ax.set_xlabel("Iteration (thinned)")
        ax.set_ylabel("ELBO")
        ax.set_title("PhenoPath ELBO")
        fig.tight_layout()
        fig.savefig(out_dir / "elbo.png")
        plt.close(fig)

    if precheck_only:
        (out_dir / "DONE").write_text("precheck_only")
        log("Precheck complete.")
        return {
            "status": "precheck_only",
            "pseudotime_csv": str(out_dir / "pseudotime.csv"),
            "output_dir": str(out_dir),
        }

    # ---- Optional targeted gene list ----
    if genes_list:
        # Handle both list/array and file path
        if isinstance(genes_list, list):
            # Direct list of genes provided
            gene_list_genes = [str(g).strip() for g in genes_list if g and str(g).strip()]
            log(f"Using provided gene list (array) with {len(gene_list_genes)} genes.")
        else:
            # File path provided
            gl_path = Path(genes_list)
            if gl_path.exists():
                gene_list_genes = [g.strip() for g in gl_path.read_text().splitlines() if g.strip()]
                if genes_list_col:
                    try:
                        df = pd.read_csv(gl_path, sep=None, engine="python")
                        if genes_list_col in df.columns:
                            gene_list_genes = df[genes_list_col].astype(str).tolist()
                    except Exception:
                        pass
                log(f"Loaded {len(gene_list_genes)} genes from file: {gl_path}")
            else:
                warnings.warn(f"genes_list path not found: {gl_path}")
                gene_list_genes = []
        
        # Process the gene list (remove duplicates and match against expression matrix)
        if gene_list_genes:
            gene_list_genes = list(dict.fromkeys(gene_list_genes))  # Remove duplicates while preserving order
            matched = [g for g in gene_list_genes if g in expr_log.index]
            if matched:
                expr_log = expr_log.loc[pd.Index(matched)]
                log(f"Restricting impulse fitting to {len(matched)} genes from genes_list (matched {len(matched)}/{len(gene_list_genes)}).")
            else:
                warnings.warn(f"No genes from genes_list found in expression matrix ({len(gene_list_genes)} genes provided). Proceeding normally.")

    # ---- Prefilter by variance and |rho| ----
    log("Prefiltering genes for impulse fitting…")
    _emit_temporal_log(workflow_logger, event_loop, "info", "Prefiltering genes for impulse fitting")
    var = expr_log.var(axis=1)
    rho = expr_log.apply(lambda r: r.corr(z, method="spearman"), axis=1)
    keep = var.ge(prefilter_min_var) & rho.abs().ge(prefilter_rho)
    kept = expr_log.index[keep]
    if max_genes > 0 and kept.size > max_genes:
        kept = rho.loc[kept].abs().sort_values(ascending=False).head(max_genes).index
    expr_fit = expr_log.loc[kept]
    log(f"Prefilter kept {expr_fit.shape[0]} genes (of {expr_log.shape[0]}).")
    _emit_temporal_log(
        workflow_logger, event_loop, "info",
        f"Prefilter: {expr_fit.shape[0]} genes kept for impulse fitting"
    )

    # ---- Impulse fits ----
    t = z.loc[expr_log.columns].values.astype(float)
    results = []
    qc_rows = []
    items = list(expr_fit.iterrows())
    total = len(items)

    if n_jobs > 1 and Parallel is not None and delayed is not None:
        log(f"Fitting impulses in parallel (n_jobs={n_jobs})…")
        _emit_temporal_log(
            workflow_logger, event_loop, "info",
            f"Fitting impulse models for {total} genes (parallel, n_jobs={n_jobs})"
        )
        chunk = max(50, checkpoint_every)
        use_parallel = True
        
        for i in range(0, total, chunk):
            batch = items[i : i + chunk]
            if use_parallel:
                try:
                    # Use 'loky' backend explicitly for better Linux compatibility
                    # 'loky' is more robust than 'multiprocessing' on Linux systems
                    outs = Parallel(
                        n_jobs=n_jobs, 
                        backend='loky',  # Explicit backend for Linux compatibility
                        prefer="processes",
                        verbose=10 if verbose else 0
                    )(
                        delayed(fit_impulse_single)(g, t, r.values.astype(float), max_iter_nonlin)
                        for g, r in batch
                    )
                except Exception as e:
                    warnings.warn(f"Parallel processing failed: {e}. Falling back to serial processing.")
                    log("Falling back to serial processing due to parallel processing error.")
                    use_parallel = False
                    # Process this batch serially
                    outs = []
                    for g, r in batch:
                        _, imp, lin, qc = fit_impulse_single(g, t, r.values.astype(float), max_iter_nonlin)
                        outs.append((g, imp, lin, qc))
            else:
                # Serial processing fallback
                outs = []
                for g, r in batch:
                    _, imp, lin, qc = fit_impulse_single(g, t, r.values.astype(float), max_iter_nonlin)
                    outs.append((g, imp, lin, qc))
            for g, imp, lin, qc in outs:
                patt, tpk, tvl = ("indeterminate", None, None)
                if imp["ok"]:
                    grid = np.linspace(t.min(), t.max(), 200)
                    yfit_grid = impulse_fn(grid, *imp["params"])
                    dy = np.gradient(yfit_grid, grid)
                    eps = np.nanmax(np.abs(yfit_grid)) * 1e-3 + 1e-6
                    dy_s = np.where(np.abs(dy) < eps, 0, np.sign(dy))
                    peaks = [k for k in range(1, len(dy_s)) if dy_s[k - 1] > 0 and dy_s[k] < 0]
                    valleys = [k for k in range(1, len(dy_s)) if dy_s[k - 1] < 0 and dy_s[k] > 0]
                    tpk = float(grid[peaks[0]]) if peaks else None
                    tvl = float(grid[valleys[0]]) if valleys else None
                    if np.all(dy_s >= 0):
                        patt = "monotonic_up"
                    elif np.all(dy_s <= 0):
                        patt = "monotonic_down"
                    elif peaks and valleys:
                        patt = "biphasic"
                    elif peaks:
                        patt = "up_then_down"
                    elif valleys:
                        patt = "down_then_up"
                p = float(imp["p_vs_linear"]) if imp["ok"] else 1.0
                aic = float(imp["aic"]) if imp["ok"] else np.nan
                results.append([g, tpk, tvl, p, patt, aic, qc["r2_impulse"]])
                qc_rows.append({"gene": g, **qc})
            # checkpoint
            pd.DataFrame(
                results,
                columns=["gene_id", "time_of_peak", "time_of_valley", "p_value", "pattern", "aic", "r2_impulse"],
            ).set_index("gene_id").to_csv(out_dir / "temporal_gene_fits.tsv", sep="\t")
            pd.DataFrame(qc_rows).set_index("gene").to_csv(out_dir / "temporal_qc.tsv", sep="\t")
            log(f"Processed {min(i + chunk, total)}/{total} genes; checkpoint written.")
    else:
        log("Fitting impulses serially…")
        _emit_temporal_log(workflow_logger, event_loop, "info", f"Fitting impulse models for {total} genes (serial)")
        for idx, (g, r) in enumerate(items, 1):
            _, imp, lin, qc = fit_impulse_single(g, t, r.values.astype(float), max_iter_nonlin)
            patt, tpk, tvl = ("indeterminate", None, None)
            if imp["ok"]:
                grid = np.linspace(t.min(), t.max(), 200)
                yfit_grid = impulse_fn(grid, *imp["params"])
                dy = np.gradient(yfit_grid, grid)
                eps = np.nanmax(np.abs(yfit_grid)) * 1e-3 + 1e-6
                dy_s = np.where(np.abs(dy) < eps, 0, np.sign(dy))
                peaks = [k for k in range(1, len(dy_s)) if dy_s[k - 1] > 0 and dy_s[k] < 0]
                valleys = [k for k in range(1, len(dy_s)) if dy_s[k - 1] < 0 and dy_s[k] > 0]
                tpk = float(grid[peaks[0]]) if peaks else None
                tvl = float(grid[valleys[0]]) if valleys else None
                if np.all(dy_s >= 0):
                    patt = "monotonic_up"
                elif np.all(dy_s <= 0):
                    patt = "monotonic_down"
                elif peaks and valleys:
                    patt = "biphasic"
                elif peaks:
                    patt = "up_then_down"
                elif valleys:
                    patt = "down_then_up"
            p = float(imp["p_vs_linear"]) if imp["ok"] else 1.0
            aic = float(imp["aic"]) if imp["ok"] else np.nan
            results.append([g, tpk, tvl, p, patt, aic, qc["r2_impulse"]])
            qc_rows.append({"gene": g, **qc})
            if idx % max(1, progress_every) == 0:
                log(f"Processed {idx}/{total} genes…")
            if idx % max(1, checkpoint_every) == 0:
                pd.DataFrame(
                    results,
                    columns=["gene_id", "time_of_peak", "time_of_valley", "p_value", "pattern", "aic", "r2_impulse"],
                ).set_index("gene_id").to_csv(out_dir / "temporal_gene_fits.tsv", sep="\t")
                pd.DataFrame(qc_rows).set_index("gene").to_csv(out_dir / "temporal_qc.tsv", sep="\t")

    # ---- Final write and FDR ----
    log("Writing final results (with FDR)…")
    fits = pd.DataFrame(
        results, columns=["gene_id", "time_of_peak", "time_of_valley", "p_value", "pattern", "aic", "r2_impulse"]
    ).set_index("gene_id")
    if multipletests is not None and not fits.empty:
        mask = fits["p_value"].notna().values
        q = np.full(fits.shape[0], np.nan)
        if mask.sum() > 0:
            q[mask] = multipletests(fits.loc[mask, "p_value"].values, method="fdr_bh")[1]
        fits["p_adj"] = q
    fits.to_csv(out_dir / "temporal_gene_fits.tsv", sep="\t")
    pd.DataFrame(qc_rows).set_index("gene").to_csv(out_dir / "temporal_qc.tsv", sep="\t")
    log(f"Final rows written: {fits.shape[0]}")
    n_sig = int((fits["p_adj"].astype(float) < 0.05).sum()) if "p_adj" in fits.columns else 0
    _emit_temporal_log(
        workflow_logger, event_loop, "info",
        f"Impulse fits done: {fits.shape[0]} genes, {n_sig} significant (FDR<0.05)"
    )

    # ---- Gene gallery & per-gene PNGs (always) ----
    try:
        fits_df = pd.read_csv(out_dir / "temporal_gene_fits.tsv", sep="\t")
        top_genes = _rank_genes_for_gallery(
            fits=fits_df, by=plot_genes_rank_by, top_n=plot_genes_top_n
        )
        if len(top_genes) > 0:
            color_by = None
            if covariate and covariate in meta.columns:
                color_by = meta.set_index("sample_id")[covariate]

            genes_dir = out_dir / "genes"
            if save_gene_plots:
                ensure_dir(genes_dir)
                for g in top_genes:
                    plot_gene_trajectory(
                        gene=g,
                        expr_log=expr_log,
                        z=z,
                        out_png=genes_dir / f"{g}.png",
                        color_by=color_by,
                    )

            n = len(top_genes)
            cols = 6 if n >= 12 else max(3, min(4, n))
            rows = int(np.ceil(n / cols))
            fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.0, rows * 2.2), dpi=140)
            if not isinstance(axes, np.ndarray):
                axes = np.array([axes])
            axes = axes.ravel()
            for i, g in enumerate(top_genes):
                ax = axes[i]
                if g in expr_log.index:
                    y = expr_log.loc[g, z.index].values.astype(float)
                    x = z.values.astype(float)
                    ax.scatter(x, y, s=10, alpha=0.7)
                    xs, ys = _smooth_trend(x, y)
                    ax.plot(xs, ys, linewidth=1.6)
                    ax.set_title(g, fontsize=9)
                    ax.set_xticks([0, 0.5, 1.0])
                ax.grid(True, alpha=0.15)
            for j in range(i + 1, rows * cols):
                axes[j].axis("off")
            for ax in axes:
                ax.tick_params(labelsize=8)
            fig.tight_layout()
            grid_png = out_dir / "temporal_genes_topN.png"
            fig.savefig(grid_png)
            plt.close(fig)
            log(f"Wrote gene gallery: {grid_png.name}")
    except Exception as e:
        warnings.warn(f"Gene plotting failed: {e}")

    # ---- Deconvolution (load before pathway export) ----
    deconv_path = deconv_csv
    if not deconv_path and input_dir is not None:
        # best-effort auto-detect in input directory
        in_dir = Path(input_dir)
        candidates = list(in_dir.glob("*bisque*proportion*.csv")) + list(in_dir.glob("*proportion*.csv"))
        if candidates:
            deconv_path = str(candidates[0])
            log(f"Auto-detected deconv_csv: {deconv_path}")
            _emit_temporal_log(workflow_logger, event_loop, "info", f"Auto-detected deconv: {Path(deconv_path).name}")

    deconv = None
    if deconv_path:
        dpath = Path(deconv_path)
        if not dpath.exists() and input_dir is not None:
            if 'in_dir' not in locals():
                in_dir = Path(input_dir)
            dpath = in_dir / deconv_path
        if dpath.exists():
            try:
                deconv = read_table_auto(dpath)
            except Exception:
                deconv = None
        if deconv is None:
            warnings.warn("Deconv CSV provided but failed to read.")
        else:
            log(f"Deconv table loaded: {deconv.shape[0]} rows × {deconv.shape[1]} cols")
            _emit_temporal_log(
                workflow_logger, event_loop, "info",
                f"Loaded deconvolution: {deconv.shape[0]} samples × {deconv.shape[1]} cell types"
            )

    # ---- ssGSEA + pathway summary (native first, then report fallback) ----
    ss = None
    pw_summary = None

    if run_ssgsea:
        # Check if gseapy is available
        from .core import gp
        if gp is None:
            log("WARNING: gseapy is not installed. Install it with: pip install gseapy")
            _emit_temporal_log(workflow_logger, event_loop, "warning", "gseapy not installed — skipping ssGSEA")
            log("Skipping ssGSEA analysis. Pathway results will not be generated.")
        else:
            log("Running ssGSEA…")
            _emit_temporal_log(workflow_logger, event_loop, "info", f"Running ssGSEA ({gene_sets}, {organism})")
            log(f"Using gene sets: {gene_sets}, organism: {organism}")
            ssg_dir = out_dir / "ssgsea"
            ss = run_ssgsea_func(expr_log, gene_sets, ssg_dir, organism=organism)
            if ss is not None:
                log(f"ssGSEA completed successfully. Found {ss.shape[0]} pathways for {ss.shape[1]} samples.")
                _emit_temporal_log(
                    workflow_logger, event_loop, "info",
                    f"ssGSEA done: {ss.shape[0]} pathways × {ss.shape[1]} samples"
                )
                pw_summary = pathway_temporal_summary(ss, z)
                if pw_summary is not None and not pw_summary.empty:
                    pw_summary.to_csv(out_dir / "pathway_temporal_summary.csv")
                    log("Wrote pathway_temporal_summary.csv")
                else:
                    log("WARNING: Pathway temporal summary is empty.")
            else:
                log("WARNING: ssGSEA returned None. Check the ssgsea/ directory for error messages.")

    if (ss is None) or (pw_summary is None) or pw_summary.empty:
        rep_path = find_ssgsea_report_in(out_dir)
        if rep_path is not None and rep_path.exists():
            log(f"Found ssGSEA report: {rep_path}")
            ss_from_report = ssgsea_report_to_matrix(rep_path)
            if ss_from_report is not None and not ss_from_report.empty:
                ss_cols = ss_from_report.columns.astype(str)
                z_idx = z.index.astype(str)
                common = [s for s in ss_cols if s in z_idx]
                if len(common) >= 3:
                    ss = ss_from_report.loc[:, common]
                    log(f"Aligned ssGSEA report to {len(common)} samples.")
                    pw_summary = pathway_temporal_summary(ss, z)
                    if pw_summary is not None and not pw_summary.empty:
                        pw_summary.to_csv(out_dir / "pathway_temporal_summary.csv")
                        log("Wrote pathway_temporal_summary.csv (from report).")
                else:
                    warnings.warn(
                        "ssGSEA report has insufficient overlap with pseudotime sample IDs; skipping pathway summary."
                    )

    # ---- Export standardized trajectory tables (pathways & cellmix) ----
    pw_long, cm_long = export_trajectory_tables(z, ss, deconv, out_dir)

    # ---- Figures (ALWAYS) & per-entity PNGs ----
    if covariate and covariate in meta.columns:
        save_boxplot_pseudotime(z, meta, covariate, out_dir / "pseudotime_by_covariate.png")

    if pw_summary is not None and not pw_summary.empty:
        save_bar_top_pathways(pw_summary, out_dir / "pseudotime_vs_pathways_top15.png", top=15)

    if pw_long is not None:
        grid_plot_trajectories(
            pw_long,
            f"Temporal Hallmark trajectories (top {top_n})",
            out_dir / "temporal_pathways_top12.png",
            top_n=top_n,
        )
        per_entity_pngs(pw_long, out_dir / "pathways", "pathway")

    if cm_long is not None:
        grid_plot_trajectories(
            cm_long,
            f"Temporal cell-type trajectories (top {top_n})",
            out_dir / "temporal_cellmix_top12.png",
            top_n=top_n,
        )
        per_entity_pngs(cm_long, out_dir / "cellmix", "cellmix")

    if covariate:
        de_and_interaction_scatter(
            expr_log,
            meta,
            z,
            covariate,
            treatment_level,
            extras,
            out_dir / "de_vs_beta.png",
            out_dir / "de_vs_interaction.csv",
        )

    # ---- Detailed HTML report (leave in out_dir to keep links valid) ----
    _emit_temporal_log(workflow_logger, event_loop, "info", "Generating HTML report and figures")
    if render_report:
        used_gene_sets = None
        try:
            if run_ssgsea:
                used_gene_sets = _resolve_gene_sets_for_gseapy(gene_sets, organism=organism)
            elif pw_summary is not None and not pw_summary.empty:
                used_gene_sets = "from_report"

            write_detailed_html_report(
                out_dir=out_dir,
                counts_path=counts_path,
                meta_path=meta_path,
                expr_counts=expr_counts,
                z=z,
                fits_df=fits,
                pw_summary=pw_summary,
                pw_long=pw_long,
                cm_long=cm_long,
                used_gene_sets=used_gene_sets,
                made_elbo=(elbo is not None and len(elbo) > 0),
                made_de_interaction=bool(covariate),
            )
            log("Wrote report_detailed.html")
        except Exception as e:
            warnings.warn(f"Failed to write detailed report: {e}")

    # ---- Build temporal pack (bundle of images) ----
    try:
        build_causal_pack(out_dir, pw_long, cm_long, top_n)  # creates temporal_pack/
    except Exception as e:
        warnings.warn(f"Failed to build temporal pack: {e}")

    (out_dir / "DONE").write_text("ok")
    log("All done.")
    _emit_temporal_log(
        workflow_logger, event_loop, "info",
        f"Temporal analysis complete: {fits.shape[0]} genes fitted, "
        f"{int((fits['p_adj'].astype(float) < 0.05).sum()) if 'p_adj' in fits.columns else 0} significant"
    )

    # Return summary dictionary
    return {
        "status": "ok",
        "output_dir": str(out_dir),
        "pseudotime_csv": str(out_dir / "pseudotime.csv"),
        "gene_fits_tsv": str(out_dir / "temporal_gene_fits.tsv"),
        "qc_tsv": str(out_dir / "temporal_qc.tsv"),
        "pathway_summary_csv": str(out_dir / "pathway_temporal_summary.csv") if pw_summary is not None and not pw_summary.empty else None,
        "pathway_profiles_csv": str(out_dir / "pathway_profiles.csv") if pw_long is not None else None,
        "cellmix_profiles_csv": str(out_dir / "cellmix_profiles.csv") if cm_long is not None else None,
        "report_html": str(out_dir / "report_detailed.html") if render_report else None,
        "n_samples": int(z.shape[0]),
        "n_genes_fitted": int(fits.shape[0]),
        "n_significant": int((fits["p_adj"].astype(float) < 0.05).sum()) if "p_adj" in fits.columns else None,
    }


def get_treatment_level_options(metadata: Union[str, Path]) -> List[str]:
    """
    Get the unique treatment levels from the metadata.
    """
    metadata = pd.read_csv(metadata)
    return list(metadata['Condition'].unique())

def main():
    """CLI entry point - parses arguments and calls run_temporal_analysis()."""
    args = build_parser().parse_args()
    
    # Convert argparse Namespace to keyword arguments
    result = run_temporal_analysis(
        output_dir=args.output_dir,
        counts=args.counts,
        metadata=args.metadata,
        input_dir=args.input_dir,
        gene_col=args.gene_col,
        time_col=args.time_col,
        covariate=args.covariate,
        treatment_level=args.treatment_level,
        extra_covariates=args.extra_covariates,
        assume_log=args.assume_log,
        use_phenopath=args.use_phenopath,
        phenopath_thin=args.phenopath_thin,
        phenopath_maxiter=args.phenopath_maxiter,
        max_iter_nonlin=args.max_iter_nonlin,
        prefilter_rho=args.prefilter_rho,
        prefilter_min_var=args.prefilter_min_var,
        max_genes=args.max_genes,
        n_jobs=args.n_jobs,
        checkpoint_every=args.checkpoint_every,
        progress_every=args.progress_every,
        precheck_only=args.precheck_only,
        verbose=args.verbose,
        genes_list=args.genes_list,
        genes_list_col=args.genes_list_col,
        signature_from_gene_list=args.signature_from_gene_list,
        restrict_pathways_by_gene_list=args.restrict_pathways_by_gene_list,
        run_ssgsea=args.run_ssgsea,
        gene_sets=args.gene_sets,
        organism=args.organism,
        ssgsea_report=args.ssgsea_report,
        deconv_csv=args.deconv_csv,
        make_figures=args.make_figures,
        top_n=args.top_n,
        render_report=args.render_report,
        plot_genes_top_n=args.plot_genes_top_n,
        plot_genes_rank_by=args.plot_genes_rank_by,
        save_gene_plots=args.save_gene_plots,
        seed=args.seed,
    )
    
    return result


if __name__ == "__main__":
    main()