#!/usr/bin/env python3
"""
CRISPR Perturb-seq Pipeline Runner (Stage 0 -> Stage 12)

Features:
- End-to-end execution (0-12)
- Auto-resume from last completed stage via marker files
- nf-core style colored + timed logging
- Per-stage log files (stdout + stderr captured)
- Multi-GSE aggregation option
- Callable API for programmatic use
- scRNA-seq integration pipeline
- Helper functions to discover available parameters
"""

import sys
import subprocess
import time
import os
from pathlib import Path
from typing import List, Optional, Union, Dict, Any

import pandas as pd

PY = sys.executable

# =============================
# Colors (nf-core style)
# =============================
BLUE = "\033[94m"
GREEN = "\033[92m"
RED = "\033[91m"
GRAY = "\033[90m"
RESET = "\033[0m"


def ts():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def log(msg, color=RESET):
    print(f"{color}[{ts()}] {msg}{RESET}", flush=True)


# =============================
# Discovery helpers
# =============================
def list_gse_folders(root: Path):
    return sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith("GSE"))


def detect_gsm_samples(gse_dir: Path):
    samples = set()
    for f in gse_dir.glob("GSM*_barcodes.tsv*"):
        parts = f.name.split("_")
        if len(parts) >= 2 and parts[0].startswith("GSM"):
            samples.add("_".join(parts[:2]))
    return sorted(samples)


def _sample_has_complete_files(gse_dir: Path, gsm_id: str) -> bool:
    """Check if sample has matrix, genes, and barcodes (required for stage0)."""
    def find_one(patterns):
        for pat in patterns:
            hits = list(gse_dir.glob(f"{gsm_id}*{pat}*"))
            if hits:
                return True
        return False
    return (
        find_one(["matrix.mtx", "matrix.mtx.txt"]) and
        find_one(["genes.tsv"]) and
        find_one(["barcodes.tsv"])
    )


# =============================
# Helper Functions for Parameter Discovery
# =============================
def get_available_samples(
    input_gse_dir: Union[str, Path],
    complete_only: bool = True,
) -> List[str]:
    """
    Discover available samples in a GSE directory.
    
    Args:
        input_gse_dir: Path to GSE directory (e.g., "input_data/GSE90546_RAW")
        complete_only: If True (default), return only samples with matrix+genes+barcodes.
            If False, return all samples found via barcodes.tsv (may include incomplete).
    
    Returns:
        List of available sample IDs (e.g., ["GSM2406675_10X001", "GSM2406677_10X005"])
    
    Example:
        >>> from pathlib import Path
        >>> from crispr.run_pipeline import get_available_samples
        >>> 
        >>> samples = get_available_samples("input_data/GSE90546_RAW")
        >>> print(f"Available samples: {samples}")
        >>> 
        >>> # Use with run_pipeline
        >>> run_pipeline(input_gse_dirs=Path("input_data/GSE90546_RAW"), samples=samples[0:2])
    """
    gse_dir = Path(input_gse_dir)
    if not gse_dir.exists():
        raise FileNotFoundError(f"GSE directory not found: {gse_dir}")
    
    samples = detect_gsm_samples(gse_dir)
    if complete_only:
        samples = [s for s in samples if _sample_has_complete_files(gse_dir, s)]
    return samples


def get_metadata_groups(input_gse_dir: Union[str, Path]) -> Dict[str, Any]:
    """
    Extract group information from metadata.csv in a GSE directory.
    
    Args:
        input_gse_dir: Path to GSE directory
    
    Returns:
        Dict with:
        - "available": bool (whether metadata.csv exists)
        - "groups": List of unique group names
        - "group_counts": Dict mapping group names to sample counts
        - "gsm_to_group": Dict mapping GSM IDs to group names
        - "suggested_control": Suggested control group (first group alphabetically)
        - "suggested_disease": Suggested disease groups (remaining groups)
    
    Example:
        >>> from crispr.run_pipeline import get_metadata_groups
        >>> 
        >>> info = get_metadata_groups("input_data/GSE90546_RAW")
        >>> if info["available"]:
        >>>     print(f"Groups: {info['groups']}")
        >>>     print(f"Suggested control: {info['suggested_control']}")
        >>>     print(f"Suggested disease: {info['suggested_disease']}")
        >>> 
        >>> # Use with run_pipeline
        >>> run_pipeline(
        >>>     input_gse_dirs=gse_dir,
        >>>     samples="all",
        >>>     scrna_config={
        >>>         "conditions": {
        >>>             "enabled": True,
        >>>             "control_groups": [info["suggested_control"]],
        >>>             "disease_groups": info["suggested_disease"]
        >>>         }
        >>>     }
        >>> )
    """
    gse_dir = Path(input_gse_dir)
    if not gse_dir.exists():
        raise FileNotFoundError(f"GSE directory not found: {gse_dir}")
    
    metadata_path = gse_dir / "metadata.csv"
    
    if not metadata_path.exists():
        return {
            "available": False,
            "groups": [],
            "group_counts": {},
            "gsm_to_group": {},
            "suggested_control": None,
            "suggested_disease": []
        }
    
    try:
        df = pd.read_csv(metadata_path)
        
        # Normalize column names
        cols_lower = {c.lower(): c for c in df.columns}
        
        # Find GSM and Group columns
        gsm_col = None
        group_col = None
        
        for candidate in ["gsm", "sample", "sample_id", "sample_name"]:
            if candidate in cols_lower:
                gsm_col = cols_lower[candidate]
                break
        
        for candidate in ["group", "condition", "treatment", "type"]:
            if candidate in cols_lower:
                group_col = cols_lower[candidate]
                break
        
        if gsm_col is None or group_col is None:
            return {
                "available": True,
                "error": f"metadata.csv missing required columns. Found: {list(df.columns)}. Need: GSM/Sample and Group/Condition columns",
                "groups": [],
                "group_counts": {},
                "gsm_to_group": {},
                "suggested_control": None,
                "suggested_disease": []
            }
        
        # Extract data
        df[gsm_col] = df[gsm_col].astype(str).str.strip()
        df[group_col] = df[group_col].astype(str).str.strip()
        
        groups = sorted(df[group_col].unique().tolist())
        group_counts = df[group_col].value_counts().to_dict()
        gsm_to_group = dict(zip(df[gsm_col], df[group_col]))
        
        # Suggest control (first alphabetically) and disease (rest)
        suggested_control = groups[0] if groups else None
        suggested_disease = groups[1:] if len(groups) > 1 else []
        
        return {
            "available": True,
            "groups": groups,
            "group_counts": group_counts,
            "gsm_to_group": gsm_to_group,
            "suggested_control": suggested_control,
            "suggested_disease": suggested_disease,
            "gsm_column": gsm_col,
            "group_column": group_col
        }
    
    except Exception as e:
        return {
            "available": True,
            "error": str(e),
            "groups": [],
            "group_counts": {},
            "gsm_to_group": {},
            "suggested_control": None,
            "suggested_disease": []
        }


def get_scrna_config_options() -> Dict[str, Any]:
    """
    Get available options for scRNA pipeline configuration.
    
    Returns:
        Dict with all available configuration options and their descriptions
    
    Example:
        >>> from crispr.run_pipeline import get_scrna_config_options
        >>> 
        >>> options = get_scrna_config_options()
        >>> print("Available integration methods:", options["integration_methods"])
        >>> print("Available CellTypist models:", options["celltypist_models"])
    """
    return {
        "integration_methods": {
            "available": ["none", "harmony", "bbknn", "scvi"],
            "descriptions": {
                "none": "Standard preprocessing without batch correction",
                "harmony": "Harmony batch correction (fast, effective)",
                "bbknn": "Batch-balanced k-nearest neighbors",
                "scvi": "Variational autoencoder-based integration (most powerful)"
            },
            "requirements": {
                "harmony": "harmonypy package",
                "bbknn": "scikit-misc package",
                "scvi": "scvi-tools package"
            }
        },
        "annotation_engines": {
            "available": ["CellTypist", "CellO", "scVI/scANVI (if available)"],
            "descriptions": {
                "CellTypist": "Pre-trained model-based annotation (fast, accurate)",
                "CellO": "Ontology-based annotation",
                "scVI/scANVI (if available)": "Deep learning-based annotation"
            }
        },
        "celltypist_models": {
            "immune": [
                "Immune_All_Low.pkl",  # Default
                "Immune_All_High.pkl",
                "Developing_Immune_System.pkl"
            ],
            "tissue_specific": [
                "Cells_Intestinal_Tract.pkl",
                "Cells_Lung_Airway.pkl",
                "Pan_Fetal_Human.pkl"
            ],
            "descriptions": {
                "Immune_All_Low.pkl": "General immune cells, low resolution (default)",
                "Immune_All_High.pkl": "General immune cells, high resolution",
                "Developing_Immune_System.pkl": "Developing immune system cells"
            }
        },
        "trajectory_modes": {
            "available": ["paga_dpt_velocity_if_possible", "paga_dpt"],
            "descriptions": {
                "paga_dpt_velocity_if_possible": "PAGA + DPT + RNA velocity if available (default)",
                "paga_dpt": "PAGA + DPT only (no velocity)"
            }
        },
        "ml_models": {
            "available": ["xgb", "rf", "lr", "mlp"],
            "descriptions": {
                "xgb": "XGBoost (gradient boosting)",
                "rf": "Random Forest",
                "lr": "Logistic Regression",
                "mlp": "Multi-layer Perceptron (neural network)"
            }
        },
        "qc_defaults": {
            "min_genes": 200,
            "min_cells": 3,
            "max_mt_pct": 20.0,
            "max_genes": 6000,
            "expected_doublet_rate": 0.06
        },
        "preprocess_defaults": {
            "hvg_n": 3000,
            "n_pcs": 50,
            "neighbors_k": 15,
            "leiden_resolution": 0.8
        }
    }


def discover_pipeline_inputs(input_gse_dir: Union[str, Path]) -> Dict[str, Any]:
    """
    Comprehensive discovery of all available pipeline inputs and metadata.
    
    This is a convenience function that combines get_available_samples() and 
    get_metadata_groups() along with configuration options.
    
    Args:
        input_gse_dir: Path to GSE directory
    
    Returns:
        Dict containing:
        - samples: List of available sample IDs
        - metadata: Metadata group information
        - config_options: Available configuration options
    
    Example:
        >>> from pathlib import Path
        >>> from crispr.run_pipeline import discover_pipeline_inputs
        >>> 
        >>> info = discover_pipeline_inputs("input_data/GSE90546_RAW")
        >>> 
        >>> print(f"Available samples ({len(info['samples'])}):")
        >>> for sample in info['samples']:
        >>>     print(f"  - {sample}")
        >>> 
        >>> if info['metadata']['available']:
        >>>     print(f"\nMetadata groups: {info['metadata']['groups']}")
        >>>     print(f"Suggested control: {info['metadata']['suggested_control']}")
        >>>     print(f"Suggested disease: {info['metadata']['suggested_disease']}")
        >>> 
        >>> print(f"\nIntegration methods: {info['config_options']['integration_methods']['available']}")
        >>> 
        >>> # Use discovered values
        >>> run_pipeline(
        >>>     input_gse_dirs=Path(input_gse_dir),
        >>>     samples=info['samples'],  # All discovered samples
        >>>     scrna_config={
        >>>         "conditions": {
        >>>             "enabled": True,
        >>>             "control_groups": [info['metadata']['suggested_control']],
        >>>             "disease_groups": info['metadata']['suggested_disease']
        >>>         }
        >>>     }
        >>> )
    """
    gse_dir = Path(input_gse_dir)
    
    return {
        "gse_name": gse_dir.name,
        "gse_path": str(gse_dir.resolve()),
        "samples": get_available_samples(gse_dir),
        "metadata": get_metadata_groups(gse_dir),
        "config_options": get_scrna_config_options()
    }


# =============================
# Command runner
# =============================
def run_cmd(cmd: List, stage: str, marker: Path, log_dir: Path):
    if marker.exists():
        log(f"[SKIP] {stage} already complete", GREEN)
        return

    cmd = [str(c) for c in cmd]
    log(f"[{stage}] STARTED", BLUE)
    log(" ".join(cmd), GRAY)

    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{stage.lower()}.log"

    t0 = time.time()
    with open(log_path, "w") as fh:
        p = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT)
    dt = time.time() - t0

    if p.returncode != 0:
        log(f"[{stage}] FAILED after {dt:.1f}s (see {log_path})", RED)
        with open(log_path) as f:
            tail = "".join(f.readlines()[-20:])
        raise RuntimeError(f"CRISPR {stage} failed (exit={p.returncode}): {log_path}\n{tail}")

    marker.touch()
    log(f"[{stage}] DONE in {dt:.1f}s", GREEN)


# =============================
# Pipeline Execution
# =============================
def run_pipeline(
    input_gse_dirs: Union[Path, List[Path]],
    samples: Union[str, List[str]] = "all",
    output_dir: Optional[Path] = None,
    full_bn: bool = True,
    models: str = "xgb,rf",
    parallel_training: bool = True,
    run_scrna: bool = True,
    scrna_config: Optional[Dict[str, Any]] = None,
    generate_report: bool = True,
) -> None:
    """
    Run the CRISPR Perturb-seq pipeline programmatically.

    Args:
        input_gse_dirs: Path or list of paths to GSE directories containing input data
        samples: Sample selection - "all", list of sample IDs (e.g. ["GSM2406675_10X001"]), 
                 or comma-separated string (e.g. "GSM2406675_10X001,GSM2406677_10X005")
        output_dir: Output directory for results (defaults to "./processed" in current working directory)
        full_bn: Enable full Bayesian network mode in Stage 10
        models: Comma-separated list of models to train in Stage 6 (e.g. "xgb,rf,lr,mlp")
        parallel_training: Enable parallel model training in Stage 6
        run_scrna: Run scRNA-seq integration pipeline after CRISPR pipeline completes
        scrna_config: Optional dict to customize scRNA pipeline configuration
                      (See SCRNA_CONFIG_GUIDE.md for available options)
        generate_report: Generate HTML report after pipeline completes (default True).
                         Requires OPENAI_API_KEY in .env for LLM-driven interpretations.

    Output Structure:
        When processing multiple GSEs, output is organized by GSE to avoid sample ID conflicts:
        
        output_dir/
        ├── GSE90546/              # GSE-specific directory
        │   ├── GSM2406675_10X001/ # Sample directory
        │   │   ├── .markers/
        │   │   ├── logs/
        │   │   ├── tables/
        │   │   ├── figures/
        │   │   └── ...
        │   └── GSM2406677_10X005/
        │   └── scrna_results/     # scRNA pipeline output (if run_scrna=True)
        └── GSE12345/              # Another GSE
            └── GSM2406675_10X001/ # Same sample ID, different GSE (no conflict)

    Example:
        >>> from pathlib import Path
        >>> from crispr.run_pipeline import run_pipeline, discover_pipeline_inputs
        >>> 
        >>> # Discover available options first
        >>> gse_dir = Path("input_data/GSE90546_RAW")
        >>> info = discover_pipeline_inputs(gse_dir)
        >>> print(f"Available samples: {info['samples']}")
        >>> 
        >>> # Single GSE (output to ./processed by default)
        >>> run_pipeline(input_gse_dirs=gse_dir, samples=["GSM2406675_10X001"], full_bn=True)
        >>> 
        >>> # Multiple GSEs with custom output directory
        >>> gse_dirs = [
        >>>     Path("input_data/GSE90546_RAW"),
        >>>     Path("input_data/GSE12345_RAW")
        >>> ]
        >>> run_pipeline(input_gse_dirs=gse_dirs, samples="all", output_dir=Path("results"))
    
    Raises:
        ValueError: If any parameter value is invalid
        FileNotFoundError: If GSE directory doesn't exist
    """
    # =============================
    # Parameter Validation
    # =============================
    
    # Setup paths
    # Stage scripts are always in the same directory as this file
    base_dir = Path(__file__).resolve().parent
    
    # Output directory defaults to ./processed in current working directory
    if output_dir is None:
        output_dir = Path.cwd() / "processed"
    else:
        output_dir = Path(output_dir).resolve()
    
    output_dir.mkdir(parents=True, exist_ok=True)

    # Normalize input_gse_dirs to list
    if isinstance(input_gse_dirs, (str, Path)):
        input_gse_dirs = [Path(input_gse_dirs)]
    else:
        input_gse_dirs = [Path(g) for g in input_gse_dirs]

    # Validate GSE directories
    for gse in input_gse_dirs:
        if not gse.exists():
            raise FileNotFoundError(
                f"❌ Error: GSE directory not found: {gse}\n\n"
                f"Please provide a valid path to a GSE directory containing input data.\n"
                f"Example structure:\n"
                f"  input_data/GSE90546_RAW/\n"
                f"    ├── GSM2406675_10X001_barcodes.tsv.gz\n"
                f"    ├── GSM2406675_10X001_genes.tsv.gz\n"
                f"    ├── GSM2406675_10X001_matrix.mtx.gz\n"
                f"    └── metadata.csv (optional)\n"
            )
    
    # Validate models parameter
    valid_models = ["xgb", "rf", "lr", "mlp"]
    model_list = [m.strip().lower() for m in models.split(",")]
    invalid_models = [m for m in model_list if m not in valid_models]
    
    if invalid_models:
        raise ValueError(
            f"❌ Error: Invalid model(s) specified: {invalid_models}\n\n"
            f"Valid models are: {valid_models}\n"
            f"You provided: {model_list}\n\n"
            f"Example usage:\n"
            f"  models='xgb,rf'  (default)\n"
            f"  models='xgb,rf,lr,mlp'  (all models)\n"
        )
    
    # Validate scrna_config if provided
    if scrna_config is not None:
        if not isinstance(scrna_config, dict):
            raise ValueError(
                f"❌ Error: scrna_config must be a dictionary, got {type(scrna_config).__name__}\n\n"
                f"Example usage:\n"
                f"  scrna_config={{\n"
                f"      'integration': {{'method': 'harmony'}},\n"
                f"      'annotation': {{'engine': 'CellTypist'}}\n"
                f"  }}\n\n"
                f"See SCRNA_CONFIG_GUIDE.md for full documentation.\n"
            )
        
        # Validate integration method if specified
        if "integration" in scrna_config and "method" in scrna_config["integration"]:
            method = scrna_config["integration"]["method"]
            valid_methods = ["none", "harmony", "bbknn", "scvi"]
            if method not in valid_methods:
                raise ValueError(
                    f"❌ Error: Invalid integration method: '{method}'\n\n"
                    f"Valid methods are: {valid_methods}\n\n"
                    f"Example usage:\n"
                    f"  scrna_config={{'integration': {{'method': 'harmony'}}}}\n"
                )
        
        # Validate annotation engine if specified
        if "annotation" in scrna_config and "engine" in scrna_config["annotation"]:
            engine = scrna_config["annotation"]["engine"]
            valid_engines = ["CellTypist", "CellO", "scVI/scANVI (if available)"]
            if engine not in valid_engines:
                raise ValueError(
                    f"❌ Error: Invalid annotation engine: '{engine}'\n\n"
                    f"Valid engines are: {valid_engines}\n\n"
                    f"Example usage:\n"
                    f"  scrna_config={{'annotation': {{'engine': 'CellTypist'}}}}\n"
                )
        
        # Validate trajectory mode if specified
        if "trajectory" in scrna_config and "mode" in scrna_config["trajectory"]:
            mode = scrna_config["trajectory"]["mode"]
            valid_modes = ["paga_dpt_velocity_if_possible", "paga_dpt"]
            if mode not in valid_modes:
                raise ValueError(
                    f"❌ Error: Invalid trajectory mode: '{mode}'\n\n"
                    f"Valid modes are: {valid_modes}\n\n"
                    f"Example usage:\n"
                    f"  scrna_config={{'trajectory': {{'mode': 'paga_dpt'}}}}\n"
                )

    # Process each GSE
    for gse in input_gse_dirs:
        # Detect available samples
        available_samples = detect_gsm_samples(gse)
        if not available_samples:
            log(f"[WARN] No samples found in {gse.name}", RED)
            continue

        # Determine which samples to process
        if samples == "all":
            chosen_samples = available_samples
        elif isinstance(samples, str):
            # Comma-separated string
            chosen_samples = [s.strip() for s in samples.split(",")]
        else:
            # List of samples
            chosen_samples = list(samples)

        # Validate selected samples exist
        invalid_samples = [s for s in chosen_samples if s not in available_samples]
        if invalid_samples:
            raise ValueError(
                f"❌ Error: Invalid sample(s) specified for {gse.name}: {invalid_samples}\n\n"
                f"Available samples in {gse.name}:\n"
                + "\n".join(f"  - {s}" for s in available_samples)
                + "\n\n"
                f"You can discover available samples using:\n"
                f"  from crispr.run_pipeline import get_available_samples\n"
                f"  samples = get_available_samples('{gse}')\n"
            )

        # Filter to valid samples
        chosen_samples = [s for s in chosen_samples if s in available_samples]
        if not chosen_samples:
            log(f"[WARN] No valid samples selected for {gse.name}", RED)
            continue

        log(f"Selected GSE={gse.name} samples={chosen_samples}", BLUE)
        
        # Validate metadata groups if conditions are enabled in scrna_config
        if scrna_config and scrna_config.get("conditions", {}).get("enabled", False):
            metadata_info = get_metadata_groups(gse)
            
            if not metadata_info["available"]:
                raise ValueError(
                    f"❌ Error: Conditions are enabled in scrna_config but no metadata.csv found in {gse}\n\n"
                    f"To use conditions, you need a metadata.csv file with columns:\n"
                    f"  - GSM/Sample column: Sample identifiers (e.g., GSM2406675_10X001)\n"
                    f"  - Group/Condition column: Group labels (e.g., Control, Treatment)\n\n"
                    f"Either:\n"
                    f"  1. Add metadata.csv to {gse}\n"
                    f"  2. Set scrna_config['conditions']['enabled'] = False\n"
                )
            
            if "error" in metadata_info:
                raise ValueError(
                    f"❌ Error: Failed to parse metadata.csv in {gse}\n\n"
                    f"Details: {metadata_info['error']}\n"
                )
            
            # Validate control groups
            control_groups = scrna_config["conditions"].get("control_groups", [])
            if control_groups:
                invalid_controls = [g for g in control_groups if g not in metadata_info["groups"]]
                if invalid_controls:
                    raise ValueError(
                        f"❌ Error: Invalid control group(s): {invalid_controls}\n\n"
                        f"Available groups in {gse}/metadata.csv:\n"
                        + "\n".join(f"  - {g} ({metadata_info['group_counts'][g]} samples)" 
                                   for g in metadata_info['groups'])
                        + "\n\n"
                        f"Suggested control: {metadata_info['suggested_control']}\n"
                        f"You can discover groups using:\n"
                        f"  from crispr.run_pipeline import get_metadata_groups\n"
                        f"  info = get_metadata_groups('{gse}')\n"
                        f"  print(info['groups'])\n"
                    )
            
            # Validate disease groups
            disease_groups = scrna_config["conditions"].get("disease_groups", [])
            if disease_groups:
                invalid_disease = [g for g in disease_groups if g not in metadata_info["groups"]]
                if invalid_disease:
                    raise ValueError(
                        f"❌ Error: Invalid disease group(s): {invalid_disease}\n\n"
                        f"Available groups in {gse}/metadata.csv:\n"
                        + "\n".join(f"  - {g} ({metadata_info['group_counts'][g]} samples)" 
                                   for g in metadata_info['groups'])
                        + "\n\n"
                        f"Suggested disease groups: {metadata_info['suggested_disease']}\n"
                        f"You can discover groups using:\n"
                        f"  from crispr.run_pipeline import get_metadata_groups\n"
                        f"  info = get_metadata_groups('{gse}')\n"
                        f"  print(info['groups'])\n"
                    )
        
        # =============================
        # Stage 0a: Validate dataset
        # =============================
        from .stage0a_dataset_validator import validate_dataset, format_validation_report

        log(f"[STAGE0a] Validating dataset: {gse.name}", BLUE)
        is_valid, val_results = validate_dataset(
            gse_dir=gse,
            samples=chosen_samples,
            out_dir=output_dir / gse.name.replace("_RAW", "") / "validation",
        )
        print(format_validation_report(val_results))

        if not is_valid:
            errors = []
            for r in val_results:
                for e in r["errors"]:
                    errors.append(f"  {r['sample_id']}: {e}")
            raise ValueError(
                f"Dataset validation failed for {gse.name}:\n"
                + "\n".join(errors)
                + "\n\nFix the errors above before running the pipeline."
            )
        log(f"[STAGE0a] Validation passed for {gse.name}", GREEN)

        # Run pipeline for each sample
        _run_pipeline_for_samples(
            gse=gse,
            samples=chosen_samples,
            base_dir=base_dir,
            output_dir=output_dir,
            full_bn=full_bn,
            models=models,
            parallel_training=parallel_training,
            run_scrna=run_scrna,
            scrna_config=scrna_config,
            generate_report=generate_report,
        )

    log("ALL PIPELINES COMPLETE 🎉", GREEN)


def _run_pipeline_for_samples(
    gse: Path,
    samples: List[str],
    base_dir: Path,
    output_dir: Path,
    full_bn: bool,
    models: str,
    parallel_training: bool,
    run_scrna: bool,
    scrna_config: Optional[dict],
    generate_report: bool = True,
) -> None:
    """
    Internal function to run pipeline for specific samples.
    
    Organizes output as: output_dir / {gse_name} / {sample_name} /
    This prevents conflicts when processing multiple GSEs with overlapping sample IDs.
    """
    # Create GSE-specific directory to avoid sample ID conflicts across datasets
    gse_output_dir = output_dir / gse.name.replace("_RAW", "")
    gse_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Track completed samples for scRNA pipeline
    completed_samples = []

    for gsm in samples:
        log(f"===== PIPELINE START {gse.name} :: {gsm} =====", BLUE)

        out = gse_output_dir / gsm
        out.mkdir(exist_ok=True)

        markers = out / ".markers"
        markers.mkdir(exist_ok=True)

        logs = out / "logs"
        logs.mkdir(exist_ok=True)

        # -------- Stage 0 --------
        run_cmd(
            [PY, base_dir / "stage0_ingestion.py",
             "--gse_dir", gse,
             "--out_dir", gse_output_dir,
             "--samples", gsm],
            "STAGE0_INGEST",
            markers / "stage0.done",
            logs,
        )

        # -------- Stage 1 --------
        run_cmd(
            [PY, base_dir / "stage1_call_perturbations.py",
             "--input_h5ad", gse_output_dir / f"raw_{gsm}.h5ad",
             "--gse_dir", gse,
             "--out_dir", out],
            "STAGE1_PERT",
            markers / "stage1.done",
            logs,
        )

        # -------- Stage 2 --------
        run_cmd(
            [PY, base_dir / "stage2_mixscape.py",
             "--input_h5ad", out / "stage1_labeled.h5ad",
             "--out_dir", out,
             "--sample_name", gsm],
            "STAGE2_MIXSCAPE",
            markers / "stage2.done",
            logs,
        )

        # -------- Stage 3 --------
        s3 = out / "processed_stage3"
        s3.mkdir(exist_ok=True)
        run_cmd(
            [PY, base_dir / "stage3_post_mixscape_analysis.py",
             "--stage2_dir", out,
             "--out_dir", s3],
            "STAGE3_POST",
            markers / "stage3.done",
            logs,
        )

        # -------- Stage 4 --------
        s4 = out / "processed_stage4"
        s4.mkdir(exist_ok=True)
        run_cmd(
            [PY, base_dir / "stage4_deg_signature.py",
             "--input_h5ad", s3 / "stage3_merged.h5ad",
             "--out_dir", s4],
            "STAGE4_DEG",
            markers / "stage4.done",
            logs,
        )

        # -------- Stage 5 --------
        run_cmd(
            [PY, base_dir / "stage5_export_ml_dataset.py",
             "--input_h5ad", s4 / "stage4_deg_results.h5ad",
             "--out_dir", out],
            "STAGE5_EXPORT",
            markers / "stage5.done",
            logs,
        )

        # -------- Stage 6 --------
        cmd6 = [
            PY, base_dir / "stage6_train_models.py",
            "--stage5_dir", out,
            "--out_dir", out,
            "--models", models,
        ]
        if parallel_training:
            cmd6.append("--parallel")
        
        run_cmd(
            cmd6,
            "STAGE6_TRAIN",
            markers / "stage6.done",
            logs,
        )

        # -------- Stage 7 --------
        run_cmd(
            [PY, base_dir / "stage7_predict_and_rank.py",
             "--input_h5ad", s4 / "stage4_deg_results.h5ad",
             "--stage5_dir", out,
             "--stage6_dir", out,
             "--out_dir", out],
            "STAGE7_PREDICT",
            markers / "stage7.done",
            logs,
        )

        # -------- Stage 8 --------
        s8 = out / "processed_stage8"
        run_cmd(
            [PY, base_dir / "stage8_qc_and_optimization.py",
             "--stage3_h5ad", s3 / "stage3_merged.h5ad",
             "--stage7_dir", out,
             "--out_dir", s8],
            "STAGE8_QC",
            markers / "stage8.done",
            logs,
        )

        # -------- Stage 9 --------
        s9 = out / "processed_stage9"
        run_cmd(
            [PY, base_dir / "stage9_causal_iv.py",
             "--input_h5ad", s8 / "stage8_qc_annotated.h5ad",
             "--out_dir", s9],
            "STAGE9_IV",
            markers / "stage9.done",
            logs,
        )

        # -------- Stage 10 --------
        s10 = out / "processed_stage10"
        cmd10 = [
            PY, base_dir / "stage10_bayesian_network.py",
            "--input_h5ad", s8 / "stage8_qc_annotated.h5ad",
            "--out_dir", s10,
        ]
        if full_bn:
            cmd10.append("--full_bn")

        run_cmd(
            cmd10,
            "STAGE10_BN",
            markers / "stage10.done",
            logs,
        )

        # -------- Stage 11 --------
        s11 = out / "processed_stage11"
        run_cmd(
            [PY, base_dir / "stage11_latent_ai_ui.py",
             "--input_h5ad", s8 / "stage8_qc_annotated.h5ad",
             "--out_dir", s11],
            "STAGE11_LATENT",
            markers / "stage11.done",
            logs,
        )

        # -------- Stage 12 --------
        s12 = out / "processed_stage12"
        run_cmd(
            [PY, base_dir / "stage12_llm_handoff_delivery.py",
             "--stage8_dir", s8,
             "--stage9_dir", s9,
             "--stage10_dir", s10,
             "--stage11_dir", s11,
             "--out_dir", s12],
            "STAGE12_HANDOFF",
            markers / "stage12.done",
            logs,
        )

        # -------- Report Generation --------
        if generate_report:
            log(f"[REPORT] Generating HTML report for {gsm}", BLUE)
            try:
                from .reporting import generate_report as _gen_report
                report_path = _gen_report(sample_dir=out)
                log(f"[REPORT] Report saved: {report_path}", GREEN)
            except Exception as e:
                log(f"[REPORT] Report generation failed: {e}", RED)
                import traceback
                traceback.print_exc()

        log(f"===== PIPELINE COMPLETE {gse.name} :: {gsm} =====", GREEN)
        
        # Track completed sample for scRNA pipeline
        completed_samples.append(gsm)
    
    # =============================
    # Run scRNA pipeline if enabled and samples completed
    # =============================
    if run_scrna and completed_samples:
        log(f"===== SCRNA PIPELINE START {gse.name} =====", BLUE)
        try:
            _run_scrna_pipeline(
                gse=gse,
                samples=completed_samples,
                gse_output_dir=gse_output_dir,
                base_dir=base_dir,
                user_config=scrna_config,
            )
            log(f"===== SCRNA PIPELINE COMPLETE {gse.name} =====", GREEN)
        except Exception as e:
            log(f"[WARN] scRNA pipeline failed: {e}", RED)
            import traceback
            traceback.print_exc()


# =============================
# scRNA Pipeline Integration
# =============================
def _run_scrna_pipeline(
    gse: Path,
    samples: List[str],
    gse_output_dir: Path,
    base_dir: Path,
    user_config: Optional[dict] = None,
) -> None:
    """
    Run scRNA-seq integration pipeline after CRISPR pipeline completes.
    
    Creates a config file and runs the scRNA pipeline on the processed samples.
    User can override defaults via user_config dict.
    """
    import yaml
    from .scrna_pipeline.run import run_from_config
    
    # Create scRNA output directory
    scrna_output_dir = gse_output_dir / "scrna_results"
    scrna_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Default configuration (matching wizard defaults)
    config = {
        "paths": {
            "input_root": str(gse.parent),
            "dataset_dir": str(gse),
            "out_root": str(scrna_output_dir),
        },
        "dataset_name": gse.name.replace("_RAW", ""),
        "gsm_selected": [s.split("_")[0] for s in samples],  # Extract GSM ID
        "run_mode": "multi" if len(samples) > 1 else "single",
        "conditions": {
            "enabled": False,
            "metadata_csv": None,
            "control_groups": [],
            "disease_groups": [],
            "deg_direction": "disease_vs_control",
        },
        "integration": {
            "run_all": True,
            "methods": ["none", "harmony", "bbknn", "scvi"],
        },
        "annotation": {
            "engines": ["CellTypist"],
            "celltypist_model": "Immune_All_Low.pkl",
            "final_label_strategy": "majority_vote",
            "allow_weak_labels_for_scanvi": True,
        },
        "trajectory": {
            "mode": "paga_dpt_velocity_if_possible",
        },
        "qc": {
            "min_genes": 200,
            "min_cells": 3,
            "max_mt_pct": 20.0,
            "max_genes": 6000,
            "doublets": {
                "enabled": True,
                "expected_doublet_rate": 0.06,
            },
        },
        "preprocess": {
            "hvg_n": 3000,
            "n_pcs": 50,
            "neighbors_k": 15,
            "leiden_resolution": 0.8,
        },
    }
    
    # Check if metadata.csv exists for automatic condition labeling
    metadata_path = gse / "metadata.csv"
    if metadata_path.exists():
        log(f"[INFO] Found metadata.csv", BLUE)
        
        # Try to auto-enable conditions unless explicitly disabled by user
        if user_config is None or user_config.get("conditions", {}).get("enabled") is not False:
            log(f"[INFO] Enabling condition labeling from metadata.csv", BLUE)
            config["conditions"]["enabled"] = True
            config["conditions"]["metadata_csv"] = str(metadata_path)
            
            # Try to infer groups from metadata
            try:
                import pandas as pd
                meta_df = pd.read_csv(metadata_path)
                if "Group" in meta_df.columns:
                    unique_groups = meta_df["Group"].unique().tolist()
                    log(f"[INFO] Found groups in metadata: {unique_groups}", BLUE)
                    
                    # Auto-assign first as control, rest as disease (user can override)
                    if len(unique_groups) >= 2:
                        config["conditions"]["control_groups"] = [unique_groups[0]]
                        config["conditions"]["disease_groups"] = unique_groups[1:]
                        log(f"[INFO] Auto-assigned control={unique_groups[0]}, disease={unique_groups[1:]}", BLUE)
            except Exception as e:
                log(f"[WARN] Could not parse metadata.csv: {e}", RED)
    
    # Merge user config with defaults (user config takes precedence)
    if user_config:
        log(f"[INFO] Applying custom scRNA config overrides", BLUE)
        _deep_merge_config(config, user_config)
    
    # Write config file
    config_path = scrna_output_dir / "scrna_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    log(f"[INFO] Created scRNA config: {config_path}", BLUE)
    log(f"[INFO] Running scRNA pipeline for {len(samples)} sample(s)...", BLUE)
    
    # Run scRNA pipeline
    run_from_config(str(config_path))


def _deep_merge_config(base: dict, override: dict) -> None:
    """
    Deep merge override dict into base dict (modifies base in-place).
    Override values take precedence.
    """
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge_config(base[key], value)
        else:
            base[key] = value


# =============================
# Interactive CLI (original behavior)
# =============================
def main_interactive():
    """Run pipeline with interactive prompts (original behavior)."""
    base = Path(__file__).resolve().parent
    input_root = base / "input_data"
    processed = base / "processed"
    processed.mkdir(exist_ok=True)

    gses = list_gse_folders(input_root)
    if not gses:
        raise RuntimeError("No GSE folders found")

    print("\nAvailable GSEs:")
    for i, g in enumerate(gses, 1):
        print(f"[{i}] {g.name}")

    multi = input("\nRun multiple GSEs? (n / all / select): ").strip().lower()

    if multi == "all":
        chosen_gses = gses
    elif multi == "select":
        picks = input("Enter GSE indices (e.g. 1,3): ").strip()
        chosen_gses = [gses[int(i)-1] for i in picks.split(",") if i.isdigit()]
    else:
        idx = int(input("Select: ")) - 1
        chosen_gses = [gses[idx]]

    for gse in chosen_gses:
        samples = detect_gsm_samples(gse)
        if not samples:
            log(f"[WARN] No samples in {gse}", RED)
            continue

        print(f"\nSamples in {gse.name}:")
        for i, s in enumerate(samples, 1):
            print(f"[{i}] {s}")

        pick = input("Samples (all or 1,3): ").strip().lower()
        if pick == "all":
            chosen_samples = samples
        else:
            chosen_samples = [samples[int(i)-1] for i in pick.split(",")]

        # Call the programmatic API
        run_pipeline(
            input_gse_dirs=gse,
            samples=chosen_samples,
            output_dir=processed,
        )


if __name__ == "__main__":
    main_interactive()

