"""
ACE Causality Analysis Core Module

Computes Average Causal Effects (ACE) from DepMap Chronos data
and performs therapeutic alignment analysis.
"""

import json
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .config import ACEConfig


# =========================================================
# Utilities
# =========================================================

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def read_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def write_csv(df: pd.DataFrame, path: str):
    ensure_dir(os.path.dirname(path))
    df.to_csv(path, index=False)


# =========================================================
# Loaders
# =========================================================

def load_tidy_chronos(tidy_csv: str) -> pd.DataFrame:
    """
    Load tidy Chronos dependency data and pivot to matrix format.
    
    Args:
        tidy_csv: Path to tidy CSV file with columns: ModelID, Gene, ChronosGeneEffect
        
    Returns:
        DataFrame with ModelID as index, Gene as columns, ChronosGeneEffect as values
    """
    df = read_csv(tidy_csv)
    required = {"ModelID", "Gene", "ChronosGeneEffect"}
    if not required.issubset(df.columns):
        raise ValueError(f"Tidy file missing columns: {required - set(df.columns)}")

    return (
        df.pivot_table(
            index="ModelID",
            columns="Gene",
            values="ChronosGeneEffect",
            aggfunc="mean"
        )
        .sort_index()
    )


def load_gene_stats(path: Optional[str]) -> Optional[pd.DataFrame]:
    """
    Load gene statistics (log2FC) for therapeutic alignment.
    
    Args:
        path: Path to gene stats CSV file with columns: gene, log2fc
        
    Returns:
        DataFrame with gene and log2fc columns, or None if path is None
    """
    if path is None:
        return None
    df = read_csv(path)
    if "gene" not in df.columns or "log2fc" not in df.columns:
        raise ValueError("GeneStats must contain columns: gene, log2fc")
    return df


# =========================================================
# Core causality
# =========================================================

def compute_gene_ace(effects: pd.DataFrame, cfg: ACEConfig) -> pd.DataFrame:
    """
    Compute Average Causal Effect (ACE) for each gene with bootstrap confidence intervals.
    
    Args:
        effects: ModelID x Gene matrix of Chronos effects
        cfg: ACEConfig with computation parameters
        
    Returns:
        DataFrame with columns: gene, ACE, CI_low, CI_high, Stability, n_models, Verdict, Direction
    """
    effects = effects.loc[effects.notna().any(axis=1)]

    ace = effects.median(axis=0) if cfg.center == "median" else effects.mean(axis=0)

    boot = np.vstack([
        effects.sample(frac=1, replace=True).median(axis=0).values
        for _ in range(cfg.n_boot)
    ])

    out = pd.DataFrame({
        "gene": ace.index,
        "ACE": ace.values,
        "CI_low": np.percentile(boot, 2.5, axis=0),
        "CI_high": np.percentile(boot, 97.5, axis=0),
        "Stability": (np.sign(boot) == np.sign(ace.values)).mean(axis=0),
        "n_models": effects.notna().sum(axis=0).values
    })

    out = out[out["n_models"] >= cfg.min_models]
    out["Verdict"] = np.where(out["Stability"] >= cfg.stability_threshold, "Robust", "Unstable")
    out["Direction"] = np.where(out["ACE"] < 0, "Decreases viability", "Increases fitness")

    return out.sort_values("ACE")


# =========================================================
# Ranking + Therapeutic Alignment
# =========================================================

def add_therapeutic_alignment(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add therapeutic alignment columns based on ACE × log2FC product.
    
    Args:
        df: DataFrame with ACE and log2fc columns
        
    Returns:
        DataFrame with added TherapeuticAlignment and TherapeuticComment columns
    """
    product = df["ACE"] * df["log2fc"]

    df["TherapeuticAlignment"] = np.where(
        product < 0, "Reversal",
        np.where(product > 0, "Aggravating", "Neutral")
    )

    df["TherapeuticComment"] = np.select(
        [
            df["TherapeuticAlignment"] == "Reversal",
            df["TherapeuticAlignment"] == "Aggravating"
        ],
        [
            "CRISPR perturbation counteracts patient expression state",
            "CRISPR perturbation reinforces patient expression state"
        ],
        default="No clear directional effect"
    )

    return df


def rank_drivers(ace: pd.DataFrame, gene_stats: Optional[pd.DataFrame]) -> pd.DataFrame:
    """
    Rank causal drivers by therapeutic alignment, verdict, and absolute ACE.
    
    Args:
        ace: DataFrame with ACE results
        gene_stats: Optional DataFrame with gene statistics (log2fc)
        
    Returns:
        Ranked DataFrame with Rank column and therapeutic alignment
    """
    df = ace.copy()

    if gene_stats is not None:
        df = df.merge(gene_stats, on="gene", how="left")

    df = add_therapeutic_alignment(df)

    df["absACE"] = df["ACE"].abs()
    df = df.sort_values(
        ["TherapeuticAlignment", "Verdict", "absACE"],
        ascending=[True, True, False]
    ).reset_index(drop=True)

    df.insert(0, "Rank", np.arange(1, len(df) + 1))
    return df.drop(columns=["absACE"])


# =========================================================
# Main Analysis Function
# =========================================================

def compute_ace_analysis(
    tidy_csv: str,
    gene_stats_csv: Optional[str],
    output_dir: Path,
    config: Optional[ACEConfig] = None,
    logger=None
) -> dict:
    """
    Run complete ACE causality analysis pipeline.
    
    Args:
        tidy_csv: Path to tidy Chronos dependency file
        gene_stats_csv: Path to gene stats file (optional, for therapeutic alignment)
        output_dir: Output directory for results
        config: ACEConfig instance (uses defaults if None)
        logger: Logger instance for progress tracking
        
    Returns:
        Dictionary with paths to generated files:
            - ace_csv: Path to CausalEffects_ACE.csv
            - ranked_csv: Path to CausalDrivers_Ranked.csv
            - manifest: Path to run_manifest.json
    """
    if logger:
        logger.info("Starting ACE causality analysis")
        
    ensure_dir(str(output_dir))
    cfg = config or ACEConfig()

    if logger:
        logger.info(f"Loading tidy Chronos data from: {tidy_csv}")
    effects = load_tidy_chronos(tidy_csv)
    
    if logger:
        logger.info(f"Loading gene statistics from: {gene_stats_csv}")
    gene_stats = load_gene_stats(gene_stats_csv)

    if logger:
        logger.info(f"Computing ACE with {cfg.n_boot} bootstrap iterations")
    ace = compute_gene_ace(effects, cfg)
    
    if logger:
        logger.info("Ranking causal drivers with therapeutic alignment")
    ranked = rank_drivers(ace, gene_stats)

    ace_path = output_dir / "CausalEffects_ACE.csv"
    ranked_path = output_dir / "CausalDrivers_Ranked.csv"
    manifest_path = output_dir / "run_manifest.json"
    
    if logger:
        logger.info(f"Saving ACE results to: {ace_path}")
    write_csv(ace, str(ace_path))
    
    if logger:
        logger.info(f"Saving ranked drivers to: {ranked_path}")
    write_csv(ranked, str(ranked_path))

    with open(manifest_path, "w") as f:
        json.dump({
            "status": "ok",
            "n_genes": int(len(ace)),
            "n_robust": int((ace["Verdict"] == "Robust").sum()),
            "config": {
                "center": cfg.center,
                "n_boot": cfg.n_boot,
                "min_models": cfg.min_models,
                "stability_threshold": cfg.stability_threshold
            }
        }, f, indent=2)

    if logger:
        logger.info(f"ACE analysis complete. Analyzed {len(ace)} genes, {(ace['Verdict'] == 'Robust').sum()} robust effects")

    return {
        "ace_csv": ace_path,
        "ranked_csv": ranked_path,
        "manifest": manifest_path
    }
