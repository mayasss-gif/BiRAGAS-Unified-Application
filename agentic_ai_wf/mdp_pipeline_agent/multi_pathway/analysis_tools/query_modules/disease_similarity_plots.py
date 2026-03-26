#!/usr/bin/env python3
"""
analysis_tools/query_modules/disease_similarity_plots.py

Disease-disease similarity plots from INSIGHTS_out tables.

Robustness:
- If ANY column is missing/zeroed, fallback ANY := (UP>0) OR (DOWN>0).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .common_plotting import plot_heatmap
from .common_scoring import jaccard_similarity_matrix


def run_disease_similarity_plots(
    tables_dir: str,
    out_dir: str,
    presence_csv: str = "presence_long.csv",
    use: str = "ANY",
) -> Path:
    """
    Build disease similarity heatmap using Jaccard over pathway presence.

    Expects long table columns:
      disease, pathway, UP, DOWN, ANY

    Writes:
      shared_disease_similarity.png
    """
    tables = Path(tables_dir).expanduser().resolve()
    out = Path(out_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    p = tables / presence_csv
    if not p.exists():
        raise FileNotFoundError(f"Missing presence table: {p}")

    df = pd.read_csv(p)

    needed = {"disease", "pathway", "UP", "DOWN"}
    if not needed.issubset(df.columns):
        raise ValueError(f"presence_long missing required columns {sorted(needed)}; got {list(df.columns)}")

    if use not in df.columns:
        df[use] = 0

    # Fix ANY if broken
    if use == "ANY":
        any_numeric = pd.to_numeric(df["ANY"], errors="coerce").fillna(0)
        if any_numeric.sum() == 0:
            df["ANY"] = ((pd.to_numeric(df["UP"], errors="coerce").fillna(0) > 0) |
                         (pd.to_numeric(df["DOWN"], errors="coerce").fillna(0) > 0)).astype(int)

    df[use] = pd.to_numeric(df[use], errors="coerce").fillna(0)

    sets = (
        df[df[use] > 0]
        .groupby("disease")["pathway"]
        .apply(lambda s: set(str(x).strip() for x in s if str(x).strip()))
        .to_dict()
    )

    if len(sets) < 2:
        # Still write an empty-but-honest heatmap?
        # Better: raise so you see the failure early.
        raise ValueError(f"Need >=2 diseases with {use}>0 pathways; got {len(sets)} diseases.")

    sim = jaccard_similarity_matrix(sets)

    fig_path = out / "shared_disease_similarity.png"
    plot_heatmap(
        sim,
        title=f"Disease-Disease Similarity (Jaccard, {use} pathways)",
        outpath=fig_path,
        xlabel="Disease",
        ylabel="Disease",
        annotate=False,
    )
    return fig_path
