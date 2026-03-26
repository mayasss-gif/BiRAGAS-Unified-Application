# path: pathway_compare/plotting.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, List
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
from PIL import Image

# NOTE: per your constraints, we use matplotlib only, no seaborn, default colors.

def _save(fig, out: Path, dpi: int):
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

def plot_upset_like(sets: Dict[str, set], out: Path, dpi: int = 300, max_intersections: int = 100):
    """Simple UpSet-like plot (no external deps)."""
    diseases = list(sets.keys())
    if not diseases:
        return
    # Build intersections
    inter_rows = []
    for mask in range(1, 1 << len(diseases)):
        group = [diseases[i] for i in range(len(diseases)) if (mask >> i) & 1]
        inter = set.intersection(*(sets[g] for g in group)) if group else set()
        # subtract items belonging to any superset to avoid overcounting (classic UpSet exactness)
        # For simplicity, we keep raw intersections and later sort by size.
        inter_rows.append((group, len(inter)))
    inter_rows.sort(key=lambda x: x[1], reverse=True)
    inter_rows = [r for r in inter_rows if r[1] > 0][:max_intersections]

    fig = plt.figure(figsize=(10, 6))
    ax_bar = fig.add_axes([0.08, 0.15, 0.75, 0.75])
    ax_dot = fig.add_axes([0.85, 0.15, 0.12, 0.75])

    sizes = [c for _, c in inter_rows]
    ax_bar.bar(range(len(sizes)), sizes)
    ax_bar.set_ylabel("intersection size")
    ax_bar.set_xticks([])

    # Dot matrix
    ax_dot.set_yticks(range(len(diseases)))
    ax_dot.set_yticklabels(diseases)
    ax_dot.set_xticks(range(len(inter_rows)))
    ax_dot.set_xticklabels([])
    ax_dot.invert_yaxis()
    for x, (group, _) in enumerate(inter_rows):
        idxs = [diseases.index(g) for g in group]
        ax_dot.scatter([x]*len(idxs), idxs, s=30)
    _save(fig, out, dpi)

def plot_jaccard_heatmap(jacc: pd.DataFrame, out: Path, dpi: int = 300):
    if jacc.empty:
        return
    # simple clustered order by average linkage on distances (1-jaccard)
    labels = list(jacc.index)
    mat = jacc.values.astype(float)
    # naive ordering by average similarity (keeps dependencies minimal)
    order = np.argsort(-np.nanmean(mat, axis=1))
    mat = mat[order][:, order]
    ordered_labels = [labels[i] for i in order]

    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111)
    im = ax.imshow(mat, aspect="auto")
    ax.set_xticks(range(len(ordered_labels)))
    ax.set_yticks(range(len(ordered_labels)))
    ax.set_xticklabels(ordered_labels, rotation=90)
    ax.set_yticklabels(ordered_labels)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Jaccard")
    _save(fig, out, dpi)

def plot_bar_coverage(cov: pd.DataFrame, out: Path, dpi: int = 300):
    if cov.empty:
        return
    df = cov.sort_values("count", ascending=False)
    fig = plt.figure(figsize=(8, 4))
    ax = fig.add_subplot(111)
    ax.barh(df["disease"], df["count"])
    ax.invert_yaxis()
    ax.set_xlabel("entities in pathway")
    _save(fig, out, dpi)

def plot_bar_unique(df_counts: pd.DataFrame, out: Path, dpi: int = 300):
    if df_counts.empty:
        return
    fig = plt.figure(figsize=(8, 4))
    ax = fig.add_subplot(111)
    ax.barh(df_counts["disease"], df_counts["count"])
    ax.invert_yaxis()
    ax.set_xlabel("unique entities (vs others)")
    _save(fig, out, dpi)

def assemble_grid_figure(upset_path: Path,
                         heatmap_path: Path,
                         coverage_path: Path,
                         unique_path: Path,
                         out_path_png: Path,
                         out_path_svg: Path,
                         dpi: int = 300,
                         title: str = ""):
    """Stitch four PNGs into a 2x2 grid to guarantee consistent layout."""
    def _load(p: Path):
        return Image.open(p) if p.exists() else Image.new("RGB", (800, 600), "white")

    A = _load(upset_path)
    B = _load(heatmap_path)
    C = _load(coverage_path)
    D = _load(unique_path)

    w = max(A.width, B.width, C.width, D.width)
    h = max(A.height, B.height, C.height, D.height)
    canvas = Image.new("RGB", (2*w, 2*h), "white")
    canvas.paste(A.resize((w, h)), (0, 0))
    canvas.paste(B.resize((w, h)), (w, 0))
    canvas.paste(C.resize((w, h)), (0, h))
    canvas.paste(D.resize((w, h)), (w, h))

    out_path_png.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path_png)
    canvas.save(out_path_svg)
