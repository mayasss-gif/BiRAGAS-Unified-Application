#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _savefig(path: Path, dpi: int = 200) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()
    return str(path)


def plot_heatmap(
    mat: pd.DataFrame,
    out_png: Path,
    title: str,
    xlabel: str = "",
    ylabel: str = "",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    rotate_x: int = 90,
    rotate_y: int = 0,
) -> str:
    if mat is None or mat.empty:
        fig = plt.figure(figsize=(10, 3))
        plt.title(title)
        plt.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=22)
        plt.axis("off")
        return _savefig(out_png)

    # dynamic sizing: huge pathway lists won’t explode the figure
    h = max(4.0, min(28.0, 0.25 * mat.shape[0]))
    w = max(6.0, min(24.0, 0.35 * mat.shape[1]))
    plt.figure(figsize=(w, h))

    arr = mat.to_numpy(dtype=float)
    im = plt.imshow(arr, aspect="auto", interpolation="nearest", vmin=vmin, vmax=vmax)
    plt.colorbar(im)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.xticks(range(mat.shape[1]), mat.columns.tolist(), rotation=rotate_x, ha="right")
    plt.yticks(range(mat.shape[0]), mat.index.tolist(), rotation=rotate_y)

    return _savefig(out_png)


def plot_barh(
    series: pd.Series,
    out_png: Path,
    title: str,
    xlabel: str = "",
    top_n: int = 30,
) -> str:
    s = series.dropna()
    if s.empty:
        fig = plt.figure(figsize=(10, 3))
        plt.title(title)
        plt.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=22)
        plt.axis("off")
        return _savefig(out_png)

    s = s.sort_values(ascending=True).tail(top_n)
    h = max(4.0, min(18.0, 0.35 * len(s)))
    plt.figure(figsize=(12, h))
    plt.barh(s.index.astype(str), s.values)
    plt.title(title)
    plt.xlabel(xlabel)
    return _savefig(out_png)


def plot_scatter(
    df: pd.DataFrame,
    x: str,
    y: str,
    out_png: Path,
    title: str,
    xlabel: str,
    ylabel: str,
    label_col: Optional[str] = None,
    label_top_n: int = 15,
) -> str:
    if df is None or df.empty or x not in df.columns or y not in df.columns:
        fig = plt.figure(figsize=(10, 3))
        plt.title(title)
        plt.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=22)
        plt.axis("off")
        return _savefig(out_png)

    plt.figure(figsize=(12, 7))
    plt.scatter(df[x].values, df[y].values, alpha=0.8)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    if label_col and label_col in df.columns:
        # label the “most interesting” by y then x
        d2 = df.copy()
        d2["_rank"] = d2[y].rank(method="average", ascending=False) + d2[x].rank(method="average", ascending=False)
        d2 = d2.sort_values("_rank").head(label_top_n)
        for _, r in d2.iterrows():
            plt.text(float(r[x]), float(r[y]), str(r[label_col]), fontsize=9)

    return _savefig(out_png)
