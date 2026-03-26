"""Lightweight Plotly -> Matplotlib PNG export.

This is a best-effort renderer for common Plotly trace types (bar, scatter,
histogram, heatmap, pie, box). It is intended as a fast alternative to
Kaleido for static PNGs. Complex plot types (sunburst, treemap, etc.) are
not supported and will be skipped.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional


def _get_text(obj, attr: str) -> Optional[str]:
    value = getattr(obj, attr, None)
    if value is None:
        return None
    text = getattr(value, "text", None)
    return text if text else None


def _to_list(value) -> list:
    if value is None:
        return []
    try:
        return list(value)
    except TypeError:
        return [value]


def save_plotly_png_with_mpl(
    fig,
    png_path: Path,
    width: Optional[int] = None,
    height: Optional[int] = None,
    scale: int = 1,
    dpi: int = 100,
) -> bool:
    """
    Render a Plotly figure to PNG using Matplotlib.

    Returns True if a PNG was written, False otherwise.
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception:
        return False

    if fig is None or not getattr(fig, "data", None):
        return False

    png_path.parent.mkdir(parents=True, exist_ok=True)

    layout = getattr(fig, "layout", None)
    layout_width = getattr(layout, "width", None) if layout else None
    layout_height = getattr(layout, "height", None) if layout else None

    w = int(width or layout_width or 900)
    h = int(height or layout_height or 600)
    fig_w = max(2.0, (w * max(1, scale)) / dpi)
    fig_h = max(2.0, (h * max(1, scale)) / dpi)

    mpl_fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
    legend_labels = []

    for trace in fig.data:
        t_type = getattr(trace, "type", "")

        if t_type in {"bar"}:
            x = _to_list(getattr(trace, "x", None))
            y = _to_list(getattr(trace, "y", None))
            orientation = getattr(trace, "orientation", "v") or "v"
            label = getattr(trace, "name", None)
            if orientation == "h":
                ax.barh(y, x, label=label)
            else:
                ax.bar(x, y, label=label)
            if label:
                legend_labels.append(label)

        elif t_type in {"scatter", "scattergl"}:
            x = _to_list(getattr(trace, "x", None))
            y = _to_list(getattr(trace, "y", None))
            if len(x) == 0 and len(y) > 0:
                x = list(range(len(y)))
            mode = getattr(trace, "mode", "markers") or "markers"
            label = getattr(trace, "name", None)
            if "lines" in mode:
                ax.plot(x, y, label=label)
            if "markers" in mode:
                ax.scatter(x, y, label=label)
            if label:
                legend_labels.append(label)

        elif t_type in {"histogram"}:
            x = _to_list(getattr(trace, "x", None))
            if len(x) > 0:
                bins = getattr(trace, "nbinsx", None) or 30
                label = getattr(trace, "name", None)
                ax.hist(x, bins=bins, alpha=0.6, label=label)
                if label:
                    legend_labels.append(label)

        elif t_type in {"heatmap"}:
            z = getattr(trace, "z", None)
            if z is None:
                continue
            z = np.array(z)
            if z.size == 0 or z.ndim < 2:
                continue
            try:
                im = ax.imshow(z, aspect="auto")
                if im is not None:
                    mpl_fig.colorbar(im, ax=ax, shrink=0.8)
            except Exception:
                continue

        elif t_type in {"pie"}:
            values = _to_list(getattr(trace, "values", None))
            labels = _to_list(getattr(trace, "labels", None))
            if len(values) > 0:
                ax.axis("equal")
                ax.pie(values, labels=labels, autopct="%1.1f%%")

        elif t_type in {"box"}:
            y = getattr(trace, "y", None)
            if y is not None:
                ax.boxplot(y)

        else:
            plt.close(mpl_fig)
            return False

    if layout:
        title = _get_text(layout, "title")
        x_title = _get_text(getattr(layout, "xaxis", None), "title")
        y_title = _get_text(getattr(layout, "yaxis", None), "title")
        if title:
            ax.set_title(title)
        if x_title:
            ax.set_xlabel(x_title)
        if y_title:
            ax.set_ylabel(y_title)

    if legend_labels:
        ax.legend(loc="best")

    mpl_fig.tight_layout()
    mpl_fig.savefig(png_path, dpi=dpi)
    plt.close(mpl_fig)
    return True
