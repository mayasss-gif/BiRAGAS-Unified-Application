# pathway_compare/tables_figs.py
from __future__ import annotations
import itertools, logging, math, re, textwrap
from pathlib import Path
from typing import Dict, List, Tuple, Set

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.stats import fisher_exact

from .config import PCConfig

# ---------------- Excel helpers ----------------

_ILLEGAL_SHEET_CHARS = re.compile(r'[:\\/*?\[\]]')

def _safe_sheet_name(name: str) -> str:
    s = str(name or "").strip()
    s = re.sub(r"\s+", " ", s)
    s = _ILLEGAL_SHEET_CHARS.sub("_", s)
    return s[:31]

def _write_lists_excel(
    out_path: Path,
    sheets: Dict[str, pd.DataFrame],
    *,
    engine: str = "openpyxl",
    **writer_kwargs,
) -> None:
    """
    Write multiple DataFrames to one XLSX; accepts arbitrary kwargs to remain
    backward-compatible with older call sites that pass extra args.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with pd.ExcelWriter(out_path, engine=engine, **writer_kwargs) as xw:
            for raw_name, df in sheets.items():
                safe = _safe_sheet_name(raw_name)
                (df if isinstance(df, pd.DataFrame) else pd.DataFrame(df)).to_excel(
                    xw, sheet_name=safe, index=False
                )
    except Exception as e:
        logging.error(f"failed to write Excel {out_path.name}: {e}")

# ---------------- dynamic plotting utilities ----------------

def _wrap_labels(labels: List[str], width: int = 22) -> List[str]:
    """Wrap long tick labels to multiple lines at word boundaries."""
    out = []
    for s in labels:
        s = str(s)
        if len(s) <= width:
            out.append(s)
        else:
            out.append(textwrap.fill(s, width=width, break_long_words=False))
    return out

def _auto_tick_font(n: int, base: int = 10, min_size: int = 6) -> int:
    """Smaller font for many ticks."""
    if n <= 10: return base
    if n >= 60: return min_size
    # linear decay
    return max(min_size, int(round(base - (base - min_size) * (n - 10) / 50.0)))

def _auto_ann_ok(n_rows: int, n_cols: int, max_cells: int = 300) -> bool:
    """Only annotate if grid is not too dense."""
    return (n_rows * n_cols) <= max_cells

def _auto_figsize_for_heatmap(n_rows: int, n_cols: int, max_lbl_len: int) -> Tuple[float, float]:
    """
    Compute a reasonable figure size based on grid size and label lengths.
    """
    # base per-cell size
    w = max(5.5, 0.55 * n_cols + 0.03 * max_lbl_len)
    h = max(4.5, 0.55 * n_rows + 0.03 * max_lbl_len)
    return (float(w), float(h))

def _auto_figsize_for_bars(n_items: int, horiz: bool, max_lbl_len: int) -> Tuple[float, float]:
    if horiz:
        # height grows with items, width with label length
        h = max(4.5, 0.38 * n_items + 1.5)
        w = max(7.0, 0.06 * max_lbl_len + 6.0)
    else:
        # width grows with items, height with label length (for rotated labels)
        w = max(6.0, 0.55 * n_items + 0.03 * max_lbl_len)
        h = max(4.5, 4.5 + 0.02 * max_lbl_len)
    return (float(w), float(h))

def _annotate_matrix(ax: plt.Axes, M: np.ndarray, fmt: str, fontsize: int) -> None:
    for (i, j), val in np.ndenumerate(M):
        try:
            ax.text(j, i, (fmt.format(val)), ha="center", va="center", fontsize=fontsize)
        except Exception:
            pass

def _sci(x: float) -> str:
    try:
        return f"{x:.1e}"
    except Exception:
        return str(x)

# ---------------- view selection ----------------

def _pick_view(views: Dict[str, pd.DataFrame], cfg: PCConfig) -> pd.DataFrame:
    mode = (cfg.direction_mode or "both").lower()
    if mode == "any":
        return views.get("ANY", pd.DataFrame()).copy()
    if mode == "up":
        return views.get("UP", pd.DataFrame()).copy()
    if mode == "down":
        return views.get("DOWN", pd.DataFrame()).copy()
    # both -> prefer ANY; fallback to union(UP,DOWN)
    base = views.get("ANY")
    if base is not None and not base.empty:
        return base.copy()
    ud = pd.concat([views.get("UP", pd.DataFrame()), views.get("DOWN", pd.DataFrame())],
                   ignore_index=True)
    if ud.empty:
        return pd.DataFrame(columns=["disease","pathway","entity_type","entity","overlap_genes"])
    return (ud.sort_values(["qval","pval","sig"], ascending=[True, True, False], na_position="last")
              .drop_duplicates(["disease","pathway","entity_type","entity"], keep="first"))

# ---------------- set building ----------------

def _collect_sets_for_pathway(df: pd.DataFrame, pathway: str) -> Tuple[Dict[str, Set[str]], Dict[str, Set[str]]]:
    sub = df[df["pathway"].astype(str) == str(pathway)].copy()
    genes_by_dis: Dict[str, Set[str]] = {}
    entities_by_dis: Dict[str, Set[str]] = {}
    for dis, g in sub.groupby("disease"):
        # genes
        og = []
        if "overlap_genes" in g.columns:
            for lst in g["overlap_genes"]:
                if isinstance(lst, (list, tuple, set)):
                    og.extend([str(x) for x in lst if isinstance(x, str)])
        genes_by_dis[dis] = set(og)
        # entities
        entities_by_dis[dis] = set(g["entity"].dropna().astype(str).tolist())
    return genes_by_dis, entities_by_dis

# ---------------- pairwise computations ----------------

def _pairwise_matrix(items: List[str], sets: Dict[str, Set[str]], how: str = "jaccard") -> pd.DataFrame:
    n = len(items)
    M = np.zeros((n, n), dtype=float)
    for i, a in enumerate(items):
        A = sets.get(a, set())
        for j, b in enumerate(items):
            B = sets.get(b, set())
            if how == "count":
                val = len(A & B)
            elif how == "jaccard":
                u = len(A | B)
                val = (len(A & B) / u) if u else 0.0
            else:
                val = np.nan
            M[i, j] = val
    return pd.DataFrame(M, index=items, columns=items)

def _pairwise_fisher(items: List[str], sets: Dict[str, Set[str]], universe: int | None) -> pd.DataFrame:
    n = len(items)
    P = np.ones((n, n), dtype=float)
    for i, a in enumerate(items):
        A = sets.get(a, set())
        for j, b in enumerate(items):
            B = sets.get(b, set())
            if i == j:
                P[i, j] = 1.0
                continue
            U = universe if (isinstance(universe, int) and universe > 0) else len(A | B)
            o = len(A & B); a_only = len(A - B); b_only = len(B - A)
            neither = max(U - (o + a_only + b_only), 0)
            try:
                _, p = fisher_exact([[o, a_only], [b_only, neither]], alternative="greater")
            except Exception:
                p = 1.0
            P[i, j] = p
    return pd.DataFrame(P, index=items, columns=items)

def _all_intersections(sets: Dict[str, Set[str]]) -> List[Tuple[Tuple[str,...], int]]:
    items = list(sets.keys())
    out: List[Tuple[Tuple[str,...], int]] = []
    for r in range(1, len(items) + 1):
        for comb in itertools.combinations(items, r):
            inter = set.intersection(*(sets[k] for k in comb)) if r > 1 else sets[comb[0]]
            s = len(inter)
            if s > 0:
                out.append((comb, s))
    out.sort(key=lambda x: (-x[1], len(x[0]), x[0]))
    return out

# ---------------- plotting (DYNAMIC) ----------------

def _save_fig(fig: plt.Figure, out_png: Path, dpi: int = 300) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    try:
        fig.savefig(out_png.with_suffix(".svg"), bbox_inches="tight")
    except Exception:
        pass
    plt.close(fig)

def _plot_upset_like(sets: Dict[str, Set[str]], out_png: Path, cfg: PCConfig) -> None:
    inters = _all_intersections(sets)
    if cfg.max_upset_intersections and len(inters) > cfg.max_upset_intersections:
        inters = inters[:cfg.max_upset_intersections]
    labels = [" ∩ ".join(k) for k, _ in inters]
    labels_wrapped = _wrap_labels(labels, width=28)
    sizes  = [s for _, s in inters]

    # horizontal if many
    fontsize = _auto_tick_font(len(labels_wrapped), base=10, min_size=6)
    w, h = _auto_figsize_for_bars(len(labels_wrapped), horiz=True, max_lbl_len=max(len(x) for x in labels_wrapped) if labels_wrapped else 10)

    fig = plt.figure(figsize=(w, h))
    ax = fig.add_subplot(111)
    y = np.arange(len(labels_wrapped))
    ax.barh(y, sizes)
    ax.set_yticks(y, labels_wrapped, fontsize=fontsize)
    ax.invert_yaxis()
    ax.set_xlabel("intersection size")
    ax.set_title("UpSet-like intersections")
    fig.tight_layout()
    _save_fig(fig, out_png, cfg.dpi)

def _plot_jaccard_heatmap(items: List[str], M: pd.DataFrame, out_png: Path, cfg: PCConfig, annotate: bool = True) -> None:
    labels_x = _wrap_labels(items, width=24)
    labels_y = _wrap_labels(items, width=24)
    max_len = max(max(len(s) for s in labels_x) if labels_x else 10,
                  max(len(s) for s in labels_y) if labels_y else 10)
    w, h = _auto_figsize_for_heatmap(len(items), len(items), max_len)
    font = _auto_tick_font(len(items), base=10, min_size=6)

    fig = plt.figure(figsize=(w, h))
    ax = fig.add_subplot(111)
    im = ax.imshow(M.values, aspect="auto")
    ax.set_xticks(range(len(items)), labels_x, rotation=45, ha="right", fontsize=font)
    ax.set_yticks(range(len(items)), labels_y, fontsize=font)
    ax.set_title("Pairwise Jaccard similarity")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel("Jaccard", rotation=270, va="bottom")
    if annotate and _auto_ann_ok(len(items), len(items)):
        _annotate_matrix(ax, M.values, "{:.2f}", fontsize=max(6, font-1))
    fig.tight_layout()
    _save_fig(fig, out_png, cfg.dpi)

def _plot_fisher_heatmap(items: List[str], P: pd.DataFrame, title: str, out_png: Path, cfg: PCConfig) -> None:
    """
    Plot Fisher results with a stable visual scale:
    - Use -log10(p) for the color map.
    - Annotate with scientific p if grid is not too dense.
    """
    # transform
    with np.errstate(divide="ignore", invalid="ignore"):
        X = -np.log10(np.clip(P.values, 1e-300, 1.0))
    labels_x = _wrap_labels(items, width=24)
    labels_y = _wrap_labels(items, width=24)
    max_len = max(max(len(s) for s in labels_x) if labels_x else 10,
                  max(len(s) for s in labels_y) if labels_y else 10)
    w, h = _auto_figsize_for_heatmap(len(items), len(items), max_len)
    font = _auto_tick_font(len(items), base=10, min_size=6)

    fig = plt.figure(figsize=(w, h))
    ax = fig.add_subplot(111)
    im = ax.imshow(X, aspect="auto")
    ax.set_xticks(range(len(items)), labels_x, rotation=45, ha="right", fontsize=font)
    ax.set_yticks(range(len(items)), labels_y, fontsize=font)
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel("-log10(p)", rotation=270, va="bottom")
    if _auto_ann_ok(len(items), len(items)):
        for (i, j), val in np.ndenumerate(P.values):
            ax.text(j, i, _sci(val), ha="center", va="center", fontsize=max(6, font-1))
    fig.tight_layout()
    _save_fig(fig, out_png, cfg.dpi)

def _plot_coverage_bars(counts: Dict[str, int], out_png: Path, cfg: PCConfig) -> None:
    items = list(counts.keys())
    vals  = [counts[k] for k in items]
    labels = _wrap_labels(items, width=20)
    many = len(items) > 16
    w, h = _auto_figsize_for_bars(len(items), horiz=many, max_lbl_len=max(len(s) for s in labels) if labels else 10)
    font = _auto_tick_font(len(items), base=10, min_size=6)

    fig = plt.figure(figsize=(w, h))
    ax = fig.add_subplot(111)
    if many:
        y = np.arange(len(labels))
        ax.barh(y, vals)
        ax.set_yticks(y, labels, fontsize=font)
        for i, v in enumerate(vals):
            if len(items) <= 35:
                ax.text(v, i, str(v), va="center", ha="left", fontsize=max(6, font-1))
        ax.set_xlabel("# pathway genes detected")
    else:
        ax.bar(labels, vals)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=font)
        for i, v in enumerate(vals):
            if len(items) <= 25:
                ax.text(i, v + max(1, 0.02 * (max(vals) if vals else 1)), str(v),
                        ha="center", va="bottom", fontsize=max(6, font-1))
        ax.set_ylabel("# pathway genes detected")
    ax.set_title("Coverage per disease")
    fig.tight_layout()
    _save_fig(fig, out_png, cfg.dpi)

def _plot_unique_shared_stack(genes_by_dis: Dict[str, Set[str]], out_png: Path, cfg: PCConfig) -> None:
    # core = genes present in all diseases, shared = present in >=2, unique = present in exactly 1
    diseases = list(genes_by_dis.keys())
    labels = _wrap_labels(diseases, width=20)
    all_genes = set().union(*genes_by_dis.values()) if diseases else set()
    uniq_counts = []
    shared_counts = []
    core_counts = []
    mem = {g: sum(g in genes_by_dis[d] for d in diseases) for g in all_genes}
    for d in diseases:
        uniq = sum(1 for g in genes_by_dis[d] if mem[g] == 1)
        core = sum(1 for g in genes_by_dis[d] if mem[g] == len(diseases))
        shared = len(genes_by_dis[d]) - uniq - core
        uniq_counts.append(uniq); shared_counts.append(shared); core_counts.append(core)

    many = len(diseases) > 16
    w, h = _auto_figsize_for_bars(len(diseases), horiz=many, max_lbl_len=max(len(s) for s in labels) if labels else 10)
    font = _auto_tick_font(len(diseases), base=10, min_size=6)

    fig = plt.figure(figsize=(w, h))
    ax = fig.add_subplot(111)

    if many:
        # horizontal stacked bars
        y = np.arange(len(labels))
        y1 = np.array(uniq_counts)
        y2 = np.array(shared_counts)
        y3 = np.array(core_counts)
        ax.barh(y, y1, label="unique")
        ax.barh(y, y2, left=y1, label="shared (>=2)")
        ax.barh(y, y3, left=y1 + y2, label="core (all)")
        ax.set_yticks(y, labels, fontsize=font)
        ax.set_xlabel("# genes")
    else:
        y1 = np.array(uniq_counts)
        y2 = np.array(shared_counts)
        y3 = np.array(core_counts)
        ax.bar(labels, y1, label="unique")
        ax.bar(labels, y2, bottom=y1, label="shared (>=2)")
        ax.bar(labels, y3, bottom=y1+y2, label="core (all)")
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=font)
        ax.set_ylabel("# genes")

    ax.legend(frameon=False, fontsize=max(7, font))
    ax.set_title("Unique vs shared vs core")
    fig.tight_layout()
    _save_fig(fig, out_png, cfg.dpi)

def _plot_pairwise_diff_grid(diseases: List[str], genes_by_dis: Dict[str, Set[str]], out_png: Path, cfg: PCConfig) -> None:
    r"""Cell text: |A∩B| / |A\B| / |B\A| for quick difference reading."""
    n = len(diseases)
    labels_x = _wrap_labels(diseases, width=22)
    labels_y = _wrap_labels(diseases, width=22)
    max_len = max(max(len(s) for s in labels_x) if labels_x else 10,
                  max(len(s) for s in labels_y) if labels_y else 10)
    w, h = _auto_figsize_for_heatmap(n, n, max_len)
    font = _auto_tick_font(n, base=10, min_size=6)

    Mtxt = [["" for _ in range(n)] for __ in range(n)]
    for i, a in enumerate(diseases):
        A = genes_by_dis[a]
        for j, b in enumerate(diseases):
            B = genes_by_dis[b]
            if i == j:
                Mtxt[i][j] = f"{len(A)}"
            else:
                Mtxt[i][j] = f"{len(A & B)}/{len(A - B)}/{len(B - A)}"

    fig = plt.figure(figsize=(w, h))
    ax = fig.add_subplot(111)
    ax.imshow(np.ones((n, n)), aspect="auto", vmin=0, vmax=1)  # white tiles
    ax.set_xticks(range(n), labels_x, rotation=45, ha="right", fontsize=font)
    ax.set_yticks(range(n), labels_y, fontsize=font)
    ax.set_title("pairwise: |shared| / |A-only| / |B-only|")

    if _auto_ann_ok(n, n, max_cells=500):
        for i in range(n):
            for j in range(n):
                ax.text(j, i, Mtxt[i][j], ha="center", va="center", fontsize=max(6, font-1))
    fig.tight_layout()
    _save_fig(fig, out_png, cfg.dpi)

def _plot_story_panel(genes_by_dis: Dict[str, Set[str]], jaccard: pd.DataFrame, out_png: Path, cfg: PCConfig) -> None:
    diseases = list(jaccard.index)
    labels = _wrap_labels(diseases, width=20)
    inters = _all_intersections(genes_by_dis)
    cov = {d: len(genes_by_dis[d]) for d in diseases}

    # panel size fixed; internal axes scale their own labels
    fig = plt.figure(figsize=(14, 8))
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1, 1], width_ratios=[1, 1.2])

    # (A) Coverage bars
    axA = fig.add_subplot(gs[0, 0])
    xs = np.arange(len(labels))
    vals = [cov[d] for d in diseases]
    horiz = len(labels) > 18
    if horiz:
        axA.barh(xs, vals)
        axA.set_yticks(xs, labels, fontsize=_auto_tick_font(len(labels)))
        axA.set_xlabel("# genes")
    else:
        axA.bar(labels, vals)
        axA.set_xticklabels(labels, rotation=45, ha="right", fontsize=_auto_tick_font(len(labels)))
        axA.set_ylabel("# genes")
    axA.set_title("A) Coverage per disease")

    # (B) Jaccard heatmap (annotated if small)
    axB = fig.add_subplot(gs[0, 1])
    im = axB.imshow(jaccard.values, aspect="auto")
    axB.set_xticks(range(len(labels)), labels, rotation=45, ha="right", fontsize=_auto_tick_font(len(labels)))
    axB.set_yticks(range(len(labels)), labels, fontsize=_auto_tick_font(len(labels)))
    axB.set_title("B) Pairwise Jaccard")
    cbar = fig.colorbar(im, ax=axB, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel("Jaccard", rotation=270, va="bottom")
    if _auto_ann_ok(len(labels), len(labels), max_cells=250):
        _annotate_matrix(axB, jaccard.values, "{:.2f}", fontsize=max(6, _auto_tick_font(len(labels))-1))

    # (C) UpSet-like intersection spectrum
    axC = fig.add_subplot(gs[1, 0])
    cap = (cfg.max_upset_intersections or 100)
    subset = inters[:cap]
    labels_inters = _wrap_labels([" ∩ ".join(k) for k, _ in subset], width=26)
    sizes  = [s for _, s in subset]
    y = np.arange(len(labels_inters))
    axC.barh(y, sizes)
    axC.set_yticks(y, labels_inters, fontsize=_auto_tick_font(len(labels_inters)))
    axC.invert_yaxis()
    axC.set_xlabel("intersection size")
    axC.set_title("C) Intersection spectrum")

    # (D) Unique vs shared vs core (stacked)
    axD = fig.add_subplot(gs[1, 1])
    all_genes = set().union(*genes_by_dis.values()) if diseases else set()
    mem = {g: sum(g in genes_by_dis[d] for d in diseases) for g in all_genes}
    uniq = [sum(1 for g in genes_by_dis[d] if mem[g] == 1) for d in diseases]
    core = [sum(1 for g in genes_by_dis[d] if mem[g] == len(diseases)) for d in diseases]
    shared = [len(genes_by_dis[d]) - u - c for d, u, c in zip(diseases, uniq, core)]
    y1, y2, y3 = np.array(uniq), np.array(shared), np.array(core)
    many = len(labels) > 18
    if many:
        pos = np.arange(len(labels))
        axD.barh(pos, y1, label="unique")
        axD.barh(pos, y2, left=y1, label="shared (>=2)")
        axD.barh(pos, y3, left=y1+y2, label="core (all)")
        axD.set_yticks(pos, labels, fontsize=_auto_tick_font(len(labels)))
        axD.set_xlabel("# genes")
    else:
        axD.bar(labels, y1, label="unique")
        axD.bar(labels, y2, bottom=y1, label="shared (>=2)")
        axD.bar(labels, y3, bottom=y1+y2, label="core (all)")
        axD.set_xticklabels(labels, rotation=45, ha="right", fontsize=_auto_tick_font(len(labels)))
        axD.set_ylabel("# genes")
    axD.legend(frameon=False, fontsize=_auto_tick_font(len(labels)))
    axD.set_title("D) Unique vs shared vs core")

    fig.suptitle("Pathway comparison — story panel", y=1.02)
    fig.tight_layout()
    _save_fig(fig, out_png, cfg.dpi)

# ---------------- utilities for ragged DataFrames ----------------

def _pad_columns_to_equal_length(cols: Dict[str, List[str]], pad: str = "-") -> Dict[str, List[str]]:
    """Right-pad all list columns to a common length so DataFrame(...) won't error."""
    max_len = 0
    for k, v in cols.items():
        if not isinstance(v, list):
            cols[k] = [v] if v is not None else [pad]
        max_len = max(max_len, len(cols[k]))
    for k, v in cols.items():
        if len(v) < max_len:
            cols[k] = v + [pad] * (max_len - len(v))
    return cols

# ---------------- NEW HELPERS FOR MULTI-PATHWAY ----------------

def _pathway_sets_across_diseases(df: pd.DataFrame, pathways: List[str]) -> Tuple[Dict[str, Set[str]], Dict[str, Dict[str, Set[str]]]]:
    """
    Build gene sets for each pathway:
      overall_by_pw: {pathway -> union-of-genes across all diseases}
      per_dis: {disease -> {pathway -> genes for that pathway in that disease}}
    """
    overall_by_pw: Dict[str, Set[str]] = {pw: set() for pw in pathways}
    per_dis: Dict[str, Dict[str, Set[str]]] = {}

    for pw in pathways:
        sub_genes, _ = _collect_sets_for_pathway(df, pw)
        for dis, gset in sub_genes.items():
            overall_by_pw[pw].update(gset)
            if dis not in per_dis:
                per_dis[dis] = {p: set() for p in pathways}
            per_dis[dis][pw].update(gset)

    for dis in list(per_dis.keys()):
        for pw in pathways:
            per_dis[dis].setdefault(pw, set())

    return overall_by_pw, per_dis

def _plot_pathway_story_panel(pathways: List[str],
                              overall_sets: Dict[str, Set[str]],
                              jacc_overall: pd.DataFrame,
                              out_png: Path,
                              cfg: PCConfig) -> None:
    cov_vals = [len(overall_sets[pw]) for pw in pathways]
    inters = _all_intersections(overall_sets)

    fig = plt.figure(figsize=(14, 8))
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1, 1], width_ratios=[1, 1.2])

    # (A) Coverage per pathway
    axA = fig.add_subplot(gs[0, 0])
    labels = _wrap_labels(pathways, width=20)
    many = len(labels) > 18
    if many:
        y = np.arange(len(labels))
        axA.barh(y, cov_vals)
        axA.set_yticks(y, labels, fontsize=_auto_tick_font(len(labels)))
        axA.set_xlabel("# genes")
    else:
        axA.bar(labels, cov_vals)
        axA.set_xticklabels(labels, rotation=45, ha="right", fontsize=_auto_tick_font(len(labels)))
        axA.set_ylabel("# genes")
    axA.set_title("A) Coverage per pathway")

    # (B) Overall Jaccard
    axB = fig.add_subplot(gs[0, 1])
    labels_heat = _wrap_labels(pathways, width=20)
    im = axB.imshow(jacc_overall.values, aspect="auto")
    axB.set_xticks(range(len(pathways)), labels_heat, rotation=45, ha="right", fontsize=_auto_tick_font(len(pathways)))
    axB.set_yticks(range(len(pathways)), labels_heat, fontsize=_auto_tick_font(len(pathways)))
    axB.set_title("B) Pathway×Pathway Jaccard (overall)")
    cbar = fig.colorbar(im, ax=axB, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel("Jaccard", rotation=270, va="bottom")
    if _auto_ann_ok(len(pathways), len(pathways)):
        _annotate_matrix(axB, jacc_overall.values, "{:.2f}", fontsize=max(6, _auto_tick_font(len(pathways))-1))

    # (C) UpSet-like across pathways
    axC = fig.add_subplot(gs[1, 0])
    cap = (cfg.max_upset_intersections or 100)
    subset = inters[:cap]
    labels_inters = _wrap_labels([" ∩ ".join(k) for k, _ in subset], width=24)
    sizes = [s for _, s in subset]
    y = np.arange(len(labels_inters))
    axC.barh(y, sizes)
    axC.set_yticks(y, labels_inters, fontsize=_auto_tick_font(len(labels_inters)))
    axC.invert_yaxis()
    axC.set_xlabel("intersection size")
    axC.set_title("C) Intersection spectrum (across pathways)")

    # (D) Unique/shared/core per pathway
    axD = fig.add_subplot(gs[1, 1])
    all_genes = set().union(*overall_sets.values()) if overall_sets else set()
    mem = {g: sum(g in overall_sets[p] for p in pathways) for g in all_genes}
    uniq = [sum(1 for g in overall_sets[pw] if mem[g] == 1) for pw in pathways]
    core = [sum(1 for g in overall_sets[pw] if mem[g] == len(pathways)) for pw in pathways]
    shared = [len(overall_sets[pw]) - u - c for pw, u, c in zip(pathways, uniq, core)]
    labels_uc = _wrap_labels(pathways, width=18)
    many2 = len(labels_uc) > 18
    if many2:
        pos = np.arange(len(labels_uc))
        axD.barh(pos, uniq, label="unique")
        axD.barh(pos, shared, left=np.array(uniq), label="shared (>=2)")
        axD.barh(pos, core, left=np.array(uniq)+np.array(shared), label="core (all)")
        axD.set_yticks(pos, labels_uc, fontsize=_auto_tick_font(len(labels_uc)))
        axD.set_xlabel("# genes")
    else:
        axD.bar(labels_uc, uniq, label="unique")
        axD.bar(labels_uc, shared, bottom=np.array(uniq), label="shared (>=2)")
        axD.bar(labels_uc, core, bottom=np.array(uniq)+np.array(shared), label="core (all)")
        axD.set_xticklabels(labels_uc, rotation=45, ha="right", fontsize=_auto_tick_font(len(labels_uc)))
        axD.set_ylabel("# genes")
    axD.legend(frameon=False, fontsize=_auto_tick_font(len(labels_uc)))
    axD.set_title("D) Unique vs shared vs core")

    fig.suptitle("Multi-pathway comparison — story panel", y=1.02)
    fig.tight_layout()
    _save_fig(fig, out_png, cfg.dpi)

# ---------------- public API (called from compute.py) ----------------

def run_one_pathway_all_outputs(pathway: str, views: Dict[str, pd.DataFrame], out_root: Path, cfg: PCConfig) -> None:
    out_dir = out_root / _safe_sheet_name(str(pathway).replace(" ", "_"))
    out_dir.mkdir(parents=True, exist_ok=True)

    df = _pick_view(views, cfg)
    if df.empty:
        logging.warning(f"[{pathway}] no entities found for selected view")
        return

    genes_by_dis, entities_by_dis = _collect_sets_for_pathway(df, pathway)
    diseases = sorted(genes_by_dis.keys())
    if not diseases:
        logging.warning(f"[{pathway}] present in none of the diseases")
        return

    # Matrices (genes)
    shared_counts = _pairwise_matrix(diseases, genes_by_dis, how="count")
    jaccard       = _pairwise_matrix(diseases, genes_by_dis, how="jaccard")
    pvals         = _pairwise_fisher(diseases, genes_by_dis, universe=None)

    # Coverage table
    coverage = pd.DataFrame(
        [{"disease": d, "n_pathway_genes_detected": len(genes_by_dis[d]),
          "n_entities": len(entities_by_dis.get(d, set()))} for d in diseases]
    ).sort_values("disease")

    # Pairwise lists (names)
    pair_rows = []
    for a, b in itertools.combinations(diseases, 2):
        A, B = genes_by_dis[a], genes_by_dis[b]
        pair_rows.append({
            "pair": f"{a} ∩ {b}",
            "shared_genes": ", ".join(sorted(A & B)) or "-",
            "n_shared": len(A & B),
            f"unique_in_{a}": ", ".join(sorted(A - B)) or "-",
            f"unique_in_{b}": ", ".join(sorted(B - A)) or "-",
        })
    shared_names = pd.DataFrame(pair_rows)

    # Per-disease detail (genes + entities)
    uniq_sheets: Dict[str, pd.DataFrame] = {}
    for d in diseases:
        uniq_sheets[f"{d} — genes"]    = pd.DataFrame({"gene": sorted(genes_by_dis[d])})
        uniq_sheets[f"{d} — entities"] = pd.DataFrame({"entity": sorted(entities_by_dis.get(d, set()))})

    # Pairwise differences workbook (pad columns to equal length)
    diffs_sheets: Dict[str, pd.DataFrame] = {}
    for a in diseases:
        for b in diseases:
            if a == b:
                continue
            A, B = genes_by_dis[a], genes_by_dis[b]
            cols = {
                "shared": sorted(A & B) or ["-"],
                f"{a}-only": sorted(A - B) or ["-"],
                f"{b}-only": sorted(B - A) or ["-"],
            }
            cols = _pad_columns_to_equal_length(cols, pad="-")
            diffs_sheets[f"{a} vs {b}"] = pd.DataFrame(cols)

    # ---- write tables
    shared_counts.to_csv(out_dir / "pairwise_shared_counts.csv")
    jaccard.to_csv(out_dir / "pairwise_jaccard.csv")
    pvals.to_csv(out_dir / "pairwise_overlap_pvals.csv")
    coverage.to_csv(out_dir / "coverage_per_disease.csv", index=False)

    _write_lists_excel(out_dir / "shared_genes.xlsx", {"shared_by_pair": shared_names})
    _write_lists_excel(out_dir / "unique_genes.xlsx", uniq_sheets)
    _write_lists_excel(out_dir / "pairwise_differences.xlsx", diffs_sheets)

    # ---- plots (dynamic)
    _plot_upset_like(genes_by_dis, out_dir / "upset_like.png", cfg)
    _plot_jaccard_heatmap(diseases, jaccard, out_dir / "jaccard_heatmap.png", cfg, annotate=True)
    _plot_pairwise_diff_grid(diseases, genes_by_dis, out_dir / "pairwise_diff_grid.png", cfg)
    _plot_coverage_bars({d: len(genes_by_dis[d]) for d in diseases}, out_dir / "coverage_bars.png", cfg)
    _plot_unique_shared_stack(genes_by_dis, out_dir / "unique_shared_core.png", cfg)

    # Fisher panel with stable scaling
    _plot_fisher_heatmap(
        diseases, pvals,
        title="Fisher (greater) p-values",
        out_png=out_dir / "pairwise_fisher_heatmap.png",
        cfg=cfg
    )

    _plot_story_panel(genes_by_dis, jaccard, out_dir / "story_panel.png", cfg)

def run_multi_pathway_outputs(requested_pathways: List[str], views: Dict[str, pd.DataFrame], out_root: Path, cfg: PCConfig) -> None:
    """
    Rich multi-pathway comparison outputs (dynamic rendering):
      Tables
      - pathway_disease_matrix.csv
      - pathway_summary.csv
      - disease_summary.csv
      - pathway_pairwise_overall_counts.csv
      - pathway_pairwise_overall_jaccard.csv
      - per_disease/pathway_pairwise_jaccard_<DISEASE>.csv
      - per_disease/pathway_pairwise_fisher_<DISEASE>.csv
      - pairwise_differences_by_pathway.xlsx
      - top_genes_per_pathway.xlsx

      Figures
      - multi_pathway_heatmap.(png|svg) [annotated if small]
      - pathway_pairwise_overall_jaccard.(png|svg)
      - per_disease/pathway_pairwise_jaccard_<DISEASE>.(png|svg)
      - per_disease/pathway_pairwise_fisher_<DISEASE>.(png|svg, -log10(p) scale)
      - upset_pathways.(png|svg)
      - unique_shared_core_pathways.(png|svg)
      - pathway_story_panel.(png|svg)
    """
    out_root.mkdir(parents=True, exist_ok=True)
    df = _pick_view(views, cfg)
    if df.empty:
        logging.warning("[multi] empty data for selected view")
        return

    pathways = [str(p) for p in requested_pathways]

    # ---------------- 1) Disease × Pathway counts ----------------
    rows = []
    per_pw_per_dis: Dict[str, Dict[str, Set[str]]] = {}
    for pw in pathways:
        sub_genes, _ = _collect_sets_for_pathway(df, pw)
        per_pw_per_dis[pw] = sub_genes
        for dis, gset in sub_genes.items():
            rows.append({"pathway": pw, "disease": dis, "n_genes": len(gset)})
    mat = pd.DataFrame(rows)
    if mat.empty:
        logging.warning("[multi] no rows created")
        return

    pivot = mat.pivot_table(index="pathway", columns="disease",
                            values="n_genes", fill_value=0, aggfunc="max")
    if set(pathways).issubset(set(pivot.index)):
        pivot = pivot.loc[pathways]
    pivot.to_csv(out_root / "pathway_disease_matrix.csv")

    # Heatmap (dynamic)
    items_r = list(pivot.index)
    items_c = list(pivot.columns)
    labels_x = _wrap_labels(items_c, width=20)
    labels_y = _wrap_labels(items_r, width=20)
    max_len = max(max(len(s) for s in labels_x) if labels_x else 10,
                  max(len(s) for s in labels_y) if labels_y else 10)
    w, h = _auto_figsize_for_heatmap(len(items_r), len(items_c), max_len)
    font_x = _auto_tick_font(len(items_c), base=10, min_size=6)
    font_y = _auto_tick_font(len(items_r), base=10, min_size=6)

    fig = plt.figure(figsize=(w, h))
    ax = fig.add_subplot(111)
    im = ax.imshow(pivot.values, aspect="auto")
    ax.set_xticks(range(len(items_c)), labels_x, rotation=45, ha="right", fontsize=font_x)
    ax.set_yticks(range(len(items_r)), labels_y, fontsize=font_y)
    ax.set_title("# pathway genes detected")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel("count", rotation=270, va="bottom")
    if _auto_ann_ok(len(items_r), len(items_c), max_cells=300):
        _annotate_matrix(ax, pivot.values, "{:.0f}", fontsize=max(6, min(font_x, font_y)-1))
    fig.tight_layout()
    _save_fig(fig, out_root / "multi_pathway_heatmap.png", cfg.dpi)

    # ---------------- 2) Summaries ----------------
    pw_summary_rows = []
    for pw in pathways:
        vals = mat.loc[mat["pathway"].eq(pw), "n_genes"].tolist()
        nz = [v for v in vals if v > 0]
        union_size = 0
        if pw in per_pw_per_dis:
            seen = set()
            for s in per_pw_per_dis[pw].values():
                seen |= set(s)
            union_size = len(seen)
        pw_summary_rows.append({
            "pathway": pw,
            "mean_genes_per_disease": float(np.mean(vals)) if vals else 0.0,
            "median_genes_per_disease": float(np.median(vals)) if vals else 0.0,
            "prevalence_n_diseases": int(len(nz)),
            "prevalence_frac": float(len(nz) / max(1, len(items_c))),
            "total_genes_union_across_diseases": int(union_size),
        })
    pd.DataFrame(pw_summary_rows).to_csv(out_root / "pathway_summary.csv", index=False)

    dis_summary_rows = []
    for dis in items_c:
        union = set()
        n_pw_hits = 0
        for pw in pathways:
            gset = per_pw_per_dis.get(pw, {}).get(dis, set())
            if gset:
                n_pw_hits += 1
            union |= set(gset)
        dis_summary_rows.append({
            "disease": dis,
            "unique_genes_across_selected_pathways": len(union),
            "n_pathways_with_hits": int(n_pw_hits),
        })
    pd.DataFrame(dis_summary_rows).to_csv(out_root / "disease_summary.csv", index=False)

    # ---------------- 3) Pathway×Pathway (overall across diseases) ----------------
    overall_sets, per_dis_sets = _pathway_sets_across_diseases(df, pathways)
    M_counts_overall = _pairwise_matrix(pathways, overall_sets, how="count")
    M_jacc_overall   = _pairwise_matrix(pathways, overall_sets, how="jaccard")
    M_counts_overall.to_csv(out_root / "pathway_pairwise_overall_counts.csv")
    M_jacc_overall.to_csv(out_root / "pathway_pairwise_overall_jaccard.csv")
    _plot_jaccard_heatmap(pathways, M_jacc_overall, out_root / "pathway_pairwise_overall_jaccard.png", cfg, annotate=True)

    # ---------------- 4) Per-disease pathway×pathway (Jaccard + Fisher) ----------------
    per_dis_dir = out_root / "per_disease"
    per_dis_dir.mkdir(parents=True, exist_ok=True)
    for dis, sets_by_pw in per_dis_sets.items():
        if not sets_by_pw:
            continue
        MJ = _pairwise_matrix(pathways, sets_by_pw, how="jaccard")
        MJ.to_csv(per_dis_dir / f"pathway_pairwise_jaccard_{_safe_sheet_name(dis)}.csv")
        _plot_jaccard_heatmap(pathways, MJ, per_dis_dir / f"pathway_pairwise_jaccard_{_safe_sheet_name(dis)}.png", cfg, annotate=True)

        MP = _pairwise_fisher(pathways, sets_by_pw, universe=None)
        MP.to_csv(per_dis_dir / f"pathway_pairwise_fisher_{_safe_sheet_name(dis)}.csv")
        _plot_fisher_heatmap(
            pathways, MP,
            title=f"Fisher (greater) p-values — {dis}",
            out_png=per_dis_dir / f"pathway_pairwise_fisher_{_safe_sheet_name(dis)}.png",
            cfg=cfg
        )

    # ---------------- 5) UpSet-like across pathways (overall sets) ----------------
    _plot_upset_like(overall_sets, out_root / "upset_pathways.png", cfg)

    # ---------------- 6) Unique/shared/core per pathway (overall sets) ----------------
    _plot_unique_shared_stack(overall_sets, out_root / "unique_shared_core_pathways.png", cfg)

    # ---------------- 7) Publication panel ----------------
    _plot_pathway_story_panel(pathways, overall_sets, M_jacc_overall, out_root / "pathway_story_panel.png", cfg)

    # ---------------- 8) Excel workbooks ----------------
    diffs_sheets: Dict[str, pd.DataFrame] = {}
    for a in pathways:
        for b in pathways:
            if a == b:
                continue
            A, B = overall_sets[a], overall_sets[b]
            cols = {
                "shared": sorted(A & B) or ["-"],
                f"{a}-only": sorted(A - B) or ["-"],
                f"{b}-only": sorted(B - A) or ["-"],
            }
            cols = _pad_columns_to_equal_length(cols, pad="-")
            diffs_sheets[f"{a} vs {b}"] = pd.DataFrame(cols)
    _write_lists_excel(out_root / "pairwise_differences_by_pathway.xlsx", diffs_sheets)

    sheets_top: Dict[str, pd.DataFrame] = {}
    for pw in pathways:
        freq = {}
        for dis, sets_by_pw in per_dis_sets.items():
            for g in sets_by_pw.get(pw, set()):
                freq[g] = freq.get(g, 0) + 1
        rows_top = [{"gene": g, "n_diseases": c} for g, c in sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))]
        sheets_top[pw] = pd.DataFrame(rows_top) if rows_top else pd.DataFrame({"gene": [], "n_diseases": []})
    _write_lists_excel(out_root / "top_genes_per_pathway.xlsx", sheets_top)

    logging.info("[multi] comparison analysis completed.")
