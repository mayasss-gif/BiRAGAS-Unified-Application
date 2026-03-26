#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import pdist


# -------------------------
# small IO helpers
# -------------------------

def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _read_tsv(p: Path) -> Optional[pd.DataFrame]:
    if not p.exists():
        return None
    try:
        df = pd.read_csv(p, sep="\t")
        if df is None or df.empty:
            return None
        return df
    except Exception:
        return None


def _find_disease_dirs(root: Path) -> List[Path]:
    """
    Works for:
      - COUNTS/DEGS: root/<Disease>/
      - GL:         root/GL_enrich/<Disease>/
      - GC:         root/GC_enrich/<Disease>/
    """
    root = root.expanduser().resolve()
    gl = root / "GL_enrich"
    gc = root / "GC_enrich"
    if gl.exists() and gl.is_dir():
        base = gl
    elif gc.exists() and gc.is_dir():
        base = gc
    else:
        base = root

    skip = {
        "baseline_consensus", "comparison", "results", "OmniPath_cache",
        "jsons_all_folder", "CATEGORY_COMPARE", "CLIENT_DASHBOARD"
    }

    out: List[Path] = []
    for p in sorted(base.iterdir()):
        if p.is_dir() and p.name not in skip:
            out.append(p)
    return out


def _pick_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols = {str(c).strip().lower(): str(c) for c in df.columns}
    for cand in candidates:
        hit = cols.get(cand.lower())
        if hit:
            return hit
    return None


def _standardize_tf_scores(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    Output: columns [TF, Score]
    """
    tf_col = _pick_col(df, ["tf", "regulator", "source", "name", "factor", "protein"])
    score_col = _pick_col(df, ["score", "activity", "nes", "viper", "statistic", "t", "z"])

    # fallback: first col as TF if object-like
    if tf_col is None and len(df.columns) >= 1:
        c0 = df.columns[0]
        if df[c0].dtype == object:
            tf_col = c0

    # fallback: second col as score
    if score_col is None and len(df.columns) >= 2:
        score_col = df.columns[1]

    if tf_col is None or score_col is None:
        return None

    out = df[[tf_col, score_col]].copy()
    out = out.rename(columns={tf_col: "TF", score_col: "Score"})
    out["TF"] = out["TF"].astype(str).str.strip()
    out["Score"] = pd.to_numeric(out["Score"], errors="coerce")
    out = out.dropna(subset=["TF", "Score"])
    out = out[out["TF"] != ""]
    if out.empty:
        return None

    # de-dup TFs: keep strongest abs score
    out["abs"] = out["Score"].abs()
    out = out.sort_values("abs", ascending=False).drop_duplicates("TF", keep="first").drop(columns=["abs"])
    return out


def load_tf_scores(root: Path) -> Dict[str, pd.DataFrame]:
    """
    Per disease tries:
      1) viper_tf_scores.tsv
      2) ulm_collectri_tf_scores.tsv
      3) fallback: *tf*score*.tsv
    Returns disease -> standardized [TF, Score]
    """
    out: Dict[str, pd.DataFrame] = {}

    for ddir in _find_disease_dirs(root):
        disease = ddir.name

        p1 = ddir / "viper_tf_scores.tsv"
        p2 = ddir / "ulm_collectri_tf_scores.tsv"

        df = _read_tsv(p1)
        if df is None:
            df = _read_tsv(p2)

        if df is None:
            # fallback
            for h in sorted(ddir.glob("*tf*score*.tsv")):
                df = _read_tsv(h)
                if df is not None:
                    break

        if df is None:
            continue

        std = _standardize_tf_scores(df)
        if std is None or std.empty:
            continue

        out[disease] = std

    return out


def build_tf_matrix(tf_by_disease: Dict[str, pd.DataFrame], top_n: int = 50) -> pd.DataFrame:
    """
    Disease × TF matrix from union of top |Score| TFs per disease.
    """
    if not tf_by_disease:
        return pd.DataFrame()

    keep: set[str] = set()
    for _, df in tf_by_disease.items():
        sub = df.copy()
        sub["abs"] = sub["Score"].abs()
        sub = sub.sort_values("abs", ascending=False).head(top_n)
        keep |= set(sub["TF"].tolist())

    keep = {k for k in keep if k}
    if not keep:
        return pd.DataFrame()

    keep_sorted = sorted(keep)
    rows = {}
    for disease, df in tf_by_disease.items():
        m = dict(zip(df["TF"], df["Score"]))
        rows[disease] = {tf: float(m.get(tf, np.nan)) for tf in keep_sorted}

    return pd.DataFrame.from_dict(rows, orient="index")


# -------------------------
# plotting (matplotlib only)
# -------------------------

def _try_cluster_order(matrix: np.ndarray) -> Optional[np.ndarray]:

    if matrix.shape[0] < 3:
        return None

    d = pdist(matrix, metric="correlation")
    Z = linkage(d, method="average")
    return leaves_list(Z)


def plot_heatmap_matrix(mat: pd.DataFrame, title: str, out_png: Path, cluster: bool = True) -> None:
    if mat.empty:
        return

    data = mat.values.astype(float)
    diseases = mat.index.tolist()
    tfs = mat.columns.tolist()

    row_order = np.arange(len(diseases))
    if cluster:
        order = _try_cluster_order(np.nan_to_num(data, nan=0.0))
        if order is not None:
            row_order = order

    data = data[row_order, :]
    diseases = [diseases[i] for i in row_order]

    fig_w = max(10, 0.18 * len(tfs))
    fig_h = max(6, 0.25 * len(diseases))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=300)

    im = ax.imshow(np.nan_to_num(data, nan=0.0), aspect="auto", interpolation="nearest")
    ax.set_title(title)
    ax.set_xlabel("TF")
    ax.set_ylabel("Disease")

    ax.set_yticks(range(len(diseases)))
    ax.set_yticklabels(diseases, fontsize=7)

    ax.set_xticks(range(len(tfs)))
    ax.set_xticklabels(tfs, rotation=90, fontsize=6)

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.ax.tick_params(labelsize=7)

    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


# -------------------------
# main runner
# -------------------------

def run_tf_activity_compare(root: str, out: str, top_n: int = 50, cluster: bool = True) -> Dict[str, str]:
    root_p = Path(root).expanduser().resolve()
    out_p = ensure_dir(Path(out).expanduser().resolve())
    tables = ensure_dir(out_p / "tables")
    plots = ensure_dir(out_p / "plots")

    tf_by_disease = load_tf_scores(root_p)
    if not tf_by_disease:
        p = tables / "tf_scores_matrix.tsv"
        p.write_text("No TF score files found.\n", encoding="utf-8")
        return {"tf_matrix": str(p)}

    mat = build_tf_matrix(tf_by_disease, top_n=top_n)
    mat_path = tables / "tf_scores_matrix.tsv"
    mat.to_csv(mat_path, sep="\t")

    heat_png = plots / "tf_scores_heatmap.png"
    plot_heatmap_matrix(
        mat,
        title=f"TF activity (union of top {top_n} TFs per disease)",
        out_png=heat_png,
        cluster=cluster,
    )

    return {"tf_matrix": str(mat_path), "tf_heatmap": str(heat_png)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--top-n", type=int, default=50)
    ap.add_argument("--no-cluster", action="store_true")
    args = ap.parse_args()

    res = run_tf_activity_compare(args.root, args.out, top_n=args.top_n, cluster=not args.no_cluster)
    print("[ok] wrote:")
    for k, v in res.items():
        print(f"  - {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
