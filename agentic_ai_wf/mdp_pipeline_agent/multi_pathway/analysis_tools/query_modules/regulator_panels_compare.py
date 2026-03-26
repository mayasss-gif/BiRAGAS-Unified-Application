#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import numpy as np
import argparse, json


from .common_io import ensure_dir, read_table, first_existing
from .common_plotting import plot_heatmap
from .common_scoring import try_cluster_order


def _list_diseases(root: Path) -> List[Path]:
    base = root
    gl = root / "GL_enrich"
    gc = root / "GC_enrich"
    if gl.exists():
        base = gl
    elif gc.exists():
        base = gc

    skip = {"baseline_consensus", "comparison", "results", "OmniPath_cache", "jsons_all_folder"}
    out: List[Path] = []
    for p in sorted(base.iterdir()):
        if p.is_dir() and p.name not in skip:
            out.append(p)
    return out


def _load_tf_scores(disease_dir: Path) -> Optional[pd.DataFrame]:
    # Prefer viper_tf_scores.tsv if present; else ulm_collectri_tf_scores.tsv
    fp = first_existing(disease_dir / "viper_tf_scores.tsv", disease_dir / "ulm_collectri_tf_scores.tsv")
    if fp is None:
        return None
    df = read_table(fp)
    return df


def run_regulator_panels_compare(root: str, out: str, top_n: int = 40, cluster: bool = True) -> Dict[str, str]:
    root_p = Path(root).expanduser().resolve()
    out_p = ensure_dir(Path(out).expanduser().resolve())
    tables = ensure_dir(out_p / "tables")
    plots = ensure_dir(out_p / "plots")

    diseases = _list_diseases(root_p)
    if not diseases:
        return {"out": str(out_p), "note": "No disease folders found."}

    # Build a union TF matrix by taking best available columns: try TF + score
    rows = []
    for d in diseases:
        df = _load_tf_scores(d)
        if df is None or df.empty:
            continue

        # heuristic columns
        tf_col = None
        score_col = None
        for c in df.columns:
            cl = str(c).lower()
            if tf_col is None and ("tf" in cl or "regulator" in cl or "name" == cl):
                tf_col = c
            if score_col is None and ("score" in cl or "activity" in cl or "nes" in cl):
                score_col = c
        if tf_col is None:
            tf_col = df.columns[0]
        if score_col is None and len(df.columns) > 1:
            score_col = df.columns[1]
        if score_col is None:
            continue

        tmp = df[[tf_col, score_col]].copy()
        tmp.columns = ["TF", "Score"]
        tmp["Score"] = pd.to_numeric(tmp["Score"], errors="coerce")
        tmp = tmp.dropna(subset=["Score"])
        # keep top by absolute score
        tmp["Abs"] = tmp["Score"].abs()
        tmp = tmp.sort_values("Abs", ascending=False).head(top_n)
        tmp["Disease"] = d.name
        rows.append(tmp[["Disease", "TF", "Score"]])

    if not rows:
        return {"out": str(out_p), "note": "No TF/regulator score files found."}

    long_df = pd.concat(rows, ignore_index=True)
    long_path = tables / "tf_scores_long.tsv"
    long_df.to_csv(long_path, sep="\t", index=False)

    mat = long_df.pivot_table(index="Disease", columns="TF", values="Score", aggfunc="mean").fillna(0.0)
    mat_path = tables / "tf_scores_matrix.tsv"
    mat.to_csv(mat_path, sep="\t")

    order = try_cluster_order(mat.values.astype(float)) if cluster else None
    plot_heatmap(mat, plots / "tf_scores_heatmap.png", "Regulator/TF activity (top |score| per disease)", cluster_order=order)

    return {"out": str(out_p), "tf_long": str(long_path), "tf_matrix": str(mat_path), "plot": str(plots / "tf_scores_heatmap.png")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--top-n", type=int, default=40)
    ap.add_argument("--no-cluster", action="store_true")
    args = ap.parse_args()
    res = run_regulator_panels_compare(args.root, args.out, top_n=args.top_n, cluster=(not args.no_cluster))
    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
