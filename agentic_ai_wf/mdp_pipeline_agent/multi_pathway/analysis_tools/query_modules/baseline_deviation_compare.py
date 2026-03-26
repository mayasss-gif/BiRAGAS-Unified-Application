#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pandas as pd
import numpy as np
import argparse, json

from .common_io import ensure_dir, read_table
from .common_plotting import plot_heatmap
from .common_scoring import try_cluster_order


def _list_diseases(root: Path) -> List[Path]:
    gl = root / "GL_enrich"
    gc = root / "GC_enrich"
    base = gl if gl.exists() else gc if gc.exists() else root

    skip = {"baseline_consensus", "comparison", "results", "OmniPath_cache", "jsons_all_folder"}
    out = []
    for p in sorted(base.iterdir()):
        if p.is_dir() and p.name not in skip:
            out.append(p)
    return out


def run_baseline_deviation_compare(root: str, out: str, top_k: int = 60, cluster: bool = True) -> Dict[str, str]:
    root_p = Path(root).expanduser().resolve()
    out_p = ensure_dir(Path(out).expanduser().resolve())
    tables = ensure_dir(out_p / "tables")
    plots = ensure_dir(out_p / "plots")

    diseases = _list_diseases(root_p)
    rows = []
    for d in diseases:
        fp = d / "delta_vs_consensus.tsv"
        if not fp.exists():
            continue
        df = read_table(fp)
        if df.empty:
            continue
        df = df.copy()

        # heuristic: try to find pathway/term column + deviation column
        term_c = None
        dev_c = None
        for c in df.columns:
            cl = str(c).lower()
            if term_c is None and ("pathway" in cl or "term" in cl or "name" in cl):
                term_c = c
            if dev_c is None and ("delta" in cl or "deviation" in cl or "diff" in cl or "z" == cl):
                dev_c = c
        if term_c is None:
            term_c = df.columns[0]
        if dev_c is None and len(df.columns) > 1:
            dev_c = df.columns[1]
        if dev_c is None:
            continue

        tmp = df[[term_c, dev_c]].copy()
        tmp.columns = ["Term", "Delta"]
        tmp["Delta"] = pd.to_numeric(tmp["Delta"], errors="coerce")
        tmp = tmp.dropna(subset=["Delta"])
        tmp["Abs"] = tmp["Delta"].abs()
        tmp = tmp.sort_values("Abs", ascending=False).head(top_k)
        tmp["Disease"] = d.name
        rows.append(tmp[["Disease", "Term", "Delta"]])

    if not rows:
        return {"out": str(out_p), "note": "No delta_vs_consensus.tsv files found."}

    long_df = pd.concat(rows, ignore_index=True)
    long_path = tables / "baseline_delta_long.tsv"
    long_df.to_csv(long_path, sep="\t", index=False)

    mat = long_df.pivot_table(index="Disease", columns="Term", values="Delta", aggfunc="mean").fillna(0.0)
    mat_path = tables / "baseline_delta_matrix.tsv"
    mat.to_csv(mat_path, sep="\t")

    order = try_cluster_order(mat.values.astype(float)) if cluster else None
    plot_heatmap(mat, plots / "baseline_deviation_heatmap.png", "Baseline deviation (delta vs consensus) — top |delta| terms", cluster_order=order)

    return {"out": str(out_p), "long": str(long_path), "matrix": str(mat_path), "plot": str(plots / "baseline_deviation_heatmap.png")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--top-k", type=int, default=60)
    ap.add_argument("--no-cluster", action="store_true")
    args = ap.parse_args()
    res = run_baseline_deviation_compare(args.root, args.out, top_k=args.top_k, cluster=(not args.no_cluster))
    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
