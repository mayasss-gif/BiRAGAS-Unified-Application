#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import argparse, json

from .common_io import ensure_dir, read_table, first_existing, col_ci
from .common_plotting import plot_heatmap
from .common_scoring import try_cluster_order


def _detect_base(root: Path) -> Path:
    gl = root / "GL_enrich"
    if gl.exists() and gl.is_dir():
        return gl
    gc = root / "GC_enrich"
    if gc.exists() and gc.is_dir():
        return gc
    return root


def _list_diseases(base: Path) -> List[Path]:
    skip = {"baseline_consensus", "comparison", "results", "OmniPath_cache", "jsons_all_folder", "CATEGORY_COMPARE"}
    out: List[Path] = []
    for p in sorted(base.iterdir()):
        if p.is_dir() and p.name not in skip:
            if (p / "gsea_prerank.tsv").exists() or (p / "gsea_prerank_classified.tsv").exists():
                out.append(p)
    return out


def run_direction_concordance(root: str, out: str, sig: float = 0.1, cap: int = 300, cluster: bool = True) -> Dict[str, str]:
    root_p = Path(root).expanduser().resolve()
    out_p = ensure_dir(Path(out).expanduser().resolve())
    tables = ensure_dir(out_p / "tables")
    plots = ensure_dir(out_p / "plots")

    base = _detect_base(root_p)
    diseases = _list_diseases(base)
    if not diseases:
        return {"out": str(out_p), "note": "No diseases with GSEA files found."}

    # Build per disease: pathway -> direction (+1/-1)
    per: Dict[str, Dict[str, int]] = {}
    for d in diseases:
        fp = first_existing(d / "gsea_prerank_classified.tsv", d / "gsea_prerank.tsv")
        if fp is None:
            continue
        df = read_table(fp)
        term_c = col_ci(df, "term") or col_ci(df, "pathway") or "Term"
        fdr_c = col_ci(df, "FDR q-val") or col_ci(df, "FDR")
        nes_c = col_ci(df, "NES")
        if term_c not in df.columns or nes_c is None:
            continue
        if fdr_c and fdr_c in df.columns:
            df[fdr_c] = pd.to_numeric(df[fdr_c], errors="coerce")
            df = df.dropna(subset=[fdr_c])
            df = df[df[fdr_c] <= sig].copy()
            df = df.sort_values(fdr_c, ascending=True).head(cap)
        df[nes_c] = pd.to_numeric(df[nes_c], errors="coerce")
        df = df.dropna(subset=[nes_c])
        mp: Dict[str, int] = {}
        for _, r in df.iterrows():
            mp[str(r[term_c])] = 1 if float(r[nes_c]) >= 0 else -1
        per[d.name] = mp

    diseases_names = sorted(per.keys())
    if len(diseases_names) < 2:
        return {"out": str(out_p), "note": "Not enough diseases with NES to compute concordance."}

    # pairwise concordance: fraction of shared pathways with same sign
    mat = pd.DataFrame(np.nan, index=diseases_names, columns=diseases_names, dtype=float)
    for a in diseases_names:
        for b in diseases_names:
            A = per[a]
            B = per[b]
            shared = set(A.keys()) & set(B.keys())
            if not shared:
                mat.loc[a, b] = np.nan
                continue
            same = sum(1 for k in shared if A[k] == B[k])
            mat.loc[a, b] = float(same) / float(len(shared))

    mat_path = tables / "direction_concordance.tsv"
    mat.to_csv(mat_path, sep="\t")

    order = try_cluster_order(mat.fillna(0.0).values.astype(float)) if cluster else None
    plot_heatmap(mat.fillna(0.0), plots / "direction_concordance.png",
                 "Direction concordance (fraction same NES sign among shared sig pathways)", cluster_order=order, vmin=0.0, vmax=1.0)

    return {"out": str(out_p), "table": str(mat_path), "plot": str(plots / "direction_concordance.png")}


def main() -> int:
    
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sig", type=float, default=0.1)
    ap.add_argument("--cap", type=int, default=300)
    ap.add_argument("--no-cluster", action="store_true")
    args = ap.parse_args()
    res = run_direction_concordance(args.root, args.out, sig=args.sig, cap=args.cap, cluster=(not args.no_cluster))
    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
