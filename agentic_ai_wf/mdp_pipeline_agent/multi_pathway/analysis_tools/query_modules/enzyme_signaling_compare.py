#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

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


def _find_ptm_file(disease_dir: Path) -> Optional[Path]:
    # New path
    p = disease_dir / "Enzyme_and_Signaling"
    if p.exists():
        for f in p.glob("*_PTM_kinase_activity.csv"):
            return f
    # older or alternate
    q = disease_dir / "NEW_omnipath_outputs"
    if q.exists():
        for f in q.glob("*_PTM_kinase_activity.tsv"):
            return f
    return None


def _find_intercell_file(disease_dir: Path) -> Optional[Path]:
    p = disease_dir / "Enzyme_and_Signaling"
    if p.exists():
        for f in p.glob("*_Intercell_roles.csv"):
            return f
    q = disease_dir / "NEW_omnipath_outputs"
    if q.exists():
        for f in q.glob("*_Intercell_roles.tsv"):
            return f
    return None


def run_enzyme_signaling_compare(root: str, out: str, top_n: int = 30, cluster: bool = True) -> Dict[str, str]:
    root_p = Path(root).expanduser().resolve()
    out_p = ensure_dir(Path(out).expanduser().resolve())
    tables = ensure_dir(out_p / "tables")
    plots = ensure_dir(out_p / "plots")

    diseases = _list_diseases(root_p)

    # PTM kinase activity
    ptm_rows = []
    for d in diseases:
        fp = _find_ptm_file(d)
        if fp is None:
            continue
        df = read_table(fp)
        if df.empty:
            continue

        # heuristic: kinase col + activity col
        kin_c = None
        act_c = None
        for c in df.columns:
            cl = str(c).lower()
            if kin_c is None and ("kinase" in cl or "enzyme" in cl or "name" == cl):
                kin_c = c
            if act_c is None and ("activity" in cl or "score" in cl or "z" == cl):
                act_c = c
        if kin_c is None:
            kin_c = df.columns[0]
        if act_c is None and len(df.columns) > 1:
            act_c = df.columns[1]
        if act_c is None:
            continue

        tmp = df[[kin_c, act_c]].copy()
        tmp.columns = ["Kinase", "Activity"]
        tmp["Activity"] = pd.to_numeric(tmp["Activity"], errors="coerce")
        tmp = tmp.dropna(subset=["Activity"])
        tmp["Abs"] = tmp["Activity"].abs()
        tmp = tmp.sort_values("Abs", ascending=False).head(top_n)
        tmp["Disease"] = d.name
        ptm_rows.append(tmp[["Disease", "Kinase", "Activity"]])

    out_dict: Dict[str, str] = {"out": str(out_p)}

    if ptm_rows:
        ptm_long = pd.concat(ptm_rows, ignore_index=True)
        ptm_long_path = tables / "ptm_kinase_activity_long.tsv"
        ptm_long.to_csv(ptm_long_path, sep="\t", index=False)

        ptm_mat = ptm_long.pivot_table(index="Disease", columns="Kinase", values="Activity", aggfunc="mean").fillna(0.0)
        ptm_mat_path = tables / "ptm_kinase_activity_matrix.tsv"
        ptm_mat.to_csv(ptm_mat_path, sep="\t")

        order = try_cluster_order(ptm_mat.values.astype(float)) if cluster else None
        plot_heatmap(ptm_mat, plots / "ptm_kinase_activity_heatmap.png", "PTM kinase activity (top |activity| per disease)", cluster_order=order)

        out_dict.update({
            "ptm_long": str(ptm_long_path),
            "ptm_matrix": str(ptm_mat_path),
            "ptm_plot": str(plots / "ptm_kinase_activity_heatmap.png"),
        })
    else:
        out_dict["ptm_note"] = "No PTM kinase activity files found."

    # Intercell roles (simple count matrix by role if possible)
    ic_rows = []
    for d in diseases:
        fp = _find_intercell_file(d)
        if fp is None:
            continue
        df = read_table(fp)
        if df.empty:
            continue
        # heuristic: role column
        role_c = None
        score_c = None
        for c in df.columns:
            cl = str(c).lower()
            if role_c is None and ("role" in cl or "category" in cl or "intercell" in cl):
                role_c = c
            if score_c is None and ("score" in cl or "count" in cl or "n_" in cl):
                score_c = c
        if role_c is None:
            role_c = df.columns[0]
        if score_c is None and len(df.columns) > 1:
            score_c = df.columns[1]
        if score_c is None:
            continue

        tmp = df[[role_c, score_c]].copy()
        tmp.columns = ["Role", "Score"]
        tmp["Score"] = pd.to_numeric(tmp["Score"], errors="coerce")
        tmp = tmp.dropna(subset=["Score"])
        tmp = tmp.sort_values("Score", ascending=False).head(top_n)
        tmp["Disease"] = d.name
        ic_rows.append(tmp[["Disease", "Role", "Score"]])

    if ic_rows:
        ic_long = pd.concat(ic_rows, ignore_index=True)
        ic_long_path = tables / "intercell_roles_long.tsv"
        ic_long.to_csv(ic_long_path, sep="\t", index=False)

        ic_mat = ic_long.pivot_table(index="Disease", columns="Role", values="Score", aggfunc="mean").fillna(0.0)
        ic_mat_path = tables / "intercell_roles_matrix.tsv"
        ic_mat.to_csv(ic_mat_path, sep="\t")

        order = try_cluster_order(ic_mat.values.astype(float)) if cluster else None
        plot_heatmap(ic_mat, plots / "intercell_roles_heatmap.png", "Intercell roles (top roles per disease)", cluster_order=order)

        out_dict.update({
            "intercell_long": str(ic_long_path),
            "intercell_matrix": str(ic_mat_path),
            "intercell_plot": str(plots / "intercell_roles_heatmap.png"),
        })
    else:
        out_dict["intercell_note"] = "No intercell roles files found."

    return out_dict


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--top-n", type=int, default=30)
    ap.add_argument("--no-cluster", action="store_true")
    args = ap.parse_args()
    res = run_enzyme_signaling_compare(args.root, args.out, top_n=args.top_n, cluster=(not args.no_cluster))
    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
