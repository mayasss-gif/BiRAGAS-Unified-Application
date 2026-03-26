#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd
import argparse, json
from .common_io import ensure_dir
from .common_plotting import plot_barh


def run_compare_two_diseases(per_term_long_tsv: str, out: str, disease_a: str, disease_b: str, top_n: int = 20) -> Dict[str, str]:
    """
    Takes the long table produced by category_landscape_compare (per_term_long.tsv)
    and produces quick client-grade comparisons between two diseases.
    """
    out_p = ensure_dir(Path(out).expanduser().resolve())
    tables = ensure_dir(out_p / "tables")
    plots = ensure_dir(out_p / "plots")

    df = pd.read_csv(per_term_long_tsv, sep="\t")
    df = df[df["Disease"].isin([disease_a, disease_b])].copy()
    if df.empty:
        return {"out": str(out_p), "note": "No matching diseases found in per_term_long.tsv"}

    # Weighted by -log10(q)
    df["Weight"] = df["QValue"].apply(lambda x: 0.0 if pd.isna(x) else (-__import__("math").log10(max(float(x), 1e-300))))

    agg = df.groupby(["Disease", "Main_Class"], as_index=False)["Weight"].sum()
    pivot = agg.pivot(index="Main_Class", columns="Disease", values="Weight").fillna(0.0)
    pivot["Delta_A_minus_B"] = pivot.get(disease_a, 0.0) - pivot.get(disease_b, 0.0)

    pivot_path = tables / "mainclass_weight_delta.tsv"
    pivot.to_csv(pivot_path, sep="\t")

    # Plots: top themes per disease + delta
    if disease_a in pivot.columns:
        plot_barh(pivot[disease_a], plots / f"{disease_a}_top_mainclass.png", f"{disease_a}: top Main_Class (sum -log10(q))", "sum(-log10(q))", top_n=top_n)
    if disease_b in pivot.columns:
        plot_barh(pivot[disease_b], plots / f"{disease_b}_top_mainclass.png", f"{disease_b}: top Main_Class (sum -log10(q))", "sum(-log10(q))", top_n=top_n)

    plot_barh(pivot["Delta_A_minus_B"].abs().sort_values(ascending=False), plots / "top_differences_abs.png",
              f"Top Main_Class differences |{disease_a} - {disease_b}|", "abs(delta weight)", top_n=top_n)

    return {
        "out": str(out_p),
        "delta_table": str(pivot_path),
        "plot_a": str(plots / f"{disease_a}_top_mainclass.png"),
        "plot_b": str(plots / f"{disease_b}_top_mainclass.png"),
        "plot_delta": str(plots / "top_differences_abs.png"),
    }


def main() -> int:
    
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-term-long", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--top-n", type=int, default=20)
    args = ap.parse_args()
    res = run_compare_two_diseases(args.per_term_long, args.out, args.a, args.b, top_n=args.top_n)
    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
