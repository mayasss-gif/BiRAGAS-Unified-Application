#!/usr/bin/env python3
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Tuple
import argparse, json
import numpy as np
import pandas as pd

from .common_io import ensure_dir, safe_write_csv
from .common_json import (
    find_strict_json_root,
    iter_json_files,
    iter_pathway_entity_rows,
    load_json,
    normalize_disease_label,
)
from .common_plotting import plot_barh, plot_heatmap, plot_scatter


def _bh_fdr(p: np.ndarray) -> np.ndarray:
    """
    Benjamini-Hochberg FDR for p-values in [0,1].
    """
    p = np.asarray(p, dtype=float)
    n = p.size
    if n == 0:
        return p
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * n / (np.arange(n) + 1.0)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0.0, 1.0)
    out = np.empty_like(q)
    out[order] = q
    return out


def _safe_sig(q: float, cap: float = 50.0) -> float:
    """
    Convert q to -log10(q) safely.
    If q<=0 or nan -> clamp to a very small positive and cap.
    """
    if q is None or not np.isfinite(q) or q <= 0:
        q = 1e-300
    val = -math.log10(q)
    return float(min(cap, val))


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    u = len(a | b)
    if u == 0:
        return 0.0
    return len(a & b) / u


def run_pathway_entity_compare(root: str, out: str, fdr_max: float = 0.10, top_pathways: int = 200) -> Dict[str, str]:
    """
    Produces:
      - tables/presence_long.csv   (UP/DOWN/ANY fixed!)
      - tables/entities_long.csv
      - tables/shared_pathway_summary.csv
      - tables/disease_interconnection.csv
      - plots/shared_presence_heatmap.png
      - plots/shared_disease_similarity.png
      - plots/shared_pathway_leaderboard.png
      - plots/shared_target_volcano_*.png
    """
    out_dir = ensure_dir(Path(out))
    tables = ensure_dir(out_dir / "tables")
    plots = ensure_dir(out_dir / "plots")

    json_root = find_strict_json_root(Path(root))
    json_files = iter_json_files(json_root)
    if not json_files:
        return {"note": f"No json files found in {json_root}"}

    # -------------------------
    # Parse all rows
    # -------------------------
    all_rows: List[Dict[str, Any]] = []
    for jf in json_files:
        disease_raw = jf.stem
        disease = normalize_disease_label(disease_raw)
        payload = load_json(jf)

        for r in iter_pathway_entity_rows(payload):
            r["disease"] = disease
            # Normalize direction labels
            d = (r.get("direction") or "ALL").upper()
            if d not in {"UP", "DOWN", "ALL"}:
                d = "ALL"
            r["direction"] = d

            # Fix pathological qvals/pvals: if missing, keep 1.0
            pval = float(r.get("pval", 1.0) or 1.0)
            qval = float(r.get("qval", 1.0) or 1.0)
            if pval < 0 or not np.isfinite(pval):
                pval = 1.0
            if qval < 0 or not np.isfinite(qval):
                qval = 1.0
            r["pval"] = pval
            r["qval"] = qval

            all_rows.append(r)

    if not all_rows:
        return {"note": "Parsed 0 pathway-entity rows from JSONs (unexpected schema or empty JSONs)."}

    df = pd.DataFrame(all_rows)

    # -------------------------
    # If qval is basically all zeros, recompute BH per (entity_type, direction) using pval
    # -------------------------
    zeroish = (df["qval"] <= 0).mean()
    if zeroish >= 0.20:
        df["qval"] = df["qval"].astype(float)
        for (etype, direc), idx in df.groupby(["entity_type", "direction"]).groups.items():
            p = df.loc[idx, "pval"].astype(float).to_numpy()
            q = _bh_fdr(np.clip(p, 0.0, 1.0))
            df.loc[idx, "qval"] = q

    df["sig"] = df["qval"].astype(float).map(lambda x: _safe_sig(x, cap=50.0))

    # Save the long entity table
    out_entities = safe_write_csv(df, tables / "entities_long.csv", index=False)

    # -------------------------
    # Presence (THIS FIXES YOUR “ANY=0” BUG)
    # ANY := (UP present) OR (DOWN present) OR (ALL present)
    # -------------------------
    pres = (
        df.groupby(["disease", "pathway", "direction"])
        .size()
        .reset_index(name="n_rows")
    )

    # direction flags
    piv = pres.pivot_table(
        index=["disease", "pathway"],
        columns="direction",
        values="n_rows",
        aggfunc="sum",
        fill_value=0,
    )
    for c in ["UP", "DOWN", "ALL"]:
        if c not in piv.columns:
            piv[c] = 0

    piv["UP"] = (piv["UP"] > 0).astype(int)
    piv["DOWN"] = (piv["DOWN"] > 0).astype(int)
    piv["ALL"] = (piv["ALL"] > 0).astype(int)
    piv["ANY"] = ((piv["UP"] > 0) | (piv["DOWN"] > 0) | (piv["ALL"] > 0)).astype(int)

    presence_long = piv.reset_index()[["disease", "pathway", "UP", "DOWN", "ANY"]]
    out_presence = safe_write_csv(presence_long, tables / "presence_long.csv", index=False)

    # -------------------------
    # Shared pathways summary
    # -------------------------
    shared = (
        presence_long.groupby("pathway")["ANY"]
        .sum()
        .reset_index(name="n_diseases_any")
        .sort_values("n_diseases_any", ascending=False)
    )
    shared = shared[shared["n_diseases_any"] >= 2]
    out_shared = safe_write_csv(shared, tables / "shared_pathway_summary.csv", index=False)

    # -------------------------
    # Disease similarity (Jaccard on ANY)
    # -------------------------
    diseases = sorted(presence_long["disease"].unique().tolist())
    pathway_sets = {
        d: set(presence_long.loc[(presence_long["disease"] == d) & (presence_long["ANY"] == 1), "pathway"].tolist())
        for d in diseases
    }

    sim = pd.DataFrame(index=diseases, columns=diseases, dtype=float)
    rows = []
    for a in diseases:
        for b in diseases:
            j = _jaccard(pathway_sets[a], pathway_sets[b])
            sim.loc[a, b] = j
            if a < b:
                rows.append(
                    {
                        "disease_a": a,
                        "disease_b": b,
                        "jaccard_any": j,
                        "shared_pathways": len(pathway_sets[a] & pathway_sets[b]),
                        "union_pathways": len(pathway_sets[a] | pathway_sets[b]),
                    }
                )
    out_inter = safe_write_csv(pd.DataFrame(rows), tables / "disease_interconnection.csv", index=False)

    # Jaccard must be 0..1 (THIS FIXES YOUR FLAT/WEIRD HEATMAP SCALE)
    out_sim_png = plot_heatmap(
        sim.fillna(0.0),
        plots / "shared_disease_similarity.png",
        title="Disease–Disease Similarity (Jaccard, ANY pathways)",
        xlabel="Disease",
        ylabel="Disease",
        vmin=0.0,
        vmax=1.0,
    )

    # -------------------------
    # Shared presence heatmap (pathways x diseases) on ANY
    # -------------------------
    if not shared.empty:
        shared_paths = shared["pathway"].head(top_pathways).tolist()
        mat_any = (
            presence_long[presence_long["pathway"].isin(shared_paths)]
            .pivot_table(index="pathway", columns="disease", values="ANY", aggfunc="max", fill_value=0)
        )
        out_presence_png = plot_heatmap(
            mat_any,
            plots / "shared_presence_heatmap.png",
            title="Shared Pathways (ANY presence)",
            xlabel="Disease",
            ylabel="Pathway",
            vmin=0.0,
            vmax=1.0,
        )
    else:
        out_presence_png = plot_heatmap(
            pd.DataFrame(),
            plots / "shared_presence_heatmap.png",
            title="Shared Pathways (ANY presence)",
            xlabel="Disease",
            ylabel="Pathway",
        )

    # -------------------------
    # Simple leaderboard: pathways ranked by #diseases + max significance
    # -------------------------
    max_sig = (
        df.groupby(["pathway"])["sig"].max().reset_index(name="max_sig")
    )
    lead = shared.merge(max_sig, on="pathway", how="left") if not shared.empty else max_sig
    lead["score"] = (lead.get("n_diseases_any", 1).fillna(1) * 10.0) + lead["max_sig"].fillna(0.0)
    lead = lead.sort_values("score", ascending=False).head(top_pathways)

    out_lead_csv = safe_write_csv(lead, tables / "top_pathways_cross_disease.csv", index=False)
    out_lead_png = plot_barh(
        lead.set_index("pathway")["score"],
        plots / "shared_pathway_leaderboard.png",
        title="Pathway Intelligence Leaderboard (fixed scoring)",
        xlabel="score",
        top_n=25,
    )

    # -------------------------
    # Volcano plots per entity_type (robust sig + labels)
    # -------------------------
    volcanos: Dict[str, str] = {}
    for etype in sorted(df["entity_type"].unique().tolist()):
        d0 = df[df["entity_type"] == etype].copy()
        if d0.empty:
            continue

        # median log10(OR) and max sig across all occurrences
        d0["log10_or"] = d0["OR"].astype(float).map(lambda x: math.log10(x) if x and x > 0 else 0.0)
        agg = (
            d0.groupby("entity")
            .agg(median_log10_or=("log10_or", "median"), max_sig=("sig", "max"), n=("entity", "size"))
            .reset_index()
            .sort_values(["max_sig", "median_log10_or"], ascending=False)
        )

        # Keep plot readable
        agg_plot = agg.head(250)

        volcanos[etype] = plot_scatter(
            agg_plot.rename(columns={"entity": "label"}),
            x="median_log10_or",
            y="max_sig",
            out_png=plots / f"shared_target_volcano_{etype}.png",
            title=f"Target Volcano — {etype.upper()}",
            xlabel="median log10(OR)",
            ylabel="max sig (−log10 q)",
            label_col="label",
            label_top_n=15,
        )

        # also export top targets table
        safe_write_csv(agg.head(200), tables / f"top200_targets_{etype}.csv", index=False)

    return {
        "json_root": str(json_root),
        "entities_long": out_entities,
        "presence_long": out_presence,
        "shared_pathway_summary": out_shared,
        "disease_interconnection": out_inter,
        "top_pathways_cross_disease": out_lead_csv,
        "plot.shared_disease_similarity": out_sim_png,
        "plot.shared_presence_heatmap": out_presence_png,
        "plot.shared_pathway_leaderboard": out_lead_png,
        **{f"plot.volcano.{k}": v for k, v in volcanos.items()},
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fdr-max", type=float, default=0.10)
    ap.add_argument("--top-pathways", type=int, default=200)
    args = ap.parse_args()

    res = run_pathway_entity_compare(args.root, args.out, fdr_max=args.fdr_max, top_pathways=args.top_pathways)
    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
