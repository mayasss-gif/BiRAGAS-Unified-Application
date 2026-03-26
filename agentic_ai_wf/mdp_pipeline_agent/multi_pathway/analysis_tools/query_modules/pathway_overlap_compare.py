#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .common_io import ensure_dir
from .common_plotting import plot_heatmap


def _find_json_bundle(root: Path) -> Path:
    # Prefer consolidated JSONs if present
    cand = root / "results" / "all_jsons"
    if cand.exists() and cand.is_dir():
        return cand
    cand2 = root / "jsons_all_folder"
    if cand2.exists() and cand2.is_dir():
        return cand2
    # fallback: look for results/all_jsons in children
    for p in root.rglob("all_jsons"):
        if p.is_dir():
            return p
    return root


def _load_one_json(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def _extract_pathway_entities(data: dict) -> Dict[str, set]:
    """
    Tries to robustly interpret your overlap JSON.
    We only need: pathway -> set(entities)
    """
    out: Dict[str, set] = {}
    if isinstance(data, dict):
        # common structure: {"pathways": {pathway: {... entities ...}}}
        if "pathways" in data and isinstance(data["pathways"], dict):
            for pw, obj in data["pathways"].items():
                ents = set()
                if isinstance(obj, dict):
                    for k in ["entities", "entity_list", "overlapping_entities", "items"]:
                        if k in obj and isinstance(obj[k], list):
                            ents = set(map(str, obj[k]))
                            break
                out[str(pw)] = ents
            return out

        # alternate: top-level keys are pathways
        for pw, obj in data.items():
            ents = set()
            if isinstance(obj, dict):
                for k in ["entities", "entity_list", "overlapping_entities", "items"]:
                    if k in obj and isinstance(obj[k], list):
                        ents = set(map(str, obj[k]))
                        break
            elif isinstance(obj, list):
                ents = set(map(str, obj))
            out[str(pw)] = ents
    return out


def run_pathway_overlap_compare(root: str, out: str, top_k_pathways: int = 40) -> Dict[str, str]:
    root_p = Path(root).expanduser().resolve()
    out_p = ensure_dir(Path(out).expanduser().resolve())
    tables = ensure_dir(out_p / "tables")
    plots = ensure_dir(out_p / "plots")

    bundle = _find_json_bundle(root_p)
    json_files = sorted([p for p in bundle.glob("*.json") if p.is_file()])
    if not json_files:
        return {"out": str(out_p), "note": f"No JSON files found under {bundle}"}

    disease_pw: Dict[str, Dict[str, set]] = {}
    for jf in json_files:
        disease = jf.stem.replace("_pathway_entity_overlap", "").replace("pathway_entity_overlap", "")
        data = _load_one_json(jf)
        disease_pw[disease] = _extract_pathway_entities(data)

    diseases = sorted(disease_pw.keys())
    # choose top pathways by global frequency (how many diseases have them)
    all_pathways = {}
    for d, mp in disease_pw.items():
        for pw in mp.keys():
            all_pathways[pw] = all_pathways.get(pw, 0) + 1
    top_pathways = [pw for pw, _ in sorted(all_pathways.items(), key=lambda x: x[1], reverse=True)[:top_k_pathways]]

    # matrix: disease×pathway = entity-count
    mat = pd.DataFrame(0, index=diseases, columns=top_pathways, dtype=float)
    for d in diseases:
        for pw in top_pathways:
            mat.loc[d, pw] = float(len(disease_pw[d].get(pw, set())))

    mat_path = tables / "disease_pathway_entitycount.tsv"
    mat.to_csv(mat_path, sep="\t")

    plot_heatmap(mat, plots / "heatmap_entitycount.png", "Disease × Pathway (entity-count from overlap JSON)")

    return {"out": str(out_p), "matrix": str(mat_path), "plot": str(plots / "heatmap_entitycount.png")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--top-k", type=int, default=40)
    args = ap.parse_args()
    res = run_pathway_overlap_compare(args.root, args.out, top_k_pathways=args.top_k)
    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
