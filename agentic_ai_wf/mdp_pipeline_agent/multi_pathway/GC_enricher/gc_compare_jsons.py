# GC_enricher/gc_compare_jsons.py
from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import Dict, List
import pandas as pd

LAYERS = ["pathways", "mechanism", "biological_function", "cell_component", "tf", "epigenetic", "metabolites"]

def load_jsons(json_dir: Path) -> Dict[str, Dict]:
    data: Dict[str, Dict] = {}
    for p in sorted(json_dir.glob("*.json")):
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
            disease = str(obj.get("disease") or p.stem)
            data[disease] = obj
        except Exception:
            pass
    if not data:
        raise RuntimeError(f"No JSON files found in: {json_dir}")
    return data

def layer_table(all_objs: Dict[str, Dict], layer: str) -> pd.DataFrame:
    # Collect unique terms across all diseases for this layer
    term_set = set()
    for dz, obj in all_objs.items():
        terms = obj.get("layers", {}).get(layer, {}).get("terms", []) or []
        term_set.update([str(t) for t in terms])
    terms_sorted = sorted(term_set)

    # Build 0/1 presence matrix
    cols = []
    for dz in sorted(all_objs.keys()):
        present = set(obj.get("layers", {}).get(layer, {}).get("terms", []) or [] for obj in [all_objs[dz]])
        # flatten present (kept as set of strings)
        present = set(list(present)[0])
        cols.append(pd.Series([1 if t in present else 0 for t in terms_sorted], name=dz))

    if not cols:
        return pd.DataFrame()

    df = pd.concat(cols, axis=1)
    df.insert(0, "Term", terms_sorted)
    df["DiseaseCount"] = df.drop(columns=["Term"]).sum(axis=1)
    return df

def build_all(json_dir: Path, out_dir: Path) -> Path:
    objs = load_jsons(json_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Per-layer CSVs
    combined_frames: List[pd.DataFrame] = []
    for layer in LAYERS:
        df = layer_table(objs, layer)
        if df.empty:
            continue
        df.insert(0, "Layer", layer)
        df.to_csv(out_dir / f"compare_{layer}.csv", index=False)
        combined_frames.append(df)

    # ALL_COMBINED with Layer + Term + diseases + DiseaseCount
    if combined_frames:
        all_df = pd.concat(combined_frames, ignore_index=True)
        all_df.to_csv(out_dir / "ALL_COMBINED.csv", index=False)
    else:
        # Even if empty, make an empty file so downstream won't crash
        pd.DataFrame(columns=["Layer", "Term", "DiseaseCount"]).to_csv(out_dir / "ALL_COMBINED.csv", index=False)

    return out_dir / "ALL_COMBINED.csv"

def main():
    ap = argparse.ArgumentParser(description="Compare per-disease JSONs and emit presence matrices.")
    ap.add_argument("--json-dir", required=True, help="Folder with per-disease JSONs (…/results/all_jsons).")
    ap.add_argument("--out", required=True, help="Output folder for comparison CSVs.")
    args = ap.parse_args()
    all_combined = build_all(Path(args.json_dir), Path(args.out))
    print(f"[GC] Comparison written. ALL_COMBINED: {all_combined}")

if __name__ == "__main__":
    main()
