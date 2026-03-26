# pipeline_agent.py — minimal agent-facing API & local folder mapper
import os, re
from typing import Dict, List, Optional
from .harmonizer_core import run_single, run_multi

TXT_EXT = (".tsv", ".csv", ".txt", ".xlsx", ".xls")
COUNT_PAT = re.compile(r"(count|expr|matrix|table|gene|prep[_\-]?count)", re.I)
META_PAT  = re.compile(r"(meta|pheno|sample|annot|group|prep[_\-]?meta)", re.I)

def _discover_preps_local(data_root: str) -> List[Dict]:
    """Recursively find 'prep*' folders (skip our own outputs) and pair counts/meta."""
    hits: List[Dict] = []
    SKIP_DIRS = {"harmonized", "figs", "multi_geo_combined", "_combined", "__combined__"}
    OUT_FILES = {
        "expression_combined.tsv", "expression_harmonized.tsv", "metadata.tsv",
        "pca_scores.tsv", "report.json", "normalization.txt", "results_bundle.zip"
    }

    for root, _, files in os.walk(data_root):
        base = os.path.basename(root).lower()

        # only consider true prep folders
        if not base.startswith("prep"):
            continue
        # skip our own output dirs
        if base in SKIP_DIRS:
            continue
        # extra guard: if any part of the path is a skipped dir, ignore
        if any(part.lower() in SKIP_DIRS for part in root.split(os.sep)):
            continue

        # candidate files (exclude our outputs)
        files = [f for f in files if f.lower().endswith(TXT_EXT) and f.lower() not in OUT_FILES]
        if not files:
            continue

        counts = [f for f in files if COUNT_PAT.search(f)]
        metas  = [f for f in files if META_PAT.search(f)]
        if not counts or not metas:
            continue

        def base_name(s: str) -> str:
            s = os.path.splitext(s)[0]
            s = re.sub("(?i)(counts?|meta(data)?)", "", s)
            return re.sub(r"[^a-z0-9]+", "", s.lower())

        metas_by = {}
        for m in metas:
            metas_by.setdefault(base_name(m), []).append(m)

        disease = os.path.basename(os.path.dirname(root))
        rel = os.path.relpath(root, data_root).replace("\\", "/")
        pretty = f"{disease}__{rel}"
        safe = re.sub(r"[^A-Za-z0-9._-]+", "_", pretty)

        used = 0
        for c in counts:
            k = base_name(c)
            m = metas_by[k].pop(0) if k in metas_by and metas_by[k] else metas[min(used, len(metas) - 1)]
            used += 1
            hits.append({
                "name": safe,
                "disease": disease,
                "counts": os.path.join(root, c),
                "counts_name": c,
                "meta": os.path.join(root, m),
                "meta_name": m,
            })
    return hits


# ---------------- Public entrypoints ----------------
def harmonize_from_local(data_root: str, combine: bool = True, out_root: Optional[str] = None) -> Dict:
    """
    Agent entrypoint for local filesystems.
    """
    datasets = _discover_preps_local(data_root)
    if not datasets:
        raise FileNotFoundError(f"No (counts, meta) pairs found under: {data_root}")
    if len(datasets) == 1:
        d = datasets[0]
        return {"mode": "single", "result": run_single(d["counts"], d["meta"], d["counts_name"], d["meta_name"], out_mode="co_locate", create_zip=False)}
    # multi: the core will now group by disease and produce one harmonizer_<disease>.zip per disease
    return {"mode": "multi", **run_multi(datasets, attempt_combine=combine, out_root=out_root)}

def harmonize_single_paths(counts_path: str, meta_path: str, out_mode: str = "co_locate", out_root: Optional[str] = None, create_zip: bool = False) -> Dict:
    """Direct single-dataset call when partner already knows the two paths."""
    return run_single(counts_path, meta_path, os.path.basename(counts_path), os.path.basename(meta_path),
                      out_root=out_root, out_mode=out_mode, create_zip=create_zip)
