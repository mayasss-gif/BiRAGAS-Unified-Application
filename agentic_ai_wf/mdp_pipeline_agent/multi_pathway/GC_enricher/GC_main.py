# GC_enricher/GC_main.py
from __future__ import annotations
import argparse, glob, json
from pathlib import Path
from typing import Optional, Dict

import pandas as pd

# Reuse your GL engine but switch it into GC mode
from ..genelist_mdp.GL_enrich import GLEnrichConfig, run_gl_enrichment_for_cohort

__all__ = ["run_gc_for_file", "run_gc_folder", "main"]

def _is_genecards_json(p: Path, gene_key: str, score_key: Optional[str]) -> bool:
    if p.suffix.lower() != ".json":
        return False
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
        return isinstance(raw, dict) and isinstance(raw.get("data"), dict) and gene_key in raw["data"]
    except Exception:
        return False

def run_gc_for_file(
    json_path: str | Path,
    out_dir: str | Path,
    gene_key: str = "gene_symbol",
    score_key: Optional[str] = "disorder_score",
    fdr: float = 0.05,
    topk: int = 500,
    perms: int = 100,
    minset: int = 10,
    maxset: int = 2000,
    seed: int = 13,
    collapse: bool = True,
    jaccard: float = 0.7,
    hgnc_map_file: Optional[str] = None,
) -> Dict:
    """
    Run GC enrichment for a single GeneCards-style JSON file.

    IMPORTANT: out_dir must be the GC_enrich BASE folder (NOT the disease subfolder).
    The GL engine will create <out_dir>/<Disease>/... and write its usual files
    (including ALL_COMBINED etc.) there.
    """
    json_path = Path(json_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not _is_genecards_json(json_path, gene_key, score_key):
        raise ValueError(f"Not a valid GeneCards JSON (missing data.{gene_key}): {json_path}")

    cfg = GLEnrichConfig(
        enabled=True,
        use_gene_list=False,          # GC mode
        top_k=topk,
        prerank_permutations=perms,
        prerank_min_size=minset,
        prerank_max_size=maxset,
        seed=seed,
        fdr_threshold=fdr,
        collapse_redundancy=collapse,
        jaccard_threshold=jaccard,
        hgnc_map_file=hgnc_map_file,
    )

    cohort = json_path.stem
    print(f"[GC] Running cohort={cohort} from {json_path}")

    # Pass BASE only; GL_enrich will create <BASE>/<cohort> internally.
    m = run_gl_enrichment_for_cohort(
        cohort_name=cohort,
        scores_table=str(json_path),     # GL_enrich will load JSON in GC mode
        out_root=out_dir,                # BASE folder (…/GC_enrich)
        cfg=cfg,
        gene_col=gene_key,
        score_col=score_key or "disorder_score",
    )

    # Convenience copy of prerank into the cohort folder that GL just created
    # (this does NOT change or remove any of GL’s native outputs, including ALL_COMBINED)
    cohort_dir = Path(m.get("out_dir", out_dir / cohort))
    cohort_dir.mkdir(parents=True, exist_ok=True)

    src = None
    if m.get("prerank_all") and Path(m["prerank_all"]).exists():
        src = Path(m["prerank_all"])
    elif m.get("prerank_sig") and Path(m["prerank_sig"]).exists():
        src = Path(m["prerank_sig"])

    if src is not None:
        dst = cohort_dir / "gsea_prerank.tsv"
        try:
            df = pd.read_csv(src, sep="\t")
            df.to_csv(dst, sep="\t", index=False)
            print(f"[GC] wrote: {dst}")
        except Exception as e:
            print(f"[GC] WARN: could not create gsea_prerank.tsv: {e}")

    return m

def run_gc_folder(
    input_dir: str,
    out_root: str,
    gene_key: str = "gene_symbol",
    score_key: Optional[str] = "disorder_score",
    fdr: float = 0.05,
    topk: int = 500,
    perms: int = 100,
    minset: int = 10,
    maxset: int = 2000,
    seed: int = 13,
    collapse: bool = True,
    jaccard: float = 0.7,
    hgnc_map_file: Optional[str] = None,
) -> bool:
    """
    Scan a folder for GeneCards JSONs and run GC enrichment per file.

    Final layout (no double nesting):
      <out_root>/GC_enrich/<Disease>/...
    """
    in_dir = Path(input_dir)
    out_root = Path(out_root)
    if not in_dir.is_dir():
        print(f"[GC] Input is not a directory: {in_dir}")
        return False

    base_out = out_root / "GC_enrich"   # BASE only
    base_out.mkdir(parents=True, exist_ok=True)

    files = sorted(glob.glob(str(in_dir / "*.json")))
    if not files:
        print(f"[GC] No .json files found in: {in_dir}")
        return False

    for fp in files:
        # DO NOT append disease here. Pass BASE so GL creates <BASE>/<Disease>.
        run_gc_for_file(
            json_path=fp,
            out_dir=base_out,           # BASE (…/GC_enrich)
            gene_key=gene_key,
            score_key=score_key,
            fdr=fdr, topk=topk, perms=perms, minset=minset, maxset=maxset,
            seed=seed, collapse=collapse, jaccard=jaccard,
            hgnc_map_file=hgnc_map_file,
        )

    print(f"[GC] DONE. Outputs under: {base_out}")
    return True

def _parse_args():
    ap = argparse.ArgumentParser(description="Run GC enrichment (GeneCards JSON inputs).")
    ap.add_argument("--in", dest="input_dir", required=True, help="Folder with GeneCards JSONs.")
    ap.add_argument("--out", dest="out_root", required=True, help="Output root directory.")
    ap.add_argument("--gene-key", default="gene_symbol")
    ap.add_argument("--score-key", default="disorder_score")
    ap.add_argument("--fdr", type=float, default=0.05)
    ap.add_argument("--topk", type=int, default=500)
    ap.add_argument("--perms", type=int, default=100)
    ap.add_argument("--minset", type=int, default=10)
    ap.add_argument("--maxset", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--no-collapse", action="store_true", help="Disable redundancy collapse.")
    ap.add_argument("--jaccard", type=float, default=0.7)
    ap.add_argument("--hgnc-map", default=None)
    return ap.parse_args()

def main() -> int:
    args = _parse_args()
    ok = run_gc_folder(
        input_dir=args.input_dir,
        out_root=args.out_root,
        gene_key=args.gene_key,
        score_key=args.score_key,
        fdr=args.fdr,
        topk=args.topk,
        perms=args.perms,
        minset=args.minset,
        maxset=args.maxset,
        seed=args.seed,
        collapse=(not args.no_collapse),
        jaccard=args.jaccard,
        hgnc_map_file=args.hgnc_map,
    )
    return 0 if ok else 1

if __name__ == "__main__":
    raise SystemExit(main())
