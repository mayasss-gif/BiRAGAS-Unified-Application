from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd

from .category_landscape_compare import run_category_landscape_compare
from .common_io import ensure_dir
from .common_scoring import cosine_similarity_matrix


def run_find_similar_diseases(
    root: str,
    out: str,
    target: str,
    sig: float = 0.1,
    cap: int = 300,
    top_n: int = 10,
) -> Dict[str, str]:
    """
    Finds diseases most similar to `target` based on cosine similarity
    of the ALL TYPES Main_Class weighted profile.
    Writes:
      - tables/target_neighbors.tsv
    """
    out_p = ensure_dir(Path(out).expanduser().resolve())
    res = run_category_landscape_compare(root=root, out=str(out_p / "category_landscape"), sig=sig, cap=cap, cluster=True)

    w = pd.read_csv(res["weighted_matrix"], sep="\t", index_col=0)
    if target not in w.index:
        raise ValueError(f"Target '{target}' not found. Available: {list(w.index)[:25]} ...")

    sim = cosine_similarity_matrix(w.values.astype(float))
    sim_df = pd.DataFrame(sim, index=w.index, columns=w.index)

    neighbors = (
        sim_df.loc[target]
        .sort_values(ascending=False)
        .drop(target)
        .head(top_n)
        .reset_index()
    )
    neighbors.columns = ["Disease", "Similarity"]

    npath = out_p / "tables" / "target_neighbors.tsv"
    npath.parent.mkdir(parents=True, exist_ok=True)
    neighbors.to_csv(npath, sep="\t", index=False)

    return {**res, "target_neighbors": str(npath)}
