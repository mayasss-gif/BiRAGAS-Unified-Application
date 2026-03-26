# mdp_engine/crosstalk.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Set

import pandas as pd
from .exceptions import DataError, ValidationError


def jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 0.0
    inter = len(a & b)
    uni = len(a | b)
    return float(inter / uni) if uni else 0.0


@dataclass(frozen=True)
class CrosstalkEdge:
    p1: str
    p2: str
    shared_genes: int
    jaccard: float
    corr: float


def build_crosstalk_network(
    pathways: Dict[str, Set[str]],
    pathway_activity: "pd.DataFrame",
    jaccard_min: float = 0.05,
    corr_min: float = 0.30,
    corr_method: str = "spearman",
) -> List[CrosstalkEdge]:
    if not pathways:
        raise DataError("pathways is empty")
    if pathway_activity is None or pathway_activity.empty:
        raise DataError("pathway_activity is empty")
    if corr_method not in {"spearman", "pearson"}:
        raise ValidationError("corr_method must be spearman or pearson")
    if not (0 <= jaccard_min <= 1):
        raise ValidationError("jaccard_min must be in [0,1]")
    if not (0 <= corr_min <= 1):
        raise ValidationError("corr_min must be in [0,1]")

    cols = set(pathway_activity.columns.astype(str))
    keys = [p for p in pathways.keys() if p in cols]
    if len(keys) < 2:
        raise DataError("Need at least 2 pathways present in pathway_activity columns")

    act = pathway_activity[keys].copy()
    act = act.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    corr = act.corr(method=corr_method).fillna(0.0)

    edges: List[CrosstalkEdge] = []
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            p1, p2 = keys[i], keys[j]
            g1, g2 = set(pathways[p1]), set(pathways[p2])
            jac = jaccard(g1, g2)
            c = float(corr.loc[p1, p2]) if (p1 in corr.index and p2 in corr.columns) else 0.0

            if jac >= jaccard_min or abs(c) >= corr_min:
                edges.append(
                    CrosstalkEdge(
                        p1=str(p1),
                        p2=str(p2),
                        shared_genes=len(g1 & g2),
                        jaccard=float(jac),
                        corr=float(c),
                    )
                )

    edges.sort(key=lambda e: max(abs(e.corr), e.jaccard), reverse=True)
    return edges
