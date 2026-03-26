# mdp_engine/integration.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Set
import pandas as pd


from .exceptions import DataError, ValidationError


@dataclass(frozen=True)
class CrossDiseaseResult:
    active_pathways: Dict[str, Set[str]]
    shared_pathways: List[Dict[str, object]]
    disease_specific: Dict[str, Set[str]]
    similarity_matrix: "pd.DataFrame"
    pathway_profiles: "pd.DataFrame"


def cross_disease_pathway_comparison(
    pathway_profiles: "pd.DataFrame",
    z_threshold: float = 2.0,
    min_shared: int = 2,
) -> CrossDiseaseResult:
    if pathway_profiles is None or pathway_profiles.empty:
        raise DataError("pathway_profiles is empty")
    if float(z_threshold) <= 0:
        raise ValidationError("z_threshold must be > 0")
    if int(min_shared) < 2:
        raise ValidationError("min_shared must be >= 2")

    df = pathway_profiles.copy()
    df = df.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    diseases = list(df.columns.astype(str))

    active: Dict[str, Set[str]] = {}
    for disease in diseases:
        x = df[disease].astype(float)
        mu = float(x.mean())
        sd = float(x.std(ddof=0))
        if sd == 0.0:
            active[disease] = set()
            continue
        z = (x - mu) / sd
        act = set(df.index[(z.abs() > float(z_threshold))].astype(str))
        active[disease] = act

    shared_list: List[Dict[str, object]] = []
    for pathway in df.index.astype(str):
        ds = [d for d in diseases if pathway in active.get(d, set())]
        if len(ds) >= int(min_shared):
            shared_list.append({"pathway": pathway, "diseases": ds, "n_diseases": len(ds)})

    disease_specific: Dict[str, Set[str]] = {}
    for disease in diseases:
        spec = set(active.get(disease, set()))
        for other in diseases:
            if other != disease:
                spec -= active.get(other, set())
        disease_specific[disease] = spec

    sim = df.T.corr(method="spearman").fillna(0.0)

    return CrossDiseaseResult(
        active_pathways=active,
        shared_pathways=shared_list,
        disease_specific=disease_specific,
        similarity_matrix=sim,
        pathway_profiles=df,
    )
