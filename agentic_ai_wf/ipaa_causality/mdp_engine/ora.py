# mdp_engine/ora.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import gseapy as gp  # type: ignore

import pandas as pd
_HAVE_PANDAS = True


from .exceptions import DataError, ValidationError
from .stats_utils import bh_fdr, fisher_exact_2x2, hypergeom_sf
from .pathway_db import Pathway

# Safe type alias: never references pd.DataFrame when pd may be None.
DataFrameT = pd.DataFrame if _HAVE_PANDAS else Any


@dataclass(frozen=True)
class OraResult:
    pid: str
    name: str
    pathway_size: int
    de_size: int
    universe_size: int
    overlap_size: int
    p_value: float
    fdr: float
    odds_ratio: float
    overlap_genes: Tuple[str, ...]
    source: str = "local"  # local or gseapy_enrichr


def hypergeometric_enrichment(
    de_genes: Set[str],
    pathway_genes: Set[str],
    universe_genes: Set[str],
) -> Tuple[float, float, Set[str]]:
    if not de_genes or not universe_genes:
        raise DataError("de_genes and universe_genes must be non-empty sets")

    if not pathway_genes:
        return 1.0, 0.0, set()

    de = set(de_genes) & set(universe_genes)
    pw = set(pathway_genes) & set(universe_genes)

    N = len(universe_genes)
    n = len(de)
    M = len(pw)
    overlap = de & pw
    k = len(overlap)

    if N == 0 or n == 0 or M == 0:
        return 1.0, 0.0, overlap

    p_value = hypergeom_sf(N=int(N), M=int(M), n=int(n), k=int(k))

    denom = n * M
    odds_ratio = (k * N) / denom if denom > 0 else 0.0
    return float(p_value), float(odds_ratio), overlap


def fisher_enrichment(
    de_genes: Set[str],
    pathway_genes: Set[str],
    universe_genes: Set[str],
    alternative: str = "greater",
) -> Tuple[float, float, Set[str]]:
    if not de_genes or not universe_genes:
        raise DataError("de_genes and universe_genes must be non-empty sets")

    de = set(de_genes) & set(universe_genes)
    pw = set(pathway_genes) & set(universe_genes)
    overlap = de & pw

    a = len(overlap)
    b = len(de - pw)
    c = len(pw - de)
    d = len(universe_genes - (de | pw))

    odds, p = fisher_exact_2x2(int(a), int(b), int(c), int(d), alternative=str(alternative))
    return float(p), float(odds), overlap


def _try_gseapy_enrichr(
    de_genes: Set[str],
    enrichr_library: str,
    organism: str = "Human",
) -> Optional[DataFrameT]:
    """
    Prefer gseapy.enrichr when available.
    NOTE: may require network (Enrichr API). If it fails, return None.
    """
    if not _HAVE_PANDAS:
        return None

    assert pd is not None  # for type checkers only


    glist = sorted({str(g).strip() for g in de_genes if g and str(g).strip()})
    if not glist:
        return None

    try:
        enr = gp.enrichr(
            gene_list=glist,
            gene_sets=enrichr_library,
            organism=organism,
            outdir=None,
            no_plot=True,
            verbose=False,
        )
    except Exception:
        return None

    res = getattr(enr, "results", None)
    if res is None or not isinstance(res, pd.DataFrame) or res.empty:
        return None
    return res


def enrichment_all_pathways(
    de_genes: Set[str],
    pathways: Dict[str, Pathway],
    universe_genes: Set[str],
    min_size: int = 10,
    method: str = "hypergeom",
    fisher_alternative: str = "greater",
    *,
    prefer_gseapy: bool = True,
    enrichr_library: Optional[str] = None,
    organism: str = "Human",
) -> Union[DataFrameT, List[OraResult]]:
    """
    Preferred behavior:
      1) If prefer_gseapy and enrichr_library provided -> try gseapy.enrichr (network).
      2) Otherwise -> local ORA across provided pathways (hypergeom or fisher), BH-FDR.

    Return type:
      - pandas.DataFrame if pandas available (and/or gseapy returns results)
      - else List[OraResult]
    """
    if method not in {"hypergeom", "fisher"}:
        raise ValidationError("method must be 'hypergeom' or 'fisher'")

    if not pathways:
        raise DataError("pathways dict is empty")
    if not universe_genes:
        raise DataError("universe_genes is empty")
    if not de_genes:
        raise DataError("de_genes is empty")

    # gseapy first (if requested)
    if prefer_gseapy and enrichr_library:
        res = _try_gseapy_enrichr(de_genes=de_genes, enrichr_library=enrichr_library, organism=organism)
        if res is not None:
            keep = res.copy()
            keep["source"] = "gseapy_enrichr"
            return keep

    # local ORA
    rows: List[OraResult] = []
    de_u = set(de_genes) & set(universe_genes)

    for pid, pw in pathways.items():
        genes = set(pw.genes) & set(universe_genes)
        if len(genes) < int(min_size):
            continue

        if method == "hypergeom":
            pval, odds, overlap = hypergeometric_enrichment(de_u, genes, universe_genes)
        else:
            pval, odds, overlap = fisher_enrichment(de_u, genes, universe_genes, alternative=fisher_alternative)

        rows.append(
            OraResult(
                pid=str(pid),
                name=str(pw.name),
                pathway_size=len(genes),
                de_size=len(de_u),
                universe_size=len(universe_genes),
                overlap_size=len(overlap),
                p_value=float(pval),
                fdr=1.0,
                odds_ratio=float(odds),
                overlap_genes=tuple(sorted(overlap)),
                source="local",
            )
        )

    if not rows:
        raise DataError("No pathways passed filters (min_size too high or no overlap).")

    qvals = bh_fdr([r.p_value for r in rows])
    rows = [
        OraResult(
            pid=r.pid,
            name=r.name,
            pathway_size=r.pathway_size,
            de_size=r.de_size,
            universe_size=r.universe_size,
            overlap_size=r.overlap_size,
            p_value=r.p_value,
            fdr=qvals[i],
            odds_ratio=r.odds_ratio,
            overlap_genes=r.overlap_genes,
            source=r.source,
        )
        for i, r in enumerate(rows)
    ]

    rows.sort(key=lambda r: (r.fdr, r.p_value))

    if _HAVE_PANDAS:
        assert pd is not None  # for type checkers only
        return pd.DataFrame([r.__dict__ for r in rows])
    return rows
