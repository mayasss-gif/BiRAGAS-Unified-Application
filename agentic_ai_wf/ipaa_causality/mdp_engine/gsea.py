# mdp_engine/gsea.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Set, Tuple, Union

import random
import numpy as np

import pandas as pd  # type: ignore
_HAVE_PANDAS = True

import gseapy as gp  # type: ignore



from .exceptions import DataError, DependencyError, ValidationError


RankedInput = Union[
    Sequence[Tuple[str, float]],   # [(gene, score), ...]
    Mapping[str, float],           # {gene: score}
    "pd.Series",                   # index=gene, values=score
]


@dataclass(frozen=True)
class GSEAResult:
    es: float
    nes: float
    p_value: float
    leading_edge: Tuple[str, ...]
    es_profile: Tuple[float, ...]
    n_permutations: int
    gene_set_size: int
    ranked_size: int
    # optional extra fields (when coming from gseapy)
    fdr_q: Optional[float] = None
    term: Optional[str] = None


def _to_ranked_list(ranked: RankedInput) -> List[Tuple[str, float]]:
    if ranked is None:
        raise ValidationError("ranked is None")

    if _HAVE_PANDAS and isinstance(ranked, pd.Series):
        items = [(str(g), float(s)) for g, s in ranked.items()]
    elif isinstance(ranked, dict):
        items = [(str(g), float(s)) for g, s in ranked.items()]
    else:
        items = [(str(g), float(s)) for g, s in ranked]  # type: ignore

    items = [(g.strip(), float(s)) for g, s in items if str(g).strip() != ""]
    items.sort(key=lambda x: x[1], reverse=True)
    if not items:
        raise DataError("ranked list is empty after parsing")
    return items


def calculate_enrichment_score(
    ranked_genes: Sequence[Tuple[str, float]],
    gene_set: Set[str],
    weight: float = 1.0,
) -> Tuple[float, Set[str], List[float]]:
    """
    Running-sum ES (classic prerank variant):
      - hit increment weighted by |score|^weight
      - miss decrement constant
      - ES = max deviation (or min if negative larger magnitude)
      - leading edge = hits up to peak
    """
    if gene_set is None:
        raise ValidationError("gene_set is None")
    if not ranked_genes:
        raise DataError("ranked_genes is empty")

    genes = [g for g, _ in ranked_genes]
    scores = np.array([s for _, s in ranked_genes], dtype=float)
    N = len(genes)

    hit = np.array([1 if g in gene_set else 0 for g in genes], dtype=int)
    Nh = int(hit.sum())
    if Nh == 0:
        return 0.0, set(), [0.0] * N
    if N == Nh:
        # all genes are hits -> trivial
        prof = [float(i + 1) / float(N) for i in range(N)]
        return 1.0, set(genes), prof

    if weight < 0:
        raise ValidationError("weight must be >= 0")

    w = np.abs(scores) ** float(weight)
    NR = float(np.sum(w * hit))
    if NR <= 0:
        return 0.0, set(), [0.0] * N

    miss_penalty = 1.0 / float(N - Nh)

    running = np.zeros(N, dtype=float)
    cur = 0.0
    for i in range(N):
        if hit[i] == 1:
            cur += float(w[i]) / NR
        else:
            cur -= miss_penalty
        running[i] = cur

    es_pos = float(np.max(running))
    es_neg = float(np.min(running))
    if abs(es_pos) >= abs(es_neg):
        es = es_pos
        peak = int(np.argmax(running))
    else:
        es = es_neg
        peak = int(np.argmin(running))

    leading_edge = {genes[i] for i in range(0, peak + 1) if hit[i] == 1}
    return es, leading_edge, running.tolist()


def _try_gseapy_prerank(
    ranked: RankedInput,
    gene_sets: Union[str, Dict[str, Set[str]]],
    n_permutations: int,
    seed: int,
    min_size: int,
    max_size: int,
) -> Optional[Dict[str, GSEAResult]]:
    """
    Try running gseapy.prerank (preferred). If missing/fails, return None.
    gene_sets: either gmt path / library name, OR dict[str,set[str]].
    """

    # Prepare ranked list for gseapy
    ranked_list = _to_ranked_list(ranked)

    if _HAVE_PANDAS:
        rnk = pd.DataFrame(ranked_list, columns=["gene", "score"])
    else:
        # gseapy expects a DataFrame or file; without pandas we won't use gseapy
        return None

    # If dict gene_sets provided, write in-memory gmt-like format via temp file (avoid: keep simple)
    # Instead, gseapy supports dict in some contexts but not consistently; safest: require gmt path/library string.
    if isinstance(gene_sets, dict):
        return None

    try:
        pre = gp.prerank(
            rnk=rnk,
            gene_sets=gene_sets,
            permutation_num=int(n_permutations),
            seed=int(seed),
            min_size=int(min_size),
            max_size=int(max_size),
            outdir=None,  # in-memory (no filesystem dependency)
            no_plot=True,
            processes=1,
            format="png",
            verbose=False,
        )
    except Exception:
        return None

    # gseapy returns res2d DataFrame with Term, ES, NES, NOM p-val, FDR q-val, Lead_genes...
    res = getattr(pre, "res2d", None)
    if res is None or not isinstance(res, pd.DataFrame) or res.empty:
        return None

    out: Dict[str, GSEAResult] = {}
    for _, row in res.iterrows():
        term = str(row.get("Term", "")).strip()
        if not term:
            continue
        es = float(row.get("ES", 0.0))
        nes = float(row.get("NES", 0.0))
        p = float(row.get("NOM p-val", row.get("NOM p-val", 1.0)))
        q = row.get("FDR q-val", None)
        fdr_q = float(q) if q is not None and str(q) != "nan" else None

        # Lead_genes can be "A;B;C" or "A/B/C"; normalize
        lg = row.get("Lead_genes", "")
        if lg is None:
            lead = ()
        else:
            txt = str(lg).replace("/", ";")
            lead = tuple(sorted({x.strip() for x in txt.split(";") if x.strip()}))

        out[term] = GSEAResult(
            es=es,
            nes=nes,
            p_value=min(max(p, 0.0), 1.0),
            leading_edge=lead,
            es_profile=tuple(),
            n_permutations=int(n_permutations),
            gene_set_size=-1,
            ranked_size=len(ranked_list),
            fdr_q=fdr_q,
            term=term,
        )
    return out or None


def gsea(
    ranked: RankedInput,
    gene_set: Set[str],
    n_permutations: int = 1000,
    weight: float = 1.0,
    seed: int = 0,
) -> GSEAResult:
    """
    Internal fallback GSEA (prerank label permutation).
    """
    ranked_list = _to_ranked_list(ranked)
    if gene_set is None or len(gene_set) == 0:
        raise DataError("gene_set is empty")

    n_permutations = int(n_permutations)
    if n_permutations <= 0:
        raise ValidationError("n_permutations must be > 0")

    es, leading, profile = calculate_enrichment_score(ranked_list, gene_set, weight=weight)

    genes = [g for g, _ in ranked_list]
    scores = [s for _, s in ranked_list]

    rng = random.Random(int(seed))
    null_es: List[float] = []

    for _ in range(n_permutations):
        perm_genes = genes[:]
        rng.shuffle(perm_genes)
        permuted = list(zip(perm_genes, scores))
        es0, _, _ = calculate_enrichment_score(permuted, gene_set, weight=weight)
        null_es.append(float(es0))

    null_arr = np.array(null_es, dtype=float)

    # empirical p-value with +1 correction
    if es >= 0:
        denom_pool = null_arr[null_arr > 0]
        denom = float(np.mean(denom_pool)) if denom_pool.size else 0.0
        nes = float(es / denom) if denom != 0 else 0.0
        p_value = float((np.sum(null_arr >= es) + 1.0) / (n_permutations + 1.0))
    else:
        denom_pool = np.abs(null_arr[null_arr < 0])
        denom = float(np.mean(denom_pool)) if denom_pool.size else 0.0
        nes = float(es / denom) if denom != 0 else 0.0
        p_value = float((np.sum(null_arr <= es) + 1.0) / (n_permutations + 1.0))

    p_value = min(max(p_value, 1.0 / (n_permutations + 1.0)), 1.0)

    return GSEAResult(
        es=float(es),
        nes=float(nes),
        p_value=float(p_value),
        leading_edge=tuple(sorted(leading)),
        es_profile=tuple(float(x) for x in profile),
        n_permutations=n_permutations,
        gene_set_size=len(gene_set),
        ranked_size=len(ranked_list),
    )


def gsea_many(
    ranked: RankedInput,
    gene_sets: Dict[str, Set[str]],
    n_permutations: int = 500,
    weight: float = 1.0,
    seed: int = 0,
    min_size: int = 10,
) -> Dict[str, GSEAResult]:
    """
    Internal fallback many-GSEA.
    """
    if not gene_sets:
        raise DataError("gene_sets is empty")

    out: Dict[str, GSEAResult] = {}
    base_seed = int(seed)
    for i, (pid, gs) in enumerate(gene_sets.items()):
        if gs is None or len(gs) < int(min_size):
            continue
        out[pid] = gsea(
            ranked=ranked,
            gene_set=set(gs),
            n_permutations=int(n_permutations),
            weight=float(weight),
            seed=base_seed + i * 9973,
        )
    if not out:
        raise DataError("No gene sets passed min_size filter.")
    return out


def gsea_many_auto(
    ranked: RankedInput,
    gene_sets: Dict[str, Set[str]],
    *,
    prefer_gseapy: bool = True,
    gseapy_gene_sets: Optional[str] = None,
    n_permutations: int = 500,
    weight: float = 1.0,
    seed: int = 0,
    min_size: int = 10,
    max_size: int = 2000,
) -> Dict[str, GSEAResult]:
    """
    Preferred behavior:
      1) If prefer_gseapy and gseapy_gene_sets provided -> run gseapy.prerank first.
      2) Otherwise fallback to internal implementation over gene_sets dict.
    """
    if prefer_gseapy and gseapy_gene_sets:
        res = _try_gseapy_prerank(
            ranked=ranked,
            gene_sets=gseapy_gene_sets,
            n_permutations=int(n_permutations),
            seed=int(seed),
            min_size=int(min_size),
            max_size=int(max_size),
        )
        if res is not None:
            return res

    # fallback internal
    return gsea_many(
        ranked=ranked,
        gene_sets=gene_sets,
        n_permutations=int(n_permutations),
        weight=float(weight),
        seed=int(seed),
        min_size=int(min_size),
    )
