# mdp_engine/stats_utils.py
from __future__ import annotations

import math
from typing import List, Sequence, Tuple

from .exceptions import ValidationError


def _logsumexp(log_terms: Sequence[float]) -> float:
    if not log_terms:
        return float("-inf")
    m = max(log_terms)
    if math.isinf(m):
        return m
    s = sum(math.exp(t - m) for t in log_terms)
    return m + math.log(s)


def _log_comb(n: int, k: int) -> float:
    if k < 0 or k > n:
        return float("-inf")
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def hypergeom_sf(N: int, M: int, n: int, k: int) -> float:
    """
    Survival function P(X >= k) for Hypergeometric(N, M, n).
    N: population size
    M: successes in population
    n: draws
    k: observed successes
    """
    for name, v in [("N", N), ("M", M), ("n", n), ("k", k)]:
        if not isinstance(v, int):
            raise ValidationError(f"{name} must be int, got {type(v)}")
        if v < 0:
            raise ValidationError(f"{name} must be >= 0, got {v}")

    if N == 0:
        return 1.0 if k <= 0 else 0.0
    if M > N:
        raise ValidationError(f"M cannot exceed N (M={M}, N={N})")
    if n > N:
        raise ValidationError(f"n cannot exceed N (n={n}, N={N})")

    k_min = max(0, n - (N - M))
    k_max = min(M, n)
    if k <= k_min:
        return 1.0
    if k > k_max:
        return 0.0

    denom = _log_comb(N, n)
    log_terms = []
    for i in range(k, k_max + 1):
        lt = _log_comb(M, i) + _log_comb(N - M, n - i) - denom
        log_terms.append(lt)

    val = math.exp(_logsumexp(log_terms))
    return min(max(val, 0.0), 1.0)


def fisher_exact_2x2(
    a: int, b: int, c: int, d: int, alternative: str = "two-sided"
) -> Tuple[float, float]:
    """
    Fisher exact test for 2x2:
        [[a, b],
         [c, d]]
    Returns: (odds_ratio, p_value)
    """
    if alternative not in {"two-sided", "greater", "less"}:
        raise ValidationError("alternative must be one of: two-sided, greater, less")

    for name, v in [("a", a), ("b", b), ("c", c), ("d", d)]:
        if not isinstance(v, int) or v < 0:
            raise ValidationError(f"{name} must be non-negative int, got {v}")

    row1 = a + b
    row2 = c + d
    col1 = a + c
    col2 = b + d
    n = row1 + row2
    if n == 0:
        return float("nan"), 1.0

    if b * c == 0:
        odds = float("inf") if a * d > 0 else 0.0
    else:
        odds = (a * d) / (b * c)

    N = n
    M = col1
    draws = row1

    a_min = max(0, draws - col2)
    a_max = min(draws, M)

    def log_p(a_val: int) -> float:
        return _log_comb(M, a_val) + _log_comb(N - M, draws - a_val) - _log_comb(N, draws)

    obs_lp = log_p(a)

    if alternative == "greater":
        log_terms = [log_p(x) for x in range(a, a_max + 1)]
        p = math.exp(_logsumexp(log_terms))
        return odds, min(max(p, 0.0), 1.0)

    if alternative == "less":
        log_terms = [log_p(x) for x in range(a_min, a + 1)]
        p = math.exp(_logsumexp(log_terms))
        return odds, min(max(p, 0.0), 1.0)

    log_ps = [log_p(x) for x in range(a_min, a_max + 1)]
    max_lp = max(log_ps)
    probs = [math.exp(lp - max_lp) for lp in log_ps]
    obs_p_scaled = math.exp(obs_lp - max_lp)

    p_sum_scaled = sum(pv for pv in probs if pv <= obs_p_scaled + 1e-15)
    p = p_sum_scaled / sum(probs)
    return odds, min(max(p, 0.0), 1.0)


def bh_fdr(p_values: Sequence[float]) -> List[float]:
    """
    Benjamini-Hochberg FDR correction. Returns q-values in original order.
    """
    if p_values is None:
        raise ValidationError("p_values is None")

    n = len(p_values)
    if n == 0:
        return []

    indexed = []
    for i, p in enumerate(p_values):
        if p is None or not isinstance(p, (int, float)) or math.isnan(float(p)):
            raise ValidationError(f"Invalid p-value at index {i}: {p}")
        pf = float(p)
        if pf < 0.0 or pf > 1.0:
            raise ValidationError(f"p-value out of range [0,1] at index {i}: {p}")
        indexed.append((i, pf))

    indexed.sort(key=lambda x: x[1])  # ascending by p
    q = [0.0] * n

    # raw BH
    for rank, (idx, p) in enumerate(indexed, start=1):
        q[idx] = (p * n) / rank

    # enforce monotone non-decreasing in sorted order by backward cumulative min
    prev = 1.0
    for idx, _p in reversed(indexed):
        cur = min(prev, q[idx])
        q[idx] = cur
        prev = cur

    return [min(max(v, 0.0), 1.0) for v in q]
