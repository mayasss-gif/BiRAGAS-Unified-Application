# mdp_engine/topology.py
from __future__ import annotations

from typing import Dict

from .exceptions import ValidationError
from .graph import SimpleDiGraph


def pagerank(
    g: SimpleDiGraph,
    damping: float = 0.85,
    max_iter: int = 200,
    tol: float = 1e-8,
) -> Dict[str, float]:
    """
    PageRank over a directed graph with power iteration.
    Returns dict[node] -> score.
    """
    if g is None:
        raise ValidationError("graph is None")
    if not (0.0 < float(damping) < 1.0):
        raise ValidationError("damping must be between 0 and 1")
    if max_iter <= 0:
        raise ValidationError("max_iter must be > 0")
    if tol <= 0:
        raise ValidationError("tol must be > 0")

    nodes = sorted(g.nodes())
    n = len(nodes)
    if n == 0:
        return {}

    idx = {node: i for i, node in enumerate(nodes)}
    outdeg = [max(g.out_degree(node), 0) for node in nodes]

    # init uniform
    pr = [1.0 / n] * n
    base = (1.0 - damping) / n

    # build incoming lists
    incoming = [[] for _ in range(n)]
    for node in nodes:
        j = idx[node]
        for e in g.predecessors(node):
            src = e.src
            if src in idx:
                incoming[j].append(idx[src])

    for _ in range(max_iter):
        new = [base] * n
        for j in range(n):
            s = 0.0
            for i in incoming[j]:
                if outdeg[i] > 0:
                    s += pr[i] / outdeg[i]
            new[j] += damping * s

        # normalize + convergence
        z = sum(new)
        if z > 0:
            new = [v / z for v in new]

        delta = sum(abs(new[i] - pr[i]) for i in range(n))
        pr = new
        if delta < tol:
            break

    return {nodes[i]: float(pr[i]) for i in range(n)}
