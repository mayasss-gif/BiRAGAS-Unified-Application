#!/usr/bin/env python3
"""
analysis_tools/query_modules/common_scoring.py

Scoring + similarity helpers.

Adds robust Jaccard similarity over sets / binary matrices.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd


def jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return float(inter) / float(union) if union else 0.0


def jaccard_similarity_matrix(sets_by_label: Dict[str, Set[str]]) -> pd.DataFrame:
    labels = list(sets_by_label.keys())
    n = len(labels)
    mat = np.zeros((n, n), dtype=float)
    for i, li in enumerate(labels):
        for j, lj in enumerate(labels):
            mat[i, j] = jaccard(sets_by_label[li], sets_by_label[lj])
    return pd.DataFrame(mat, index=labels, columns=labels)


def cosine_similarity_matrix(X: pd.DataFrame) -> pd.DataFrame:
    """
    Cosine similarity over rows of X (row vectors).
    """
    A = X.to_numpy(dtype=float)
    norms = np.linalg.norm(A, axis=1)
    norms[norms == 0] = 1.0
    A = A / norms[:, None]
    sim = A @ A.T
    return pd.DataFrame(sim, index=X.index, columns=X.index)
