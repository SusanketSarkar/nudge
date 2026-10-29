"""Avoid shortlists of 20 near-duplicates: greedy selection over behavioural profiles."""
from __future__ import annotations

from typing import Sequence

import numpy as np


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else 0.0


def greedy_diverse(scores: Sequence[float], profiles: np.ndarray, k: int, max_cosine: float = 0.9) -> list[int]:
    """Indices of up to ``k`` high-scoring candidates whose profiles are pairwise < ``max_cosine`` similar.

    Candidates are visited in descending score; one is skipped if it is too similar to an already
    chosen one. If that leaves fewer than ``k``, the shortlist is simply shorter (no padding with
    near-duplicates). Clusters are an organisational aid, not evidence of distinct mechanisms.
    """
    order = np.argsort(-np.asarray(scores, dtype=float), kind="stable")
    chosen: list[int] = []
    for i in order:
        if all(_cos(profiles[i], profiles[j]) < max_cosine for j in chosen):
            chosen.append(int(i))
        if len(chosen) >= k:
            break
    return chosen


def cluster_labels(profiles: np.ndarray, threshold: float = 0.9) -> list[int]:
    """Greedy leader clustering by cosine similarity (stable, dependency-free)."""
    leaders: list[int] = []
    labels: list[int] = []
    for i in range(len(profiles)):
        for li, j in enumerate(leaders):
            if _cos(profiles[i], profiles[j]) >= threshold:
                labels.append(li)
                break
        else:
            leaders.append(i)
            labels.append(len(leaders) - 1)
    return labels
