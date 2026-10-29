"""Semantic target/contrast scoring.

    S_T(y) = mean_t cos(E(y), E(t))      S_C(y) = mean_c cos(E(y), E(c))      J(y) = S_T(y) - S_C(y)

Complete outputs are embedded (never tokens independently). Raw similarities are stored, not only J.
J measures movement toward the *described* behaviour; it does not show that a neuron represents it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from nudge.scoring.embeddings import Embedder
from nudge.targets.schema import TargetSpec

FORMULA_VERSION = "J=mean_cos(target)-mean_cos(contrast);v1"


@dataclass
class SemanticScore:
    s_target: float
    s_contrast: float
    j: float
    target_sims: list[float]
    contrast_sims: list[float]


class SemanticScorer:
    def __init__(self, embedder: Embedder, spec: TargetSpec) -> None:
        self.embedder = embedder
        self.spec = spec
        self._t = embedder.encode(spec.target_descriptions)
        self._c = embedder.encode(spec.contrast_descriptions)

    def score(self, texts: Sequence[str]) -> list[SemanticScore]:
        if not texts:
            return []
        e = self.embedder.encode(list(texts))
        ts, cs = e @ self._t.T, e @ self._c.T  # [n, |T|], [n, |C|]
        out = []
        for t, c in zip(ts, cs):
            st, sc = float(np.mean(t)), float(np.mean(c))
            out.append(SemanticScore(st, sc, st - sc, [float(x) for x in t], [float(x) for x in c]))
        return out
