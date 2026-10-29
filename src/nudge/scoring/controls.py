"""Statistics for random-direction controls and multiple comparisons."""
from __future__ import annotations

from typing import Sequence

import numpy as np


def bootstrap_ci(values: Sequence[float], level: float = 0.95, n_boot: int = 4000, seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap CI of the mean, resampling *prompts*. (nan, nan) if fewer than 2 values."""
    v = np.asarray([x for x in values if np.isfinite(x)], dtype=float)
    if len(v) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, len(v), size=(n_boot, len(v)))].mean(axis=1)
    a = (1 - level) / 2
    return float(np.quantile(means, a)), float(np.quantile(means, 1 - a))


def control_comparison(candidate_mean: float, control_means: Sequence[float]) -> dict[str, float]:
    """Where does the candidate's mean effect fall in the matched random-control distribution?

    percentile: fraction of controls strictly below the candidate.
    p_value: one-sided empirical p, (1 + #controls >= candidate) / (K + 1)  (never exactly 0).
    """
    c = np.asarray([x for x in control_means if np.isfinite(x)], dtype=float)
    if len(c) == 0 or not np.isfinite(candidate_mean):
        return {"percentile": float("nan"), "p_value": float("nan"), "n_controls": float(len(c))}
    return {"percentile": float(np.mean(c < candidate_mean)),
            "p_value": float((1 + np.sum(c >= candidate_mean)) / (len(c) + 1)),
            "n_controls": float(len(c))}


def benjamini_hochberg(pvals: Sequence[float]) -> np.ndarray:
    """BH false-discovery-rate adjusted q-values; NaNs are passed through and ignored."""
    p = np.asarray(pvals, dtype=float)
    q = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    if not ok.any():
        return q
    pv = p[ok]
    order = np.argsort(pv)
    ranked = pv[order] * len(pv) / (np.arange(len(pv)) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty_like(pv)
    out[order] = np.minimum(ranked, 1.0)
    q[ok] = out
    return q
