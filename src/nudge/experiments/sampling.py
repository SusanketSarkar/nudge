"""Reproducible candidate and control sampling."""
from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from nudge.steering.interventions import Intervention
from nudge.utils.hashing import stable_seed


def layers_by_depth(n_layers: int, k: int, skip_first: int = 0) -> list[int]:
    """``k`` layers spread evenly from ``skip_first`` to the last layer (cross-depth survey).

    ``k <= 0`` surveys every layer, i.e. ``survey_layers = -1`` means all layers.
    """
    lo, hi = skip_first, n_layers - 1
    if k <= 0 or k >= hi - lo + 1:
        return list(range(lo, hi + 1))
    return sorted({int(round(x)) for x in np.linspace(lo, hi, k)})


def sample_neurons(layer: int, n_neurons_in_layer: int, k: int, seed: int) -> list[int]:
    """Uniform sample without replacement; stable per (seed, layer) and independent of other layers.

    ``k < 0`` inspects every neuron in the layer, i.e. ``neurons_per_layer = -1`` means all neurons.
    """
    if k < 0:
        return list(range(n_neurons_in_layer))
    rng = np.random.default_rng(stable_seed("neurons", seed, layer))
    k = min(k, n_neurons_in_layer)
    return sorted(int(x) for x in rng.choice(n_neurons_in_layer, size=k, replace=False))


def sample_from_statistics(layer: int, stat: Sequence[float], k: int, seed: int, bins: int = 5) -> list[int]:
    """Optional mode: stratify by quantile bins of a per-neuron statistic (e.g. activation frequency).

    Stratified rather than top-k on purpose: causally useful neurons are not necessarily the loudest.
    """
    stat = np.asarray(stat, dtype=float)
    if k < 0:
        return list(range(len(stat)))
    order = np.argsort(stat, kind="stable")
    strata = np.array_split(order, min(bins, len(order)))
    rng = np.random.default_rng(stable_seed("stats", seed, layer))
    per = max(1, k // len(strata))
    picks: list[int] = []
    for s in strata:
        picks.extend(int(x) for x in rng.choice(s, size=min(per, len(s)), replace=False))
    return sorted(set(picks))[:k] if len(picks) > k else sorted(set(picks))


def candidate_neurons(layers: Sequence[int], neurons_per_layer: int, intermediate_sizes: Mapping[int, int],
                      seed: int) -> list[tuple[int, int]]:
    return [(l, n) for l in layers for n in sample_neurons(l, intermediate_sizes[l], neurons_per_layer, seed)]


def make_candidate_interventions(pairs: Sequence[tuple[int, int]], doses: Sequence[float],
                                 dose_unit: str) -> list[Intervention]:
    return [Intervention(l, n, float(d), dose_unit) for (l, n) in pairs for d in doses]


def make_controls(layer: int, doses: Sequence[float], dose_unit: str, per_layer_dose: int,
                  n_neurons_in_layer: int, seed: int) -> list[Intervention]:
    """Per (layer, dose): ``per_layer_dose`` random-direction controls with distinct seeds.

    Each control is anchored to a sampled neuron index (only used for magnitude under the
    ``raw_activation`` unit; irrelevant under ``rel_mlp_out_norm``).
    """
    rng = np.random.default_rng(stable_seed("controls", seed, layer))
    anchors = [int(x) for x in rng.integers(0, n_neurons_in_layer, size=per_layer_dose)]
    return [Intervention(layer, anchors[k], float(d), dose_unit, "random", k)
            for d in doses for k in range(per_layer_dose)]
