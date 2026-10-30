"""Ad-hoc manual check of one neuron + dose: baseline vs steered, with the measured shift.

This is the human-in-the-loop companion to the leaderboard. It generates a fresh answer for each query
with and without the intervention, scores both with the same target spec/embedder as the run, and
optionally compares the shift against matched random-direction controls.
"""
from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

from nudge.experiments.sampling import make_controls
from nudge.models.base import GenerationConfig, SteeringBackend
from nudge.scoring.behavioral import coherence_features
from nudge.scoring.controls import control_comparison
from nudge.scoring.embeddings import get_embedder
from nudge.scoring.semantic import SemanticScorer
from nudge.steering.interventions import Intervention
from nudge.targets.schema import TargetSpec

_NEURON_ID = re.compile(r"^\s*[Ll]?\s*(\d+)\s*[.,:_/\s-]+\s*[Nn]?\s*(\d+)\s*$")


def parse_neuron_id(text: str) -> tuple[int, int]:
    """Parse ``L12.N4254`` / ``12.4254`` / ``12 4254`` into ``(layer, neuron)``."""
    m = _NEURON_ID.match(text)
    if not m:
        raise ValueError(f"invalid neuron id {text!r}; expected something like L12.N4254")
    return int(m.group(1)), int(m.group(2))


def _evaluate(embedder, scorer: SemanticScorer, prompt: str, text: str) -> dict[str, Any]:
    sem = scorer.score([text])[0]
    coh = coherence_features(text)
    relevance = float((embedder.encode([text]) * embedder.encode([prompt])).sum(axis=1)[0])
    return {"text": text, "j": sem.j, "s_target": sem.s_target, "s_contrast": sem.s_contrast,
            "coherence": coh["coherence"], "flags": coh["flags"], "relevance": relevance}


def _generated(result) -> dict[str, Any]:
    return {"status": result.status, "error": result.error, "n_output_tokens": len(result.token_ids),
            "latency_s": result.latency_s}


def run_check(*, backend: SteeringBackend, spec: TargetSpec, queries: Sequence[str],
              layer: int, neuron: int, doses: Sequence[float], dose_unit: str = "rel_mlp_out_norm",
              generation: GenerationConfig | None = None, embedder_name: str = "hashing",
              n_controls: int = 0, seed: int = 0) -> list[dict[str, Any]]:
    """Generate and score baseline/steered pairs for each query and dose."""
    gen = generation or GenerationConfig()
    embedder = get_embedder(embedder_name)
    scorer = SemanticScorer(embedder, spec)
    n_neurons = len(backend.list_neurons(layer))
    out: list[dict[str, Any]] = []
    for query in queries:
        base_res = backend.run_baseline(query, gen)
        base = {**_generated(base_res), **_evaluate(embedder, scorer, query, base_res.text)}
        rows = []
        for dose in doses:
            iv = Intervention(layer, neuron, float(dose), dose_unit)
            res = backend.run_steered(query, iv, gen)
            row = {**_generated(res), **_evaluate(embedder, scorer, query, res.text),
                   "layer": layer, "neuron": neuron, "dose": float(dose), "dose_unit": dose_unit}
            if base_res.status == "ok" and res.status == "ok":
                row["dj"] = row["j"] - base["j"]
                row["d_coherence"] = row["coherence"] - base["coherence"]
                row["d_relevance"] = row["relevance"] - base["relevance"]
            if n_controls > 0 and base_res.status == "ok":
                control_djs = []
                for ctrl in make_controls(layer, [float(dose)], dose_unit, n_controls, n_neurons, seed):
                    cres = backend.run_steered(query, ctrl, gen)
                    if cres.status != "ok":
                        continue
                    c = _evaluate(embedder, scorer, query, cres.text)
                    control_djs.append(c["j"] - base["j"])
                row["controls"] = control_comparison(row.get("dj", float("nan")), control_djs)
            rows.append(row)
        out.append({"query": query, "baseline": {**_generated(base_res), **{
            k: base[k] for k in ("text", "j", "s_target", "s_contrast", "coherence", "flags", "relevance")}},
            "doses": rows})
    return out
