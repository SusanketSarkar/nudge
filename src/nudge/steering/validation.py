"""Phase-0 self-checks: prove the intervention is wired correctly *before* any experiment.

Run via ``nudge doctor``. Each check returns a :class:`CheckResult`; a failure means results from this
backend must not be trusted.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from nudge.models.base import GenerationConfig
from nudge.models.hf_backend import HFBackend
from nudge.steering.hooks import count_forward_hooks, additive_mlp_out, neuron_activation
from nudge.steering.interventions import Intervention

PROBE = "The old lighthouse keeper looked out over the water and"


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str


def check_delta_shape_and_orientation(b: HFBackend, layer: int, neuron: int) -> CheckResult:
    """delta must be [hidden] and, for raw_activation, equal alpha * W_down[:, n] (column, not row)."""
    iv = Intervention(layer, neuron, 2.0, "raw_activation")
    d = b.delta_vector(iv)
    w = b._down[layer].weight.detach().float()
    ok_shape = tuple(d.shape) == (b.hidden_size,)
    ok_val = torch.allclose(d, 2.0 * w[:, neuron], atol=1e-6)
    return CheckResult("delta_shape_and_orientation", ok_shape and ok_val,
                       f"delta shape {tuple(d.shape)}, hidden={b.hidden_size}; equals 2*W_down[:, n]: {ok_val}")


def check_equivalence(b: HFBackend, layer: int, neuron: int, alpha: float = 3.0) -> CheckResult:
    """Output-side additive hook == adding alpha to the neuron activation at down_proj's input."""
    iv = Intervention(layer, neuron, alpha, "raw_activation")
    via_out = b.logits(PROBE, iv)
    with neuron_activation(b._down, {layer: [(neuron, alpha)]}):
        with torch.inference_mode():
            via_act = b.model(**b._encode(PROBE)).logits[0, -1].float().cpu()
    base = b.logits(PROBE)
    max_diff = float((via_out - via_act).abs().max())
    effect = float((via_out - base).abs().max())
    tol = 1e-3 * max(1.0, float(via_out.abs().max()))
    return CheckResult("hook_point_equivalence", max_diff < tol and effect > 0,
                       f"max|mlp_out - activation| = {max_diff:.2e} (tol {tol:.1e}); effect on logits {effect:.3e}")


def check_zero_dose_identity(b: HFBackend, layer: int, neuron: int) -> CheckResult:
    base = b.logits(PROBE)
    z = b.logits(PROBE, Intervention(layer, neuron, 0.0))
    d = float((base - z).abs().max())
    return CheckResult("zero_dose_is_identity", d < 1e-6, f"max logit diff {d:.2e}")


def check_hook_cleanup(b: HFBackend, layer: int, neuron: int, cfg: GenerationConfig | None = None) -> CheckResult:
    cfg = cfg or GenerationConfig(max_new_tokens=12)
    n0 = count_forward_hooks(b.model)
    first = b.run_baseline(PROBE, cfg)
    steered = b.run_steered(PROBE, Intervention(layer, neuron, 2.0), cfg)
    n1 = count_forward_hooks(b.model)
    again = b.run_baseline(PROBE, cfg)
    # exception safety: a failing generation must not leak hooks
    try:
        with additive_mlp_out(b._down, {layer: b.delta_vector(Intervention(layer, neuron, 1.0))}):
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    n2 = count_forward_hooks(b.model)
    ok = (n0 == n1 == n2 and first.text == again.text and first.status == again.status == "ok"
          and steered.status == "ok")
    return CheckResult("hook_cleanup", ok,
                       f"hooks before/after run/after exception: {n0}/{n1}/{n2}; "
                       f"baseline unchanged after steered run: {first.text == again.text}")


def check_steering_has_effect(b: HFBackend, layer: int, neuron: int) -> CheckResult:
    base = b.logits(PROBE)
    big = b.logits(PROBE, Intervention(layer, neuron, 1.0, "rel_mlp_out_norm"))
    d = float((base - big).abs().max())
    return CheckResult("steering_changes_logits", d > 1e-5, f"max logit change at dose 1.0: {d:.3e}")


def check_random_control_matched(b: HFBackend, layer: int, neuron: int) -> CheckResult:
    iv = Intervention(layer, neuron, 0.7, "rel_mlp_out_norm")
    a, c = b.delta_vector(iv), b.delta_vector(iv.as_control(1))
    c2 = b.delta_vector(iv.as_control(1))
    cos = float(torch.nn.functional.cosine_similarity(a, c, dim=0))
    ok = abs(float(a.norm()) - float(c.norm())) < 1e-4 * float(a.norm()) and torch.equal(c, c2)
    return CheckResult("random_control_matched_and_seeded", ok,
                       f"|delta|={float(a.norm()):.4f} vs control {float(c.norm()):.4f}; cos={cos:+.3f}; deterministic={torch.equal(c, c2)}")


def check_baseline_repeatability(b: HFBackend, cfg: GenerationConfig | None = None) -> CheckResult:
    cfg = cfg or GenerationConfig(max_new_tokens=16)
    a, c = b.run_baseline(PROBE, cfg), b.run_baseline(PROBE, cfg)
    return CheckResult("baseline_repeatability", a.token_ids == c.token_ids and a.status == "ok",
                       f"identical token ids across repeats: {a.token_ids == c.token_ids}")


def run_self_checks(b: HFBackend, layer: int | None = None, neuron: int = 0) -> list[CheckResult]:
    layers = b.list_supported_layers()
    layer = layers[len(layers) // 2] if layer is None else layer
    return [
        check_delta_shape_and_orientation(b, layer, neuron),
        check_zero_dose_identity(b, layer, neuron),
        check_equivalence(b, layer, neuron),
        check_steering_has_effect(b, layer, neuron),
        check_hook_cleanup(b, layer, neuron),
        check_random_control_matched(b, layer, neuron),
        check_baseline_repeatability(b),
    ]
