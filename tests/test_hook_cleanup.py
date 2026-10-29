import pytest

from nudge.models.base import GenerationConfig
from nudge.steering.hooks import additive_mlp_out, count_forward_hooks
from nudge.steering.interventions import Intervention

CFG = GenerationConfig(max_new_tokens=10)
P = "The old lighthouse keeper looked out over the water and"


def test_baseline_after_steered_run_is_unchanged(backend):
    first = backend.run_baseline(P, CFG)
    steered = backend.run_steered(P, Intervention(3, 7, 3.0), CFG)
    again = backend.run_baseline(P, CFG)
    assert steered.status == "ok" and steered.token_ids != first.token_ids
    assert first.token_ids == again.token_ids
    assert count_forward_hooks(backend.model) == 0


def test_hooks_removed_even_if_generation_raises(backend):
    with pytest.raises(RuntimeError):
        with additive_mlp_out(backend._down, {2: backend.delta_vector(Intervention(2, 1, 1.0))}):
            assert count_forward_hooks(backend.model) == 1
            raise RuntimeError("boom")
    assert count_forward_hooks(backend.model) == 0


def test_backend_error_is_recorded_not_raised(backend):
    res = backend.run_steered(P, Intervention(99, 0, 1.0), CFG)
    assert res.status == "error" and "layer 99" in res.error
    assert count_forward_hooks(backend.model) == 0


def test_multi_neuron_same_layer_adds(backend):
    import torch
    a, b = Intervention(2, 1, 0.5), Intervention(2, 2, 0.5)
    d = backend._deltas([a, b])
    assert torch.allclose(d[2], backend.delta_vector(a) + backend.delta_vector(b))
