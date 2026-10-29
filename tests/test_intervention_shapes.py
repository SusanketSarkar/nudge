import pytest
import torch

from nudge.steering.interventions import Intervention, interventions_key


def test_delta_shape_and_column_orientation(backend):
    iv = Intervention(2, 5, 2.0, "raw_activation")
    d = backend.delta_vector(iv)
    w = backend._down[2].weight
    assert d.shape == (backend.hidden_size,)
    assert w.shape == (backend.hidden_size, backend.intermediate_size)  # [hidden, intermediate]
    assert torch.allclose(d, 2.0 * w[:, 5].detach().float())
    assert not torch.allclose(d, 2.0 * w[5, :backend.hidden_size].detach().float())  # not the row


def test_relative_unit_has_layer_norm_magnitude(backend):
    d = backend.delta_vector(Intervention(3, 9, 0.5, "rel_mlp_out_norm"))
    assert float(d.norm()) == pytest.approx(0.5 * backend.mlp_out_norm(3), rel=1e-4)
    d2 = backend.delta_vector(Intervention(3, 10, 0.5, "rel_mlp_out_norm"))
    assert float(d.norm()) == pytest.approx(float(d2.norm()), rel=1e-4)  # same push for every neuron


def test_sign_flips_delta(backend):
    a = backend.delta_vector(Intervention(2, 1, 0.7))
    b = backend.delta_vector(Intervention(2, 1, -0.7))
    assert torch.allclose(a, -b)


def test_output_and_activation_hooks_are_equivalent(backend):
    from nudge.steering.validation import check_equivalence
    assert check_equivalence(backend, 3, 11).passed


def test_random_control_matched_magnitude_and_seeded(backend):
    iv = Intervention(2, 4, 0.8)
    c1, c1b, c2 = (backend.delta_vector(iv.as_control(s)) for s in (1, 1, 2))
    assert torch.equal(c1, c1b) and not torch.equal(c1, c2)
    assert float(c1.norm()) == pytest.approx(float(backend.delta_vector(iv).norm()), rel=1e-4)


def test_invalid_interventions():
    with pytest.raises(ValueError):
        Intervention(1, 1, 0.5, kind="random")           # needs seed
    with pytest.raises(ValueError):
        Intervention(1, 1, 0.5, random_seed=3)           # neuron kind must not have a seed
    with pytest.raises(ValueError):
        Intervention(-1, 1, 0.5)


def test_backend_rejects_bad_requests(backend):
    with pytest.raises(ValueError):
        backend.delta_vector(Intervention(99, 0, 1.0))
    with pytest.raises(ValueError):
        backend.delta_vector(Intervention(1, 10_000, 1.0))
    with pytest.raises(ValueError):
        backend.delta_vector(Intervention(1, 1, 1.0, "bogus_unit"))


def test_key_is_order_independent():
    a, b = Intervention(1, 2, 0.5), Intervention(3, 4, -0.5)
    assert interventions_key([a, b]) == interventions_key([b, a]) != interventions_key([a])
