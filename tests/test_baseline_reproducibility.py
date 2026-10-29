from nudge.models.base import GenerationConfig
from nudge.steering.interventions import Intervention

P = "Describe a market in the early morning."


def test_greedy_baseline_is_repeatable(backend):
    cfg = GenerationConfig(max_new_tokens=16)
    assert backend.run_baseline(P, cfg).token_ids == backend.run_baseline(P, cfg).token_ids


def test_steered_is_repeatable_and_seeded_sampling_is_repeatable(backend):
    iv = Intervention(3, 3, 1.0)
    g = GenerationConfig(max_new_tokens=12)
    assert backend.run_steered(P, iv, g).token_ids == backend.run_steered(P, iv, g).token_ids
    s = GenerationConfig(max_new_tokens=12, temperature=0.8, seed=5)
    assert backend.run_baseline(P, s).token_ids == backend.run_baseline(P, s).token_ids


def test_zero_dose_matches_baseline(backend):
    import torch
    assert torch.equal(backend.logits(P), backend.logits(P, Intervention(2, 0, 0.0)))


def test_self_checks_all_pass(backend):
    from nudge.steering.validation import run_self_checks
    bad = [r for r in run_self_checks(backend) if not r.passed]
    assert not bad, bad
