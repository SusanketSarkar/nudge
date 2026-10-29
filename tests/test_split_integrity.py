import pytest

from nudge.experiments.prompts import Prompt, PromptSuite, SplitIntegrityError, load_suite, split_flat_list
from nudge.experiments.sampling import (candidate_neurons, layers_by_depth, make_controls,
                                        sample_from_statistics, sample_neurons)


@pytest.mark.parametrize("name", ["fiction_scenes", "uncertainty_qa"])
def test_builtin_suites_are_disjoint_and_sized(name):
    s = load_suite(name)
    assert s.counts() == {"screening": 12, "validation": 12, "test": 20}
    parts = {p: {" ".join(x.text.lower().split()) for x in s.partition(p)} for p in ("screening", "validation", "test")}
    assert not (parts["screening"] & parts["validation"] or parts["screening"] & parts["test"] or parts["validation"] & parts["test"])


def test_leak_detected():
    with pytest.raises(SplitIntegrityError):
        PromptSuite("x", [Prompt("a", "Hello  World", "screening"), Prompt("b", "hello world", "test")])
    with pytest.raises(SplitIntegrityError):
        PromptSuite("x", [Prompt("a", "one", "screening"), Prompt("a", "two", "test")])


def test_split_flat_list_is_deterministic_and_order_independent():
    texts = [f"prompt number {i}" for i in range(20)]
    a, b = split_flat_list(texts, seed=3), split_flat_list(list(reversed(texts)), seed=3)
    assert {p.id: p.partition for p in a.prompts} == {p.id: p.partition for p in b.prompts}
    assert sum(a.counts().values()) == 20


def test_uncertainty_suite_has_balanced_answerability():
    s = load_suite("uncertainty_qa")
    for part in ("screening", "validation", "test"):
        ans = [p.meta["answerable"] for p in s.partition(part)]
        assert sum(ans) == len(ans) - sum(ans)


def test_sampler_reproducible_and_layer_independent():
    assert sample_neurons(3, 192, 10, seed=7) == sample_neurons(3, 192, 10, seed=7)
    assert sample_neurons(3, 192, 10, seed=7) != sample_neurons(3, 192, 10, seed=8)
    a = candidate_neurons([2, 3], 5, {2: 100, 3: 100}, 1)
    b = candidate_neurons([3], 5, {3: 100}, 1)
    assert [x for x in a if x[0] == 3] == b  # adding a layer does not reshuffle the others
    assert len(set(sample_neurons(1, 50, 50, 0))) == 50 and len(sample_neurons(1, 5, 50, 0)) == 5


def test_stats_assisted_sampling_stratifies():
    stat = list(range(100))
    picks = sample_from_statistics(0, stat, 10, seed=0, bins=5)
    assert len(picks) == 10 and any(p < 20 for p in picks) and any(p >= 80 for p in picks)


def test_layers_by_depth_and_controls():
    assert layers_by_depth(12, 3, skip_first=2) == [2, 6, 11]
    c = make_controls(3, [-0.5, 0.5], "rel_mlp_out_norm", 4, 100, seed=0)
    assert len(c) == 8 and all(x.kind == "random" for x in c)
    assert len({(x.dose, x.random_seed) for x in c}) == 8
