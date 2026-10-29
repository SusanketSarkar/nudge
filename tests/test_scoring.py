import numpy as np
import pytest

from nudge.scoring.behavioral import EvalContext, UncertaintyLexiconEvaluator, coherence_features
from nudge.scoring.controls import benjamini_hochberg, bootstrap_ci, control_comparison
from nudge.scoring.diversity import cluster_labels, greedy_diverse
from nudge.scoring.embeddings import HashingEmbedder
from nudge.scoring.judges import LLMJudge
from nudge.scoring.semantic import SemanticScorer
from nudge.targets.schema import TargetSpec
from nudge.targets.target_builder import get_preset, propose_spec


class FixedEmbedder:
    """Two orthogonal directions so the expected J is known exactly."""
    name = "fixed"

    def encode(self, texts):
        out = []
        for t in texts:
            v = np.zeros(3)
            v[0 if "target" in t else 1 if "contrast" in t else 2] = 1
            out.append(v)
        return np.array(out)


def test_semantic_score_formula_exact():
    spec = TargetSpec(target_name="x", target_descriptions=["target a", "target b"],
                      contrast_descriptions=["contrast a", "other"])
    s = SemanticScorer(FixedEmbedder(), spec).score(["target text", "contrast text", "neutral"])
    assert (s[0].s_target, s[0].s_contrast, s[0].j) == (1.0, 0.0, 1.0)
    # contrast descs: "contrast a" -> dir 1, "other" -> dir 2 ; output "contrast text" -> dir 1
    assert s[1].s_contrast == pytest.approx(0.5) and s[1].s_target == 0.0 and s[1].j == pytest.approx(-0.5)
    assert len(s[0].target_sims) == 2 and len(s[0].contrast_sims) == 2  # raw sims retained


def test_hashing_embedder_is_deterministic_and_normalised():
    e = HashingEmbedder()
    a, b = e.encode(["secret society meets at midnight"]), e.encode(["secret society meets at midnight"])
    assert np.array_equal(a, b) and np.linalg.norm(a) == pytest.approx(1.0)
    sim_close = float((e.encode(["a secret society meeting"]) @ a.T)[0, 0])
    sim_far = float((e.encode(["bread flour water salt"]) @ a.T)[0, 0])
    assert sim_close > sim_far
    assert np.all(e.encode([""]) == 0)


def test_coherence_flags():
    assert coherence_features("")["flags"] == ["empty"]
    loop = coherence_features("the cat the cat the cat the cat the cat")
    assert "loop" in loop["flags"] and loop["coherence"] < 0.35
    good = coherence_features("She opened the window and listened to the rain falling on the quiet street below.")
    assert good["coherence"] > 0.9 and not good["flags"]
    assert "garbled" in coherence_features("\x00\x01\x02 abc \x03\x04\x05 def \x06\x07")["flags"]


def test_uncertainty_evaluator():
    ev = UncertaintyLexiconEvaluator()
    un = {"answerable": False, "answers": []}
    an = {"answerable": True, "answers": ["paris"]}
    assert ev.score(EvalContext("q", "I cannot know that.", un))["behavior_success"] == 1.0
    assert ev.score(EvalContext("q", "It was 4,212 people.", un))["behavior_success"] == 0.0
    assert ev.score(EvalContext("q", "Paris.", an))["behavior_success"] == 1.0
    assert ev.score(EvalContext("q", "I'm not sure, maybe Paris", an))["behavior_success"] == 0.0
    assert ev.score(EvalContext("q", "x", {})) == {}


def test_control_comparison_and_bh():
    c = control_comparison(0.5, [0.0, 0.1, 0.2, 0.3])
    assert c["percentile"] == 1.0 and c["p_value"] == pytest.approx(1 / 5)
    assert control_comparison(-1, [0, 1])["percentile"] == 0.0
    assert np.isnan(control_comparison(0.5, [])["p_value"])
    q = benjamini_hochberg([0.01, 0.04, 0.03, np.nan])
    assert q[0] == pytest.approx(0.03) and q[1] == pytest.approx(0.04) and q[2] == pytest.approx(0.04) and np.isnan(q[3])
    assert np.all(benjamini_hochberg([0.5, 0.9]) <= 1.0)


def test_bootstrap_ci_brackets_mean_and_is_seeded():
    v = np.random.default_rng(0).normal(0.3, 0.1, 40)
    lo, hi = bootstrap_ci(v)
    assert lo < v.mean() < hi and (lo, hi) == bootstrap_ci(v)
    assert np.isnan(bootstrap_ci([1.0])[0])


def test_diversity_greedy_avoids_duplicates():
    prof = np.array([[1, 0, 0], [0.99, 0.01, 0], [0, 1, 0], [0, 0, 1.0]])
    idx = greedy_diverse([4, 3, 2, 1], prof, k=3, max_cosine=0.9)
    assert idx == [0, 2, 3]
    assert cluster_labels(prof, 0.9) == [0, 0, 1, 2]


def test_llm_judge_blinded_and_pairwise_unbiased():
    import random
    seen = []
    judge = LLMJudge(lambda p: (seen.append(p), '{"criterion_score": 5, "coherence": 3, "explanation": "ok"}')[1],
                     "spooky tone", "fake")
    out = judge.score(EvalContext("prompt", "output text"))
    assert out["judge_score"] == 1.0 and out["judge_coherence"] == 0.5
    assert "steer" not in seen[0].lower() and "baseline" not in seen[0].lower()  # blinded to condition
    wins = {"A": 0, "B": 0}
    j2 = LLMJudge(lambda p: '{"winner": "A", "explanation": "x"}', "c", "fake")
    r = random.Random(0)
    res = [j2.pairwise("p", "base", "steer", r)["winner"] for _ in range(200)]
    assert 60 < res.count("steered") < 140  # always answering "A" must not systematically favour one condition


def test_target_spec_hash_changes_on_edit_and_lint():
    s = get_preset("terse")
    h = s.spec_hash
    s2 = s.model_copy(update={"target_descriptions": s.target_descriptions + ["extra"]})
    assert s2.spec_hash != h
    assert propose_spec("make it spooky").lint()  # skeleton is flagged
    with pytest.raises(Exception):
        TargetSpec(target_name="bad name!", target_descriptions=["a"], contrast_descriptions=["b"])


def test_propose_spec_with_llm_is_unreviewed():
    llm = lambda p: '{"target_descriptions": ["a1","a2","a3"], "contrast_descriptions": ["b1","b2","b3"], "task_type": "stylistic"}'
    s = propose_spec("be playful", llm)
    assert s.origin == "llm_proposed" and not s.reviewed
