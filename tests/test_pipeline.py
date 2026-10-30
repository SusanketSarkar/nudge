"""End-to-end on the tiny fixture: resume, caching, held-out guard, storage robustness, report, UI."""
import sys
from pathlib import Path

import pytest

from nudge.experiments.confirm import HeldOutError
from nudge.experiments.runner import ExperimentRunner
from nudge.experiments.storage import RunStore
from nudge.factory import build_runner
from nudge.reports.markdown_report import build_report

APP = str(Path(__file__).resolve().parents[1] / "src" / "nudge" / "ui" / "app.py")


def test_screen_resume_uses_cache_and_scores_are_stable(small_cfg, backend):
    r = build_runner(small_cfg, backend=backend)
    est = r.estimate()
    assert est["screening_generations"] == est["baseline"] + est["steered"] + est["control"] > 0
    lb1 = r.screen()
    n = len(r.store.generations())
    assert n == est["screening_generations"]
    r2 = build_runner(small_cfg, backend=backend)  # fresh runner, same run dir => resume
    assert r2.estimate()["already_cached"] == n
    calls = []
    orig = backend.run_steered
    backend.run_steered = lambda *a, **k: (calls.append(1), orig(*a, **k))[1]
    try:
        lb2 = r2.screen()
    finally:
        backend.run_steered = orig
    assert not calls, "cached cells must not be regenerated"
    assert list(lb1.neuron_id) == list(lb2.neuron_id) and list(lb1.mean_dj.round(9)) == list(lb2.mean_dj.round(9))


def test_cancel_then_resume(small_cfg, backend):
    r = build_runner(small_cfg, backend=backend)
    r.cancel.set()
    r.screen()
    assert len(r.store.generations()) == 0
    r.cancel.clear()
    r.screen()
    assert len(r.store.generations()) == r.estimate()["screening_generations"]


def test_failures_are_recorded_and_excluded(small_cfg, backend):
    r = build_runner(small_cfg, backend=backend)
    orig = backend.run_steered

    def flaky(prompt, iv, cfg):
        from nudge.steering.interventions import as_tuple
        if as_tuple(iv)[0].neuron % 2 == 0 and as_tuple(iv)[0].kind == "neuron":
            from nudge.models.base import GenerationResult
            return GenerationResult(text="", status="error", error="synthetic")
        return orig(prompt, iv, cfg)

    backend.run_steered = flaky
    try:
        r.screen()
    finally:
        backend.run_steered = orig
    errs = [g for g in r.store.generations() if g.status == "error"]
    assert errs and all(g.error == "synthetic" for g in errs)
    rows = r.rows()
    assert not rows[rows.key.isin([g.key for g in errs])].ok.any()


def test_heldout_guard(small_cfg, backend):
    r = build_runner(small_cfg, backend=backend)
    r.screen()
    with pytest.raises(HeldOutError):
        r.test()  # not frozen yet
    assert not any(g.partition == "test" for g in r.store.generations())
    fin = r.pick_finalists(r.validate(r.shortlist()))
    assert fin
    r.freeze(fin)
    res = r.test()
    assert not res["contaminated"] and res["n_finalists"] == len(fin)
    with pytest.raises(HeldOutError):
        r.test()  # second look refused
    res2 = r.test(force=True)
    assert res2["contaminated"]
    assert "contaminated" in build_report(r.store)


def test_full_run_and_report(small_cfg, backend):
    r = build_runner(small_cfg, backend=backend)
    summary = r.run_all()
    assert summary["screening_rows"] > 0
    text = build_report(r.store)
    for h in ("## 1. Research question", "## 3. Search procedure and budget", "## 5. Exploratory findings",
              "## 7. Held-out confirmation", "leads", "Benjamini"):
        assert h in text
    # model/seed/dose provenance present
    assert "weights_fingerprint" in text and "rel_mlp_out_norm" in text and "seed 1" in text


def test_every_generation_record_is_auditable(small_cfg, backend):
    r = build_runner(small_cfg, backend=backend)
    r.screen()
    for g in r.store.generations()[:20]:
        assert g.model_id and g.tokenizer_id and g.gen_config and g.seed == 0 and g.prompt
        assert g.condition == "baseline" or g.interventions
    assert all(i["dose_unit"] for g in r.store.generations() for i in g.interventions)


def test_store_survives_truncated_last_line(tmp_path):
    s = RunStore(tmp_path / "x")
    s.gen_path.write_text('{"key": "a"')  # crash mid-write
    s2 = RunStore(tmp_path / "x")
    assert s2.generations() == []


def test_rescoring_with_new_spec_keeps_old_scores(small_cfg, backend):
    r = build_runner(small_cfg, backend=backend)
    r.screen()
    n0 = len(r.store.scorer_ids())
    spec2 = r.spec.model_copy(update={"target_descriptions": r.spec.target_descriptions + ["a new description"]})
    r2 = ExperimentRunner(backend, r.store, spec2, r.suite, small_cfg)
    r2.score()
    assert r2.scorer.scorer_id != r.scorer.scorer_id
    assert len(r.store.scorer_ids()) == n0 + 1


def test_candidate_beating_controls_is_detected(small_cfg, backend):
    """Sanity of the statistics path: a fabricated large effect must rank above matched controls."""
    import numpy as np
    from nudge.scoring.controls import control_comparison
    ctrl = np.random.default_rng(0).normal(0, 0.01, 50)
    assert control_comparison(0.2, ctrl)["percentile"] == 1.0
    assert control_comparison(0.0, ctrl)["percentile"] < 0.8


def test_survey_and_neuron_all_sentinels(tmp_path, backend):
    from nudge.experiments.config import config_from_dict

    cfg = config_from_dict({
        "experiment": {"name": "all", "seed": 0, "run_dir": str(tmp_path)},
        "backend": {"kind": "tiny"},
        "search": {"survey_layers": -1, "neurons_per_layer": -1, "doses": [1.0]},
        "prompts": {"suite": "fiction_scenes", "n_screening": 1},
    })
    r = build_runner(cfg, backend=backend)
    assert r.layers() == backend.list_supported_layers()
    assert r.estimate()["n_candidates"] == len(r.layers()) * backend.intermediate_size


def test_run_check_reports_shift(backend):
    from nudge.experiments.check import parse_neuron_id, run_check
    from nudge.models.base import GenerationConfig
    from nudge.targets.target_builder import get_preset

    assert parse_neuron_id("L12.N4254") == (12, 4254)
    assert parse_neuron_id("3.7") == (3, 7)
    out = run_check(backend=backend, spec=get_preset("secret_society_thriller"), queries=["hello"],
                    layer=2, neuron=1, doses=[1.0], generation=GenerationConfig(max_new_tokens=4),
                    n_controls=2)
    row = out[0]["doses"][0]
    assert out[0]["query"] == "hello" and row["status"] == "ok"
    assert row["dj"] == pytest.approx(row["j"] - out[0]["baseline"]["j"])
    assert "controls" in row and row["controls"]["n_controls"] == 2


def test_cli_check_prints_shift(small_cfg, backend, capsys):
    from nudge.cli import main

    r = build_runner(small_cfg, backend=backend)
    r.screen()
    rc = main(["check", "L2.N1", "--run", str(small_cfg.run_path), "--dose", "1.0",
               "--query", "hello", "--controls", "2"])
    out = capsys.readouterr().out
    assert rc == 0 and "baseline" in out and "dose +1" in out and "shift deltaJ" in out


def test_cli_doctor_and_target(capsys):
    from nudge.cli import main
    assert main(["doctor"]) == 0
    assert main(["target", "--list"]) == 0
    assert "secret_society_thriller" in capsys.readouterr().out


def test_cli_target_auto_uses_llm_and_stays_unreviewed(tmp_path, monkeypatch):
    import nudge.targets.target_builder as tb
    from nudge.cli import main
    from nudge.targets.schema import TargetSpec

    seen = {}

    def fake_call(base_url, model, api_key=None, timeout=120.0):
        seen.update(base_url=base_url, model=model)
        return lambda prompt: ('{"target_descriptions": ["z1", "z2", "z3"], '
                               '"contrast_descriptions": ["c1", "c2", "c3"], "task_type": "stylistic"}')

    monkeypatch.setattr(tb, "openai_compatible_call", fake_call)
    out = tmp_path / "zombie.json"
    rc = main(["target", "--auto", "--goal", "sound like a zombie", "--model", "fake-model",
               "--base-url", "http://example/v1", "--out", str(out)])
    assert rc == 0
    spec = TargetSpec.load(out)
    assert spec.origin == "llm_proposed" and not spec.reviewed
    assert spec.target_descriptions == ["z1", "z2", "z3"]
    assert seen == {"base_url": "http://example/v1", "model": "fake-model"}


def test_cli_target_auto_requires_goal(capsys):
    from nudge.cli import main
    assert main(["target", "--auto", "--model", "fake-model", "--out", "unused.json"]) == 2
    assert "--auto requires --goal" in capsys.readouterr().err


def test_ui_pages_render(small_cfg, backend, tmp_path, monkeypatch):
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest
    r = build_runner(small_cfg, backend=backend)
    r.run_all()
    monkeypatch.setattr(sys, "argv", ["app", "--runs", str(small_cfg.run_dir)])
    for page in ["New experiment", "Run progress", "Leaderboard", "Candidate detail", "Report"]:
        at = AppTest.from_file(APP, default_timeout=90).run()
        at.sidebar.radio[0].set_value(page).run()
        assert not at.exception, (page, [e.value for e in at.exception])
