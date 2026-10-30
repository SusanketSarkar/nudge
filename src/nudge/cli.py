"""``nudge`` command line."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from nudge.experiments.config import ExperimentConfig, load_config
from nudge.experiments.storage import RunStore


def _cfg(path: str | None, run_dir: str | None, name: str | None) -> ExperimentConfig:
    cfg = load_config(path) if path else ExperimentConfig()
    if run_dir:
        cfg.run_dir = run_dir
    if name:
        cfg.name = name
    return cfg


def _print_df(df, cols, n=10) -> None:
    if df is None or len(df) == 0:
        print("  (empty)")
        return
    cols = [c for c in cols if c in df.columns]
    print(df.head(n)[cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))


def cmd_doctor(a) -> int:
    from nudge.steering.validation import run_self_checks

    if a.backend == "tiny":
        from nudge.models.tiny import build_tiny_backend

        backend = build_tiny_backend()
    else:
        from nudge.models.hf_backend import HFBackend

        backend = HFBackend.from_pretrained(a.model, device=a.device)
    meta = backend.get_model_metadata()
    print(f"model: {meta['model_id']}  layers={meta['n_layers']} hidden={meta['hidden_size']} "
          f"intermediate={meta['intermediate_size']} dtype={meta['dtype']} device={meta['device']}")
    results = run_self_checks(backend, layer=a.layer, neuron=a.neuron)
    for r in results:
        print(f"  [{'PASS' if r.passed else 'FAIL'}] {r.name}: {r.detail}")
    ok = all(r.passed for r in results)
    print("doctor:", "all checks passed" if ok else "FAILED: do not trust results from this backend")
    return 0 if ok else 1


def cmd_target(a) -> int:
    from nudge.targets.target_builder import (
        get_preset,
        list_openai_models,
        list_presets,
        openai_compatible_call,
        propose_spec,
    )

    if a.list:
        print("\n".join(list_presets()))
        return 0
    if a.auto:
        if not a.goal:
            print("--auto requires --goal", file=sys.stderr)
            return 2
        if a.preset:
            print("--auto cannot be combined with --preset", file=sys.stderr)
            return 2
        model = a.model
        try:
            if not model:
                model = next(iter(list_openai_models(a.base_url, a.api_key)), "")
                if not model:
                    print(f"--auto needs --model (or NUDGE_LLM_MODEL); none advertised at {a.base_url}",
                          file=sys.stderr)
                    return 2
            llm = openai_compatible_call(a.base_url, model, a.api_key, a.llm_timeout)
            spec = propose_spec(a.goal, llm)
        except (RuntimeError, ValueError, KeyError, TypeError) as e:
            print(f"auto proposal failed: {e}", file=sys.stderr)
            return 1
        print(f"auto: {model} proposed this spec via {a.base_url}; review the descriptions and set "
              "reviewed=true before using it", file=sys.stderr)
    else:
        spec = get_preset(a.preset) if a.preset else propose_spec(a.goal)
    if a.out:
        spec.save(a.out)
        print(f"wrote {a.out}")
    else:
        print(json.dumps(spec.model_dump(), indent=2))
    for w in spec.lint():
        print("lint:", w, file=sys.stderr)
    return 0


def cmd_plan(a) -> int:
    from nudge.factory import build_runner

    r = build_runner(_cfg(a.config, a.run_dir, a.name))
    print(json.dumps(r.estimate(), indent=2))
    return 0


def cmd_run(a) -> int:
    from nudge.factory import build_runner
    from nudge.reports.markdown_report import write_report

    cfg = _cfg(a.config, a.run_dir, a.name)
    def show(p) -> None:
        if p.done == p.total or p.done % 50 == 0:
            eta = f", eta {p.eta_s:.0f}s" if p.eta_s else ""
            print(f"[{p.stage}] {p.done}/{p.total} (cached {p.cached}, failures {p.failures}{eta})", flush=True)

    r = build_runner(cfg, progress=show)
    print("plan:", json.dumps(r.estimate()))
    stage = a.stage
    if stage in ("screen", "all"):
        lb = r.screen()
        print("\n== leaderboard (screening, exploratory) ==")
        _print_df(lb, ["rank", "neuron_id", "dose", "n_prompts", "mean_dj", "success_rate", "mean_coherence",
                       "control_percentile", "p_control", "q_value"], 10)
    if stage == "all":
        short = r.shortlist()
        print(f"\n== shortlist: {len(short)} ==")
        val = r.validate(short)
        fin = r.pick_finalists(val)
        print(f"== finalists: {[f['neuron_id'] + format(f['dose'], '+g') for f in fin]} ==")
        if fin:
            r.freeze(fin)
            res = r.test(force=a.force_test)
            print(f"\n== held-out: {res['n_confirmed']}/{res['n_finalists']} confirmed ==")
            for c in res["candidates"]:
                print(f"  {c['neuron_id']} {c['dose']:+g}: {c['status']}  {'; '.join(c['reasons'])}")
        else:
            print("no finalists to freeze; held-out set untouched")
    elif stage == "validate":
        short = r.shortlist()
        val = r.validate(short)
        fin = r.pick_finalists(val)
        print(f"\nfinalists proposed: {fin}")
        if a.freeze and fin:
            r.freeze(fin)
            print("frozen finalists.json")
    elif stage == "test":
        res = r.test(force=a.force_test)
        print(json.dumps({k: v for k, v in res.items() if k != "candidates"}, indent=2))
    print("\nreport:", write_report(r.store))
    return 0


def _fmt_score(x) -> str:
    return "n/a" if x is None else f"{x:+.3f}"


def _print_check_row(label: str, r: dict) -> None:
    status = "" if r.get("status") == "ok" else f" [status={r.get('status')} {r.get('error') or ''}]"
    print(f"  {label:<8} J={_fmt_score(r.get('j'))}  target={_fmt_score(r.get('s_target'))}  "
          f"contrast={_fmt_score(r.get('s_contrast'))}  coh={r.get('coherence', 0):.2f}  "
          f"rel={_fmt_score(r.get('relevance'))}  flags={','.join(r.get('flags') or []) or '-'}{status}")
    print(f"           {r.get('text', '').strip() or '(empty)'}")


def cmd_check(a) -> int:
    from nudge.experiments.check import parse_neuron_id, run_check
    from nudge.experiments.config import config_from_dict, load_config
    from nudge.experiments.storage import RunStore
    from nudge.factory import build_backend, load_spec
    from nudge.models.base import GenerationConfig
    from nudge.targets.schema import TargetSpec

    if a.run:
        store = RunStore(a.run)
        cfg_doc = store.read_json("config.json")
        if cfg_doc is None or not store.exists("spec.json"):
            print(f"{a.run} is not a run directory (missing config.json/spec.json)", file=sys.stderr)
            return 2
        cfg = config_from_dict(cfg_doc)
        spec = TargetSpec.load(store.path("spec.json"))
    else:
        cfg = load_config(a.config)
        spec = load_spec(cfg.target)

    if a.neuron_id:
        try:
            layer, neuron = parse_neuron_id(a.neuron_id)
        except ValueError as e:
            print(e, file=sys.stderr)
            return 2
    elif a.layer is not None and a.neuron is not None:
        layer, neuron = a.layer, a.neuron
    else:
        print("give a neuron id (e.g. L12.N4254) or both --layer and --neuron", file=sys.stderr)
        return 2

    queries = list(a.query)
    if a.query_file:
        queries += [ln.strip() for ln in Path(a.query_file).read_text(encoding="utf-8").splitlines() if ln.strip()]
    if not queries:
        print("give at least one --query (repeatable) or --query-file", file=sys.stderr)
        return 2

    doses = a.dose or list(cfg.search.doses)
    if a.dose_unit:
        cfg.search.dose_unit = a.dose_unit
    if a.device:
        cfg.backend.device = a.device
    gen = cfg.generation
    if a.max_new_tokens is not None:
        gen = GenerationConfig(**{**gen.to_dict(), "max_new_tokens": a.max_new_tokens})
    n_controls = a.controls if a.controls >= 0 else cfg.controls.per_layer_dose

    try:
        results = run_check(backend=build_backend(cfg), spec=spec, queries=queries, layer=layer,
                            neuron=neuron, doses=doses, dose_unit=cfg.search.dose_unit,
                            generation=gen, embedder_name=cfg.scoring.embedder,
                            n_controls=n_controls, seed=cfg.seed)
    except ValueError as e:
        print(f"check failed: {e}", file=sys.stderr)
        return 1

    if a.json:
        print(json.dumps({"model": cfg.backend.model_id, "layer": layer, "neuron": neuron,
                          "spec": spec.target_name, "results": results}, indent=2))
        return 0

    print(f"check L{layer}.N{neuron}  model={cfg.backend.model_id or cfg.backend.kind}  "
          f"spec={spec.target_name}  unit={cfg.search.dose_unit}")
    for q in results:
        print(f"\nQUERY: {q['query']}")
        _print_check_row("baseline", q["baseline"])
        for r in q["doses"]:
            _print_check_row(f"dose {r['dose']:+g}", r)
            if "dj" in r:
                print(f"           shift deltaJ={r['dj']:+.4f}  d_coherence={r['d_coherence']:+.3f}  "
                      f"d_relevance={r['d_relevance']:+.3f}")
            if "controls" in r:
                c = r["controls"]
                print(f"           controls K={int(c['n_controls'])}  percentile={c['percentile']:.3f}  "
                      f"p={c['p_value']:.3f}" if c["n_controls"] else "           controls: none completed")
    print("\nNote: a positive shift means the output moved toward the target descriptions, not that the "
          "neuron encodes the concept. Read the text and compare with controls.")
    return 0


def cmd_report(a) -> int:
    from nudge.reports.markdown_report import write_report

    print(write_report(RunStore(a.run_dir), a.out))
    return 0


def cmd_demo(a) -> int:
    a.config = str(Path(__file__).resolve().parents[2] / "configs" / "demo_tiny.toml")
    a.stage, a.force_test, a.freeze = "all", False, False
    return cmd_run(a)


def cmd_ui(a) -> int:
    import subprocess

    app = Path(__file__).parent / "ui" / "app.py"
    return subprocess.call([sys.executable, "-m", "streamlit", "run", str(app), "--", "--runs", a.runs])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="nudge", description="Agentic lab for neuron-steering discovery")
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("doctor", help="Phase-0 self-checks of the steering wiring")
    d.add_argument("--backend", choices=["tiny", "hf"], default="tiny")
    d.add_argument("--model", default="")
    d.add_argument("--device", default="auto")
    d.add_argument("--layer", type=int)
    d.add_argument("--neuron", type=int, default=0)
    d.set_defaults(fn=cmd_doctor)

    t = sub.add_parser("target", help="create / inspect a target specification")
    t.add_argument("--list", action="store_true")
    t.add_argument("--preset")
    t.add_argument("--goal")
    t.add_argument("--out")
    t.add_argument("--auto", action="store_true",
                   help="have an OpenAI-compatible LLM (e.g. a local vLLM server) propose the spec; "
                        "the result stays unreviewed for a human to edit")
    t.add_argument("--base-url", default=os.environ.get("NUDGE_LLM_BASE_URL", "http://localhost:8000/v1"),
                   help="OpenAI-compatible base URL for --auto (default: $NUDGE_LLM_BASE_URL or "
                        "http://localhost:8000/v1)")
    t.add_argument("--model", default=os.environ.get("NUDGE_LLM_MODEL", ""),
                   help="model id for --auto (default: $NUDGE_LLM_MODEL, else first model the server advertises)")
    t.add_argument("--api-key", default=os.environ.get("NUDGE_LLM_API_KEY") or os.environ.get("OPENAI_API_KEY"),
                   help="API key for --auto (default: $NUDGE_LLM_API_KEY or $OPENAI_API_KEY)")
    t.add_argument("--llm-timeout", type=float, default=120.0, help="request timeout in seconds for --auto")
    t.set_defaults(fn=cmd_target)

    for name, fn, h in (("plan", cmd_plan, "estimate generation counts"), ("run", cmd_run, "run an experiment")):
        s = sub.add_parser(name, help=h)
        s.add_argument("config", nargs="?")
        s.add_argument("--run-dir")
        s.add_argument("--name")
        if name == "run":
            s.add_argument("--stage", choices=["screen", "validate", "test", "all"], default="all")
            s.add_argument("--freeze", action="store_true", help="with --stage validate: freeze proposed finalists")
            s.add_argument("--force-test", action="store_true", help="re-run held-out test (marks it contaminated)")
        s.set_defaults(fn=fn)

    c = sub.add_parser("check", help="manually compare baseline vs steered output for one neuron + dose")
    c.add_argument("neuron_id", nargs="?", help="e.g. L12.N4254 (or use --layer/--neuron)")
    c.add_argument("--layer", type=int)
    c.add_argument("--neuron", type=int)
    c.add_argument("--dose", type=float, action="append",
                   help="repeatable; defaults to the config's search.doses")
    c.add_argument("--dose-unit", help="override the config's dose unit")
    c.add_argument("--query", action="append", default=[], help="repeatable ad-hoc prompt")
    c.add_argument("--query-file", help="text file with one query per line")
    src = c.add_mutually_exclusive_group(required=True)
    src.add_argument("--run", help="run directory containing config.json + spec.json")
    src.add_argument("--config", help="experiment TOML; uses its target spec")
    c.add_argument("--controls", type=int, default=0,
                   help="matched random-direction controls per (query, dose); -1 = config per_layer_dose")
    c.add_argument("--max-new-tokens", type=int)
    c.add_argument("--device", help="override the config's backend.device")
    c.add_argument("--json", action="store_true", help="machine-readable output")
    c.set_defaults(fn=cmd_check)

    r = sub.add_parser("report", help="export the Markdown report for a run directory")
    r.add_argument("run_dir")
    r.add_argument("--out")
    r.set_defaults(fn=cmd_report)

    dm = sub.add_parser("demo", help="end-to-end plumbing demo on the tiny random model")
    dm.add_argument("--run-dir")
    dm.add_argument("--name")
    dm.set_defaults(fn=cmd_demo)

    u = sub.add_parser("ui", help="launch the Streamlit UI")
    u.add_argument("--runs", default="runs")
    u.set_defaults(fn=cmd_ui)

    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
