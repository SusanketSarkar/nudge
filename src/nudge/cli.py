"""``nudge`` command line."""
from __future__ import annotations

import argparse
import json
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
    from nudge.targets.target_builder import get_preset, list_presets, propose_spec

    if a.list:
        print("\n".join(list_presets()))
        return 0
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
