"""Streamlit UI: new experiment, run progress, leaderboard, candidate detail, report.

Launch with ``nudge ui`` (or ``streamlit run src/nudge/ui/app.py``). The UI is a thin viewer/launcher over
the same run directories the CLI writes; it contains no scoring logic of its own.
"""
from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path

import pandas as pd
import streamlit as st

from nudge.experiments.config import ExperimentConfig
from nudge.experiments.prompts import BUILTIN_DIR
from nudge.experiments.storage import RunStore
from nudge.reports.markdown_report import build_report
from nudge.scoring import ranking as rk
from nudge.targets.schema import TargetSpec
from nudge.targets.target_builder import get_preset, list_presets


def _args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    a, _ = ap.parse_known_args(sys.argv[1:])
    return a


def list_runs(root: Path) -> list[str]:
    return sorted(p.name for p in root.glob("*") if (p / "generations.jsonl").exists() or (p / "config.json").exists())


def _params(cfg: dict) -> rk.RankingParams:
    s = cfg.get("selection", {})
    return rk.RankingParams(min_valid_prompts=s.get("min_valid_prompts", 5), min_coherence=s.get("min_coherence", 0.0),
                            control_percentile=s.get("control_percentile", 0.95))


@st.cache_data(show_spinner=False)
def _rows(run: str, root: str, n_gens: int, scorer_id: str) -> pd.DataFrame:  # n_gens busts the cache as the run grows
    return rk.build_rows(RunStore(Path(root) / run), scorer_id)


def _scorer_id(store: RunStore) -> str | None:
    return (store.read_json("meta.json", {}) or {}).get("scorer_id") or (store.scorer_ids() or [None])[-1]


# ------------------------------------------------------------------------------------------------ pages
def page_new(root: Path) -> None:
    st.header("New experiment")
    st.caption("Choose a backend, define the target and contrast descriptions, review the budget, launch.")
    c1, c2 = st.columns(2)
    with c1:
        kind = st.selectbox("Backend", ["tiny", "hf"], help="`tiny` is a random plumbing fixture; `hf` loads a real model.")
        model_id = st.text_input("HF model id", "Qwen/Qwen2.5-0.5B-Instruct", disabled=kind != "hf")
        name = st.text_input("Run name", "ui-run")
        preset = st.selectbox("Target preset (starting point)", list_presets())
    spec0 = get_preset(preset)
    with c2:
        goal = st.text_area("Target behavior (natural language)", spec0.goal, height=90)
        suites = sorted(p.stem for p in BUILTIN_DIR.glob("*.jsonl"))
        suite = st.selectbox("Prompt suite", suites, index=suites.index("fiction_scenes") if "fiction_scenes" in suites else 0)

    st.subheader("Target and contrast descriptions (edit freely)")
    t_txt = st.text_area("Target descriptions, one per line", "\n".join(spec0.target_descriptions), height=130)
    c_txt = st.text_area("Contrast descriptions, one per line", "\n".join(spec0.contrast_descriptions), height=130)
    try:
        spec = spec0.model_copy(update={"goal": goal, "target_descriptions": [x for x in t_txt.splitlines() if x.strip()],
                                        "contrast_descriptions": [x for x in c_txt.splitlines() if x.strip()],
                                        "reviewed": True, "origin": "edited"})
        TargetSpec.model_validate(spec.model_dump())
        for w in spec.lint():
            st.warning(w)
    except Exception as e:  # pydantic validation
        st.error(f"Invalid specification: {e}")
        return

    st.subheader("Search")
    a, b, c = st.columns(3)
    layers_txt = a.text_input("Layers (comma separated)", "2,3,4")
    n_neurons = b.number_input("Neurons per layer", 1, 500, 6)
    doses_txt = c.text_input("Doses (comma separated)", "-1.0,1.0")
    d, e, f = st.columns(3)
    max_new = d.number_input("Max new tokens", 8, 512, 24)
    n_controls = e.number_input("Random controls per (layer, dose)", 0, 64, 8)
    embedder = f.text_input("Embedder", "hashing", help="'hashing' (offline baseline) or a sentence-transformers model name")

    try:
        cfg = ExperimentConfig(name=name, run_dir=str(root), target=preset)
        cfg.backend.kind, cfg.backend.model_id = kind, model_id
        cfg.search.layers = [int(x) for x in layers_txt.split(",") if x.strip()]
        cfg.search.neurons_per_layer = int(n_neurons)
        cfg.search.doses = [float(x) for x in doses_txt.split(",") if x.strip()]
        cfg.generation = type(cfg.generation)(max_new_tokens=int(max_new))
        cfg.controls.per_layer_dose = int(n_controls)
        cfg.prompts.suite, cfg.scoring.embedder = suite, embedder
        n_cand = len(cfg.search.layers) * cfg.search.neurons_per_layer
        n_prompt = 12  # built-in suites ship 12 screening prompts
        est = n_prompt * (1 + n_cand * len(cfg.search.doses)
                          + cfg.controls.per_layer_dose * len(cfg.search.layers) * len(cfg.search.doses))
        st.info(f"≈ **{est:,} screening generations** ({n_cand} candidate neurons × {len(cfg.search.doses)} doses × "
                f"~{n_prompt} prompts, plus baselines and controls). Validation/held-out add generations for the shortlist only.")
    except ValueError as e:
        st.error(str(e))
        return

    if st.button("Launch screening → validation → held-out", type="primary"):
        from nudge.factory import build_runner

        cancel = threading.Event()
        runner = build_runner(cfg, spec=spec, cancel=cancel)

        def work() -> None:
            try:
                runner.run_all()
            except Exception as ex:  # surfaced on the progress page
                runner.store.write_json("error.json", {"error": f"{type(ex).__name__}: {ex}"})

        th = threading.Thread(target=work, daemon=True)
        th.start()
        st.session_state.update(thread=th, cancel=cancel, active_run=name)
        st.success(f"Started run `{name}`. Open **Run progress**.")


def page_progress(root: Path, run: str | None) -> None:
    st.header("Run progress")
    if not run:
        st.info("No runs yet.")
        return
    store = RunStore(root / run)
    p = store.read_json("progress.json")
    if store.read_json("error.json"):
        st.error(store.read_json("error.json")["error"])
    if not p:
        st.info("No progress recorded for this run.")
        return
    st.progress(min(1.0, p["done"] / max(1, p["total"])), text=f"{p['stage']}: {p['done']}/{p['total']}")
    a, b, c = st.columns(3)
    a.metric("Cached (skipped)", p["cached"])
    b.metric("Failures", p["failures"])
    c.metric("ETA (s)", "–" if p.get("eta_s") is None else f"{p['eta_s']:.0f}")
    st.caption(f"Current: {p['current']}")
    th = st.session_state.get("thread")
    live = th is not None and th.is_alive() and st.session_state.get("active_run") == run
    if live:
        if st.button("Cancel (resume later by re-running)"):
            st.session_state["cancel"].set()
        time.sleep(1.5)
        st.rerun()
    for w in (store.read_json("meta.json", {}) or {}).get("warnings", []):
        st.warning(w)
    st.json(store.summary())


def page_leaderboard(root: Path, run: str | None) -> None:
    st.header("Leaderboard")
    if not run:
        st.info("No runs yet.")
        return
    store = RunStore(root / run)
    sid = _scorer_id(store)
    if not sid:
        st.info("Nothing scored yet.")
        return
    cfg = store.read_json("config.json", {})
    rows = _rows(run, str(root), len(store.generations()), sid)
    part = st.radio("Partition", ["screening", "validation", "test"], horizontal=True)
    params = _params(cfg)
    table = rk.candidate_table(rows, params, part)
    conf = store.read_json("confirmation.json", {})
    vs = {f"{c['neuron_id']}@{c['dose']:+g}": c["status"] for c in conf.get("candidates", [])}
    lb = rk.leaderboard(table, params, vs)
    if part == "screening":
        st.caption("Exploratory: scored on the same prompts used to select. Treat as leads.")
    if part == "test" and not conf:
        st.info("The held-out set has not been evaluated.")
    if lb.empty:
        st.info("No eligible candidates in this partition.")
        return
    show = ["rank", "neuron_id", "dose", "n_prompts", "mean_dj", "std_dj", "success_rate", "mean_coherence",
            "behavior_delta", "control_percentile", "p_control", "q_value", "validation_status"]
    st.dataframe(lb[[c for c in show if c in lb]], width="stretch", hide_index=True)
    st.session_state["lb"] = lb
    st.caption(f"{len(table)} (neuron, dose) cells screened in this partition; q = Benjamini–Hochberg across them.")


def page_candidate(root: Path, run: str | None) -> None:
    st.header("Candidate detail")
    if not run:
        st.info("No runs yet.")
        return
    store = RunStore(root / run)
    sid = _scorer_id(store)
    if not sid:
        return
    rows = _rows(run, str(root), len(store.generations()), sid)
    cand = rows[(rows.condition == "steered") & (rows.n_iv == 1)]
    if cand.empty:
        st.info("No steered generations.")
        return
    ids = sorted({f"L{int(l)}.N{int(n)}" for l, n in zip(cand.layer, cand.neuron)})
    pick = st.selectbox("Neuron", ids)
    layer, neuron = (int(x) for x in pick[1:].split(".N"))
    sub = cand[(cand.layer == layer) & (cand.neuron == neuron)]
    doses = sorted(sub.dose.unique())
    part = st.radio("Partition", sorted(sub.partition.unique()), horizontal=True)
    sub = sub[sub.partition == part]
    summ = sub.groupby("dose").agg(n=("dj", "count"), mean_dj=("dj", "mean"), coherence=("coherence", "mean"),
                                   success_rate=("dj", lambda x: float((x > 0).mean()))).reset_index()
    st.dataframe(summ, hide_index=True, width="stretch")
    cfg = store.read_json("config.json", {})
    t = rk.candidate_table(rows, _params(cfg), part)
    t = t[(t.layer == layer) & (t.neuron == neuron)]
    if len(t):
        st.caption("Matched random-control comparison per dose")
        st.dataframe(t[["dose", "mean_dj", "control_percentile", "p_control", "n_controls"]], hide_index=True)
    if doses:
        st.subheader("Prompt-by-prompt: baseline vs steered")
        for pid in sorted(sub.prompt_id.unique())[:12]:
            r0 = rows[(rows.prompt_id == pid) & (rows.condition == "baseline")]
            with st.expander(f"{pid}"):
                cols = st.columns(1 + len(doses))
                cols[0].markdown("**baseline**")
                cols[0].code(r0.output.iloc[0] if len(r0) else "(none)", language=None)
                for col, d in zip(cols[1:], doses):
                    r = sub[(sub.prompt_id == pid) & (sub.dose == d)]
                    col.markdown(f"**dose {d:+g}**" + (f" · ΔJ {r.dj.iloc[0]:+.3f}" if len(r) and pd.notna(r.dj.iloc[0]) else ""))
                    col.code(r.output.iloc[0] if len(r) else "(none)", language=None)
    frozen = store.exists("finalists.json")
    st.button("Test combination (phase 2)", disabled=True,
              help="Disabled until the candidate is confirmed on held-out prompts; pair search is not part of the MVP.")
    if not frozen:
        st.caption("Validation/held-out runs are launched from the CLI or the New experiment page.")


def page_report(root: Path, run: str | None) -> None:
    st.header("Report")
    if not run:
        st.info("No runs yet.")
        return
    store = RunStore(root / run)
    if not store.exists("spec.json"):
        st.info("Run has no spec yet.")
        return
    text = build_report(store)
    st.download_button("Download report.md", text, file_name=f"{run}-report.md", mime="text/markdown")
    st.download_button("Download generations.jsonl", store.gen_path.read_bytes() if store.gen_path.exists() else b"",
                       file_name=f"{run}-generations.jsonl")
    st.markdown(text)


def main() -> None:
    st.set_page_config(page_title="Nudge", page_icon="🧠", layout="wide")
    root = Path(_args().runs)
    root.mkdir(parents=True, exist_ok=True)
    st.sidebar.title("Nudge")
    st.sidebar.caption("Use the agent to automate experiments, not to automate certainty.")
    page = st.sidebar.radio("Page", ["New experiment", "Run progress", "Leaderboard", "Candidate detail", "Report"])
    runs = list_runs(root)
    default = runs.index(st.session_state["active_run"]) if st.session_state.get("active_run") in runs else (len(runs) - 1 if runs else 0)
    run = st.sidebar.selectbox("Run", runs, index=default if runs else None) if runs else None
    {"New experiment": lambda: page_new(root), "Run progress": lambda: page_progress(root, run),
     "Leaderboard": lambda: page_leaderboard(root, run), "Candidate detail": lambda: page_candidate(root, run),
     "Report": lambda: page_report(root, run)}[page]()


main()
