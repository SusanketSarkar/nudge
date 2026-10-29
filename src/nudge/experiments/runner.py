"""Experiment runner: screening -> shortlist -> validation -> freeze -> held-out test.

Design points
-------------
* Every (model, prompt, interventions, generation-config) cell has a stable key; cells already stored
  are never regenerated, so an interrupted run resumes by simply running again.
* Failures are recorded (status="error") and retried once, never swallowed.
* Scoring is separate from generation (``ScoringPipeline``) and can be redone with a new scorer.
* The held-out partition is protected by ``confirm.check_test_allowed``.
"""
from __future__ import annotations

import platform
import sys
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd

from nudge.experiments import confirm as cf
from nudge.experiments.config import ExperimentConfig
from nudge.experiments.prompts import Prompt, PromptSuite
from nudge.experiments.sampling import (candidate_neurons, layers_by_depth, make_candidate_interventions,
                                        make_controls)
from nudge.experiments.storage import (COND_BASELINE, COND_CONTROL, COND_STEERED, GenerationRecord,
                                       RunStore, generation_key)
from nudge.models.base import GenerationConfig, SteeringBackend
from nudge.scoring import ranking as rk
from nudge.scoring.behavioral import EVALUATORS, Evaluator
from nudge.scoring.diversity import greedy_diverse
from nudge.scoring.embeddings import Embedder, get_embedder
from nudge.scoring.pipeline import ScoringPipeline
from nudge.steering.interventions import Intervention, as_tuple
from nudge.targets.schema import TargetSpec

ProgressFn = Callable[["Progress"], None]


@dataclass
class Progress:
    stage: str
    done: int
    total: int
    cached: int
    failures: int
    current: str
    elapsed_s: float
    eta_s: float | None
    cancelled: bool = False

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


@dataclass
class Job:
    stage: str
    prompt: Prompt
    condition: str
    interventions: tuple[Intervention, ...]


class ExperimentRunner:
    def __init__(self, backend: SteeringBackend, store: RunStore, spec: TargetSpec, suite: PromptSuite,
                 config: ExperimentConfig, embedder: Embedder | None = None,
                 evaluators: Sequence[Evaluator] | None = None, progress: ProgressFn | None = None,
                 cancel: threading.Event | None = None, max_retries: int = 1) -> None:
        self.backend, self.store, self.spec, self.suite, self.cfg = backend, store, spec, suite, config
        self.progress_cb, self.cancel, self.max_retries = progress, cancel or threading.Event(), max_retries
        if evaluators is None:
            evaluators = [EVALUATORS[spec.behavior_evaluator]()] if spec.behavior_evaluator else []
        self.embedder = embedder or get_embedder(config.scoring.embedder)
        self.scorer = ScoringPipeline(spec, self.embedder, evaluators)
        self.gen_cfg: GenerationConfig = config.generation
        self.meta = backend.get_model_metadata()
        self.params = rk.RankingParams(
            min_valid_prompts=config.selection.min_valid_prompts, min_coherence=config.selection.min_coherence,
            control_percentile=config.selection.control_percentile)
        self._failures = 0

    # ---------------------------------------------------------------- planning -------------------------
    def layers(self) -> list[int]:
        s = self.cfg.search
        avail = self.backend.list_supported_layers()
        if s.layers is not None:
            bad = [l for l in s.layers if l not in avail]
            if bad:
                raise ValueError(f"layers {bad} not supported by backend (available: {avail[0]}..{avail[-1]})")
            return list(s.layers)
        if s.survey_layers:
            return [l for l in layers_by_depth(len(avail), s.survey_layers, avail[0]) if l in avail]
        return [avail[len(avail) // 2]]

    def _sizes(self, layers: Sequence[int]) -> dict[int, int]:
        return {l: len(self.backend.list_neurons(l)) for l in layers}

    def _prompts(self, partition: str, cap: int | None = None) -> list[Prompt]:
        ps = self.suite.partition(partition)
        return ps[:cap] if cap else ps

    def candidate_list(self) -> list[Intervention]:
        layers = self.layers()
        pairs = candidate_neurons(layers, self.cfg.search.neurons_per_layer, self._sizes(layers), self.cfg.seed)
        return make_candidate_interventions(pairs, self.cfg.search.doses, self.cfg.search.dose_unit)

    def control_list(self, layers_doses: Sequence[tuple[int, float]]) -> list[Intervention]:
        out: list[Intervention] = []
        c = self.cfg.controls
        for layer in sorted({l for l, _ in layers_doses}):
            doses = sorted({d for l, d in layers_doses if l == layer})
            out += make_controls(layer, doses, self.cfg.search.dose_unit, c.per_layer_dose,
                                 len(self.backend.list_neurons(layer)), self.cfg.seed)
        return out

    def plan_screening(self) -> list[Job]:
        jobs = [Job("screening", p, COND_BASELINE, ()) for p in self._prompts("screening", self.cfg.prompts.n_screening)]
        prompts = self._prompts("screening", self.cfg.prompts.n_screening)
        for iv in self.candidate_list():
            jobs += [Job("screening", p, COND_STEERED, (iv,)) for p in prompts]
        ld = [(iv.layer, iv.dose) for iv in self.candidate_list()]
        cprompts = prompts[: self.cfg.controls.n_prompts] if self.cfg.controls.n_prompts else prompts
        for iv in self.control_list(sorted(set(ld))):
            jobs += [Job("screening", p, COND_CONTROL, (iv,)) for p in cprompts]
        return jobs

    def estimate(self) -> dict[str, Any]:
        jobs = self.plan_screening()
        by = {c: sum(j.condition == c for j in jobs) for c in (COND_BASELINE, COND_STEERED, COND_CONTROL)}
        cached = sum(self.store.has_generation(self._key(j)) for j in jobs)
        return {"layers": self.layers(), "n_candidates": len({(i.layer, i.neuron) for i in self.candidate_list()}),
                "screening_generations": len(jobs), "already_cached": cached, **by,
                "full_pipeline_note": "validation and held-out stages add generations for the shortlist only"}

    # ---------------------------------------------------------------- execution ------------------------
    def _key(self, job: Job) -> str:
        return generation_key(self.meta["model_id"], self.meta.get("weights_fingerprint"), job.prompt.text,
                              [i.to_dict() for i in job.interventions], self.gen_cfg.to_dict())

    def _emit(self, stage: str, done: int, total: int, cached: int, current: str, t0: float, cancelled=False) -> None:
        el = time.time() - t0
        fresh = done - cached
        eta = (el / fresh) * (total - done) if fresh > 0 and done < total else None
        p = Progress(stage, done, total, cached, self._failures, current, el, eta, cancelled)
        self.store.write_json("progress.json", p.to_dict())
        if self.progress_cb:
            self.progress_cb(p)

    def execute(self, jobs: Sequence[Job], stage: str) -> dict[str, int]:
        t0, cached, done = time.time(), 0, 0
        self._failures = 0
        total = len(jobs)
        for job in jobs:
            if self.cancel.is_set():
                self._emit(stage, done, total, cached, "cancelled", t0, cancelled=True)
                return {"done": done, "cached": cached, "failures": self._failures, "cancelled": 1}
            key = self._key(job)
            if self.store.has_generation(key):
                cached += 1
            else:
                res = None
                for _ in range(self.max_retries + 1):
                    res = (self.backend.run_baseline(job.prompt.text, self.gen_cfg) if not job.interventions
                           else self.backend.run_steered(job.prompt.text, job.interventions, self.gen_cfg))
                    if res.status == "ok":
                        break
                if res.status != "ok":
                    self._failures += 1
                self.store.add_generation(GenerationRecord(
                    key=key, stage=job.stage, partition=job.prompt.partition, prompt_id=job.prompt.id,
                    prompt=job.prompt.text, condition=job.condition,
                    interventions=[i.to_dict() for i in job.interventions], gen_config=self.gen_cfg.to_dict(),
                    seed=self.gen_cfg.seed, output=res.text, status=res.status, error=res.error,
                    latency_s=res.latency_s, n_prompt_tokens=res.n_prompt_tokens, n_output_tokens=len(res.token_ids),
                    model_id=self.meta["model_id"], tokenizer_id=str(self.meta.get("tokenizer_id", "")),
                    prompt_meta=job.prompt.meta))
            done += 1
            self._emit(stage, done, total, cached, f"{job.condition} {job.prompt.id} "
                       + ("/".join(i.id for i in job.interventions) or "baseline"), t0)
        return {"done": done, "cached": cached, "failures": self._failures, "cancelled": 0}

    # ---------------------------------------------------------------- bookkeeping ----------------------
    def write_run_files(self) -> None:
        self.spec.save(self.store.path("spec.json"))
        self.store.write_json("config.json", self.cfg.to_dict())
        old = self.store.read_json("meta.json", {})
        layers = self.layers()
        warns = []
        if any(l < 3 for l in layers):
            warns.append("Layers < 3 selected: relative doses are unreliable in the first layers "
                         "(many empty/looping outputs, controls included); inspect coherence closely.")
        if self.cfg.backend.kind == "tiny":
            warns.append("Backend is the random tiny fixture: outputs are gibberish; results are plumbing tests only.")
        self.store.write_json("meta.json", {
            **old, "model": self.meta, "spec_hash": self.spec.spec_hash, "scorer_id": self.scorer.scorer_id,
            "embedder": self.embedder.name, "prompt_suite": self.suite.name,
            "prompt_fingerprint": self.suite.fingerprint, "prompt_counts": self.suite.counts(),
            "search_budget": {"layers": layers, "neurons_per_layer": self.cfg.search.neurons_per_layer,
                              "n_candidate_neurons": len(layers) * self.cfg.search.neurons_per_layer,
                              "doses": self.cfg.search.doses, "controls_per_layer_dose": self.cfg.controls.per_layer_dose},
            "environment": {"python": sys.version.split()[0], "platform": platform.platform(),
                            **_versions()},
            "warnings": warns, "started_at": old.get("started_at", time.time())})

    def score(self) -> int:
        return self.scorer.score_store(self.store)

    def rows(self) -> pd.DataFrame:
        return rk.build_rows(self.store, self.scorer.scorer_id)

    def _validation_status(self) -> dict[str, str]:
        conf = self.store.read_json("confirmation.json", {})
        return {f"{c['neuron_id']}@{c['dose']:+g}": c["status"] for c in conf.get("candidates", [])}

    # ---------------------------------------------------------------- stages ---------------------------
    def screen(self) -> pd.DataFrame:
        self.write_run_files()
        stats = self.execute(self.plan_screening(), "screening")
        self.score()
        self.store.write_json("screening_summary.json", stats)
        return self.leaderboard("screening")

    def leaderboard(self, partition: str = "screening") -> pd.DataFrame:
        table = rk.candidate_table(self.rows(), self.params, partition)
        return rk.leaderboard(table, self.params, self._validation_status())

    def shortlist(self) -> pd.DataFrame:
        """Top screening candidates: positive, eligible, above controls (if run), diverse."""
        s = self.cfg.selection
        rows = self.rows()
        table = rk.candidate_table(rows, self.params, "screening")
        lb = rk.leaderboard(table, self.params)
        if lb.empty:
            return lb
        lb = lb[lb.mean_dj > 0]
        if s.require_control and lb.control_percentile.notna().any():
            lb = lb[lb.beats_control == True]  # noqa: E712 (object column)
        if lb.empty:
            return lb
        prof = rk.profile_matrix(rows, lb, "screening")
        idx = greedy_diverse(lb.mean_dj.to_numpy(), prof, s.shortlist, s.max_profile_cosine)
        return lb.iloc[idx].reset_index(drop=True)

    def validate(self, shortlist: pd.DataFrame) -> pd.DataFrame:
        """Re-test shortlisted neurons on the validation partition (screening doses + extra doses)."""
        if shortlist.empty:
            return shortlist
        doses = sorted(set(self.cfg.search.doses) | set(self.cfg.selection.validation_extra_doses))
        vp = self._prompts("validation", self.cfg.prompts.n_validation)
        jobs = [Job("validation", p, COND_BASELINE, ()) for p in vp]
        ld: set[tuple[int, float]] = set()
        for r in shortlist.itertuples():
            for d in doses:
                iv = Intervention(int(r.layer), int(r.neuron), float(d), self.cfg.search.dose_unit)
                jobs += [Job("validation", p, COND_STEERED, (iv,)) for p in vp]
                ld.add((iv.layer, iv.dose))
        cp = vp[: self.cfg.controls.n_prompts] if self.cfg.controls.n_prompts else vp
        for iv in self.control_list(sorted(ld)):
            jobs += [Job("validation", p, COND_CONTROL, (iv,)) for p in cp]
        self.execute(jobs, "validation")
        self.score()
        table = rk.candidate_table(self.rows(), self.params, "validation")
        keep = {(int(r.layer), int(r.neuron)) for r in shortlist.itertuples()}
        return rk.leaderboard(table[[(l, n) in keep for l, n in zip(table.layer, table.neuron)]], self.params)

    def pick_finalists(self, validated: pd.DataFrame) -> list[dict[str, Any]]:
        """Finalists = best validation dose per neuron that is positive (and above controls if run)."""
        s = self.cfg.selection
        if validated.empty:
            return []
        v = validated[validated.mean_dj > 0]
        if s.require_control and v.control_percentile.notna().any():
            v = v[v.beats_control == True]  # noqa: E712
        out = []
        for r in v.head(s.finalists).itertuples():
            out.append({"neuron_id": r.neuron_id, "layer": int(r.layer), "neuron": int(r.neuron),
                        "dose": float(r.dose), "dose_unit": self.cfg.search.dose_unit,
                        "validation_mean_dj": float(r.mean_dj),
                        "validation_control_percentile": None if pd.isna(r.control_percentile) else float(r.control_percentile)})
        return out

    def freeze(self, finalists: list[dict[str, Any]]) -> dict[str, Any]:
        return cf.freeze_finalists(self.store, finalists, self.cfg.criteria, self.spec.spec_hash, self.scorer.scorer_id)

    def test(self, force: bool = False) -> dict[str, Any]:
        doc, contaminated = cf.check_test_allowed(self.store, force)
        fin = doc["finalists"]
        tp = self._prompts("test", self.cfg.prompts.n_test)
        jobs = [Job("test", p, COND_BASELINE, ()) for p in tp]
        ld: set[tuple[int, float]] = set()
        for f in fin:
            iv = Intervention(f["layer"], f["neuron"], f["dose"], f["dose_unit"])
            jobs += [Job("test", p, COND_STEERED, (iv,)) for p in tp]
            ld.add((iv.layer, iv.dose))
        cp = tp[: self.cfg.controls.n_prompts] if self.cfg.controls.n_prompts else tp
        for iv in self.control_list(sorted(ld)):
            jobs += [Job("test", p, COND_CONTROL, (iv,)) for p in cp]
        stats = self.execute(jobs, "test")
        self.score()
        self.store.write_json("test_done.json", {"at": time.time(), "contaminated": contaminated, **stats})
        return self.confirm(doc, contaminated)

    def confirm(self, doc: dict[str, Any] | None = None, contaminated: bool = False) -> dict[str, Any]:
        doc = doc or self.store.read_json("finalists.json")
        crit = cf.ConfirmationCriteria(**doc["criteria"])
        rows = self.rows()
        loose = rk.RankingParams(min_valid_prompts=1, min_coherence=0.0, min_control_seeds=self.params.min_control_seeds)
        table = rk.candidate_table(rows, loose, "test")
        cand_rows = rows[(rows.partition == "test") & (rows.condition == COND_STEERED) & (rows.n_iv == 1) & rows.ok & rows.dj.notna()]
        out = []
        for f in doc["finalists"]:
            g = cand_rows[(cand_rows.layer == f["layer"]) & (cand_rows.neuron == f["neuron"]) & (cand_rows.dose == f["dose"])]
            t = table[(table.layer == f["layer"]) & (table.neuron == f["neuron"]) & (table.dose == f["dose"])]
            res = cf.evaluate_candidate(g, t.iloc[0].to_dict() if len(t) else None, crit)
            out.append({"neuron_id": f["neuron_id"], "dose": f["dose"], **_py(res)})
        n_conf = sum(c["status"] == "confirmed" for c in out)
        result = {"contaminated": contaminated, "criteria": doc["criteria"], "finalists_sha": doc["sha"],
                  "n_finalists": len(out), "n_confirmed": n_conf, "candidates": out}
        self.store.write_json("confirmation.json", result)
        return result

    def run_all(self, force_test: bool = False) -> dict[str, Any]:
        lb = self.screen()
        short = self.shortlist()
        validated = self.validate(short)
        fin = self.pick_finalists(validated)
        summary = {"screening_rows": len(lb), "shortlist": len(short), "finalists": len(fin)}
        if fin:
            self.freeze(fin)
            summary["confirmation"] = self.test(force=force_test)
        self.store.write_json("run_summary.json", summary)
        return summary


def _py(o: Any) -> Any:
    if isinstance(o, dict):
        return {k: _py(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_py(v) for v in o]
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, float) and np.isnan(o):
        return None
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def _versions() -> dict[str, str]:
    out = {}
    for m in ("torch", "transformers", "numpy", "pandas"):
        try:
            out[m] = __import__(m).__version__
        except Exception:
            pass
    return out
