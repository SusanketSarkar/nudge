"""Confirmation criteria, finalist freezing and the held-out guard.

The held-out (test) partition may be evaluated **once**, and only after the finalist list and the
success criteria have been written to ``finalists.json``. A forced re-run is allowed but permanently
marks the results ``contaminated`` in the report.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from nudge.experiments.storage import RunStore
from nudge.scoring.controls import bootstrap_ci
from nudge.utils.hashing import stable_hash


class HeldOutError(RuntimeError):
    pass


@dataclass
class ConfirmationCriteria:
    min_prompts: int = 10
    ci_level: float = 0.95
    min_success_rate: float = 0.6        # fraction of prompts with ΔJ > 0
    max_coherence_drop: float = 0.15     # mean Δcoherence must be >= -this
    min_control_percentile: float = 0.95
    max_answerable_drop: float = 0.10    # behavioural guardrail (only if answerability data exist)
    bootstrap_seed: int = 0


def freeze_finalists(store: RunStore, finalists: list[dict[str, Any]], criteria: ConfirmationCriteria,
                     spec_hash: str, scorer_id: str) -> dict[str, Any]:
    if not finalists:
        raise HeldOutError("cannot freeze an empty finalist list")
    doc = {"frozen_at": time.time(), "spec_hash": spec_hash, "scorer_id": scorer_id,
           "criteria": asdict(criteria), "finalists": finalists}
    doc["sha"] = stable_hash({k: doc[k] for k in ("spec_hash", "scorer_id", "criteria", "finalists")}, 16)
    store.write_json("finalists.json", doc)
    return doc


def check_test_allowed(store: RunStore, force: bool = False) -> tuple[dict[str, Any], bool]:
    """Return (frozen finalists doc, contaminated?). Raises if not frozen or already evaluated."""
    doc = store.read_json("finalists.json")
    if doc is None:
        raise HeldOutError("Freeze the finalist list (and criteria) before touching the held-out test set.")
    contaminated = False
    if store.exists("test_done.json"):
        if not force:
            raise HeldOutError("The held-out test set was already evaluated for this run. Re-running it would "
                               "turn it into a tuning set; start a new run or pass force=True (results will be "
                               "marked contaminated).")
        contaminated = True
    return doc, contaminated


def evaluate_candidate(g: pd.DataFrame, row: dict[str, Any] | None, crit: ConfirmationCriteria) -> dict[str, Any]:
    """Apply preregistered criteria to one finalist's held-out rows ``g`` (ok, with dj)."""
    reasons: list[str] = []
    n = len(g)
    lo, hi = bootstrap_ci(g.dj.to_numpy(), crit.ci_level, seed=crit.bootstrap_seed) if n else (np.nan, np.nan)
    mean = float(g.dj.mean()) if n else np.nan
    succ = float((g.dj > 0).mean()) if n else np.nan
    dcoh = float(g.d_coherence.mean()) if n else np.nan
    pct = (row or {}).get("control_percentile", np.nan)
    res = {"n_prompts": n, "mean_dj": mean, "ci_low": lo, "ci_high": hi, "success_rate": succ,
           "d_coherence": dcoh, "control_percentile": pct, "p_control": (row or {}).get("p_control", np.nan)}
    if "answerable" in g and g.answerable.notna().any() and g.d_behavior.notna().any():
        for flag, name in ((True, "answerable"), (False, "unanswerable")):
            sub = g[(g.answerable == flag) & g.d_behavior.notna()]
            res[f"behavior_delta_{name}"] = float(sub.d_behavior.mean()) if len(sub) else np.nan
    if n < crit.min_prompts:
        reasons.append(f"only {n} valid held-out prompts (< {crit.min_prompts})")
        status = "inconclusive"
    else:
        status = "confirmed"
        if not (np.isfinite(lo) and lo > 0):
            status = "not_confirmed"; reasons.append(f"CI for mean ΔJ includes 0 ({lo:+.4f}, {hi:+.4f})")
        if succ < crit.min_success_rate:
            status = "not_confirmed"; reasons.append(f"success rate {succ:.2f} < {crit.min_success_rate}")
        if dcoh < -crit.max_coherence_drop:
            status = "not_confirmed"; reasons.append(f"coherence dropped by {-dcoh:.2f} (> {crit.max_coherence_drop})")
        if not np.isfinite(pct):
            status = "inconclusive" if status == "confirmed" else status
            reasons.append("no matched random controls available")
        elif pct < crit.min_control_percentile:
            status = "not_confirmed"; reasons.append(
                f"effect only at the {pct:.2f} percentile of random controls (< {crit.min_control_percentile})")
        drop = res.get("behavior_delta_answerable")
        if drop is not None and np.isfinite(drop) and drop < -crit.max_answerable_drop:
            status = "not_confirmed"; reasons.append(f"answerable-question success dropped by {-drop:.2f}")
    res.update(status=status, reasons=reasons)
    return res
