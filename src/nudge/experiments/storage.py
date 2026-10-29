"""Append-only JSONL run store: one structured record per generation, scores kept separately.

Layout of a run directory::

    spec.json            frozen target specification (+ hash)
    config.json          experiment configuration
    meta.json            model metadata, environment, timings
    generations.jsonl    one record per generation (baseline / steered / control)   [append-only]
    scores.jsonl         one record per (generation, scorer_id)                       [append-only]
    finalists.json       frozen finalist list + preregistered criteria (before test)
    test_done.json       written when the held-out partition has been evaluated
    confirmation.json    confirmatory results
    report.md            exported report
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterator

from nudge.utils.hashing import stable_hash

COND_BASELINE, COND_STEERED, COND_CONTROL = "baseline", "steered", "control"


def generation_key(model_id: str, weights_fp: str | None, prompt: str, interventions: list[dict],
                   gen_config: dict) -> str:
    """Cache / resume identity. Same key <=> same experiment cell, so it is never regenerated."""
    iv = sorted(interventions, key=lambda d: json.dumps(d, sort_keys=True))
    return stable_hash({"m": model_id, "w": weights_fp, "p": prompt, "i": iv, "g": gen_config}, 20)


@dataclass
class GenerationRecord:
    key: str
    stage: str  # screening | validation | test
    partition: str
    prompt_id: str
    prompt: str
    condition: str  # baseline | steered | control
    interventions: list[dict[str, Any]]
    gen_config: dict[str, Any]
    seed: int
    output: str
    status: str = "ok"
    error: str | None = None
    latency_s: float = 0.0
    n_prompt_tokens: int = 0
    n_output_tokens: int = 0
    model_id: str = ""
    tokenizer_id: str = ""
    created_at: float = field(default_factory=time.time)
    prompt_meta: dict[str, Any] = field(default_factory=dict)

    # convenient views -------------------------------------------------------------------------
    @property
    def layer(self) -> int | None:
        return self.interventions[0]["layer"] if len(self.interventions) == 1 else None

    @property
    def neuron(self) -> int | None:
        return self.interventions[0]["neuron"] if len(self.interventions) == 1 else None

    @property
    def dose(self) -> float | None:
        return self.interventions[0]["dose"] if len(self.interventions) == 1 else None

    @property
    def control_seed(self) -> int | None:
        return self.interventions[0].get("random_seed") if len(self.interventions) == 1 else None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "GenerationRecord":
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})


@dataclass
class ScoreRecord:
    key: str  # generation key
    scorer_id: str
    s_target: float
    s_contrast: float
    j: float
    target_sims: list[float]
    contrast_sims: list[float]
    coherence: float
    relevance: float
    flags: list[str] = field(default_factory=list)
    behavior: dict[str, Any] = field(default_factory=dict)
    evaluator_raw: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ScoreRecord":
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})


def _read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    if not path.exists():
        return
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:  # truncated final line after a crash: skip, resume will redo it
                continue


class RunStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._gens: dict[str, GenerationRecord] = {}
        self._scores: dict[tuple[str, str], ScoreRecord] = {}
        for d in _read_jsonl(self.gen_path):
            r = GenerationRecord.from_dict(d)
            self._gens[r.key] = r
        for d in _read_jsonl(self.score_path):
            s = ScoreRecord.from_dict(d)
            self._scores[(s.key, s.scorer_id)] = s

    # paths ----------------------------------------------------------------------------------------
    @property
    def gen_path(self) -> Path:
        return self.root / "generations.jsonl"

    @property
    def score_path(self) -> Path:
        return self.root / "scores.jsonl"

    def path(self, name: str) -> Path:
        return self.root / name

    # generations ----------------------------------------------------------------------------------
    def has_generation(self, key: str, ok_only: bool = True) -> bool:
        r = self._gens.get(key)
        return r is not None and (r.status == "ok" or not ok_only)

    def get_generation(self, key: str) -> GenerationRecord | None:
        return self._gens.get(key)

    def add_generation(self, rec: GenerationRecord) -> None:
        with self.gen_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec.to_dict(), ensure_ascii=False) + "\n")
            f.flush()
        self._gens[rec.key] = rec  # a successful retry supersedes an earlier error

    def generations(self) -> list[GenerationRecord]:
        return list(self._gens.values())

    # scores ---------------------------------------------------------------------------------------
    def has_score(self, key: str, scorer_id: str) -> bool:
        return (key, scorer_id) in self._scores

    def get_score(self, key: str, scorer_id: str) -> ScoreRecord | None:
        return self._scores.get((key, scorer_id))

    def add_scores(self, recs: list[ScoreRecord]) -> None:
        with self.score_path.open("a", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")
            f.flush()
        for r in recs:
            self._scores[(r.key, r.scorer_id)] = r

    def scores(self, scorer_id: str) -> dict[str, ScoreRecord]:
        return {k: s for (k, sid), s in self._scores.items() if sid == scorer_id}

    def scorer_ids(self) -> list[str]:
        seen: dict[str, float] = {}
        for (_, sid), s in self._scores.items():
            seen[sid] = max(seen.get(sid, 0.0), s.created_at)
        return [k for k, _ in sorted(seen.items(), key=lambda kv: kv[1])]

    # small json documents -------------------------------------------------------------------------
    def write_json(self, name: str, obj: Any) -> None:
        tmp = self.path(name + ".tmp")
        tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        tmp.replace(self.path(name))

    def read_json(self, name: str, default: Any = None) -> Any:
        p = self.path(name)
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else default

    def exists(self, name: str) -> bool:
        return self.path(name).exists()

    def summary(self) -> dict[str, Any]:
        gens = self.generations()
        by: dict[str, int] = {}
        for g in gens:
            by[g.condition] = by.get(g.condition, 0) + 1
        return {"generations": len(gens), "errors": sum(g.status != "ok" for g in gens), "by_condition": by}
