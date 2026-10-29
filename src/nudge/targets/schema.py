"""Structured, versioned target specification (spec section 4.A)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from nudge.utils.hashing import stable_hash

DEFAULT_DIMENSIONS = ["target_semantic_shift", "coherence", "relevance"]


class TargetSpec(BaseModel):
    target_name: str = Field(min_length=1, pattern=r"^[A-Za-z0-9_\-]+$")
    goal: str = ""
    target_descriptions: list[str] = Field(min_length=1)
    contrast_descriptions: list[str] = Field(min_length=1)
    task_type: Literal["behavioral", "stylistic"] = "stylistic"
    guardrails: list[str] = Field(default_factory=list)
    evaluation_dimensions: list[str] = Field(default_factory=lambda: list(DEFAULT_DIMENSIONS))
    behavior_evaluator: str | None = None  # name in nudge.scoring.behavioral.EVALUATORS
    reviewed: bool = False  # a human has inspected/edited the descriptions
    origin: str = "manual"  # manual | preset | llm_proposed | edited

    @field_validator("target_descriptions", "contrast_descriptions")
    @classmethod
    def _non_empty_strings(cls, v: list[str]) -> list[str]:
        v = [" ".join(s.split()) for s in v]
        if any(not s for s in v):
            raise ValueError("descriptions must be non-empty strings")
        return v

    @property
    def spec_hash(self) -> str:
        """Content hash of everything that affects scoring; changes whenever the spec is edited."""
        return stable_hash({"t": self.target_descriptions, "c": self.contrast_descriptions,
                            "e": self.behavior_evaluator, "n": self.target_name}, 12)

    def lint(self) -> list[str]:
        w: list[str] = []
        if len(self.target_descriptions) < 2 or len(self.contrast_descriptions) < 2:
            w.append("Use at least 2 target and 2 contrast descriptions so the score is not tied to one phrasing.")
        if set(map(str.lower, self.target_descriptions)) & set(map(str.lower, self.contrast_descriptions)):
            w.append("A description appears in both target and contrast sets.")
        if not self.reviewed:
            w.append("Descriptions have not been reviewed by a human; review before trusting scores.")
        if self.task_type == "stylistic" and any("accura" in g.lower() or "factual" in g.lower() for g in self.guardrails):
            w.append("Style experiments should be kept distinct from factual-accuracy experiments.")
        return w

    def lint_overlap(self, embedder, threshold: float = 0.85) -> list[str]:
        """Warn when target and contrast descriptions are semantically near-identical for the embedder."""
        import numpy as np

        t = embedder.encode(self.target_descriptions)
        c = embedder.encode(self.contrast_descriptions)
        sim = float(np.mean(t @ c.T))
        return ([f"Target/contrast descriptions are very similar under this embedder (mean cos {sim:.2f}); "
                 "J = S_T - S_C will have little dynamic range."] if sim > threshold else [])

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({**self.model_dump(), "spec_hash": self.spec_hash}, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "TargetSpec":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        d.pop("spec_hash", None)
        return cls(**d)
