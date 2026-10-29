"""Experiment configuration (TOML or dict). Every field has a documented default."""
from __future__ import annotations

import sys
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

from nudge.experiments.confirm import ConfirmationCriteria
from nudge.models.base import GenerationConfig

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover
    import tomli as tomllib


@dataclass
class BackendConfig:
    kind: str = "tiny"            # "tiny" (random plumbing fixture) | "hf"
    model_id: str = ""            # HF id/path for kind="hf"
    device: str = "auto"
    dtype: str = "auto"
    min_layer: int = 0            # skip the first layers (relative doses are unreliable there)
    force_memory: bool = False


@dataclass
class SearchConfig:
    layers: list[int] | None = None
    survey_layers: int | None = None   # alternatively: k layers spread across depth
    neurons_per_layer: int = 8
    doses: list[float] = field(default_factory=lambda: [-0.7, 0.7])
    dose_unit: str = "rel_mlp_out_norm"


@dataclass
class ControlConfig:
    per_layer_dose: int = 8            # random-direction controls per (layer, dose)
    n_prompts: int | None = None       # controls run on the first n prompts of a partition (None = all)


@dataclass
class SelectionConfig:
    shortlist: int = 6
    finalists: int = 3
    min_valid_prompts: int = 5
    min_coherence: float = 0.0
    control_percentile: float = 0.95
    max_profile_cosine: float = 0.9
    require_control: bool = True       # drop candidates not above controls (when controls exist)
    validation_extra_doses: list[float] = field(default_factory=list)


@dataclass
class PromptConfig:
    suite: str = "fiction_scenes"
    n_screening: int | None = None
    n_validation: int | None = None
    n_test: int | None = None


@dataclass
class ScoringConfig:
    embedder: str = "hashing"          # "hashing" (offline baseline) or a sentence-transformers model name
    judge: str | None = None           # reserved: name of an LLM-judge preset


@dataclass
class ExperimentConfig:
    name: str = "demo"
    seed: int = 0
    run_dir: str = "runs"
    target: str = "secret_society_thriller"   # preset name or path to a spec .json
    backend: BackendConfig = field(default_factory=BackendConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    generation: GenerationConfig = field(default_factory=lambda: GenerationConfig(max_new_tokens=48))
    controls: ControlConfig = field(default_factory=ControlConfig)
    selection: SelectionConfig = field(default_factory=SelectionConfig)
    prompts: PromptConfig = field(default_factory=PromptConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    criteria: ConfirmationCriteria = field(default_factory=ConfirmationCriteria)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def run_path(self) -> Path:
        return Path(self.run_dir) / self.name


_SECTIONS = {"backend": BackendConfig, "search": SearchConfig, "generation": GenerationConfig,
             "controls": ControlConfig, "selection": SelectionConfig, "prompts": PromptConfig,
             "scoring": ScoringConfig, "criteria": ConfirmationCriteria}


def _check_keys(cls, d: dict[str, Any], where: str) -> None:
    unknown = set(d) - {f.name for f in fields(cls)}
    if unknown:
        raise ValueError(f"unknown key(s) in [{where}]: {sorted(unknown)}")


def config_from_dict(d: dict[str, Any]) -> ExperimentConfig:
    d = dict(d)
    top = d.pop("experiment", {})
    d = {**top, **d}
    kwargs: dict[str, Any] = {}
    for k, v in d.items():
        if k in _SECTIONS:
            _check_keys(_SECTIONS[k], v, k)
            kwargs[k] = _SECTIONS[k](**v)
        else:
            kwargs[k] = v
    _check_keys(ExperimentConfig, kwargs, "experiment")
    return ExperimentConfig(**kwargs)


def load_config(path: str | Path) -> ExperimentConfig:
    with open(path, "rb") as f:
        return config_from_dict(tomllib.load(f))
