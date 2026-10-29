"""Build backend / spec / suite / runner from an :class:`ExperimentConfig`."""
from __future__ import annotations

from pathlib import Path

from nudge.experiments.config import ExperimentConfig
from nudge.experiments.prompts import PromptSuite, load_suite
from nudge.experiments.runner import ExperimentRunner
from nudge.experiments.storage import RunStore
from nudge.models.base import SteeringBackend
from nudge.targets.schema import TargetSpec
from nudge.targets.target_builder import get_preset


def build_backend(cfg: ExperimentConfig) -> SteeringBackend:
    b = cfg.backend
    if b.kind == "tiny":
        from nudge.models.tiny import build_tiny_backend

        return build_tiny_backend(seed=cfg.seed, min_layer=b.min_layer)
    if b.kind == "hf":
        from nudge.models.hf_backend import HFBackend

        if not b.model_id:
            raise ValueError("backend.model_id is required for kind='hf'")
        return HFBackend.from_pretrained(b.model_id, device=b.device, dtype=b.dtype,
                                         force=b.force_memory, min_layer=b.min_layer)
    raise ValueError(f"unknown backend kind {b.kind!r} (expected 'tiny' or 'hf')")


def load_spec(target: str) -> TargetSpec:
    return TargetSpec.load(target) if Path(target).suffix == ".json" else get_preset(target)


def build_suite(cfg: ExperimentConfig) -> PromptSuite:
    return load_suite(cfg.prompts.suite)


def build_runner(cfg: ExperimentConfig, backend: SteeringBackend | None = None, spec: TargetSpec | None = None,
                 **kw) -> ExperimentRunner:
    return ExperimentRunner(backend or build_backend(cfg), RunStore(cfg.run_path), spec or load_spec(cfg.target),
                            build_suite(cfg), cfg, **kw)
