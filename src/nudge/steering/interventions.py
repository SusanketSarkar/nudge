"""Intervention specification.

An :class:`Intervention` is a *declarative* description of one steering push; a backend decides how to
realise it. It is deliberately explicit about the dose unit, because doses are only comparable within
one backend + unit (see ``docs/METHODOLOGY.md``).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable, Sequence

from nudge.utils.hashing import stable_hash

HOOK_MLP_OUT = "mlp_out"
KIND_NEURON = "neuron"
KIND_RANDOM = "random"


@dataclass(frozen=True)
class Intervention:
    layer: int
    neuron: int
    dose: float
    dose_unit: str = "rel_mlp_out_norm"
    kind: str = KIND_NEURON  # "neuron" | "random" (matched-magnitude random-direction control)
    random_seed: int | None = None
    hook_point: str = HOOK_MLP_OUT

    def __post_init__(self) -> None:
        if self.kind not in (KIND_NEURON, KIND_RANDOM):
            raise ValueError(f"unknown intervention kind {self.kind!r}")
        if self.kind == KIND_RANDOM and self.random_seed is None:
            raise ValueError("random-direction interventions need a random_seed")
        if self.kind == KIND_NEURON and self.random_seed is not None:
            raise ValueError("neuron interventions must not carry a random_seed")
        if self.layer < 0 or self.neuron < 0:
            raise ValueError("layer and neuron must be non-negative")

    @property
    def id(self) -> str:
        base = f"L{self.layer}.N{self.neuron}@{self.dose:+g}"
        return base if self.kind == KIND_NEURON else f"{base}~rand{self.random_seed}"

    @property
    def neuron_id(self) -> str:
        return f"L{self.layer}.N{self.neuron}"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Intervention":
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})

    def with_dose(self, dose: float) -> "Intervention":
        return Intervention(self.layer, self.neuron, dose, self.dose_unit, self.kind,
                            self.random_seed, self.hook_point)

    def as_control(self, random_seed: int) -> "Intervention":
        """Random-direction control with the same layer, dose, unit and magnitude."""
        return Intervention(self.layer, self.neuron, self.dose, self.dose_unit, KIND_RANDOM,
                            random_seed, self.hook_point)


def as_tuple(interventions: "Intervention | Iterable[Intervention] | None") -> tuple[Intervention, ...]:
    if interventions is None:
        return ()
    if isinstance(interventions, Intervention):
        return (interventions,)
    return tuple(interventions)


def interventions_key(interventions: Sequence[Intervention]) -> str:
    """Order-independent identity of a (possibly multi-neuron) intervention."""
    if not interventions:
        return "baseline"
    return stable_hash(sorted((i.to_dict() for i in interventions), key=lambda d: sorted(d.items(), key=str)), 12)
