"""Backend interface. Everything else in Nudge talks to models only through this."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any, Sequence

from nudge.steering.interventions import Intervention
from nudge.utils.hashing import stable_hash


@dataclass(frozen=True)
class GenerationConfig:
    max_new_tokens: int = 64
    temperature: float = 0.0  # 0 => greedy decoding
    top_p: float = 1.0
    seed: int = 0
    use_chat_template: bool = False
    system_prompt: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def key(self) -> str:
        return stable_hash(self.to_dict(), 10)


@dataclass
class GenerationResult:
    text: str
    token_ids: list[int] = field(default_factory=list)
    n_prompt_tokens: int = 0
    latency_s: float = 0.0
    status: str = "ok"  # "ok" | "error"
    error: str | None = None
    extras: dict[str, Any] = field(default_factory=dict)


class SteeringBackend(ABC):
    """Narrow adapter interface (see spec section 4.D)."""

    name: str = "abstract"

    @abstractmethod
    def list_supported_layers(self) -> list[int]: ...

    @abstractmethod
    def list_neurons(self, layer_id: int) -> range: ...

    @abstractmethod
    def run_baseline(self, prompt: str, generation_config: GenerationConfig) -> GenerationResult: ...

    @abstractmethod
    def run_steered(self, prompt: str, intervention: "Intervention | Sequence[Intervention]",
                    generation_config: GenerationConfig) -> GenerationResult: ...

    @abstractmethod
    def get_model_metadata(self) -> dict[str, Any]: ...

    @abstractmethod
    def supported_dose_units(self) -> dict[str, str]:
        """Map ``dose_unit`` -> human-readable definition. Doses are only comparable within a unit."""

    @property
    def model_id(self) -> str:
        return str(self.get_model_metadata().get("model_id", "unknown"))
