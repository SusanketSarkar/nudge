"""Placeholder adapter for the Neuron Explorer sweep stack (vLLM steering fork).

Not implemented, on purpose. The upstream repository currently ships no license, and its fast path needs
a CUDA pod plus a vLLM fork, neither of which is available in the default environment. If you want
it, implement :class:`nudge.models.base.SteeringBackend` here (``run_steered`` -> a request carrying
``steering_vectors={"mlp_out": {layer: {"vector": W_down[:, n], "scale": dose * R}}}``) and verify it
against :class:`HFBackend` with ``nudge doctor`` before trusting any result. See ``NOTICE.md``.
"""
from __future__ import annotations

from nudge.models.base import SteeringBackend


class ExplorerAdapter(SteeringBackend):  # pragma: no cover - documented stub
    name = "explorer"

    def __init__(self, *args, **kwargs) -> None:
        raise NotImplementedError(
            "ExplorerAdapter is a documented stub. Use HFBackend, or implement this adapter "
            "(see the module docstring and NOTICE.md)."
        )

    def list_supported_layers(self): ...
    def list_neurons(self, layer_id): ...
    def run_baseline(self, prompt, generation_config): ...
    def run_steered(self, prompt, intervention, generation_config): ...
    def get_model_metadata(self): ...
    def supported_dose_units(self): ...
