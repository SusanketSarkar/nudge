"""Forward hooks that apply and *always* remove steering.

Two equivalent realisations of "push neuron n of an MLP by alpha" are provided so that the backend can
verify its own wiring (``steering.validation``):

* ``additive_mlp_out``: add ``alpha * W_down[:, n]`` to the output of ``down_proj`` (what Nudge uses).
* ``neuron_activation``: add ``alpha`` to channel ``n`` of ``down_proj``'s *input* (the neuron's
  activation h_n). For a bias-free linear ``down_proj`` the two are mathematically identical.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator, Mapping, Sequence

import torch


def count_forward_hooks(model: torch.nn.Module) -> int:
    return sum(len(m._forward_hooks) + len(m._forward_pre_hooks) for m in model.modules())


@contextmanager
def additive_mlp_out(down_projs: Mapping[int, torch.nn.Module],
                     deltas: Mapping[int, torch.Tensor]) -> Iterator[None]:
    """Add ``deltas[layer]`` (shape ``[hidden]``) to every position of each layer's MLP output."""
    handles = []
    try:
        for layer, delta in deltas.items():
            module = down_projs[layer]

            def hook(_m, _args, output, _delta=delta):
                return output + _delta.to(device=output.device, dtype=output.dtype)

            handles.append(module.register_forward_hook(hook))
        yield
    finally:
        for h in handles:
            h.remove()


@contextmanager
def neuron_activation(down_projs: Mapping[int, torch.nn.Module],
                      pushes: Mapping[int, Sequence[tuple[int, float]]]) -> Iterator[None]:
    """Add ``alpha`` to channel ``n`` of ``down_proj``'s input for each ``(n, alpha)``."""
    handles = []
    try:
        for layer, items in pushes.items():
            module = down_projs[layer]

            def pre_hook(_m, args, _items=tuple(items)):
                x = args[0].clone()
                for n, alpha in _items:
                    x[..., n] = x[..., n] + alpha
                return (x, *args[1:])

            handles.append(module.register_forward_pre_hook(pre_hook))
        yield
    finally:
        for h in handles:
            h.remove()
