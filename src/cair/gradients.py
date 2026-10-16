"""Utilities for extracting and applying flat policy gradients."""
from __future__ import annotations

from typing import Iterable, List, Sequence

import torch


def _parameters(parameters: Iterable[torch.nn.Parameter]) -> List[torch.nn.Parameter]:
    return [p for p in parameters if p.requires_grad]


def task_gradients(
    losses: Sequence[torch.Tensor],
    parameters: Iterable[torch.nn.Parameter],
    allow_unused: bool = True,
) -> List[torch.Tensor]:
    """Return one flat gradient vector for each loss.

    Parameters unused by a particular loss contribute zeros.  This is useful
    for policy modules whose actor/value heads share part of a network.
    """
    params = _parameters(parameters)
    result: List[torch.Tensor] = []
    for i, loss in enumerate(losses):
        grads = torch.autograd.grad(
            loss,
            params,
            retain_graph=i < len(losses) - 1,
            allow_unused=allow_unused,
        )
        filled = [torch.zeros_like(p) if g is None else g for p, g in zip(params, grads)]
        result.append(torch.cat([g.contiguous().reshape(-1) for g in filled], dim=0))
    return result


def assign_flat_gradient(parameters: Iterable[torch.nn.Parameter], flat_grad: torch.Tensor) -> None:
    """Assign a flat gradient vector to ``Parameter.grad`` fields."""
    params = _parameters(parameters)
    offset = 0
    for p in params:
        n = p.numel()
        if offset + n > flat_grad.numel():
            raise ValueError("Flat gradient is shorter than the parameter vector.")
        p.grad = flat_grad[offset : offset + n].view_as(p).to(device=p.device, dtype=p.dtype).clone()
        offset += n
    if offset != flat_grad.numel():
        raise ValueError(
            "Flat gradient has {} entries but parameters consume {}.".format(flat_grad.numel(), offset)
        )
