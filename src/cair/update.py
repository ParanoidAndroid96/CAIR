"""Algorithm-agnostic CAIR actor-step helper."""
from __future__ import annotations

from typing import Dict, Iterable, Optional, Tuple

import torch

from .core import CAIRController
from .gradients import assign_flat_gradient, task_gradients


def cair_actor_step(
    il_loss: torch.Tensor,
    rl_loss: torch.Tensor,
    parameters: Iterable[torch.nn.Parameter],
    optimizer: torch.optim.Optimizer,
    controller: CAIRController,
    episode: int,
    zero_grad: bool = True,
) -> Tuple[torch.Tensor, Dict[str, object]]:
    """Apply one CAIR policy-gradient update.

    The RL learner remains responsible for constructing ``rl_loss`` (SAC,
    DDPG, PPO, etc.) and ``il_loss`` (BC or DAgger MSE).  CAIR only combines
    the two policy gradients.
    """
    params = [p for p in parameters if p.requires_grad]
    if zero_grad:
        optimizer.zero_grad()
    g_il, g_rl = task_gradients([il_loss, rl_loss], params)
    direction, metrics = controller.policy_direction(g_il, g_rl, episode)
    assign_flat_gradient(params, direction)
    optimizer.step()
    return direction, metrics
