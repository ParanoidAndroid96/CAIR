"""Small imitation-loss helpers shared by BC and DAgger integrations."""
from __future__ import annotations

import torch


def imitation_mse(policy_actions: torch.Tensor, expert_actions: torch.Tensor) -> torch.Tensor:
    """Mean-squared action imitation loss used by the supplied implementations."""
    return torch.nn.functional.mse_loss(policy_actions, expert_actions)
