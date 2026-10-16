"""Core CAIR update logic.

This module implements the method-level update described in the camera-ready
paper, independent of SAC/DDPG/PPO data collection or critic/value updates.

Conventions
-----------
``g_il`` and ``g_rl`` are gradients of losses to be *minimized*.  The vector
returned by :meth:`CAIRController.policy_direction` is therefore assigned to
``.grad`` and passed to the policy optimizer in the usual way.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F


@dataclass
class CAIRConfig:
    """Method hyperparameters from the final paper specification.

    ``sync_interval`` differs by benchmark family: 10 for manipulation and 5
    for locomotion.  The config resolver applies that domain-specific value.
    """

    cagrad_c: float = 0.7
    lambda_lr: float = 1e-4
    beta: float = 0.9
    mu_init: float = 1.0
    mu_lr: float = 1.0
    lambda_floor: float = 0.05
    warmup_episodes: int = 40
    sync_interval: int = 5
    cagrad_steps: int = 20
    cagrad_lr: float = 2.0
    cagrad_momentum: float = 0.5
    eps: float = 1e-8


def _flat(x: torch.Tensor) -> torch.Tensor:
    return x.reshape(-1)


def normalize_gradient(g: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Return a unit-norm copy of ``g`` when nonzero, else the zero vector."""
    g = _flat(g)
    n = g.norm()
    if float(n.detach().item()) <= eps:
        return torch.zeros_like(g)
    return g / n


def cosine_similarity(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Cosine similarity with the paper's zero-vector convention.

    The camera-ready appendix assigns zero similarity when the mixed direction
    is exactly zero.  We use the same behavior for any zero-norm argument.
    """
    a = _flat(a)
    b = _flat(b)
    if float(a.norm().detach().item()) <= eps or float(b.norm().detach().item()) <= eps:
        return torch.zeros((), dtype=torch.promote_types(a.dtype, b.dtype), device=a.device)
    return F.cosine_similarity(a, b.to(a.device), dim=0, eps=eps)


def mixed_gradient(g_il: torch.Tensor, g_rl: torch.Tensor, lam: torch.Tensor) -> torch.Tensor:
    return lam * g_il + (1.0 - lam) * g_rl


def cagrad_direction(
    g_il: torch.Tensor,
    g_rl: torch.Tensor,
    lam: float,
    c: float = 0.7,
    steps: int = 20,
    lr: float = 2.0,
    momentum: float = 0.5,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Finite two-task CAGrad solve used by CAIR.

    Inputs are expected to be normalized IL/RL gradients.  We minimize the
    simplex dual

        <g_w, v_lambda> + c ||v_lambda|| ||g_w||,

    starting from equal logits, take ``steps`` SGD steps, retain the best
    evaluated iterate, and reconstruct

        d = v_lambda + c ||v_lambda|| g_w / ||g_w||.

    This matches the algorithmic description in the camera-ready appendix.
    """
    gil = _flat(g_il)
    grl = _flat(g_rl).to(gil.device)
    dtype = torch.promote_types(gil.dtype, grl.dtype)
    grads = torch.stack([gil.to(dtype), grl.to(dtype)], dim=0)

    lam_t = torch.tensor(float(lam), device=grads.device, dtype=dtype)
    v = lam_t * grads[0] + (1.0 - lam_t) * grads[1]
    v_norm = v.norm()
    radius = float(c) * v_norm

    # With v_lambda == 0 the feasible ball collapses to the origin.
    if float(v_norm.detach().item()) <= eps or float(c) <= 0.0:
        return v.clone(), {
            "cagrad_weight_il": 0.5,
            "cagrad_weight_rl": 0.5,
            "cagrad_radius": float(radius.detach().item()),
        }

    logits = torch.zeros(2, 1, device=grads.device, dtype=dtype, requires_grad=True)
    optimizer = torch.optim.SGD([logits], lr=lr, momentum=momentum)

    best_logits = logits.detach().clone()
    best_obj = float("inf")

    # Evaluate the initial iterate plus every iterate after each of ``steps``
    # SGD updates.  This is equivalent to the historical 21-evaluation loop
    # with 20 optimizer steps.
    for i in range(steps + 1):
        weights = torch.softmax(logits, dim=0)
        gw = weights.t().mm(grads).reshape(-1)
        obj = torch.dot(gw, v) + radius * gw.norm()
        value = float(obj.detach().item())
        if value < best_obj:
            best_obj = value
            best_logits = logits.detach().clone()
        if i == steps:
            break
        optimizer.zero_grad()
        obj.backward()
        optimizer.step()

    weights = torch.softmax(best_logits, dim=0)
    gw = weights.t().mm(grads).reshape(-1)
    gw_norm = gw.norm()
    if float(gw_norm.detach().item()) <= eps:
        direction = v.clone()
    else:
        direction = v + radius * gw / gw_norm

    return direction, {
        "cagrad_weight_il": float(weights[0].detach().item()),
        "cagrad_weight_rl": float(weights[1].detach().item()),
        "cagrad_radius": float(radius.detach().item()),
    }


class CAIRController:
    """Stateful adaptive annealing + CAGrad controller.

    The controller separates *policy-gradient updates* from *learner training
    calls*.  Call :meth:`policy_direction` once per policy-gradient update and
    :meth:`end_training_call` once after the learner's training call.  This
    mirrors Algorithm 1 in the paper: the shadow weight adapts each policy
    update, while the applied weight is refreshed periodically by episode.
    """

    def __init__(self, config: Optional[CAIRConfig] = None, device: Optional[torch.device] = None):
        self.config = config or CAIRConfig()
        self.device = torch.device("cpu") if device is None else torch.device(device)
        self.dtype = torch.float64

        self.lambda_opt = torch.tensor(
            1.0, dtype=self.dtype, device=self.device, requires_grad=True
        )
        self.lambda_optimizer = torch.optim.Adam([self.lambda_opt], lr=self.config.lambda_lr)
        self.lambda_applied = 1.0

        self.mu = torch.tensor(self.config.mu_init, dtype=self.dtype, device=self.device)
        self.s_il = torch.tensor(1.0, dtype=self.dtype, device=self.device)
        self.s_rl = torch.tensor(0.0, dtype=self.dtype, device=self.device)

    @property
    def completed(self) -> bool:
        return self.lambda_applied == 0.0

    def shadow_update(self, g_il: torch.Tensor, g_rl: torch.Tensor) -> Dict[str, float]:
        """One persistent OptimizeLambda update (Algorithm 2)."""
        gil_raw = _flat(g_il).detach().to(device=self.device, dtype=self.dtype)
        grl_raw = _flat(g_rl).detach().to(device=self.device, dtype=self.dtype)

        if float(gil_raw.norm().item()) <= self.config.eps or float(grl_raw.norm().item()) <= self.config.eps:
            return {
                "shadow_update_skipped": 1.0,
                "lambda_opt": float(self.lambda_opt.detach().item()),
                "mu": float(self.mu.detach().item()),
                "s_il": float(self.s_il.detach().item()),
                "s_rl": float(self.s_rl.detach().item()),
            }

        gil = normalize_gradient(gil_raw, self.config.eps)
        grl = normalize_gradient(grl_raw, self.config.eps)
        # Loss gradients are held fixed during the lambda update.
        gil = gil.detach()
        grl = grl.detach()

        v = self.lambda_opt * gil + (1.0 - self.lambda_opt) * grl
        cos_il = cosine_similarity(gil, v, self.config.eps)
        cos_rl = cosine_similarity(grl, v, self.config.eps)

        beta = self.config.beta
        # stop-gradient through previous histories, but preserve the current
        # cosine's dependence on lambda_opt.
        s_il_new = beta * self.s_il.detach() + (1.0 - beta) * cos_il
        s_rl_new = beta * self.s_rl.detach() + (1.0 - beta) * cos_rl
        objective = -s_rl_new + self.mu.detach() * torch.relu(-s_il_new)

        self.lambda_optimizer.zero_grad()
        objective.backward()
        self.lambda_optimizer.step()

        with torch.no_grad():
            self.s_il = s_il_new.detach()
            self.s_rl = s_rl_new.detach()
            self.mu = torch.clamp(self.mu - self.config.mu_lr * self.s_il, min=0.0)
            self.lambda_opt.clamp_(0.0, 1.0)

        return {
            "shadow_update_skipped": 0.0,
            "lambda_opt": float(self.lambda_opt.detach().item()),
            "mu": float(self.mu.detach().item()),
            "s_il": float(self.s_il.detach().item()),
            "s_rl": float(self.s_rl.detach().item()),
            "cos_il_shadow": float(cos_il.detach().item()),
            "cos_rl_shadow": float(cos_rl.detach().item()),
            "shadow_objective": float(objective.detach().item()),
        }

    def policy_direction(
        self,
        g_il: torch.Tensor,
        g_rl: torch.Tensor,
        episode: int,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Return the gradient supplied to the policy optimizer.

        Pure IL and post-completion RL phases use *unnormalized* gradients.
        Only the mixed phase uses normalized gradients and CAGrad, matching the
        final paper's Algorithm 1.
        """
        gil_raw = _flat(g_il)
        grl_raw = _flat(g_rl).to(gil_raw.device)

        if self.lambda_applied <= self.config.lambda_floor:
            self.lambda_applied = 0.0
            return grl_raw.clone(), {
                "phase": "rl",
                "lambda_applied": 0.0,
                "lambda_opt": float(self.lambda_opt.detach().item()),
            }

        shadow_metrics: Dict[str, float] = {}

        if episode < self.config.warmup_episodes or self.lambda_applied == 1.0:
            direction = gil_raw.clone()
            phase = "il"
        else:
            gil = normalize_gradient(gil_raw, self.config.eps)
            grl = normalize_gradient(grl_raw, self.config.eps)
            direction, cagrad_metrics = cagrad_direction(
                gil,
                grl,
                lam=self.lambda_applied,
                c=self.config.cagrad_c,
                steps=self.config.cagrad_steps,
                lr=self.config.cagrad_lr,
                momentum=self.config.cagrad_momentum,
                eps=self.config.eps,
            )
            shadow_metrics.update(cagrad_metrics)
            shadow_metrics.update({
                "cos_il_direction": float(cosine_similarity(gil, direction, self.config.eps).detach().item()),
                "cos_rl_direction": float(cosine_similarity(grl, direction, self.config.eps).detach().item()),
            })
            phase = "mixed"

        # The shadow weight begins adapting at W even while lambda_applied may
        # still be exactly one.  Post-completion RL bypasses this branch above.
        if episode >= self.config.warmup_episodes:
            shadow_metrics.update(self.shadow_update(gil_raw, grl_raw))

        shadow_metrics.update({
            "phase": phase,
            "lambda_applied": float(self.lambda_applied),
            "lambda_opt": float(self.lambda_opt.detach().item()),
        })
        return direction, shadow_metrics

    def end_training_call(self, episode: int) -> float:
        """Refresh the applied lambda after a learner training call when due."""
        if (
            self.lambda_applied > 0.0
            and episode >= self.config.warmup_episodes
            and episode % self.config.sync_interval == 0
        ):
            self.lambda_applied = float(self.lambda_opt.detach().item())
        return self.lambda_applied

    def state_dict(self) -> Dict[str, object]:
        return {
            "config": asdict(self.config),
            "lambda_opt": float(self.lambda_opt.detach().item()),
            "lambda_applied": float(self.lambda_applied),
            "mu": float(self.mu.detach().item()),
            "s_il": float(self.s_il.detach().item()),
            "s_rl": float(self.s_rl.detach().item()),
            "lambda_optimizer": self.lambda_optimizer.state_dict(),
        }

    def load_state_dict(self, state: Dict[str, object]) -> None:
        with torch.no_grad():
            self.lambda_opt.fill_(float(state["lambda_opt"]))
            self.lambda_applied = float(state["lambda_applied"])
            self.mu.fill_(float(state["mu"]))
            self.s_il.fill_(float(state["s_il"]))
            self.s_rl.fill_(float(state["s_rl"]))
        if "lambda_optimizer" in state:
            self.lambda_optimizer.load_state_dict(state["lambda_optimizer"])  # type: ignore[arg-type]
