import math

import torch

from cair.core import (
    CAIRConfig,
    CAIRController,
    cagrad_direction,
    cosine_similarity,
    normalize_gradient,
)


def test_normalize_gradient_and_zero_vector():
    g = torch.tensor([3.0, 4.0])
    out = normalize_gradient(g)
    assert torch.allclose(out.norm(), torch.tensor(1.0))
    z = normalize_gradient(torch.zeros(3))
    assert torch.equal(z, torch.zeros(3))
    assert cosine_similarity(z, torch.ones(3)).item() == 0.0


def test_cagrad_returns_finite_direction():
    gil = normalize_gradient(torch.tensor([1.0, 0.0]))
    grl = normalize_gradient(torch.tensor([-0.5, 1.0]))
    d, metrics = cagrad_direction(gil, grl, lam=0.5, c=0.7)
    assert d.shape == gil.shape
    assert torch.isfinite(d).all()
    assert math.isclose(metrics["cagrad_weight_il"] + metrics["cagrad_weight_rl"], 1.0, rel_tol=1e-5)


def test_controller_phases_and_lambda_bounds():
    cfg = CAIRConfig(warmup_episodes=2, sync_interval=1, lambda_lr=1e-2)
    ctl = CAIRController(cfg)
    gil = torch.tensor([2.0, 0.0])
    grl = torch.tensor([-1.0, 1.0])

    d0, m0 = ctl.policy_direction(gil, grl, episode=0)
    assert m0["phase"] == "il"
    assert torch.equal(d0, gil)

    # At/after warmup the shadow weight adapts even while applied lambda is 1.
    ctl.policy_direction(gil, grl, episode=2)
    ctl.end_training_call(2)
    assert 0.0 <= ctl.lambda_applied <= 1.0
    assert 0.0 <= ctl.lambda_opt.item() <= 1.0


def test_post_completion_uses_raw_rl_gradient():
    ctl = CAIRController(CAIRConfig(lambda_floor=0.05))
    ctl.lambda_applied = 0.04
    gil = torch.tensor([10.0, 0.0])
    grl = torch.tensor([0.0, 3.0])
    d, metrics = ctl.policy_direction(gil, grl, episode=100)
    assert metrics["phase"] == "rl"
    assert torch.equal(d, grl)
    assert ctl.lambda_applied == 0.0


def test_shadow_penalty_reacts_to_negative_il_alignment():
    ctl = CAIRController(CAIRConfig(lambda_lr=5e-3, beta=0.0))
    before = ctl.lambda_opt.item()
    # RL points directly against IL; the constraint penalty should be active
    # once the mixture crosses into negative IL alignment.
    ctl.lambda_opt.data.fill_(0.2)
    metrics = ctl.shadow_update(torch.tensor([1.0, 0.0]), torch.tensor([-1.0, 0.0]))
    assert metrics["cos_il_shadow"] < 0.0
    assert 0.0 <= ctl.lambda_opt.item() <= 1.0
    assert ctl.lambda_opt.item() >= 0.2 - 1e-6
