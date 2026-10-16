#!/usr/bin/env python3
"""Fast installation smoke test: config + core CAIR update."""
import torch

from cair.config import resolve_experiment, cair_config_from_experiment
from cair.core import CAIRController


def main():
    exp = resolve_experiment("ant", "dagger", "sac")
    ctl = CAIRController(cair_config_from_experiment(exp))
    ctl.lambda_applied = 0.5
    d, metrics = ctl.policy_direction(
        torch.tensor([1.0, 0.0]), torch.tensor([-0.2, 1.0]), episode=50
    )
    assert torch.isfinite(d).all()
    assert metrics["phase"] == "mixed"
    print("CAIR smoke test passed")
    print("resolved pairing:", exp["pairing"], exp["env_id"])
    print("direction:", d.tolist())


if __name__ == "__main__":
    main()
