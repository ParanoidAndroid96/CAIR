"""Tiny CAIR smoke test with conflicting synthetic gradients.

Run:
    python examples/synthetic_demo.py
"""
import torch

from cair.core import CAIRConfig, CAIRController, cosine_similarity


def main():
    controller = CAIRController(
        CAIRConfig(warmup_episodes=0, sync_interval=1, lambda_lr=1e-3)
    )
    controller.lambda_applied = 0.5

    g_il = torch.tensor([1.0, 0.0])
    g_rl = torch.tensor([-0.4, 1.0])

    direction, metrics = controller.policy_direction(g_il, g_rl, episode=1)
    controller.end_training_call(1)

    print("phase:", metrics["phase"])
    print("lambda_applied:", metrics["lambda_applied"])
    print("lambda_opt:", metrics["lambda_opt"])
    print("cos(direction, IL):", float(cosine_similarity(direction, g_il)))
    print("cos(direction, RL):", float(cosine_similarity(direction, g_rl)))
    print("direction:", direction.tolist())


if __name__ == "__main__":
    main()
