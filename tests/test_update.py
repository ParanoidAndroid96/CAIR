import torch

from cair.core import CAIRConfig, CAIRController
from cair.update import cair_actor_step


def test_actor_step_updates_parameters():
    torch.manual_seed(0)
    model = torch.nn.Linear(2, 1, bias=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    ctl = CAIRController(CAIRConfig(warmup_episodes=0, sync_interval=1))
    ctl.lambda_applied = 0.5

    x = torch.tensor([[1.0, 2.0], [-1.0, 1.0]])
    y_il = torch.tensor([[1.0], [0.0]])
    y_rl = torch.tensor([[0.0], [1.0]])
    pred = model(x)
    il_loss = torch.nn.functional.mse_loss(pred, y_il)
    rl_loss = torch.nn.functional.mse_loss(pred, y_rl)
    before = model.weight.detach().clone()

    _, metrics = cair_actor_step(
        il_loss, rl_loss, model.parameters(), optimizer, ctl, episode=1
    )
    assert metrics["phase"] == "mixed"
    assert not torch.equal(before, model.weight.detach())
