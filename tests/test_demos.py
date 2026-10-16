import numpy as np

from cair.demos.fetch import make_fetch_expert
from cair.demos.minari import DATASETS


def _obs():
    raw = np.zeros(25, dtype=np.float64)
    raw[0:3] = [1.30, 0.75, 0.55]  # gripper
    raw[3:6] = [1.35, 0.75, 0.42]  # object
    raw[6:9] = raw[3:6] - raw[0:3]
    raw[9:11] = [0.04, 0.04]
    return {
        "observation": raw,
        "achieved_goal": raw[3:6].copy(),
        "desired_goal": np.array([1.50, 0.80, 0.45]),
    }


def test_fetch_experts_return_finite_actions():
    for name in ("push", "slide", "pick_and_place"):
        expert = make_fetch_expert(name)
        action = expert.act(_obs())
        assert action.shape == (4,)
        assert np.isfinite(action).all()


def test_minari_dataset_ids_are_explicit():
    assert DATASETS["ant"] == "mujoco/ant/simple-v0"
    assert DATASETS["swimmer"] == "mujoco/swimmer/medium-v0"
