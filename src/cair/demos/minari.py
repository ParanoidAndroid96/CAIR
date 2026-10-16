"""Minari resources used by the locomotion experiments."""
from __future__ import annotations

from typing import Dict


DATASETS: Dict[str, str] = {
    "halfcheetah": "mujoco/halfcheetah/simple-v0",
    "hopper": "mujoco/hopper/simple-v0",
    "walker2d": "mujoco/walker2d/simple-v0",
    "ant": "mujoco/ant/simple-v0",
    "swimmer": "mujoco/swimmer/medium-v0",
    "humanoid": "mujoco/humanoid/simple-v0",
}


def load_dataset(task: str, download: bool = True):
    """Load the paper's Minari dataset lazily."""
    try:
        import minari
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("Install the 'paper' extras to use Minari datasets.") from exc
    key = task.lower()
    if key not in DATASETS:
        raise KeyError("Unknown locomotion task: {}".format(task))
    return minari.load_dataset(DATASETS[key], download=download)
