"""Load and resolve paper experiment configurations."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from .core import CAIRConfig

PAIRING_IL = {"bc", "dagger"}
PAIRING_RL = {"sac", "ddpg", "ppo"}


def _repo_config_path() -> Path:
    return Path(__file__).resolve().parents[2] / "configs" / "paper_main.yaml"


def load_paper_config(path: Optional[str] = None) -> Dict[str, Any]:
    if path is not None:
        p = Path(path)
        with p.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    p = _repo_config_path()
    if p.exists():
        with p.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    # Installed-wheel fallback compatible with Python 3.8.
    import importlib.resources as resources

    text = resources.read_text("cair.resources", "paper_main.yaml", encoding="utf-8")
    return yaml.safe_load(text)


def resolve_experiment(task: str, il: str, rl: str, path: Optional[str] = None) -> Dict[str, Any]:
    cfg = load_paper_config(path)
    il = il.lower()
    rl = rl.lower()
    if il not in PAIRING_IL:
        raise ValueError("Unknown IL method {!r}; choose bc or dagger.".format(il))
    if rl not in PAIRING_RL:
        raise ValueError("Unknown RL method {!r}; choose sac, ddpg, or ppo.".format(rl))
    if task not in cfg["tasks"]:
        raise KeyError("Unknown task {!r}. Available: {}".format(task, ", ".join(sorted(cfg["tasks"]))))

    t = deepcopy(cfg["tasks"][task])
    benchmark = t["benchmark"]
    result = {
        "task": task,
        "il": il,
        "rl": rl,
        "pairing": "{}+{}".format(il, rl),
        "benchmark": benchmark,
        "env_id": t["paper_env_id"],
        "episode_length": t["episode_length"],
        "train_steps": t["train_steps"],
        "demonstrations": t["demonstrations"],
        "rl_config": t["rl"][rl],
        "seeds": cfg["benchmark_defaults"][benchmark]["seeds"],
    }
    cair = deepcopy(cfg["cair"])
    cair["sync_interval"] = cfg["benchmark_defaults"][benchmark]["sync_interval"]
    result["cair"] = cair
    return result


def cair_config_from_experiment(exp: Dict[str, Any]) -> CAIRConfig:
    return CAIRConfig(**exp["cair"])
