"""Conflict-Averse IL-RL (CAIR)."""
from .config import cair_config_from_experiment, load_paper_config, resolve_experiment
from .core import CAIRConfig, CAIRController, cagrad_direction, cosine_similarity, normalize_gradient
from .losses import imitation_mse
from .update import cair_actor_step

__all__ = [
    "CAIRConfig",
    "CAIRController",
    "cagrad_direction",
    "cosine_similarity",
    "normalize_gradient",
    "cair_actor_step",
    "imitation_mse",
    "load_paper_config",
    "resolve_experiment",
    "cair_config_from_experiment",
]
