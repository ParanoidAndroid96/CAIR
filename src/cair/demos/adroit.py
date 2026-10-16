"""Small loaders for the historical Adroit demonstration formats.

The release does not redistribute the benchmark demonstration/expert binaries.
Supply files obtained from the Rajeswaran/D4RL benchmark at runtime.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict

import numpy as np


def load_hammer_hdf5(path: str) -> Dict[str, np.ndarray]:
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover
        raise ImportError("h5py is required for Hammer HDF5 demonstrations.") from exc
    with h5py.File(path, "r") as f:
        required = ["observations", "actions", "rewards", "terminals"]
        missing = [k for k in required if k not in f]
        if missing:
            raise KeyError("Hammer HDF5 is missing keys: {}".format(missing))
        return {k: np.asarray(f[k]) for k in required}


def load_pen_demos(path: str) -> Dict[str, np.ndarray]:
    with Path(path).open("rb") as f:
        demos = pickle.load(f)
    # Historical code used demos[0] and the standard DAPG keys below.
    item = demos[0] if isinstance(demos, (list, tuple)) else demos
    required = ["observations", "actions", "rewards"]
    missing = [k for k in required if k not in item]
    if missing:
        raise KeyError("Pen demonstration pickle is missing keys: {}".format(missing))
    return {k: np.asarray(item[k]) for k in required}


def load_pen_expert(path: str):
    """Load the legacy DAPG Pen expert object (must provide get_action)."""
    with Path(path).open("rb") as f:
        expert = pickle.load(f)
    if not hasattr(expert, "get_action"):
        raise TypeError("Expected a DAPG-style expert with get_action(state).")
    return expert
