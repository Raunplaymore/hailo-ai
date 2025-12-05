"""Common path helpers for local and Colab environments."""

from __future__ import annotations

import os
from pathlib import Path


def _is_colab() -> bool:
    """Return True when running inside Google Colab."""
    if "COLAB_GPU" in os.environ:
        return True
    try:
        import google.colab  # type: ignore
    except ImportError:
        return False
    else:
        return True

def get_data_root() -> Path:
    """
    Determine the base data directory.

    Priority order:
    1) DATA_ROOT environment variable (manual override)
    2) Colab default: /content/hailo-ai/data
    3) Local default: <project_root>/data
    """
    # 1) User override
    env_override = os.environ.get("DATA_ROOT")
    if env_override:
        return Path(env_override).expanduser().resolve()

    # 2) Colab default path
    if _is_colab():
        return Path("/content/hailo-ai/data").resolve()

    # 3) Local project data folder
    project_root = Path(__file__).resolve().parents[2]
    return (project_root / "data").resolve()


def get_data_path(*subpaths: str) -> Path:
    """
    Join one or more subpaths under the data root.
    Examples:
        get_data_path("golf_db")
            -> <data_root>/golf_db

        get_data_path("splits", "train_split_1.pkl")
            -> <data_root>/splits/train_split_1.pkl

        get_data_path("yolo", "images", "train")
            -> <data_root>/yolo/images/train
    """
    root = get_data_root()
    for sp in subpaths:
        root = root / sp
    return root.resolve()


__all__ = ["get_data_root", "get_data_path"]