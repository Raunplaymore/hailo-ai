"""Common path helpers for local and Colab environments."""

from __future__ import annotations

import os
import sys
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

    Priority:
    1) DATA_ROOT environment variable
    2) Colab default: /content/hailo-ai/data
    3) Local default: <project_root>/data
    """
    env_override = os.environ.get("DATA_ROOT")
    if env_override:
        return Path(env_override).expanduser().resolve()

    if _is_colab():
        return Path("/content/hailo-ai/data")

    project_root = Path(__file__).resolve().parents[2]
    return project_root / "data"


def get_data_path(subdir: str) -> Path:
    """
    Get a path inside the data directory.

    Example:
        get_data_path("raw") -> <data_root>/raw
    """
    return get_data_root() / subdir


__all__ = ["get_data_root", "get_data_path"]
