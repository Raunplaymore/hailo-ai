"""Deprecated compatibility shim.

The GolfDB dataset was moved to src/dataset/golfdb_dataset.py.
Import from there instead:
    from src.dataset.golfdb_dataset import GolfDB, ToTensor, Normalize
"""

from .golfdb_dataset import GolfDB, Normalize, ToTensor

__all__ = ["GolfDB", "ToTensor", "Normalize"]
       

