"""Deprecated compatibility shim for utility functions.

The utilities were moved to src/utils/video_utils.py.
Import from there instead:
    from src.utils.video_utils import AverageMeter, correct_preds, freeze_layers
"""

from .video_utils import AverageMeter, correct_preds, freeze_layers

__all__ = ["AverageMeter", "correct_preds", "freeze_layers"]
