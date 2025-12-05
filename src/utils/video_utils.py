"""Video-related utilities for training and evaluation."""

from __future__ import annotations

import numpy as np


class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def correct_preds(probs, labels, tol: int = -1):
    """
    Compute event correctness with tolerance.

    Args:
        probs: (sequence_length, 9) probabilities or logits.
        labels: (sequence_length,) ground truth labels.
        tol: tolerance in frames; if -1, derive from address-to-impact span.
    Returns:
        events, preds, deltas, tol, correct
    """
    events = np.where(labels < 8)[0]
    preds = np.zeros(len(events))
    if tol == -1:
        tol = int(max(np.round((events[5] - events[0]) / 30), 1))
    for i in range(len(events)):
        preds[i] = np.argsort(probs[:, i])[-1]
    deltas = np.abs(events - preds)
    correct = (deltas <= tol).astype(np.uint8)
    return events, preds, deltas, tol, correct


def freeze_layers(num_freeze: int, net) -> None:
    """Freeze first num_freeze layers of a backbone network."""
    i = 1
    for child in net.children():
        if i == 1:
            j = 1
            for child_child in child.children():
                if j <= num_freeze:
                    for param in child_child.parameters():
                        param.requires_grad = False
                j += 1
        i += 1


__all__ = ["AverageMeter", "correct_preds", "freeze_layers"]
