"""Deprecated compatibility shim for EventDetector location.

EventDetector has moved to src/models/event_detector.py.
"""

from .event_detector import EventDetector

__all__ = ["EventDetector"]
