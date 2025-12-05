"""Model package exports."""

from .event_detector import EventDetector
from .backbone.mobilenet_v2 import MobileNetV2

__all__ = ["EventDetector", "MobileNetV2"]
