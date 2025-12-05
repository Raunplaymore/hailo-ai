"""Deprecated compatibility shim for MobileNetV2 location.

MobileNetV2 has moved to src/models/backbone/mobilenet_v2.py.
"""

from .backbone.mobilenet_v2 import MobileNetV2

__all__ = ["MobileNetV2"]
