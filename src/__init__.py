"""LoRetta source package.

Future training, datasets, losses, metrics, and evaluation packages should be
added as peers of ``config``, ``inference``, and ``loretta``.
"""

from .loretta import LoRetta

__all__ = ["LoRetta"]
