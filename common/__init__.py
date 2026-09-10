"""Shared, GUI-free helpers used by every module of the app.

* :mod:`common.naming` - the ``plot(N)`` / ``aug(N)`` filename conventions that tie
  augmented copies back to their original plot (and keep ML splits leakage-free).
* :mod:`common.las_io` - reading, copying, subsetting and writing whole ``.las/.laz``
  clouds while preserving *every* point dimension (needed by augmentation).
* :mod:`common.session` - the small state object the three modules hand off to each
  other.
"""

from .naming import (
    LAS_SUFFIXES,
    aug_number_from_name,
    build_aug_name,
    build_copy_name,
    plot_number_from_name,
)
from .session import Session

__all__ = [
    "LAS_SUFFIXES",
    "plot_number_from_name",
    "aug_number_from_name",
    "build_aug_name",
    "build_copy_name",
    "Session",
]
