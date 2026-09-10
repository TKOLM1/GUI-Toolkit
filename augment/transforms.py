"""The augmentation transforms and their registry.

This mirrors the feature registry in :mod:`featuregen.features`: a single ordered
list :data:`AUGMENTATIONS` is the source of truth, so the GUI builds its controls and
the runner applies transforms from the *same* definitions and they can never drift.

Each :class:`AugmentDef` separates two steps so the run is reproducible and the
manifest can record exactly what happened:

* ``sample(ranges, rng)`` draws the concrete parameters for one application (e.g. a
  rotation angle, a noise sigma) from the user's ``min..max`` ranges. The returned dict
  is what gets written to the manifest.
* ``apply(las, params, rng, height_channel)`` mutates a *copy* of the cloud in place
  using those parameters (plus ``rng`` for the per-point noise / index draws).

Every transform preserves all point dimensions: geometric ops only rewrite the
affected coordinate arrays, and dropout/bootstrap reindex the whole packed point
record so RGB, classification and ``RelativeHeight`` ride along.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import laspy
import numpy as np

# A range the GUI exposes for one parameter: (min, max).
Range = tuple[float, float]


@dataclass(frozen=True)
class AugParam:
    """One tunable parameter of a transform, with the default range shown in the GUI."""

    name: str
    label: str
    default_min: float
    default_max: float
    kind: str = "float"  # "float" | "int"
    tooltip: str = ""


@dataclass(frozen=True)
class AugmentDef:
    """One augmentation method (registry entry)."""

    key: str
    label: str
    group: str
    tooltip: str
    params: tuple[AugParam, ...]
    sample: Callable[[dict[str, Range], np.random.Generator], dict]
    apply: Callable[[laspy.LasData, dict, np.random.Generator, str], None]


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #

def _uniform(ranges: dict[str, Range], name: str, rng: np.random.Generator) -> float:
    """Draw one value uniformly from the (possibly user-overridden) range of ``name``."""
    lo, hi = ranges[name]
    return float(rng.uniform(lo, hi))


def _n(las: laspy.LasData) -> int:
    return int(len(las.points))


# --------------------------------------------------------------------------- #
# Rotation about the vertical axis                                            #
# --------------------------------------------------------------------------- #

def _sample_rotation(ranges, rng):
    return {"angle_deg": _uniform(ranges, "angle_deg", rng)}


def _apply_rotation(las, params, rng, height_channel):
    if _n(las) < 1:
        return
    ang = np.deg2rad(params["angle_deg"])
    ca, sa = np.cos(ang), np.sin(ang)
    x = np.asarray(las.x, dtype=np.float64)
    y = np.asarray(las.y, dtype=np.float64)
    cx, cy = x.mean(), y.mean()
    dx, dy = x - cx, y - cy
    las.x = cx + dx * ca - dy * sa
    las.y = cy + dx * sa + dy * ca


# --------------------------------------------------------------------------- #
# Horizontal flip (mirror across the centroid along a random axis)            #
# --------------------------------------------------------------------------- #

def _sample_flip(ranges, rng):
    return {"axis": "x" if rng.integers(0, 2) == 0 else "y"}


def _apply_flip(las, params, rng, height_channel):
    if _n(las) < 1:
        return
    if params["axis"] == "x":
        x = np.asarray(las.x, dtype=np.float64)
        las.x = 2.0 * x.mean() - x
    else:
        y = np.asarray(las.y, dtype=np.float64)
        las.y = 2.0 * y.mean() - y


# --------------------------------------------------------------------------- #
# x/y jitter                                                                  #
# --------------------------------------------------------------------------- #

def _sample_jitter_xy(ranges, rng):
    return {"sigma_m": _uniform(ranges, "sigma_m", rng)}


def _apply_jitter_xy(las, params, rng, height_channel):
    n = _n(las)
    if n < 1:
        return
    sigma = params["sigma_m"]
    las.x = np.asarray(las.x, dtype=np.float64) + rng.normal(0.0, sigma, n)
    las.y = np.asarray(las.y, dtype=np.float64) + rng.normal(0.0, sigma, n)


# --------------------------------------------------------------------------- #
# z jitter - one shared delta added to BOTH z and the height channel          #
# --------------------------------------------------------------------------- #

def _sample_jitter_z(ranges, rng):
    return {"sigma_m": _uniform(ranges, "sigma_m", rng)}


def _apply_jitter_z(las, params, rng, height_channel):
    n = _n(las)
    if n < 1:
        return
    delta = rng.normal(0.0, params["sigma_m"], n)
    las.z = np.asarray(las.z, dtype=np.float64) + delta
    # Shift the height-above-ground channel by the *same* delta so z and RelativeHeight
    # never diverge. Skip silently if the file carries no such channel.
    if height_channel in set(las.point_format.dimension_names):
        las[height_channel] = np.asarray(las[height_channel], dtype=np.float64) + delta


# --------------------------------------------------------------------------- #
# Point dropout (drop a fraction of all points, uniformly)                    #
# --------------------------------------------------------------------------- #

def _sample_dropout(ranges, rng):
    return {"fraction": _uniform(ranges, "fraction", rng)}


def _apply_dropout(las, params, rng, height_channel):
    n = _n(las)
    if n < 1:
        return
    keep = max(1, int(round(n * (1.0 - params["fraction"]))))
    idx = rng.choice(n, size=keep, replace=False)
    idx.sort()  # keep original order for tidiness
    las.points = las.points[idx].copy()


# --------------------------------------------------------------------------- #
# Bootstrap resampling (resample points with replacement, uniformly)          #
# --------------------------------------------------------------------------- #

def _sample_bootstrap(ranges, rng):
    return {"n_fraction": _uniform(ranges, "n_fraction", rng)}


def _apply_bootstrap(las, params, rng, height_channel):
    n = _n(las)
    if n < 1:
        return
    take = max(1, int(round(n * params["n_fraction"])))
    idx = rng.integers(0, n, size=take)
    las.points = las.points[idx].copy()


# --------------------------------------------------------------------------- #
# The registry - order here is the order transforms compose in a plan          #
# --------------------------------------------------------------------------- #

AUGMENTATIONS: list[AugmentDef] = [
    AugmentDef(
        "rotation_z", "Rotation (vertical axis)", "Geometric",
        "Rotate the (x, y) of every point about the plot centroid by a random angle. "
        "Heights (z, RelativeHeight) and all other dimensions are unchanged.",
        (AugParam("angle_deg", "Angle (deg)", 0.0, 360.0, "float",
                  "Rotation angle drawn uniformly from this range, in degrees."),),
        _sample_rotation, _apply_rotation),
    AugmentDef(
        "flip_h", "Horizontal flip", "Geometric",
        "Mirror the (x, y) of every point across the plot centroid along a randomly "
        "chosen horizontal axis (x or y). Heights are unchanged.",
        (),  # no continuous parameter; the axis is chosen at random per sample
        _sample_flip, _apply_flip),
    AugmentDef(
        "jitter_xy", "x/y jitter", "Noise",
        "Add independent Gaussian noise to each point's x and y. The standard "
        "deviation (metres) is drawn from this range, then applied per point.",
        (AugParam("sigma_m", "Sigma (m)", 0.0, 0.02, "float",
                  "Per-point Gaussian noise std for x and y, in metres."),),
        _sample_jitter_xy, _apply_jitter_xy),
    AugmentDef(
        "jitter_z", "z jitter", "Noise",
        "Add Gaussian noise to height: one shared per-point delta is added to BOTH "
        "the raw z and the height channel (RelativeHeight) so they stay consistent. The "
        "std (metres) is drawn from this range. This is the main height augmentation "
        "that actually changes the computed height features.",
        (AugParam("sigma_m", "Sigma (m)", 0.0, 0.02, "float",
                  "Per-point Gaussian noise std added to z and RelativeHeight together."),),
        _sample_jitter_z, _apply_jitter_z),
    AugmentDef(
        "dropout", "Point dropout", "Sampling",
        "Randomly drop a fraction of ALL points (uniformly across classes). The drop "
        "fraction is drawn from this range. Changes point density, PPR and counts.",
        (AugParam("fraction", "Drop fraction", 0.0, 0.3, "float",
                  "Fraction of points removed, drawn uniformly from this range (0..1)."),),
        _sample_dropout, _apply_dropout),
    AugmentDef(
        "bootstrap", "Bootstrap resample", "Sampling",
        "Resample points with replacement (uniformly across all points). The number "
        "kept is this fraction of the original count (1.0 = same size).",
        (AugParam("n_fraction", "Size fraction", 1.0, 1.0, "float",
                  "Resampled point count as a fraction of the original, drawn from this range."),),
        _sample_bootstrap, _apply_bootstrap),
]

AUG_BY_KEY: dict[str, AugmentDef] = {a.key: a for a in AUGMENTATIONS}
