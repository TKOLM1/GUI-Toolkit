"""Reading a single .las/.laz plot file into the arrays the metrics need.

The whole tool only ever needs two point sets per file:

* **VEG** - the off-ground / vegetation points (``classification == veg_code``).
  Their ``RelativeHeight`` values are the "heights" feeding nearly every metric, their
  ``(x, y)`` feed the Sigma_Z plane fit, and their RGB feed the colour means.
* **ALL** - every point in the file. Used only by the Pulse Penetration Ratio,
  which compares ground hits against the total.

:class:`PlotData` extracts and caches both sets once, so each metric function in
``features.py`` is a clean one-liner over ready-made arrays.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import laspy
import numpy as np


@dataclass(frozen=True)
class Config:
    """User-tunable parameters, surfaced in the GUI's "Advanced" panel.

    Nothing about the point semantics is hard-coded: if a future export renames
    the height channel or shuffles the class codes, only these values change.
    """

    height_channel: str = "RelativeHeight"  # point dimension holding height-above-ground
    veg_code: int = 1                   # classification value for off-ground / vegetation
    ground_code: int = 2                # classification value for ground
    entropy_bin_size: float = 0.1       # vertical bin width (metres) for Shannon entropy
    # The colour channels, so a file that spells them differently (or carries none at all) is
    # handled by configuration rather than by hardcoded "red"/"green"/"blue" lookups. An empty
    # name means "this colour is not available" and the colour features come out NaN.
    red_channel: str = "red"
    green_channel: str = "green"
    blue_channel: str = "blue"
    slope_strata_pct: float = 5.0       # top/bottom height % for the directional slope-angle features


@dataclass
class PlotData:
    """Pre-extracted arrays for one plot file.

    ``h``, ``x``, ``y`` and the RGB arrays are already filtered to the vegetation
    points; ``classification`` is kept for the whole cloud so PPR can be computed.
    Arrays may be empty (no vegetation points) and the RGB arrays may be ``None``
    (file carries no colour) - the metric functions handle both via NaN guards.
    """

    name: str                       # source file name (becomes the features .csv row label)
    h: np.ndarray                   # VEG RelativeHeight values (float64)
    x: np.ndarray                   # VEG scaled x (float64)
    y: np.ndarray                   # VEG scaled y (float64)
    red: np.ndarray | None          # VEG red channel (raw stored values) or None
    green: np.ndarray | None        # VEG green channel or None
    blue: np.ndarray | None         # VEG blue channel or None
    classification: np.ndarray      # classification of ALL points (uint)
    n_all: int                      # total point count in the file
    config: Config = field(default_factory=Config)

    @property
    def n(self) -> int:
        """Number of vegetation points."""
        return int(self.h.size)


def _has_dimension(las: "laspy.LasData", name: str) -> bool:
    """True if ``name`` is a point dimension of this file (case-sensitive)."""
    return name in set(las.point_format.dimension_names)


# The three coordinate axes, as listed in the point-format dimension names (and so as
# offered in the GUI's height-channel dropdown). laspy is case-sensitive: ``las["Z"]``
# returns the *raw stored integer*, while ``las.z`` applies the header scale+offset and
# returns metres. So when the chosen height channel is one of these axes we must read
# the scaled lowercase attribute - otherwise picking "Z" yields height in raw integer
# units (e.g. metres * 1/scale), which is the 1000x-too-large bug this guards against.
_AXIS_CHANNELS = {"X": "x", "Y": "y", "Z": "z"}


def read_channel(las: "laspy.LasData", name: str) -> np.ndarray:
    """Return point dimension ``name`` as float64, scaled if it is an X/Y/Z axis.

    For the coordinate axes (``X``/``Y``/``Z``, as the dimension-name list spells them)
    this reads the scaled ``las.x``/``.y``/``.z`` so a height channel set to e.g. ``Z``
    comes back in metres, not raw integers. Any other (extra) dimension is returned
    verbatim, since those carry their real values directly.
    """
    attr = _AXIS_CHANNELS.get(name)
    source = getattr(las, attr) if attr is not None else las[name]
    return np.asarray(source, dtype=np.float64)


def list_dimensions(path: str | Path) -> list[str]:
    """The point-dimension names of ``path``, read from the header only (no points loaded).

    Used by the Feature tab to populate the "Height channel" dropdown with the channels a plot
    actually carries (e.g. ``RelativeHeight``), so the user picks a real dimension instead of typing
    one. ``laspy.open`` reads just the header, so this stays cheap even on large clouds.
    """
    with laspy.open(Path(path)) as reader:
        return list(reader.header.point_format.dimension_names)


def load_plot(path: str | Path, config: Config | None = None) -> PlotData:
    """Read ``path`` and return a :class:`PlotData`.

    Raises
    ------
    KeyError
        If the configured height channel is absent from the file - the caller
        (pipeline) catches this and records the file as failed so a single bad
        file never aborts a batch.
    """
    config = config or Config()
    path = Path(path)
    las = laspy.read(path)

    if not _has_dimension(las, config.height_channel):
        available = ", ".join(las.point_format.dimension_names)
        raise KeyError(
            f"Height channel '{config.height_channel}' not found in {path.name}. "
            f"Available dimensions: {available}"
        )

    classification = np.asarray(las.classification)
    veg_mask = classification == config.veg_code

    # Height-above-ground for the vegetation points (the value every height metric uses).
    # read_channel scales the axis channels, so a height channel of e.g. "Z" returns metres.
    height = read_channel(las, config.height_channel)
    h = height[veg_mask]

    # Scaled coordinates for the Sigma_Z plane fit.
    x = np.asarray(las.x, dtype=np.float64)[veg_mask]
    y = np.asarray(las.y, dtype=np.float64)[veg_mask]

    # RGB is optional (only colour-bearing point formats have it) and its channel names are
    # configurable, so each is read independently: an unset or absent channel simply stays None
    # and the colour metrics built on it return NaN.
    def _colour(name: str):
        if not name or not _has_dimension(las, name):
            return None
        return read_channel(las, name)[veg_mask]

    red = _colour(config.red_channel)
    green = _colour(config.green_channel)
    blue = _colour(config.blue_channel)

    return PlotData(
        name=path.name,
        h=h,
        x=x,
        y=y,
        red=red,
        green=green,
        blue=blue,
        classification=classification,
        n_all=int(classification.size),
        config=config,
    )
