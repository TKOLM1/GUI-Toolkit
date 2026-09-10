"""Vegetation / structure metrics and the feature registry.

Every metric is a small pure function of a :class:`~featuregen.io_las.PlotData`.
They are collected in :data:`FEATURES`, an ordered registry that is the **single
source of truth** for the whole tool: the GUI builds its checkboxes (labels +
tooltips) from it, and the pipeline computes columns from it, so the two can never
drift. Adding a feature later is just appending one :class:`FeatureDef`.

Metric definitions follow the LiDAR_Vegetation_Metrics reference notebook
(``4_compute_vegetation_metrics.ipynb``), collapsed to one value per file. Two
deliberate corrections to the reference are made and flagged inline:

* ``h_cv`` uses the sample std (``ddof=1``) to match ``h_std`` (the reference
  accidentally used the population std there).
* the entropy bin floor is extended below zero so noisy slightly-negative heights
  are binned rather than silently dropped.

Every feature carries two grouping labels: a broad ``cls`` ("Height", "Color" or
"2D") and a finer ``group`` type within it. The GUI nests its checkboxes class ->
type and makes both levels togglable; the docs use the same two-level structure.

Groups A-F follow that reference: the height metrics (A-E) form the **Height** class
and the mean RGB (F) the **Color** class. The **2D** class (group
"Horizontal (hand-crafted)") does **not** follow the reference: those features were
hand-crafted for this project to describe horizontal canopy structure in the plot's
own PCA axis frame, and are flagged as such inline.

Degenerate inputs (too few points, zero variance, non-positive mean, missing RGB)
return ``NaN`` so a column is always numeric and never raises.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import floor, nan
from typing import Callable

import numpy as np
from scipy.stats import kurtosis, skew

from . import geometry as geo
from .geometry import fit_plane
from .io_las import PlotData

# Type of a metric function: maps one plot's data to a single scalar.
FeatureFunc = Callable[[PlotData], float]


@dataclass(frozen=True)
class FeatureDef:
    """One selectable output column.

    Features are organised two levels deep for the GUI and the docs: a small set of
    broad **classes** (``cls`` — "Height", "Color", "2D"), each split into finer
    **types** (``group`` — e.g. "Central tendency", "Shape"). The GUI nests its
    checkboxes class -> type, and both levels are togglable.

    Attributes
    ----------
    key:     exact column name written to the features .csv.
    label:   human-readable checkbox text in the GUI.
    cls:     broad feature class ("Height" / "Color" / "2D") — the outer GUI grouping.
    group:   finer type within the class (e.g. "Spread") — the inner GUI grouping.
    tooltip: precise description of exactly how the value is computed (shown on
             hover in the GUI and attached as the Excel header cell comment).
    func:    the metric function.
    """

    key: str
    label: str
    cls: str
    group: str
    tooltip: str
    func: FeatureFunc


# --------------------------------------------------------------------------- #
# Group A - central tendency (where the heights sit)                          #
# --------------------------------------------------------------------------- #

def h_mean(p: PlotData) -> float:
    """Arithmetic mean of vegetation heights."""
    return float(np.mean(p.h)) if p.n >= 1 else nan


def h_median(p: PlotData) -> float:
    """Median (50th percentile) of vegetation heights - the robust centre."""
    return float(np.median(p.h)) if p.n >= 1 else nan


def h_max(p: PlotData) -> float:
    """Tallest vegetation height - canopy-top proxy (fragile to a single outlier)."""
    return float(np.max(p.h)) if p.n >= 1 else nan


def h_p25(p: PlotData) -> float:
    """25th height percentile (linear interpolation) - lower canopy / understory."""
    return float(np.percentile(p.h, 25)) if p.n >= 1 else nan


def h_p75(p: PlotData) -> float:
    """75th height percentile (linear interpolation) - upper canopy."""
    return float(np.percentile(p.h, 75)) if p.n >= 1 else nan


def h_p95(p: PlotData) -> float:
    """95th height percentile - robust canopy top (top 5% of returns trimmed)."""
    return float(np.percentile(p.h, 95)) if p.n >= 1 else nan


# --------------------------------------------------------------------------- #
# Group B - spread (how variable the heights are)                             #
# --------------------------------------------------------------------------- #

def h_var(p: PlotData) -> float:
    """Sample variance (ddof=1) of vegetation heights, in m**2."""
    return float(np.var(p.h, ddof=1)) if p.n >= 2 else nan


def h_std(p: PlotData) -> float:
    """Sample standard deviation (ddof=1) of vegetation heights, in m."""
    return float(np.std(p.h, ddof=1)) if p.n >= 2 else nan


def h_cv(p: PlotData) -> float:
    """Coefficient of variation: sample std (ddof=1) / mean.

    Unitless relative spread, comparable across tall and short stands. Uses the
    sample std to stay consistent with ``h_std`` (the reference used the
    population std here, an inconsistency this tool corrects). NaN when the mean
    is not strictly positive.
    """
    if p.n < 2:
        return nan
    mean = float(np.mean(p.h))
    if mean <= 0:
        return nan
    return float(np.std(p.h, ddof=1)) / mean


# --------------------------------------------------------------------------- #
# Group C - shape (higher moments of the height distribution)                 #
# --------------------------------------------------------------------------- #

def h_skew(p: PlotData) -> float:
    """Skewness (scipy default biased g1) of vegetation heights.

    Positive: mostly low returns with a thin tall tail. Negative: mostly high
    returns with a tail trailing down. Needs >= 3 points and non-zero variance.
    """
    if p.n < 3 or np.var(p.h) == 0:
        return nan
    return float(skew(p.h))


def h_kurtosis(p: PlotData) -> float:
    """Excess kurtosis (Fisher, scipy default biased) of vegetation heights.

    0 for a normal distribution; positive = peaked/heavy-tailed (one dominant
    layer with occasional far returns); negative = flat/broad. Needs >= 4 points
    and non-zero variance.
    """
    if p.n < 4 or np.var(p.h) == 0:
        return nan
    return float(kurtosis(p.h, fisher=True))


# --------------------------------------------------------------------------- #
# Group D - structure (layering and surface roughness)                        #
# --------------------------------------------------------------------------- #

def entropy(p: PlotData) -> float:
    """Shannon entropy (bits) of the vertical height distribution.

    Heights are sliced into fixed-width vertical bins (default 0.1 m); each bin's
    share of points is a proportion and entropy is ``-sum(prop * log2(prop))``
    over non-empty bins. Near 0 = single layer; high = many occupied layers. The
    bin grid is floored below zero so noisy slightly-negative heights are kept.
    """
    if p.n < 1:
        return nan
    bin_size = p.config.entropy_bin_size
    lo = floor(min(0.0, float(p.h.min())))
    hi = float(p.h.max())
    bins = np.arange(lo, hi + bin_size, bin_size)
    hist, _ = np.histogram(p.h, bins=bins)
    total = hist.sum()
    if total == 0:
        return nan
    proportions = hist[hist > 0] / total
    return float(-(proportions * np.log2(proportions)).sum())


def sigma_z(p: PlotData) -> float:
    """Roughness: RMS of residuals after fitting a plane z = a*x + b*y + c.

    Fitting a plane first removes any systematic tilt (a uniformly sloped plot is
    not "rough"); what remains is genuine bumpiness around the local trend. Uses
    population normalisation (ddof=0) by design - this is an RMS roughness
    descriptor, not an estimator of population spread. Needs >= 3 points.
    """
    if p.n < 3:
        return nan
    # The plane fit lives in featuregen.geometry so the viewer can draw the *same* plane.
    _, residuals = fit_plane(p.x, p.y, p.h)
    return float(np.std(residuals, ddof=0))


# --------------------------------------------------------------------------- #
# Group E - density and penetration                                           #
# --------------------------------------------------------------------------- #

def frac_above_mean(p: PlotData) -> float:
    """Fraction of vegetation points above the vegetation mean height.

    A proportion in [0, 1], not a raw count: dividing by the point total removes the
    drone-driven point-density variation that contaminated the old count, leaving a
    pure shape descriptor (what *share* of returns sit in the upper canopy).
    """
    if p.n < 1:
        return nan
    return float(np.sum(p.h > np.mean(p.h))) / p.n


def ppr(p: PlotData) -> float:
    """Pulse Penetration Ratio: ground points / total points (all classes).

    The only metric using the whole cloud. High = many pulses reached the ground
    (sparse/open canopy); low = the canopy intercepted nearly everything (dense,
    closed). Always in [0, 1].
    """
    if p.n_all < 1:
        return nan
    ground = int(np.sum(p.classification == p.config.ground_code))
    return ground / p.n_all


# --------------------------------------------------------------------------- #
# Group F - colour (mean RGB over vegetation points)                          #
# --------------------------------------------------------------------------- #

def _channel_mean(channel: np.ndarray | None, n: int) -> float:
    if channel is None or n < 1:
        return nan
    return float(np.mean(channel))


def r_mean(p: PlotData) -> float:
    """Mean of the red channel over vegetation points (raw stored values, no rescale)."""
    return _channel_mean(p.red, p.n)


def g_mean(p: PlotData) -> float:
    """Mean of the green channel over vegetation points (raw stored values, no rescale)."""
    return _channel_mean(p.green, p.n)


def b_mean(p: PlotData) -> float:
    """Mean of the blue channel over vegetation points (raw stored values, no rescale)."""
    return _channel_mean(p.blue, p.n)


# --------------------------------------------------------------------------- #
# Class "2D" / Group G - horizontal structure (HAND-CRAFTED, not from reference) #
# --------------------------------------------------------------------------- #
# Everything above this point follows the open-source LiDAR_Vegetation_Metrics
# reference notebook. The features below are different: they were **hand-crafted
# for this project** to capture *horizontal* canopy structure that the purely
# vertical height metrics cannot see. Each is computed in the plot's own axis frame
# (a 2-D PCA of the vegetation XY), so it is invariant to how the plot is rotated in
# the world. The shared maths lives in ``featuregen.geometry`` so the viewer can draw
# the same axes/planes. All operate on vegetation points only.


def _axes(p: PlotData):
    """The plot's PCA frame ``(centroid, axes, eigvals)`` or ``None`` (cached per call site)."""
    return geo.pca_axes_2d(p.x, p.y)


def pca_spread_major(p: PlotData) -> float:
    """Point spread along the plot's major (long) axis: sqrt of the larger PCA eigenvalue, in m."""
    res = _axes(p)
    return float(np.sqrt(res[2][0])) if res is not None else nan


def pca_spread_minor(p: PlotData) -> float:
    """Point spread along the plot's minor (short) axis: sqrt of the smaller PCA eigenvalue, in m."""
    res = _axes(p)
    return float(np.sqrt(res[2][1])) if res is not None else nan


def pca_anisotropy(p: PlotData) -> float:
    """Elongation of the XY point pattern: (sqrt_l1 - sqrt_l2) / (sqrt_l1 + sqrt_l2), in [0, 1).

    0 = points spread equally in both directions (round); near 1 = strongly stretched
    along one axis. Scale- and rotation-free. NaN if undefined or both spreads are 0.
    """
    res = _axes(p)
    if res is None:
        return nan
    a, b = float(np.sqrt(res[2][0])), float(np.sqrt(res[2][1]))
    return (a - b) / (a + b) if (a + b) > 0 else nan


def asym_major(p: PlotData) -> float:
    """Front/back point imbalance along the major axis: (front - back) / total, in [-1, 1].

    0 = points balanced across the plot's long midline; sign follows the deterministic
    axis orientation (the side of the centroid with more points is positive). NaN if
    < 2 points.
    """
    res = _axes(p)
    if res is None:
        return nan
    centroid, axes, _ = res
    s_major, _ = geo.project_to_axes(p.x, p.y, centroid, axes)
    return geo.balance_index(int(np.sum(s_major > 0)), int(np.sum(s_major < 0)))


def asym_minor(p: PlotData) -> float:
    """Left/right point imbalance along the minor axis: (left - right) / total, in [-1, 1].

    0 = points balanced across the plot's short midline. NaN if < 2 points.
    """
    res = _axes(p)
    if res is None:
        return nan
    centroid, axes, _ = res
    _, s_minor = geo.project_to_axes(p.x, p.y, centroid, axes)
    return geo.balance_index(int(np.sum(s_minor > 0)), int(np.sum(s_minor < 0)))


def slope_angle_major(p: PlotData) -> float:
    """Angle (deg) between the top- and bottom-stratum surfaces tilting along the major axis.

    Fit a line to the top X% and another to the bottom X% of points (by height) in the
    (major-axis, height) plane, and take the angle between them. Terrain-robust (a shared
    ground tilt cancels). X is ``config.slope_strata_pct``. NaN if a stratum is too small.
    """
    res = _axes(p)
    if res is None:
        return nan
    centroid, axes, _ = res
    s_major, _ = geo.project_to_axes(p.x, p.y, centroid, axes)
    out = geo.directional_slope(s_major, p.h, p.config.slope_strata_pct)
    return out[2] if out is not None else nan


def slope_angle_minor(p: PlotData) -> float:
    """Angle (deg) between the top- and bottom-stratum surfaces tilting along the minor axis.

    As ``slope_angle_major`` but in the (minor-axis, height) plane. NaN if a stratum is
    too small.
    """
    res = _axes(p)
    if res is None:
        return nan
    centroid, axes, _ = res
    _, s_minor = geo.project_to_axes(p.x, p.y, centroid, axes)
    out = geo.directional_slope(s_minor, p.h, p.config.slope_strata_pct)
    return out[2] if out is not None else nan


def rough_major(p: PlotData) -> float:
    """Directional roughness along the major axis: RMS of heights about their major-axis trend, m.

    The 1-D analogue of sigma_z along the long axis - how bumpy the canopy is *along the
    rows*. NaN if < 2 points or degenerate.
    """
    res = _axes(p)
    if res is None:
        return nan
    centroid, axes, _ = res
    s_major, _ = geo.project_to_axes(p.x, p.y, centroid, axes)
    return geo.directional_roughness(s_major, p.h)


def rough_minor(p: PlotData) -> float:
    """Directional roughness along the minor axis: RMS of heights about their minor-axis trend, m.

    The 1-D analogue of sigma_z across the short axis. NaN if < 2 points or degenerate.
    """
    res = _axes(p)
    if res is None:
        return nan
    centroid, axes, _ = res
    _, s_minor = geo.project_to_axes(p.x, p.y, centroid, axes)
    return geo.directional_roughness(s_minor, p.h)


# --------------------------------------------------------------------------- #
# The registry - order here is the column order in the features .csv          #
# --------------------------------------------------------------------------- #

FEATURES: list[FeatureDef] = [
    # === Class: Height (vertical RelativeHeight distribution; Groups A-E) ============
    # Type: Central tendency (Group A)
    FeatureDef("h_mean", "Mean height", "Height", "Central tendency",
               "Arithmetic mean of vegetation RelativeHeight (off-ground points). "
               "Outlier-sensitive average height.", h_mean),
    FeatureDef("h_median", "Median height", "Height", "Central tendency",
               "Median (50th percentile) of vegetation RelativeHeight. Robust "
               "counterpart to the mean.", h_median),
    FeatureDef("h_max", "Max height", "Height", "Central tendency",
               "Single tallest vegetation RelativeHeight. Canopy-top proxy; most "
               "informative but fragile to a single high outlier.", h_max),
    FeatureDef("h_p25", "Height p25", "Height", "Central tendency",
               "25th percentile of vegetation RelativeHeight (numpy linear "
               "interpolation). Characterises the lower canopy/understory.", h_p25),
    FeatureDef("h_p75", "Height p75", "Height", "Central tendency",
               "75th percentile of vegetation RelativeHeight (linear interpolation). "
               "Characterises the upper canopy.", h_p75),
    FeatureDef("h_p95", "Height p95", "Height", "Central tendency",
               "95th percentile of vegetation RelativeHeight. Robust canopy top with "
               "the top 5% of returns trimmed off.", h_p95),
    # Type: Spread (Group B)
    FeatureDef("h_var", "Height variance", "Height", "Spread",
               "Sample variance (ddof=1) of vegetation RelativeHeight, in m^2. Bigger "
               "means more vertical spread.", h_var),
    FeatureDef("h_std", "Height std dev", "Height", "Spread",
               "Sample standard deviation (ddof=1) of vegetation RelativeHeight, in m.",
               h_std),
    FeatureDef("h_cv", "Coefficient of variation", "Height", "Spread",
               "Sample std (ddof=1) divided by the mean of vegetation RelativeHeight. "
               "Unitless relative spread, comparable across tall and short stands. "
               "NaN if the mean is <= 0.", h_cv),
    # Type: Shape (Group C)
    FeatureDef("h_skew", "Skewness", "Height", "Shape",
               "Skewness (scipy biased g1) of vegetation RelativeHeight. Positive: "
               "mostly low returns with a thin tall tail; negative: mostly high "
               "with a low tail. NaN if < 3 points or zero variance.", h_skew),
    FeatureDef("h_kurtosis", "Kurtosis", "Height", "Shape",
               "Excess kurtosis (Fisher, scipy biased) of vegetation RelativeHeight. "
               "0 = normal; positive = peaked/heavy-tailed; negative = flat/broad. "
               "NaN if < 4 points or zero variance.", h_kurtosis),
    # Type: Structure (Group D)
    FeatureDef("entropy", "Vertical entropy", "Height", "Structure",
               "Shannon entropy (bits) of vegetation RelativeHeight binned into 0.1 m "
               "vertical layers: -sum(p*log2(p)) over non-empty bins. Near 0 = "
               "single layer; high = many occupied layers.", entropy),
    FeatureDef("sigma_z", "Roughness (sigma_z)", "Height", "Structure",
               "RMS (ddof=0) of residuals after least-squares fitting a plane "
               "z = a*x + b*y + c through the vegetation points. Detrends tilt, "
               "leaving genuine surface bumpiness. NaN if < 3 points.", sigma_z),
    # Type: Density (Group E)
    FeatureDef("frac_above_mean", "Fraction above mean", "Height", "Density",
               "Fraction (0..1) of vegetation points whose RelativeHeight exceeds the "
               "vegetation mean height. A proportion, not a count, so it is unaffected "
               "by the drone-driven variation in point density.", frac_above_mean),
    FeatureDef("ppr", "Pulse penetration ratio", "Height", "Density",
               "Ground points (classification == ground code) divided by the total "
               "number of points in the file. High = open/sparse canopy; low = "
               "dense closed canopy. Always in [0, 1].", ppr),
    # === Class: Color (mean RGB over vegetation points; Group F) =====================
    FeatureDef("R_mean", "Mean red", "Color", "Colour",
               "Mean of the red channel over vegetation points (raw stored values, "
               "typically 16-bit 0-65535; no rescaling). NaN if the file has no RGB.",
               r_mean),
    FeatureDef("G_mean", "Mean green", "Color", "Colour",
               "Mean of the green channel over vegetation points (raw stored "
               "values; no rescaling). NaN if the file has no RGB.", g_mean),
    FeatureDef("B_mean", "Mean blue", "Color", "Colour",
               "Mean of the blue channel over vegetation points (raw stored "
               "values; no rescaling). NaN if the file has no RGB.", b_mean),
    # === Class: 2D (HAND-CRAFTED horizontal structure, in the plot's own PCA frame; Group G) =
    FeatureDef("pca_spread_major", "Spread (major axis)", "2D", "Horizontal (hand-crafted)",
               "Point spread along the plot's major (long) axis: sqrt of the larger "
               "eigenvalue of the 2-D PCA of the vegetation XY, in metres. Rotation-"
               "invariant. NaN if < 2 points.", pca_spread_major),
    FeatureDef("pca_spread_minor", "Spread (minor axis)", "2D", "Horizontal (hand-crafted)",
               "Point spread along the plot's minor (short) axis: sqrt of the smaller "
               "PCA eigenvalue, in metres. Rotation-invariant. NaN if < 2 points.",
               pca_spread_minor),
    FeatureDef("pca_anisotropy", "Anisotropy", "2D", "Horizontal (hand-crafted)",
               "Elongation of the XY point pattern: (sqrt_l1 - sqrt_l2) / (sqrt_l1 + "
               "sqrt_l2), in [0, 1). 0 = round/even spread; near 1 = strongly stretched "
               "along one axis. Scale- and rotation-free.", pca_anisotropy),
    FeatureDef("asym_major", "Asymmetry (major axis)", "2D", "Horizontal (hand-crafted)",
               "Front/back point imbalance along the major axis: (front - back) / total, "
               "in [-1, 1]. 0 = balanced about the long midline; sign uses a deterministic "
               "axis orientation (the side of the centroid with more points is positive). "
               "NaN if < 2 points.",
               asym_major),
    FeatureDef("asym_minor", "Asymmetry (minor axis)", "2D", "Horizontal (hand-crafted)",
               "Left/right point imbalance along the minor axis: (left - right) / total, "
               "in [-1, 1]. 0 = balanced about the short midline. NaN if < 2 points.",
               asym_minor),
    FeatureDef("slope_angle_major", "Top-bottom angle (major axis)", "2D", "Horizontal (hand-crafted)",
               "Angle (degrees) between two lines fitted to the top X% and bottom X% of "
               "points (by height) in the (major-axis, height) plane. Terrain-robust "
               "(shared ground tilt cancels). X = slope_strata_pct (Advanced). NaN if a "
               "stratum is too small.", slope_angle_major),
    FeatureDef("slope_angle_minor", "Top-bottom angle (minor axis)", "2D", "Horizontal (hand-crafted)",
               "As the major-axis top-bottom angle but tilting along the minor axis. "
               "NaN if a stratum is too small.", slope_angle_minor),
    FeatureDef("rough_major", "Directional roughness (major axis)", "2D", "Horizontal (hand-crafted)",
               "1-D roughness along the major axis: RMS (ddof=0) of heights about their "
               "linear trend in the major-axis coordinate. The directional analogue of "
               "sigma_z along the long axis. NaN if < 2 points.", rough_major),
    FeatureDef("rough_minor", "Directional roughness (minor axis)", "2D", "Horizontal (hand-crafted)",
               "1-D roughness along the minor axis: RMS of heights about their minor-axis "
               "trend. The directional analogue of sigma_z across the short axis. NaN if "
               "< 2 points.", rough_minor),
]

# Convenience lookups.
FEATURES_BY_KEY: dict[str, FeatureDef] = {f.key: f for f in FEATURES}


def compute_features(
    plot: PlotData, keys: list[str] | None = None
) -> dict[str, float]:
    """Compute the selected features for one plot.

    Parameters
    ----------
    plot:
        The plot data to evaluate.
    keys:
        Feature keys to compute, in any order; ``None`` means all features.
        Output preserves the canonical :data:`FEATURES` order regardless.

    Returns
    -------
    dict mapping feature key -> scalar value (``NaN`` where not computable).
    """
    selected = set(keys) if keys is not None else None
    return {
        f.key: f.func(plot)
        for f in FEATURES
        if selected is None or f.key in selected
    }
