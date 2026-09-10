"""Unit tests for the vegetation metrics and the file reader.

Most tests build a :class:`PlotData` directly from synthetic arrays with
hand-computed answers; one round-trips a real (written) .las file through laspy to
exercise ``io_las.load_plot`` end to end.
"""

from __future__ import annotations

import math
from pathlib import Path

import laspy
import numpy as np
import pytest

from featuregen import Config, compute_features, load_plot, run_batch
from featuregen.external import load_external, plot_number_from_name
from featuregen.features import FEATURES, FEATURES_BY_KEY
from featuregen.io_las import PlotData


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #

def make_plot(
    h,
    *,
    x=None,
    y=None,
    rgb=None,
    classification=None,
    n_all=None,
    config: Config | None = None,
) -> PlotData:
    """Build a PlotData straight from arrays for metric-math tests."""
    h = np.asarray(h, dtype=np.float64)
    n = h.size
    x = np.zeros(n) if x is None else np.asarray(x, dtype=np.float64)
    y = np.zeros(n) if y is None else np.asarray(y, dtype=np.float64)
    if rgb is None:
        red = green = blue = None
    else:
        red, green, blue = (np.asarray(c, dtype=np.float64) for c in rgb)
    config = config or Config()
    if classification is None:
        classification = np.full(n, config.veg_code, dtype=np.uint8)
    else:
        classification = np.asarray(classification)
    n_all = classification.size if n_all is None else n_all
    return PlotData(
        name="synthetic", h=h, x=x, y=y, red=red, green=green, blue=blue,
        classification=classification, n_all=n_all, config=config,
    )


def feat(plot: PlotData, key: str) -> float:
    return FEATURES_BY_KEY[key].func(plot)


# --------------------------------------------------------------------------- #
# Central tendency / percentiles                                              #
# --------------------------------------------------------------------------- #

def test_central_tendency_and_percentiles():
    p = make_plot([0, 1, 2, 3, 4])
    assert feat(p, "h_mean") == pytest.approx(2.0)
    assert feat(p, "h_median") == pytest.approx(2.0)
    assert feat(p, "h_max") == pytest.approx(4.0)
    assert feat(p, "h_p25") == pytest.approx(1.0)
    assert feat(p, "h_p75") == pytest.approx(3.0)
    assert feat(p, "h_p95") == pytest.approx(3.8)


def test_spread_sample_statistics():
    h = [0.0, 1.0, 2.0, 3.0, 4.0]
    p = make_plot(h)
    assert feat(p, "h_var") == pytest.approx(np.var(h, ddof=1))
    assert feat(p, "h_std") == pytest.approx(np.std(h, ddof=1))
    # CV must use the sample std (ddof=1), matching h_std - the corrected behaviour.
    assert feat(p, "h_cv") == pytest.approx(np.std(h, ddof=1) / np.mean(h))


# --------------------------------------------------------------------------- #
# Constant-height degenerate cell                                             #
# --------------------------------------------------------------------------- #

def test_constant_height_cell():
    p = make_plot([5.0, 5.0, 5.0, 5.0])
    assert feat(p, "h_var") == pytest.approx(0.0)
    assert feat(p, "h_std") == pytest.approx(0.0)
    assert feat(p, "h_cv") == pytest.approx(0.0)
    assert feat(p, "entropy") == pytest.approx(0.0)   # all points in one bin
    assert math.isnan(feat(p, "h_skew"))              # zero-variance guard
    assert math.isnan(feat(p, "h_kurtosis"))


# --------------------------------------------------------------------------- #
# Shape                                                                       #
# --------------------------------------------------------------------------- #

def test_skew_sign_and_kurtosis_finite():
    right_skewed = make_plot([0, 0, 0, 0, 10])
    assert feat(right_skewed, "h_skew") > 0
    assert math.isfinite(feat(right_skewed, "h_kurtosis"))


# --------------------------------------------------------------------------- #
# Structure                                                                   #
# --------------------------------------------------------------------------- #

def test_entropy_two_equal_layers_is_one_bit():
    # Two equally-populated 0.5 m bins -> entropy = 1 bit (explicit bin size).
    p = make_plot([0.1, 0.2, 5.1, 5.2], config=Config(entropy_bin_size=0.5))
    assert feat(p, "entropy") == pytest.approx(1.0)


def test_entropy_default_bin_is_ten_cm():
    # Default bin size is now 0.1 m: four points each in their own 0.1 m bin -> 2 bits.
    assert Config().entropy_bin_size == pytest.approx(0.1)
    p = make_plot([0.05, 0.15, 0.25, 0.35])  # one point per 0.1 m bin
    assert feat(p, "entropy") == pytest.approx(2.0)


def test_sigma_z_detrends_a_tilted_plane():
    # Heights lie exactly on z = 2x + 3y + 1 -> roughness ~ 0, raw std > 0.
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 10, 200)
    y = rng.uniform(0, 10, 200)
    h = 2 * x + 3 * y + 1
    p = make_plot(h, x=x, y=y)
    assert feat(p, "sigma_z") == pytest.approx(0.0, abs=1e-6)
    assert feat(p, "h_std") > 1.0


def test_sigma_z_captures_roughness_on_flat_plot():
    rng = np.random.default_rng(1)
    x = rng.uniform(0, 10, 500)
    y = rng.uniform(0, 10, 500)
    noise = rng.normal(0, 0.25, 500)  # no tilt, pure bumpiness
    p = make_plot(noise, x=x, y=y)
    assert feat(p, "sigma_z") == pytest.approx(0.25, abs=0.05)


# --------------------------------------------------------------------------- #
# Density and penetration                                                     #
# --------------------------------------------------------------------------- #

def test_frac_above_mean_is_a_fraction():
    p = make_plot([0, 0, 0, 10])  # mean 2.5 -> one of four points above
    assert feat(p, "frac_above_mean") == pytest.approx(0.25)


def test_ppr_equals_ground_fraction():
    # 3 ground (code 2) out of 10 total -> PPR 0.3. Veg heights irrelevant to PPR.
    classification = np.array([1, 1, 1, 1, 1, 1, 1, 2, 2, 2], dtype=np.uint8)
    p = make_plot([1.0, 2.0], classification=classification, n_all=10)
    assert feat(p, "ppr") == pytest.approx(0.3)


# --------------------------------------------------------------------------- #
# Colour                                                                      #
# --------------------------------------------------------------------------- #

def test_rgb_means_over_vegetation():
    p = make_plot([1, 2, 3], rgb=([10, 20, 30], [40, 50, 60], [70, 80, 90]))
    assert feat(p, "R_mean") == pytest.approx(20.0)
    assert feat(p, "G_mean") == pytest.approx(50.0)
    assert feat(p, "B_mean") == pytest.approx(80.0)


def test_rgb_missing_is_nan():
    p = make_plot([1, 2, 3])  # no rgb
    assert math.isnan(feat(p, "R_mean"))
    assert math.isnan(feat(p, "G_mean"))
    assert math.isnan(feat(p, "B_mean"))


# --------------------------------------------------------------------------- #
# Horizontal (hand-crafted) features - PCA axis frame                          #
# --------------------------------------------------------------------------- #

_HORIZONTAL_KEYS = [
    "pca_spread_major", "pca_spread_minor", "pca_anisotropy",
    "asym_major", "asym_minor", "slope_angle_major", "slope_angle_minor",
    "rough_major", "rough_minor",
]


def _wheat_plot(angle_deg=0.0, dx=0.0, dy=0.0, lopsided=False, seed=0):
    """A synthetic 4 m x 1 m wheat-like plot, optionally rotated/translated/lopsided."""
    rng = np.random.default_rng(seed)
    n = 600
    sx = rng.uniform(0, 4, n)   # along the long axis
    sy = rng.uniform(0, 1, n)   # along the short axis
    if lopsided:               # bias points toward the high end of the long axis
        keep = rng.random(n) < (0.2 + 0.8 * sx / 4)
        sx, sy = sx[keep], sy[keep]
    h = 0.5 + 0.05 * sx + rng.normal(0, 0.05, sx.size)
    a = np.deg2rad(angle_deg)
    x = sx * np.cos(a) - sy * np.sin(a) + dx
    y = sx * np.sin(a) + sy * np.cos(a) + dy
    return make_plot(h, x=x, y=y)


def test_pca_finds_long_axis():
    p = _wheat_plot()
    # The 4 m extent must register as more spread than the 1 m extent.
    assert feat(p, "pca_spread_major") > feat(p, "pca_spread_minor")
    assert 0.0 <= feat(p, "pca_anisotropy") < 1.0


def test_horizontal_features_are_rotation_and_translation_invariant():
    base = {k: feat(_wheat_plot(seed=3), k) for k in _HORIZONTAL_KEYS}
    moved = {k: feat(_wheat_plot(angle_deg=53.0, dx=1000.0, dy=-200.0, seed=3), k)
             for k in _HORIZONTAL_KEYS}
    for k in _HORIZONTAL_KEYS:
        assert moved[k] == pytest.approx(base[k], abs=1e-6), k


def test_asymmetry_indices_are_bounded_and_signed():
    # A balanced plot -> asymmetry near 0; a lopsided plot -> clearly non-zero, in [-1, 1].
    balanced = feat(_wheat_plot(seed=4), "asym_major")
    lopsided = feat(_wheat_plot(lopsided=True, seed=4), "asym_major")
    assert abs(balanced) < 0.1
    assert -1.0 <= lopsided <= 1.0
    assert abs(lopsided) > abs(balanced)


def test_directional_roughness_matches_a_known_tilt_plus_noise():
    # Heights = pure tilt along the long axis + small bump noise; the long-axis trend is
    # removed, so rough_major ~ the injected noise std, and it stays finite and small.
    rng = np.random.default_rng(7)
    sx = rng.uniform(0, 4, 800)
    sy = rng.uniform(0, 1, 800)
    h = 0.3 * sx + rng.normal(0, 0.1, 800)  # tilt only along the long axis
    p = make_plot(h, x=sx, y=sy)
    assert feat(p, "rough_major") == pytest.approx(0.1, abs=0.03)


def test_horizontal_features_nan_on_too_few_points():
    one = make_plot([3.0], x=[0.0], y=[0.0])
    for k in _HORIZONTAL_KEYS:
        assert math.isnan(feat(one, k)), k


# --------------------------------------------------------------------------- #
# Degenerate-input guards                                                      #
# --------------------------------------------------------------------------- #

def test_small_sample_guards():
    empty = make_plot([])
    for key in ("h_mean", "h_median", "h_max", "entropy", "frac_above_mean"):
        assert math.isnan(feat(empty, key))

    one = make_plot([3.0])
    assert math.isnan(feat(one, "h_var"))      # needs >= 2
    assert math.isnan(feat(one, "h_cv"))
    assert math.isnan(feat(one, "sigma_z"))    # needs >= 3

    two = make_plot([1.0, 2.0])
    assert math.isnan(feat(two, "h_skew"))     # needs >= 3

    three = make_plot([1.0, 2.0, 4.0])
    assert math.isnan(feat(three, "h_kurtosis"))  # needs >= 4


def test_cv_nan_for_nonpositive_mean():
    p = make_plot([-1.0, 0.0, 1.0])  # mean 0
    assert math.isnan(feat(p, "h_cv"))


# --------------------------------------------------------------------------- #
# io_las + pipeline round trip with a written .las file                       #
# --------------------------------------------------------------------------- #

def _write_las(path: Path) -> None:
    """Write a small point-format-3 (RGB) LAS file with a RelativeHeight extra dim."""
    header = laspy.LasHeader(point_format=3, version="1.2")
    header.add_extra_dim(laspy.ExtraBytesParams(name="RelativeHeight", type=np.float32))
    las = laspy.LasData(header)

    # 4 vegetation points (class 1) + 2 ground points (class 2).
    n = 6
    las.x = np.array([0, 1, 2, 3, 4, 5], dtype=np.float64)
    las.y = np.array([0, 0, 0, 0, 0, 0], dtype=np.float64)
    las.z = np.array([1, 2, 3, 4, 0, 0], dtype=np.float64)
    las.classification = np.array([1, 1, 1, 1, 2, 2], dtype=np.uint8)
    las.RelativeHeight = np.array([1.0, 2.0, 3.0, 4.0, 0.0, 0.0], dtype=np.float32)
    las.red = np.array([10, 10, 10, 10, 99, 99], dtype=np.uint16)
    las.green = np.array([20, 20, 20, 20, 99, 99], dtype=np.uint16)
    las.blue = np.array([30, 30, 30, 30, 99, 99], dtype=np.uint16)
    assert len(las.points) == n
    las.write(path)


def test_load_plot_round_trip(tmp_path):
    las_path = tmp_path / "plot.las"
    _write_las(las_path)
    plot = load_plot(las_path)

    assert plot.n == 4                      # only the class-1 vegetation points
    assert plot.n_all == 6
    np.testing.assert_allclose(plot.h, [1.0, 2.0, 3.0, 4.0])
    # RGB means computed over vegetation points only (ground's 99s excluded).
    assert feat(plot, "R_mean") == pytest.approx(10.0)
    assert feat(plot, "ppr") == pytest.approx(2 / 6)


def test_missing_height_channel_raises(tmp_path):
    las_path = tmp_path / "no_norm.las"
    header = laspy.LasHeader(point_format=3, version="1.2")
    las = laspy.LasData(header)
    las.x = np.array([0.0]); las.y = np.array([0.0]); las.z = np.array([0.0])
    las.classification = np.array([1], dtype=np.uint8)
    las.write(las_path)
    with pytest.raises(KeyError):
        load_plot(las_path)


def test_run_batch_writes_csv(tmp_path):
    good = tmp_path / "good.las"
    _write_las(good)
    bad = tmp_path / "bad.las"  # missing RelativeHeight -> should be recorded as failed
    header = laspy.LasHeader(point_format=3, version="1.2")
    las = laspy.LasData(header)
    las.x = np.array([0.0]); las.y = np.array([0.0]); las.z = np.array([0.0])
    las.classification = np.array([1], dtype=np.uint8)
    las.write(bad)

    out = tmp_path / "out.csv"
    result = run_batch([good, bad], out)

    assert out.exists()
    assert result.n_succeeded == 1
    assert result.n_failed == 1
    assert "good.las" in result.frame.index
    assert list(result.frame.columns) == [f.key for f in FEATURES]
    # The written CSV round-trips, skipping the leading '#' description comment lines.
    reread = pd.read_csv(out, index_col=0, comment="#")
    assert "good.las" in reread.index


# --------------------------------------------------------------------------- #
# Imported reference features (external spreadsheet)                           #
# --------------------------------------------------------------------------- #

import pandas as pd  # noqa: E402 - kept beside the external-feature tests


def test_plot_number_from_name():
    assert plot_number_from_name("test_plot(429).laz") == 429
    assert plot_number_from_name("PLOT( 12 ).las") == 12
    assert plot_number_from_name("no_number_here.laz") is None


def test_plot_number_from_label_custom_format():
    from common.naming import LabelFormat, plot_number_from_label

    fmt = LabelFormat(start="P_", end="")
    assert plot_number_from_label("P_429.laz", fmt) == 429
    # A bare integer still reads even under a custom format (reference cells are often plain numbers).
    assert plot_number_from_label("430", fmt) == 430
    assert plot_number_from_label("430.0", fmt) == 430
    # The default format is plot(N).
    assert plot_number_from_label("plot(7).laz") == 7
    assert plot_number_from_label("nothing") is None


def _write_reference_xlsx(path: Path) -> None:
    """A tiny reference sheet: first column = plot number, then mixed columns."""
    df = pd.DataFrame(
        {
            "F.PLT": [429, 430, 431],
            "Block": [1, 1, 1],
            "Species": ["T", "T", "M"],
            "Plant height": [74.0, 78.0, 61.0],
        }
    )
    df.to_excel(path, index=False)


def test_load_external_and_lookup(tmp_path):
    xlsx = tmp_path / "ref.xlsx"
    _write_reference_xlsx(xlsx)
    ext = load_external(xlsx)

    assert ext.columns == ["Block", "Species", "Plant height"]
    matched = ext.lookup("test_plot(430).laz", ["Species", "Plant height"])
    assert matched["Species"] == "T"
    assert matched["Plant height"] == pytest.approx(78.0)

    # Unmatched plot number -> NaN for every requested column.
    missing = ext.lookup("test_plot(999).laz", ["Plant height"])
    assert math.isnan(missing["Plant height"])


def test_load_external_csv_with_chosen_key_column(tmp_path):
    """A .csv reference keyed on a named column whose cells are spelled in a custom label format."""
    from common.naming import LabelFormat

    csv = tmp_path / "ref.csv"
    pd.DataFrame(
        {
            "name": ["P_429_", "P_430_"],   # plot number wrapped as P_<n>_
            "Block": [1, 2],
            "Plant height": [74.0, 78.0],
        }
    ).to_csv(csv, index=False)

    ext = load_external(csv, key_column="name", label_format=LabelFormat(start="P_", end="_"))
    assert ext.columns == ["Block", "Plant height"]
    # The plot(N) file still matches by its number, read through the custom-format key.
    row = ext.lookup("test_plot(430).laz", ["Plant height"])
    assert row["Plant height"] == pytest.approx(78.0)


def test_list_reference_columns_returns_every_header(tmp_path):
    """The Label-key selector must offer *every* column, including the key column.

    Regression: reading the header with ``nrows=0`` then ``dropna(how="all")`` dropped all
    columns (a zero-row frame looks all-empty), so the Label dropdown only had its placeholder.
    """
    from featuregen.external import list_reference_columns

    csv = tmp_path / "ref.csv"
    pd.DataFrame(
        {"F.PLT": [429, 430], "Block": [1, 1], "Plant height": [74.0, 78.0]}
    ).to_csv(csv, index=False)

    assert list_reference_columns(csv) == ["F.PLT", "Block", "Plant height"]


def test_list_reference_columns_drops_trailing_blank(tmp_path):
    """A genuinely all-blank trailing column is still dropped (matches load_external)."""
    from featuregen.external import list_reference_columns

    csv = tmp_path / "ref.csv"
    csv.write_text("F.PLT,Block,\n429,1,\n430,1,\n")
    assert list_reference_columns(csv) == ["F.PLT", "Block"]


def test_run_batch_appends_external_columns(tmp_path):
    las_path = tmp_path / "test_plot(429).las"
    _write_las(las_path)
    xlsx = tmp_path / "ref.xlsx"
    _write_reference_xlsx(xlsx)
    ext = load_external(xlsx)

    out = tmp_path / "out.csv"
    result = run_batch(
        [las_path],
        out,
        feature_keys=["h_mean"],
        external=ext,
        external_columns=["Species", "Plant height"],
    )

    assert list(result.frame.columns) == ["h_mean", "Species", "Plant height"]
    row = result.frame.loc["test_plot(429).las"]
    assert row["Species"] == "T"
    assert row["Plant height"] == pytest.approx(74.0)


def test_run_batch_one_hot_encodes_coded_column(tmp_path):
    """A coded reference column ticked for splitting becomes one binary column per value."""
    a = tmp_path / "test_plot(429).las"   # Species == "T"
    b = tmp_path / "test_plot(431).las"   # Species == "M"
    _write_las(a)
    _write_las(b)
    xlsx = tmp_path / "ref.xlsx"
    _write_reference_xlsx(xlsx)
    ext = load_external(xlsx)

    result = run_batch(
        [a, b], tmp_path / "out.csv",
        feature_keys=["h_mean"],
        external=ext,
        external_columns=["Species"],
        encode_columns=["Species"],
    )

    # The raw "Species" column is replaced by one binary column per value.
    assert "Species" not in result.frame.columns
    assert {"Species=M", "Species=T"} <= set(result.frame.columns)
    assert result.frame.loc["test_plot(429).las", "Species=T"] == pytest.approx(1.0)
    assert result.frame.loc["test_plot(429).las", "Species=M"] == pytest.approx(0.0)
    assert result.frame.loc["test_plot(431).las", "Species=M"] == pytest.approx(1.0)


def test_run_batch_one_hot_unmatched_is_nan(tmp_path):
    """A plot absent from the reference sheet gets NaN (not a spurious 0) across the dummies."""
    las_path = tmp_path / "test_plot(999).las"  # not in the sheet
    _write_las(las_path)
    xlsx = tmp_path / "ref.xlsx"
    _write_reference_xlsx(xlsx)
    ext = load_external(xlsx)

    result = run_batch(
        [las_path], tmp_path / "out.csv",
        feature_keys=["h_mean"],
        external=ext,
        external_columns=["Species"],
        encode_columns=["Species"],
    )
    for col in [c for c in result.frame.columns if c.startswith("Species=")]:
        assert math.isnan(result.frame.loc["test_plot(999).las", col])


def test_run_batch_one_hot_over_cap_raises(tmp_path):
    """Splitting a column with too many unique values is refused."""
    from featuregen.pipeline import MAX_ONEHOT_COLUMNS

    las_path = tmp_path / "test_plot(429).las"
    _write_las(las_path)
    n = MAX_ONEHOT_COLUMNS + 5
    xlsx = tmp_path / "big.xlsx"
    # A Code column with more than the cap of distinct values.
    pd.DataFrame({"F.PLT": list(range(n)), "Code": list(range(n))}).to_excel(xlsx, index=False)
    ext = load_external(xlsx)

    with pytest.raises(ValueError, match="binary features"):
        run_batch(
            [las_path], tmp_path / "out.csv",
            feature_keys=["h_mean"],
            external=ext,
            external_columns=["Code"],
            encode_columns=["Code"],
        )
