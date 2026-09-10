"""Unit tests for the results core (field layout + per-plot predictions)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml import MODELS, TrainConfig
from ml.dataset import Dataset
from ml.plot_layout import N_COLS, N_ROWS, PLOT_TO_CELL, PLOTS
from ml.results import compute_plot_predictions
from ml.validate import validate_procedure


# --------------------------------------------------------------------------- #
# Field layout                                                                #
# --------------------------------------------------------------------------- #

def test_layout_is_an_8x10_snake_of_80_unique_cells():
    assert len(PLOTS) == N_ROWS * N_COLS == 80
    assert len(set(PLOT_TO_CELL.values())) == 80          # no two plots share a cell
    # Corners of the boustrophedon match the field map.
    assert PLOT_TO_CELL[438] == (0, 0)
    assert PLOT_TO_CELL[429] == (0, 9)
    assert PLOT_TO_CELL[439] == (1, 0)
    assert PLOT_TO_CELL[448] == (1, 9)
    assert PLOT_TO_CELL[499] == (7, 0)
    assert PLOT_TO_CELL[508] == (7, 9)


def test_build_layout_default_is_top_left_row_major():
    from ml.plot_layout import build_layout

    layout = build_layout()  # the map's default: start top-left, run right, step down
    assert layout[429] == (0, 0)   # base plot at the top-left
    assert layout[438] == (0, 9)   # end of the first row
    assert layout[439] == (1, 0)   # next row restarts at the left (no snake)
    assert layout[508] == (7, 9)   # last plot bottom-right
    assert len(set(layout.values())) == 80


def test_build_layout_covers_orientation_choices():
    from ml.plot_layout import build_layout

    # A small grid is easiest to reason about: 2 rows x 3 cols, base 0.
    rm = build_layout(rows=2, cols=3, base=0, start="top-left", major="row", snake=False)
    assert rm == {0: (0, 0), 1: (0, 1), 2: (0, 2), 3: (1, 0), 4: (1, 1), 5: (1, 2)}
    # Column-major from the bottom-right, snaking.
    cm = build_layout(rows=2, cols=3, base=0, start="bottom-right", major="column", snake=True)
    assert len(set(cm.values())) == 6 and set(cm.values()) == {(r, c) for r in range(2) for c in range(3)}
    assert cm[0] == (1, 2)  # bottom-right corner is the start


def test_build_plot_geoms_reads_centroid_and_hull(tmp_path):
    import laspy

    from ml.plot_geometry import build_plot_geoms

    file_map = {}
    for plot, (ox, oy) in {429: (0.0, 0.0), 430: (50.0, 0.0)}.items():
        las = laspy.LasData(laspy.LasHeader(point_format=3))
        rng = np.random.default_rng(plot)
        las.x = ox + rng.uniform(0, 10, 40)
        las.y = oy + rng.uniform(0, 10, 40)
        las.z = rng.uniform(0, 1, 40)
        path = tmp_path / f"plot_{plot}.las"
        las.write(path)
        file_map[(plot, 0)] = path
        file_map[(plot, 1)] = path  # an augmented copy that must be ignored at aug_index=0

    geoms = build_plot_geoms(file_map, aug_index=0)
    assert set(geoms) == {429, 430}                    # only the originals
    assert 0 <= geoms[429].centroid[0] <= 10           # plot 429 sits near the origin
    assert 50 <= geoms[430].centroid[0] <= 60          # plot 430 is shifted east
    # The footprint is the 4-corner oriented bounding box (kept small so the field view is fast).
    assert geoms[429].hull is not None and geoms[429].hull.shape == (4, 2)


def test_min_area_rect_recovers_a_rotated_rectangle():
    from ml.plot_geometry import min_area_rect

    rng = np.random.default_rng(0)
    pts = np.column_stack([rng.uniform(0, 20, 400), rng.uniform(0, 6, 400)])
    theta = np.radians(25)
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    rotated = pts @ rot.T
    rect = min_area_rect(rotated[:, 0], rotated[:, 1])
    assert rect.shape == (4, 2)
    sides = sorted(float(np.hypot(*(rect[(i + 1) % 4] - rect[i]))) for i in range(4))
    # Two short (~6) and two long (~20) sides, area ~120.
    assert sides[0] == pytest.approx(6, abs=0.5) and sides[-1] == pytest.approx(20, abs=0.5)
    assert min_area_rect(np.array([0.0, 1.0, 2.0]), np.array([0.0, 1.0, 2.0])) is None  # collinear


def test_trim_outliers_keeps_a_stray_point_from_inflating_the_footprint():
    from ml.plot_geometry import _trim_outliers, min_area_rect

    rng = np.random.default_rng(3)
    x = rng.uniform(0, 4, 2000)
    y = rng.uniform(0, 1.2, 2000)
    clean_area = _rect_area(min_area_rect(x, y))
    # One return 50 m away would otherwise stretch the convex hull (and thus the box) enormously.
    x2 = np.append(x, 54.0)
    y2 = np.append(y, 51.0)
    assert _rect_area(min_area_rect(x2, y2)) > 50 * clean_area  # the raw box explodes
    tx, ty = _trim_outliers(x2, y2)
    assert _rect_area(min_area_rect(tx, ty)) == pytest.approx(clean_area, rel=0.1)  # trimmed stays tight


def test_box_angle_matches_the_footprint_orientation():
    from gui.results.field_canvas import box_angle_deg, _box_sides

    # An axis-aligned tall box (long axis vertical) -> the long-axis angle is ~90 -> folded to 90.
    upright = np.array([[0.0, 0.0], [1.2, 0.0], [1.2, 4.0], [0.0, 4.0]])
    short, long_ = _box_sides(upright)
    assert short == pytest.approx(1.2) and long_ == pytest.approx(4.0)
    assert abs(box_angle_deg(upright)) == pytest.approx(90.0, abs=1e-6)

    # A 30-degree-rotated long box: the reported label angle folds into (-90, 90].
    theta = np.radians(30)
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    rotated = upright @ rot.T
    ang = box_angle_deg(rotated)
    assert -90 < ang <= 90


def _rect_area(corners) -> float:
    return float(np.hypot(*(corners[1] - corners[0])) * np.hypot(*(corners[2] - corners[1])))


# --------------------------------------------------------------------------- #
# Per-plot predictions                                                        #
# --------------------------------------------------------------------------- #

def _dataset() -> Dataset:
    idx = [
        "plot(429).laz", "plot(429)_aug(1).laz",
        "plot(430).laz", "plot(431).laz", "plot(432).laz", "plot(433).laz",
    ]
    frame = pd.DataFrame(
        {"f1": [1.0, 1.1, 2.0, 3.0, 4.0, 5.0], "y": [10.0, 10.0, 20.0, 30.0, 40.0, 50.0]},
        index=idx,
    )
    frame.index.name = "filename"
    return Dataset(frame=frame, feature_columns=["f1"], target_column="y")


def test_compute_plot_predictions_one_row_per_file_with_roles_and_error():
    dataset = _dataset()
    config = TrainConfig(model_key=MODELS[0].key, test_size=0.25, seed=0)
    history = validate_procedure(dataset, config, do_optimize=False, n_outer_splits=3).history

    table = compute_plot_predictions(dataset, history)
    assert list(table.columns) == ["plot", "aug", "actual", "predicted", "error_pct", "role"]
    assert len(table) == 6                                  # one row per labelled file
    # Per-plot aug id: the augmented copy is aug 1, originals are 0.
    assert table.loc["plot(429)_aug(1).laz", "aug"] == 1
    assert table.loc["plot(429).laz", "aug"] == 0
    assert set(table["role"]) <= {"T", "V"}
    # Every plot is either train or test in the best split (the sets partition the plots).
    assert history.best_train_plots | history.best_test_plots >= set(table["plot"])
    assert np.isfinite(table["error_pct"]).all()


# --------------------------------------------------------------------------- #
# Point-cloud view colouring matches the map view (full-range norm)           #
# --------------------------------------------------------------------------- #

def test_solid_colours_use_full_map_range_not_just_selection():
    """A cloud's colour must equal its map-view square's colour regardless of the selection.

    Regression: the 3-D view used to stretch the colour map over only the *selected* plots, so a
    set of all-low-error plots spanned green->red among themselves instead of all reading green.
    """
    from gui.results.map_view import solid_colours
    from gui.results.colormaps import full_range_norm, get_cmap

    # A wide error range across the whole map; the user selects four of the lowest-error plots.
    all_values = {429: 3.0, 430: 3.6, 431: 4.1, 432: 4.2, 433: 28.9, 444: 45.8}
    selected = {429, 430, 431, 432}

    colours = solid_colours(all_values, selected, "green→red")

    # The colour each selected plot gets must be what the map view assigns over the FULL range.
    cmap, norm = get_cmap("green→red"), full_range_norm(all_values.values())
    for plot in selected:
        expected = list(cmap(norm(all_values[plot]))[:3])
        assert colours[plot] == expected

    # And all four low-error plots stay on the green side (not stretched to red among themselves).
    for plot in selected:
        r, g, b = colours[plot]
        assert g > r, f"plot {plot} should read green-ish, got rgb=({r:.2f},{g:.2f},{b:.2f})"


def test_solid_colours_grey_for_missing_or_nan():
    from gui.results.map_view import solid_colours

    all_values = {429: 3.0, 430: float("nan")}
    colours = solid_colours(all_values, {429, 430, 999}, "green→red")
    assert colours[430] == [0.6, 0.6, 0.6]   # NaN value -> grey
    assert colours[999] == [0.6, 0.6, 0.6]   # not on the map -> grey


def test_effective_aug_state_reports_off_when_no_augmented_rows():
    """The per-model badge must read off when the run had no augmented plots, even if the
    'use augmented data' setting was ticked — ticking it over an originals-only folder does nothing.
    """
    from gui.results_page import effective_aug_state

    # The bug being fixed: setting ON but the run had zero augmented rows -> show OFF (orig-only).
    assert effective_aug_state(True, had_augmented=False) is False
    # Setting ON and the run genuinely had augmented rows -> show ON.
    assert effective_aug_state(True, had_augmented=True) is True
    # Setting OFF stays OFF regardless of whether augmented rows existed.
    assert effective_aug_state(False, had_augmented=True) is False
    assert effective_aug_state(False, had_augmented=False) is False
    # Legacy per-model flag not recorded (None) stays None -> faint "unknown" badge, never coerced.
    assert effective_aug_state(None, had_augmented=False) is None
    assert effective_aug_state(None, had_augmented=True) is None
    # Legacy bundle with unknown had_augmented (None) leaves the setting untouched (no regression).
    assert effective_aug_state(True, had_augmented=None) is True
    assert effective_aug_state(False, had_augmented=None) is False
