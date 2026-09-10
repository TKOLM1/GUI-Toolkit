"""Unit tests for the import route: the .pcd reader and the folder importer."""

from __future__ import annotations

import csv
import struct

import laspy
import numpy as np
import pytest

from clip import GridLayout, MANIFEST_NAME, import_plots, list_sources, plan_import, read_pcd
from clip.pcd import PcdError
from common import las_io
from common.naming import KeyFormat, LabelFormat, PlotKey, key_stride


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #

def write_pcd(path, xyzi: list[tuple[float, float, float, float]], binary: bool = True) -> None:
    """Write a minimal 'x y z intensity' PCD, in the same layout the SGCBP files use."""
    header = (
        "# .PCD v0.7 - Point Cloud Data file format\n"
        "VERSION 0.7\n"
        "FIELDS x y z intensity\n"
        "SIZE 4 4 4 4\n"
        "TYPE F F F F\n"
        "COUNT 1 1 1 1\n"
        f"WIDTH {len(xyzi)}\n"
        "HEIGHT 1\n"
        "VIEWPOINT 0 0 0 1 0 0 0\n"
        f"POINTS {len(xyzi)}\n"
        f"DATA {'binary' if binary else 'ascii'}\n"
    ).encode("ascii")
    if binary:
        body = b"".join(struct.pack("<ffff", *row) for row in xyzi)
    else:
        body = "".join(" ".join(f"{v:g}" for v in row) + "\n" for row in xyzi).encode("ascii")
    path.write_bytes(header + body)


def write_las(path, xy: list[tuple[float, float]]) -> None:
    """A tiny point-format-3 .laz at the given (x, y), z = 0."""
    header = laspy.LasHeader(point_format=3, version="1.2")
    header.scales = np.array([1e-4, 1e-4, 1e-4])
    header.offsets = np.array([0.0, 0.0, 0.0])
    las = laspy.LasData(header)
    las.x = np.array([p[0] for p in xy], dtype=np.float64)
    las.y = np.array([p[1] for p in xy], dtype=np.float64)
    las.z = np.zeros(len(xy), dtype=np.float64)
    las_io.write_cloud(las, path)


_FMT = KeyFormat(col=LabelFormat(start="", end=""))  # bare integers, single-numbered


# --------------------------------------------------------------------------- #
# .pcd reader                                                                  #
# --------------------------------------------------------------------------- #

def test_read_pcd_binary_maps_xyz_and_intensity(tmp_path):
    path = tmp_path / "a.pcd"
    write_pcd(path, [(1.0, 2.0, 3.0, 7.0), (4.0, 5.0, 6.0, 9.0)])

    las = read_pcd(path)

    assert len(las.x) == 2
    assert np.allclose(np.asarray(las.x), [1.0, 4.0])
    assert np.allclose(np.asarray(las.z), [3.0, 6.0])
    assert list(np.asarray(las.intensity)) == [7, 9]
    # No class field in the source, so every point gets the project's default.
    assert set(np.unique(np.asarray(las.classification))) == {1}


def test_read_pcd_ascii_matches_binary(tmp_path):
    points = [(1.5, 2.5, 3.5, 4.0), (-1.0, 0.0, 2.0, 0.0)]
    binary, ascii_ = tmp_path / "b.pcd", tmp_path / "a.pcd"
    write_pcd(binary, points, binary=True)
    write_pcd(ascii_, points, binary=False)

    from_binary, from_ascii = read_pcd(binary), read_pcd(ascii_)
    assert np.allclose(np.asarray(from_binary.x), np.asarray(from_ascii.x))
    assert np.allclose(np.asarray(from_binary.z), np.asarray(from_ascii.z))


def test_read_pcd_rejects_compressed(tmp_path):
    path = tmp_path / "c.pcd"
    path.write_bytes(
        b"VERSION 0.7\nFIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1\n"
        b"WIDTH 1\nHEIGHT 1\nPOINTS 1\nDATA binary_compressed\n\x00\x00"
    )
    with pytest.raises(PcdError, match="binary_compressed|not supported"):
        read_pcd(path)


# --------------------------------------------------------------------------- #
# Numbering                                                                    #
# --------------------------------------------------------------------------- #

def test_plan_reads_numbers_through_the_label_format(tmp_path):
    for name in ("1-1-1-b.pcd", "1-12-1-b.pcd"):
        write_pcd(tmp_path / name, [(0.0, 0.0, 0.0, 1.0)])

    plan = plan_import(
        list_sources(tmp_path), KeyFormat(col=LabelFormat(start="1-", end="-1-b"))
    )

    assert [n for _, _, n in plan.entries] == [1, 12]
    assert plan.is_clean


def test_row_and_column_keep_grid_plots_distinct(tmp_path):
    """The SGCBP shape: <run>-<range>-1-b, where the range alone repeats across runs."""
    for name in ("1-12-1-b.pcd", "2-12-1-b.pcd"):
        write_pcd(tmp_path / name, [(0.0, 0.0, 0.0, 1.0)])

    col = LabelFormat(start="-", end="-1-b")
    row = LabelFormat(start="", end="-")

    # The column alone cannot tell the two files apart.
    flat = plan_import(list_sources(tmp_path), KeyFormat(col=col))
    assert flat.duplicates and not flat.is_clean

    # The pair can, and packs to a readable row*stride + column.
    paired = plan_import(list_sources(tmp_path), KeyFormat(col=col, row=row))
    assert paired.is_clean
    assert [(k.row, k.col) for _, k, _ in paired.entries] == [(1, 12), (2, 12)]
    assert [n for _, _, n in paired.entries] == [112, 212]


def test_single_number_datasets_keep_their_numbers(tmp_path):
    """With no row, the plot number passes through untouched — 429 stays 429."""
    write_las(tmp_path / "plot(429).las", [(0.0, 0.0)])

    plan = plan_import(list_sources(tmp_path), KeyFormat(col=LabelFormat()))

    assert [n for _, _, n in plan.entries] == [429]


def test_stride_clears_the_widest_column():
    assert key_stride([PlotKey(col=9)]) == 10
    assert key_stride([PlotKey(col=18)]) == 100
    assert key_stride([PlotKey(col=100)]) == 1000


def test_sources_are_found_recursively(tmp_path):
    """Datasets split their plots across sub-folders; importing per folder would renumber each."""
    for session in ("run_001", "run_002"):
        (tmp_path / session).mkdir()
        write_pcd(tmp_path / session / f"{session[-1]}-1-1-b.pcd", [(0.0, 0.0, 0.0, 1.0)])

    assert len(list_sources(tmp_path)) == 2


def test_plan_flags_duplicates_and_unreadable_names(tmp_path):
    # Two files carry the number 3; "notes" carries none.
    for name in ("plot(3).las", "other_plot(3).las", "notes.las"):
        write_las(tmp_path / name, [(0.0, 0.0)])

    plan = plan_import(list_sources(tmp_path), KeyFormat(col=LabelFormat()))

    assert set(plan.duplicates) == {3}
    assert [p.name for p in plan.unreadable] == ["notes.las"]
    assert not plan.is_clean


# --------------------------------------------------------------------------- #
# import_plots                                                                 #
# --------------------------------------------------------------------------- #

def test_import_converts_pcd_and_writes_canonical_names(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    write_pcd(src / "7.pcd", [(1.0, 2.0, 3.0, 4.0)])

    result = import_plots(list_sources(src), tmp_path / "out", _FMT, universal_string="_siteA")

    assert result.n_written == 1
    assert result.written[0].name == "plot(7)_siteA.laz"
    assert las_io.point_count(result.written[0]) == 1


def test_import_scales_units(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    write_pcd(src / "1.pcd", [(1000.0, 2000.0, 3000.0, 1.0)])

    result = import_plots(list_sources(src), tmp_path / "out", _FMT, scale=0.001)

    las = las_io.read_cloud(result.written[0])
    assert np.allclose(np.asarray(las.z), [3.0])  # millimetres became metres


def test_import_writes_manifest_linking_source_to_plot(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    write_pcd(src / "1-12-1-b.pcd", [(0.0, 0.0, 0.0, 1.0)])

    result = import_plots(
        list_sources(src), tmp_path / "out",
        KeyFormat(col=LabelFormat(start="1-", end="-1-b")),
    )

    manifest = result.output_dir / MANIFEST_NAME
    assert manifest == result.manifest_path
    rows = list(csv.DictReader(manifest.open(encoding="utf-8")))
    assert rows[0]["plot"] == "12"
    assert rows[0]["source_file"] == "1-12-1-b.pcd"
    assert rows[0]["file"] == "plot(12).laz"


def test_import_skips_unreadable_names_but_finishes(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    write_las(src / "plot(1).las", [(0.0, 0.0)])
    write_las(src / "notes.las", [(0.0, 0.0)])

    result = import_plots(list_sources(src), tmp_path / "out", KeyFormat(col=LabelFormat()))

    assert result.n_written == 1
    assert [name for name, _ in result.skipped] == ["notes.las"]


def test_grid_separates_plots_that_overlapped(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    # Three plots all sitting on the same 2 m x 2 m footprint at the origin.
    for number in (1, 2, 3):
        write_las(src / f"{number}.las", [(-1.0, -1.0), (1.0, 1.0)])

    result = import_plots(
        list_sources(src), tmp_path / "out", _FMT,
        grid=GridLayout(columns=2, padding=0.10),
    )

    centres = {}
    for path in result.written:
        las = las_io.read_cloud(path)
        x, y = np.asarray(las.x), np.asarray(las.y)
        centres[path.name] = ((x.min() + x.max()) / 2, (y.min() + y.max()) / 2)

    # 2 m span + 10% => a 2.2 m pitch; plot 1 at the origin cell, 2 beside it, 3 on the next row.
    assert centres["plot(1).laz"] == pytest.approx((0.0, 0.0), abs=1e-3)
    assert centres["plot(2).laz"] == pytest.approx((2.2, 0.0), abs=1e-3)
    assert centres["plot(3).laz"] == pytest.approx((0.0, -2.2), abs=1e-3)


def test_grid_leaves_z_alone(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    write_pcd(src / "1.pcd", [(0.0, 0.0, 5.0, 1.0), (1.0, 1.0, 9.0, 1.0)])

    result = import_plots(
        list_sources(src), tmp_path / "out", _FMT, grid=GridLayout(columns=1)
    )

    las = las_io.read_cloud(result.written[0])
    assert sorted(np.asarray(las.z).round(3)) == [5.0, 9.0]  # heights are real data


# --------------------------------------------------------------------------- #
# The reference-sheet join                                                     #
# --------------------------------------------------------------------------- #

def test_paired_keys_join_files_and_reference_sheet(tmp_path):
    """The point of the pair: both sides derive the same plot number, so they cannot decouple.

    Mirrors SGCBP — files named <run>-<range>-1-b, ground truth keyed on (runNo, rangeNo) — where
    neither number identifies a plot on its own.
    """
    import pandas as pd
    from featuregen.external import load_external
    from common.naming import plot_number_from_name

    src = tmp_path / "src"
    src.mkdir()
    for run, rng in ((1, 12), (2, 12), (2, 3)):
        write_pcd(src / f"{run}-{rng}-1-b.pcd", [(0.0, 0.0, 0.0, 1.0)])

    fmt = KeyFormat(
        col=LabelFormat(start="-", end="-1-b"), row=LabelFormat(start="", end="-")
    )
    plan = plan_import(list_sources(src), fmt)
    result = import_plots(list_sources(src), tmp_path / "out", fmt)

    sheet = tmp_path / "gt.csv"
    pd.DataFrame(
        {"runNo": [1, 2, 2], "rangeNo": [12, 12, 3], "biomass": [10.0, 20.0, 30.0]}
    ).to_csv(sheet, index=False)

    external = load_external(
        sheet, key_column="rangeNo", row_column="runNo",
        label_format=LabelFormat(start="", end=""), stride=plan.stride,
    )

    # Every imported plot finds its row, and finds the *right* one.
    for path in result.written:
        number = plot_number_from_name(path.name)
        assert number in external.frame.index
    assert external.frame.loc[plot_number_from_name("plot(112).laz"), "biomass"] == 10.0
    assert external.frame.loc[plot_number_from_name("plot(212).laz"), "biomass"] == 20.0
    # runNo was folded into the index, so it is not offered as a feature.
    assert "runNo" not in external.columns


def test_grid_rebuilds_the_field_layout_from_row_and_column(tmp_path):
    """With a row/column pair the grid is the real field arrangement, not an arbitrary fill."""
    src = tmp_path / "src"
    src.mkdir()
    for run, rng in ((1, 1), (1, 2), (2, 1)):
        write_pcd(src / f"{run}-{rng}-1-b.pcd",
                  [(-1.0, -1.0, 0.0, 1.0), (1.0, 1.0, 0.0, 1.0)])

    fmt = KeyFormat(
        col=LabelFormat(start="-", end="-1-b"), row=LabelFormat(start="", end="-")
    )
    result = import_plots(
        list_sources(src), tmp_path / "out", fmt, grid=GridLayout(columns=99, padding=0.10)
    )

    centres = {}
    for path in result.written:
        las = las_io.read_cloud(path)
        x, y = np.asarray(las.x), np.asarray(las.y)
        centres[path.name] = ((x.min() + x.max()) / 2, (y.min() + y.max()) / 2)

    # 2 m span + 10% => 2.2 m pitch. Column advances X, row advances -Y; the "columns=99"
    # setting is ignored, because the data itself says where each plot goes.
    assert centres["plot(11).laz"] == pytest.approx((0.0, 0.0), abs=1e-3)
    assert centres["plot(12).laz"] == pytest.approx((2.2, 0.0), abs=1e-3)
    assert centres["plot(21).laz"] == pytest.approx((0.0, -2.2), abs=1e-3)


def test_far_from_origin_clouds_survive_the_round_trip(tmp_path):
    """A survey coordinate far from zero must not overflow the LAS integer store.

    LAS keeps coordinates as (value - offset) / scale in an int32; with the offset left at zero
    that caps out around 214 m at this precision, and real scan coordinates sit far beyond it.
    """
    src = tmp_path / "src"
    src.mkdir()
    write_pcd(src / "1.pcd", [(0.0, 218_000.0, 0.0, 1.0), (1000.0, 219_000.0, 1500.0, 1.0)])

    result = import_plots(list_sources(src), tmp_path / "out", _FMT, scale=0.001)

    assert result.n_written == 1 and not result.skipped
    las = las_io.read_cloud(result.written[0])
    assert float(np.asarray(las.y).max()) == pytest.approx(219.0, abs=1e-3)


def test_duplicate_reference_keys_are_reported(tmp_path):
    """A sheet covering two dates repeats every plot; the match must not look clean.

    lookup() takes the first matching row, so an unreported duplicate hands the plot a
    real-looking value that may belong to the other date.
    """
    import pandas as pd
    from featuregen.external import load_external

    sheet = tmp_path / "gt.csv"
    pd.DataFrame({
        "runNo": [1, 1], "rangeNo": [12, 12],
        "stage": ["Z31", "Z65"], "biomass": [10.0, 99.0],
    }).to_csv(sheet, index=False)

    external = load_external(
        sheet, key_column="rangeNo", row_column="runNo",
        label_format=LabelFormat(start="", end=""), stride=100,
    )

    assert external.duplicate_keys == [112]


def test_reference_filter_narrows_an_ambiguous_sheet(tmp_path):
    """Filtering to one stage removes the duplicate keys that made the join ambiguous."""
    import pandas as pd
    from featuregen.external import column_values, load_external

    sheet = tmp_path / "gt.csv"
    pd.DataFrame({
        "runNo": [1, 1], "rangeNo": [12, 12],
        "stage": ["Z31", "Z65"], "biomass": [10.0, 99.0],
    }).to_csv(sheet, index=False)

    assert column_values(sheet, "stage") == ["Z31", "Z65"]

    external = load_external(
        sheet, key_column="rangeNo", row_column="runNo",
        label_format=LabelFormat(start="", end=""), stride=100,
        filter_column="stage", filter_value="Z65",
    )

    assert external.duplicate_keys == []
    assert external.frame.loc[112, "biomass"] == 99.0  # the Z65 row, not the first one


def test_reference_filter_that_matches_nothing_is_an_error(tmp_path):
    """Silently returning an empty sheet would look like 'no plots matched' instead of a typo."""
    import pandas as pd
    from featuregen.external import load_external

    sheet = tmp_path / "gt.csv"
    pd.DataFrame({"plot": [1], "stage": ["Z31"], "biomass": [1.0]}).to_csv(sheet, index=False)

    with pytest.raises(ValueError, match="Z99"):
        load_external(sheet, key_column="plot", filter_column="stage", filter_value="Z99")


# --------------------------------------------------------------------------- #
# Unit detection                                                               #
# --------------------------------------------------------------------------- #

def test_units_come_from_the_declared_crs(tmp_path):
    """A file that states its unit is believed; that declaration is the only evidence accepted."""
    import laspy
    import pyproj
    from clip import detect_units

    header = laspy.LasHeader(point_format=3, version="1.4")
    header.scales = np.array([1e-4, 1e-4, 1e-4])
    header.offsets = np.array([0.0, 0.0, 0.0])
    header.add_crs(pyproj.CRS.from_epsg(32636))  # UTM 36N, axis unit metre
    las = laspy.LasData(header)
    las.x = np.array([0.0, 1080.0]); las.y = np.array([0.0, 500.0]); las.z = np.array([0.0, 900.0])
    las_io.write_cloud(las, tmp_path / "1.laz")

    name, reason = detect_units(list_sources(tmp_path))
    assert name == "Already in metres"
    assert "declares" in reason


def test_files_that_declare_nothing_return_none(tmp_path):
    """No declaration means unknown, not metres - the caller has to ask rather than guess.

    A .pcd carries no coordinate system at all, and its size alone must not be used to infer one:
    that would rest on an assumption about what is being scanned.
    """
    from clip import detect_units

    write_pcd(tmp_path / "1.pcd", [(0.0, 0.0, 0.0, 1.0), (1080.0, 500.0, 900.0, 1.0)])

    assert detect_units(list_sources(tmp_path)) is None
