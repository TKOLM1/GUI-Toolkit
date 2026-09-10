"""Unit tests for the clipping core (masks reader + exact polygon clip)."""

from __future__ import annotations

import laspy
import numpy as np
from shapely.geometry import Polygon

from clip import PlotMask, clip_to_masks, list_mask_fields, load_masks
from common import las_io
from common.naming import LabelFormat


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #

def make_cloud(coords: list[tuple[float, float]]) -> laspy.LasData:
    """A point-format-3 cloud (RGB) with a RelativeHeight extra dim at the given (x, y)."""
    n = len(coords)
    header = laspy.LasHeader(point_format=3, version="1.2")
    header.add_extra_dim(laspy.ExtraBytesParams(name="RelativeHeight", type=np.float64))
    header.scales = np.array([1e-4, 1e-4, 1e-4])
    header.offsets = np.array([0.0, 0.0, 0.0])
    las = laspy.LasData(header)
    las.x = np.array([c[0] for c in coords], dtype=np.float64)
    las.y = np.array([c[1] for c in coords], dtype=np.float64)
    las.z = np.zeros(n, dtype=np.float64)
    las.RelativeHeight = np.arange(n, dtype=np.float64)
    las.classification = np.ones(n, dtype=np.uint8)
    las.red = np.full(n, 100, dtype=np.uint16)
    las.green = np.full(n, 200, dtype=np.uint16)
    las.blue = np.full(n, 300, dtype=np.uint16)
    return las


_SQUARE = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])


# --------------------------------------------------------------------------- #
# clip_to_masks                                                               #
# --------------------------------------------------------------------------- #

def test_clip_keeps_only_inside_points_and_all_dims(tmp_path):
    cloud = tmp_path / "field.laz"
    las_io.write_cloud(make_cloud([(1, 1), (9, 9), (20, 20)]), cloud)

    result = clip_to_masks(cloud, [PlotMask(429, _SQUARE)], tmp_path / "out",
                           universal_string="_siteA")

    assert result.n_written == 1
    out = result.written[0]
    assert out.name == "plot(429)_siteA.laz"          # universal string sits after plot(N)

    sub = las_io.read_cloud(out)
    assert len(sub.points) == 2                        # the (20, 20) point is outside
    assert "RelativeHeight" in set(sub.point_format.dimension_names)  # every dim carried
    assert sorted(np.asarray(sub.x).round().astype(int)) == [1, 9]


def test_write_cloud_strips_copc_header(tmp_path):
    """A COPC source (which laspy refuses to write) is rebuilt with a clean header on write."""
    las = make_cloud([(1, 1), (2, 2)])
    las.header.vlrs.append(
        laspy.VLR(user_id="copc", record_id=1, description="copc info", record_data=b"\x00" * 160)
    )
    assert las_io.is_copc(las)

    out = tmp_path / "clean.laz"
    las_io.write_cloud(las, out)  # must not raise "Writing COPC is not supported"

    back = las_io.read_cloud(out)
    assert not las_io.is_copc(back)
    assert back.header.point_count == 2
    assert "RelativeHeight" in set(back.point_format.dimension_names)  # dims preserved


def test_clip_excludes_boundary_and_outside(tmp_path):
    # (0, 5) is exactly on the left edge -> strict contains excludes it.
    cloud = tmp_path / "field.laz"
    las_io.write_cloud(make_cloud([(0, 5), (5, 5), (50, 50)]), cloud)

    result = clip_to_masks(cloud, [PlotMask(1, _SQUARE)], tmp_path / "out")
    sub = las_io.read_cloud(result.written[0])
    assert len(sub.points) == 1                        # only the interior (5, 5) point


def test_clip_empty_polygon_recorded(tmp_path):
    cloud = tmp_path / "field.laz"
    las_io.write_cloud(make_cloud([(1, 1)]), cloud)
    far = Polygon([(100, 100), (110, 100), (110, 110), (100, 110)])

    result = clip_to_masks(cloud, [PlotMask(99, far)], tmp_path / "out")
    assert result.n_written == 0
    assert result.empty == ["99"]


# --------------------------------------------------------------------------- #
# GeoPackage reader                                                           #
# --------------------------------------------------------------------------- #

def test_gpkg_round_trip(tmp_path):
    import geopandas as gpd

    gdf = gpd.GeoDataFrame(
        {"plot_id": [429, 430]},
        geometry=[_SQUARE, Polygon([(20, 20), (30, 20), (30, 30), (20, 30)])],
        crs="EPSG:32636",
    )
    path = tmp_path / "masks.gpkg"
    gdf.to_file(path, driver="GPKG")

    assert "plot_id" in list_mask_fields(path)
    masks, crs, unreadable = load_masks(path, "plot_id")
    assert {m.label for m in masks} == {429, 430}      # labels coerced to int
    assert unreadable == []
    assert "32636" in crs


def test_mask_labels_read_through_the_label_format(tmp_path):
    """A decorated mask label is reduced to the plain number the pipeline keys off.

    Without this the file is written as plot(P_429_a).laz, which the canonical plot(N) pattern
    cannot read — so it exists on disk and is invisible to every later tab.
    """
    import geopandas as gpd

    gdf = gpd.GeoDataFrame(
        {"tag": ["P_429_a", "junk"]},
        geometry=[_SQUARE, Polygon([(20, 20), (30, 20), (30, 30), (20, 30)])],
        crs="EPSG:32636",
    )
    path = tmp_path / "masks.gpkg"
    gdf.to_file(path, driver="GPKG")

    masks, _crs, unreadable = load_masks(path, "tag", LabelFormat(start="P_", end="_a"))

    assert [m.label for m in masks] == [429]
    assert unreadable == ["junk"]      # reported, not exported under an unusable name
