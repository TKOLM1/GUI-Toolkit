"""Unit tests for the data-augmentation core (transforms, plan, runner, manifest)."""

from __future__ import annotations

from pathlib import Path

import laspy
import numpy as np
import pandas as pd
import pytest

from augment import AugmentConfig, load_previous_output, run_augment, sample_plan
from augment.transforms import AUG_BY_KEY
from common import las_io
from common.naming import aug_number_from_name, plot_number_from_name


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #

def make_cloud(n: int = 100, *, seed: int = 0) -> laspy.LasData:
    """An in-memory point-format-3 cloud with a RelativeHeight extra dim and fine scale.

    The scale is set fine (1e-5 m) so LAS integer-quantisation of z does not blur the
    'z and RelativeHeight shift by the same delta' check.
    """
    rng = np.random.default_rng(seed)
    header = laspy.LasHeader(point_format=3, version="1.2")
    header.add_extra_dim(laspy.ExtraBytesParams(name="RelativeHeight", type=np.float64))
    header.scales = np.array([1e-5, 1e-5, 1e-5])
    header.offsets = np.array([0.0, 0.0, 0.0])
    las = laspy.LasData(header)
    las.x = rng.uniform(0, 10, n)
    las.y = rng.uniform(0, 10, n)
    z = rng.uniform(0, 5, n)
    las.z = z
    las.RelativeHeight = z.copy()                       # RelativeHeight tracks z here
    las.classification = rng.integers(1, 3, n).astype(np.uint8)  # mix of 1 (veg) and 2 (ground)
    las.red = (rng.integers(0, 65535, n)).astype(np.uint16)
    las.green = (rng.integers(0, 65535, n)).astype(np.uint16)
    las.blue = (rng.integers(0, 65535, n)).astype(np.uint16)
    return las


def apply_one(key: str, params: dict, las: laspy.LasData, *, seed: int = 1) -> laspy.LasData:
    """Copy the cloud and apply a single transform, returning the mutated copy."""
    out = las_io.copy_cloud(las)
    AUG_BY_KEY[key].apply(out, params, np.random.default_rng(seed), "RelativeHeight")
    return out


# --------------------------------------------------------------------------- #
# Geometric transforms                                                        #
# --------------------------------------------------------------------------- #

def test_rotation_preserves_heights_classification_and_count():
    las = make_cloud()
    out = apply_one("rotation_z", {"angle_deg": 90.0}, las)
    assert len(out.points) == len(las.points)
    np.testing.assert_allclose(np.asarray(out.z), np.asarray(las.z))
    np.testing.assert_allclose(np.asarray(out.RelativeHeight), np.asarray(las.RelativeHeight))
    np.testing.assert_array_equal(np.asarray(out.classification), np.asarray(las.classification))
    # x/y actually moved.
    assert not np.allclose(np.asarray(out.x), np.asarray(las.x))


def test_rotation_360_is_identity_in_xy():
    las = make_cloud()
    out = apply_one("rotation_z", {"angle_deg": 360.0}, las)
    np.testing.assert_allclose(np.asarray(out.x), np.asarray(las.x), atol=1e-3)
    np.testing.assert_allclose(np.asarray(out.y), np.asarray(las.y), atol=1e-3)


def test_flip_mirrors_one_axis_about_centroid():
    las = make_cloud()
    out = apply_one("flip_h", {"axis": "x"}, las)
    x0 = np.asarray(las.x)
    np.testing.assert_allclose(np.asarray(out.x), 2 * x0.mean() - x0, atol=1e-3)
    np.testing.assert_allclose(np.asarray(out.y), np.asarray(las.y), atol=1e-3)


# --------------------------------------------------------------------------- #
# Noise transforms                                                            #
# --------------------------------------------------------------------------- #

def test_jitter_z_shifts_z_and_normheight_by_the_same_delta():
    las = make_cloud()
    out = apply_one("jitter_z", {"sigma_m": 0.02}, las)
    dz = np.asarray(out.z) - np.asarray(las.z)
    dh = np.asarray(out.RelativeHeight) - np.asarray(las.RelativeHeight)
    # The SAME per-point delta was added to both (within z's 1e-5 storage scale).
    np.testing.assert_allclose(dz, dh, atol=1e-4)
    assert np.std(dz) > 0  # noise was actually applied


def test_jitter_xy_leaves_heights_untouched():
    las = make_cloud()
    out = apply_one("jitter_xy", {"sigma_m": 0.02}, las)
    np.testing.assert_allclose(np.asarray(out.z), np.asarray(las.z))
    np.testing.assert_allclose(np.asarray(out.RelativeHeight), np.asarray(las.RelativeHeight))
    assert not np.allclose(np.asarray(out.x), np.asarray(las.x))


# --------------------------------------------------------------------------- #
# Sampling transforms                                                         #
# --------------------------------------------------------------------------- #

def test_dropout_reduces_count_by_fraction():
    las = make_cloud(n=100)
    out = apply_one("dropout", {"fraction": 0.3}, las)
    assert len(out.points) == 70  # round(100 * 0.7)
    # All dimensions still line up (no ragged record).
    assert len(np.asarray(out.RelativeHeight)) == 70


def test_bootstrap_keeps_count_full_fraction():
    las = make_cloud(n=80)
    out = apply_one("bootstrap", {"n_fraction": 1.0}, las)
    assert len(out.points) == 80


# --------------------------------------------------------------------------- #
# Plan sampling                                                               #
# --------------------------------------------------------------------------- #

def test_sample_plan_respects_method_bounds_and_selection():
    config = AugmentConfig(
        selected=["rotation_z", "jitter_z", "dropout"],
        ranges={
            "rotation_z": {"angle_deg": (0.0, 360.0)},
            "jitter_z": {"sigma_m": (0.0, 0.02)},
            "dropout": {"fraction": (0.0, 0.3)},
        },
        min_methods=2,
        max_methods=2,
    )
    rng = np.random.default_rng(0)
    for _ in range(50):
        plan = sample_plan(config, rng)
        keys = [k for k, _ in plan]
        assert len(keys) == 2
        assert len(set(keys)) == 2                     # distinct
        assert all(k in config.selected for k in keys)
    # A drawn angle is inside the configured range.
    angle_params = [prm for _ in range(50)
                    for k, prm in sample_plan(config, rng) if k == "rotation_z"]
    for params in angle_params:
        assert 0.0 <= params["angle_deg"] <= 360.0


# --------------------------------------------------------------------------- #
# Runner + manifest + previous-output import                                  #
# --------------------------------------------------------------------------- #

def _write_named_cloud(path: Path, n: int = 50) -> None:
    las_io.write_cloud(make_cloud(n=n), path)


def test_run_augment_writes_augmented_and_manifest(tmp_path):
    # Input and output share one folder (the merged plots/ layout): the originals are already there
    # and are not re-copied — only the augmented copies are written.
    plots_dir = tmp_path / "plots"
    plots_dir.mkdir()
    for plot in (429, 430):
        _write_named_cloud(plots_dir / f"field_plot({plot}).laz")

    config = AugmentConfig(
        selected=["rotation_z", "jitter_z"],
        ranges={"rotation_z": {"angle_deg": (0.0, 360.0)}, "jitter_z": {"sigma_m": (0.0, 0.02)}},
        min_methods=1,
        max_methods=2,
        n_per_sample=3,
        seed=42,
    )
    originals = sorted(p for p in plots_dir.glob("*.laz") if aug_number_from_name(p.name) is None)
    result = run_augment(originals, plots_dir, config)

    assert result.n_augmented == 6                      # 2 files x 3 samples
    assert not result.failed
    assert result.manifest_path.exists()

    # No verbatim copies are made; the originals stay and the 6 augmented join them in one folder.
    all_laz = sorted(plots_dir.glob("*.laz"))
    aug_files = [p for p in all_laz if aug_number_from_name(p.name) is not None]
    assert len(all_laz) == 8                             # 2 originals + 6 augmented
    assert len(aug_files) == 6

    # plot(N) preserved on every file; aug(N) restarts at 1 per plot (so 1..3 appears twice).
    aug_ids = [aug_number_from_name(p.name) for p in result.augmented]
    assert all(plot_number_from_name(p.name) in (429, 430) for p in all_laz)
    assert sorted(aug_ids) == [1, 1, 2, 2, 3, 3]

    # Manifest (CSV, '#'-comment provenance header) lists every file: 2 originals (aug_id 0) + 6.
    manifest = pd.read_csv(result.manifest_path, comment="#")
    assert len(manifest) == 8
    assert set(manifest["plot"]) == {429, 430}
    originals_rows = manifest[manifest["aug_id"] == 0]
    assert len(originals_rows) == 2
    assert (originals_rows["n_methods"] == 0).all()
    assert set(aug_files) <= {plots_dir / n for n in manifest["output_file"]}


def test_load_previous_output_lists_files(tmp_path):
    out_dir = tmp_path / "prev"
    out_dir.mkdir()
    _write_named_cloud(out_dir / "field_plot(429).laz")
    _write_named_cloud(out_dir / "field_plot(429)_aug(1).laz")

    prev = load_previous_output(out_dir)
    assert len(prev.files) == 2
    assert prev.manifest_path is None  # no manifest written here
