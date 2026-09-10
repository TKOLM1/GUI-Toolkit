"""Tests for the project-folder model (common.project)."""

from __future__ import annotations

from pathlib import Path

import pytest

from common.project import (
    PINNED_KEYS,
    SUBFOLDERS,
    create_project,
    is_project,
    open_project,
)


def test_create_makes_subfolders_and_config(tmp_path: Path) -> None:
    project = create_project(tmp_path, "field A")
    assert project.root == tmp_path / "field A"
    assert is_project(project.root)
    for sub in SUBFOLDERS:
        assert (project.root / sub).is_dir()
    # The convenience accessors point at those sub-folders.
    assert project.plots_dir == project.root / "plots"
    assert project.features_dir == project.root / "features"
    assert project.model_dir == project.root / "model"


def test_create_seeds_pins_and_persists(tmp_path: Path) -> None:
    cloud = tmp_path / "cloud.laz"
    cloud.write_bytes(b"")
    project = create_project(tmp_path / "p", "proj", seed_pins={"cloud": str(cloud)})
    assert project.pin("cloud") == str(cloud)
    # Re-opening restores the seeded pin from project.json.
    reopened = open_project(project.root)
    assert reopened.pin("cloud") == str(cloud)


def test_set_pin_round_trips(tmp_path: Path) -> None:
    project = create_project(tmp_path, "proj")
    project.set_pin("mask", r"C:\masks\plots.gpkg")
    assert open_project(project.root).pin("mask") == r"C:\masks\plots.gpkg"
    # Blanking clears it.
    project.set_pin("mask", "")
    assert open_project(project.root).pin("mask") == ""


def test_seed_round_trips(tmp_path: Path) -> None:
    project = create_project(tmp_path, "proj")
    assert project.seed == 0  # a fresh project starts at seed 0
    project.set_seed(12345)
    assert open_project(project.root).seed == 12345
    # The seed survives alongside the pins.
    project.set_pin("mask", r"C:\masks\plots.gpkg")
    reopened = open_project(project.root)
    assert reopened.seed == 12345
    assert reopened.pin("mask") == r"C:\masks\plots.gpkg"


def test_open_tolerates_missing_seed(tmp_path: Path) -> None:
    """A project.json from before the seed existed (no 'seed' key) opens with seed 0."""
    folder = tmp_path / "old"
    folder.mkdir()
    (folder / "project.json").write_text('{"pins": {}}', encoding="utf-8")
    assert open_project(folder).seed == 0


def test_create_existing_raises(tmp_path: Path) -> None:
    create_project(tmp_path, "dup")
    with pytest.raises(FileExistsError):
        create_project(tmp_path, "dup")


def test_unknown_keys_rejected(tmp_path: Path) -> None:
    project = create_project(tmp_path, "proj")
    with pytest.raises(ValueError):
        project.subdir("not_a_stage")
    with pytest.raises(ValueError):
        project.pin("bogus")
    with pytest.raises(ValueError):
        project.set_pin("bogus", "x")


def test_open_missing_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        open_project(tmp_path / "nope")


def test_open_tolerates_missing_or_broken_config(tmp_path: Path) -> None:
    # A plain folder with no project.json opens with empty pins.
    folder = tmp_path / "bare"
    folder.mkdir()
    project = open_project(folder)
    assert all(project.pin(k) == "" for k in PINNED_KEYS)
    # A corrupt project.json is tolerated (empty pins, no raise).
    (folder / "project.json").write_text("{ not json", encoding="utf-8")
    assert all(open_project(folder).pin(k) == "" for k in PINNED_KEYS)
