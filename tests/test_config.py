"""Unit tests for the preset-based user config (presets.txt)."""

from __future__ import annotations


from common.config import (
    DEFAULT_PRESET_NAME,
    Preset,
    load_presets,
    parse_config,
)


def test_parse_config_reads_brace_blocks_and_active():
    text = """\
# a comment
active = field

[default] {
    label_start = plot(
    label_end = )
}

[field] {
    cloud_dir = C:\\Data\\clouds
    label_start = P_
    label_end =
    augment_selected = jitter_z, rotate
    augment_min_methods = 2
    augment_copies = 8
    augment_range.jitter_z.sigma = 0.1, 0.5
    features_enabled = h_mean, h_max
    features_disabled = R_mean
}
"""
    presets, active = parse_config(text)
    assert active == "field"
    assert set(presets) == {"default", "field"}

    field = presets["field"]
    # Paths are literal — single backslashes are kept verbatim (the whole point of .txt).
    assert field.cloud_dir == r"C:\Data\clouds"
    assert field.label_start == "P_"
    assert field.label_end == ""  # an explicit blank value
    assert field.augment_selected == ["jitter_z", "rotate"]
    assert field.augment_min_methods == 2
    assert field.augment_copies == 8
    assert field.augment_ranges == {"jitter_z": {"sigma": (0.1, 0.5)}}
    assert field.features_enabled == ["h_mean", "h_max"]
    assert field.features_disabled == ["R_mean"]


def test_parse_config_accepts_braceless_blocks_for_back_compat():
    presets, _ = parse_config("[old]\nlabel_start = X_\n")
    assert presets["old"].label_start == "X_"


def test_parse_config_unknown_active_falls_back_to_default():
    _presets, active = parse_config("active = missing\n[default] {\n}\n")
    assert active == DEFAULT_PRESET_NAME


def test_parse_config_empty_yields_a_default_preset():
    presets, active = parse_config("")
    assert active == DEFAULT_PRESET_NAME
    assert isinstance(presets[DEFAULT_PRESET_NAME], Preset)


def test_preset_start_dir_returns_existing_folder_only(tmp_path):
    clouds = tmp_path / "clouds"
    clouds.mkdir()
    preset = Preset(cloud_dir=str(clouds), mask_dir=str(tmp_path / "gone"))
    assert preset.start_dir("cloud") == str(clouds)
    assert preset.start_dir("mask") == ""   # stale folder ignored
    assert preset.start_dir("nope") == ""   # unknown kind is empty, not an error


def test_enabled_features_whitelist_then_blacklist():
    universe = ["a", "b", "c"]
    # Whitelist wins when present (and ignores unknown names).
    assert Preset(features_enabled=["b", "zz"]).enabled_features(universe) == ["b"]
    # Blacklist applies only when the whitelist is blank.
    assert Preset(features_disabled=["b"]).enabled_features(universe) == ["a", "c"]
    # Both blank -> everything on (features default all-on).
    assert Preset().enabled_features(universe) == ["a", "b", "c"]


def test_enabled_reference_defaults_to_none_when_unspecified():
    universe = ["x", "y"]
    # Reference columns default all-OFF (the user opts in per column) when both lists are blank.
    assert Preset().enabled_reference(universe) == []
    # A whitelist or blacklist switches to the normal resolve.
    assert Preset(reference_features_enabled=["y"]).enabled_reference(universe) == ["y"]
    assert Preset(reference_features_disabled=["x"]).enabled_reference(universe) == ["y"]


def test_load_presets_writes_template_with_reference_block(tmp_path, monkeypatch):
    path = tmp_path / "presets.txt"
    monkeypatch.setattr("common.config.CONFIG_PATH", path)
    monkeypatch.setattr("common.config.EXPORT_SPEC_PATH", tmp_path / "export_spec.json")

    presets, active = load_presets()
    assert active == DEFAULT_PRESET_NAME
    assert path.exists()
    written = path.read_text(encoding="utf-8")
    # The auto-generated reference block lists real feature keys + augmentation methods from code.
    assert "BEGIN AUTO-GENERATED REFERENCE" in written
    assert "h_mean" in written            # a real feature key
    assert "jitter_z" in written          # a real augmentation method key
    assert "[default] {" in written       # blocks are brace-enclosed
    # The body parses back cleanly to a default preset with the documented output names.
    presets2, _ = parse_config(written)
    default = presets2[DEFAULT_PRESET_NAME]
    assert default.label_start == "plot("
    assert default.feature_table_name == "features"
    assert default.targets_table_name == "target"
    assert default.augmentation_table_name == "Data augmentation"


def test_load_presets_refreshes_reference_block_in_place(tmp_path, monkeypatch):
    path = tmp_path / "presets.txt"
    monkeypatch.setattr("common.config.CONFIG_PATH", path)
    monkeypatch.setattr("common.config.EXPORT_SPEC_PATH", tmp_path / "es.json")
    # A user file with a stale reference block but a real preset block below it.
    path.write_text(
        "# >>> BEGIN AUTO-GENERATED REFERENCE (edited automatically — do not hand-edit) >>>\n"
        "# stale junk\n"
        "# <<< END AUTO-GENERATED REFERENCE <<<\n"
        "\n"
        "active = mine\n"
        "[mine] {\n"
        "    cloud_dir = C:\\keepme\n"
        "}\n",
        encoding="utf-8",
    )
    presets, active = load_presets()
    assert active == "mine"
    assert presets["mine"].cloud_dir == r"C:\keepme"   # the user's block survived untouched
    refreshed = path.read_text(encoding="utf-8")
    assert "stale junk" not in refreshed               # the old block was replaced
    assert "h_mean" in refreshed                        # with a fresh, real listing


def test_load_presets_tolerates_raw_windows_backslash_paths(tmp_path, monkeypatch):
    path = tmp_path / "presets.txt"
    monkeypatch.setattr("common.config.CONFIG_PATH", path)
    monkeypatch.setattr("common.config.EXPORT_SPEC_PATH", tmp_path / "es.json")
    # Single backslashes, including a folder starting with a JSON-escape letter (\final) that broke
    # the old JSON config — here they are read literally with no doubling or corruption.
    path.write_text(
        "[default] {\n"
        r"    cloud_dir = C:\Projects\Data\final clouds" + "\n"
        r"    mask_dir = C:\temp\new masks" + "\n"
        "}\n",
        encoding="utf-8",
    )
    presets, _ = load_presets()
    assert presets["default"].cloud_dir == r"C:\Projects\Data\final clouds"
    assert presets["default"].mask_dir == r"C:\temp\new masks"


def test_relative_input_paths_resolve_against_the_project_root(tmp_path, monkeypatch):
    """A preset shipped with the repo names Data/... and must work from any checkout."""
    import common.config as config

    monkeypatch.setattr(config, "ROOT", tmp_path)
    (tmp_path / "Data").mkdir()
    (tmp_path / "Data" / "sheet.csv").write_text("a,b\n1,2\n")

    preset = config.Preset(name="p", reference_file="Data/sheet.csv")
    assert preset.input_path("reference_file") == str(tmp_path / "Data" / "sheet.csv")


def test_missing_input_paths_are_not_pre_filled(tmp_path, monkeypatch):
    """A preset written for another machine should leave the field empty, not plant a dead path."""
    import common.config as config

    monkeypatch.setattr(config, "ROOT", tmp_path)
    preset = config.Preset(name="p", reference_file="Data/nope.csv")
    assert preset.input_path("reference_file") == ""


def test_shipped_presets_resolve_their_inputs():
    """The presets in presets.txt must point at files that actually exist in this repo."""
    from pathlib import Path
    from common.config import parse_config

    presets, _ = parse_config(Path("presets.txt").read_text(encoding="utf-8"))
    expected = {
        "CYENS": ["cloud_file", "mask_file", "reference_file"],
        "SGCBP 2019-08-28": ["import_folder", "reference_file"],
        "SGCBP 2019-10-02": ["import_folder", "reference_file"],
    }
    for name, keys in expected.items():
        assert name in presets, f"{name} missing from presets.txt"
        for key in keys:
            assert presets[name].input_path(key), f"{name}.{key} does not resolve"


def test_shipped_presets_choose_their_features():
    """CYENS keeps every feature; the SGCBP scans drop the colour ones (their .pcd has no RGB)."""
    from pathlib import Path
    from common.config import parse_config
    from featuregen.features import FEATURES

    presets, _ = parse_config(Path("presets.txt").read_text(encoding="utf-8"))
    universe = [f.key for f in FEATURES]
    rgb = {"R_mean", "G_mean", "B_mean"}

    assert set(presets["CYENS"].enabled_features(universe)) == set(universe)
    for name in ("SGCBP 2019-08-28", "SGCBP 2019-10-02"):
        on = set(presets[name].enabled_features(universe))
        assert on == set(universe) - rgb
