"""The project folder: one folder that owns every input and output of a pipeline run.

Instead of choosing import and output locations on every tab, the user picks (or creates) a
single **project folder** on the first tab. Everything else is derived from it:

* the pipeline stages write into fixed sub-folders. Clipping and augmentation share a single
  ``plots/`` folder (the clipped ``plot(N).laz`` originals *and* their ``plot(N)_aug(k).laz``
  copies live side by side, with no duplicated original), then ``features/`` -> ``model/``; each
  stage reads its input from the previous one's folder, so the stages chain automatically;
* the few genuinely *external* inputs that don't come from a previous stage - the whole-field
  cloud and ``.gpkg`` masks (clip) and the reference table (feature generation) - are *pinned* in
  a small ``project.json`` inside the folder, so re-opening a project restores them.

A new project seeds its pinned inputs from the global :mod:`common.config` defaults; opening an
existing project restores whatever was saved last. Selecting a different file in a tab updates the
project (and is persisted), so a project always reflects the last-used inputs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

# The fixed per-stage sub-folder names. Order matters: each stage reads the previous one's output.
# Clipping and augmentation share ``plots/`` (originals + augmented copies in one place).
SUBFOLDERS = ("plots", "features", "model")

# The pinned external-input keys stored in project.json. Mostly paths to specific files;
# "reference_key" is the reference sheet's plot-number column, pinned alongside the sheet
# itself so re-opening a project restores the whole link, not just the file.
PINNED_KEYS = (
    "cloud", "mask", "reference", "reference_key", "reference_row", "import_folder",
)

# The results-map grid-layout keys stored in project.json (dimensions + fill order). These match
# the arguments of ml.plot_layout.build_layout; an empty dict means "use the built-in default".
LAYOUT_KEYS = ("rows", "cols", "base", "start", "major", "snake")

PROJECT_FILE = "project.json"


@dataclass
class Project:
    """An open project: its root folder plus the pinned external input files.

    The pipeline sub-folders are always ``root/<name>`` for ``name`` in :data:`SUBFOLDERS`;
    :meth:`subdir` returns them (creating on demand). The pinned inputs (:data:`PINNED_KEYS`)
    are paths the user has chosen for the cloud / masks / remap / reference; they are saved to and
    loaded from ``project.json`` so a re-opened project restores them.
    """

    root: Path
    pins: dict[str, str] = field(default_factory=dict)
    # The one project-wide random seed: every stage (augmentation, ML/DL splits, both
    # hyperparameter optimizers) reads this so a project reproduces end-to-end. Set on the Setup
    # tab and persisted to project.json.
    seed: int = 0
    # The results-map grid layout: the field's dimensions + fill order, set in the Results tab's
    # grid view and persisted so the map draws the same grid on reopen. Empty until the user
    # saves one; the map falls back to its built-in default (ml.plot_layout.build_layout()).
    layout: dict = field(default_factory=dict)
    # The top/bottom height percentage the hand-crafted slope-angle features were generated with
    # (Config.slope_strata_pct). Persisted on feature generation so the Results viewer can draw the
    # slope planes at the *same* percentile the numbers used. Defaults to Config's default.
    slope_strata_pct: float = 5.0
    # The point dimension the features used for height-above-ground (Config.height_channel, the
    # Feature-tab "Height channel" pick). Persisted on feature generation so the Results viewer
    # draws the cloud + feature geometry from the *same* channel the numbers used. Defaults to
    # Config's default ("RelativeHeight").
    height_channel: str = "RelativeHeight"
    # Custom display names for the Results "Model comparison" sub-tab, keyed by bundle *filename*
    # (so they survive the project folder moving). Empty until the user renames a model; an absent
    # key means "fall back to the bundle's auto label". Persisted to project.json.
    model_names: dict[str, str] = field(default_factory=dict)

    # -- folders ---------------------------------------------------------- #
    @property
    def name(self) -> str:
        return self.root.name

    def subdir(self, name: str, *, create: bool = True) -> Path:
        """The pipeline sub-folder ``root/<name>`` (created on demand by default)."""
        if name not in SUBFOLDERS:
            raise ValueError(f"unknown pipeline sub-folder: {name!r}")
        path = self.root / name
        if create:
            path.mkdir(parents=True, exist_ok=True)
        return path

    # Convenience accessors used by the pages.
    @property
    def plots_dir(self) -> Path:
        """The shared clip+augment folder: ``plot(N).laz`` originals and ``plot(N)_aug(k).laz`` copies."""
        return self.subdir("plots")

    @property
    def features_dir(self) -> Path:
        return self.subdir("features")

    @property
    def model_dir(self) -> Path:
        return self.subdir("model")

    # -- pinned inputs ---------------------------------------------------- #
    def pin(self, key: str) -> str:
        """The pinned path for ``key`` (one of :data:`PINNED_KEYS`), or ``""`` if unset."""
        if key not in PINNED_KEYS:
            raise ValueError(f"unknown pinned input: {key!r}")
        return self.pins.get(key, "")

    def set_pin(self, key: str, value: str) -> None:
        """Pin ``value`` for ``key`` and persist the project (a blank value clears the pin)."""
        if key not in PINNED_KEYS:
            raise ValueError(f"unknown pinned input: {key!r}")
        self.pins[key] = value or ""
        self.save()

    def set_seed(self, value: int) -> None:
        """Set the project-wide random seed and persist the project."""
        self.seed = int(value)
        self.save()

    def set_slope_strata_pct(self, value: float) -> None:
        """Record the slope-strata % the features were generated with and persist the project."""
        self.slope_strata_pct = float(value)
        self.save()

    def set_height_channel(self, value: str) -> None:
        """Record the height channel the features were generated with and persist the project."""
        value = (value or "").strip()
        if value:
            self.height_channel = value
        self.save()

    def set_model_name(self, bundle_filename: str, name: str) -> None:
        """Set (or clear) a model's custom display name in the comparison sub-tab and persist.

        ``bundle_filename`` is the bundle file's name (``Path.name``); a blank ``name`` clears the
        override so the bundle falls back to its auto-generated label.
        """
        name = name.strip()
        if name:
            self.model_names[bundle_filename] = name
        else:
            self.model_names.pop(bundle_filename, None)
        self.save()

    def set_layout(self, layout: dict | None) -> None:
        """Set (or clear) the results-map grid layout and persist the project.

        ``layout`` holds the :data:`LAYOUT_KEYS` (matching ``build_layout``'s arguments); a
        falsy value clears it so the map reverts to its built-in default.
        """
        self.layout = {k: layout[k] for k in LAYOUT_KEYS if k in layout} if layout else {}
        self.save()

    # -- persistence ------------------------------------------------------ #
    @property
    def config_path(self) -> Path:
        return self.root / PROJECT_FILE

    def save(self) -> None:
        """Write ``project.json`` (seed + pinned inputs + grid layout; folders are derived).

        Never raises.
        """
        try:
            data = {
                "seed": int(self.seed),
                "pins": {k: self.pins.get(k, "") for k in PINNED_KEYS},
                "layout": {k: self.layout[k] for k in LAYOUT_KEYS if k in self.layout},
                "slope_strata_pct": float(self.slope_strata_pct),
                "height_channel": str(self.height_channel),
                "model_names": {str(k): str(v) for k, v in self.model_names.items()},
            }
            self.config_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except OSError:
            pass


def _read_pins(root: Path) -> dict[str, str]:
    """Read the pinned inputs from ``root/project.json``; tolerate a missing/broken file."""
    path = root / PROJECT_FILE
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    raw = data.get("pins") or {}
    return {k: str(raw.get(k, "") or "") for k in PINNED_KEYS}


def _read_seed(root: Path) -> int:
    """Read the project-wide seed from ``root/project.json``; tolerate a missing/broken file."""
    path = root / PROJECT_FILE
    if not path.exists():
        return 0
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return int(data.get("seed", 0) or 0)
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return 0


def _read_slope_strata_pct(root: Path) -> float:
    """Read the persisted slope-strata % from ``root/project.json``; tolerate a missing/broken file."""
    path = root / PROJECT_FILE
    if not path.exists():
        return 5.0
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return float(data.get("slope_strata_pct", 5.0) or 5.0)
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return 5.0


def _read_height_channel(root: Path) -> str:
    """Read the persisted height channel from ``root/project.json``; tolerate a missing/broken file."""
    path = root / PROJECT_FILE
    if not path.exists():
        return "RelativeHeight"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return str(data.get("height_channel") or "RelativeHeight").strip() or "RelativeHeight"
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return "RelativeHeight"


def _read_layout(root: Path) -> dict:
    """Read the results-map grid layout from ``root/project.json``; tolerate a missing/broken file."""
    path = root / PROJECT_FILE
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    raw = data.get("layout") or {}
    return {k: raw[k] for k in LAYOUT_KEYS if k in raw}


def _read_model_names(root: Path) -> dict[str, str]:
    """Read the comparison-tab custom model names from ``root/project.json``; tolerate a broken file."""
    path = root / PROJECT_FILE
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    raw = data.get("model_names") or {}
    return {str(k): str(v) for k, v in raw.items() if str(v).strip()}


def open_project(root: str | Path) -> Project:
    """Open an existing project folder, restoring seed + pinned inputs + layout from ``project.json``."""
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"Project folder does not exist: {root}")
    return Project(
        root=root,
        pins=_read_pins(root),
        seed=_read_seed(root),
        layout=_read_layout(root),
        slope_strata_pct=_read_slope_strata_pct(root),
        height_channel=_read_height_channel(root),
        model_names=_read_model_names(root),
    )


def create_project(parent: str | Path, name: str, *, seed_pins: dict[str, str] | None = None) -> Project:
    """Create ``parent/name`` (with its pipeline sub-folders) and seed its pinned inputs.

    ``seed_pins`` (usually the global config defaults) pre-fills the pinned external inputs for a
    fresh project. Raises ``FileExistsError`` if a non-empty project of that name already exists.
    """
    name = name.strip()
    if not name:
        raise ValueError("Project name cannot be empty.")
    root = Path(parent) / name
    if (root / PROJECT_FILE).exists():
        raise FileExistsError(f"A project already exists at: {root}")
    root.mkdir(parents=True, exist_ok=True)
    project = Project(root=root, pins={k: (seed_pins or {}).get(k, "") for k in PINNED_KEYS})
    for sub in SUBFOLDERS:
        project.subdir(sub)  # create the empty stage folders up front
    project.save()
    return project


def create_project_here(root: str | Path, *, seed_pins: dict[str, str] | None = None) -> Project:
    """Turn the **already chosen** folder ``root`` itself into a project (no extra sub-folder).

    The Setup tab's "Create new project" flow picks one folder and uses it directly, so the user is
    never asked for a name after already naming/picking a folder. The project's name is the folder's
    own name. Raises ``FileExistsError`` if that folder already holds a project.
    """
    root = Path(root)
    if (root / PROJECT_FILE).exists():
        raise FileExistsError(f"A project already exists at: {root}")
    root.mkdir(parents=True, exist_ok=True)
    project = Project(root=root, pins={k: (seed_pins or {}).get(k, "") for k in PINNED_KEYS})
    for sub in SUBFOLDERS:
        project.subdir(sub)  # create the empty stage folders up front
    project.save()
    return project


def is_project(root: str | Path) -> bool:
    """True if ``root`` looks like a project folder (has a ``project.json``)."""
    return (Path(root) / PROJECT_FILE).exists()
