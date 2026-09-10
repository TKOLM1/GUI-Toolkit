r"""User configuration: named **presets** of defaults the GUI applies across the first four tabs.

A small text file (``presets.txt`` in the project root) holds one or more **presets**. A preset is a
named block of default values — import folders, the plot-label format, the clipping universal
string, augmentation settings, which features and reference columns to enable, and the output file
names — that pre-fill the Setup → Clipping → Augmentation → Feature tabs. The user picks the active
preset on the Setup tab; selecting one applies all of its values.

Why a ``.txt`` (not JSON): the values are mostly **raw Windows paths**, and JSON treats ``\`` as an
escape (``C:\final`` silently became a form-feed). This format takes everything after ``=`` as a
literal string, so a pasted path "just works" with no backslash doubling and no workaround.

Format
------
``# ...``                  a comment line (ignored)
``active = <name>``        (top, before any block) the preset to use this session
``[<name>] {`` ... ``}``   a preset block: the name, then ``{``, its ``key = value`` lines, and ``}``
``key = value``            a setting inside the current block; the value is literal to end-of-line

List-valued keys (``features_enabled``, ``augment_selected`` ...) are comma-separated. A blank value
means "unset / use the built-in default". Unknown keys are ignored, so the file tolerates hand-edits.

The top of the file carries an auto-generated **reference block** (between two marker lines) that
lists every feature key and augmentation method/parameter actually present in the code, so the user
can see exactly what to type. That block is regenerated from the live registries on every launch, so
adding a feature or augmentation in code automatically updates ``presets.txt``; the user's own preset
blocks below it are never touched.

Every key is optional and the file is optional: a missing/blank/corrupt file simply yields the
built-in :data:`DEFAULT_PRESET`, so the app always runs.

The Results tab's figure-export style is a separate concern (not part of a preset) and is persisted
to its own ``export_spec.json`` sidecar via :func:`load_export_spec` / :func:`save_export_spec`.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field, fields
from pathlib import Path

# presets.txt and the export-style sidecar sit next to this package's parent (the project root).
ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "presets.txt"
EXPORT_SPEC_PATH = ROOT / "export_spec.json"

# Per-augmentation-method parameter ranges: {method_key: {param_name: (lo, hi)}}.
Ranges = dict[str, dict[str, tuple[float, float]]]


@dataclass
class Preset:
    """One named set of defaults applied across the first four tabs (blank = "use built-in default").

    Folder keys seed the matching "Browse…" dialogs; the label format and output-name keys feed the
    pages directly. ``augment_*`` mirror :class:`augment.AugmentConfig`; ``features_*`` and the
    ``reference_features_*`` lists pre-tick the feature / reference-column checkboxes.
    """

    name: str = "default"

    # -- import folders (where each Browse dialog opens) ------------------- #
    cloud_dir: str = ""          # whole-field .las/.laz clouds (clip)
    mask_dir: str = ""           # .gpkg plot masks (clip)
    import_clipped_dir: str = ""  # folder of already-clipped plots (clip "import" button)
    reference_dir: str = ""      # reference table (.csv/.xlsx) for feature generation

    # -- the actual inputs, pre-filled into the Import tab's fields -------- #
    # Unlike the ``*_dir`` keys above (which only say where a Browse dialog opens), these name the
    # files themselves. A relative path is resolved against the project root, so a preset that
    # ships with the repo keeps working on any machine; an absolute path is used as-is. A path
    # that does not exist is ignored rather than pre-filled, so a preset written for another
    # checkout never puts a dead path in front of the user.
    cloud_file: str = ""         # whole-field cloud (masks route)
    mask_file: str = ""          # .gpkg of plot masks (masks route)
    import_folder: str = ""      # folder of per-plot files (folder route)
    reference_file: str = ""     # the reference sheet

    # -- plot-label format (how labels are read from external sources) ----- #
    label_start: str = "plot("   # text before the plot (column) number
    label_end: str = ")"         # text after the plot (column) number
    label_field: str = ""        # default mask attribute column to use as the label

    # Plots identified by a grid position rather than one id: the row number's own separators,
    # and the switch that turns the pair on. See common.naming.PlotKey for why a pair is needed
    # (a dataset like SGCBP repeats every column number across rows).
    use_row: bool = False
    label_row_start: str = ""    # text before the row number
    label_row_end: str = "-"     # text after the row number

    # -- import (folder of per-plot files) --------------------------------- #
    # Must name one of clip.importer.UNIT_SCALES; blank = "Already in metres".
    source_units: str = ""
    grid_enabled: bool = False   # spread the plots onto a grid on import
    grid_columns: int = 10       # plots per row, when there is no row number to use
    grid_padding: float = 10.0   # gap between plots, as a percentage of the largest plot

    # -- reference sheet (chosen on the Import tab) ------------------------ #
    reference_key_column: str = ""     # column holding the plot (column) number
    reference_row_column: str = ""     # column holding the row number, when plots are paired
    # Keep only the rows where ``reference_filter_column`` equals ``reference_filter_value``.
    # A sheet covering several scan dates repeats every plot, which makes the key ambiguous;
    # filtering to one date is what makes the join unique.
    reference_filter_column: str = ""
    reference_filter_value: str = ""

    # -- clipping ---------------------------------------------------------- #
    universal_string: str = ""   # text inserted into every clipped file name

    # -- augmentation ------------------------------------------------------ #
    augment_selected: list[str] = field(default_factory=list)  # enabled method keys
    augment_min_methods: int = 1
    augment_max_methods: int = 2
    augment_copies: int = 5      # augmented copies per plot (n_per_sample)
    augment_ranges: Ranges = field(default_factory=dict)       # {key: {param: (lo, hi)}}

    # -- feature generation ------------------------------------------------ #
    # The point dimension holding height-above-ground. Blank = no forced default: the Feature tab's
    # "Height channel" dropdown then just pre-selects the first channel detected in the project's
    # plots. Set it to e.g. ``RelativeHeight`` to pre-select that channel whenever it is present.
    height_channel: str = ""
    # ``*_enabled`` is a whitelist (only these on); when it is blank ``*_disabled`` acts as a
    # blacklist (all on except these). If both are blank, everything is on.
    features_enabled: list[str] = field(default_factory=list)   # feature keys to tick ([] = all on)
    features_disabled: list[str] = field(default_factory=list)  # feature keys to untick (only if enabled blank)
    reference_features_enabled: list[str] = field(default_factory=list)   # reference cols to use
    reference_features_disabled: list[str] = field(default_factory=list)  # reference cols to skip (only if enabled blank)
    reference_features_split: list[str] = field(default_factory=list)     # reference cols to one-hot

    # -- output file names (no extension; .csv is added) ------------------- #
    feature_table_name: str = "features"
    targets_table_name: str = "target"
    augmentation_table_name: str = "Data augmentation"

    # ------------------------------------------------------------------ #
    def start_dir(self, kind: str) -> str:
        """The folder a Browse dialog for ``kind`` should open in, if it still exists.

        ``kind`` is one of ``"cloud"``, ``"mask"``, ``"clipped"``, ``"reference"``; an unset or
        stale path returns ``""`` (the OS default).
        """
        value = {
            "cloud": self.cloud_dir,
            "mask": self.mask_dir,
            "clipped": self.import_clipped_dir,
            "reference": self.reference_dir,
        }.get(kind, "")
        return value if value and Path(value).is_dir() else ""

    def input_path(self, key: str) -> str:
        """The absolute path for one of the ``*_file`` / ``import_folder`` keys, or ``""``.

        A relative value is taken against the project root (:data:`ROOT`), so the presets shipped
        with the repo point at ``Data/...`` and work from any checkout. A value naming something
        that does not exist returns ``""``: a preset written elsewhere should quietly not pre-fill
        rather than plant a dead path in the field.
        """
        raw = str(getattr(self, key, "") or "").strip().strip('"').strip("'")
        if not raw:
            return ""
        path = Path(raw)
        if not path.is_absolute():
            path = ROOT / path
        return str(path) if path.exists() else ""

    def resolve_enabled(self, universe: list[str], enabled: list[str], disabled: list[str]) -> list[str]:
        """Resolve an enabled-whitelist / disabled-blacklist pair against ``universe`` (order kept).

        ``enabled`` (a whitelist) wins when non-empty: only its members that exist in ``universe``
        are on. When ``enabled`` is blank, every name is on *except* those in ``disabled``. With both
        blank, the whole universe is on. Names not in ``universe`` are ignored.
        """
        if enabled:
            wanted = set(enabled)
            return [n for n in universe if n in wanted]
        skip = set(disabled)
        return [n for n in universe if n not in skip]

    def enabled_features(self, universe: list[str]) -> list[str]:
        """The feature keys to turn on, applying ``features_enabled`` / ``features_disabled``."""
        return self.resolve_enabled(universe, self.features_enabled, self.features_disabled)

    def enabled_reference(self, universe: list[str]) -> list[str]:
        """The reference columns to turn on, applying the ``reference_features_*`` lists.

        Unlike features (which default all-on), reference columns default all-**off**: with both
        lists blank the user opts in per column in the GUI, so nothing is pre-ticked. A non-empty
        ``enabled`` whitelist or ``disabled`` blacklist switches to the normal resolve behaviour.
        """
        if not self.reference_features_enabled and not self.reference_features_disabled:
            return []
        return self.resolve_enabled(
            universe, self.reference_features_enabled, self.reference_features_disabled
        )


# The built-in default preset, also the template written on first run (with documentation comments).
DEFAULT_PRESET = Preset(name="default")
DEFAULT_PRESET_NAME = "default"

# Keys whose value is a comma-separated list (everything else is a plain scalar string / number).
_LIST_KEYS = {
    "augment_selected",
    "features_enabled",
    "features_disabled",
    "reference_features_enabled",
    "reference_features_disabled",
    "reference_features_split",
}
_INT_KEYS = {"augment_min_methods", "augment_max_methods", "augment_copies", "grid_columns"}
_FLOAT_KEYS = {"grid_padding"}
_BOOL_KEYS = {"use_row", "grid_enabled"}
# Spellings accepted for a boolean key; anything else (including a blank) reads as False.
_TRUE_WORDS = {"true", "yes", "on", "1"}


def _split_list(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def _parse_range_key(key: str, value: str, ranges: Ranges) -> None:
    """Parse ``augment_range.<method>.<param> = lo, hi`` into ``ranges`` (best-effort)."""
    rest = key[len("augment_range."):]
    method, _, param = rest.partition(".")
    if not method or not param:
        return
    parts = _split_list(value)
    if len(parts) != 2:
        return
    try:
        ranges.setdefault(method, {})[param] = (float(parts[0]), float(parts[1]))
    except ValueError:
        return


def _apply_setting(preset: Preset, key: str, value: str) -> None:
    """Set one ``key = value`` on ``preset`` (tolerant: unknown keys and bad numbers are ignored)."""
    if key.startswith("augment_range."):
        _parse_range_key(key, value, preset.augment_ranges)
        return
    if not hasattr(preset, key):
        return
    if key in _LIST_KEYS:
        setattr(preset, key, _split_list(value))
    elif key in _BOOL_KEYS:
        setattr(preset, key, value.strip().lower() in _TRUE_WORDS)
    elif key in _INT_KEYS:
        try:
            setattr(preset, key, int(value))
        except ValueError:
            pass
    elif key in _FLOAT_KEYS:
        try:
            setattr(preset, key, float(value.replace(",", ".")))
        except ValueError:
            pass
    else:
        setattr(preset, key, value)


def parse_config(text: str) -> tuple[dict[str, Preset], str]:
    """Parse ``presets.txt`` text into ``(presets_by_name, active_name)``; never raises.

    A line-based reader: ``# ...`` comments and blanks are skipped, ``active = name`` (before any
    block) records the active preset, ``[name] {`` opens a block, ``}`` closes it, and ``key = value``
    sets a literal value on the current block. The braces are optional (a bare ``[name]`` also opens a
    block) so older brace-less files keep working. Always returns at least the built-in ``default``.
    """
    presets: dict[str, Preset] = {}
    active = ""
    current: Preset | None = None
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line == "}":                       # close the current block
            current = None
            continue
        if line.startswith("["):              # "[name]" or "[name] {"
            inner = line[1:]
            inner = inner.rstrip("{").strip()  # drop a trailing "{"
            name = inner[:-1].strip() if inner.endswith("]") else inner.strip("]").strip()
            name = name or "preset"
            current = Preset(name=name)
            presets[name] = current
            continue
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if current is None:
            if key == "active":
                active = value
            continue  # settings outside a block (other than `active`) are ignored
        _apply_setting(current, key, value)
    if not presets:
        presets[DEFAULT_PRESET_NAME] = Preset(name=DEFAULT_PRESET_NAME)
    if active not in presets:
        active = DEFAULT_PRESET_NAME if DEFAULT_PRESET_NAME in presets else next(iter(presets))
    return presets, active


# --------------------------------------------------------------------------- #
# Auto-generated reference block (lists the code's real features + augmentations) #
# --------------------------------------------------------------------------- #
_REF_START = "# >>> BEGIN AUTO-GENERATED REFERENCE (edited automatically — do not hand-edit) >>>"
_REF_END = "# <<< END AUTO-GENERATED REFERENCE <<<"


def _reference_block() -> str:
    """Build the ``#``-comment reference listing from the live feature + augmentation registries.

    Imported lazily so the heavy registries (numpy/scipy/laspy) are only pulled when the block is
    (re)generated, not on every ``import common.config``. Falls back to a short note if an import
    fails, so config loading never breaks because of a registry error.
    """
    lines = [_REF_START,
             "# This block is rebuilt from the code on every launch — it always reflects what is",
             "# actually available. Use the keys below in the list-valued settings of a preset.",
             "#"]
    try:
        from featuregen.features import FEATURES
        lines.append("# FEATURES  (use these keys in features_enabled / features_disabled):")
        cls = grp = None
        for f in FEATURES:
            if f.cls != cls:
                cls = f.cls
                grp = None
                lines.append(f"#   Class: {cls}")
            if f.group != grp:
                grp = f.group
                lines.append(f"#     Type: {grp}")
            lines.append(f"#       {f.key:<20} {f.label}")
    except Exception as exc:  # noqa: BLE001 - never break config loading on a registry import error
        lines.append(f"#   (feature registry unavailable: {exc})")

    lines.append("#")
    try:
        from augment.transforms import AUGMENTATIONS
        lines.append("# AUGMENTATION METHODS  (use these keys in augment_selected; ranges as")
        lines.append("#   augment_range.<method>.<param> = <min>, <max>):")
        for a in AUGMENTATIONS:
            params = ", ".join(p.name for p in a.params) or "(no parameters)"
            lines.append(f"#   {a.key:<14} {a.label}  [{a.group}]  params: {params}")
            for p in a.params:
                lines.append(
                    f"#       range.{a.key}.{p.name} default {p.default_min} .. {p.default_max}  ({p.label})"
                )
    except Exception as exc:  # noqa: BLE001
        lines.append(f"#   (augmentation registry unavailable: {exc})")

    lines.append(_REF_END)
    return "\n".join(lines)


# The documented body written under the reference block on first run: one fully-spelled-out preset.
_TEMPLATE_BODY = """\
# ====================================================================
# Pipeline configuration — presets of defaults for the first four tabs.
#
# A preset is a NAMED BLOCK:   [name] {  ...settings...  }
# It pre-fills the Setup -> Clipping -> Augmentation -> Feature tabs.
# Pick the active one on the Setup tab (or set `active` below).
#
# Paths are LITERAL: paste a raw Windows path with single backslashes,
# e.g.  cloud_dir = C:\\Data\\clouds   (no doubling needed).
# Blank value = "use the built-in default". `#` starts a comment line.
# List values are comma-separated; see the reference block above for the
# exact feature keys and augmentation methods/parameters you can use.
# Copy the whole [default] { ... } block to make your own preset.
# ====================================================================

active = default

[default] {

    # -- Import folders (where each "Browse..." dialog opens) --------
    cloud_dir          =
    mask_dir           =
    import_clipped_dir =
    reference_dir      =

    # -- Plot-label format: how a plot NUMBER is spelled in external
    #    data (imported filenames and the reference-table key column).
    #    The app always WRITES files as plot(N); these only affect what
    #    it READS.
    label_start = plot(
    label_end   = )
    # Default mask attribute column to read the label from (blank = first):
    label_field =

    # -- Clipping ---------------------------------------------------
    # Text inserted into every clipped file name, right after plot(N):
    universal_string =

    # -- Augmentation -----------------------------------------------
    # Enabled method keys (comma-separated; blank = none pre-selected):
    augment_selected    =
    augment_min_methods = 1
    augment_max_methods = 2
    # Augmented copies per plot (the original is always kept as well):
    augment_copies      = 5
    # Per-method parameter ranges, one line each, e.g.:
    #   augment_range.jitter_z.sigma_m = 0.0, 0.02

    # -- Feature generation -----------------------------------------
    # Point dimension holding height-above-ground. Blank = no forced
    # default: the "Height channel" dropdown pre-selects the first
    # channel detected in the plots. Set e.g. RelativeHeight to
    # pre-select that channel when present.
    height_channel =
    # features_enabled is a whitelist (only these on). When it is blank,
    # features_disabled is a blacklist (all on except these).
    features_enabled  =
    features_disabled =
    # Imported reference columns: same enabled/disabled rule, plus which
    # to one-hot ("split").
    reference_features_enabled  =
    reference_features_disabled =
    reference_features_split    =

    # -- Output file names (no extension; written as .csv) ----------
    feature_table_name      = features
    targets_table_name      = target
    augmentation_table_name = Data augmentation
}
"""


def _template_text() -> str:
    """The full first-run file: the auto-generated reference block then the documented body."""
    return _reference_block() + "\n\n" + _TEMPLATE_BODY


def _refresh_reference_block(text: str) -> str:
    """Return ``text`` with its reference block replaced by a freshly generated one (in place).

    If the markers are present the block between them is swapped; otherwise a fresh block is
    prepended. The user's preset blocks (below the reference) are never touched.
    """
    block = _reference_block()
    start = text.find(_REF_START)
    end = text.find(_REF_END)
    if start != -1 and end != -1 and end > start:
        end += len(_REF_END)
        return text[:start] + block + text[end:]
    return block + "\n\n" + text


def load_presets() -> tuple[dict[str, Preset], str]:
    """Load all presets from ``presets.txt``; write the template on first run; never raises.

    On an existing file the auto-generated reference block is refreshed in place (so it always
    mirrors the code) without disturbing the user's preset blocks.
    """
    if not CONFIG_PATH.exists():
        try:
            CONFIG_PATH.write_text(_template_text(), encoding="utf-8")
        except OSError:
            pass
        return {DEFAULT_PRESET_NAME: Preset(name=DEFAULT_PRESET_NAME)}, DEFAULT_PRESET_NAME
    try:
        text = CONFIG_PATH.read_text(encoding="utf-8")
    except OSError:
        return {DEFAULT_PRESET_NAME: Preset(name=DEFAULT_PRESET_NAME)}, DEFAULT_PRESET_NAME
    refreshed = _refresh_reference_block(text)
    if refreshed != text:
        try:
            CONFIG_PATH.write_text(refreshed, encoding="utf-8")
        except OSError:
            pass
    return parse_config(refreshed)


def set_active_preset(name: str) -> None:
    """Persist the chosen active-preset ``name`` back into ``presets.txt`` (best-effort).

    Rewrites only the ``active = ...`` line (adding one if absent); the preset blocks are left
    untouched. Updates the in-memory :data:`ACTIVE_NAME` and copies the chosen preset's fields into
    the existing :data:`CONFIG` object *in place* (so every page that imported ``CONFIG`` by name
    sees the new values without re-importing).
    """
    global ACTIVE_NAME
    if name in PRESETS:
        ACTIVE_NAME = name
        for f in fields(Preset):
            # Deep-copy: CONFIG must never share (or alias) a stored preset's lists/dicts, or the
            # first switch would overwrite the preset it came from and later switches back would
            # find their own values already gone.
            setattr(CONFIG, f.name, copy.deepcopy(getattr(PRESETS[name], f.name)))
    try:
        text = CONFIG_PATH.read_text(encoding="utf-8") if CONFIG_PATH.exists() else _template_text()
        lines = text.splitlines()
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith("active") and "=" in stripped and not stripped.startswith("#"):
                lines[i] = f"active = {name}"
                break
        else:
            lines.insert(0, f"active = {name}")
        CONFIG_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    except OSError:
        pass


# --------------------------------------------------------------------------- #
# Export-style sidecar (Results tab figure export; not part of a preset).      #
# --------------------------------------------------------------------------- #
def load_export_spec() -> dict:
    """Read the saved figure-export style from ``export_spec.json`` ({} if missing/unreadable)."""
    if not EXPORT_SPEC_PATH.exists():
        return {}
    try:
        data = json.loads(EXPORT_SPEC_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def save_export_spec(spec: dict) -> None:
    """Persist the figure-export style ``spec`` to ``export_spec.json`` (best-effort, never raises)."""
    try:
        EXPORT_SPEC_PATH.write_text(json.dumps(dict(spec), indent=2), encoding="utf-8")
    except OSError:
        pass


# Loaded once at import; treated as static for the session (the active preset can change via the
# Setup tab, which updates CONFIG in place through set_active_preset).
PRESETS, ACTIVE_NAME = load_presets()
# A copy, never the stored preset itself: PRESETS is the pristine record of presets.txt, while
# CONFIG is the live object every page holds a reference to and set_active_preset() mutates.
CONFIG = copy.deepcopy(PRESETS[ACTIVE_NAME])
