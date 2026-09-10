"""Small reusable widgets shared by the module pages.

These were the drag-and-drop helpers originally in the feature generator's window;
they are generic, so they now live here and are used by every page.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

from PySide6.QtCore import QEvent, QObject, Qt
from PySide6.QtGui import QFontDatabase
from PySide6.QtWidgets import (
    QAbstractSpinBox,
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from common.naming import LAS_SUFFIXES


# Role colours for the accent buttons (background, hover) - readable white-on-colour text.
_BUTTON_ROLES = {
    "primary": ("#1971c2", "#1864ab"),  # blue   - the page's main action
    "next": ("#2f9e44", "#2b8a3e"),     # green  - advance to the next tab
    "clear": ("#e03131", "#c92a2a"),    # red    - destructive "Clear …" actions
}


class NoScrollFilter(QObject):
    """App-wide event filter that stops the mouse wheel from ever changing spinbox/combo values.

    The dense ML/DL panels live in scroll areas full of spinboxes and combos. By default a wheel turn
    over any of them mutates its value, so scrolling the panel silently changes settings — a minefield.
    Installed once on the ``QApplication`` (see :func:`gui.run`), this intercepts wheel events for
    spinboxes/combos and **re-sends them to the widget's parent**, so the value stays put *and* the
    enclosing scroll area still scrolls. Values are still changed by typing, the up/down arrows, or the
    arrow keys — just never by an incidental scroll. (Blocking unconditionally, rather than only when
    unfocused, keeps it predictable: a wheel turn never moves a value, full stop.)
    """

    def eventFilter(self, obj, event):  # noqa: N802 - Qt override name
        if event.type() == QEvent.Wheel and isinstance(obj, (QAbstractSpinBox, QComboBox)):
            parent = obj.parentWidget()
            if parent is not None:
                QApplication.sendEvent(parent, event)  # let the scroll area handle the scroll
            return True  # consume on the field so its value never changes
        return super().eventFilter(obj, event)


def wire_parent_toggle(parent, children: list[QCheckBox]) -> None:
    """Make a ``parent`` toggle drive a list of child checkboxes (a "select all" header).

    Clicking the parent sets every child to its new state. Toggling any child refreshes the
    parent's *display* to "any child on -> on, none on -> off" without re-driving the children; a
    guard flag breaks the parent<->child feedback loop. The header is strictly binary (checked /
    unchecked) — there is no tri-state "partial" display.

    Used for the two-level class -> type feature pickers on the feature and ML pages. Works for both
    a checkable ``QGroupBox`` parent and a ``QCheckBox`` header parent (treated the same way).
    """
    guard = {"busy": False}

    def on_parent(state: bool) -> None:
        if guard["busy"]:
            return
        guard["busy"] = True
        for c in children:
            c.setChecked(state)
        guard["busy"] = False

    def refresh_parent(_=None) -> None:
        if guard["busy"]:
            return
        guard["busy"] = True
        parent.setChecked(any(c.isChecked() for c in children))  # binary: any child on -> on
        guard["busy"] = False

    if isinstance(parent, QCheckBox):
        # clicked() fires only on user action (not programmatic state changes), so it never echoes
        # the display refresh below; an off header turns all on, an on header turns all off.
        parent.clicked.connect(lambda checked: on_parent(checked))
    else:
        parent.toggled.connect(on_parent)
    for c in children:
        c.toggled.connect(refresh_parent)


def make_collapsible_section(title: str, content: QWidget, *, expanded: bool = True) -> QWidget:
    """A collapsible section: an arrow header button (▾/▸) that shows/hides ``content`` below it.

    Qt has no built-in collapsible container. A *checkable group box* reads like an enable/disable
    toggle (users think unchecking it turns the section off), so this uses an explicit expand/collapse
    **arrow** header instead — ``▾ Title`` when open, ``▸ Title`` when collapsed — which reads
    unambiguously as show/hide. Unlike :func:`wire_parent_toggle` (which drives child *checkboxes*),
    this only collapses its single content widget. Used to stack collapsible sections in one panel
    without either dominating the height. Starts ``expanded`` by default.
    """
    container = QWidget()
    outer = QVBoxLayout(container)
    outer.setContentsMargins(0, 0, 0, 0)
    outer.setSpacing(2)

    header = QPushButton()
    header.setCheckable(True)
    header.setChecked(expanded)
    header.setCursor(Qt.PointingHandCursor)
    # Flat, left-aligned header so it reads as a section title with a disclosure arrow, not a button.
    header.setStyleSheet(
        "QPushButton { text-align: left; font-weight: bold; border: none; padding: 4px 2px; } "
        "QPushButton:hover { color: #1971c2; }"
    )

    def _sync(checked: bool) -> None:
        header.setText(f"{'▾' if checked else '▸'}  {title}")
        content.setVisible(checked)

    header.toggled.connect(_sync)
    outer.addWidget(header)
    outer.addWidget(content)
    _sync(expanded)
    return container


def style_button(button, role: str):
    """Give ``button`` an accent colour by role (``primary`` blue / ``next`` green / ``clear`` red).

    A thin shared helper so the whole app colours its main-action, navigation and clear buttons
    consistently. Returns the button for convenient inline use.
    """
    base, hover = _BUTTON_ROLES[role]
    button.setStyleSheet(
        f"QPushButton {{ background-color: {base}; color: white; border: none; "
        f"padding: 5px 10px; border-radius: 3px; }} "
        f"QPushButton:hover {{ background-color: {hover}; }} "
        f"QPushButton:disabled {{ background-color: #adb5bd; color: #f1f3f5; }}"
    )
    return button


def make_console(parent, console=None, *, title: str = "Console", extra_header: list | None = None):
    """A console block — bold header, optional header widgets, a red bin, and the log below.

    Every tab logs its run into one of these, so they all read the same way: the same header, the
    same monospace text area (fixed-pitch, because several logs align columns by padding), and the
    same confirm-then-clear bin at the top right. Pass ``console`` to wrap an existing widget (the
    ML tab's rich-text ``QTextBrowser``); otherwise a read-only :class:`QPlainTextEdit` is made.
    ``extra_header`` widgets sit between the title and the bin. Returns ``(container, console)``.
    """
    if console is None:
        console = QPlainTextEdit()
        console.setReadOnly(True)
    console.setFont(QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont))

    container = QWidget(parent)
    outer = QVBoxLayout(container)
    outer.setContentsMargins(0, 0, 0, 0)

    header = QHBoxLayout()
    label = QLabel(title)
    label.setStyleSheet("font-weight: bold;")
    header.addWidget(label, 1)
    for widget in extra_header or []:
        header.addWidget(widget)

    clear_button = QPushButton("\U0001f5d1")
    clear_button.setFixedWidth(32)
    clear_button.setToolTip("Clear the console.")
    style_button(clear_button, "clear")

    def _clear() -> None:
        if QMessageBox.question(
            container, "Clear console", "Clear the console? This can't be undone."
        ) == QMessageBox.Yes:
            console.clear()

    clear_button.clicked.connect(_clear)
    header.addWidget(clear_button)
    outer.addLayout(header)
    outer.addWidget(console, 1)
    return container, console


def make_status_indicator() -> QLabel:
    """A small word-wrapped label used by every tab to show whether its data is loaded.

    Pair it with :func:`set_status_indicator` to flip between a green ``✓ ready`` state and a
    gray ``○ waiting`` state, so each tab gives the same at-a-glance "is this step's data here?"
    feedback for both new and opened projects.
    """
    label = QLabel()
    label.setWordWrap(True)
    return label


def set_status_indicator(label: QLabel, ready: bool, message: str) -> None:
    """Set ``label`` to a green check (``ready``) or a gray circle, followed by ``message``."""
    if ready:
        mark, colour = "✓", "#2f9e44"  # green - matches the "next" accent
    else:
        mark, colour = "○", "#868e96"  # gray - nothing loaded yet
    label.setText(f"{mark}  {message}")
    label.setStyleSheet(f"color: {colour}; font-weight: bold;")


def confirm_and_delete(parent, title: str, paths: list[Path]) -> int:
    """Confirm (listing the exact files) then delete ``paths``; return the count removed.

    Used by the "Clear …" buttons so the user always sees precisely which generated files will
    be removed before anything is deleted. Returns 0 (and deletes nothing) if there is nothing to
    remove or the user cancels.
    """
    existing = [p for p in paths if p.exists()]
    if not existing:
        QMessageBox.information(parent, title, "Nothing to delete.")
        return 0
    listing = "\n".join(f"  • {p.name}" for p in existing[:20])
    more = "" if len(existing) <= 20 else f"\n  … and {len(existing) - 20} more"
    if QMessageBox.question(
        parent, title,
        f"Delete these {len(existing)} file(s)?\n\n{listing}{more}",
    ) != QMessageBox.Yes:
        return 0
    removed = 0
    for path in existing:
        try:
            path.unlink()
            removed += 1
        except OSError as exc:
            QMessageBox.warning(parent, title, f"Could not delete {path.name}: {exc}")
    return removed


@dataclass
class ClearSection:
    """One stage's deletable files in the cascade-clear dialog.

    ``primary`` marks the stage the user actually clicked "Clear" on (its box is checked and
    disabled — it always goes); the downstream stages are checked but toggleable so the user can
    keep one if they want, with everything-downstream-on as the default.
    """

    key: str
    label: str            # shown next to the checkbox, e.g. "Augmented clouds + manifest"
    files: list[Path]
    primary: bool = False
    checked: bool = True
    _box: QCheckBox | None = field(default=None, repr=False, compare=False)

    @property
    def existing(self) -> list[Path]:
        return [p for p in self.files if p.exists()]


def cascade_clear_dialog(parent, title: str, sections: list[ClearSection]) -> dict[str, int] | None:
    """One popup to clear a stage **and** its now-stale downstream stages; return per-key removed.

    Each section is a checkbox (file count shown). The primary section is checked and locked;
    downstream sections default to checked (clearing a stage makes everything derived from it stale)
    but can be unticked. Returns ``{key: n_removed}`` for the sections the user kept ticked, or
    ``None`` if cancelled / nothing to delete.
    """
    sections = [s for s in sections if s.existing]
    if not sections:
        QMessageBox.information(parent, title, "Nothing to delete.")
        return None

    dialog = QDialog(parent)
    dialog.setWindowTitle(title)
    layout = QVBoxLayout(dialog)
    layout.addWidget(QLabel(
        "Clearing a stage makes everything derived from it stale. Choose what to delete "
        "(downstream stages are pre-selected):"
    ))
    for section in sections:
        box = QCheckBox(f"{section.label}  ({len(section.existing)} file(s))")
        box.setChecked(True if section.primary else section.checked)
        box.setEnabled(not section.primary)  # the clicked stage always goes
        section._box = box
        layout.addWidget(box)

    buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
    buttons.button(QDialogButtonBox.Ok).setText("Delete selected")
    buttons.accepted.connect(dialog.accept)
    buttons.rejected.connect(dialog.reject)
    layout.addWidget(buttons)

    if dialog.exec() != QDialog.Accepted:
        return None

    removed: dict[str, int] = {}
    for section in sections:
        if section._box is None or not section._box.isChecked():
            continue
        count = 0
        for path in section.existing:
            try:
                path.unlink()
                count += 1
            except OSError as exc:
                QMessageBox.warning(parent, title, f"Could not delete {path.name}: {exc}")
        removed[section.key] = count
    return removed


def select_and_delete(parent, title: str, items: list[tuple[Path, str]]) -> int:
    """Multi-select delete dialog: ``items`` is ``(path, label)`` pairs; return the count removed.

    Used for the Results tab's per-model delete, where each entry is shown by its descriptive
    bundle label rather than its raw file name. Nothing is pre-selected (the user picks exactly
    which to remove). Returns 0 if nothing exists, none are picked, or the user cancels.
    """
    items = [(p, lbl) for p, lbl in items if p.exists()]
    if not items:
        QMessageBox.information(parent, title, "Nothing to delete.")
        return 0

    dialog = QDialog(parent)
    dialog.setWindowTitle(title)
    layout = QVBoxLayout(dialog)
    layout.addWidget(QLabel("Tick the model(s) to delete:"))
    boxes: list[tuple[QCheckBox, Path]] = []
    for path, label in items:
        box = QCheckBox(label)
        layout.addWidget(box)
        boxes.append((box, path))

    buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
    buttons.button(QDialogButtonBox.Ok).setText("Delete selected")
    buttons.accepted.connect(dialog.accept)
    buttons.rejected.connect(dialog.reject)
    layout.addWidget(buttons)

    if dialog.exec() != QDialog.Accepted:
        return 0
    chosen = [path for box, path in boxes if box.isChecked()]
    if not chosen:
        return 0
    removed = 0
    for path in chosen:
        try:
            path.unlink()
            removed += 1
        except OSError as exc:
            QMessageBox.warning(parent, title, f"Could not delete {path.name}: {exc}")
    return removed


def downstream_feature_files(features_dir: Path) -> list[Path]:
    """The feature + targets .csv tables in ``features_dir`` (the feature stage's outputs).

    The single source of truth for "what feature generation produced", shared by the Feature tab's
    own Clear button and the cascade triggered from the Augment tab, so they always agree: any
    ``.csv`` mentioning "feature" (but not "manifest"/"interaction") plus every ``target`` table.
    """
    if not features_dir.is_dir():
        return []
    features = [
        p for p in features_dir.glob("*.csv")
        if "feature" in p.name.lower()
        and "manifest" not in p.name.lower()
        and "interaction" not in p.name.lower()
    ]
    targets = [p for p in features_dir.glob("*.csv") if "target" in p.name.lower()]
    return list(dict.fromkeys(features + targets))  # de-dup, preserve order


def downstream_model_files(model_dir: Path) -> list[Path]:
    """Every saved model bundle in ``model_dir`` (the model stage's outputs).

    Used by the cascade clears (re-augmenting / re-generating features makes a saved model's field
    map stale) and by the Results tab's per-model delete. Kept here so the gatherer is shared.
    """
    if not model_dir.is_dir():
        return []
    return sorted(model_dir.glob("*.joblib"))


def open_folder(folder: str | Path) -> None:
    """Open ``folder`` in the OS file browser (best effort, cross-platform)."""
    folder = str(folder)
    if sys.platform.startswith("win"):
        os.startfile(folder)  # noqa: S606 - intended: reveal the output folder
    elif sys.platform == "darwin":
        os.system(f'open "{folder}"')
    else:
        os.system(f'xdg-open "{folder}"')


# The per-step output sub-folder names (clip + augment now share ``plots/``).
PIPELINE_SUBFOLDERS = ("plots", "features", "model")


def subfolder(base: Path, name: str) -> Path:
    """``base/name``, but avoid double-nesting if ``base`` already ends in ``name``.

    Used so each pipeline step writes into its own dedicated sub-folder of the chosen output
    folder (e.g. ``<out>/plots``, ``<out>/features``) without piling
    ``features/features`` when a step is repeated against the already-nested folder.
    """
    base = Path(base)
    return base if base.name == name else base / name


def pipeline_base(folder: Path | str | None) -> Path | None:
    """The shared base folder of a pipeline output, stripping a known step sub-folder leaf.

    A hand-off path points at one step's sub-folder (e.g. ``<base>/plots``); siblings
    (``<base>/features``, ``<base>/model``) are derived from the **base**. If ``folder`` already
    is such a sub-folder its parent is returned; otherwise it is treated as the base itself.
    """
    if not folder:
        return None
    folder = Path(folder)
    return folder.parent if folder.name in PIPELINE_SUBFOLDERS else folder


def collect_las_files(paths: list[str]) -> list[Path]:
    """Expand dropped/selected paths into a flat list of .las/.laz files (folders recursed)."""
    found: list[Path] = []
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            found.extend(
                p for p in sorted(path.rglob("*")) if p.suffix.lower() in LAS_SUFFIXES
            )
        elif path.suffix.lower() in LAS_SUFFIXES:
            found.append(path)
    return found


class FileListWidget(QListWidget):
    """A list of queued files that accepts dropped files and folders."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setSelectionMode(QListWidget.ExtendedSelection)
        self.setAcceptDrops(True)
        self.setToolTip("Drag .las/.laz files or folders here")

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        paths = [url.toLocalFile() for url in event.mimeData().urls()]
        self.add_paths(paths)
        event.acceptProposedAction()

    def add_paths(self, paths: list[str]) -> None:
        existing = self.file_paths()
        for f in collect_las_files(paths):
            if f not in existing:
                item = QListWidgetItem(str(f))
                item.setData(Qt.UserRole, str(f))
                self.addItem(item)
                existing.append(f)

    def set_files(self, paths: list[Path]) -> None:
        """Replace the list contents with ``paths``."""
        self.clear()
        for f in paths:
            item = QListWidgetItem(str(f))
            item.setData(Qt.UserRole, str(f))
            self.addItem(item)

    def file_paths(self) -> list[Path]:
        return [Path(self.item(i).data(Qt.UserRole)) for i in range(self.count())]

    def remove_selected(self) -> None:
        for item in self.selectedItems():
            self.takeItem(self.row(item))


class DropLineEdit(QLineEdit):
    """A line edit that accepts a dropped file or folder.

    By default the dropped path is taken **verbatim** - so a file field (a .las/.gpkg/.xlsx
    path) receives the file the user dropped, not its parent folder. Set ``folder_only=True``
    for output-directory fields, where dropping a file should resolve to its containing folder.
    """

    def __init__(self, parent=None, folder_only: bool = False) -> None:
        super().__init__(parent)
        self._folder_only = folder_only
        self.setAcceptDrops(True)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        if urls:
            path = Path(urls[0].toLocalFile())
            if self._folder_only and not path.is_dir():
                path = path.parent
            self.setText(str(path))
        event.acceptProposedAction()


# ---------------------------------------------------------------------- #
# Control snapshots (used to reset a page when a new project is opened)   #
# ---------------------------------------------------------------------- #
def snapshot_controls(root: QWidget) -> list[tuple]:
    """Record the current value of every input control under ``root``.

    Pages carry a lot of user-tweakable state (option checkboxes, spin boxes, dropdowns, text
    fields) that is built once and never re-seeded, so without this an opened project inherited
    whatever the previous one was left on. Taking the snapshot right after a page is built - i.e.
    at its defaults - gives :func:`restore_controls` something to put back on the next project.
    """
    entries: list[tuple] = []
    for widget in root.findChildren(QWidget):
        if isinstance(widget, QCheckBox):
            entries.append((widget, "check", widget.isChecked()))
        elif isinstance(widget, QAbstractSpinBox) and hasattr(widget, "value"):
            entries.append((widget, "value", widget.value()))
        elif isinstance(widget, QComboBox):
            entries.append((widget, "combo", widget.currentText()))
        elif isinstance(widget, QLineEdit):
            # Combo boxes and spin boxes own an internal QLineEdit; the parent control already
            # carries their value, so recording the inner field too would fight with it.
            if isinstance(widget.parent(), (QComboBox, QAbstractSpinBox)):
                continue
            entries.append((widget, "text", widget.text()))
    return entries


def restore_controls(entries: list[tuple]) -> None:
    """Put back the values captured by :func:`snapshot_controls`.

    Signals stay blocked throughout: several fields persist themselves into the *active* project
    when edited, and this runs after the new project is already active - re-emitting here would
    overwrite the new project's saved values with the old page state. Widgets that were rebuilt or
    deleted since the snapshot (dynamic column checkboxes, re-filled dropdowns) are skipped; the
    pages rebuild those from the new project themselves.
    """
    for widget, kind, value in entries:
        try:
            widget.isVisible()  # cheap liveness probe: a deleted C++ object raises here
        except RuntimeError:
            continue
        blocked = widget.blockSignals(True)
        try:
            if kind == "check":
                widget.setChecked(value)
            elif kind == "value":
                widget.setValue(value)
            elif kind == "combo":
                if widget.isEditable():
                    widget.setCurrentText(value)
                else:
                    index = widget.findText(value)
                    widget.setCurrentIndex(index if index >= 0 else -1)
            elif kind == "text":
                widget.setText(value)
        finally:
            widget.blockSignals(blocked)
