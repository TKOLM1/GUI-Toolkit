"""Tab 0: the project folder — the single place inputs and outputs are chosen.

Create a new project (a fresh folder with the pipeline sub-folders) or open an existing one.
Once a project is active every other tab derives its input and output locations from it, so the
per-tab folder pickers are gone; the external inputs (cloud / masks / reference) are pinned in the
project itself and restored on reopen.

A **preset selector** here chooses which named block of defaults (from ``presets.txt``) pre-fills
the Clipping / Augmentation / Feature tabs. The page emits :attr:`project_changed` and
:attr:`preset_changed` so the shell can enable / re-seed the pipeline tabs.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from common import config as app_config
from common.config import CONFIG
from common.naming import set_label_format
from common.project import create_project_here, is_project, open_project
from common.session import Session
from .widgets import make_status_indicator, open_folder, set_status_indicator, style_button


class ProjectPage(QWidget):
    """The project-folder module UI (the first tab)."""

    project_changed = Signal()  # a project was created or opened; the shell re-enables the tabs
    seed_changed = Signal()      # the project-wide seed changed (shell refreshes its status readout)
    preset_changed = Signal()    # the active preset changed; the shell re-seeds every page from it
    request_next = Signal()      # ask the shell to switch to the clipping tab

    def __init__(self, session: Session) -> None:
        super().__init__()
        self._session = session

        root = QVBoxLayout(self)
        root.addWidget(self._build_actions_panel())
        root.addWidget(self._build_status_panel(), 1)
        self._refresh_status()

    # ------------------------------------------------------------------ #
    # Panels                                                             #
    # ------------------------------------------------------------------ #
    def _build_actions_panel(self) -> QWidget:
        box = QGroupBox("Project folder")
        layout = QVBoxLayout(box)

        buttons = QHBoxLayout()
        self.new_button = QPushButton("Create new project…")
        self.new_button.setToolTip(
            "Choose a folder; it becomes the project, with empty plots / features / model "
            "sub-folders created inside it."
        )
        self.new_button.clicked.connect(self._on_create)
        style_button(self.new_button, "primary")

        self.open_button = QPushButton("Open existing project…")
        self.open_button.setToolTip("Open a previously created project folder and autofill everything.")
        self.open_button.clicked.connect(self._on_open)
        style_button(self.open_button, "next")

        buttons.addWidget(self.new_button)
        buttons.addWidget(self.open_button)
        buttons.addStretch(1)
        layout.addLayout(buttons)

        # Preset selector: the named set of defaults applied across the first four tabs.
        preset_form = QFormLayout()
        self.preset_combo = QComboBox()
        self.preset_combo.addItems(sorted(app_config.PRESETS.keys()))
        self.preset_combo.setCurrentText(app_config.ACTIVE_NAME)
        self.preset_combo.setToolTip(
            "The configuration preset (from presets.txt) whose defaults pre-fill the Clipping, "
            "Augmentation and Feature tabs. Edit presets.txt to add or change presets."
        )
        self.preset_combo.currentTextChanged.connect(self._on_preset_changed)
        preset_form.addRow("Preset", self.preset_combo)
        layout.addLayout(preset_form)
        return box

    def _build_status_panel(self) -> QWidget:
        box = QGroupBox("Active project")
        layout = QVBoxLayout(box)
        self.indicator = make_status_indicator()
        layout.addWidget(self.indicator)
        self.status = QLabel()
        self.status.setWordWrap(True)
        layout.addWidget(self.status)

        # One project-wide random seed: every stage (augmentation, ML/DL splits, both hyperparameter
        # optimizers) reads this, so a project reproduces end-to-end. Saved to project.json.
        seed_form = QFormLayout()
        self.seed = QSpinBox()
        self.seed.setRange(0, 2_000_000_000)
        self.seed.setToolTip(
            "The single random seed for the whole project: augmentation, every train/test split, "
            "and both hyperparameter optimizers. Saved to the project folder whenever it changes."
        )
        self.seed.valueChanged.connect(self._on_seed_changed)
        seed_form.addRow("Random seed", self.seed)
        layout.addLayout(seed_form)

        layout.addStretch(1)
        self.open_folder_button = QPushButton("Open project folder")
        self.open_folder_button.clicked.connect(self._on_open_folder)
        layout.addWidget(self.open_folder_button)

        self.next_button = QPushButton("Go to clipping →")
        self.next_button.clicked.connect(lambda: self.request_next.emit())
        style_button(self.next_button, "next")
        layout.addWidget(self.next_button)
        return box

    # ------------------------------------------------------------------ #
    # Actions                                                            #
    # ------------------------------------------------------------------ #
    def _on_create(self) -> None:
        """Create a project **in the folder the user picks** — no follow-up name prompt.

        The folder dialog already lets the user create and name a folder, so asking for a project
        name afterwards (and nesting a second folder inside the chosen one) was redundant: the
        chosen folder *is* the project, and its own name is the project name.
        """
        folder = QFileDialog.getExistingDirectory(
            self, "Choose (or create) the folder to use as the project", CONFIG.start_dir("")
        )
        if not folder:
            return
        try:
            project = create_project_here(folder)
        except FileExistsError:
            # Offer to open it instead of failing outright.
            if QMessageBox.question(
                self, "Project exists",
                f"A project already exists at:\n{folder}\n\nOpen it instead?",
            ) == QMessageBox.Yes:
                self._activate(open_project(folder))
            return
        except (OSError, ValueError) as exc:
            QMessageBox.critical(self, "Could not create project", str(exc))
            return
        self._activate(project)

    def _on_open(self) -> None:
        folder = QFileDialog.getExistingDirectory(
            self, "Open project folder", CONFIG.start_dir("")
        )
        if not folder:
            return
        if not is_project(folder) and QMessageBox.question(
            self, "Not a project folder",
            f"{folder}\n\nThis folder has no project.json. Open it as a project anyway "
            "(its sub-folders will be created as needed)?",
        ) != QMessageBox.Yes:
            return
        try:
            project = open_project(folder)
        except (OSError, FileNotFoundError) as exc:
            QMessageBox.critical(self, "Could not open project", str(exc))
            return
        self._activate(project)

    def _activate(self, project) -> None:
        # Wipe the previous project's hand-off state before switching, so the new project starts
        # clean (the shell additionally resets each page's own caches via reset_for_project).
        self._session.reset_handoff()
        self._session.project = project
        self._refresh_status()
        self.project_changed.emit()

    def _on_seed_changed(self, value: int) -> None:
        if self._session.project is not None:
            self._session.project.set_seed(int(value))
            self.seed_changed.emit()

    def _on_preset_changed(self, name: str) -> None:
        """Apply the chosen preset: persist it, update the label format, re-seed every page."""
        if not name or name not in app_config.PRESETS:
            return
        app_config.set_active_preset(name)  # updates the in-memory CONFIG too
        set_label_format(app_config.CONFIG.label_start, app_config.CONFIG.label_end)
        self.preset_changed.emit()

    def _on_open_folder(self) -> None:
        if self._session.project is not None:
            open_folder(self._session.project.root)

    # ------------------------------------------------------------------ #
    def _refresh_status(self) -> None:
        project = self._session.project
        has = project is not None
        self.open_folder_button.setEnabled(has)
        self.next_button.setEnabled(has)
        self.seed.setEnabled(has)
        # Reflect the active project's saved seed without re-triggering a save.
        self.seed.blockSignals(True)
        self.seed.setValue(int(project.seed) if has else 0)
        self.seed.blockSignals(False)
        if not has:
            set_status_indicator(self.indicator, False, "No project open.")
            self.status.setText(
                "Create a new project or open an existing one to begin."
            )
            return
        set_status_indicator(self.indicator, True, f"Project '{project.name}' is active.")
        self.status.setText(f"Location: {project.root}")

    def on_enter(self) -> None:  # called by the shell when this page is shown
        self._refresh_status()
