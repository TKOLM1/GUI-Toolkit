"""The application shell: a nav bar over a stack of the five module pages.

The shell owns the single :class:`Session`. The nav buttons let the user move freely
between modules; the per-page "Go to ..." buttons emit ``request_next`` to advance and
carry their output forward through the session. Whenever a page is shown its
``on_enter`` runs so it can pick up anything the previous module produced.
"""

from __future__ import annotations

from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import (
    QButtonGroup,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from common.naming import LAS_SUFFIXES, aug_number_from_name, plot_number_from_name
from common.session import Session
from .augment_page import AugmentPage
from .import_page import ImportPage
from .feature_page import FeaturePage
from .ml_page import MLPage
from .project_page import ProjectPage
from .results_page import ResultsPage
from .widgets import restore_controls, snapshot_controls


class MainShell(QMainWindow):
    """Top-level window hosting the pipeline modules."""

    def _size_to_screen(self) -> None:
        """Open as a normal (non-maximised) window filling the screen's usable area."""
        screen = QGuiApplication.screenAt(self.pos()) or QGuiApplication.primaryScreen()
        if screen is None:
            self.resize(1400, 820)
            return
        avail = screen.availableGeometry()
        self.resize(avail.size())
        self.move(avail.topLeft())

    def showEvent(self, event) -> None:  # noqa: N802 - Qt naming
        """Trim the window once its frame exists, if the frame spills off-screen.

        ``resize`` sizes the *client* area, so a title bar/border can push the frame past the
        screen edge. macOS clamps this for us; other platforms may not, and the frame margins are
        only known once the native window exists.
        """
        super().showEvent(event)
        if self._fitted:
            return
        self._fitted = True
        self._trim_to_screen()
        # Some platforms only report the real frame margins a beat after the first show.
        QTimer.singleShot(0, self._trim_to_screen)

    def _trim_to_screen(self) -> None:
        """Shrink so the whole window *frame* fits the screen's available area."""
        screen = self.screen() or QGuiApplication.primaryScreen()
        if screen is None:
            return
        avail = screen.availableGeometry()
        frame = self.frameGeometry()
        over_w = max(frame.right() - avail.right(), 0)
        over_h = max(frame.bottom() - avail.bottom(), 0)
        if over_w or over_h:
            self.resize(max(self.width() - over_w, 640), max(self.height() - over_h, 480))

    def __init__(self) -> None:
        super().__init__()
        self._fitted = False
        self.setWindowTitle("Wheat Biomass Toolkit - Import, Augment, Features, ML, Results")
        self._size_to_screen()
        self.session = Session()

        self.project_page = ProjectPage(self.session)
        self.import_page = ImportPage(self.session)
        self.augment_page = AugmentPage(self.session)
        self.feature_page = FeaturePage(self.session)
        self.ml_page = MLPage(self.session)
        self.results_page = ResultsPage(self.session)
        self._pages = [
            self.project_page,
            self.import_page,
            self.augment_page,
            self.feature_page,
            self.ml_page,
            self.results_page,
        ]

        # Each page gets its own scroll area: a QStackedWidget's layout forces every tab to be at
        # least as large as the biggest one (Results wants ~1330px of height, taller than a laptop
        # screen), which would push the window off the bottom of the display. Wrapping the pages
        # keeps the shell's minimum small and scrolls only the tab that actually needs it.
        self.stack = QStackedWidget()
        for page in self._pages:
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setFrameShape(QFrame.NoFrame)
            scroll.setWidget(page)
            self.stack.addWidget(scroll)

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.addLayout(self._build_nav())
        root.addWidget(self.stack, 1)

        # "Go to next" hand-offs (each page advances to the following tab). The ML tab
        # advances to Results, which is the last tab.
        results_index = len(self._pages) - 1
        self.project_page.request_next.connect(lambda: self._show(1))
        self.import_page.request_next.connect(lambda: self._show(2))
        self.augment_page.request_next.connect(lambda: self._show(3))
        self.feature_page.request_next.connect(lambda: self._show(4))
        self.ml_page.request_next.connect(lambda: self._show(results_index))

        # When the clouds/workbooks change on the Import/Augment/Feature tabs — a Clear deletes them or a
        # re-run rewrites them — drop the ML tab's cached dataset so it can't train/optimise on files
        # that no longer match disk (the in-memory desync that read vanished clouds).
        self.import_page.data_invalidated.connect(self._on_data_invalidated)
        self.augment_page.data_invalidated.connect(self._on_data_invalidated)
        self.feature_page.data_invalidated.connect(self._on_data_invalidated)

        # Opening/creating a project enables the pipeline tabs and re-seeds every page from it.
        self.project_page.project_changed.connect(self._on_project_changed)
        # The project-wide seed changing only affects the top-right readout's context, not the tabs.
        self.project_page.seed_changed.connect(self.update_status_readout)
        # Switching preset re-seeds every page's preset-driven defaults on next entry.
        self.project_page.preset_changed.connect(self._on_preset_changed)

        # The defaults every pipeline page starts on. Opening a project puts these back, so a new
        # project never inherits the previous one's ticked options, spin-box values or dropdown
        # picks (the pages' own reset_for_project handles their caches and derived state).
        self._page_defaults = {id(page): snapshot_controls(page) for page in self._pages[1:]}

        self._set_pipeline_enabled(False)  # tabs 1.. stay disabled until a project is active
        self._show(0)
        self.update_status_readout()

    def _build_nav(self) -> QHBoxLayout:
        nav = QHBoxLayout()
        self._nav_group = QButtonGroup(self)
        self._nav_group.setExclusive(True)
        labels = [
            "0. Setup",
            "1. Import",
            "2. Data augmentation",
            "3. Feature generation",
            "4. ML",
            "5. Results",
        ]
        for i, label in enumerate(labels):
            button = QPushButton(label)
            button.setCheckable(True)
            button.clicked.connect(lambda _=False, idx=i: self._show(idx))
            self._nav_group.addButton(button, i)
            nav.addWidget(button)
        nav.addStretch(1)
        # A persistent status readout (plots loaded / augmented + target feature), visible on every
        # tab, right-aligned in the nav bar.
        self.status_readout = QLabel("")
        self.status_readout.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.status_readout.setStyleSheet("color: gray;")
        nav.addWidget(self.status_readout)
        return nav

    def _on_data_invalidated(self) -> None:
        """Drop the ML tab's cached dataset when the data changed, and refresh the readout."""
        self.ml_page.invalidate_dataset()
        self.update_status_readout()

    def update_status_readout(self) -> None:
        """Refresh the top-right readout from the project's clouds + the session's target feature."""
        project = self.session.project
        total = aug = 0
        if project is not None:
            folder = project.plots_dir
            if folder.is_dir():
                for path in folder.iterdir():
                    if path.suffix.lower() not in LAS_SUFFIXES:
                        continue
                    if plot_number_from_name(path.name) is None:
                        continue
                    total += 1
                    if aug_number_from_name(path.name):
                        aug += 1
        target = self.session.target_column or "—"
        self.status_readout.setText(
            f"Plots loaded: {total} (augmented: {aug})    Target feature: {target}"
        )

    def _show(self, index: int) -> None:
        self.stack.setCurrentIndex(index)
        button = self._nav_group.button(index)
        if button is not None:
            button.setChecked(True)
        self._pages[index].on_enter()
        # A tab's on_enter may load data or change the target; refresh the readout afterwards.
        self.update_status_readout()

    def _set_pipeline_enabled(self, enabled: bool) -> None:
        """Enable/disable the pipeline tabs (1..); the Setup tab (0) is always available."""
        for i in range(1, len(self._pages)):
            button = self._nav_group.button(i)
            if button is not None:
                button.setEnabled(enabled)

    def _on_project_changed(self) -> None:
        """A project was created/opened: enable the pipeline tabs and re-seed every page from it."""
        self._set_pipeline_enabled(True)
        # Two passes: first wipe every page's cached/visible state from the previous project, then
        # let each re-read the now-active project. Resetting all pages before any re-entry avoids
        # ordering coupling between them (the session hand-off is already cleared in _activate).
        for page in self._pages[1:]:
            restore_controls(self._page_defaults.get(id(page), []))
            page.reset_for_project()
        for page in self._pages[1:]:
            page.on_enter()
        self.update_status_readout()

    def _on_preset_changed(self) -> None:
        """The active preset changed: re-apply its defaults across the pipeline pages."""
        # Pages whose controls are preset-driven re-seed them first; the snapshot is then retaken so
        # opening a project later restores *this* preset's defaults, not the start-up preset's.
        for page in self._pages[1:]:
            apply_defaults = getattr(page, "apply_preset_defaults", None)
            if apply_defaults is not None:
                apply_defaults()
        for page in self._pages[1:]:
            page.on_enter()
        self._page_defaults = {id(page): snapshot_controls(page) for page in self._pages[1:]}
        self.update_status_readout()

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt override
        """Tear down the shared polyscope viewer when the window closes."""
        self.results_page.close_viewer()
        super().closeEvent(event)
