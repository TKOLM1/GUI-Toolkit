"""The Feature tab must never keep a stale reference sheet.

Both SGCBP presets read the same ground-truth file, filtered to a different growth stage. The tab used
to decide "same sheet" by file path, so after a preset switch it kept the old stage's rows and wrote
the late clouds' targets from the early stage. These run the real page headless (offscreen Qt).
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pandas as pd
import pytest

QtWidgets = pytest.importorskip("PySide6.QtWidgets")

from common.session import Session
from featuregen.external import ExternalData
from gui.feature_page import FeaturePage


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _sheet(path, stage: str, biomass: float) -> ExternalData:
    """One loaded copy of the sheet, as the Import tab builds it for a given stage filter."""
    frame = pd.DataFrame({"trt": ["Hi"], "stage": [stage], "biomass_g_mSq": [biomass]}, index=[101])
    return ExternalData(frame=frame, columns=list(frame.columns), source_path=path)


def test_reload_of_the_same_file_is_adopted(app, tmp_path):
    session = Session()
    page = FeaturePage(session)
    path = tmp_path / "Ground_truth_data_final.csv"

    session.external = _sheet(path, "Z31", 477.15)
    page._sync_external()
    # The Import tab reloads the same file under the late preset's stage filter.
    session.external = _sheet(path, "Z65", 1192.2)
    page._sync_external()

    assert page._external is session.external
    assert page._external.lookup("plot(101).laz", ["biomass_g_mSq"])["biomass_g_mSq"] == 1192.2


def test_reentry_without_a_reload_keeps_the_users_ticks(app, tmp_path):
    session = Session()
    page = FeaturePage(session)
    session.external = _sheet(tmp_path / "sheet.csv", "Z31", 477.15)
    page._sync_external()
    page._external_checks["trt"].setChecked(True)

    page._sync_external()  # re-entering the tab: same loaded sheet, nothing to rebuild

    assert page._external_checks["trt"].isChecked()
