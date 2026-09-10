"""Loading the feature (X) and target (y) tables and turning them into arrays.

Feature generation writes two ``.csv`` tables, both indexed by file name: a features table
(X) and a single-column targets table (y). Here they are joined back together and
turned into the ``X, y, groups`` the trainer needs, where ``groups`` is the plot number
of each row (the anti-leakage key from :mod:`common.naming`).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from common.naming import plot_number_from_name


@dataclass
class Dataset:
    """Joined feature + target table, plus the column roles."""

    frame: pd.DataFrame          # indexed by file name; feature columns + the target column
    feature_columns: list[str]   # candidate X columns (from the features file)
    target_column: str           # the single y column (from the targets file)


def _read_indexed(path: str | Path) -> pd.DataFrame:
    """Read a .csv whose first column is the file-name index (skipping ``#`` comment lines)."""
    return pd.read_csv(path, index_col=0, comment="#")


def load_dataset(features_path: str | Path, targets_path: str | Path) -> Dataset:
    """Read the features (X) and targets (y) ``.csv`` tables and join them on file name."""
    feats = _read_indexed(features_path)
    tgts = _read_indexed(targets_path)
    if tgts.shape[1] < 1:
        raise ValueError(f"Targets file {Path(targets_path).name} has no target column.")
    target_column = str(tgts.columns[0])
    feature_columns = [str(c) for c in feats.columns]

    joined = feats.join(tgts.iloc[:, [0]].rename(columns={tgts.columns[0]: target_column}))
    return Dataset(
        frame=joined,
        feature_columns=feature_columns,
        target_column=target_column,
    )


def _group_label(name: str) -> str:
    """Stable group key: the plot number, or a unique per-row key if it has none."""
    plot = plot_number_from_name(name)
    return f"plot:{plot}" if plot is not None else f"row:{name}"


def make_xy(
    dataset: Dataset, feature_cols: list[str] | None = None
) -> tuple[pd.DataFrame, pd.Series, np.ndarray]:
    """Return ``(X, y, groups)`` for training.

    Rows whose target is missing/non-numeric are dropped. Feature values are coerced to
    numeric (non-numeric become NaN and are imputed inside the model pipeline). ``groups``
    holds each surviving row's plot group, so the split can keep plots intact.
    """
    cols = feature_cols if feature_cols is not None else dataset.feature_columns
    frame = dataset.frame
    y_all = pd.to_numeric(frame[dataset.target_column], errors="coerce")
    keep = y_all.notna()
    sub = frame.loc[keep]
    X = sub[cols].apply(pd.to_numeric, errors="coerce")
    y = y_all.loc[keep]
    groups = np.array([_group_label(str(name)) for name in sub.index])
    return X, y, groups
