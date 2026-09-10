"""Per-plot predictions for the results map.

The results tab colours every plot by how closely the trained model predicts it. The
trainer only keeps the best split's *held-out* predictions, but the map needs a value for
**every** plot (and every augmented copy), so :func:`compute_plot_predictions` runs the
best saved model over the whole dataset. Plots that were in that split's training set are
flagged ``'T'`` and the held-out ones ``'V'`` (a small honesty marker, since the training
plots' errors are in-sample).

The error shown is the per-plot **relative error** ``|predicted - actual| / |actual| * 100``
(percent), the same scale-free quantity as the rRMSE loss.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from common.naming import aug_number_from_name, plot_number_from_name
from .dataset import Dataset, make_xy
from .trainer import TrainHistory, active_split_model


def compute_plot_predictions(dataset: Dataset, history: TrainHistory) -> pd.DataFrame:
    """Predict every row with the **active split's** model; return a per-file results table.

    Columns: ``plot`` (number), ``aug`` (per-plot copy id, 0 for the original), ``actual``,
    ``predicted``, ``error_pct``, ``role`` (``'T'`` train / ``'V'`` validation in the active split).
    Indexed by file name. Rows whose target is missing are dropped (they never appear on the map).
    """
    X, y, _ = make_xy(dataset, history.feature_columns)
    if len(X) == 0:
        return pd.DataFrame(
            columns=["plot", "aug", "actual", "predicted", "error_pct", "role"]
        )

    active = active_split_model(history)
    model = active.model if active is not None else history.model
    train_plots = active.train_plots if active is not None else history.best_train_plots

    predicted = np.ravel(model.predict(X))
    actual = y.to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        error_pct = np.where(actual != 0, np.abs(predicted - actual) / np.abs(actual) * 100.0, np.nan)

    plots = [plot_number_from_name(str(i)) for i in X.index]
    augs = [aug_number_from_name(str(i)) or 0 for i in X.index]
    roles = ["T" if p in train_plots else "V" for p in plots]

    frame = pd.DataFrame(
        {
            "plot": plots,
            "aug": augs,
            "actual": actual,
            "predicted": predicted,
            "error_pct": error_pct,
            "role": roles,
        },
        index=X.index,
    )
    frame.index.name = "filename"
    return frame
