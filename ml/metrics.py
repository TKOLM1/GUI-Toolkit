"""Regression metrics used for the loss curve and the final report.

The "loss" reported per cycle (train and held-out) is the relative RMSE (rRMSE):
the RMSE expressed as a percentage of the mean actual value, so it is scale-free and
comparable across targets. The standardised report set surfaced everywhere is
**rRMSE, R², R (Pearson) and MAPE**; :func:`standard_metrics` bundles them (plus ``n``
and ``mae`` for backward compatibility) so every tab reports the same numbers.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)


def rmse(y_true, y_pred) -> float:
    """Root mean squared error, in the units of the target."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def rrmse(y_true, y_pred) -> float:
    """Relative RMSE as a percentage: ``rmse(y) / mean(y) * 100`` (the per-cycle loss).

    Scale-free version of :func:`rmse`, normalised by the mean of the actual values, so a
    value of ``12.5`` means the typical error is 12.5% of the average target. This is the
    "loss" used for the loss curve, for picking the best split and for hyperparameter tuning.
    ``mean(y)`` is taken in absolute value so the result stays a non-negative error magnitude;
    it falls back to ``inf`` when the mean is zero (no meaningful relative scale).
    """
    y_true = np.asarray(y_true, dtype=float)
    denom = abs(float(np.mean(y_true)))
    if denom == 0.0:
        return float("inf")
    return float(np.sqrt(mean_squared_error(y_true, y_pred)) / denom * 100.0)


def mae(y_true, y_pred) -> float:
    """Mean absolute error."""
    return float(mean_absolute_error(y_true, y_pred))


def r2(y_true, y_pred) -> float:
    """Coefficient of determination R^2."""
    return float(r2_score(y_true, y_pred))


def r(y_true, y_pred) -> float:
    """Pearson correlation coefficient between actual and predicted (sign preserved).

    Returns ``nan`` for fewer than two points or when either series has zero variance
    (no meaningful correlation). Unlike R², this keeps its sign, so a negative value flags
    predictions that move opposite to the target.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.size < 2 or np.std(y_true) == 0 or np.std(y_pred) == 0:
        return float("nan")
    val = np.corrcoef(y_true, y_pred)[0, 1]
    return float(val) if np.isfinite(val) else float("nan")


def mape(y_true, y_pred) -> float:
    """Mean absolute percentage error, as a percentage (``0`` = perfect)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.size == 0:
        return float("nan")
    return float(mean_absolute_percentage_error(y_true, y_pred) * 100.0)


def standard_metrics(y_true, y_pred) -> dict:
    """The standardised report set for one prediction series.

    Returns ``{'rrmse','r2','r','mape','mae','n'}``. ``rrmse``/``r2``/``r``/``mape`` are the
    four metrics surfaced across the app; ``mae`` is kept for backward compatibility with the
    saved metrics workbook and ``n`` is the sample count.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "rrmse": rrmse(y_true, y_pred),
        "r2": r2(y_true, y_pred),
        "r": r(y_true, y_pred),
        "mape": mape(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "n": int(y_true.size),
    }
