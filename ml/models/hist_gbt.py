"""Histogram-based gradient-boosted trees (scikit-learn's HistGradientBoostingRegressor)."""

from __future__ import annotations

from sklearn.ensemble import HistGradientBoostingRegressor

from .base import HParam, ModelDef


def _factory(p: dict) -> HistGradientBoostingRegressor:
    max_depth = int(p["max_depth"])
    return HistGradientBoostingRegressor(
        max_iter=int(p["max_iter"]),
        learning_rate=float(p["learning_rate"]),
        max_depth=None if max_depth <= 0 else max_depth,
        l2_regularization=float(p["l2_regularization"]),
    )


MODEL = ModelDef(
    key="hist_gbt",
    label="Gradient boosting (HistGBT)",
    tooltip="Fast histogram-based gradient boosting; strong tabular baseline, no scaling.",
    hparams=(
        HParam("max_iter", "Boosting iterations", "int", 200, 10, 2000, step=10,
               tooltip="Number of boosting stages (trees added sequentially)."),
        HParam("learning_rate", "Learning rate", "float", 0.1, 0.001, 1.0, step=0.01,
               tooltip="Shrinkage applied to each tree; lower = needs more iterations."),
        HParam("max_depth", "Max depth (0 = unlimited)", "int", 0, 0, 50,
               tooltip="Maximum depth of each tree; 0 means no explicit limit."),
        HParam("l2_regularization", "L2 regularisation", "float", 0.0, 0.0, 10.0, step=0.1,
               tooltip="L2 penalty on leaf values; larger values regularise."),
    ),
    needs_scaling=False,
    factory=_factory,
)
