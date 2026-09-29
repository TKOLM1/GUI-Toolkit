"""Random Forest regressor."""

from __future__ import annotations

from sklearn.ensemble import RandomForestRegressor

from ..parallel import inner_fit_n_jobs
from .base import HParam, ModelDef


def _factory(p: dict) -> RandomForestRegressor:
    max_depth = int(p["max_depth"])
    # Use all cores when a single model is trained, but defer to the arbiter when an outer stage is
    # fanning fits across threads (optimizer folds / importance repeats): then it returns 1 so the
    # forest builds single-threaded and the two levels of parallelism don't oversubscribe the CPU.
    return RandomForestRegressor(
        n_estimators=int(p["n_estimators"]),
        max_depth=None if max_depth <= 0 else max_depth,
        min_samples_leaf=int(p["min_samples_leaf"]),
        n_jobs=inner_fit_n_jobs(-1),
    )


MODEL = ModelDef(
    key="random_forest",
    label="Random Forest",
    tooltip="Ensemble of decision trees on bootstrap samples; robust, needs no scaling.",
    hparams=(
        HParam("n_estimators", "Number of trees", "int", 300, 10, 2000, step=10,
               tooltip="How many trees in the forest. More = steadier but slower."),
        HParam("max_depth", "Max depth (0 = unlimited)", "int", 0, 0, 100,
               tooltip="Maximum tree depth; 0 lets trees grow until leaves are pure."),
        HParam("min_samples_leaf", "Min samples per leaf", "int", 1, 1, 50,
               tooltip="Minimum samples in a leaf; larger values regularise."),
    ),
    needs_scaling=False,
    factory=_factory,
)
