"""The model interface: a hyperparameter spec + a pipeline builder, behind a registry.

Every model lives in its own file and exports one :class:`ModelDef`. A ``ModelDef``
declares its hyperparameters (which drive the GUI form, the same registry idea as the
feature registry) and a ``factory`` that turns a parameter dict into a bare sklearn
estimator. :meth:`ModelDef.build` wraps that estimator in a leakage-safe pipeline:

    SimpleImputer (median)  ->  [StandardScaler if needs_scaling]  ->  estimator

The imputer fills missing feature values (e.g. files with no RGB); the scaler is fit on
the training fold *inside* the pipeline, so standardisation never sees the test data -
matching the project's "no batch normalisation" design.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class HParam:
    """One hyperparameter, with enough metadata for the GUI to build a control."""

    name: str
    label: str
    kind: str                       # "int" | "float" | "choice" | "bool"
    default: object
    min: float | None = None
    max: float | None = None
    choices: tuple | None = None
    step: float | None = None
    tooltip: str = ""


@dataclass(frozen=True)
class ModelDef:
    """One selectable regression model."""

    key: str
    label: str
    tooltip: str
    hparams: tuple[HParam, ...]
    needs_scaling: bool
    factory: Callable[[dict], BaseEstimator]
    # Names of the hparams that are solver iteration limits (e.g. "max_iter", "tol"). The GUI groups
    # these under a separate "Iteration limits" header so the user sees, and can raise, the cap that
    # produces sklearn's ConvergenceWarning. Empty for models with no iterative solver cap (the
    # default); for those models there is nothing to converge and the header is hidden. Note this is
    # only for *convergence* caps — HistGBT's max_iter is a boosting-round count (a real modelling
    # hyperparameter, not a convergence cap), so it is deliberately *not* listed here.
    iteration_hparams: tuple[str, ...] = ()

    def defaults(self) -> dict:
        """The default hyperparameter dict (used when the GUI has not overridden one)."""
        return {h.name: h.default for h in self.hparams}

    def core_hparams(self) -> tuple[HParam, ...]:
        """The model's hyperparameters that are *not* iteration limits (shown under 'Hyperparameters')."""
        return tuple(h for h in self.hparams if h.name not in self.iteration_hparams)

    def iteration_limit_hparams(self) -> tuple[HParam, ...]:
        """The model's iteration-limit hyperparameters, in declaration order (shown under their own header)."""
        names = set(self.iteration_hparams)
        return tuple(h for h in self.hparams if h.name in names)

    def build(
        self,
        params: dict | None = None,
        *,
        normalize_columns: list[str] | None = None,
        feature_columns: list[str] | None = None,
    ) -> Pipeline:
        """Build the full estimator pipeline for the given hyperparameters.

        ``normalize_columns`` selects which feature columns are standardised (the per-feature
        normalisation toggles in the GUI); ``feature_columns`` is the column order of the X the
        pipeline will see, used to turn those names into positions. Scaling only ever applies to
        models that ``need_scaling`` (tree models are scale-invariant). The scaler is still fit
        inside the pipeline on the training fold only, so this remains leakage-safe.

        * ``normalize_columns is None`` -> standardise every column (the original behaviour).
        * a subset -> a ``ColumnTransformer`` standardises those columns, others pass through.
        * an empty list -> no scaling at all ("normalize none").
        """
        merged = self.defaults()
        if params:
            merged.update(params)
        # keep_empty_features keeps all-NaN columns (filled with 0) so the column count is
        # stable - the per-feature ColumnTransformer below selects columns by integer position.
        steps = [("impute", SimpleImputer(strategy="median", keep_empty_features=True))]
        if self.needs_scaling:
            if normalize_columns is None:
                steps.append(("scale", StandardScaler()))
            elif feature_columns is not None and normalize_columns:
                # SimpleImputer emits a bare numpy array, so select by integer position.
                idx = [feature_columns.index(c) for c in normalize_columns if c in feature_columns]
                if idx:
                    steps.append((
                        "scale",
                        ColumnTransformer(
                            [("std", StandardScaler(), idx)], remainder="passthrough"
                        ),
                    ))
            elif normalize_columns:
                # No column order given: fall back to standardising everything.
                steps.append(("scale", StandardScaler()))
            # normalize_columns == [] -> no scale step ("normalize none").
        steps.append(("model", self.factory(merged)))
        return Pipeline(steps)
