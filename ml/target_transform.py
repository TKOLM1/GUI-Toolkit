"""Fitting on a transformed target (log) while reporting in the target's own units.

Biomass-like targets are multiplicative and right-skewed: the error grows with the plot, and a
model fit on the raw values spends its capacity on the few large plots. Fitting on ``log(y)``
fixes that — but only if the predictions are brought *back* honestly, which is where the obvious
implementation goes wrong.

**The exponential trap.** A model fit on the log target predicts the conditional mean of
``log y``. Exponentiating that gives the conditional **median** of ``y``, not its mean: for a
residual spread ``σ`` the value is low by a factor of about ``exp(σ²/2)``. Naively exponentiating
therefore under-predicts every plot systematically — a real bias, not noise, so it does not average
out, and it lands directly in the rRMSE/MAPE the whole app reports. On a target with 25% residual
scatter that is roughly a 3% systematic shortfall.

The fix used here is **Duan's smearing estimator**: at fit time the residuals on the log scale are
kept, and the back-transform multiplies by ``S = mean(exp(residual))``. That is a non-parametric
estimate of the very factor the trap loses — it needs no normality assumption (unlike the
parametric ``exp(σ²/2)``), and it collapses to the naive exponential when the residuals are
symmetric on the raw scale. ``S`` is computed from the fold's **training** rows only, so it is as
leakage-safe as the scaler inside the pipeline.

The transform lives in :class:`TransformedTargetPipeline`, which wraps the model pipeline and does
the whole round trip inside ``fit``/``predict``. That placement is deliberate: everything in this
package — the metric cube, the per-plot prediction tables, the optimiser's objective, the
permutation importances, the Results tab — reads predictions through ``predict``, so with the
back-transform there, **every reported number is already in the target's original units** and no
consumer needs to know a transform happened.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin

# The transform keys the GUI and TrainConfig speak.
NONE = "none"
LOG = "log"
TRANSFORM_KEYS = (NONE, LOG)

# How the log back-transform corrects the exponential trap.
SMEARING = "smearing"   # Duan's non-parametric factor, mean(exp(train residual)) — the default
NAIVE = "naive"         # plain exponential: the conditional MEDIAN, knowingly biased low
BIAS_KEYS = (SMEARING, NAIVE)

# ``log1p``/``expm1`` rather than ``log``/``exp``: biomass targets legitimately contain exact zeros
# (a dead or unemerged plot), which plain log cannot represent. For values well above 1 the two are
# indistinguishable, so this costs nothing and removes a whole class of run-time failure. The one
# caveat: for a target whose values are of order 1 or below, the ``+1`` offset compresses the log
# scale, which weakens both the transform and the smearing correction (measured: at a target mean of
# ~4 the correction removes about half the median bias, at ~250 essentially all of it). Biomass in any
# normal unit (kg/ha, g/m²) sits far above that, so this is a note, not a limitation in practice.
_FLOOR = -1.0  # log1p is defined for y > -1


def forward(y):
    """Map the target onto the log scale (``log1p``, so exact zeros are allowed)."""
    y = np.asarray(y, dtype=float)
    if np.any(y <= _FLOOR):
        raise ValueError(
            "The log target transform needs values greater than -1 (biomass is non-negative); "
            f"the target has {int(np.sum(y <= _FLOOR))} row(s) at or below that."
        )
    return np.log1p(y)


def smearing_factor(residuals) -> float:
    """Duan's smearing factor ``mean(exp(residual))`` from the training residuals on the log scale.

    This is the multiplicative correction the naive exponential loses (see the module docstring).
    Falls back to ``1.0`` (i.e. the naive back-transform) when there are no residuals or the mean is
    not finite, so a degenerate fold degrades to the old behaviour instead of producing NaNs.
    """
    res = np.asarray(residuals, dtype=float)
    res = res[np.isfinite(res)]
    if res.size == 0:
        return 1.0
    factor = float(np.mean(np.exp(res)))
    return factor if np.isfinite(factor) and factor > 0 else 1.0


def inverse(z, factor: float = 1.0, *, clip_zero: bool = True):
    """Bring log-scale predictions ``z`` back to the target's units, corrected by ``factor``.

    ``E[y|x] = exp(z) · S - 1`` for the ``log1p`` forward map (``S = 1`` reproduces plain ``expm1``).
    With ``clip_zero`` the result is floored at 0, because a negative biomass prediction is not a
    value the rest of the app should ever have to reason about.
    """
    out = np.exp(np.asarray(z, dtype=float)) * float(factor) - 1.0
    return np.clip(out, 0.0, None) if clip_zero else out


class TransformedTargetPipeline(BaseEstimator, RegressorMixin):
    """A model pipeline fit on ``log1p(y)`` that predicts in ``y``'s own units.

    ``fit`` transforms the target, fits the wrapped pipeline on it, and records the smearing factor
    from that fit's own training residuals; ``predict`` inverts the transform with that factor. The
    inner pipeline is untouched sklearn, so the imputer/scaler stay fold-local and leakage-safe.

    Introspection is delegated: ``named_steps`` and attribute lookup fall through to the wrapped
    pipeline, so the code that reads a fitted model's hyperparameters off its ``"model"`` step (the
    Results tab's nested-CV panel, the bundle metadata) keeps working unchanged.
    """

    def __init__(self, pipeline, bias_correction: str = SMEARING, clip_zero: bool = True):
        self.pipeline = pipeline
        self.bias_correction = bias_correction
        self.clip_zero = clip_zero
        self.smearing_ = 1.0

    def fit(self, X, y):
        z = forward(y)
        self.pipeline.fit(X, z)
        if self.bias_correction == SMEARING:
            self.smearing_ = smearing_factor(z - np.ravel(self.pipeline.predict(X)))
        else:
            self.smearing_ = 1.0
        return self

    def predict(self, X):
        return inverse(np.ravel(self.pipeline.predict(X)), self.smearing_, clip_zero=self.clip_zero)

    # -- introspection passthrough ------------------------------------------ #
    @property
    def named_steps(self):
        return self.pipeline.named_steps

    def __getattr__(self, name):
        # Only reached for attributes this object doesn't define; ``pipeline`` itself is set in
        # __init__, so guard against the recursive lookup during unpickling.
        if name in ("pipeline",):
            raise AttributeError(name)
        return getattr(self.__dict__["pipeline"], name)


def build_estimator(model_def, params: dict, *, normalize_columns, feature_columns,
                    target_transform: str = NONE, bias_correction: str = SMEARING, seed: int = 0):
    """Build the fitted-model object for a run: the model pipeline, wrapped if the target is logged.

    The single place the target transform is applied, called by both the validation folds and the
    optimiser's inner folds so tuning optimises the same units the run finally reports.
    """
    pipe = model_def.build(
        params, normalize_columns=normalize_columns, feature_columns=feature_columns, seed=seed
    )
    if target_transform == LOG:
        return TransformedTargetPipeline(pipe, bias_correction=bias_correction)
    return pipe


def describe(target_transform: str, bias_correction: str = SMEARING) -> str:
    """One-line human description of the target handling, for logs and the Results summary."""
    if target_transform != LOG:
        return "Target: raw (no transform)."
    how = ("Duan smearing back-transform" if bias_correction == SMEARING
           else "plain exponential back-transform (median-biased)")
    return f"Target: log1p, {how}; all metrics reported in the target's original units."
