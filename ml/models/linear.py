"""Penalised linear regressions: Ridge, Lasso and Elastic Net.

Three separate :class:`ModelDef` entries that share the same scale-sensitive nature
(features are standardised in the pipeline).
"""

from __future__ import annotations

from sklearn.linear_model import ElasticNet, Lasso, Ridge

from .base import HParam, ModelDef

_ALPHA = HParam(
    "alpha", "Alpha (penalty strength)", "float", 1.0, 0.0001, 1000.0, step=0.1,
    tooltip="Regularisation strength; larger shrinks coefficients more.",
)

# Lasso and Elastic Net are fit by coordinate descent, an *iterative* solver that stops either when
# the solution settles (within ``tol``) or when it hits ``max_iter`` — and sklearn's default cap of
# 1000 is far too low for this project's many highly-collinear lidar features, so the solver stops
# mid-descent and raises a ConvergenceWarning with a not-quite-final (and, for Lasso, a different and
# less stable) coefficient set. We default ``max_iter`` to 200000. That number is empirical, not
# round-for-its-own-sake: on Experiment 1 the *full-data* fit converges by ~21k iterations, but the
# smaller, worse-conditioned per-split train subsets need far more — up to ~120k across the ten
# default random splits — so a 50k cap still warned on half of them. 200k clears every split, and
# because coordinate descent stops itself the instant it converges, a cap above what a fit actually
# needs costs nothing (a full 10-split Lasso train measured the same at 200k and 500k). ``tol`` is
# left at the sklearn default (1e-4). Both are exposed as editable "Iteration limits" controls so a
# user who still sees a warning (e.g. at a very small alpha Optuna explores) can raise the cap or
# loosen the tolerance. Ridge is closed-form (no iteration), so it has neither knob and never warns.
_MAX_ITER = HParam(
    "max_iter", "Max iterations", "int", 200000, 100, 5_000_000, step=1000,
    tooltip="Coordinate-descent iteration cap. If you see a ConvergenceWarning in the console, the "
            "solver hit this cap before settling — raise it (the solver stops itself once converged, "
            "so a higher cap only costs time when it is actually needed).",
)
_TOL = HParam(
    "tol", "Tolerance", "float", 1e-4, 1e-7, 1e-1, step=1e-5,
    tooltip="Convergence tolerance: the solver stops once the update falls below this. Loosening it "
            "(larger value) lets the solver declare convergence sooner — useful to speed up the many "
            "throwaway fits inside a hyperparameter search.",
)


RIDGE = ModelDef(
    key="ridge",
    label="Ridge regression",
    tooltip="Linear regression with an L2 penalty; keeps all features, shrinks coefficients.",
    hparams=(_ALPHA,),
    needs_scaling=True,
    factory=lambda p: Ridge(alpha=float(p["alpha"]), random_state=0),
)

LASSO = ModelDef(
    key="lasso",
    label="Lasso regression",
    tooltip="Linear regression with an L1 penalty; can drive some coefficients to zero.",
    hparams=(_ALPHA, _MAX_ITER, _TOL),
    needs_scaling=True,
    factory=lambda p: Lasso(
        alpha=float(p["alpha"]), max_iter=int(p["max_iter"]), tol=float(p["tol"]), random_state=0
    ),
    iteration_hparams=("max_iter", "tol"),
)

ELASTIC_NET = ModelDef(
    key="elastic_net",
    label="Elastic Net regression",
    tooltip="Linear regression blending L1 and L2 penalties (l1_ratio mixes them).",
    hparams=(
        _ALPHA,
        HParam("l1_ratio", "L1 ratio", "float", 0.5, 0.0, 1.0, step=0.05,
               tooltip="0 = pure L2 (ridge), 1 = pure L1 (lasso)."),
        _MAX_ITER,
        _TOL,
    ),
    needs_scaling=True,
    factory=lambda p: ElasticNet(
        alpha=float(p["alpha"]), l1_ratio=float(p["l1_ratio"]),
        max_iter=int(p["max_iter"]), tol=float(p["tol"]), random_state=0
    ),
    iteration_hparams=("max_iter", "tol"),
)
