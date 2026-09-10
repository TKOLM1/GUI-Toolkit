"""k-Nearest Neighbors regression."""

from __future__ import annotations

from sklearn.neighbors import KNeighborsRegressor

from .base import HParam, ModelDef


class _ClampedKNN(KNeighborsRegressor):
    """kNN that caps ``n_neighbors`` at the training-set size when it is fit.

    ``n_neighbors`` can exceed the rows in a given training fold: the HParam max is 50, but a CV
    fold (after the hold-out block, and dropped augmented rows, are removed) can be smaller — the
    nested-CV search would otherwise pick e.g. k=48 against a 45-row fold and sklearn raises
    "Expected n_neighbors <= n_samples_fit". Capping at fit time keeps every caller (optimizer,
    feature selection, validate, final trainer) safe in one place, and only ever looks at the
    training rows, so it stays leakage-free. The cap is silent — a fold smaller than k just averages
    over all of its rows, which is the sensible degenerate behaviour.
    """

    def fit(self, X, y):
        n_samples = len(X)
        if self.n_neighbors > n_samples:
            self.n_neighbors = max(1, n_samples)
        return super().fit(X, y)


def _factory(p: dict) -> KNeighborsRegressor:
    return _ClampedKNN(
        n_neighbors=int(p["n_neighbors"]),
        weights=str(p["weights"]),
    )


MODEL = ModelDef(
    key="knn",
    label="k-Nearest Neighbors",
    tooltip="Predicts from the k closest plots in feature space; scale-sensitive.",
    hparams=(
        HParam("n_neighbors", "Neighbors (k)", "int", 5, 1, 50,
               tooltip="How many nearest neighbours to average over."),
        HParam("weights", "Weighting", "choice", "distance",
               choices=("uniform", "distance"),
               tooltip="'distance' weights closer neighbours more heavily."),
    ),
    needs_scaling=True,
    factory=_factory,
)
