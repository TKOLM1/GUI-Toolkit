"""Results panel: which features and hyperparameters a saved model ended up using.

A bundle saved from a nested-CV (rotating-vault) run carries ``vault_meta`` — the feature set the run
used and the per-fold record of *what it tuned* in each outer fold (``fold_features`` /
``fold_params``), plus the honest mean ± std. Hyperparameters stable across folds ⇒ the headline
number is trustworthy; jumpy ⇒ it leaned on which plots happened to be held out.

An ordinary fitted model has no outer folds, but it still ended up with one feature set and one
hyperparameter set. :func:`summary_meta` packs those into the *same* one-"fold" shape so this panel
can render for any saved model: each hyperparameter shows a single constant value
(min == median == max). That is the degenerate case of the same stability view, not a different one —
so a user opening any model gets the same "what did this end up using" answer here.

This is the selection-stability half of the old (removed) vault report dialog, moved onto Results so
it appears wherever a saved model is opened.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtWidgets import (
    QHeaderView,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)


def summary_meta(loaded) -> dict | None:
    """Build a one-"fold" ``vault_meta``-shaped dict from a regular (non-nested) :class:`LoadedModel`.

    The nested-CV panel aggregates ``fold_features`` / ``fold_params`` across outer folds. A fitted
    model has no folds, so we hand it a single fold: the model's feature columns and the fitted
    estimator's declared hyperparameters. Rendered, that reads as every feature at 100% and every
    hyperparameter constant — correct, if uninformative, which is the whole point of showing it for
    any model. Returns ``None`` only if there is nothing at all to show (no model loaded).

    No ``mean_held_out_rrmse`` is set: a single model carries no honest cross-fold number, so the
    panel's headline line is simply omitted (the splits' own rRMSE already shows in the split picker).
    """
    if loaded is None:
        return None
    features = list(getattr(loaded, "feature_columns", None) or [])
    params = _fitted_params(loaded)
    return {
        "fold_features": [features],
        "fold_params": [params],
        "did_optimize": bool(params),
    }


def _fitted_params(loaded) -> dict:
    """The hyperparameters the saved (classical) model was actually built with, by declared name.

    Reads each of the model's *declared* :class:`~ml.models.base.HParam` values off the fitted
    estimator (the pipeline's ``"model"`` step) rather than dumping ``get_params()``, which buries the
    few real knobs under sklearn defaults. Best-effort: any lookup failure — or a non-classical bundle
    whose estimator carries no declared hparams — yields ``{}``, which the panel renders as the
    "hyperparameters were fixed" note.
    """
    try:
        from ml.models import MODELS_BY_KEY

        active = _active_model_object(loaded)
        model_def = MODELS_BY_KEY.get(getattr(loaded, "model_key", None))
        if active is None or model_def is None:
            return {}
        estimator = active.named_steps.get("model") if hasattr(active, "named_steps") else active
        if estimator is None:
            return {}
        out: dict = {}
        for h in model_def.hparams:
            if hasattr(estimator, h.name):
                out[h.name] = getattr(estimator, h.name)
        return out
    except Exception:  # noqa: BLE001 - the summary is best-effort, never block the load
        return {}


def _active_model_object(loaded):
    """The fitted model object for the bundle's active split (pipeline or deep state), or ``None``."""
    from ml.trainer import active_split_model

    history = getattr(loaded, "history", None)
    if history is None:
        return None
    active = active_split_model(history)
    if active is not None and getattr(active, "model", None) is not None:
        return active.model
    return getattr(history, "model", None)


def with_target_note(meta: dict | None, history) -> dict | None:
    """Copy of ``meta`` carrying the run's target handling, read off its :class:`~ml.trainer.TrainHistory`.

    Bundles saved before the target transform existed have no such fields and default to ``"none"``,
    which renders no line — so old models look exactly as they did.
    """
    if meta is None or history is None:
        return meta
    out = dict(meta)
    out["target_transform"] = str(getattr(history, "target_transform", "none") or "none")
    out["target_bias_correction"] = str(getattr(history, "target_bias_correction", "smearing")
                                        or "smearing")
    return out


class NestedCvStability(QWidget):
    """Per-fold feature-selection + hyperparameter stability for a nested-CV bundle's ``vault_meta``."""

    def __init__(self) -> None:
        super().__init__()
        self._root = QVBoxLayout(self)
        self._title = QLabel("Selection stability")
        self._title.setStyleSheet("font-weight: bold; font-size: 14px;")
        self._root.addWidget(self._title)
        self._body = QWidget()
        self._root.addWidget(self._body, 1)
        self.set_meta(None)

    def set_meta(self, meta: dict | None) -> None:
        """Render the panel from a bundle's ``vault_meta`` dict, or show an empty note when ``None``."""
        # Replace the body widget wholesale each refresh (the tables differ by which stages ran).
        self._root.removeWidget(self._body)
        self._body.deleteLater()
        self._body = self._build_body(meta)
        self._root.addWidget(self._body, 1)

    # ------------------------------------------------------------------ #
    def _build_body(self, meta: dict | None) -> QWidget:
        panel = QWidget()
        col = QVBoxLayout(panel)
        if not meta:
            note = QLabel("No model is loaded, so there is nothing to summarise here.")
            note.setWordWrap(True)
            note.setStyleSheet("color: gray;")
            col.addWidget(note)
            return panel

        fold_features = meta.get("fold_features") or []
        fold_params = meta.get("fold_params") or []
        n = len(fold_features)
        # A single "fold" is a plain fitted model summarised through the same shape (see summary_meta):
        # there is no cross-fold spread, so headers/notes drop the fold framing and percentages.
        single = n <= 1
        did_optimize = bool(meta.get("did_optimize"))

        # Target handling first: it says what units every number below (and on every other Results
        # view) is in. Omitted entirely for a raw-target model, which is the norm.
        if str(meta.get("target_transform", "none")) == "log":
            from ml.target_transform import describe

            tnote = QLabel(describe("log", str(meta.get("target_bias_correction", "smearing"))))
            tnote.setWordWrap(True)
            col.addWidget(tnote)

        mean = meta.get("mean_held_out_rrmse")
        std = meta.get("std_held_out_rrmse")
        if mean is not None:
            head = QLabel(
                f"Honest held-out rRMSE = <b>{mean:.4g}%</b> ± {std:.3g}% across {n} outer fold(s)."
            )
            head.setWordWrap(True)
            col.addWidget(head)
        elif single:
            head = QLabel("The features and hyperparameters this saved model ended up using.")
            head.setWordWrap(True)
            head.setStyleSheet("color: gray;")
            col.addWidget(head)

        # --- features used ---
        # The feature set is fixed for the whole run (it is whatever was ticked on the ML tab), so this
        # is a plain list rather than a per-fold frequency: it answers "what did this model train on".
        features = sorted({f for feats in fold_features for f in feats})
        if features:
            col.addWidget(QLabel("Features this model uses:"))
            feat_table = QTableWidget(len(features), 1)
            feat_table.setHorizontalHeaderLabels(["Feature"])
            feat_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
            for r, name in enumerate(features):
                feat_table.setItem(r, 0, QTableWidgetItem(name))
            feat_table.setEditTriggers(QTableWidget.NoEditTriggers)
            col.addWidget(feat_table, 1)
        else:
            note = QLabel("This model records no feature set to list.")
            note.setWordWrap(True)
            note.setStyleSheet("color: gray;")
            col.addWidget(note)

        # --- hyperparameter stability ---
        if did_optimize:
            col.addWidget(QLabel(
                "Hyperparameters this model was built with:" if single
                else "Hyperparameters (spread across folds):"
            ))
            names = sorted({k for p in fold_params for k in p})
            numeric = [
                name for name in names
                if all(isinstance(p.get(name), (int, float)) and not isinstance(p.get(name), bool)
                       for p in fold_params if name in p)
            ]
            if single:
                param_table = QTableWidget(len(numeric), 2)
                param_table.setHorizontalHeaderLabels(["Param", "Value"])
                param_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
                for r, name in enumerate(numeric):
                    val = float(fold_params[0][name])
                    param_table.setItem(r, 0, QTableWidgetItem(name))
                    param_table.setItem(r, 1, QTableWidgetItem(f"{val:.4g}"))
            else:
                param_table = QTableWidget(len(numeric), 4)
                param_table.setHorizontalHeaderLabels(["Param", "min", "median", "max"])
                param_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
                for r, name in enumerate(numeric):
                    vals = np.asarray([p[name] for p in fold_params if name in p], dtype=float)
                    param_table.setItem(r, 0, QTableWidgetItem(name))
                    param_table.setItem(r, 1, QTableWidgetItem(f"{np.min(vals):.4g}"))
                    param_table.setItem(r, 2, QTableWidgetItem(f"{np.median(vals):.4g}"))
                    param_table.setItem(r, 3, QTableWidgetItem(f"{np.max(vals):.4g}"))
            param_table.setEditTriggers(QTableWidget.NoEditTriggers)
            col.addWidget(param_table, 1)
        else:
            note = QLabel(
                "This model records no tunable hyperparameters to list."
                if single
                else "Hyperparameters were FIXED (the optimizer was not part of this run) — every "
                     "fold used the current form values, so there is no tuning spread to report."
            )
            note.setWordWrap(True)
            note.setStyleSheet("color: gray;")
            col.addWidget(note)

        return panel
