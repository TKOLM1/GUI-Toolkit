"""A self-describing saved-model bundle for classical (scikit-learn) models.

The Results tab no longer reads the last in-memory training run; instead the user picks a **saved
model** and Results computes the field map from it. For that to work a saved bundle has to carry
everything needed to (a) redraw the per-plot map by predicting a data source, and (b) show the
model-performance scatter from the held-out splits.

A bundle is one ``.joblib`` dict holding the full :class:`~ml.trainer.TrainHistory` (every split,
each split's fitted model + held-out predictions + plot sets) plus a small ``kind`` tag and the
input spec. :func:`load_bundle` wraps it in a :class:`LoadedModel` whose :meth:`predict_table`
returns the same ``plot, aug, actual, predicted, error_pct, role`` table
:func:`ml.results.compute_plot_predictions` produces, predicting from the feature frame.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from common.naming import aug_number_from_name, plot_number_from_name
from .trainer import TrainHistory, active_split_model

BUNDLE_SUFFIX = ".joblib"
# A tiny companion file written next to each bundle holding just the BundleInfo metadata, so the
# Results dropdown can list models without deserialising the whole (multi-MB) bundle. Read by
# bundle_info(); regenerated from the bundle if missing (back-fills old bundles on first listing).
INFO_SUFFIX = ".info.json"
# Bundle schema version, so a future change can be detected on load.
# 3: each SplitModel carries a precomputed metric cube (SplitModel.metric_variants).
# 4: each SplitModel also carries a full per-split prediction table (SplitModel.plot_predictions),
#    so Results builds its per-plot table by lookup without rebuilding the data source or re-predicting.
_FORMAT = 4


def _safe_stem(name: str | None) -> str:
    """Sanitise a user-typed bundle name into a safe file stem (or ``""`` if nothing usable).

    Strips any ``.joblib`` extension, drops path separators and characters illegal on Windows
    filesystems, and collapses whitespace to underscores — so a free-text Save name can never write
    outside ``out_dir`` or produce an invalid filename. An empty result signals "fall back to the
    default stem".
    """
    if not name:
        return ""
    stem = name.strip()
    if stem.lower().endswith(BUNDLE_SUFFIX):
        stem = stem[: -len(BUNDLE_SUFFIX)]
    stem = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "", stem)  # illegal on Windows / path separators
    stem = re.sub(r"\s+", "_", stem).strip("._ ")
    return stem


def _assemble_table(names, actual, predicted, train_plots, plots=None, augs=None) -> pd.DataFrame:
    """Build the standard per-plot results table from raw predictions.

    Shared by the stored-prediction path (:meth:`LoadedModel.stored_table`) and the live-predict
    path (:meth:`LoadedModel.predict_table`) so both emit identical columns: ``plot, aug, actual,
    predicted, error_pct, role`` indexed by file name. ``error_pct`` is derived from actual/predicted
    and ``role`` from membership in ``train_plots`` (``'T'`` train / ``'V'`` held-out). ``plots`` /
    ``augs`` may be supplied (stored path) to skip re-parsing the file names.
    """
    if len(names) == 0:
        return pd.DataFrame(columns=["plot", "aug", "actual", "predicted", "error_pct", "role"])
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        error_pct = np.where(actual != 0, np.abs(predicted - actual) / np.abs(actual) * 100.0, np.nan)
    if plots is None:
        plots = [plot_number_from_name(str(n)) for n in names]
    if augs is None:
        augs = [aug_number_from_name(str(n)) or 0 for n in names]
    roles = ["T" if p in train_plots else "V" for p in plots]
    frame = pd.DataFrame(
        {"plot": plots, "aug": augs, "actual": actual,
         "predicted": predicted, "error_pct": error_pct, "role": roles},
        index=list(names),
    )
    frame.index.name = "filename"
    return frame


def save_bundle(
    history: TrainHistory,
    out_dir: str | Path,
    *,
    model_key: str,
    vault_meta: dict | None = None,
    name: str | None = None,
) -> Path:
    """Write ``history`` as a ``<model_key>_<timestamp>.joblib`` bundle in ``out_dir``; return its path.

    The whole :class:`TrainHistory` is stored (all splits), so the Results tab can recompute the map
    for any split and draw the performance scatter.

    ``vault_meta`` marks a bundle saved from a rotating-vault (nested-CV) run: its splits are the
    *outer folds* of an honest measurement, not random train/test cycles. When present it carries the
    per-fold selection stability (``fold_features`` / ``fold_params``), the honest mean ± std and which
    stages the vault wrapped, so the Results tab can label and report it as a nested-CV bundle. ``None``
    (the default) is a normal trained model.

    ``name`` is an optional user-chosen file stem (the ``.joblib`` extension is added). It only sets the
    on-disk file name — the Results dropdown label is derived from the bundle metadata regardless — and
    is sanitised to a safe stem; ``None`` falls back to ``<model_key>_<timestamp>``.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = _safe_stem(name) or f"{model_key}_{stamp}"
    path = out_dir / f"{stem}{BUNDLE_SUFFIX}"

    payload: dict[str, Any] = {
        "format": _FORMAT,
        "kind": "classical",
        "model_key": model_key,
        "saved_at": stamp,
        "source": "nested_cv" if vault_meta is not None else "train",
        "vault_meta": vault_meta,
        "target_column": history.target_column,
        "feature_columns": list(history.feature_columns),
        "history": history,
    }
    joblib.dump(payload, path)
    _write_info_sidecar(path, _info_from_payload(path, payload))
    return path


@dataclass
class BundleInfo:
    """Lightweight metadata for listing bundles in the Results dropdown (no model loaded)."""

    path: Path
    kind: str
    model_key: str
    saved_at: str
    target_column: str
    rrmse: float
    validate_on_augmented: bool = False  # were augmented copies scored in the held-out set?
    source: str = "train"  # "train" (normal model) or "nested_cv" (rotating-vault measurement)

    @property
    def label(self) -> str:
        aug_tag = " (AUG)" if self.validate_on_augmented else ""
        cv_tag = " (nested CV)" if self.source == "nested_cv" else ""
        return (
            f"[ML] {self.model_key}{aug_tag}{cv_tag} — {self.target_column} "
            f"(rRMSE {self.rrmse:.3g}%, {self.saved_at})"
        )


@dataclass
class LoadedModel:
    """A loaded bundle ready to drive the Results tab."""

    path: Path
    kind: str
    model_key: str
    target_column: str
    feature_columns: list[str]
    history: TrainHistory
    source: str = "train"  # "train" or "nested_cv"
    vault_meta: dict | None = None  # per-fold stability + honest label, for nested-CV bundles

    @property
    def metrics(self) -> dict:
        return self.history.final_metrics

    @property
    def is_nested_cv(self) -> bool:
        """True when this bundle is a rotating-vault (nested-CV) measurement rather than a model."""
        return self.source == "nested_cv"

    @property
    def has_stored_predictions(self) -> bool:
        """True when the bundle carries per-split stored predictions (so :meth:`stored_table` works)."""
        from .trainer import has_stored_predictions
        return has_stored_predictions(self.history)

    def stored_table(self) -> pd.DataFrame:
        """The per-plot table for the active split, built purely from stored predictions.

        No data source, no model — just the active split's stored ``plot_predictions``
        with ``error_pct`` and ``role`` derived (role from that split's ``train_plots``). Same columns
        as :meth:`predict_table` / :func:`ml.results.compute_plot_predictions`. Raises if the bundle
        has no stored predictions, so the caller can fall back to :meth:`predict_table` for older
        bundles (see :attr:`has_stored_predictions`).
        """
        active = active_split_model(self.history)
        stored = getattr(active, "plot_predictions", None) if active is not None else None
        if stored is None or stored.empty:
            raise ValueError("This bundle has no stored predictions for the active split.")
        plots = stored["plot"].tolist() if "plot" in stored else None
        augs = stored["aug"].tolist() if "aug" in stored else None
        return _assemble_table(
            names=list(stored.index),
            actual=stored["actual"].to_numpy(dtype=float),
            predicted=stored["predicted"].to_numpy(dtype=float),
            train_plots=active.train_plots,
            plots=plots,
            augs=augs,
        )

    def predict_table(self, source) -> pd.DataFrame:
        """Per-plot predictions over ``source`` using the active split's model.

        ``source`` is an :class:`ml.dataset.Dataset`. Returns the same columns as
        :func:`ml.results.compute_plot_predictions`: ``plot, aug, actual, predicted, error_pct, role``
        (indexed by file name). ``role`` is ``'T'`` if the plot was in the active split's train set,
        else ``'V'``. The legacy path for bundles without stored predictions (format ≤ 3); newer
        bundles read :meth:`stored_table` instead and never touch the source.
        """
        active = active_split_model(self.history)
        model = active.model if active is not None else self.history.model
        train_plots = active.train_plots if active is not None else self.history.best_train_plots

        names, actual, predicted = self._predict_classical(model, source)

        frame = _assemble_table(
            names=list(names), actual=actual, predicted=predicted, train_plots=train_plots,
        )
        return frame

    # -- per-kind prediction ------------------------------------------------ #
    def _predict_classical(self, model, dataset):
        from .dataset import make_xy  # local import keeps ml.bundle import-light

        X, y, _ = make_xy(dataset, self.feature_columns)
        if len(X) == 0:
            return [], np.empty(0), np.empty(0)
        predicted = np.ravel(model.predict(X))
        return list(X.index), y.to_numpy(dtype=float), predicted


def _peek(path: Path) -> dict:
    """Read a bundle's payload dict (the model objects load lazily with joblib's mmap-free read)."""
    return joblib.load(path)


# The BundleInfo fields persisted in the sidecar (everything except ``path``, which is the location).
_INFO_FIELDS = ("kind", "model_key", "saved_at", "target_column", "rrmse", "validate_on_augmented",
                "source")


def _info_from_payload(path: Path, payload: dict) -> BundleInfo:
    """Build a :class:`BundleInfo` from a loaded bundle payload dict.

    Shared by :func:`save_bundle` (to write the sidecar) and the joblib fallback in
    :func:`bundle_info`, so the metadata is identical however it was produced. Raises if ``payload``
    is not a recognised bundle (no ``history``).
    """
    if not isinstance(payload, dict) or "history" not in payload:
        raise ValueError(f"{path.name} is not a recognised model bundle.")
    history = payload.get("history")
    rrmse = float(history.final_metrics.get("rrmse", float("nan"))) if history is not None else float("nan")
    return BundleInfo(
        path=path,
        kind=str(payload.get("kind", "classical")),
        model_key=str(payload.get("model_key", path.stem)),
        saved_at=str(payload.get("saved_at", "")),
        target_column=str(payload.get("target_column", "")),
        rrmse=rrmse,
        validate_on_augmented=bool(getattr(history, "validate_on_augmented", False)),
        source=str(payload.get("source", "train")),
    )


def _info_sidecar_path(bundle_path: Path) -> Path:
    """The sidecar path for a bundle (``foo.joblib`` -> ``foo.info.json``)."""
    return bundle_path.with_name(bundle_path.name[: -len(BUNDLE_SUFFIX)] + INFO_SUFFIX)


def _write_info_sidecar(bundle_path: Path, info: BundleInfo) -> None:
    """Write ``info`` next to its bundle as JSON. Best-effort: failure never blocks save/listing."""
    data = {field: getattr(info, field) for field in _INFO_FIELDS}
    try:
        _info_sidecar_path(bundle_path).write_text(json.dumps(data), encoding="utf-8")
    except OSError:
        pass  # a missing sidecar just means the next listing reads the bundle once more


def _info_from_sidecar(bundle_path: Path) -> BundleInfo | None:
    """Read a bundle's metadata from its JSON sidecar, or None if it's absent/unreadable/stale.

    Returns None (rather than raising) on any problem so the caller falls back to the bundle itself;
    a sidecar missing a field is treated as stale so a schema change can't surface partial metadata.
    A sidecar older than its bundle is also treated as stale, so a bundle rewritten by anything other
    than :func:`save_bundle` (a restored backup, a manual copy) self-corrects on the next listing.
    """
    sidecar = _info_sidecar_path(bundle_path)
    if not sidecar.is_file():
        return None
    try:
        if sidecar.stat().st_mtime < bundle_path.stat().st_mtime:
            return None  # bundle rewritten after the sidecar -> stale, re-read the bundle
    except OSError:
        return None
    try:
        data = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict) or any(field not in data for field in _INFO_FIELDS):
        return None
    return BundleInfo(path=bundle_path, **{field: data[field] for field in _INFO_FIELDS})


def bundle_info(path: str | Path) -> BundleInfo:
    """Read just the metadata of a bundle for the dropdown.

    Reads the tiny JSON sidecar when present (instant); otherwise loads the bundle once to build the
    metadata and back-fills the sidecar so subsequent listings are fast. Raises if ``path`` is not a
    recognised bundle (no ``history``), so :func:`list_bundles` can skip stray ``.joblib`` files.
    """
    path = Path(path)
    cached = _info_from_sidecar(path)
    if cached is not None:
        return cached
    info = _info_from_payload(path, _peek(path))
    _write_info_sidecar(path, info)  # back-fill so the slow read happens at most once
    return info


def delete_bundle(path: str | Path) -> None:
    """Delete a bundle and its JSON sidecar (if any). Missing files are ignored."""
    path = Path(path)
    for target in (path, _info_sidecar_path(path)):
        try:
            target.unlink()
        except FileNotFoundError:
            pass


def prune_orphan_info_sidecars(folder: str | Path) -> None:
    """Delete ``*.info.json`` sidecars in ``folder`` whose bundle no longer exists.

    Called after a bundle is deleted by other means (the generic delete dialog removes the
    ``.joblib`` directly), so the folder doesn't accumulate stale metadata files. Best-effort.
    """
    folder = Path(folder)
    if not folder.is_dir():
        return
    for sidecar in folder.glob(f"*{INFO_SUFFIX}"):
        bundle = sidecar.with_name(sidecar.name[: -len(INFO_SUFFIX)] + BUNDLE_SUFFIX)
        if not bundle.exists():
            try:
                sidecar.unlink()
            except OSError:
                pass


def load_bundle(path: str | Path) -> LoadedModel:
    """Load a bundle written by :func:`save_bundle` into a :class:`LoadedModel`."""
    path = Path(path)
    payload = _peek(path)
    if "history" not in payload:
        raise ValueError(f"{path.name} is not a recognised model bundle.")
    return LoadedModel(
        path=path,
        kind=str(payload.get("kind", "classical")),
        model_key=str(payload.get("model_key", path.stem)),
        target_column=str(payload.get("target_column", "")),
        feature_columns=list(payload.get("feature_columns", [])),
        history=payload["history"],
        source=str(payload.get("source", "train")),
        vault_meta=payload.get("vault_meta"),
    )


def list_bundles(folder: str | Path) -> list[BundleInfo]:
    """All bundles in ``folder`` (newest first), as lightweight :class:`BundleInfo`.

    Unreadable / non-bundle ``.joblib`` files are skipped so a stray file never breaks the dropdown.
    """
    folder = Path(folder)
    if not folder.is_dir():
        return []
    infos: list[BundleInfo] = []
    for path in folder.glob(f"*{BUNDLE_SUFFIX}"):
        try:
            infos.append(bundle_info(path))
        except Exception:  # noqa: BLE001 - tolerate foreign .joblib files
            continue
    infos.sort(key=lambda i: i.saved_at, reverse=True)
    return infos
