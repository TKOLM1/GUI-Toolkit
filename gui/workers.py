"""Background worker threads so heavy work never blocks the UI.

One worker per module, each wrapping the module's GUI-free core function and re-emitting
its progress as Qt signals the pages connect to.
"""

from __future__ import annotations

import warnings
from contextlib import contextmanager
from pathlib import Path

from PySide6.QtCore import QThread, Signal


def _warning_kind(category, message) -> str:
    """A stable de-dup key for a warning: its category + first sentence, with run-specific numbers
    stripped. sklearn's ConvergenceWarning embeds the per-fit duality gap and tolerance (e.g.
    "Duality gap: 2.48e+07, tolerance: 7.39e+04"), so the raw text differs every fit and naive
    string de-dup never collapses them. Keying on the leading sentence groups all the convergence
    warnings under one kind, so the console shows it once with a repeat count."""
    head = str(message).split(".")[0].strip()  # first sentence — the stable part
    return f"{category.__name__}: {head}"


@contextmanager
def _capture_warnings(emit, *, is_enabled=lambda: True):
    """Route every :mod:`warnings` warning raised *inside this block* to ``emit(text)``.

    The classical fitting/optimization code (sklearn, numpy) reports problems — slow convergence,
    degenerate folds, deprecations — via ``warnings.warn``. By default those only reach the
    background terminal; this context manager redirects them into the in-UI console for the duration
    of a worker's run, so the user sees what the training is complaining about. Scoped to the block:
    ``catch_warnings`` restores the previous handler (and filters) on exit, so it never leaks into
    other threads' work.

    De-dup: warnings are grouped by :func:`_warning_kind` (category + first sentence, numbers
    stripped), so a convergence warning fired once per CV fold — each with a different duality-gap
    number — collapses to one console line plus a final "(N more)" tally instead of flooding it.

    ``is_enabled`` is polled per warning (it reads a GUI checkbox): when it returns False the warning
    is dropped entirely — neither emitted nor tallied — so the user can silence console warnings.
    Capturing still happens regardless, so re-enabling mid-run takes effect for later warnings.

    Only one classical worker runs at a time (a run disables the Run button), so this process-global
    handler swap is safe — there is no second run concurrently catching warnings.
    """
    counts: dict[str, int] = {}

    def _show(message, category, filename, lineno, file=None, line=None):
        if not is_enabled():
            return  # warnings muted by the console toggle
        kind = _warning_kind(category, message)
        counts[kind] = counts.get(kind, 0) + 1
        if counts[kind] == 1:  # emit each distinct KIND once, live; tally the repeats for the summary
            emit(f"⚠ {category.__name__}: {message}")

    with warnings.catch_warnings():
        warnings.simplefilter("always")  # defeat the "once per location" cache so every fold's warning is seen
        warnings.showwarning = _show
        try:
            yield
        finally:
            repeats = {k: n for k, n in counts.items() if n > 1}
            if repeats and is_enabled():
                total_extra = sum(n - 1 for n in repeats.values())
                kinds = "kind" if len(repeats) == 1 else f"{len(repeats)} kinds"
                emit(f"⚠ (+{total_extra} more of the above {kinds} — the same warning(s) repeated "
                     "across splits/folds/trials.)")

from augment import AugmentConfig, run_augment
from clip import GridLayout, clip_to_masks, import_plots
from featuregen.external import ExternalData
from featuregen.io_las import Config
from featuregen.pipeline import run_batch
from ml import (
    Dataset,
    TrainConfig,
    TrainControl,
    TrainingStopped,
    permutation_importance,
    sweep_models,
    validate_procedure,
)


class BatchWorker(QThread):
    """Runs one feature-generation batch off the UI thread."""

    progressed = Signal(int, int, str, object)  # done, total, file_name, error
    finished_ok = Signal(object)                # BatchResult
    failed = Signal(str)

    def __init__(
        self,
        files,
        output_path: Path,
        feature_keys: list[str],
        config: Config,
        external: ExternalData | None = None,
        external_columns: list[str] | None = None,
        encode_columns: list[str] | None = None,
        target_column: str | None = None,
        targets_path: Path | None = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._files = files
        self._output_path = output_path
        self._feature_keys = feature_keys
        self._config = config
        self._external = external
        self._external_columns = external_columns or []
        self._encode_columns = encode_columns or []
        self._target_column = target_column
        self._targets_path = targets_path

    def run(self) -> None:
        try:
            result = run_batch(
                self._files,
                self._output_path,
                feature_keys=self._feature_keys,
                config=self._config,
                external=self._external,
                external_columns=self._external_columns,
                encode_columns=self._encode_columns,
                target_column=self._target_column,
                targets_path=self._targets_path,
                progress=lambda d, t, n, e: self.progressed.emit(d, t, n, e),
            )
            self.finished_ok.emit(result)
        except Exception as exc:  # noqa: BLE001 - report any failure to the UI
            self.failed.emit(str(exc))


class ClipWorker(QThread):
    """Runs one clipping batch (cloud -> per-plot files) off the UI thread."""

    progressed = Signal(int, int, str, object)  # done, total, label, error
    finished_ok = Signal(object)                # ClipResult
    failed = Signal(str)

    def __init__(self, cloud_path, masks, output_dir: Path, universal_string: str, parent=None) -> None:
        super().__init__(parent)
        self._cloud_path = cloud_path
        self._masks = masks
        self._output_dir = output_dir
        self._universal_string = universal_string

    def run(self) -> None:
        try:
            result = clip_to_masks(
                self._cloud_path,
                self._masks,
                self._output_dir,
                universal_string=self._universal_string,
                progress=lambda d, t, n, e: self.progressed.emit(d, t, n, e),
            )
            self.finished_ok.emit(result)
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc))


class ImportWorker(QThread):
    """Runs one folder import (per-plot files -> canonical plot(N) files) off the UI thread.

    The read/convert/write loop is heavy for a large dataset - a few hundred ``.pcd`` files are
    parsed, optionally rescaled and re-positioned, then LAZ-compressed - so it gets the same
    treatment as clipping rather than freezing the window.
    """

    progressed = Signal(int, int, str, object)  # done, total, source_name, error
    finished_ok = Signal(object)                # ImportResult
    failed = Signal(str)

    def __init__(self, sources, output_dir: Path, fmt, universal_string: str,
                 scale: float, grid: GridLayout | None, parent=None) -> None:
        super().__init__(parent)
        self._sources = sources
        self._output_dir = output_dir
        self._fmt = fmt
        self._universal_string = universal_string
        self._scale = scale
        self._grid = grid

    def run(self) -> None:
        try:
            result = import_plots(
                self._sources,
                self._output_dir,
                self._fmt,
                universal_string=self._universal_string,
                scale=self._scale,
                grid=self._grid,
                progress=lambda d, t, n, e: self.progressed.emit(d, t, n, e),
            )
            self.finished_ok.emit(result)
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc))


class AugmentWorker(QThread):
    """Runs one augmentation batch off the UI thread."""

    progressed = Signal(int, int, str, object)  # done, total, file_name, error
    finished_ok = Signal(object)                # AugmentResult
    failed = Signal(str)

    def __init__(self, files, output_dir: Path, config: AugmentConfig,
                 manifest_name: str | None = None, parent=None) -> None:
        super().__init__(parent)
        self._files = files
        self._output_dir = output_dir
        self._config = config
        self._manifest_name = manifest_name

    def run(self) -> None:
        try:
            kwargs = {"manifest_name": self._manifest_name} if self._manifest_name else {}
            result = run_augment(
                self._files,
                self._output_dir,
                self._config,
                progress=lambda d, t, n, e: self.progressed.emit(d, t, n, e),
                **kwargs,
            )
            self.finished_ok.emit(result)
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc))


class ValidateWorker(QThread):
    """Runs the rotating-vault nested CV (validate_procedure) off the UI thread.

    The procedure re-runs the whole optimize→train inside each outer fold, so this is the heaviest
    classical worker; the single ``TrainControl`` is threaded into every inner optimizer call
    (validate_procedure handles that) so Pause/Stop reach the loops.

    With ``model_keys`` given the worker instead sweeps :func:`ml.sweep_models` over those models under
    the same settings and emits ``model_done`` per finished model; ``finished_ok`` then carries the
    winning :class:`~ml.VaultResult` and ``sweep_done`` the full ranked list.
    """

    # fold, total, held_out_rrmse, train_rrmse, train_metrics, test_metrics. The two trailing dicts are
    # the fold's full standard_metrics() for the outer-train fit and outer-test rows, so the GUI can
    # plot every metric per outer fold, not only rRMSE.
    fold_done = Signal(int, int, float, float, object, object)
    # fold, n_outer, stage, trial, total — per inner feature-selection/optimizer trial, so the GUI's
    # live status line keeps ticking through a fold's long inner search.
    inner_progressed = Signal(int, int, str, int, int)
    # model_index, n_models, model_key, VaultResult|None — one per model of a sweep, so the GUI can
    # report each model's score as it lands rather than only at the end.
    model_done = Signal(int, int, str, object)
    sweep_done = Signal(object)                  # list[ml.VaultResult], best first (sweep runs only)
    noted = Signal(str)                          # one-off informational line for the log
    warned = Signal(str)                         # a training-run warning, for the console
    finished_ok = Signal(object)                 # ml.VaultResult (the winner, for a sweep)
    failed = Signal(str)

    # Polled per warning to decide whether to surface it in the console; the page replaces this
    # with its "show warnings" checkbox reader. Default-on so the worker is usable without the GUI.
    warnings_enabled = staticmethod(lambda: True)

    def __init__(self, dataset: Dataset, config: TrainConfig, kwargs: dict,
                 model_keys: list[str] | None = None, parent=None) -> None:
        super().__init__(parent)
        self._dataset = dataset
        self._config = config
        # The do_optimize flag + inner knobs (mirrors the GUI widgets); passed straight through to
        # validate_procedure so the worker stays a thin wrapper.
        self._kwargs = dict(kwargs)
        # None -> validate the one model in ``config``; a list -> sweep those models and keep the best.
        self._model_keys = list(model_keys) if model_keys else None
        self._control = TrainControl()

    def pause(self) -> None:
        self._control.pause()

    def resume(self) -> None:
        self._control.resume()

    def stop(self) -> None:
        self._control.stop()

    def run(self) -> None:
        try:
            with _capture_warnings(self.warned.emit, is_enabled=self.warnings_enabled):
                shared = dict(
                    note=lambda msg: self.noted.emit(msg),
                    progress=lambda i, t, ho, tr, trm, tem: self.fold_done.emit(
                        i, t, ho, tr, trm, tem),
                    inner_progress=lambda f, n, stage, tr, tot: self.inner_progressed.emit(
                        f, n, stage, tr, tot),
                    control=self._control,
                )
                if self._model_keys is None:
                    result = validate_procedure(
                        self._dataset, self._config, **shared, **self._kwargs
                    )
                    ranked = None
                else:
                    ranked = sweep_models(
                        self._dataset, self._config, model_keys=self._model_keys,
                        on_model=lambda i, n, key, res: self.model_done.emit(i, n, key, res),
                        **shared, **self._kwargs,
                    )
                    result = ranked[0]  # sweep_models sorts best-first (and raises if none finished)
            if ranked is not None:
                self.sweep_done.emit(ranked)
            self.finished_ok.emit(result)
        except TrainingStopped:
            self.failed.emit("Validation was stopped before any outer fold finished — nothing to show.")
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc))


class ResultsWorker(QThread):
    """Computes per-plot predictions for the results map off the UI thread, from a loaded bundle."""

    finished_ok = Signal(object)  # per-plot predictions DataFrame
    failed = Signal(str)

    def __init__(self, loaded, source, parent=None) -> None:
        super().__init__(parent)
        self._loaded = loaded   # ml.LoadedModel
        self._source = source   # ml.Dataset

    def run(self) -> None:
        try:
            table = self._loaded.predict_table(self._source)
            self.finished_ok.emit(table)
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc))


class ImportanceWorker(QThread):
    """Computes permutation feature importance off the UI thread (≈ features × repeats predictions)."""

    finished_ok = Signal(object)  # ml.ImportanceResult
    failed = Signal(str)

    def __init__(
        self, dataset, history, n_repeats: int, score_on: str, include_aug: bool, seed: int,
        n_jobs: int = 1, parent=None,
    ) -> None:
        super().__init__(parent)
        self._dataset = dataset      # ml.Dataset (classical only)
        self._history = history      # ml.TrainHistory of the active split's bundle
        self._n_repeats = n_repeats
        self._score_on = score_on
        self._include_aug = include_aug  # honour the Results-tab aug toggle for the scored subset
        self._seed = seed
        self._n_jobs = n_jobs

    def run(self) -> None:
        try:
            result = permutation_importance(
                self._dataset,
                self._history,
                n_repeats=self._n_repeats,
                score_on=self._score_on,
                include_aug=self._include_aug,
                seed=self._seed,
                n_jobs=self._n_jobs,
            )
            self.finished_ok.emit(result)
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc))
