"""The augmentation batch: augmented copies + a manifest, written beside the originals.

:func:`run_augment` is GUI-free (the GUI's worker thread calls it and forwards the
``progress`` callback) and is the augmentation counterpart of
:func:`featuregen.pipeline.run_batch`.

Input and output share the project's ``plots/`` folder: the ``plot(N).laz`` originals are
already there (written by the clip/import step), so augmentation does **not** re-copy them —
that would just duplicate every file. For every input file it:

* writes ``n_per_sample`` **augmented** copies, each named ``<stem>_aug(N).laz`` where
  ``aug(N)`` restarts at 1 for every plot (per-plot numbering) while ``plot(N)`` is preserved.

The original is still recorded in the manifest as ``aug_id = 0`` (pointing at the existing
``plot(N).laz``), so the manifest accounts for every plot file even though no copy was made.

Reproducibility: a master :class:`numpy.random.SeedSequence` is built from
``config.seed`` and spawns one child generator per augmented sample, so the same
config + seed reproduces the same plans and noise.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from common import las_io
from common.naming import build_aug_name, build_copy_name, plot_number_from_name
from .manifest import MANIFEST_NAME, ManifestRow, write_manifest
from .plan import AugmentConfig, sample_plan
from .transforms import AUG_BY_KEY

# Called once per unit of work with (done, total, file_name, error_or_None).
ProgressCallback = Callable[[int, int, str, str | None], None]


@dataclass
class AugmentResult:
    """Outcome of an augmentation run."""

    output_dir: Path
    manifest_path: Path
    augmented: list[Path] = field(default_factory=list)
    failed: list[tuple[str, str]] = field(default_factory=list)  # (file_name, reason)

    @property
    def n_augmented(self) -> int:
        return len(self.augmented)


def run_augment(
    files: list[str | Path],
    output_dir: str | Path,
    config: AugmentConfig,
    manifest_name: str = MANIFEST_NAME,
    progress: ProgressCallback | None = None,
) -> AugmentResult:
    """Augment ``files`` into ``output_dir`` and write the manifest. See module docstring.

    ``files`` are the ``plot(N).laz`` originals (already in ``output_dir``); they are not re-copied,
    only augmented. ``manifest_name`` is the manifest file name (``.csv``).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    files = [Path(f) for f in files]

    suffix = config.out_suffix
    total = len(files) * config.n_per_sample  # only augmented copies are written
    done = 0

    seed_seq = np.random.SeedSequence(config.seed)
    child_seeds = iter(seed_seq.spawn(max(1, len(files) * config.n_per_sample)))

    result = AugmentResult(output_dir=output_dir, manifest_path=output_dir / manifest_name)
    rows: list[ManifestRow] = []

    for src in files:
        plot = plot_number_from_name(src.name)
        # Record the existing original as aug_id=0 (no file is written — it is already on disk as
        # the canonical plot(N).laz the clip/import step produced).
        rows.append(ManifestRow(0, src.name, build_copy_name(src.name, suffix), plot, []))

        # Read the source cloud once; each augmented sample mutates a fresh copy of it.
        try:
            base_cloud = las_io.read_cloud(src) if config.n_per_sample > 0 else None
        except Exception as exc:  # noqa: BLE001 - cannot augment an unreadable file
            base_cloud = None
            for _ in range(config.n_per_sample):
                next(child_seeds, None)
                result.failed.append((src.name, f"read failed: {exc}"))
                done += 1
                if progress is not None:
                    progress(done, total, src.name, str(exc))
            continue

        # Augmented copies - numbered aug(1..n_per_sample) *within this plot*.
        for j in range(config.n_per_sample):
            aug_id = j + 1
            rng = np.random.default_rng(next(child_seeds))
            out_name = build_aug_name(src.name, aug_id, suffix)
            out_path = output_dir / out_name
            try:
                plan = sample_plan(config, rng)
                cloud = las_io.copy_cloud(base_cloud)
                for key, params in plan:
                    AUG_BY_KEY[key].apply(cloud, params, rng, config.height_channel)
                las_io.write_cloud(cloud, out_path)
                result.augmented.append(out_path)
                rows.append(ManifestRow(aug_id, src.name, out_name, plot, plan))
                err = None
            except Exception as exc:  # noqa: BLE001 - record and continue
                result.failed.append((out_name, str(exc)))
                err = str(exc)
            done += 1
            if progress is not None:
                progress(done, total, out_name, err)

    write_manifest(rows, config, result.manifest_path)
    return result
