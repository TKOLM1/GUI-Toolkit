"""Whole-cloud read / copy / subset / write that preserves *every* point dimension.

Feature generation only needs a handful of arrays per file, so it has its own slim
reader (:func:`featuregen.io_las.load_plot`). Augmentation is different: it rewrites a
cloud and must hand back a file that is still a valid plot for *every* downstream
metric - RGB, classification, ``RelativeHeight`` and all. So here we work on the full
:class:`laspy.LasData` and its packed point record, never a reduced copy.

The four primitives the augmentation transforms build on:

* :func:`read_cloud`   - read a file into a ``LasData``.
* :func:`copy_cloud`   - an independent deep copy safe to mutate in place.
* :func:`take_points`  - a new cloud keeping only the selected point indices (all dims).
* :func:`write_cloud`  - write a ``LasData`` out (``.laz`` is compressed via lazrs).
"""

from __future__ import annotations

import copy
import shutil
from pathlib import Path

import laspy
import numpy as np


def read_cloud(path: str | Path) -> laspy.LasData:
    """Read ``path`` into a :class:`laspy.LasData` (all dimensions intact)."""
    return laspy.read(Path(path))


def copy_cloud(las: laspy.LasData) -> laspy.LasData:
    """Return an independent deep copy of ``las`` that is safe to mutate.

    The point record and the header are both copied, so transforms can overwrite
    coordinates / heights on the copy without touching the source cloud.
    """
    return laspy.LasData(header=copy.deepcopy(las.header), points=las.points.copy())


def take_points(las: laspy.LasData, indices: np.ndarray) -> laspy.LasData:
    """A new cloud containing only ``indices`` (a mask or integer index) of ``las``.

    Every point dimension is carried through because we index the packed point
    record itself. ``indices`` may repeat values (used by bootstrap resampling).
    """
    new = laspy.LasData(header=copy.deepcopy(las.header), points=las.points[indices].copy())
    return new


def is_copc(las: laspy.LasData) -> bool:
    """True if ``las`` carries COPC marker VLRs (a Cloud Optimized Point Cloud).

    A COPC file is read by laspy with a ``copc`` VLR on its header; laspy refuses to
    *write* a COPC, so any clipped/augmented output derived from such a source fails
    with "Writing COPC is not supported". We detect it so :func:`write_cloud` can
    rebuild a plain header before writing.
    """
    for store in (las.header.vlrs, las.header.evlrs):
        for vlr in store or []:
            if str(getattr(vlr, "user_id", "")).lower() == "copc":
                return True
    return False


def _decopc(las: laspy.LasData) -> laspy.LasData:
    """Return a copy of ``las`` with a clean, non-COPC header (points unchanged).

    A fresh :class:`laspy.LasHeader` is built from the source's version and point
    format, carrying over the scales/offsets and every VLR/EVLR *except* the ``copc``
    ones (the CRS lives in the WKT/GeoTIFF VLRs, so it is preserved). The packed point
    record is copied verbatim, so all dimensions survive.
    """
    src = las.header
    header = laspy.LasHeader(version=src.version, point_format=src.point_format)
    header.scales = src.scales
    header.offsets = src.offsets
    header.vlrs.extend(v for v in (src.vlrs or []) if str(getattr(v, "user_id", "")).lower() != "copc")
    evlrs = [v for v in (src.evlrs or []) if str(getattr(v, "user_id", "")).lower() != "copc"]
    if evlrs:
        header.evlrs = evlrs
    out = laspy.LasData(header)
    out.points = las.points.copy()
    return out


def write_cloud(las: laspy.LasData, path: str | Path) -> None:
    """Write ``las`` to ``path``; a ``.laz`` suffix triggers lazrs compression.

    If ``las`` came from a COPC source its header is rebuilt without the COPC VLRs
    first (laspy cannot write a COPC), so clipped/augmented outputs save cleanly. A
    normal (non-COPC) write path is unchanged.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if is_copc(las):
        las = _decopc(las)
    las.write(path)


def copy_to(src: str | Path, dst: str | Path) -> None:
    """Copy point cloud ``src`` to ``dst`` (``.laz``), re-encoding only when needed.

    A verbatim file copy is used for the ``.laz``→``.laz`` case, *except* for COPC sources: a raw
    copy would keep the COPC layout (which laspy cannot later re-write), so those are routed through
    :func:`write_cloud`, which strips the COPC header. Other formats are read and re-written as
    ``.laz``. Used when importing already-clipped plots and (historically) copying originals.
    """
    src, dst = Path(src), Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.suffix.lower() == ".laz" and dst.suffix.lower() == ".laz":
        las = read_cloud(src)
        if is_copc(las):
            write_cloud(las, dst)
        else:
            shutil.copyfile(src, dst)
    else:
        write_cloud(read_cloud(src), dst)


def point_count(path: str | Path) -> int:
    """The number of points in ``path``, read from its header only (no points loaded).

    Header-only so a whole project's plots can be totalled cheaply — the Clipping tab shows the
    total / per-plot average point count next to its "N plots loaded" indicator. Returns 0 for a
    file that cannot be opened, so one bad plot never breaks the readout.
    """
    try:
        with laspy.open(Path(path)) as reader:
            return int(reader.header.point_count)
    except Exception:  # noqa: BLE001 - unreadable / not a cloud
        return 0
