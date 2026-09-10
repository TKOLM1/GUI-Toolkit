"""Reading Point Cloud Data (``.pcd``) files into the same ``LasData`` the rest of the app uses.

Some plot datasets ship as per-plot ``.pcd`` files rather than ``.las``/``.laz`` (the SGCBP
wheat set is one), so the import tab accepts them and converts on the way in. This module is
the GUI-free reader: :func:`read_pcd` parses the text header, reads the point block, and hands
back a :class:`laspy.LasData` that :mod:`common.las_io` can write out as ``.laz`` unchanged.

Only the two layouts a scanner actually writes are supported - ``DATA ascii`` and
``DATA binary``. ``binary_compressed`` raises a clear error rather than guessing.

Field mapping is by name, so a file's own ``FIELDS`` line decides what is carried through:

* ``x y z``                     -> coordinates (required)
* ``intensity``                 -> LAS Intensity, rounded and clipped to uint16
* ``r``/``red``, ``g``, ``b``   -> LAS RGB, scaled from 0-255 to the LAS 16-bit range
* ``label``/``classification``  -> LAS Classification

Any other field is dropped: LAS has no place to put it, and the feature stack reads only the
dimensions above. Every point gets :data:`DEFAULT_CLASSIFICATION` when the file carries no
class field, so downstream ground/vegetation filters see a populated channel.

Coordinates are written through verbatim. A file recorded in millimetres stays in millimetres
here; converting to metres is the import tab's ``scale`` option, applied in
:mod:`clip.importer`, because the unit is a property of the dataset and not of the format.
"""

from __future__ import annotations

import re
from pathlib import Path

import laspy
import numpy as np

# Written to every point when the .pcd carries no class/label field. LAS code 1 is formally
# "Unclassified"; this project uses it as its plain "a real point" value, matching what the
# clipping path produces for clouds with no classification of their own.
DEFAULT_CLASSIFICATION = 1

# LAS stores coordinates as scaled integers. 0.0001 keeps sub-millimetre detail whether the
# source is in metres or millimetres, without overflowing the 32-bit integer range.
LAS_SCALE = 0.0001

# PCD numeric TYPE + SIZE -> numpy dtype. 'F' float, 'U' unsigned, 'I' signed.
_TYPE_MAP = {
    ("F", 4): np.float32, ("F", 8): np.float64,
    ("U", 1): np.uint8, ("U", 2): np.uint16, ("U", 4): np.uint32, ("U", 8): np.uint64,
    ("I", 1): np.int8, ("I", 2): np.int16, ("I", 4): np.int32, ("I", 8): np.int64,
}

# Field-name aliases, lower-cased, mapped onto the LAS dimension they feed.
_RED = {"r", "red"}
_GREEN = {"g", "green"}
_BLUE = {"b", "blue"}
_CLASS = {"label", "classification", "class"}

# PCD colour is conventionally 0-255 per channel; LAS RGB is 16-bit.
_RGB_GAIN = 257  # 255 * 257 == 65535, so full-scale maps to full-scale exactly


class PcdError(ValueError):
    """A ``.pcd`` file that cannot be read (bad header, unsupported layout, truncated data)."""


def _parse_header(raw: bytes) -> tuple[dict[str, list[str]], int]:
    """Parse the leading text header; return ``(fields, offset_of_point_data)``.

    The header is a run of ``KEY value...`` lines and ends at the ``DATA`` line - everything
    after that line's newline is the point block. Keys are upper-cased; comment lines (``#``)
    and blank lines are skipped.
    """
    header: dict[str, list[str]] = {}
    offset = 0
    while True:
        end = raw.find(b"\n", offset)
        if end < 0:
            raise PcdError("File ended inside the PCD header (no DATA line found).")
        line = raw[offset:end].decode("ascii", errors="replace").strip()
        offset = end + 1
        if not line or line.startswith("#"):
            continue
        key, _, rest = line.partition(" ")
        header[key.upper()] = rest.split()
        if key.upper() == "DATA":
            return header, offset


def _point_dtype(header: dict[str, list[str]]) -> np.dtype:
    """Build the structured numpy dtype describing one point, from FIELDS/SIZE/TYPE/COUNT."""
    names = header.get("FIELDS")
    sizes = header.get("SIZE")
    types = header.get("TYPE")
    if not (names and sizes and types):
        raise PcdError("PCD header is missing one of FIELDS / SIZE / TYPE.")
    if not (len(names) == len(sizes) == len(types)):
        raise PcdError("PCD header's FIELDS / SIZE / TYPE lines have different lengths.")
    counts = header.get("COUNT") or ["1"] * len(names)

    spec: list[tuple[str, object, tuple[int, ...]]] = []
    seen: dict[str, int] = {}
    for i, name in enumerate(names):
        try:
            dtype = _TYPE_MAP[(types[i].upper(), int(sizes[i]))]
        except (KeyError, ValueError) as exc:
            raise PcdError(f"Unsupported PCD field type '{types[i]}{sizes[i]}' for '{name}'.") from exc
        # '_' is PCD's padding-field name and may repeat; de-duplicate so numpy accepts the dtype.
        seen[name] = seen.get(name, 0) + 1
        unique = name if seen[name] == 1 else f"{name}__{seen[name]}"
        count = max(int(counts[i]), 1)
        spec.append((unique, dtype, (count,) if count > 1 else ()))
    return np.dtype([(n, d, s) if s else (n, d) for n, d, s in spec])


def _column(points: np.ndarray, names: set[str] | str) -> np.ndarray | None:
    """The first field of ``points`` whose lower-cased name is in ``names``, or ``None``."""
    wanted = {names} if isinstance(names, str) else names
    for field in points.dtype.names or ():
        if field.lower() in wanted:
            column = points[field]
            return column[:, 0] if column.ndim > 1 else column
    return None


def read_pcd(path: str | Path) -> laspy.LasData:
    """Read the ``.pcd`` at ``path`` into a :class:`laspy.LasData`.

    Coordinates are carried through in the file's own units (see the module note on scaling).
    Intensity, RGB and classification are mapped across when the file has them; classification
    defaults to :data:`DEFAULT_CLASSIFICATION`.

    Raises
    ------
    PcdError
        If the header is malformed, the layout is ``binary_compressed``, the file has no
        ``x``/``y``/``z`` fields, or the point block is truncated.
    """
    path = Path(path)
    raw = path.read_bytes()
    header, offset = _parse_header(raw)

    encoding = (header.get("DATA") or ["ascii"])[0].lower()
    dtype = _point_dtype(header)
    n_points = _declared_count(header)

    if encoding == "ascii":
        points = _read_ascii(raw[offset:], dtype, n_points, path)
    elif encoding == "binary":
        points = _read_binary(raw[offset:], dtype, n_points, path)
    else:
        raise PcdError(
            f"{path.name}: PCD encoding '{encoding}' is not supported (only 'ascii' and "
            f"'binary'). Re-save the file as one of those, e.g. with pcl_convert_pcd_ascii_binary."
        )
    return _to_las(points, path)


def _declared_count(header: dict[str, list[str]]) -> int | None:
    """The point count from POINTS, or WIDTH*HEIGHT, or ``None`` when neither is usable."""
    for key in ("POINTS",):
        values = header.get(key)
        if values:
            try:
                return int(values[0])
            except ValueError:
                pass
    try:
        return int((header.get("WIDTH") or [""])[0]) * int((header.get("HEIGHT") or [""])[0])
    except ValueError:
        return None


def _read_binary(body: bytes, dtype: np.dtype, n_points: int | None, path: Path) -> np.ndarray:
    """Interpret ``body`` as a packed array of ``dtype``, honouring the declared point count."""
    usable = len(body) - (len(body) % dtype.itemsize)
    points = np.frombuffer(body[:usable], dtype=dtype)
    if n_points is not None:
        if len(points) < n_points:
            raise PcdError(
                f"{path.name}: header declares {n_points} points but only {len(points)} are "
                f"present - the file looks truncated."
            )
        points = points[:n_points]
    if len(points) == 0:
        raise PcdError(f"{path.name}: the PCD contains no points.")
    return points


def _read_ascii(body: bytes, dtype: np.dtype, n_points: int | None, path: Path) -> np.ndarray:
    """Parse whitespace-separated ASCII rows into an array of ``dtype``.

    Each row holds one value per dtype *column* (a COUNT>1 field spreads over several), so the
    flat number stream is reshaped by the total column count and then packed field by field.
    """
    widths = [int(np.prod(dtype[name].shape or (1,))) for name in dtype.names or ()]
    total = sum(widths)
    try:
        flat = np.fromstring(body.decode("ascii", errors="replace"), sep=" ")  # type: ignore[attr-defined]
    except AttributeError:  # numpy >= 2 removed fromstring
        flat = np.array(body.decode("ascii", errors="replace").split(), dtype=np.float64)
    if total == 0 or flat.size < total:
        raise PcdError(f"{path.name}: the PCD contains no readable point rows.")
    rows = flat[: flat.size - (flat.size % total)].reshape(-1, total)
    if n_points is not None:
        rows = rows[:n_points]

    points = np.zeros(len(rows), dtype=dtype)
    at = 0
    for name, width in zip(dtype.names or (), widths):
        block = rows[:, at:at + width]
        points[name] = block.astype(dtype[name].base) if width > 1 else \
            block[:, 0].astype(dtype[name].base)
        at += width
    return points


def _to_las(points: np.ndarray, path: Path) -> laspy.LasData:
    """Map the parsed PCD fields onto a ``LasData`` (XYZ, intensity, class, and RGB when present)."""
    x, y, z = (_column(points, axis) for axis in ("x", "y", "z"))
    if x is None or y is None or z is None:
        present = ", ".join(points.dtype.names or ())
        raise PcdError(f"{path.name}: PCD has no x/y/z fields (found: {present}).")

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)

    # Only promote to a colour-carrying point format when the PCD actually has colour. Format 3
    # always allocates red/green/blue, so writing it unconditionally left every plot with three
    # all-zero colour channels - which then showed up in the Feature tab's channel dropdowns as
    # though the scan had colour. Format 0 carries XYZ, intensity and classification, which is all
    # a colourless PCD can fill.
    red, green, blue = _column(points, _RED), _column(points, _GREEN), _column(points, _BLUE)
    has_colour = red is not None and green is not None and blue is not None
    header = laspy.LasHeader(point_format=3 if has_colour else 0, version="1.2")
    header.scales = [LAS_SCALE, LAS_SCALE, LAS_SCALE]
    # Anchor the offsets on the cloud's own minimum corner rather than zero. LAS stores each
    # coordinate as the 32-bit integer (value - offset) / scale, so an offset of zero caps the
    # usable range at about 2.1e9 * scale - only ~214 metres at this precision, and a scan
    # carrying real survey coordinates blows straight through it (SGCBP's later runs sit at
    # y ~ 218,000 mm, which overflows and makes laspy refuse the file). Offsetting by the minimum
    # means only the plot's own extent has to fit, which it always does.
    header.offsets = [float(x.min()), float(y.min()), float(z.min())]
    las = laspy.LasData(header)
    las.x = x
    las.y = y
    las.z = z

    intensity = _column(points, "intensity")
    if intensity is not None:
        las.intensity = np.clip(np.rint(np.asarray(intensity, dtype=np.float64)),
                                0, 65535).astype(np.uint16)

    if has_colour:
        for channel, values in (("red", red), ("green", green), ("blue", blue)):
            scaled = np.clip(np.rint(np.asarray(values, dtype=np.float64)), 0, 255)
            setattr(las, channel, (scaled * _RGB_GAIN).astype(np.uint16))

    classification = _column(points, _CLASS)
    if classification is not None:
        las.classification = np.clip(
            np.rint(np.asarray(classification, dtype=np.float64)), 0, 255
        ).astype(np.uint8)
    else:
        las.classification = np.full(len(las.x), DEFAULT_CLASSIFICATION, dtype=np.uint8)
    return las
