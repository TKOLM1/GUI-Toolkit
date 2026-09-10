"""Grouped splitting - the single most important guard against data leakage.

Every augmented copy of a plot shares that plot's ``plot(N)`` number, and the target
value is identical across those copies. If a plot's copies were split across train and
test, the model would effectively be tested on data it trained on. So the train/test
split is done **by group** (plot number): a whole plot lands entirely on one side.

A training run repeats this grouped split several times; a whole plot always lands entirely on one
side of every split, so no copy can leak across. Four split **modes** are offered:

* ``"random"`` — each cycle draws a fresh random grouped split (``GroupShuffleSplit``), seeded by
  ``seed + cycle`` so the run reproduces. Successive cycles draw **independently** (with replacement
  across cycles), so a plot may be held out in several cycles or none — the only mode whose folds are
  not a partition. The number of splits is chosen by the user and is unbounded by the ratio.
* ``"random_systematic"`` — a randomised **partition**: the unique plot groups are shuffled *once*
  (seeded by ``seed`` only, shared across cycles) and then dealt into ``n_splits`` contiguous blocks of
  ``round(test_size · n_groups)`` plots each, exactly like sequential but on a random order instead of
  the numeric one. So cycle 0 draws a random hold-out block, cycle 1 draws another random block **from
  the plots not yet drawn**, and so on — random sampling **without replacement** across the splits. This
  needs ``n_splits · block ≤ n_groups`` (the GUI enforces ``splits · ratio ≤ 1``), so every cycle gets a
  full-sized hold-out from the undrawn pool and no plot is held out twice. See
  :func:`random_systematic_group_split_by_count`.
* ``"sequential"`` — the sorted plot groups are dealt into **contiguous blocks of ``round(test_size ·
  n_groups)`` plots** and each cycle holds out one block, sweeping the field in order rather than
  sampling it. Like random-systematic it is a deterministic without-replacement **partition** driven by
  **both** the ratio (block size) and the count (``n_splits · block ≤ n_groups``, the GUI's splits·ratio
  ≤ 1 cap): with ``splits · ratio == 1`` it is an exact k-fold (every plot held out once), and with the
  product below 1 the trailing plots are simply never held out. The user sets the ratio and the count
  freely; no plot is held out twice. See :func:`sequential_group_split_by_count`.
* ``"systematic"`` — like sequential a deterministic, leakage-free without-replacement **partition**
  driven by the ratio (block size) and count, but instead of contiguous blocks it holds out a strided
  sample: with plots in numeric order, fold ``c`` takes the first ``block`` of positions
  ``c, c + n_splits, c + 2·n_splits, …``. Each fold's hold-out is spread evenly across the whole field
  rather than covering one contiguous stretch, which is the right choice when there is a spatial
  gradient along the plot order (every fold then samples the whole range). Same free ratio/count controls
  and the same ``splits · ratio ≤ 1`` cap as sequential; no plot is held out twice across
  ``cycle in range(n_splits)``. See :func:`systematic_group_split_by_count`.

  (The older ratio-driven sequential split — :func:`n_sequential_splits` deriving the count from a
  test-size ratio, and :func:`sequential_group_split` sliding a ratio-sized window — is retained for
  back-compat but is no longer the path the trainer/CV use.)
"""

from __future__ import annotations

import math

import numpy as np
from sklearn.model_selection import GroupShuffleSplit

SPLIT_MODES = ("random", "random_systematic", "sequential", "systematic")


def n_sequential_splits(n_groups: int, test_size: float) -> int:
    """How many contiguous, non-overlapping blocks tile ``n_groups`` plots at this test ratio.

    *Legacy ratio-driven helper.* Sequential mode now takes the split **count** directly
    (:func:`sequential_group_split_by_count`); this maps a test-size ratio to a block count for the
    rare callers/tests that still derive it. Block of ``round(test_size * n_groups)`` plots per split,
    sliding one block at a time, so ``ceil(1 / test_size)`` blocks cover the whole field.
    """
    if n_groups < 2:
        return 1
    block = max(1, min(n_groups - 1, int(round(test_size * n_groups))))
    return max(1, math.ceil(n_groups / block))


def grouped_split(
    groups, test_size: float, seed: int, *, mode: str = "random", cycle: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Split row indices into (train, test) so no group spans both sides.

    ``mode`` selects random (``GroupShuffleSplit``) or sequential (a sliding contiguous block of
    groups) splitting; ``cycle`` is the 0-based split index, used to advance the sequential block
    (and is already folded into ``seed`` by the callers for the random mode).

    Raises
    ------
    ValueError
        If there are too few distinct groups to form the requested split.
    """
    groups = np.asarray(groups)
    n_groups = len(np.unique(groups))
    if n_groups < 2:
        raise ValueError(
            f"Need at least 2 distinct plot groups to split; found {n_groups}. "
            "Check that file names contain plot(N)."
        )
    if mode == "sequential":
        return sequential_group_split(groups, test_size, cycle)
    if mode == "systematic":
        # The ratio-based entry point maps a test-size ratio to a fold count the same way the legacy
        # sequential path does (``ceil(1 / test_size)`` folds), then strides. Callers that drive the
        # partition by count use :func:`systematic_group_split_by_count` directly.
        n_splits = n_sequential_splits(n_groups, test_size)
        return systematic_group_split_by_count(groups, n_splits, test_size, cycle)
    if mode == "random_systematic":
        # Ratio-driven entry point: derive the block count the same way (so the shared shuffle tiles the
        # field without overlap), then deal block ``cycle``. Callers that drive it by count use
        # :func:`random_systematic_group_split_by_count` directly.
        n_splits = max_splits_for_test_size(n_groups, test_size)
        return random_systematic_group_split_by_count(groups, n_splits, test_size, seed, cycle)
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(splitter.split(np.zeros(len(groups)), groups=groups))
    return train_idx, test_idx


def _group_sort_key(group) -> tuple[int, object]:
    """Sort key ordering ``plot:N`` groups by the integer ``N`` (the physical plot number).

    Returns ``(0, plot_number)`` for a parseable ``…:<int>`` group and ``(1, str)`` otherwise, so
    numbered groups sort numerically and any oddly-named group falls back to the end in name order.
    """
    text = str(group)
    tail = text.rsplit(":", 1)[-1]
    try:
        return (0, int(tail))
    except ValueError:
        return (1, text)


def sequential_group_split(
    groups, test_size: float, cycle: int
) -> tuple[np.ndarray, np.ndarray]:
    """Hold out a contiguous block of plot groups (sorted order), sliding by ``cycle``.

    The block holds ``round(test_size * n_groups)`` groups (at least one, never all), starting at
    ``(cycle * block) % n_groups`` and wrapping around, so successive cycles cover different,
    contiguous parts of the field. No group spans both sides. When the caller runs exactly
    :func:`n_sequential_splits` cycles the blocks tile the field with no overlap (a K-fold
    partition); the wrap-around only repeats blocks if more cycles than that are requested.
    """
    groups = np.asarray(groups)
    # Order the blocks by the numeric plot number (the int after "plot:") so a contiguous block is
    # a contiguous plot-number range even when plot numbers differ in digit count (e.g. 99 vs 100);
    # groups without a numeric suffix fall back to the end in name order. (Plain ``sorted`` on the
    # key tuples — not np.argsort, which would build a 2-D array and sort the wrong axis.)
    unique = sorted(np.unique(groups).tolist(), key=_group_sort_key)
    n_groups = len(unique)
    block = max(1, min(n_groups - 1, int(round(test_size * n_groups))))
    start = (cycle * block) % n_groups
    # Contiguous wrap-around window of `block` groups (indices into the numerically-sorted list).
    order = [(start + i) % n_groups for i in range(block)]
    test_groups = {unique[i] for i in order}
    is_test = np.array([g in test_groups for g in groups])
    test_idx = np.flatnonzero(is_test)
    train_idx = np.flatnonzero(~is_test)
    return train_idx, test_idx


def _block_size(n_groups: int, test_size: float, n_splits: int = 1) -> int:
    """Plots per hold-out block for a ratio-driven partition: ``round(test_size · n_groups)``, ≥ 1.

    Shared by every without-replacement partition mode (sequential / systematic / random-systematic):
    the user's "Split ratio" picks the block size in plots, and the "Splits" count picks how many such
    disjoint blocks to deal. Both inputs are honoured: the block is the rounded ratio, **shrunk just
    enough that ``n_splits`` disjoint blocks still tile the field** (``block ≤ ⌊n_groups / n_splits⌋``).
    So when the rounded ratio leaves room for the requested count nothing changes, but when rounding it
    *up* would overrun the field (e.g. ``round(0.25 · 150) = 38`` ⇒ ``4 · 38 = 152 > 150``) the block
    drops by the minimum needed (38 → ⌊150/4⌋ = 37) rather than the count being silently reduced. The
    GUI's ``splits · ratio ≤ 1`` cap keeps the shrink to ≤ 1 plot. Never zero (a degenerate ratio still
    yields a 1-plot block) and never more than ``n_groups``.
    """
    block = int(round(test_size * n_groups))
    fits = n_groups // max(1, n_splits)  # largest block that lets n_splits disjoint blocks tile
    return max(1, min(n_groups, block, fits))


def sequential_group_split_by_count(
    groups, n_splits: int, test_size: float, cycle: int
) -> tuple[np.ndarray, np.ndarray]:
    """Hold out block ``cycle`` of a contiguous without-replacement partition of the plot groups.

    The numerically-sorted unique groups are dealt into contiguous blocks of ``round(test_size ·
    n_groups)`` plots each (at least one); block ``cycle`` is the hold-out. ``n_splits`` blocks are dealt
    off the front, so the user controls **both** how big each hold-out is (``test_size``) and how many
    blocks (``n_splits``). Both are honoured: the requested count of disjoint blocks always tiles the
    field because the block is shrunk to ``⌊n_groups / n_splits⌋`` when the rounded ratio would overrun
    (see :func:`_block_size`). When ``n_splits · block == n_groups`` this is an exact partition (every
    plot held out once); when it is less, the trailing plots are simply never held out (a deterministic
    partial sweep). Over ``cycle in range(n_splits)`` no plot is held out more than once, and no group
    spans both sides.
    """
    groups = np.asarray(groups)
    unique = sorted(np.unique(groups).tolist(), key=_group_sort_key)
    n_groups = len(unique)
    n_splits = max(1, min(n_splits, n_groups))
    block = _block_size(n_groups, test_size, n_splits)
    c = cycle % n_splits
    start = c * block
    stop = min(start + block, n_groups)
    test_groups = {unique[i] for i in range(start, stop)}
    is_test = np.array([g in test_groups for g in groups])
    test_idx = np.flatnonzero(is_test)
    train_idx = np.flatnonzero(~is_test)
    return train_idx, test_idx


def systematic_group_split_by_count(
    groups, n_splits: int, test_size: float, cycle: int
) -> tuple[np.ndarray, np.ndarray]:
    """Hold out fold ``cycle`` of a **systematic** (strided) without-replacement partition of plots.

    Each fold holds out ``round(test_size · n_groups)`` plots (at least one) spread evenly across the
    field by striding with step ``n_splits``: fold ``cycle`` takes the first ``block`` of the positions
    ``cycle, cycle + n_splits, cycle + 2·n_splits, …`` (in numeric plot order). With ``n_splits`` folds
    and ``block`` plots each, distinct folds take disjoint positions because the block is shrunk to
    ``⌊n_groups / n_splits⌋`` when the rounded ratio would overrun (see :func:`_block_size`), so
    ``n_splits · block ≤ n_groups`` always holds (the GUI's splits·ratio ≤ 1 cap). Each fold's hold-out
    samples the whole field rather than one contiguous stretch — the right choice under a spatial
    gradient along the plot order — the difference from :func:`sequential_group_split_by_count`. No
    group spans both sides; over ``cycle in range(n_splits)`` no plot is held out more than once.
    """
    groups = np.asarray(groups)
    unique = sorted(np.unique(groups).tolist(), key=_group_sort_key)
    n_groups = len(unique)
    n_splits = max(1, min(n_splits, n_groups))
    block = _block_size(n_groups, test_size, n_splits)
    c = cycle % n_splits
    # The first ``block`` strided positions for this fold: c, c+n_splits, c+2*n_splits, … (block of them).
    positions = [c + k * n_splits for k in range(block) if c + k * n_splits < n_groups]
    test_groups = {unique[i] for i in positions}
    is_test = np.array([g in test_groups for g in groups])
    test_idx = np.flatnonzero(is_test)
    train_idx = np.flatnonzero(~is_test)
    return train_idx, test_idx


def max_splits_for_test_size(n_groups: int, test_size: float) -> int:
    """How many disjoint hold-out blocks of ``test_size`` of the field fit — the ``splits · ratio ≤ 1`` cap.

    The cap that makes a **random-systematic** (or any without-replacement) partition possible. It is the
    *continuous* ``splits · ratio ≤ 1`` ⇒ ``splits ≤ ⌊1 / test_size⌋`` — deliberately NOT
    ``⌊n_groups / round(test_size · n_groups)⌋``: the per-block plot count is rounded *to fit the
    requested count* (:func:`_block_size` shrinks the block to ``⌊n_groups / n_splits⌋`` when the rounded
    ratio would overrun), so the count itself is bounded only by the ratio, not by where the rounding of a
    *single* block happens to land. This keeps the GUI's "Splits" / "Test split" controls honouring both
    inputs (e.g. ratio 0.25 always allows 4 splits, even on 150 plots where ``round(0.25·150)=38`` and
    ``4·38>150``). Returns at least 1 (a single block always fits) and never more than ``n_groups``.
    """
    if n_groups < 1 or test_size <= 0:
        return max(1, n_groups)
    return max(1, min(n_groups, int(1.0 / test_size)))


def random_systematic_group_split_by_count(
    groups, n_splits: int, test_size: float, seed: int, cycle: int
) -> tuple[np.ndarray, np.ndarray]:
    """Hold out block ``cycle`` of a *randomised* without-replacement partition of the plot groups.

    The unique groups are shuffled **once** with an RNG seeded by ``seed`` alone (deliberately *not*
    ``seed + cycle``, so every cycle of the same run shares one shuffle and therefore dovetails into a
    single partition), then dealt into contiguous blocks of ``round(test_size · n_groups)`` plots. Cycle
    ``c`` holds out block ``c`` of that shuffled order — so cycle 0 is a random hold-out, cycle 1 is a
    random hold-out drawn from the plots cycle 0 did **not** take, and so on: random sampling *without
    replacement* across the splits, the randomised twin of :func:`sequential_group_split_by_count`.

    Both inputs are honoured: the block is shrunk to ``⌊n_groups / n_splits⌋`` when the rounded ratio
    would overrun (see :func:`_block_size`), so ``n_splits · block ≤ n_groups`` always holds (the GUI
    enforces ``splits · ratio ≤ 1``) and no degenerate request can make blocks overlap. No group spans
    both sides; over ``cycle in range(n_splits)`` no plot is held out more than once.
    """
    groups = np.asarray(groups)
    # Start from the same canonical order as the deterministic partitions, then shuffle it once so the
    # result is reproducible and order-independent of the row layout (same guarantee as systematic).
    unique = sorted(np.unique(groups).tolist(), key=_group_sort_key)
    n_groups = len(unique)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_groups)
    shuffled = [unique[i] for i in perm]
    n_splits = max(1, min(n_splits, n_groups))
    block = _block_size(n_groups, test_size, n_splits)
    c = cycle % n_splits
    start = c * block
    stop = min(start + block, n_groups)
    test_groups = {shuffled[i] for i in range(start, stop)}
    is_test = np.array([g in test_groups for g in groups])
    test_idx = np.flatnonzero(is_test)
    train_idx = np.flatnonzero(~is_test)
    return train_idx, test_idx
