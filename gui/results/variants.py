"""The augmented-data choice the Results sub-tabs read from the host's two toggles.

The Results page has two display toggles — *include augmented data in training metrics* and *in
validation metrics* — that pick which precomputed metric cell every sub-tab shows (held-out /
training / overall) without any recomputation. This module turns the two booleans into the cube keys
:func:`ml.trainer.variant_metrics` expects, in one immutable object the host hands to each sub-tab.

The "overall" key encodes *both* choices (``"<train>_<val>"``) because an overall figure mixes
training and held-out plots, so an augmented row's inclusion depends on which side it sits on. The
``available`` flag is False for pre-cube (legacy) bundles; sub-tabs then render n/a and the host
greys the toggles out.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class VariantSelection:
    """Which metric-cube cell each scope resolves to, derived from the host's two aug toggles."""

    held_out: str   # "orig_only" | "with_aug"
    train: str      # "orig_only" | "with_aug"
    overall: str    # "orig_orig" | "aug_orig" | "orig_aug" | "aug_aug"  (train_val)
    available: bool  # the loaded model carries a precomputed cube (format-3 bundle)

    @classmethod
    def from_toggles(cls, train_aug: bool, val_aug: bool, available: bool) -> "VariantSelection":
        """Build a selection from the two toggle states (train on/off, validation on/off)."""
        return cls(
            held_out="with_aug" if val_aug else "orig_only",
            train="with_aug" if train_aug else "orig_only",
            overall=f"{'aug' if train_aug else 'orig'}_{'aug' if val_aug else 'orig'}",
            available=available,
        )

    @classmethod
    def legacy(cls) -> "VariantSelection":
        """The fallback when no cube is available: original-only everywhere, overall unavailable."""
        return cls(held_out="orig_only", train="orig_only", overall="", available=False)

    def key(self, scope: str) -> str:
        """The cube key for ``scope`` (``"held_out"``/``"train"``/``"overall"``)."""
        return getattr(self, scope)
