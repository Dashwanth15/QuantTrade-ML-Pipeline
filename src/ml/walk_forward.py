"""
QuantTrade ML Pipeline — Walk-Forward Validator
Implements strict time-series aware cross-validation with embargo periods.
No data leakage — train and test windows are always chronologically separated.

Resolution-Agnostic Design:
    Uses pd.Timedelta-based timestamp arithmetic rather than integer row offsets.
    Correctly handles millisecond, second, minute, hourly, and daily bar data
    without any hardcoded frequency multipliers.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import settings


@dataclass
class WalkForwardFold:
    """Metadata for a single walk-forward fold."""
    fold_index: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    embargo_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    train_mask: np.ndarray
    test_mask: np.ndarray

    @property
    def n_train(self) -> int:
        return int(self.train_mask.sum())

    @property
    def n_test(self) -> int:
        return int(self.test_mask.sum())

    def to_dict(self) -> dict:
        return {
            "fold_index": self.fold_index,
            "train_start": self.train_start,
            "train_end": self.train_end,
            "test_start": self.test_start,
            "test_end": self.test_end,
            "n_train": self.n_train,
            "n_test": self.n_test,
        }


class WalkForwardValidator:
    """
    Production walk-forward validator with embargo period.

    Resolution-agnostic: uses pd.Timedelta timestamp arithmetic, so it
    correctly handles any bar frequency (milliseconds → daily).

    Each fold:
    ├── Training window: train_days calendar days
    ├── Embargo period: embargo_days calendar days (prevents lag-feature leakage)
    └── Test window: test_days calendar days

    The anchor slides forward by step_days after each fold.

    Example (train=90d, test=30d, step=30d, embargo=5d, hourly data):
        Fold 0: Train[2015-01-01 .. 2015-04-01), Embargo 5d, Test[2015-04-06 .. 2015-05-06)
        Fold 1: Train[2015-01-31 .. 2015-05-01), Embargo 5d, Test[2015-05-06 .. 2015-06-05)
        ...

    Backward-compatible with original API:
        WalkForwardValidator(train_days=90, test_days=30, step_days=30, embargo_days=5)
    """

    def __init__(
        self,
        train_days: int | None = None,
        test_days: int | None = None,
        step_days: int | None = None,
        embargo_days: int | None = None,
    ) -> None:
        # Resolve from settings if not supplied
        train_d  = train_days   if train_days   is not None else settings.wf_train_days
        test_d   = test_days    if test_days    is not None else settings.wf_test_days
        step_d   = step_days    if step_days    is not None else settings.wf_step_days
        emb_d    = embargo_days if embargo_days is not None else settings.wf_embargo_days

        # Store as pd.Timedelta — resolution-agnostic, no hardcoded × 24
        self.train_delta   = pd.Timedelta(days=train_d)
        self.test_delta    = pd.Timedelta(days=test_d)
        self.step_delta    = pd.Timedelta(days=step_d)
        self.embargo_delta = pd.Timedelta(days=emb_d)

        # Keep raw day counts for logging
        self._train_days   = train_d
        self._test_days    = test_d
        self._step_days    = step_d
        self._embargo_days = emb_d

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def split(self, df: pd.DataFrame) -> list[WalkForwardFold]:
        """
        Generate walk-forward folds for any bar-frequency DataFrame.

        Args:
            df: DataFrame with a sorted, timezone-aware or naive DatetimeIndex.

        Returns:
            List of WalkForwardFold objects.

        Raises:
            AssertionError: If the index is not monotonically increasing.
            TypeError: If the index is not a DatetimeIndex.
        """
        if not isinstance(df.index, pd.DatetimeIndex):
            raise TypeError(
                f"WalkForwardValidator.split() requires a DatetimeIndex, "
                f"got {type(df.index).__name__}."
            )
        assert df.index.is_monotonic_increasing, "Index must be sorted chronologically."

        index = df.index
        n     = len(index)
        folds: list[WalkForwardFold] = []

        logger.info(
            "Generating walk-forward folds | n_bars={} | freq={} | "
            "train={}d | test={}d | embargo={}d | step={}d",
            n,
            _infer_freq(index),
            self._train_days,
            self._test_days,
            self._embargo_days,
            self._step_days,
        )

        anchor = index[0]
        fold_idx = 0

        while True:
            # ── Compute boundary timestamps ──────────────────────────────
            train_start_ts  = anchor
            train_end_ts    = anchor + self.train_delta
            embargo_end_ts  = train_end_ts + self.embargo_delta
            test_start_ts   = embargo_end_ts
            test_end_ts     = embargo_end_ts + self.test_delta

            # Stop when the full test window exceeds the data range
            if test_end_ts > index[-1]:
                break

            # ── Build boolean masks from timestamps ──────────────────────
            train_mask = (index >= train_start_ts) & (index < train_end_ts)
            test_mask  = (index >= test_start_ts)  & (index < test_end_ts)

            if train_mask.sum() == 0 or test_mask.sum() == 0:
                anchor += self.step_delta
                fold_idx += 1
                continue

            # ── Resolve actual boundary timestamps from index ────────────
            train_idx = index[train_mask]
            test_idx  = index[test_mask]

            # embargo_end is the last timestamp strictly in the embargo gap
            embargo_rows = (index >= train_end_ts) & (index < embargo_end_ts)
            if embargo_rows.any():
                embargo_end_actual = index[embargo_rows][-1]
            else:
                embargo_end_actual = train_end_ts  # no bars in embargo (e.g. weekend gap)

            fold = WalkForwardFold(
                fold_index=fold_idx,
                train_start=train_idx[0],
                train_end=train_idx[-1],
                embargo_end=embargo_end_actual,
                test_start=test_idx[0],
                test_end=test_idx[-1],
                train_mask=train_mask,
                test_mask=test_mask,
            )
            folds.append(fold)

            logger.debug(
                "Fold {} | Train: {:%Y-%m-%d} → {:%Y-%m-%d} (n={}) | "
                "Test: {:%Y-%m-%d} → {:%Y-%m-%d} (n={})",
                fold_idx,
                fold.train_start, fold.train_end, fold.n_train,
                fold.test_start,  fold.test_end,  fold.n_test,
            )

            anchor   += self.step_delta
            fold_idx += 1

        logger.info("Generated {} walk-forward folds", len(folds))
        return folds

    def verify_no_leakage(self, folds: list[WalkForwardFold]) -> bool:
        """Verify that no fold has training data overlapping with test data."""
        for fold in folds:
            if fold.train_end >= fold.test_start:
                logger.error(
                    "DATA LEAKAGE in fold {}! train_end={} >= test_start={}",
                    fold.fold_index, fold.train_end, fold.test_start,
                )
                return False
            if self.embargo_delta.total_seconds() > 0 and fold.train_end >= fold.embargo_end:
                logger.error("Embargo violation in fold {}!", fold.fold_index)
                return False
        logger.debug("Leakage check passed for all {} folds", len(folds))
        return True

    @property
    def embargo_delta(self) -> pd.Timedelta:
        return self._embargo_delta

    @embargo_delta.setter
    def embargo_delta(self, value: pd.Timedelta) -> None:
        self._embargo_delta = value


# ── Helper ────────────────────────────────────────────────────────────────────

def _infer_freq(index: pd.DatetimeIndex) -> str:
    """Best-effort detection of bar frequency for logging."""
    if len(index) < 2:
        return "unknown"
    delta = index[1] - index[0]
    seconds = delta.total_seconds()
    if seconds < 1:
        return f"{int(delta.microseconds / 1000)}ms"
    if seconds < 60:
        return f"{int(seconds)}s"
    if seconds < 3600:
        return f"{int(seconds / 60)}min"
    if seconds < 86400:
        return f"{int(seconds / 3600)}h"
    return f"{int(seconds / 86400)}d"
