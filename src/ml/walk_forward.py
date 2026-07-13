"""
QuantTrade ML Pipeline — Walk-Forward Validator
Implements strict time-series aware cross-validation with embargo periods.
No data leakage — train and test windows are always chronologically separated.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
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

    Each fold:
    ├── Training window: train_days
    ├── Embargo period: embargo_days (buffer to prevent leakage from lag features)
    └── Test window: test_days

    The window slides forward by step_days after each fold.

    Example (train=90, test=30, step=30, embargo=5):
    Fold 0: Train[0..89], Embargo[90..94], Test[95..124]
    Fold 1: Train[30..119], Embargo[120..124], Test[125..154]
    ...
    """

    def __init__(
        self,
        train_days: int | None = None,
        test_days: int | None = None,
        step_days: int | None = None,
        embargo_days: int | None = None,
    ) -> None:
        # Use explicit None checks so callers can pass 0 without triggering the default
        self.train_days = (train_days if train_days is not None else settings.wf_train_days) * 24
        self.test_days = (test_days if test_days is not None else settings.wf_test_days) * 24
        self.step_days = (step_days if step_days is not None else settings.wf_step_days) * 24
        self.embargo_days = (embargo_days if embargo_days is not None else settings.wf_embargo_days) * 24

    def split(self, df: pd.DataFrame) -> list[WalkForwardFold]:
        """
        Generate walk-forward folds for the given DataFrame.

        Args:
            df: DataFrame with UTC DatetimeIndex (must be sorted)

        Returns:
            List of WalkForwardFold objects
        """
        assert df.index.is_monotonic_increasing, "Index must be sorted chronologically"
        n = len(df)
        folds = []
        fold_idx = 0
        window_start = 0

        logger.info(
            "Generating walk-forward folds | n={} | train={}h | test={}h | embargo={}h | step={}h",
            n, self.train_days, self.test_days, self.embargo_days, self.step_days,
        )

        while True:
            train_end = window_start + self.train_days
            embargo_end = train_end + self.embargo_days
            test_end = embargo_end + self.test_days

            if test_end > n:
                break  # Not enough data for this fold

            train_mask = np.zeros(n, dtype=bool)
            test_mask = np.zeros(n, dtype=bool)
            train_mask[window_start:train_end] = True
            test_mask[embargo_end:test_end] = True

            fold = WalkForwardFold(
                fold_index=fold_idx,
                train_start=df.index[window_start],
                train_end=df.index[train_end - 1],
                embargo_end=df.index[min(embargo_end, n - 1)],
                test_start=df.index[embargo_end],
                test_end=df.index[test_end - 1],
                train_mask=train_mask,
                test_mask=test_mask,
            )
            folds.append(fold)

            logger.debug(
                "Fold {} | Train: {:%Y-%m-%d} → {:%Y-%m-%d} (n={}) | "
                "Test: {:%Y-%m-%d} → {:%Y-%m-%d} (n={})",
                fold_idx,
                fold.train_start, fold.train_end, fold.n_train,
                fold.test_start, fold.test_end, fold.n_test,
            )

            window_start += self.step_days
            fold_idx += 1

        logger.info("Generated {} walk-forward folds", len(folds))
        return folds

    def verify_no_leakage(self, folds: list[WalkForwardFold]) -> bool:
        """Verify that no fold has training data after test data."""
        for fold in folds:
            if fold.train_end >= fold.test_start:
                logger.error(
                    "DATA LEAKAGE detected in fold {}! "
                    "train_end={} >= test_start={}",
                    fold.fold_index, fold.train_end, fold.test_start,
                )
                return False
            if fold.train_end >= fold.embargo_end:
                logger.error(
                    "Embargo violation in fold {}!", fold.fold_index
                )
                return False
        logger.debug("Leakage check passed for all {} folds", len(folds))
        return True
