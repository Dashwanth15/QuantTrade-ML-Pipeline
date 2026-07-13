"""
QuantTrade ML Pipeline — Walk-Forward Validation Tests
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.ml.walk_forward import WalkForwardValidator


@pytest.fixture
def sample_wf_df():
    """300 days of hourly data = 7200 bars."""
    dates = pd.date_range("2010-01-01", periods=7200, freq="h", tz="UTC")
    df = pd.DataFrame({"price": np.random.randn(7200)}, index=dates)
    return df


class TestWalkForwardValidator:
    def test_generates_folds(self, sample_wf_df):
        wf = WalkForwardValidator(train_days=60, test_days=20, step_days=20, embargo_days=5)
        folds = wf.split(sample_wf_df)
        assert len(folds) > 0, "Should generate at least one fold"

    def test_no_data_leakage(self, sample_wf_df):
        wf = WalkForwardValidator(train_days=60, test_days=20, step_days=20, embargo_days=5)
        folds = wf.split(sample_wf_df)
        assert wf.verify_no_leakage(folds), "Data leakage detected!"

    def test_train_before_test(self, sample_wf_df):
        wf = WalkForwardValidator(train_days=60, test_days=20, step_days=20, embargo_days=5)
        folds = wf.split(sample_wf_df)
        for fold in folds:
            assert fold.train_end < fold.test_start, (
                f"Fold {fold.fold_index}: train_end {fold.train_end} >= test_start {fold.test_start}"
            )

    def test_embargo_respected(self, sample_wf_df):
        embargo_days = 5
        wf = WalkForwardValidator(train_days=60, test_days=20, step_days=20, embargo_days=embargo_days)
        folds = wf.split(sample_wf_df)
        for fold in folds:
            gap_hours = (fold.test_start - fold.train_end).total_seconds() / 3600
            assert gap_hours >= embargo_days * 24 - 1, (
                f"Embargo not respected in fold {fold.fold_index}: gap={gap_hours:.0f}h"
            )

    def test_train_test_non_overlapping(self, sample_wf_df):
        wf = WalkForwardValidator(train_days=60, test_days=20, step_days=20, embargo_days=5)
        folds = wf.split(sample_wf_df)
        for fold in folds:
            overlap = fold.train_mask & fold.test_mask
            assert not overlap.any(), f"Train/test masks overlap in fold {fold.fold_index}!"

    def test_fold_sizes(self, sample_wf_df):
        train_days, test_days = 60, 20
        wf = WalkForwardValidator(train_days=train_days, test_days=test_days, step_days=20, embargo_days=5)
        folds = wf.split(sample_wf_df)
        for fold in folds:
            expected_train = train_days * 24
            expected_test = test_days * 24
            # Allow small tolerance for index alignment
            assert abs(fold.n_train - expected_train) <= 1, (
                f"Fold {fold.fold_index} train size {fold.n_train} != expected {expected_train}"
            )
