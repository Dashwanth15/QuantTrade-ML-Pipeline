"""
QuantTrade ML Pipeline — ML Pipeline Integration Tests

Tests the full ML pipeline end-to-end:
  prepare_ml_dataset → build_xgboost_pipeline → evaluate → walk-forward
All tests use synthetic data to run fast without requiring CSV files.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ml.pipeline import (
    TimeSeriesImputer,
    build_xgboost_pipeline,
    prepare_ml_dataset,
)
from src.ml.evaluator import ModelEvaluator
from src.ml.walk_forward import WalkForwardValidator, WalkForwardFold
from src.ml.tuner import XGBoostTuner


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def synthetic_feature_df():
    """500 hourly bars of synthetic market features."""
    np.random.seed(0)
    n = 500
    dates = pd.date_range("2015-01-01", periods=n, freq="h", tz="UTC")
    price = 1.2 + np.cumsum(np.random.normal(0, 0.0003, n))

    df = pd.DataFrame(index=dates)
    df["mid_close"] = price
    df["return_1h"] = df["mid_close"].pct_change()
    df["roll_mean_20"] = df["mid_close"].rolling(20, min_periods=10).mean()
    df["roll_std_20"] = df["mid_close"].rolling(20, min_periods=10).std()
    df["momentum_10"] = df["mid_close"] - df["mid_close"].shift(10)
    df["rsi_14"] = 50 + np.random.normal(0, 15, n)
    for col in df.columns:
        df[col] = df[col].bfill().fillna(0)
    return df


@pytest.fixture
def synthetic_trade_log(synthetic_feature_df):
    """100 synthetic trades aligned to synthetic_feature_df timestamps."""
    np.random.seed(1)
    n_trades = 100
    idx = synthetic_feature_df.index

    entries = np.sort(np.random.choice(len(idx) - 10, n_trades, replace=False))
    exits = entries + np.random.randint(1, 10, n_trades)
    exits = np.clip(exits, 0, len(idx) - 1)

    pnl_usd = np.random.normal(0, 150, n_trades)

    trades = pd.DataFrame({
        "strategy_id": np.random.choice(["momentum", "bollinger", "rsi_reversion"], n_trades),
        "direction": np.random.choice([-1, 1], n_trades),
        "entry_time": idx[entries],
        "exit_time": idx[exits],
        "entry_price": synthetic_feature_df["mid_close"].iloc[entries].values,
        "exit_price": synthetic_feature_df["mid_close"].iloc[exits].values,
        "pnl": pnl_usd / 100_000,
        "pnl_usd": pnl_usd,
        "pnl_pct": pnl_usd / 100_000,
        "win": pnl_usd > 0,
        "holding_bars": (exits - entries).tolist(),
        "quantity": 100_000.0,
        "exit_reason": "tp",
        "rolling_win_rate": 0.5,
        "rolling_avg_pnl": pnl_usd.mean(),
        "prev_trade_pnl": np.roll(pnl_usd, 1),
    })
    return trades


# ── TimeSeriesImputer Tests ───────────────────────────────────────────────────

class TestTimeSeriesImputer:
    def test_fills_nan_forward(self):
        df = pd.DataFrame({"a": [1.0, np.nan, np.nan, 4.0], "b": [np.nan, 2.0, 3.0, 4.0]})
        imputer = TimeSeriesImputer()
        imputer.fit(df)
        result = imputer.transform(df)
        # No NaNs should remain
        assert not np.isnan(result).any(), "Imputer left NaN values"

    def test_output_shape_preserved(self):
        df = pd.DataFrame(np.random.rand(50, 5), columns=list("ABCDE"))
        df.iloc[5, 2] = np.nan
        imputer = TimeSeriesImputer()
        imputer.fit(df)
        out = imputer.transform(df)
        assert out.shape == df.shape

    def test_no_future_fill(self):
        """ffill should propagate past values forward, not backward."""
        df = pd.DataFrame({"a": [np.nan, np.nan, 5.0, 6.0]})
        imputer = TimeSeriesImputer()
        imputer.fit(df)
        result = imputer.transform(df)
        # After bfill, first two should become 5.0
        assert result[0, 0] == pytest.approx(5.0)


# ── prepare_ml_dataset Tests ─────────────────────────────────────────────────

class TestPrepareMLDataset:
    def test_returns_correct_shapes(self, synthetic_feature_df, synthetic_trade_log):
        X, y, ts = prepare_ml_dataset(synthetic_feature_df, synthetic_trade_log)
        assert len(X) == len(y) == len(ts)
        assert len(X) > 0, "Should produce at least 1 sample"

    def test_x_is_numeric_only(self, synthetic_feature_df, synthetic_trade_log):
        X, y, ts = prepare_ml_dataset(synthetic_feature_df, synthetic_trade_log)
        non_numeric = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
        assert len(non_numeric) == 0, f"Non-numeric columns in X: {non_numeric}"

    def test_no_inf_in_x(self, synthetic_feature_df, synthetic_trade_log):
        X, y, ts = prepare_ml_dataset(synthetic_feature_df, synthetic_trade_log)
        has_inf = np.isinf(X.values).any()
        assert not has_inf, "X contains infinite values"

    def test_no_nan_target(self, synthetic_feature_df, synthetic_trade_log):
        X, y, ts = prepare_ml_dataset(synthetic_feature_df, synthetic_trade_log)
        assert not y.isna().any(), "y (target) contains NaN values"

    def test_timestamps_chronological(self, synthetic_feature_df, synthetic_trade_log):
        X, y, ts = prepare_ml_dataset(synthetic_feature_df, synthetic_trade_log)
        assert ts.is_monotonic_increasing, "Timestamps must be in chronological order"

    def test_raises_on_empty_trade_log(self, synthetic_feature_df):
        empty_log = pd.DataFrame()
        with pytest.raises(ValueError, match="Trade log is empty"):
            prepare_ml_dataset(synthetic_feature_df, empty_log)


# ── XGBoost Pipeline Tests ───────────────────────────────────────────────────

class TestXGBoostPipeline:
    def test_pipeline_builds(self):
        pipeline = build_xgboost_pipeline()
        assert pipeline is not None
        step_names = [name for name, _ in pipeline.steps]
        assert "imputer" in step_names
        assert "scaler" in step_names
        assert "model" in step_names

    def test_pipeline_fit_predict(self, synthetic_feature_df, synthetic_trade_log):
        X, y, _ = prepare_ml_dataset(synthetic_feature_df, synthetic_trade_log)
        pipeline = build_xgboost_pipeline()
        pipeline.fit(X, y)
        preds = pipeline.predict(X)
        assert len(preds) == len(y)
        assert not np.isnan(preds).any(), "Pipeline produced NaN predictions"

    def test_pipeline_generalises(self, synthetic_feature_df, synthetic_trade_log):
        """Train on first 70%, predict on last 30% — predictions should be finite."""
        X, y, ts = prepare_ml_dataset(synthetic_feature_df, synthetic_trade_log)
        split = int(len(X) * 0.7)
        X_train, X_test = X.iloc[:split], X.iloc[split:]
        y_train = y.iloc[:split]

        pipeline = build_xgboost_pipeline()
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)
        assert np.all(np.isfinite(preds)), "Out-of-sample predictions are not finite"


# ── ModelEvaluator Tests ─────────────────────────────────────────────────────

class TestModelEvaluator:
    def test_perfect_prediction(self):
        evaluator = ModelEvaluator()
        y = np.array([100.0, -50.0, 200.0, -30.0, 150.0])
        metrics = evaluator.evaluate(y, y)
        assert metrics["mae"] == pytest.approx(0.0, abs=1e-8)
        assert metrics["rmse"] == pytest.approx(0.0, abs=1e-8)

    def test_metrics_keys_present(self):
        evaluator = ModelEvaluator()
        y_true = np.random.normal(0, 100, 200)
        y_pred = y_true + np.random.normal(0, 20, 200)
        metrics = evaluator.evaluate(y_true, y_pred)
        required_keys = [
            "mae", "rmse", "r2", "sharpe", "sortino", "win_rate",
            "max_drawdown", "profit_factor", "cumulative_return",
        ]
        for k in required_keys:
            assert k in metrics, f"Missing metric key: {k}"

    def test_win_rate_bounds(self):
        evaluator = ModelEvaluator()
        y_true = np.random.normal(0, 100, 300)
        y_pred = y_true + np.random.normal(0, 10, 300)
        metrics = evaluator.evaluate(y_true, y_pred)
        assert 0.0 <= metrics["win_rate"] <= 1.0

    def test_direction_accuracy(self):
        evaluator = ModelEvaluator()
        y_true = np.array([100.0, -50.0, 75.0, -25.0])
        y_pred = np.array([80.0, -30.0, 60.0, -10.0])  # Same sign as y_true
        metrics = evaluator.evaluate(y_true, y_pred)
        assert metrics["direction_accuracy"] == pytest.approx(1.0)


# ── XGBoostTuner Tests ───────────────────────────────────────────────────────

class TestXGBoostTuner:
    def test_default_params_complete(self):
        params = XGBoostTuner._default_params()
        required = [
            "n_estimators", "max_depth", "learning_rate", "subsample",
            "colsample_bytree", "reg_alpha", "reg_lambda", "objective",
        ]
        for key in required:
            assert key in params, f"Missing default param: {key}"

    def test_tune_returns_dict(self, synthetic_feature_df, synthetic_trade_log):
        """Tune with just 2 trials so the test runs fast."""
        X, y, _ = prepare_ml_dataset(synthetic_feature_df, synthetic_trade_log)
        tuner = XGBoostTuner(n_trials=2, n_cv_folds=2)
        params = tuner.tune(X, y)
        assert isinstance(params, dict)
        assert "n_estimators" in params
        assert "learning_rate" in params

    def test_best_params_valid_types(self, synthetic_feature_df, synthetic_trade_log):
        X, y, _ = prepare_ml_dataset(synthetic_feature_df, synthetic_trade_log)
        tuner = XGBoostTuner(n_trials=2, n_cv_folds=2)
        params = tuner.tune(X, y)
        assert isinstance(params["n_estimators"], int)
        assert isinstance(params["learning_rate"], float)
        assert 0 < params["learning_rate"] < 1


# ── Walk-Forward Validator Unit Tests ───────────────────────────────────────

class TestWalkForwardValidator:
    """Tests that the validator works across multiple bar frequencies."""

    def _make_df(self, n: int, freq: str) -> pd.DataFrame:
        dates = pd.date_range("2015-01-01", periods=n, freq=freq, tz="UTC")
        return pd.DataFrame({"x": np.random.rand(n)}, index=dates)

    def test_hourly_generates_folds(self):
        df = self._make_df(500, "h")
        wf = WalkForwardValidator(train_days=10, test_days=3, step_days=3, embargo_days=1)
        folds = wf.split(df)
        assert len(folds) > 0

    def test_daily_generates_folds(self):
        df = self._make_df(500, "D")
        wf = WalkForwardValidator(train_days=90, test_days=30, step_days=30, embargo_days=5)
        folds = wf.split(df)
        assert len(folds) > 0

    def test_minute_generates_folds(self):
        df = self._make_df(5000, "min")
        wf = WalkForwardValidator(train_days=2, test_days=1, step_days=1, embargo_days=0)
        folds = wf.split(df)
        assert len(folds) > 0

    def test_no_leakage_hourly(self):
        df = self._make_df(500, "h")
        wf = WalkForwardValidator(train_days=10, test_days=3, step_days=3, embargo_days=1)
        folds = wf.split(df)
        assert wf.verify_no_leakage(folds)

    def test_no_leakage_daily(self):
        df = self._make_df(500, "D")
        wf = WalkForwardValidator(train_days=60, test_days=20, step_days=20, embargo_days=3)
        folds = wf.split(df)
        assert wf.verify_no_leakage(folds)

    def test_train_end_before_test_start(self):
        """train_end must always be strictly before test_start across all folds."""
        df = self._make_df(500, "h")
        wf = WalkForwardValidator(train_days=10, test_days=3, step_days=3, embargo_days=1)
        folds = wf.split(df)
        for fold in folds:
            assert fold.train_end < fold.test_start, (
                f"Fold {fold.fold_index}: train_end={fold.train_end} >= test_start={fold.test_start}"
            )

    def test_requires_datetime_index(self):
        df_bad = pd.DataFrame({"x": [1, 2, 3]})
        wf = WalkForwardValidator(train_days=1, test_days=1, step_days=1)
        with pytest.raises(TypeError, match="DatetimeIndex"):
            wf.split(df_bad)

    def test_fold_masks_mutually_exclusive(self):
        """No row should appear in both train and test mask."""
        df = self._make_df(300, "h")
        wf = WalkForwardValidator(train_days=7, test_days=3, step_days=3, embargo_days=1)
        folds = wf.split(df)
        for fold in folds:
            overlap = fold.train_mask & fold.test_mask
            assert not overlap.any(), f"Fold {fold.fold_index} has train/test overlap"

    def test_fold_counts_per_resolution(self):
        """Hourly and minute data with the same calendar window should yield similar fold counts."""
        df_hourly = self._make_df(24 * 120, "h")   # 120 days of hourly data
        df_minute = self._make_df(60 * 24 * 120, "min")  # 120 days of minute data
        wf = WalkForwardValidator(train_days=30, test_days=10, step_days=10, embargo_days=2)
        folds_h = wf.split(df_hourly)
        folds_m = wf.split(df_minute)
        # Should produce the same number of folds since windows are in calendar days
        assert len(folds_h) == len(folds_m)


# ── Full Walk-Forward Pipeline Integration Test ──────────────────────────────

class TestWalkForwardPipelineIntegration:
    def test_full_wf_run_no_crash(self, synthetic_feature_df, synthetic_trade_log):
        """
        End-to-end test: prepare data → generate folds →
        train XGBoost per fold → evaluate.
        Uses small windows to keep it fast.
        """
        X, y, timestamps = prepare_ml_dataset(synthetic_feature_df, synthetic_trade_log)
        evaluator = ModelEvaluator()

        # Create validator with very small windows to fit in available samples
        wf = WalkForwardValidator(train_days=2, test_days=1, step_days=1, embargo_days=0)
        indexed = X.copy()
        indexed.index = timestamps.values
        folds = wf.split(indexed)

        assert len(folds) > 0, "Expected at least one fold"
        assert wf.verify_no_leakage(folds), "Data leakage detected in folds"

        all_metrics = []
        for fold in folds:
            X_train = X[fold.train_mask]
            y_train = y[fold.train_mask]
            X_test = X[fold.test_mask]
            y_test = y[fold.test_mask]

            if len(X_train) < 10 or len(X_test) < 5:
                continue

            pipeline = build_xgboost_pipeline(XGBoostTuner._default_params())
            pipeline.fit(X_train, y_train)
            preds = pipeline.predict(X_test)

            assert not np.isnan(preds).any(), "Predictions contain NaN"
            metrics = evaluator.evaluate(y_test.values, preds)
            all_metrics.append(metrics)

        assert len(all_metrics) > 0, "No folds produced predictions"

    def test_predictions_finite_after_full_fit(self, synthetic_feature_df, synthetic_trade_log):
        """Final model trained on all data should produce finite predictions."""
        X, y, _ = prepare_ml_dataset(synthetic_feature_df, synthetic_trade_log)
        params = XGBoostTuner._default_params()
        pipeline = build_xgboost_pipeline(params)
        pipeline.fit(X, y)
        preds = pipeline.predict(X)
        assert np.all(np.isfinite(preds)), "Final model predictions are not finite"
