"""
QuantTrade ML Pipeline — XGBoost Model Pipeline
Full sklearn-compatible pipeline with preprocessing and XGBoost.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import settings


class TimeSeriesImputer(BaseEstimator, TransformerMixin):
    """Forward-fill imputer that respects time series order (no look-ahead)."""

    def fit(self, X: pd.DataFrame, y=None):
        self.feature_names_in_ = list(X.columns) if isinstance(X, pd.DataFrame) else None
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X.copy()
        # Forward-fill then backward-fill for any remaining NaN at start
        df = df.ffill().bfill()
        # If still NaN (all-NaN column), fill with 0
        df = df.fillna(0)
        return df.values


class FeatureNamePassthrough(BaseEstimator, TransformerMixin):
    """Keeps track of feature names through the pipeline."""

    def fit(self, X: pd.DataFrame, y=None):
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = list(X.columns)
        return self

    def transform(self, X):
        return X


def build_xgboost_pipeline(xgb_params: dict | None = None) -> Pipeline:
    """
    Build a full sklearn Pipeline:
    Imputer → RobustScaler → XGBRegressor

    Args:
        xgb_params: XGBoost hyperparameters (from tuner)

    Returns:
        sklearn Pipeline ready for fit/predict
    """
    params = xgb_params or {
        "n_estimators": 500,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 5,
        "reg_alpha": 0.5,
        "reg_lambda": 1.0,
        "gamma": 0.1,
        "tree_method": "hist",
        "random_state": settings.random_seed,
        "verbosity": 0,
        "n_jobs": -1,
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
    }

    pipeline = Pipeline([
        ("imputer", TimeSeriesImputer()),
        ("scaler", RobustScaler()),
        ("model", xgb.XGBRegressor(**params)),
    ])
    return pipeline


def prepare_ml_dataset(
    feature_df: pd.DataFrame,
    trade_log: pd.DataFrame,
    target_col: str = "pnl_usd",
    min_samples: int = 100,
) -> tuple[pd.DataFrame, pd.Series, pd.Index]:
    """
    Align features with trade targets to create the ML dataset.

    Strategy: For each trade, match market features at entry_time.
    The target is the actual trade PnL.

    Args:
        feature_df: Full feature DataFrame with DatetimeIndex
        trade_log: Trade log with entry_time and PnL columns
        target_col: Column to use as prediction target
        min_samples: Minimum required samples

    Returns:
        (X, y, timestamps)
    """
    if trade_log.empty:
        raise ValueError("Trade log is empty — run simulation first")

    # Ensure entry_time is UTC
    trade_log = trade_log.copy()
    trade_log["entry_time"] = pd.to_datetime(trade_log["entry_time"], utc=True)

    # Sort both by time
    feature_df = feature_df.sort_index()
    trade_log = trade_log.sort_values("entry_time")

    # Match each trade to the nearest bar in feature_df
    feature_times = feature_df.index
    trade_times = trade_log["entry_time"]

    # Merge-asof: each trade gets the features from the bar just before entry
    feature_reset = feature_df.reset_index()
    trade_reset = trade_log.rename(columns={"entry_time": "timestamp"})

    # Normalise the index column name to "timestamp" regardless of whether the
    # DatetimeIndex was named (real pipeline) or unnamed (tests / ad-hoc usage)
    idx_col = feature_reset.columns[0]  # reset_index() puts index as first column
    if idx_col != "timestamp":
        feature_reset = feature_reset.rename(columns={idx_col: "timestamp"})

    # Remove tz for merge compatibility
    feature_reset["timestamp"] = feature_reset["timestamp"].dt.tz_localize(None)
    trade_reset["timestamp"] = trade_reset["timestamp"].dt.tz_localize(None)

    merged = pd.merge_asof(
        trade_reset.sort_values("timestamp"),
        feature_reset.sort_values("timestamp"),
        on="timestamp",
        direction="backward",
    )

    # Drop non-feature columns from trade log columns that ended up in merged
    trade_cols_to_drop = [c for c in trade_log.columns if c in merged.columns and c != "timestamp"]
    trade_only_cols = ["strategy_id", "direction", "entry_price", "exit_price",
                       "quantity", "pnl", "pnl_usd", "pnl_pct", "win", "holding_bars",
                       "exit_reason", "rolling_win_rate", "rolling_avg_pnl",
                       "prev_trade_pnl", target_col]

    # Get target
    if target_col not in merged.columns:
        raise ValueError(f"Target column '{target_col}' not found in merged DataFrame")

    y = merged[target_col].copy()

    # Drop target, metadata, and all-string columns from features
    exclude = set(trade_only_cols) | {"timestamp", "instrument"}
    feature_cols = [c for c in merged.columns
                    if c not in exclude
                    and pd.api.types.is_numeric_dtype(merged[c])]

    X = merged[feature_cols].copy()
    timestamps = pd.to_datetime(merged["timestamp"], utc=True)

    # Remove rows with NaN target
    valid_mask = y.notna() & np.isfinite(y)
    X = X[valid_mask]
    y = y[valid_mask]
    timestamps = timestamps[valid_mask]

    # Remove columns with >50% NaN
    null_rate = X.isnull().mean()
    good_cols = null_rate[null_rate < 0.5].index
    X = X[good_cols]

    # Replace inf
    X = X.replace([np.inf, -np.inf], np.nan)

    if len(X) < min_samples:
        raise ValueError(f"Insufficient ML samples: {len(X)} < {min_samples}")

    return X, y, timestamps
