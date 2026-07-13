"""
QuantTrade ML Pipeline — Feature Store
Central registry for managing, versioning, and serving feature sets.
Orchestrates all feature engineers and provides a unified interface.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import settings
from src.features.macro_features import add_macro_features
from src.features.price_features import add_price_features
from src.features.technical import add_technical_indicators
from src.features.time_features import add_time_features


# ------------------------------------------------------------------ #
# Feature Groups Registry
# ------------------------------------------------------------------ #
FEATURE_GROUPS: dict[str, list[str]] = {
    "time": [],          # Populated dynamically
    "price": [],
    "technical": [],
    "macro": [],
}

# Features that should NOT be used as ML inputs (metadata columns)
METADATA_COLS = {
    "instrument", "is_outlier", "is_weekend_gap", "is_spread_anomaly",
    "data_quality_score", "hourly_return", "bid_change", "ask_change",
}

# Features to always exclude from the model (direct target leakage)
LEAKAGE_COLS = {
    "mid_close", "bid_close", "ask_close",
    "mid_open", "mid_high", "mid_low",
    "bid_open", "bid_high", "bid_low",
    "ask_open", "ask_high", "ask_low",
    "spread", "spread_pips", "typical_price",
    "candle_body", "candle_range", "upper_shadow", "lower_shadow",
}


class FeatureStore:
    """
    Central orchestrator for feature engineering.

    Responsibilities:
    - Run all feature engineers in the correct order
    - Track feature schemas and versions
    - Provide feature groups for model selection
    - Cache computed features to parquet for reuse
    - Handle feature selection and validation
    """

    def __init__(self, cache_dir: Path | None = None) -> None:
        self.cache_dir = Path(cache_dir or settings.data_processed_path)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._feature_schema: dict[str, str] = {}
        self._feature_groups: dict[str, list[str]] = {}

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def build_features(
        self,
        forex_df: pd.DataFrame,
        macro_df: pd.DataFrame | None = None,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """
        Build the full feature set from raw forex + macro data.

        Args:
            forex_df: Cleaned forex DataFrame
            macro_df: Macro events DataFrame (optional)
            use_cache: Load from parquet cache if available

        Returns:
            Feature DataFrame ready for ML
        """
        # Check cache
        cache_key = self._compute_cache_key(forex_df, macro_df)
        cache_path = self.cache_dir / f"features_{cache_key}.parquet"

        if use_cache and cache_path.exists():
            logger.info("Loading cached features from {}", cache_path)
            df = pd.read_parquet(cache_path)
            self._infer_feature_groups(df)
            return df

        logger.info("Building feature matrix | forex_rows={}", len(forex_df))
        df = forex_df.copy()

        # ---- Apply feature engineers ----
        time_cols_before = set(df.columns)
        df = add_time_features(df)
        self._feature_groups["time"] = list(set(df.columns) - time_cols_before)

        price_cols_before = set(df.columns)
        df = add_price_features(df)
        self._feature_groups["price"] = list(set(df.columns) - price_cols_before)

        tech_cols_before = set(df.columns)
        df = add_technical_indicators(df)
        self._feature_groups["technical"] = list(set(df.columns) - tech_cols_before)

        macro_cols_before = set(df.columns)
        df = add_macro_features(df, macro_df)
        self._feature_groups["macro"] = list(set(df.columns) - macro_cols_before)

        # ---- Post-processing ----
        df = self._handle_infinities(df)
        df = self._log_feature_summary(df)

        # ---- Cache ----
        df.to_parquet(cache_path, index=True)
        logger.info("Features cached to {}", cache_path)

        return df

    def get_ml_features(
        self,
        df: pd.DataFrame,
        exclude_groups: list[str] | None = None,
        max_features: int | None = None,
    ) -> tuple[pd.DataFrame, list[str]]:
        """
        Return only the columns appropriate for ML training.

        Args:
            df: Full feature DataFrame
            exclude_groups: Feature groups to exclude
            max_features: Limit number of features

        Returns:
            (X_df, feature_names)
        """
        exclude_groups = exclude_groups or []

        # Start with all columns
        all_cols = set(df.columns)

        # Remove metadata and leakage columns
        exclude_cols = METADATA_COLS | LEAKAGE_COLS

        # Remove excluded group columns
        for grp in exclude_groups:
            exclude_cols.update(self._feature_groups.get(grp, []))

        # Keep only numeric columns not in exclusion list
        feature_cols = []
        for col in df.columns:
            if col in exclude_cols:
                continue
            if not pd.api.types.is_numeric_dtype(df[col]):
                continue
            feature_cols.append(col)

        X = df[feature_cols].copy()

        # Limit features if requested
        if max_features and len(feature_cols) > max_features:
            # Keep features with most variance (proxy for informativeness)
            variances = X.var()
            top_cols = variances.nlargest(max_features).index.tolist()
            X = X[top_cols]
            feature_cols = top_cols

        logger.info("ML feature matrix: {} samples × {} features", len(X), len(feature_cols))
        return X, feature_cols

    def get_feature_groups(self) -> dict[str, list[str]]:
        """Return the feature group mapping."""
        return self._feature_groups.copy()

    def get_feature_schema(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return a DataFrame describing each feature."""
        records = []
        X, feature_cols = self.get_ml_features(df)
        for col in feature_cols:
            series = X[col].dropna()
            group = self._get_column_group(col)
            records.append({
                "feature": col,
                "group": group,
                "dtype": str(X[col].dtype),
                "null_rate": X[col].isnull().mean(),
                "mean": series.mean() if len(series) > 0 else np.nan,
                "std": series.std() if len(series) > 0 else np.nan,
                "min": series.min() if len(series) > 0 else np.nan,
                "max": series.max() if len(series) > 0 else np.nan,
            })
        return pd.DataFrame(records)

    # ------------------------------------------------------------------ #
    # Private Methods
    # ------------------------------------------------------------------ #

    def _handle_infinities(self, df: pd.DataFrame) -> pd.DataFrame:
        """Replace inf/-inf with NaN."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        n_inf = np.isinf(df[numeric_cols].values).sum()
        if n_inf > 0:
            logger.warning("Replacing {} infinite values with NaN", n_inf)
            df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
        return df

    def _log_feature_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        total_cols = len(df.columns)
        numeric_cols = df.select_dtypes(include=[np.number]).shape[1]
        null_rate = df.select_dtypes(include=[np.number]).isnull().mean().mean()
        logger.success(
            "Feature matrix built: {} total cols | {} numeric | {:.2%} null rate",
            total_cols, numeric_cols, null_rate,
        )
        return df

    def _compute_cache_key(
        self, forex_df: pd.DataFrame, macro_df: pd.DataFrame | None
    ) -> str:
        """Compute a hash key for the input data to use for caching."""
        forex_hash = hashlib.md5(
            f"{len(forex_df)}_{str(forex_df.index[0])}_{str(forex_df.index[-1])}".encode()
        ).hexdigest()[:8]
        macro_hash = hashlib.md5(
            f"{len(macro_df) if macro_df is not None else 0}".encode()
        ).hexdigest()[:8]
        return f"{forex_hash}_{macro_hash}"

    def _infer_feature_groups(self, df: pd.DataFrame) -> None:
        """Infer feature groups from column naming conventions."""
        time_prefixes = ("hour", "day", "month", "quarter", "year", "week", "is_",
                         "dow", "session", "london", "asian", "new_york", "overlap",
                         "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos")
        price_prefixes = ("return_", "log_return", "roll_", "realized_vol", "momentum_",
                          "roc_", "price_pos", "price_z", "true_range", "bar_range",
                          "spread_roll", "candle_dir", "hl_ratio", "overnight", "abs_return",
                          "price_acceleration", "return_acceleration", "activity_ratio")
        tech_prefixes = ("rsi", "macd", "ema_", "sma_", "bb_", "atr_", "stoch",
                         "adx", "williams", "cci", "ichimoku", "obv", "composite",
                         "golden", "death", "price_above_", "price_vs_")
        macro_prefixes = ("mins_", "event_", "surprise", "nearby_")

        self._feature_groups = {"time": [], "price": [], "technical": [], "macro": [], "other": []}
        for col in df.columns:
            if any(col.startswith(p) for p in time_prefixes):
                self._feature_groups["time"].append(col)
            elif any(col.startswith(p) for p in price_prefixes):
                self._feature_groups["price"].append(col)
            elif any(col.startswith(p) for p in tech_prefixes):
                self._feature_groups["technical"].append(col)
            elif any(col.startswith(p) for p in macro_prefixes):
                self._feature_groups["macro"].append(col)
            else:
                self._feature_groups["other"].append(col)

    def _get_column_group(self, col: str) -> str:
        for group, cols in self._feature_groups.items():
            if col in cols:
                return group
        return "other"


# Module-level instance for convenience
feature_store = FeatureStore()
