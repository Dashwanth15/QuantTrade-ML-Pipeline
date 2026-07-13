"""
QuantTrade ML Pipeline — Data Cleaner
Performs market-microstructure-aware cleaning of the EUR/USD dataset:
- Missing value handling (forward-fill for OHLC)
- Weekend gap detection and flagging
- Spread anomaly removal
- Outlier detection via IQR + Z-score on returns
- Price continuity checks
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import settings

PRICE_COLS = [
    "bid_open", "bid_high", "bid_low", "bid_close",
    "ask_open", "ask_high", "ask_low", "ask_close",
    "mid_open", "mid_high", "mid_low", "mid_close",
]

MID_OHLC = ["mid_open", "mid_high", "mid_low", "mid_close"]


class DataCleaner:
    """
    Cleans raw EUR/USD DataFrame for production use.
    All operations are market-microstructure-aware:
    - Weekend gaps are expected, not errors
    - Large spreads during news events may be real, flagged not dropped
    - Outliers are flagged with metadata columns, not silently removed
    """

    # Thresholds
    MAX_SPREAD_PIPS = 10.0          # Above this is suspicious
    MAX_RETURN_ABS = 0.03           # 3% hourly move — extreme
    MIN_PRICE = 0.8                 # EUR/USD historical minimum
    MAX_PRICE = 2.0                 # EUR/USD historical maximum
    ZSCORE_OUTLIER_THRESHOLD = 5.0  # Z-score for return outliers
    IQR_MULTIPLIER = 3.0            # IQR multiplier for outlier fence

    def __init__(self) -> None:
        self.cleaning_report: dict[str, int | float] = {}

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Full cleaning pipeline.

        Args:
            df: Raw DataFrame from ForexLoader

        Returns:
            Cleaned DataFrame with quality metadata columns added
        """
        logger.info("Starting data cleaning pipeline | rows={}", len(df))
        original_len = len(df)

        df = self._add_quality_flags(df)
        df = self._handle_missing_values(df)
        df = self._validate_ohlc_integrity(df)
        df = self._detect_weekend_gaps(df)
        df = self._detect_spread_anomalies(df)
        df = self._detect_return_outliers(df)
        df = self._fill_derived_columns(df)

        self.cleaning_report["original_rows"] = original_len
        self.cleaning_report["final_rows"] = len(df)
        self.cleaning_report["rows_removed"] = original_len - len(df)
        self.cleaning_report["outlier_rows"] = int(df.get("is_outlier", pd.Series([False])).sum())

        logger.success(
            "Cleaning complete | rows={} | outliers_flagged={}",
            len(df),
            self.cleaning_report["outlier_rows"],
        )
        return df

    def get_report(self) -> dict:
        """Return cleaning summary statistics."""
        return self.cleaning_report

    # ------------------------------------------------------------------ #
    # Private Methods
    # ------------------------------------------------------------------ #

    def _add_quality_flags(self, df: pd.DataFrame) -> pd.DataFrame:
        """Initialize quality flag columns."""
        df = df.copy()
        df["is_outlier"] = False
        df["is_weekend_gap"] = False
        df["is_spread_anomaly"] = False
        df["data_quality_score"] = 1.0  # 1.0 = perfect, 0.0 = bad
        return df

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values with market-appropriate strategies.
        - OHLC prices: forward-fill (last known price carries forward)
        - Change columns: recalculate from prices
        - Derived columns: recalculate
        """
        # Count before
        null_counts = df[PRICE_COLS].isnull().sum()
        total_nulls = null_counts.sum()

        if total_nulls == 0:
            logger.debug("No missing price values detected")
            return df

        logger.warning("Filling {} missing price values", total_nulls)
        self.cleaning_report["null_values_filled"] = int(total_nulls)

        # Forward-fill OHLC (last known price carries forward in forex)
        available_price_cols = [c for c in PRICE_COLS if c in df.columns]
        df[available_price_cols] = df[available_price_cols].ffill()

        # Back-fill any remaining at start of series
        df[available_price_cols] = df[available_price_cols].bfill()

        # Rows with no price data at all — remove
        all_null_mask = df[MID_OHLC].isnull().all(axis=1)
        if all_null_mask.sum() > 0:
            logger.warning("Removing {} rows with no price data", all_null_mask.sum())
            df = df[~all_null_mask]

        return df

    def _validate_ohlc_integrity(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Enforce OHLC constraints:
        - High >= max(Open, Close)
        - Low <= min(Open, Close)
        - All prices within historical bounds
        """
        prefix_pairs = [("bid", "bid"), ("ask", "ask"), ("mid", "mid")]
        violations = 0

        for pfx, _ in prefix_pairs:
            open_col = f"{pfx}_open"
            high_col = f"{pfx}_high"
            low_col = f"{pfx}_low"
            close_col = f"{pfx}_close"

            if not all(c in df.columns for c in [open_col, high_col, low_col, close_col]):
                continue

            # High should be >= open and close
            high_violation = df[high_col] < df[[open_col, close_col]].max(axis=1)
            if high_violation.sum() > 0:
                violations += high_violation.sum()
                df.loc[high_violation, high_col] = df.loc[
                    high_violation, [open_col, close_col]
                ].max(axis=1)

            # Low should be <= open and close
            low_violation = df[low_col] > df[[open_col, close_col]].min(axis=1)
            if low_violation.sum() > 0:
                violations += low_violation.sum()
                df.loc[low_violation, low_col] = df.loc[
                    low_violation, [open_col, close_col]
                ].min(axis=1)

        if violations > 0:
            logger.warning("Corrected {} OHLC integrity violations", violations)
            self.cleaning_report["ohlc_violations_corrected"] = violations

        # Price bounds check
        for col in ["mid_close"]:
            if col not in df.columns:
                continue
            out_of_bounds = (df[col] < self.MIN_PRICE) | (df[col] > self.MAX_PRICE)
            if out_of_bounds.sum() > 0:
                logger.warning(
                    "{} prices out of historical bounds [{}–{}]",
                    out_of_bounds.sum(), self.MIN_PRICE, self.MAX_PRICE,
                )
                df.loc[out_of_bounds, "is_outlier"] = True
                df.loc[out_of_bounds, "data_quality_score"] = 0.0

        return df

    def _detect_weekend_gaps(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Flag bars that follow a weekend gap.
        Forex closes ~22:00 UTC Friday, re-opens ~22:00 UTC Sunday.
        A gap > 30 hours between consecutive bars is a weekend gap.
        """
        time_diffs = df.index.to_series().diff().dt.total_seconds() / 3600  # hours
        weekend_mask = time_diffs > 30  # Normal is 1 hour; > 30h = gap
        df.loc[weekend_mask, "is_weekend_gap"] = True
        weekend_count = weekend_mask.sum()
        logger.debug("Detected {} weekend/holiday gaps", weekend_count)
        self.cleaning_report["weekend_gaps"] = int(weekend_count)
        return df

    def _detect_spread_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        """Flag bars where bid-ask spread exceeds threshold."""
        if "spread_pips" not in df.columns:
            return df

        anomaly_mask = df["spread_pips"] > self.MAX_SPREAD_PIPS
        df.loc[anomaly_mask, "is_spread_anomaly"] = True
        df.loc[anomaly_mask, "data_quality_score"] -= 0.3

        anomaly_count = anomaly_mask.sum()
        if anomaly_count > 0:
            logger.info(
                "Flagged {} spread anomaly bars (>{} pips)",
                anomaly_count, self.MAX_SPREAD_PIPS,
            )
        self.cleaning_report["spread_anomalies"] = int(anomaly_count)
        return df

    def _detect_return_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect return outliers using IQR and Z-score.
        Outliers are FLAGGED not removed — they may be real news events.
        """
        if "mid_close" not in df.columns:
            return df

        returns = df["mid_close"].pct_change()
        df["hourly_return"] = returns

        # Z-score method
        mean_r = returns.mean()
        std_r = returns.std()
        z_scores = (returns - mean_r) / std_r
        zscore_outliers = z_scores.abs() > self.ZSCORE_OUTLIER_THRESHOLD

        # IQR method
        q1 = returns.quantile(0.25)
        q3 = returns.quantile(0.75)
        iqr = q3 - q1
        lower_fence = q1 - self.IQR_MULTIPLIER * iqr
        upper_fence = q3 + self.IQR_MULTIPLIER * iqr
        iqr_outliers = (returns < lower_fence) | (returns > upper_fence)

        # Flag if either method triggers
        outlier_mask = zscore_outliers | iqr_outliers
        df.loc[outlier_mask, "is_outlier"] = True
        df.loc[outlier_mask, "data_quality_score"] -= 0.2

        # Also flag extreme absolute moves
        extreme_mask = returns.abs() > self.MAX_RETURN_ABS
        df.loc[extreme_mask, "is_outlier"] = True

        outlier_count = outlier_mask.sum()
        logger.info(
            "Flagged {} return outliers | zscore={} | iqr={}",
            outlier_count, zscore_outliers.sum(), iqr_outliers.sum(),
        )
        self.cleaning_report["return_outliers"] = int(outlier_count)
        return df

    def _fill_derived_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Recalculate any derived columns that might be stale after cleaning."""
        if "bid_close" in df.columns and "ask_close" in df.columns:
            df["spread"] = df["ask_close"] - df["bid_close"]
            df["spread_pips"] = df["spread"] / settings.pip_value

        if "mid_open" not in df.columns and "bid_open" in df.columns:
            df["mid_open"] = (df["bid_open"] + df["ask_open"]) / 2
            df["mid_high"] = (df["bid_high"] + df["ask_high"]) / 2
            df["mid_low"] = (df["bid_low"] + df["ask_low"]) / 2
            df["mid_close"] = (df["bid_close"] + df["ask_close"]) / 2

        # Clip quality score to [0, 1]
        if "data_quality_score" in df.columns:
            df["data_quality_score"] = df["data_quality_score"].clip(0.0, 1.0)

        return df


def clean_forex_data(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Clean a raw forex DataFrame.

    Returns:
        (cleaned_df, cleaning_report)
    """
    cleaner = DataCleaner()
    cleaned = cleaner.clean(df)
    return cleaned, cleaner.get_report()
