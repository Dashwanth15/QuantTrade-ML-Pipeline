"""
QuantTrade ML Pipeline — Price Feature Engineering
Computes returns, rolling statistics, momentum indicators, and
price-derived features. All rolling calculations respect time-series
order to prevent look-ahead bias.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import settings

WINDOWS = settings.rolling_windows  # [5, 10, 20, 50]


class PriceFeatureEngineer:
    """
    Generates price-based features using only past data at each timestep.
    All rolling operations use closed='left' equivalent (shift by 1 where needed).
    """

    def __init__(self, price_col: str = "mid_close") -> None:
        self.price_col = price_col

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        logger.debug("Engineering price features for {} rows", len(df))
        price = df[self.price_col]

        # ---- Returns ----
        df["return_1h"] = price.pct_change(1)
        df["return_4h"] = price.pct_change(4)
        df["return_12h"] = price.pct_change(12)
        df["return_24h"] = price.pct_change(24)
        df["return_48h"] = price.pct_change(48)
        df["return_1w"] = price.pct_change(168)  # 7 days

        # ---- Log Returns ----
        log_price = np.log(price)
        df["log_return_1h"] = log_price.diff(1)
        df["log_return_4h"] = log_price.diff(4)
        df["log_return_24h"] = log_price.diff(24)

        # ---- Absolute Returns ----
        df["abs_return_1h"] = df["return_1h"].abs()

        # ---- Rolling Statistics ----
        for window in WINDOWS:
            rolled = price.rolling(window, min_periods=window // 2)
            df[f"roll_mean_{window}"] = rolled.mean()
            df[f"roll_std_{window}"] = rolled.std()
            df[f"roll_min_{window}"] = rolled.min()
            df[f"roll_max_{window}"] = rolled.max()
            df[f"roll_median_{window}"] = rolled.median()

            # Return statistics
            ret_rolled = df["return_1h"].rolling(window, min_periods=window // 2)
            df[f"return_mean_{window}"] = ret_rolled.mean()
            df[f"return_std_{window}"] = ret_rolled.std()
            df[f"return_skew_{window}"] = ret_rolled.skew()

        # ---- Realized Volatility ----
        for window in [12, 24, 48, 168]:
            df[f"realized_vol_{window}h"] = (
                df["log_return_1h"].rolling(window, min_periods=window // 2).std()
                * np.sqrt(window)
            )

        # ---- Momentum ----
        for window in WINDOWS + [100]:
            df[f"momentum_{window}"] = price - price.shift(window)
            df[f"roc_{window}"] = price.pct_change(window)  # Rate of change

        # ---- Price Position Features ----
        for window in [20, 50]:
            roll_min = price.rolling(window).min()
            roll_max = price.rolling(window).max()
            df[f"price_position_{window}"] = (price - roll_min) / (roll_max - roll_min + 1e-10)

        # ---- Z-Score of Price ----
        for window in [20, 50]:
            roll_mean = price.rolling(window).mean()
            roll_std = price.rolling(window).std()
            df[f"price_zscore_{window}"] = (price - roll_mean) / (roll_std + 1e-10)

        # ---- Range-Based Features ----
        if "mid_high" in df.columns and "mid_low" in df.columns:
            high = df["mid_high"]
            low = df["mid_low"]
            open_ = df["mid_open"]

            df["true_range"] = np.maximum(
                high - low,
                np.maximum(
                    (high - price.shift(1)).abs(),
                    (low - price.shift(1)).abs(),
                )
            )
            df["bar_range"] = high - low
            df["bar_range_pct"] = df["bar_range"] / price

            # Candle direction
            df["candle_direction"] = np.sign(price - open_)

            # High-Low ratio
            df["hl_ratio"] = (high - price) / (high - low + 1e-10)

            # Gap from previous close
            df["overnight_gap"] = open_ - price.shift(1)
            df["overnight_gap_pct"] = df["overnight_gap"] / price.shift(1)

        # ---- Acceleration ----
        df["return_acceleration"] = df["return_1h"].diff()
        df["price_acceleration"] = df["return_1h"] - df["return_1h"].shift(5)

        # ---- Spread Features ----
        if "spread_pips" in df.columns:
            for window in [10, 24]:
                df[f"spread_roll_mean_{window}"] = df["spread_pips"].rolling(window).mean()
                df[f"spread_zscore_{window}"] = (
                    df["spread_pips"] - df[f"spread_roll_mean_{window}"]
                ) / (df["spread_pips"].rolling(window).std() + 1e-10)

        # ---- Volume Proxy (tick activity via candle range) ----
        # In absence of volume, range is a proxy for activity
        if "bar_range" in df.columns:
            for window in [12, 24]:
                df[f"activity_ratio_{window}"] = (
                    df["bar_range"] / df["bar_range"].rolling(window).mean()
                )

        logger.debug("Price features engineered successfully")
        return df


def add_price_features(df: pd.DataFrame, price_col: str = "mid_close") -> pd.DataFrame:
    """Add all price-derived features."""
    return PriceFeatureEngineer(price_col).transform(df)
