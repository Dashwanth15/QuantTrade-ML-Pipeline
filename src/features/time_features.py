"""
QuantTrade ML Pipeline — Time Feature Engineering
Extracts rich calendar and market-session features from the DatetimeIndex.
All features are strictly computed from the timestamp — zero look-ahead.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# ------------------------------------------------------------------ #
# Trading Session Definitions (UTC hours)
# ------------------------------------------------------------------ #
SESSIONS = {
    "asian":    (22, 8),   # Tokyo: 22:00 – 08:00 UTC (wraps midnight)
    "london":   (8, 17),   # London: 08:00 – 17:00 UTC
    "new_york": (13, 22),  # New York: 13:00 – 22:00 UTC
}

SESSION_OVERLAPS = {
    "tokyo_london":   (8, 9),
    "london_new_york": (13, 17),
}


class TimeFeatureEngineer:
    """
    Generates temporal and session-based features from a DatetimeIndex.

    Features generated:
    - Calendar: hour, day_of_week, month, quarter, year, week_of_year
    - Cyclical encodings: sin/cos of hour, day_of_week, month
    - Session flags: is_asian, is_london, is_new_york, session_overlap
    - Special flags: is_weekend, is_monday_open, is_friday_close
    - Session start/end proximity
    """

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time features in-place (returns copy)."""
        df = df.copy()
        logger.debug("Engineering time features for {} rows", len(df))
        idx = df.index

        # ---- Calendar features ----
        df["hour"] = idx.hour
        df["day_of_week"] = idx.dayofweek          # 0=Monday, 6=Sunday
        df["month"] = idx.month
        df["quarter"] = idx.quarter
        df["year"] = idx.year
        df["week_of_year"] = idx.isocalendar().week.astype(int)
        df["day_of_year"] = idx.dayofyear
        df["days_in_month"] = idx.days_in_month

        # ---- Cyclical encodings (preserve periodicity) ----
        df["hour_sin"] = np.sin(2 * np.pi * idx.hour / 24)
        df["hour_cos"] = np.cos(2 * np.pi * idx.hour / 24)
        df["dow_sin"] = np.sin(2 * np.pi * idx.dayofweek / 7)
        df["dow_cos"] = np.cos(2 * np.pi * idx.dayofweek / 7)
        df["month_sin"] = np.sin(2 * np.pi * (idx.month - 1) / 12)
        df["month_cos"] = np.cos(2 * np.pi * (idx.month - 1) / 12)

        # ---- Boolean flags ----
        df["is_weekend"] = (idx.dayofweek >= 5).astype(int)
        df["is_monday"] = (idx.dayofweek == 0).astype(int)
        df["is_friday"] = (idx.dayofweek == 4).astype(int)
        df["is_month_start"] = idx.is_month_start.astype(int)
        df["is_month_end"] = idx.is_month_end.astype(int)
        df["is_quarter_start"] = idx.is_quarter_start.astype(int)
        df["is_quarter_end"] = idx.is_quarter_end.astype(int)

        # ---- Trading sessions ----
        hour = idx.hour
        df["is_asian_session"] = self._asian_session_mask(hour).astype(int)
        df["is_london_session"] = ((hour >= 8) & (hour < 17)).astype(int)
        df["is_new_york_session"] = ((hour >= 13) & (hour < 22)).astype(int)

        # Session overlaps (highest liquidity periods)
        df["is_tokyo_london_overlap"] = ((hour >= 8) & (hour < 9)).astype(int)
        df["is_london_ny_overlap"] = ((hour >= 13) & (hour < 17)).astype(int)
        df["is_any_overlap"] = (
            df["is_tokyo_london_overlap"] | df["is_london_ny_overlap"]
        ).astype(int)

        # Session label (for display)
        df["trading_session"] = self._assign_session_label(hour)

        # Active trading hours (all 3 sessions combined)
        df["is_active_hours"] = (
            df["is_asian_session"] |
            df["is_london_session"] |
            df["is_new_york_session"]
        ).astype(int)

        # ---- Intraday positioning ----
        # Hours since market open (London as reference)
        london_open_hour = 8
        df["hours_since_london_open"] = (hour - london_open_hour) % 24
        df["hours_to_london_close"] = (17 - hour) % 24

        # NY open proximity
        df["hours_since_ny_open"] = (hour - 13) % 24
        df["hours_to_ny_close"] = (22 - hour) % 24

        logger.debug("Time features engineered: {} new columns", 30)
        return df

    @staticmethod
    def _asian_session_mask(hour: pd.Index) -> pd.Series:
        """Asian session wraps midnight: 22:00–00:00 and 00:00–08:00."""
        return pd.Series((hour >= 22) | (hour < 8), index=hour.index if hasattr(hour, 'index') else None)

    @staticmethod
    def _assign_session_label(hour: pd.Index) -> pd.Series:
        """Assign a string session label for the most active session."""
        conditions = [
            (hour >= 13) & (hour < 17),   # London + NY overlap
            (hour >= 8) & (hour < 13),    # London
            (hour >= 17) & (hour < 22),   # New York
            (hour >= 22) | (hour < 8),    # Asian
        ]
        labels = ["london_ny_overlap", "london", "new_york", "asian"]
        result = pd.Series("off_hours", index=range(len(hour)))
        for cond, label in zip(reversed(conditions), reversed(labels)):
            result[cond] = label
        return result.values


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add all time features to a DataFrame with UTC DatetimeIndex."""
    return TimeFeatureEngineer().transform(df)
