"""
QuantTrade ML Pipeline — Macro Feature Engineering
Merges macroeconomic events with market data and engineers
event-proximity, surprise, and impact features.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

CATEGORIES = ["inflation", "monetary_policy", "growth", "employment", "consumption", "trade"]
CURRENCIES = ["USD", "EUR"]


class MacroFeatureEngineer:
    """
    Merges macro events with the forex OHLC DataFrame and engineers:
    - Event proximity features (minutes to/from each event type)
    - Surprise magnitude and direction
    - Impact score aggregation
    - Categorical encodings
    """

    def transform(
        self, df: pd.DataFrame, macro_df: pd.DataFrame | None
    ) -> pd.DataFrame:
        """
        Merge macro events and create features.

        Args:
            df: Forex DataFrame with UTC DatetimeIndex
            macro_df: Macro events DataFrame from MacroEventScraper

        Returns:
            DataFrame with macro features added
        """
        df = df.copy()

        if macro_df is None or macro_df.empty:
            logger.warning("No macro events available — adding zero-filled macro features")
            return self._add_empty_macro_features(df)

        logger.info("Engineering macro features | events={}", len(macro_df))
        macro_df = self._prepare_macro_df(macro_df)

        df = self._add_event_proximity_features(df, macro_df)
        df = self._add_impact_window_features(df, macro_df)
        df = self._add_surprise_features(df, macro_df)
        df = self._add_event_count_features(df, macro_df)

        logger.debug("Macro features engineered successfully")
        return df

    # ------------------------------------------------------------------ #
    # Private Methods
    # ------------------------------------------------------------------ #

    def _prepare_macro_df(self, macro_df: pd.DataFrame) -> pd.DataFrame:
        """Ensure macro_df has proper UTC timestamp and is sorted."""
        df = macro_df.copy()
        if "timestamp_utc" not in df.columns:
            raise ValueError("macro_df must have 'timestamp_utc' column")
        df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
        df = df.sort_values("timestamp_utc").reset_index(drop=True)
        return df

    def _add_event_proximity_features(
        self, df: pd.DataFrame, macro_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Add minutes-since and minutes-until features for each event category."""
        event_times = macro_df["timestamp_utc"].values.astype(np.int64)
        bar_times = df.index.values.astype(np.int64)
        ns_per_min = 60 * 1_000_000_000

        # Global (all events)
        df["mins_since_last_event"] = self._minutes_since(bar_times, event_times, ns_per_min)
        df["mins_until_next_event"] = self._minutes_until(bar_times, event_times, ns_per_min)

        # Per category
        for cat in CATEGORIES:
            cat_mask = macro_df["category"] == cat
            if cat_mask.sum() == 0:
                df[f"mins_since_{cat}"] = np.nan
                df[f"mins_until_{cat}"] = np.nan
                continue
            cat_times = macro_df.loc[cat_mask, "timestamp_utc"].values.astype(np.int64)
            df[f"mins_since_{cat}"] = self._minutes_since(bar_times, cat_times, ns_per_min)
            df[f"mins_until_{cat}"] = self._minutes_until(bar_times, cat_times, ns_per_min)

        # High-impact only
        hi_mask = macro_df["impact_score"] >= 3
        if hi_mask.sum() > 0:
            hi_times = macro_df.loc[hi_mask, "timestamp_utc"].values.astype(np.int64)
            df["mins_since_high_impact"] = self._minutes_since(bar_times, hi_times, ns_per_min)
            df["mins_until_high_impact"] = self._minutes_until(bar_times, hi_times, ns_per_min)
        else:
            df["mins_since_high_impact"] = np.nan
            df["mins_until_high_impact"] = np.nan

        # EUR/USD specific
        for ccy in CURRENCIES:
            ccy_mask = macro_df.get("currency", pd.Series()) == ccy
            if isinstance(ccy_mask, pd.Series) and ccy_mask.sum() > 0:
                ccy_times = macro_df.loc[ccy_mask, "timestamp_utc"].values.astype(np.int64)
                df[f"mins_since_{ccy}_event"] = self._minutes_since(bar_times, ccy_times, ns_per_min)
                df[f"mins_until_{ccy}_event"] = self._minutes_until(bar_times, ccy_times, ns_per_min)
            else:
                df[f"mins_since_{ccy}_event"] = np.nan
                df[f"mins_until_{ccy}_event"] = np.nan

        return df

    def _add_impact_window_features(
        self, df: pd.DataFrame, macro_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Add features for bars within ±N hours of a macro event.
        These capture the volatility spike around events.
        """
        windows_hours = [1, 2, 4, 12, 24]

        for window_h in windows_hours:
            window_ns = window_h * 3600 * 1_000_000_000
            bar_times = df.index.values.astype(np.int64)
            event_times = macro_df["timestamp_utc"].values.astype(np.int64)

            # Is any event within window?
            in_window = np.zeros(len(df), dtype=int)
            for et in event_times:
                near = np.abs(bar_times - et) <= window_ns
                in_window |= near.astype(int)

            df[f"event_within_{window_h}h"] = in_window

        # Max impact score of events within 24h window
        impact_scores = np.zeros(len(df))
        surprise_magnitudes = np.zeros(len(df))
        window_24h_ns = 24 * 3600 * 1_000_000_000
        bar_times = df.index.values.astype(np.int64)

        for _, event_row in macro_df.iterrows():
            et = event_row["timestamp_utc"].value
            near_mask = np.abs(bar_times - et) <= window_24h_ns
            impact = float(event_row.get("impact_score", 1))
            surprise = float(event_row.get("surprise_magnitude", 0) or 0)
            impact_scores = np.where(near_mask, np.maximum(impact_scores, impact), impact_scores)
            surprise_magnitudes = np.where(near_mask, np.maximum(surprise_magnitudes, surprise), surprise_magnitudes)

        df["nearby_event_impact_score"] = impact_scores
        df["nearby_event_surprise"] = surprise_magnitudes

        return df

    def _add_surprise_features(
        self, df: pd.DataFrame, macro_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Add rolling surprise aggregation features."""
        macro_utc = macro_df["timestamp_utc"]
        bar_times = df.index

        # For each bar, find events in the past 30 days
        surprise_30d = np.zeros(len(df))
        surprise_dir_30d = np.zeros(len(df))
        n_events_30d = np.zeros(len(df))

        window_30d_ns = 30 * 24 * 3600 * 1_000_000_000

        macro_ns = macro_utc.values.astype(np.int64)
        bar_ns = bar_times.values.astype(np.int64)

        for i, bt in enumerate(bar_ns):
            past_mask = (macro_ns >= bt - window_30d_ns) & (macro_ns <= bt)
            if past_mask.sum() > 0:
                subset = macro_df[past_mask]
                surprises = subset["surprise"].dropna()
                if len(surprises) > 0:
                    surprise_30d[i] = surprises.abs().mean()
                    surprise_dir_30d[i] = surprises.mean()
                n_events_30d[i] = past_mask.sum()

        df["surprise_magnitude_30d"] = surprise_30d
        df["surprise_direction_30d"] = surprise_dir_30d
        df["event_count_30d"] = n_events_30d
        return df

    def _add_event_count_features(
        self, df: pd.DataFrame, macro_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Count events in recent windows."""
        macro_ns = macro_df["timestamp_utc"].values.astype(np.int64)
        bar_ns = df.index.values.astype(np.int64)

        for window_h in [24, 48, 168]:
            window_ns = window_h * 3600 * 1_000_000_000
            counts = np.array([
                np.sum((macro_ns >= bt - window_ns) & (macro_ns <= bt))
                for bt in bar_ns
            ])
            df[f"event_count_{window_h}h"] = counts

        return df

    def _add_empty_macro_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add NaN/zero macro features when no events are available."""
        feature_cols = (
            ["mins_since_last_event", "mins_until_next_event",
             "mins_since_high_impact", "mins_until_high_impact",
             "nearby_event_impact_score", "nearby_event_surprise",
             "surprise_magnitude_30d", "surprise_direction_30d", "event_count_30d"]
            + [f"mins_since_{cat}" for cat in CATEGORIES]
            + [f"mins_until_{cat}" for cat in CATEGORIES]
            + [f"mins_since_{c}_event" for c in CURRENCIES]
            + [f"mins_until_{c}_event" for c in CURRENCIES]
            + [f"event_within_{h}h" for h in [1, 2, 4, 12, 24]]
            + [f"event_count_{h}h" for h in [24, 48, 168]]
        )
        for col in feature_cols:
            df[col] = 0.0
        return df

    @staticmethod
    def _minutes_since(
        bar_times: np.ndarray, event_times: np.ndarray, ns_per_min: int
    ) -> np.ndarray:
        """Minutes elapsed since the most recent past event."""
        result = np.full(len(bar_times), np.nan)
        for i, bt in enumerate(bar_times):
            past = event_times[event_times <= bt]
            if len(past) > 0:
                result[i] = (bt - past[-1]) / ns_per_min
        return result

    @staticmethod
    def _minutes_until(
        bar_times: np.ndarray, event_times: np.ndarray, ns_per_min: int
    ) -> np.ndarray:
        """Minutes remaining until the next future event."""
        result = np.full(len(bar_times), np.nan)
        for i, bt in enumerate(bar_times):
            future = event_times[event_times >= bt]
            if len(future) > 0:
                result[i] = (future[0] - bt) / ns_per_min
        return result


def add_macro_features(
    df: pd.DataFrame, macro_df: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Add macro event features to the forex DataFrame."""
    return MacroFeatureEngineer().transform(df, macro_df)
