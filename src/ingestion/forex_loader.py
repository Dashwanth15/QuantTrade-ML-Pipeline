"""
QuantTrade ML Pipeline — Forex Data Loader
Loads, validates, and normalises the EUR/USD hourly CSV dataset.
Returns a clean, UTC-indexed DataFrame ready for preprocessing.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import settings

# ------------------------------------------------------------------ #
# Column Mapping
# ------------------------------------------------------------------ #
RAW_COLUMNS = {
    "Date": "date",
    "Time": "time",
    "BO": "bid_open",
    "BH": "bid_high",
    "BL": "bid_low",
    "BC": "bid_close",
    "BCh": "bid_change",
    "AO": "ask_open",
    "AH": "ask_high",
    "AL": "ask_low",
    "AC": "ask_close",
    "ACh": "ask_change",
}

PRICE_COLS = [
    "bid_open", "bid_high", "bid_low", "bid_close",
    "ask_open", "ask_high", "ask_low", "ask_close",
]


class ForexLoader:
    """
    Loads the EUR/USD hourly CSV and returns a validated DataFrame.

    Responsibilities:
    - Column renaming and type coercion
    - DateTime parsing + UTC localization
    - Initial structural validation
    - Mid-price and spread computation
    """

    def __init__(self, file_path: Path | str | None = None) -> None:
        self.file_path = Path(file_path or settings.data_raw_path)
        self._df: pd.DataFrame | None = None

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def load(self) -> pd.DataFrame:
        """Load and return the full cleaned DataFrame."""
        logger.info("Loading EUR/USD data from {}", self.file_path)

        if not self.file_path.exists():
            raise FileNotFoundError(f"EUR/USD CSV not found: {self.file_path}")

        raw = self._read_csv()
        df = self._rename_columns(raw)
        df = self._parse_datetime(df)
        df = self._coerce_types(df)
        df = self._add_derived_columns(df)
        df = self._sort_and_deduplicate(df)
        self._validate_structure(df)

        self._df = df
        logger.success(
            "Loaded {} rows | {} to {}",
            len(df),
            df.index[0],
            df.index[-1],
        )
        return df

    @property
    def data(self) -> pd.DataFrame:
        if self._df is None:
            raise RuntimeError("Call .load() first")
        return self._df

    # ------------------------------------------------------------------ #
    # Private Methods
    # ------------------------------------------------------------------ #

    def _read_csv(self) -> pd.DataFrame:
        """Read raw CSV with minimal parsing."""
        try:
            df = pd.read_csv(
                self.file_path,
                dtype=str,  # Read everything as string first for safe parsing
                na_values=["", "NA", "N/A", "nan", "null"],
                low_memory=False,
            )
            logger.debug("Raw CSV shape: {}", df.shape)
            return df
        except Exception as exc:
            logger.exception("Failed to read CSV: {}", exc)
            raise

    def _rename_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rename raw columns to snake_case names."""
        missing = set(RAW_COLUMNS.keys()) - set(df.columns)
        if missing:
            raise ValueError(f"Missing expected columns: {missing}")
        return df.rename(columns=RAW_COLUMNS)

    def _parse_datetime(self, df: pd.DataFrame) -> pd.DataFrame:
        """Combine Date + Time into a UTC-aware DatetimeIndex."""
        datetime_str = df["date"].str.strip() + " " + df["time"].str.strip()
        timestamps = pd.to_datetime(datetime_str, format="%Y-%m-%d %H:%M", utc=True)
        df = df.drop(columns=["date", "time"])
        df.index = timestamps
        df.index.name = "timestamp"
        return df

    def _coerce_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert all price/change columns to float64."""
        numeric_cols = PRICE_COLS + ["bid_change", "ask_change"]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        n_nulls = df[PRICE_COLS].isnull().sum().sum()
        if n_nulls > 0:
            logger.warning("Found {} NaN values in price columns after coercion", n_nulls)
        return df

    def _add_derived_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add mid-price, spread, and instrument metadata."""
        # Mid prices
        df["mid_open"] = (df["bid_open"] + df["ask_open"]) / 2
        df["mid_high"] = (df["bid_high"] + df["ask_high"]) / 2
        df["mid_low"] = (df["bid_low"] + df["ask_low"]) / 2
        df["mid_close"] = (df["bid_close"] + df["ask_close"]) / 2

        # Bid-ask spread in pips (1 pip = 0.0001 for EUR/USD)
        df["spread"] = df["ask_close"] - df["bid_close"]
        df["spread_pips"] = df["spread"] / settings.pip_value

        # Typical price
        df["typical_price"] = (df["mid_high"] + df["mid_low"] + df["mid_close"]) / 3

        # Candle body / shadow
        df["candle_body"] = (df["mid_close"] - df["mid_open"]).abs()
        df["candle_range"] = df["mid_high"] - df["mid_low"]
        df["upper_shadow"] = df["mid_high"] - df[["mid_open", "mid_close"]].max(axis=1)
        df["lower_shadow"] = df[["mid_open", "mid_close"]].min(axis=1) - df["mid_low"]

        # Instrument
        df["instrument"] = "EURUSD"

        return df

    def _sort_and_deduplicate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure chronological order and no duplicate timestamps."""
        df = df.sort_index()
        n_dups = df.index.duplicated().sum()
        if n_dups > 0:
            logger.warning("Removing {} duplicate timestamps", n_dups)
            df = df[~df.index.duplicated(keep="last")]
        return df

    def _validate_structure(self, df: pd.DataFrame) -> None:
        """Run basic sanity checks on loaded data."""
        assert len(df) > 10_000, "Dataset appears too small — check file"
        assert df.index.is_monotonic_increasing, "Timestamps not sorted"
        assert (df["spread"] >= 0).all(), "Negative spreads detected"
        assert df["mid_close"].notna().mean() > 0.99, "Too many NaN mid prices"
        logger.debug("Structure validation passed")


def load_forex_data(file_path: Path | str | None = None) -> pd.DataFrame:
    """
    Convenience function to load and return EUR/USD data.

    Returns:
        pd.DataFrame with UTC DatetimeIndex and all price/derived columns.
    """
    return ForexLoader(file_path).load()


if __name__ == "__main__":
    df = load_forex_data()
    print(df.head())
    print(df.dtypes)
    print(f"\nShape: {df.shape}")
    print(f"Date range: {df.index[0]} → {df.index[-1]}")
    print(f"\nSpread stats (pips):\n{df['spread_pips'].describe()}")
