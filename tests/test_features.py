"""
QuantTrade ML Pipeline — Feature Engineering Tests
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.features.time_features import add_time_features
from src.features.price_features import add_price_features
from src.features.technical import add_technical_indicators


@pytest.fixture
def sample_forex_df():
    """Create a sample forex DataFrame for testing."""
    n = 500
    np.random.seed(42)
    dates = pd.date_range("2010-01-01", periods=n, freq="h", tz="UTC")
    price = 1.2 + np.cumsum(np.random.normal(0, 0.0005, n))
    spread = np.abs(np.random.normal(0.0002, 0.00005, n))

    df = pd.DataFrame({
        "mid_open": price + np.random.normal(0, 0.0001, n),
        "mid_high": price + np.abs(np.random.normal(0, 0.0003, n)),
        "mid_low": price - np.abs(np.random.normal(0, 0.0003, n)),
        "mid_close": price,
        "bid_open": price - spread / 2,
        "bid_high": price + np.abs(np.random.normal(0, 0.0003, n)) - spread / 2,
        "bid_low": price - np.abs(np.random.normal(0, 0.0003, n)) - spread / 2,
        "bid_close": price - spread / 2,
        "ask_close": price + spread / 2,
        "spread": spread,
        "spread_pips": spread / 0.0001,
        "bar_range": np.abs(np.random.normal(0.0006, 0.0002, n)),
    }, index=dates)
    return df


class TestTimeFeatures:
    def test_hour_feature(self, sample_forex_df):
        df = add_time_features(sample_forex_df)
        assert "hour" in df.columns
        assert df["hour"].between(0, 23).all()

    def test_cyclical_encoding(self, sample_forex_df):
        df = add_time_features(sample_forex_df)
        assert "hour_sin" in df.columns
        assert "hour_cos" in df.columns
        # Cyclical values should be in [-1, 1]
        assert df["hour_sin"].between(-1.01, 1.01).all()

    def test_session_flags(self, sample_forex_df):
        df = add_time_features(sample_forex_df)
        assert "is_london_session" in df.columns
        assert "is_new_york_session" in df.columns
        assert "is_asian_session" in df.columns
        # Flags should be 0 or 1
        assert df["is_london_session"].isin([0, 1]).all()

    def test_no_lookahead(self, sample_forex_df):
        """Time features should depend only on the timestamp, not future data."""
        df = add_time_features(sample_forex_df)
        assert "hour" in df.columns
        # Hour at same index should be same regardless of data after
        df_short = add_time_features(sample_forex_df.head(100))
        pd.testing.assert_series_equal(df["hour"].head(100), df_short["hour"])


class TestPriceFeatures:
    def test_returns(self, sample_forex_df):
        df = add_price_features(sample_forex_df)
        assert "return_1h" in df.columns
        assert "log_return_1h" in df.columns
        # First return should be NaN
        assert pd.isna(df["return_1h"].iloc[0])

    def test_rolling_calculations(self, sample_forex_df):
        df = add_price_features(sample_forex_df)
        for window in [5, 10, 20, 50]:
            assert f"roll_mean_{window}" in df.columns
            assert f"roll_std_{window}" in df.columns
        # Implementation uses min_periods=window//2 (i.e. 10 for window=20),
        # so first min_periods-1 = 9 rows should be NaN, not window-1 = 19 rows.
        min_periods = 20 // 2  # 10
        assert df["roll_mean_20"].iloc[:min_periods - 1].isna().all(), (
            f"Expected first {min_periods - 1} rows to be NaN"
        )

    def test_no_future_leakage_in_rolling(self, sample_forex_df):
        """Rolling features on full dataset should match those on truncated dataset."""
        df_full = add_price_features(sample_forex_df)
        df_half = add_price_features(sample_forex_df.head(200))
        # The 100th row's rolling mean (with window 20) should be identical
        val_full = df_full["roll_mean_20"].iloc[199]
        val_half = df_half["roll_mean_20"].iloc[199]
        assert abs(val_full - val_half) < 1e-10, "Rolling features have look-ahead bias!"


class TestTechnicalIndicators:
    def test_rsi_bounds(self, sample_forex_df):
        df = add_technical_indicators(sample_forex_df)
        assert "rsi_14" in df.columns
        rsi_valid = df["rsi_14"].dropna()
        assert (rsi_valid >= 0).all() and (rsi_valid <= 100).all(), "RSI must be in [0, 100]"

    def test_bollinger_bands(self, sample_forex_df):
        df = add_technical_indicators(sample_forex_df)
        assert "bb_upper" in df.columns
        assert "bb_lower" in df.columns
        # Upper must always >= lower
        valid = df.dropna(subset=["bb_upper", "bb_lower"])
        assert (valid["bb_upper"] >= valid["bb_lower"]).all()

    def test_macd(self, sample_forex_df):
        df = add_technical_indicators(sample_forex_df)
        assert "macd" in df.columns
        assert "macd_signal" in df.columns
        assert "macd_histogram" in df.columns
        # Histogram = MACD - signal
        valid = df.dropna(subset=["macd", "macd_signal", "macd_histogram"])
        diff = (valid["macd"] - valid["macd_signal"] - valid["macd_histogram"]).abs()
        assert (diff < 1e-10).all()

    def test_atr_positive(self, sample_forex_df):
        df = add_technical_indicators(sample_forex_df)
        assert "atr_14" in df.columns
        valid = df["atr_14"].dropna()
        assert (valid >= 0).all(), "ATR must be non-negative"
