"""
QuantTrade ML Pipeline — Strategy Tests
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.features.price_features import add_price_features
from src.features.technical import add_technical_indicators
from src.simulation.strategies import MomentumStrategy, RSIMeanReversionStrategy, BollingerBandStrategy


@pytest.fixture
def feature_df():
    """Minimal feature-enriched DataFrame."""
    n = 2000
    np.random.seed(42)
    dates = pd.date_range("2010-01-01", periods=n, freq="h", tz="UTC")
    price = 1.2 + np.cumsum(np.random.normal(0, 0.0003, n))
    df = pd.DataFrame({
        "mid_open": price + np.random.normal(0, 0.0001, n),
        "mid_high": price + np.abs(np.random.normal(0, 0.0002, n)),
        "mid_low": price - np.abs(np.random.normal(0, 0.0002, n)),
        "mid_close": price,
        "ask_close": price + 0.0002,
        "bid_close": price,
        "spread": 0.0002,
        "spread_pips": 2.0,
        "bar_range": np.abs(np.random.normal(0.0004, 0.0001, n)),
    }, index=dates)
    df = add_price_features(df)
    df = add_technical_indicators(df)
    return df


class TestStrategies:
    def test_momentum_signals_valid(self, feature_df):
        strategy = MomentumStrategy()
        signals = strategy.generate_signals(feature_df)
        assert set(signals.unique()).issubset({-1, 0, 1}), "Signals must be -1, 0, or 1"
        assert signals.index.equals(feature_df.index)

    def test_rsi_reversion_signals(self, feature_df):
        strategy = RSIMeanReversionStrategy()
        signals = strategy.generate_signals(feature_df)
        assert set(signals.unique()).issubset({-1, 0, 1})

    def test_bollinger_signals(self, feature_df):
        strategy = BollingerBandStrategy()
        signals = strategy.generate_signals(feature_df)
        assert set(signals.unique()).issubset({-1, 0, 1})

    def test_strategy_generates_trades(self, feature_df):
        strategy = MomentumStrategy()
        trades = strategy.run(feature_df)
        assert isinstance(trades, list)
        if len(trades) > 0:
            trade = trades[0]
            assert hasattr(trade, "pnl")
            assert hasattr(trade, "win")
            assert hasattr(trade, "entry_price")
            assert hasattr(trade, "exit_price")

    def test_pnl_calculation(self, feature_df):
        strategy = RSIMeanReversionStrategy()
        trades = strategy.run(feature_df)
        if len(trades) > 0:
            for trade in trades:
                # PnL should be positive for wins, negative for losses
                if trade.win:
                    assert trade.pnl > 0, f"Win trade has negative PnL: {trade.pnl}"
                else:
                    assert trade.pnl <= 0, f"Loss trade has positive PnL: {trade.pnl}"

    def test_trades_chronological(self, feature_df):
        strategy = MomentumStrategy()
        trades = strategy.run(feature_df)
        if len(trades) > 1:
            for i in range(1, len(trades)):
                assert trades[i].entry_time >= trades[i-1].entry_time, (
                    "Trades are not in chronological order!"
                )
