"""
QuantTrade Simulation — Technical Trading Strategies
Defines the rule-based quantitative strategies used to generate trading signals
from historical EUR/USD forex data.

Module: src.simulation.strategies
Purpose: Houses all quantitative strategy class implementations.
Responsibilities:
  - Generate long (+1), short (-1), or flat (0) signals based on technical indicators.
  - Expose parameters for hyperparameter optimization and strategy analysis.
Dependencies: numpy, pandas, src.simulation.base_strategy.BaseStrategy
Inputs: pd.DataFrame containing historical forex candles and engineered features.
Outputs: pd.Series of trading signals (-1, 0, 1) indexed by time.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.simulation.base_strategy import BaseStrategy


class MACrossoverStrategy(BaseStrategy):
    """
    Classic dual moving average crossover strategy.
    
    Purpose:
      Generates signals based on the crossover of fast and slow exponential moving averages.
    Usage:
      strategy = MACrossoverStrategy(fast_period=20, slow_period=50)
      signals = strategy.generate_signals(df)
    Parameters:
      fast_period (int): Period for the fast EMA (default: 20).
      slow_period (int): Period for the slow EMA (default: 50).
    Attributes:
      fast_period (int): Period for the fast EMA.
      slow_period (int): Period for the slow EMA.
    """

    STRATEGY_ID = "ma_crossover"

    def __init__(self, fast_period: int = 20, slow_period: int = 50, **kwargs) -> None:
        super().__init__(**kwargs)
        self.fast_period = fast_period
        self.slow_period = slow_period

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        Generates buy/sell signals based on golden and death crosses of EMAs.
        
        Parameters:
          df (pd.DataFrame): Historical price and features dataset.
        Return value:
          pd.Series: Signal values (-1, 0, 1) matching the dataframe index.
        """
        signals = pd.Series(0, index=df.index)
        close = df["mid_close"]

        fast_col = f"ema_{self.fast_period}"
        slow_col = f"ema_{self.slow_period}"
        fast = df.get(fast_col, close.ewm(span=self.fast_period, adjust=False).mean())
        slow = df.get(slow_col, close.ewm(span=self.slow_period, adjust=False).mean())

        # Crossover detection
        fast_above = fast > slow
        prev_fast_above = fast_above.shift(1)

        # Golden cross: fast crosses above slow
        golden_cross = fast_above & ~prev_fast_above.fillna(False)
        # Death cross: fast crosses below slow
        death_cross = ~fast_above & prev_fast_above.fillna(False)

        signals[golden_cross] = 1
        signals[death_cross] = -1

        # Activity filter: only trade when bar range is above average
        bar_range = df.get("bar_range", (df.get("mid_high", close) - df.get("mid_low", close)))
        avg_range = bar_range.rolling(20).mean()
        active = bar_range >= avg_range * 0.5
        signals = signals.where(active, 0)

        return signals

    def get_parameters(self) -> dict:
        return {"fast_period": self.fast_period, "slow_period": self.slow_period}


class MomentumStrategy(BaseStrategy):
    """
    Momentum strategy using Rate-of-Change + EMA trend filter + ADX strength filter.
    
    Purpose:
      Generates trend-following signals during high-strength directional regimes.
    Usage:
      strategy = MomentumStrategy(roc_period=10, roc_threshold=0.002)
    Parameters:
      roc_period (int): Lookback period for Rate of Change (default: 10).
      roc_threshold (float): Threshold to trigger momentum entry (default: 0.002).
      ema_period (int): Period for trend filter EMA (default: 50).
      adx_threshold (float): Minimum ADX to confirm trade (default: 20.0).
    """

    STRATEGY_ID = "momentum"

    def __init__(self, roc_period: int = 10, roc_threshold: float = 0.002,
                 ema_period: int = 50, adx_threshold: float = 20.0, **kwargs) -> None:
        super().__init__(**kwargs)
        self.roc_period = roc_period
        self.roc_threshold = roc_threshold
        self.ema_period = ema_period
        self.adx_threshold = adx_threshold

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        Generates buy/sell signals when Rate-of-Change exceeds threshold and filters align.
        
        Parameters:
          df (pd.DataFrame): Input market features.
        Return value:
          pd.Series: Trading signals.
        """
        signals = pd.Series(0, index=df.index)
        close = df["mid_close"]
        roc_col = f"roc_{self.roc_period}"
        ema_col = f"ema_{self.ema_period}"
        roc = df.get(roc_col, close.pct_change(self.roc_period))
        ema = df.get(ema_col, close.ewm(span=self.ema_period, adjust=False).mean())
        adx = df.get("adx", pd.Series(30, index=df.index))

        # Momentum conditions
        long_cond = (roc > self.roc_threshold) & (close > ema) & (adx > self.adx_threshold)
        short_cond = (roc < -self.roc_threshold) & (close < ema) & (adx > self.adx_threshold)

        signals[long_cond] = 1
        signals[short_cond] = -1

        # Avoid signal spam — only trigger on transitions
        signals = signals.where(signals != signals.shift(1), 0)
        return signals

    def get_parameters(self) -> dict:
        return {"roc_period": self.roc_period, "roc_threshold": self.roc_threshold,
                "ema_period": self.ema_period, "adx_threshold": self.adx_threshold}


class TrendFollowingStrategy(BaseStrategy):
    """
    Multi-confirmation trend following strategy.
    
    Purpose:
      Captures long-term trends by trading bounces off short EMAs in direction of EMA200.
    Usage:
      strategy = TrendFollowingStrategy()
    """

    STRATEGY_ID = "trend_following"

    def __init__(self, trend_ema: int = 200, pullback_ema: int = 21,
                 adx_threshold: float = 25.0, pullback_pct: float = 0.002, **kwargs) -> None:
        super().__init__(**kwargs)
        self.trend_ema = trend_ema
        self.pullback_ema = pullback_ema
        self.adx_threshold = adx_threshold
        self.pullback_pct = pullback_pct

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """Generates pull-back entry signals in strong trends."""
        signals = pd.Series(0, index=df.index)
        close = df["mid_close"]
        low = df.get("mid_low", close)
        high = df.get("mid_high", close)

        ema_trend = df.get(f"ema_{self.trend_ema}", close.ewm(span=self.trend_ema, adjust=False).mean())
        ema_pb = df.get(f"ema_{self.pullback_ema}", close.ewm(span=self.pullback_ema, adjust=False).mean())
        adx = df.get("adx", pd.Series(30, index=df.index))
        adx_dir = df.get("adx_direction", pd.Series(1, index=df.index))

        trending = adx > self.adx_threshold
        uptrend = trending & (close > ema_trend) & (adx_dir > 0)
        downtrend = trending & (close < ema_trend) & (adx_dir < 0)

        # Pullback to EMA21 and bounce
        near_ema_long = (low <= ema_pb * (1 + self.pullback_pct)) & (close > ema_pb)
        near_ema_short = (high >= ema_pb * (1 - self.pullback_pct)) & (close < ema_pb)

        signals[uptrend & near_ema_long] = 1
        signals[downtrend & near_ema_short] = -1

        signals = signals.where(signals != signals.shift(1), 0)
        return signals

    def get_parameters(self) -> dict:
        return {"trend_ema": self.trend_ema, "pullback_ema": self.pullback_ema,
                "adx_threshold": self.adx_threshold, "pullback_pct": self.pullback_pct}


class BreakoutStrategy(BaseStrategy):
    """
    N-period high/low breakout strategy.
    
    Purpose:
      Triggers positions when price breaches rolling price extremes.
    Usage:
      strategy = BreakoutStrategy(lookback=20)
    """

    STRATEGY_ID = "breakout"

    def __init__(self, lookback: int = 20, cooldown_bars: int = 10,
                 range_mult: float = 1.2, **kwargs) -> None:
        super().__init__(**kwargs)
        self.lookback = lookback
        self.cooldown_bars = cooldown_bars
        self.range_mult = range_mult

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """Generates breakout signals with volume/ATR confirmation and signal cooldowns."""
        signals = pd.Series(0, index=df.index)
        close = df["mid_close"]
        high = df.get("mid_high", close)
        low = df.get("mid_low", close)

        period_high = close.rolling(self.lookback).max().shift(1)
        period_low = close.rolling(self.lookback).min().shift(1)

        bar_range = high - low
        avg_range = bar_range.rolling(20).mean()
        strong_bar = bar_range > avg_range * self.range_mult

        long_break = (close > period_high) & strong_bar
        short_break = (close < period_low) & strong_bar

        signals[long_break] = 1
        signals[short_break] = -1

        # Apply cooldown window
        raw = signals.copy()
        for i in range(1, len(signals)):
            if signals.iloc[i] != 0:
                start_idx = max(0, i - self.cooldown_bars)
                if raw.iloc[start_idx:i].abs().sum() > 0:
                    signals.iloc[i] = 0

        return signals

    def get_parameters(self) -> dict:
        return {"lookback": self.lookback, "cooldown_bars": self.cooldown_bars,
                "range_mult": self.range_mult}


class RSIMeanReversionStrategy(BaseStrategy):
    """
    RSI-based mean reversion strategy.
    
    Purpose:
      Trades reversals in range-bound markets.
    """

    STRATEGY_ID = "rsi_reversion"

    def __init__(self, rsi_period: int = 14, oversold: float = 30.0,
                 overbought: float = 70.0, adx_max: float = 25.0, **kwargs) -> None:
        super().__init__(**kwargs)
        self.rsi_period = rsi_period
        self.oversold = oversold
        self.overbought = overbought
        self.adx_max = adx_max

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """Generates signal bounds based on oversold/overbought thresholds."""
        signals = pd.Series(0, index=df.index)
        rsi_col = f"rsi_{self.rsi_period}"
        rsi = df.get(rsi_col, self._compute_rsi(df["mid_close"], self.rsi_period))
        adx = df.get("adx", pd.Series(15.0, index=df.index))
        macd_hist = df.get("macd_histogram", pd.Series(0.0, index=df.index))

        ranging = adx < self.adx_max
        long_cond = ranging & (rsi < self.oversold) & (rsi > rsi.shift(1)) & (macd_hist > macd_hist.shift(1))
        short_cond = ranging & (rsi > self.overbought) & (rsi < rsi.shift(1)) & (macd_hist < macd_hist.shift(1))

        signals[long_cond] = 1
        signals[short_cond] = -1

        # Debounce: skip duplicate signals in adjacent bars
        for i in range(1, 3):
            duplicate = signals == signals.shift(i)
            signals = signals.where(~duplicate | (signals == 0), 0)

        return signals

    @staticmethod
    def _compute_rsi(close: pd.Series, period: int) -> pd.Series:
        delta = close.diff()
        gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
        loss = (-delta).clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
        rs = gain / (loss + 1e-10)
        return 100 - (100 / (1 + rs))

    def get_parameters(self) -> dict:
        return {"rsi_period": self.rsi_period, "oversold": self.oversold,
                "overbought": self.overbought, "adx_max": self.adx_max}


class BollingerBandStrategy(BaseStrategy):
    """
    Bollinger Band mean-reversion + squeeze breakout strategy.
    
    Purpose:
      Trades bounces off outer bands or breakouts following low-volatility squeezes.
    """

    STRATEGY_ID = "bollinger"

    def __init__(self, bb_period: int = 20, bb_std: float = 2.0,
                 rsi_threshold: float = 35.0, use_squeeze: bool = True, **kwargs) -> None:
        super().__init__(**kwargs)
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.rsi_threshold = rsi_threshold
        self.use_squeeze = use_squeeze

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """Generates standard Bollinger Band reversion or squeeze breakout signals."""
        signals = pd.Series(0, index=df.index)
        close = df["mid_close"]

        bb_upper = df.get("bb_upper", self._compute_bb(close, self.bb_period, self.bb_std, "upper"))
        bb_lower = df.get("bb_lower", self._compute_bb(close, self.bb_period, self.bb_std, "lower"))
        bb_squeeze = df.get("bb_squeeze", pd.Series(False, index=df.index))
        rsi = df.get("rsi_14", pd.Series(50.0, index=df.index))

        long_cond = (close <= bb_lower) & (rsi < self.rsi_threshold + 5)
        short_cond = (close >= bb_upper) & (rsi > (100 - self.rsi_threshold) - 5)

        signals[long_cond] = 1
        signals[short_cond] = -1

        if self.use_squeeze:
            bb_squeeze_bool = bb_squeeze.fillna(False).astype(bool)
            prev_squeeze = bb_squeeze_bool.shift(1).fillna(False).astype(bool)
            squeeze_end = (~bb_squeeze_bool) & prev_squeeze
            bb_middle = df.get("bb_middle", close.rolling(self.bb_period).mean())
            squeeze_long = squeeze_end & (close > bb_middle)
            squeeze_short = squeeze_end & (close < bb_middle)
            signals[squeeze_long] = 1
            signals[squeeze_short] = -1

        signals = signals.where(signals != signals.shift(1), 0)
        return signals

    @staticmethod
    def _compute_bb(close: pd.Series, period: int, std_mult: float, band: str) -> pd.Series:
        mid = close.rolling(period).mean()
        std = close.rolling(period).std()
        return mid + std_mult * std if band == "upper" else mid - std_mult * std

    def get_parameters(self) -> dict:
        return {"bb_period": self.bb_period, "bb_std": self.bb_std,
                "rsi_threshold": self.rsi_threshold, "use_squeeze": self.use_squeeze}


class SupportResistanceStrategy(BaseStrategy):
    """
    Pivot-point based support/resistance strategy.
    
    Purpose:
      Trades candle reversals occurring near daily support/resistance levels.
    """

    STRATEGY_ID = "support_resistance"

    def __init__(self, proximity_pips: float = 5.0, confirmation_bars: int = 2, **kwargs) -> None:
        super().__init__(**kwargs)
        self.proximity_pips = proximity_pips
        self.confirmation_bars = confirmation_bars
        self.proximity = proximity_pips * 0.0001

    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """Generates bounces off S/R pivot levels using yesterday's ranges."""
        signals = pd.Series(0, index=df.index)
        close = df["mid_close"]
        high = df.get("mid_high", close)
        low = df.get("mid_low", close)

        daily_high = high.resample("D").max().reindex(df.index, method="ffill")
        daily_low = low.resample("D").min().reindex(df.index, method="ffill")
        daily_close = close.resample("D").last().reindex(df.index, method="ffill")

        prev_high = daily_high.shift(1, freq="D").reindex(df.index, method="ffill")
        prev_low = daily_low.shift(1, freq="D").reindex(df.index, method="ffill")
        prev_close = daily_close.shift(1, freq="D").reindex(df.index, method="ffill")

        pivot = (prev_high + prev_low + prev_close) / 3
        support1 = 2 * pivot - prev_high
        support2 = pivot - (prev_high - prev_low)
        resist1 = 2 * pivot - prev_low
        resist2 = pivot + (prev_high - prev_low)

        near_support = ((close - support1).abs() < self.proximity) | ((close - support2).abs() < self.proximity)
        near_resist = ((close - resist1).abs() < self.proximity) | ((close - resist2).abs() < self.proximity)

        candle_up = close > df.get("mid_open", close.shift(1))
        candle_down = close < df.get("mid_open", close.shift(1))

        signals[near_support & candle_up] = 1
        signals[near_resist & candle_down] = -1

        signals = signals.where(signals != signals.shift(1), 0)
        return signals

    def get_parameters(self) -> dict:
        return {"proximity_pips": self.proximity_pips, "confirmation_bars": self.confirmation_bars}
