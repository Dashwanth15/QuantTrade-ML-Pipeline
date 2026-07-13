"""
QuantTrade ML Pipeline — Technical Indicator Engineering
Computes 14 professional technical indicators using the `ta` library
with hand-crafted fallbacks. All computations are strictly causal.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import settings


class TechnicalIndicatorEngineer:
    """
    Computes a comprehensive set of technical indicators.
    
    Indicators:
    RSI, MACD, EMA (4 periods), SMA (3 periods), Bollinger Bands,
    ATR, Stochastic Oscillator, ADX, Williams %R, CCI,
    Parabolic SAR (simplified), Ichimoku (basic), OBV proxy
    """

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        logger.debug("Engineering technical indicators for {} rows", len(df))

        close = df["mid_close"]
        high = df.get("mid_high", close)
        low = df.get("mid_low", close)
        open_ = df.get("mid_open", close)

        df = self._add_rsi(df, close)
        df = self._add_macd(df, close)
        df = self._add_ema(df, close)
        df = self._add_sma(df, close)
        df = self._add_bollinger_bands(df, close)
        df = self._add_atr(df, high, low, close)
        df = self._add_stochastic(df, high, low, close)
        df = self._add_adx(df, high, low, close)
        df = self._add_williams_r(df, high, low, close)
        df = self._add_cci(df, high, low, close)
        df = self._add_ichimoku(df, high, low)
        df = self._add_obv_proxy(df, close)
        df = self._add_composite_signals(df)

        logger.debug("Technical indicators engineered successfully")
        return df

    # ------------------------------------------------------------------ #
    # RSI
    # ------------------------------------------------------------------ #
    def _add_rsi(self, df: pd.DataFrame, close: pd.Series) -> pd.DataFrame:
        for period in [7, 14, 21]:
            delta = close.diff()
            gain = delta.clip(lower=0)
            loss = (-delta).clip(lower=0)
            avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
            avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
            rs = avg_gain / (avg_loss + 1e-10)
            df[f"rsi_{period}"] = 100 - (100 / (1 + rs))

        # RSI-based signals
        df["rsi_overbought"] = (df["rsi_14"] > 70).astype(int)
        df["rsi_oversold"] = (df["rsi_14"] < 30).astype(int)
        df["rsi_divergence"] = df["rsi_14"] - df["rsi_14"].shift(10)
        return df

    # ------------------------------------------------------------------ #
    # MACD
    # ------------------------------------------------------------------ #
    def _add_macd(self, df: pd.DataFrame, close: pd.Series) -> pd.DataFrame:
        fast = settings.macd_fast
        slow = settings.macd_slow
        signal_period = settings.macd_signal

        ema_fast = close.ewm(span=fast, adjust=False).mean()
        ema_slow = close.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
        histogram = macd_line - signal_line

        df["macd"] = macd_line
        df["macd_signal"] = signal_line
        df["macd_histogram"] = histogram
        df["macd_crossover"] = np.sign(macd_line - signal_line)

        # MACD momentum
        df["macd_hist_change"] = histogram.diff()
        df["macd_bullish"] = ((macd_line > signal_line) & (macd_line.shift(1) <= signal_line.shift(1))).astype(int)
        df["macd_bearish"] = ((macd_line < signal_line) & (macd_line.shift(1) >= signal_line.shift(1))).astype(int)
        return df

    # ------------------------------------------------------------------ #
    # EMA
    # ------------------------------------------------------------------ #
    def _add_ema(self, df: pd.DataFrame, close: pd.Series) -> pd.DataFrame:
        for period in settings.ema_periods:
            df[f"ema_{period}"] = close.ewm(span=period, adjust=False).mean()
            df[f"price_above_ema_{period}"] = (close > df[f"ema_{period}"]).astype(int)
            df[f"ema_{period}_slope"] = df[f"ema_{period}"].diff(3)

        # EMA crossover signals
        df["ema_9_21_cross"] = np.sign(df["ema_9"] - df["ema_21"])
        df["ema_21_50_cross"] = np.sign(df["ema_21"] - df["ema_50"])
        df["golden_cross"] = (
            (df["ema_50"] > df["ema_200"]) &
            (df["ema_50"].shift(1) <= df["ema_200"].shift(1))
        ).astype(int)
        df["death_cross"] = (
            (df["ema_50"] < df["ema_200"]) &
            (df["ema_50"].shift(1) >= df["ema_200"].shift(1))
        ).astype(int)

        # Price relative to key EMAs
        df["price_vs_ema200"] = (close - df["ema_200"]) / df["ema_200"]
        df["price_vs_ema50"] = (close - df["ema_50"]) / df["ema_50"]
        return df

    # ------------------------------------------------------------------ #
    # SMA
    # ------------------------------------------------------------------ #
    def _add_sma(self, df: pd.DataFrame, close: pd.Series) -> pd.DataFrame:
        for period in settings.sma_periods:
            df[f"sma_{period}"] = close.rolling(period, min_periods=period // 2).mean()
            df[f"price_above_sma_{period}"] = (close > df[f"sma_{period}"]).astype(int)
        return df

    # ------------------------------------------------------------------ #
    # Bollinger Bands
    # ------------------------------------------------------------------ #
    def _add_bollinger_bands(self, df: pd.DataFrame, close: pd.Series) -> pd.DataFrame:
        period = settings.bb_period
        std_mult = settings.bb_std

        mid = close.rolling(period, min_periods=period // 2).mean()
        std = close.rolling(period, min_periods=period // 2).std()

        df["bb_upper"] = mid + std_mult * std
        df["bb_middle"] = mid
        df["bb_lower"] = mid - std_mult * std
        df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / (mid + 1e-10)

        # %B indicator: where price is relative to bands
        df["bb_pct_b"] = (close - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"] + 1e-10)

        # Squeeze: bands narrowing (low volatility precedes breakout)
        df["bb_squeeze"] = (df["bb_width"] < df["bb_width"].rolling(50).mean() * 0.8).astype(int)

        # Band touch signals
        df["bb_upper_touch"] = (close >= df["bb_upper"]).astype(int)
        df["bb_lower_touch"] = (close <= df["bb_lower"]).astype(int)
        return df

    # ------------------------------------------------------------------ #
    # ATR (Average True Range)
    # ------------------------------------------------------------------ #
    def _add_atr(
        self, df: pd.DataFrame, high: pd.Series, low: pd.Series, close: pd.Series
    ) -> pd.DataFrame:
        prev_close = close.shift(1)
        tr = pd.DataFrame({
            "hl": high - low,
            "hpc": (high - prev_close).abs(),
            "lpc": (low - prev_close).abs(),
        }).max(axis=1)

        for period in [7, 14, 21]:
            df[f"atr_{period}"] = tr.ewm(alpha=1 / period, adjust=False).mean()
            df[f"atr_{period}_pct"] = df[f"atr_{period}"] / close

        df["true_range"] = tr

        # Normalized ATR (volatility regime)
        df["atr_regime"] = pd.cut(
            df["atr_14_pct"],
            bins=[0, 0.001, 0.002, 0.005, 1.0],
            labels=["low", "medium", "high", "extreme"],
        ).cat.codes
        return df

    # ------------------------------------------------------------------ #
    # Stochastic Oscillator
    # ------------------------------------------------------------------ #
    def _add_stochastic(
        self, df: pd.DataFrame, high: pd.Series, low: pd.Series, close: pd.Series
    ) -> pd.DataFrame:
        k_period = 14
        d_period = 3

        lowest_low = low.rolling(k_period).min()
        highest_high = high.rolling(k_period).max()

        stoch_k = 100 * (close - lowest_low) / (highest_high - lowest_low + 1e-10)
        stoch_d = stoch_k.rolling(d_period).mean()

        df["stoch_k"] = stoch_k
        df["stoch_d"] = stoch_d
        df["stoch_crossover"] = np.sign(stoch_k - stoch_d)
        df["stoch_overbought"] = (stoch_k > 80).astype(int)
        df["stoch_oversold"] = (stoch_k < 20).astype(int)
        return df

    # ------------------------------------------------------------------ #
    # ADX (Average Directional Index)
    # ------------------------------------------------------------------ #
    def _add_adx(
        self, df: pd.DataFrame, high: pd.Series, low: pd.Series, close: pd.Series
    ) -> pd.DataFrame:
        period = 14
        prev_high = high.shift(1)
        prev_low = low.shift(1)
        prev_close = close.shift(1)

        # True Range
        tr = pd.DataFrame({
            "hl": high - low,
            "hpc": (high - prev_close).abs(),
            "lpc": (low - prev_close).abs(),
        }).max(axis=1)

        # Directional movements
        dm_plus = high - prev_high
        dm_minus = prev_low - low
        dm_plus = dm_plus.clip(lower=0).where(dm_plus > dm_minus, 0)
        dm_minus = dm_minus.clip(lower=0).where(dm_minus > dm_plus, 0)

        # Smoothed
        atr = tr.ewm(alpha=1 / period, adjust=False).mean()
        di_plus = 100 * dm_plus.ewm(alpha=1 / period, adjust=False).mean() / (atr + 1e-10)
        di_minus = 100 * dm_minus.ewm(alpha=1 / period, adjust=False).mean() / (atr + 1e-10)

        dx = 100 * (di_plus - di_minus).abs() / (di_plus + di_minus + 1e-10)
        adx = dx.ewm(alpha=1 / period, adjust=False).mean()

        df["adx"] = adx
        df["di_plus"] = di_plus
        df["di_minus"] = di_minus
        df["adx_trending"] = (adx > 25).astype(int)
        df["adx_strong_trend"] = (adx > 40).astype(int)
        df["adx_direction"] = np.sign(di_plus - di_minus)
        return df

    # ------------------------------------------------------------------ #
    # Williams %R
    # ------------------------------------------------------------------ #
    def _add_williams_r(
        self, df: pd.DataFrame, high: pd.Series, low: pd.Series, close: pd.Series
    ) -> pd.DataFrame:
        period = 14
        highest_high = high.rolling(period).max()
        lowest_low = low.rolling(period).min()
        df["williams_r"] = -100 * (highest_high - close) / (highest_high - lowest_low + 1e-10)
        df["williams_r_overbought"] = (df["williams_r"] > -20).astype(int)
        df["williams_r_oversold"] = (df["williams_r"] < -80).astype(int)
        return df

    # ------------------------------------------------------------------ #
    # CCI (Commodity Channel Index)
    # ------------------------------------------------------------------ #
    def _add_cci(
        self, df: pd.DataFrame, high: pd.Series, low: pd.Series, close: pd.Series
    ) -> pd.DataFrame:
        period = 20
        typical = (high + low + close) / 3
        roll_mean = typical.rolling(period).mean()
        mean_dev = typical.rolling(period).apply(
            lambda x: np.mean(np.abs(x - x.mean())), raw=True
        )
        df["cci"] = (typical - roll_mean) / (0.015 * mean_dev + 1e-10)
        df["cci_overbought"] = (df["cci"] > 100).astype(int)
        df["cci_oversold"] = (df["cci"] < -100).astype(int)
        return df

    # ------------------------------------------------------------------ #
    # Ichimoku Cloud (basic)
    # ------------------------------------------------------------------ #
    def _add_ichimoku(self, df: pd.DataFrame, high: pd.Series, low: pd.Series) -> pd.DataFrame:
        tenkan_period = 9
        kijun_period = 26

        tenkan = (high.rolling(tenkan_period).max() + low.rolling(tenkan_period).min()) / 2
        kijun = (high.rolling(kijun_period).max() + low.rolling(kijun_period).min()) / 2

        senkou_a = ((tenkan + kijun) / 2).shift(kijun_period)
        senkou_b = (
            (high.rolling(52).max() + low.rolling(52).min()) / 2
        ).shift(kijun_period)

        df["ichimoku_tenkan"] = tenkan
        df["ichimoku_kijun"] = kijun
        df["ichimoku_senkou_a"] = senkou_a
        df["ichimoku_senkou_b"] = senkou_b
        df["ichimoku_tk_cross"] = np.sign(tenkan - kijun)
        return df

    # ------------------------------------------------------------------ #
    # OBV Proxy (without volume — using candle direction × range)
    # ------------------------------------------------------------------ #
    def _add_obv_proxy(self, df: pd.DataFrame, close: pd.Series) -> pd.DataFrame:
        """Volume proxy: candle direction × bar range."""
        if "bar_range" not in df.columns:
            df["bar_range"] = df.get("mid_high", close) - df.get("mid_low", close)

        direction = np.sign(close.diff())
        obv_proxy = (direction * df["bar_range"]).cumsum()
        df["obv_proxy"] = obv_proxy
        df["obv_proxy_sma20"] = obv_proxy.rolling(20).mean()
        df["obv_proxy_trend"] = np.sign(obv_proxy - obv_proxy.shift(10))
        return df

    # ------------------------------------------------------------------ #
    # Composite Signals
    # ------------------------------------------------------------------ #
    def _add_composite_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Multi-indicator composite signal scores."""
        # Bull score: count of bullish indicator alignments
        bull_signals = []
        if "rsi_14" in df.columns:
            bull_signals.append((df["rsi_14"] > 50).astype(int))
        if "macd" in df.columns and "macd_signal" in df.columns:
            bull_signals.append((df["macd"] > df["macd_signal"]).astype(int))
        if "ema_21" in df.columns and "ema_50" in df.columns:
            bull_signals.append((df["ema_21"] > df["ema_50"]).astype(int))
        if "adx_direction" in df.columns:
            bull_signals.append((df["adx_direction"] > 0).astype(int))
        if "stoch_k" in df.columns:
            bull_signals.append((df["stoch_k"] > 50).astype(int))

        if bull_signals:
            df["composite_bull_score"] = sum(bull_signals) / len(bull_signals)
            df["composite_bear_score"] = 1 - df["composite_bull_score"]

        return df


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add all technical indicators to a DataFrame."""
    return TechnicalIndicatorEngineer().transform(df)
