"""
QuantTrade ML Pipeline — Base Strategy & Trade Dataclass
Defines the contract that all trading strategies must implement.
"""
from __future__ import annotations

import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import settings


# ------------------------------------------------------------------ #
# Trade Dataclass
# ------------------------------------------------------------------ #
@dataclass
class Trade:
    """Represents a single completed trade."""
    strategy_id: str
    direction: int                  # +1 = long, -1 = short
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price: float
    exit_price: float
    quantity: float                 # Units in base currency
    position_size_usd: float        # Notional value in USD
    stop_loss: float
    take_profit: float
    exit_reason: str                # "tp", "sl", "signal", "timeout"
    holding_bars: int               # Number of 1h bars held
    pnl: float                      # In price points
    pnl_usd: float                  # In USD
    pnl_pct: float                  # As % of position
    win: bool
    strategy_params: dict[str, Any] = field(default_factory=dict)

    @property
    def holding_hours(self) -> float:
        return self.holding_bars

    @property
    def return_pct(self) -> float:
        return self.pnl_pct

    def to_dict(self) -> dict[str, Any]:
        d = {
            "strategy_id": self.strategy_id,
            "direction": self.direction,
            "entry_time": self.entry_time,
            "exit_time": self.exit_time,
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "quantity": self.quantity,
            "position_size_usd": self.position_size_usd,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "exit_reason": self.exit_reason,
            "holding_bars": self.holding_bars,
            "pnl": self.pnl,
            "pnl_usd": self.pnl_usd,
            "pnl_pct": self.pnl_pct,
            "win": self.win,
        }
        # Flatten strategy params
        for k, v in self.strategy_params.items():
            d[f"param_{k}"] = v
        return d


# ------------------------------------------------------------------ #
# Abstract Base Strategy
# ------------------------------------------------------------------ #
class BaseStrategy(ABC):
    """
    Abstract base class for all trading strategies.

    Each strategy must implement:
    - generate_signals(df) -> pd.Series of {-1, 0, +1}
    - get_parameters() -> dict of strategy configuration

    The base class handles:
    - Position sizing (ATR-based with risk_per_trade % risk)
    - Stop loss and take profit calculation
    - Trade execution simulation
    - Spread and slippage costs
    """

    STRATEGY_ID: str = "base"
    MAX_HOLDING_BARS: int = 100    # Force-exit after N bars
    MIN_ATR_MULTIPLIER: float = 1.0
    MAX_ATR_MULTIPLIER: float = 4.0

    def __init__(
        self,
        initial_capital: float | None = None,
        risk_per_trade: float | None = None,
        atr_stop_mult: float = 1.5,
        atr_tp_mult: float = 3.0,
        max_holding_bars: int | None = None,
    ) -> None:
        self.initial_capital = initial_capital or settings.initial_capital
        self.risk_per_trade = risk_per_trade or settings.risk_per_trade
        self.atr_stop_mult = atr_stop_mult
        self.atr_tp_mult = atr_tp_mult
        self.max_holding_bars = max_holding_bars or self.MAX_HOLDING_BARS
        self.spread_cost = settings.max_spread_pips * settings.pip_value
        self.slippage = settings.slippage_pips * settings.pip_value

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.Series:
        """
        Generate entry signals for each bar.

        Returns:
            pd.Series with values: +1 (long), -1 (short), 0 (no signal)
        """
        ...

    @abstractmethod
    def get_parameters(self) -> dict[str, Any]:
        """Return strategy parameters as a dict."""
        ...

    def run(self, df: pd.DataFrame) -> list[Trade]:
        """
        Execute the strategy on the full dataset.

        Args:
            df: Feature-enriched DataFrame with ATR column

        Returns:
            List of completed Trade objects
        """
        logger.debug("Running strategy: {}", self.STRATEGY_ID)

        signals = self.generate_signals(df)
        trades = self._execute_trades(df, signals)

        logger.debug(
            "Strategy {} | trades={} | win_rate={:.1%}",
            self.STRATEGY_ID,
            len(trades),
            sum(t.win for t in trades) / max(len(trades), 1),
        )
        return trades

    # ------------------------------------------------------------------ #
    # Trade Execution
    # ------------------------------------------------------------------ #

    def _execute_trades(self, df: pd.DataFrame, signals: pd.Series) -> list[Trade]:
        """Simulate trade execution bar by bar."""
        trades = []
        in_trade = False
        entry_idx = None
        entry_price = None
        direction = None
        stop_loss = None
        take_profit = None
        quantity = None
        position_size_usd = None

        close = df["mid_close"]
        high = df.get("mid_high", close)
        low = df.get("mid_low", close)
        atr = df.get("atr_14", pd.Series(0.001, index=df.index))

        prices = close.values
        highs = high.values
        lows = low.values
        atrs = atr.fillna(0.001).values
        sig_values = signals.values
        idx = df.index

        for i in range(1, len(df)):
            current_price = prices[i]
            current_high = highs[i]
            current_low = lows[i]

            if not in_trade:
                # Check for entry signal
                sig = sig_values[i]
                if sig != 0:
                    atr_val = atrs[i]
                    effective_entry = current_price + (sig * (self.slippage + self.spread_cost / 2))
                    sl = effective_entry - sig * self.atr_stop_mult * atr_val
                    tp = effective_entry + sig * self.atr_tp_mult * atr_val

                    # Position sizing: risk_per_trade % of capital per pip risk
                    pip_risk = abs(effective_entry - sl) / settings.pip_value
                    if pip_risk > 0:
                        qty = (self.initial_capital * self.risk_per_trade) / (pip_risk * settings.pip_value)
                        qty = min(qty, 1_000_000)  # Cap at 1M units
                    else:
                        qty = 10_000  # Fallback: 1 mini lot

                    in_trade = True
                    entry_idx = i
                    entry_price = effective_entry
                    direction = int(sig)
                    stop_loss = sl
                    take_profit = tp
                    quantity = qty
                    position_size_usd = qty * effective_entry

            else:
                # Check for exit conditions
                bars_in_trade = i - entry_idx
                exit_reason = None
                exit_price = current_price

                if direction == 1:
                    if current_low <= stop_loss:
                        exit_reason = "sl"
                        exit_price = stop_loss
                    elif current_high >= take_profit:
                        exit_reason = "tp"
                        exit_price = take_profit
                    elif sig_values[i] == -1:
                        exit_reason = "signal"
                elif direction == -1:
                    if current_high >= stop_loss:
                        exit_reason = "sl"
                        exit_price = stop_loss
                    elif current_low <= take_profit:
                        exit_reason = "tp"
                        exit_price = take_profit
                    elif sig_values[i] == 1:
                        exit_reason = "signal"

                if bars_in_trade >= self.max_holding_bars:
                    exit_reason = "timeout"

                if exit_reason:
                    # Calculate PnL
                    exit_effective = exit_price - direction * (self.slippage + self.spread_cost / 2)
                    raw_pnl = direction * (exit_effective - entry_price)
                    pnl_usd = raw_pnl * quantity
                    pnl_pct = raw_pnl / entry_price

                    trades.append(Trade(
                        strategy_id=self.STRATEGY_ID,
                        direction=direction,
                        entry_time=idx[entry_idx],
                        exit_time=idx[i],
                        entry_price=entry_price,
                        exit_price=exit_effective,
                        quantity=quantity,
                        position_size_usd=position_size_usd,
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                        exit_reason=exit_reason,
                        holding_bars=bars_in_trade,
                        pnl=raw_pnl,
                        pnl_usd=pnl_usd,
                        pnl_pct=pnl_pct,
                        win=raw_pnl > 0,
                        strategy_params=self.get_parameters(),
                    ))

                    in_trade = False
                    entry_idx = None

        return trades

    def _compute_position_size(
        self,
        price: float,
        atr: float,
        capital: float,
    ) -> float:
        """Compute position size using ATR-based risk management."""
        pip_risk = (self.atr_stop_mult * atr) / settings.pip_value
        if pip_risk <= 0:
            return 10_000
        size = (capital * self.risk_per_trade) / (pip_risk * settings.pip_value)
        return max(min(size, 1_000_000), 1_000)
