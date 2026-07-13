"""
QuantTrade ML Pipeline — Risk Manager
Implements professional position sizing and risk controls.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import settings


class RiskManager:
    """
    Position sizing and risk control engine.
    
    Methods:
    - Kelly Criterion sizing
    - ATR-based fixed-fractional sizing
    - Correlation-adjusted exposure limits
    - Max drawdown circuit breaker
    """

    def __init__(
        self,
        initial_capital: float | None = None,
        max_risk_per_trade: float = 0.02,
        max_portfolio_risk: float = 0.06,
        max_drawdown_pct: float = 0.20,
        kelly_fraction: float = 0.25,
    ) -> None:
        self.capital = initial_capital or settings.initial_capital
        self.current_capital = self.capital
        self.max_risk_per_trade = max_risk_per_trade
        self.max_portfolio_risk = max_portfolio_risk
        self.max_drawdown_pct = max_drawdown_pct
        self.kelly_fraction = kelly_fraction  # Fractional Kelly
        self._peak_capital = self.capital
        self._circuit_broken = False

    def position_size_fixed_fractional(
        self,
        entry_price: float,
        stop_loss_price: float,
        win_rate: float | None = None,
    ) -> float:
        """
        ATR-based fixed-fractional position sizing.
        
        Returns quantity in base currency units.
        """
        if self._circuit_broken:
            return 0.0

        pip_risk = abs(entry_price - stop_loss_price) / settings.pip_value
        if pip_risk <= 0:
            return 10_000  # 1 mini lot fallback

        # Base: risk max_risk_per_trade % of capital
        risk_amount = self.current_capital * self.max_risk_per_trade
        pip_value_usd = settings.pip_value * 1.0  # For EUR/USD ≈ $1 per pip per 10k lots

        quantity = risk_amount / (pip_risk * pip_value_usd * 1e-4)
        quantity = max(min(quantity, 1_000_000), 1_000)  # Clamp to 1k–1M units

        return quantity

    def kelly_position_size(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        entry_price: float,
        stop_loss: float,
    ) -> float:
        """
        Fractional Kelly Criterion position sizing.
        f* = (bp - q) / b where b = win/loss ratio, p = win rate, q = 1-p
        """
        if avg_loss == 0 or win_rate <= 0:
            return self.position_size_fixed_fractional(entry_price, stop_loss)

        b = abs(avg_win / avg_loss)
        p = win_rate
        q = 1 - p
        kelly_f = (b * p - q) / b

        # Apply fractional Kelly (safer)
        kelly_f = max(0.0, kelly_f * self.kelly_fraction)
        kelly_f = min(kelly_f, self.max_risk_per_trade * 3)  # Cap at 3× base risk

        risk_amount = self.current_capital * kelly_f
        pip_risk = abs(entry_price - stop_loss) / settings.pip_value
        if pip_risk <= 0:
            return 10_000

        quantity = risk_amount / (pip_risk * settings.pip_value)
        return max(min(quantity, 1_000_000), 1_000)

    def update_capital(self, pnl_usd: float) -> None:
        """Update capital after a trade and check circuit breaker."""
        self.current_capital += pnl_usd
        self._peak_capital = max(self._peak_capital, self.current_capital)
        drawdown_pct = (self._peak_capital - self.current_capital) / self._peak_capital
        if drawdown_pct >= self.max_drawdown_pct:
            if not self._circuit_broken:
                logger.warning(
                    "Circuit breaker triggered! Drawdown={:.1%}", drawdown_pct
                )
                self._circuit_broken = True

    def reset(self) -> None:
        """Reset to initial state."""
        self.current_capital = self.capital
        self._peak_capital = self.capital
        self._circuit_broken = False

    @property
    def is_active(self) -> bool:
        return not self._circuit_broken

    @property
    def current_drawdown(self) -> float:
        return (self._peak_capital - self.current_capital) / self._peak_capital
