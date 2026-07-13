"""
QuantTrade ML Pipeline — Model Evaluator
Computes regression metrics and trading-specific performance metrics.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class ModelEvaluator:
    """
    Evaluates XGBoost model predictions using:
    - Regression metrics: MAE, RMSE, R², MAPE
    - Trading metrics: Sharpe, Sortino, Max Drawdown, Win Rate,
                       Profit Factor, Cumulative Return, Calmar
    """

    def evaluate(
        self,
        y_true: pd.Series | np.ndarray,
        y_pred: pd.Series | np.ndarray,
        timestamps: pd.Index | None = None,
    ) -> dict[str, float]:
        """
        Full evaluation suite.

        Args:
            y_true: Actual PnL values
            y_pred: Predicted PnL values
            timestamps: Optional timestamps for time-series metrics

        Returns:
            Dict of all metric values
        """
        y_true = np.asarray(y_true).flatten()
        y_pred = np.asarray(y_pred).flatten()

        metrics = {}
        metrics.update(self._regression_metrics(y_true, y_pred))
        metrics.update(self._trading_metrics(y_true, y_pred))

        return metrics

    def _regression_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> dict[str, float]:
        """Standard regression metrics."""
        residuals = y_true - y_pred
        mae = float(np.mean(np.abs(residuals)))
        mse = float(np.mean(residuals ** 2))
        rmse = float(np.sqrt(mse))

        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = float(1 - ss_res / (ss_tot + 1e-10))

        # MAPE — avoid division by zero
        nonzero = y_true != 0
        mape = float(np.mean(np.abs(residuals[nonzero] / y_true[nonzero]))) if nonzero.sum() > 0 else np.nan

        # Direction accuracy
        dir_true = np.sign(y_true)
        dir_pred = np.sign(y_pred)
        dir_accuracy = float(np.mean(dir_true == dir_pred))

        return {
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
            "mape": mape,
            "direction_accuracy": dir_accuracy,
        }

    def _trading_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> dict[str, float]:
        """
        Trading performance metrics based on actual PnL.
        Uses y_pred > 0 as trade signal (long when model predicts positive PnL).
        """
        # Simulated strategy: only take trades where model predicts profit
        pred_positive = y_pred > 0
        
        # Strategy returns: take actual PnL when predicted positive, else 0
        strategy_returns = np.where(pred_positive, y_true, 0.0)

        # Baseline: all trades
        all_returns = y_true

        metrics = {}

        # Win rate on selected trades
        selected = y_true[pred_positive]
        metrics["win_rate"] = float((selected > 0).mean()) if len(selected) > 0 else 0.0
        metrics["n_predicted_trades"] = int(pred_positive.sum())

        # Sharpe ratio (annualized — hourly data, ~5040 trading hours/year)
        annual_factor = np.sqrt(5040)
        metrics["sharpe"] = self._sharpe(strategy_returns, annual_factor)
        metrics["sortino"] = self._sortino(strategy_returns, annual_factor)

        # Cumulative and drawdown
        cum_returns = np.cumsum(strategy_returns)
        metrics["cumulative_return"] = float(cum_returns[-1]) if len(cum_returns) > 0 else 0.0
        metrics["max_drawdown"] = self._max_drawdown(strategy_returns)
        metrics["calmar"] = abs(metrics["cumulative_return"] / (abs(metrics["max_drawdown"]) + 1e-10))

        # Profit factor
        wins = strategy_returns[strategy_returns > 0]
        losses = strategy_returns[strategy_returns < 0]
        metrics["profit_factor"] = (
            float(wins.sum() / (abs(losses.sum()) + 1e-10)) if len(wins) > 0 else 0.0
        )

        # Average trade
        metrics["avg_trade_pnl"] = float(strategy_returns[pred_positive].mean()) if pred_positive.sum() > 0 else 0.0

        return metrics

    @staticmethod
    def _sharpe(returns: np.ndarray, annual_factor: float = 1.0) -> float:
        if len(returns) < 2 or returns.std() == 0:
            return 0.0
        return float(returns.mean() / (returns.std() + 1e-10) * annual_factor)

    @staticmethod
    def _sortino(returns: np.ndarray, annual_factor: float = 1.0) -> float:
        downside = returns[returns < 0]
        if len(downside) == 0:
            return float("inf")
        downside_std = np.sqrt(np.mean(downside ** 2))
        return float(returns.mean() / (downside_std + 1e-10) * annual_factor)

    @staticmethod
    def _max_drawdown(returns: np.ndarray) -> float:
        cum = np.cumsum(returns)
        peak = np.maximum.accumulate(cum)
        dd = cum - peak
        return float(dd.min()) if len(dd) > 0 else 0.0

    def format_report(self, metrics: dict[str, float]) -> str:
        """Format metrics as a readable report string."""
        lines = [
            "=" * 50,
            "MODEL EVALUATION REPORT",
            "=" * 50,
            "REGRESSION METRICS",
            f"  MAE:               {metrics.get('mae', 0):.6f}",
            f"  RMSE:              {metrics.get('rmse', 0):.6f}",
            f"  R²:                {metrics.get('r2', 0):.4f}",
            f"  MAPE:              {metrics.get('mape', 0):.2%}",
            f"  Direction Acc:     {metrics.get('direction_accuracy', 0):.2%}",
            "-" * 50,
            "TRADING METRICS",
            f"  Sharpe Ratio:      {metrics.get('sharpe', 0):.2f}",
            f"  Sortino Ratio:     {metrics.get('sortino', 0):.2f}",
            f"  Max Drawdown:      {metrics.get('max_drawdown', 0):.4f}",
            f"  Win Rate:          {metrics.get('win_rate', 0):.2%}",
            f"  Profit Factor:     {metrics.get('profit_factor', 0):.2f}",
            f"  Cumulative Return: {metrics.get('cumulative_return', 0):.4f}",
            f"  Calmar Ratio:      {metrics.get('calmar', 0):.2f}",
            f"  Avg Trade PnL:     {metrics.get('avg_trade_pnl', 0):.4f}",
            f"  Predicted Trades:  {metrics.get('n_predicted_trades', 0):,}",
            "=" * 50,
        ]
        return "\n".join(lines)
