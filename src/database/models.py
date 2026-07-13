"""
QuantTrade ML Pipeline — SQLAlchemy ORM Models
Defines the complete database schema for all data artifacts.
"""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    JSON, Boolean, Column, DateTime, Float, Index, Integer,
    String, Text, UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """Declarative base for all ORM models."""
    pass


class ForexCandle(Base):
    """Hourly EUR/USD market data."""
    __tablename__ = "forex_candles"
    __table_args__ = (
        UniqueConstraint("timestamp", name="uq_forex_timestamp"),
        Index("ix_forex_timestamp", "timestamp"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    instrument = Column(String(10), default="EURUSD")
    bid_open = Column(Float, nullable=False)
    bid_high = Column(Float, nullable=False)
    bid_low = Column(Float, nullable=False)
    bid_close = Column(Float, nullable=False)
    ask_open = Column(Float, nullable=False)
    ask_high = Column(Float, nullable=False)
    ask_low = Column(Float, nullable=False)
    ask_close = Column(Float, nullable=False)
    mid_open = Column(Float)
    mid_high = Column(Float)
    mid_low = Column(Float)
    mid_close = Column(Float)
    spread = Column(Float)
    spread_pips = Column(Float)
    is_outlier = Column(Boolean, default=False)
    is_weekend_gap = Column(Boolean, default=False)
    data_quality_score = Column(Float, default=1.0)


class MacroEvent(Base):
    """Macroeconomic events from Apify."""
    __tablename__ = "macro_events"
    __table_args__ = (
        Index("ix_macro_timestamp", "timestamp_utc"),
        Index("ix_macro_category", "category"),
        Index("ix_macro_impact", "impact_score"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    event_name = Column(String(255), nullable=False)
    country = Column(String(100))
    currency = Column(String(10))
    timestamp_utc = Column(DateTime, nullable=False)
    forecast = Column(Float)
    actual = Column(Float)
    previous = Column(Float)
    surprise = Column(Float)
    surprise_direction = Column(Integer)
    surprise_magnitude = Column(Float)
    impact = Column(String(20))
    impact_score = Column(Integer, default=1)
    category = Column(String(50))
    is_high_impact = Column(Boolean, default=False)
    eurusd_relevant = Column(Boolean, default=True)
    scraped_at = Column(DateTime, default=datetime.utcnow)


class Trade(Base):
    """Simulated trade records from strategy engine."""
    __tablename__ = "trades"
    __table_args__ = (
        Index("ix_trade_entry_time", "entry_time"),
        Index("ix_trade_strategy", "strategy_id"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    strategy_id = Column(String(50), nullable=False)
    direction = Column(Integer, nullable=False)
    entry_time = Column(DateTime, nullable=False)
    exit_time = Column(DateTime, nullable=False)
    entry_price = Column(Float, nullable=False)
    exit_price = Column(Float, nullable=False)
    quantity = Column(Float)
    position_size_usd = Column(Float)
    stop_loss = Column(Float)
    take_profit = Column(Float)
    exit_reason = Column(String(20))
    holding_bars = Column(Integer)
    pnl = Column(Float)
    pnl_usd = Column(Float)
    pnl_pct = Column(Float)
    win = Column(Boolean)
    rolling_win_rate = Column(Float)
    rolling_avg_pnl = Column(Float)
    prev_trade_pnl = Column(Float)
    consecutive_wins = Column(Integer, default=0)
    consecutive_losses = Column(Integer, default=0)
    strategy_params = Column(JSON)
    sim_run_id = Column(String(50))


class ModelRun(Base):
    """XGBoost model training run metadata."""
    __tablename__ = "model_runs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(String(50), unique=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    model_type = Column(String(50), default="xgboost")
    target_variable = Column(String(50))
    n_features = Column(Integer)
    n_train_samples = Column(Integer)
    n_test_samples = Column(Integer)
    n_folds = Column(Integer)
    train_days = Column(Integer)
    test_days = Column(Integer)
    embargo_days = Column(Integer)
    best_params = Column(JSON)
    mae = Column(Float)
    rmse = Column(Float)
    r2 = Column(Float)
    sharpe = Column(Float)
    sortino = Column(Float)
    max_drawdown = Column(Float)
    win_rate = Column(Float)
    profit_factor = Column(Float)
    model_path = Column(String(500))
    feature_importance = Column(JSON)
    status = Column(String(20), default="pending")
    notes = Column(Text)


class WalkForwardFold(Base):
    """Per-fold results from walk-forward validation."""
    __tablename__ = "walk_forward_folds"
    __table_args__ = (
        Index("ix_wf_run_id", "run_id"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(String(50), nullable=False)
    fold_index = Column(Integer, nullable=False)
    train_start = Column(DateTime)
    train_end = Column(DateTime)
    test_start = Column(DateTime)
    test_end = Column(DateTime)
    n_train = Column(Integer)
    n_test = Column(Integer)
    mae = Column(Float)
    rmse = Column(Float)
    r2 = Column(Float)
    sharpe = Column(Float)
    win_rate = Column(Float)
    best_iteration = Column(Integer)
    best_params = Column(JSON)
    top_features = Column(JSON)


class Prediction(Base):
    """Model prediction records."""
    __tablename__ = "predictions"
    __table_args__ = (
        Index("ix_pred_timestamp", "timestamp"),
        Index("ix_pred_strategy", "strategy_id"),
        Index("ix_pred_run_id", "run_id"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(String(50))
    timestamp = Column(DateTime, nullable=False)
    strategy_id = Column(String(50))
    predicted_pnl = Column(Float)
    predicted_win_rate = Column(Float)
    confidence_lower = Column(Float)
    confidence_upper = Column(Float)
    actual_pnl = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)


class StrategyPerformance(Base):
    """Aggregated per-strategy performance statistics."""
    __tablename__ = "strategy_performance"
    __table_args__ = (
        Index("ix_sp_run_id", "run_id"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(String(50))
    strategy_id = Column(String(50), nullable=False)
    total_trades = Column(Integer)
    win_rate = Column(Float)
    total_pnl_usd = Column(Float)
    avg_pnl_usd = Column(Float)
    sharpe = Column(Float)
    sortino = Column(Float)
    max_drawdown = Column(Float)
    profit_factor = Column(Float)
    avg_holding_bars = Column(Float)
    best_month_pnl = Column(Float)
    worst_month_pnl = Column(Float)
    computed_at = Column(DateTime, default=datetime.utcnow)


class FeatureImportanceRecord(Base):
    """Stored feature importance scores."""
    __tablename__ = "feature_importance"
    __table_args__ = (
        Index("ix_fi_run_id", "run_id"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(String(50), nullable=False)
    feature_name = Column(String(255), nullable=False)
    importance_gain = Column(Float)
    importance_cover = Column(Float)
    importance_frequency = Column(Float)
    shap_mean_abs = Column(Float)
    rank = Column(Integer)
    group = Column(String(50))
