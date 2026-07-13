"""
QuantTrade ML Pipeline — Centralized Settings
Uses pydantic-settings for type-safe, 12-factor-compliant configuration.
All values can be overridden via environment variables or .env file.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Project root — always reliable regardless of working directory
PROJECT_ROOT = Path(__file__).parent.parent.resolve()


class Settings(BaseSettings):
    """Production-grade settings for QuantTrade ML Pipeline."""

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ------------------------------------------------------------------ #
    # API Keys
    # ------------------------------------------------------------------ #
    apify_api_key: str = Field(
        default="",
        description="Apify API key for macro event scraping",
    )

    # ------------------------------------------------------------------ #
    # Paths
    # ------------------------------------------------------------------ #
    project_root: Path = Field(default=PROJECT_ROOT)
    data_raw_path: Path = Field(default=PROJECT_ROOT / "data" / "raw" / "eurusd_hour.csv")
    data_processed_path: Path = Field(default=PROJECT_ROOT / "data" / "processed")
    db_path: Path = Field(default=PROJECT_ROOT / "data" / "db" / "quanttrade.db")
    model_path: Path = Field(default=PROJECT_ROOT / "models")
    log_path: Path = Field(default=PROJECT_ROOT / "logs")

    # ------------------------------------------------------------------ #
    # Logging
    # ------------------------------------------------------------------ #
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    log_format: Literal["colored", "json"] = "colored"

    # ------------------------------------------------------------------ #
    # Walk-Forward Validation
    # ------------------------------------------------------------------ #
    wf_train_days: int = Field(default=90, ge=30, le=730)
    wf_test_days: int = Field(default=30, ge=7, le=180)
    wf_step_days: int = Field(default=30, ge=7, le=180)
    wf_embargo_days: int = Field(default=5, ge=0, le=30)

    # ------------------------------------------------------------------ #
    # Machine Learning
    # ------------------------------------------------------------------ #
    random_seed: int = Field(default=42)
    n_optuna_trials: int = Field(default=50, ge=10, le=500)
    max_features: int = Field(default=50, ge=10, le=200)

    # ------------------------------------------------------------------ #
    # Trade Simulation
    # ------------------------------------------------------------------ #
    initial_capital: float = Field(default=100_000.0, gt=0)
    risk_per_trade: float = Field(default=0.01, gt=0, le=0.1)
    max_spread_pips: float = Field(default=3.0, gt=0)
    slippage_pips: float = Field(default=0.5, ge=0)
    pip_value: float = Field(default=0.0001, description="1 pip in price units for EUR/USD")

    # ------------------------------------------------------------------ #
    # Apify / Macro Scraping
    # ------------------------------------------------------------------ #
    apify_macro_days: int = Field(default=180, ge=30, le=365)

    # ------------------------------------------------------------------ #
    # Feature Engineering
    # ------------------------------------------------------------------ #
    rsi_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    atr_period: int = 14
    bb_period: int = 20
    bb_std: float = 2.0
    ema_periods: list[int] = Field(default=[9, 21, 50, 200])
    sma_periods: list[int] = Field(default=[20, 50, 200])
    rolling_windows: list[int] = Field(default=[5, 10, 20, 50])

    @field_validator("data_processed_path", "db_path", "model_path", "log_path", "data_raw_path", mode="before")
    @classmethod
    def ensure_path(cls, v: str | Path) -> Path:
        p = Path(v)
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        if p.suffix == "":
            p.mkdir(parents=True, exist_ok=True)
        return p

    @field_validator("db_path", "data_raw_path", mode="before")
    @classmethod
    def ensure_file_parent(cls, v: str | Path) -> Path:
        p = Path(v)
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        p.parent.mkdir(parents=True, exist_ok=True)
        return p


# Singleton — import this everywhere
settings = Settings()
