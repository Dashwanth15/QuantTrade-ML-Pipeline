"""
QuantTrade ML Pipeline — Database Layer Tests

Tests all repository classes against a transient in-memory SQLite database.
No files are written to disk; each test starts with a clean slate.
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.database.models import (
    Base, ForexCandle, MacroEvent, Trade, ModelRun,
    WalkForwardFold as WalkForwardFoldModel,
    Prediction, StrategyPerformance, FeatureImportanceRecord,
)
from src.database.repository import (
    DatabaseManager, ForexRepository, MacroRepository,
    TradeRepository, ModelRepository,
)


# ── In-memory SQLite fixture ──────────────────────────────────────────────────

class InMemoryDB(DatabaseManager):
    """DatabaseManager backed by a fresh in-memory SQLite for each test."""

    def __init__(self):
        self.engine = create_engine(
            "sqlite:///:memory:",
            echo=False,
            connect_args={"check_same_thread": False},
        )
        self.SessionLocal = sessionmaker(bind=self.engine, expire_on_commit=False)
        Base.metadata.create_all(self.engine)


@pytest.fixture
def db():
    return InMemoryDB()


@pytest.fixture
def forex_repo(db):
    return ForexRepository(db)


@pytest.fixture
def macro_repo(db):
    return MacroRepository(db)


@pytest.fixture
def trade_repo(db):
    return TradeRepository(db)


@pytest.fixture
def model_repo(db):
    return ModelRepository(db)


# ── Sample DataFrames ─────────────────────────────────────────────────────────

def make_forex_df(n: int = 50) -> pd.DataFrame:
    np.random.seed(42)
    dates = pd.date_range("2015-01-01", periods=n, freq="h", tz="UTC")
    price = 1.2 + np.cumsum(np.random.normal(0, 0.0003, n))
    spread = 0.0002
    return pd.DataFrame({
        "bid_open": price - spread / 2,
        "bid_high": price + 0.0003 - spread / 2,
        "bid_low": price - 0.0003 - spread / 2,
        "bid_close": price - spread / 2,
        "ask_open": price + spread / 2,
        "ask_high": price + 0.0003 + spread / 2,
        "ask_low": price - 0.0003 + spread / 2,
        "ask_close": price + spread / 2,
        "mid_open": price,
        "mid_high": price + 0.0003,
        "mid_low": price - 0.0003,
        "mid_close": price,
        "spread": spread,
        "spread_pips": spread / 0.0001,
    }, index=dates)


def make_macro_df(n: int = 10) -> pd.DataFrame:
    dates = pd.date_range("2015-01-01", periods=n, freq="W", tz="UTC")
    return pd.DataFrame({
        "event_name": [f"Event {i}" for i in range(n)],
        "country": ["US"] * n,
        "currency": ["USD"] * n,
        "timestamp_utc": dates,
        "forecast": np.random.normal(0, 0.5, n),
        "actual": np.random.normal(0, 0.5, n),
        "previous": np.random.normal(0, 0.5, n),
        "surprise": np.random.normal(0, 0.1, n),
        "surprise_direction": np.random.choice([-1, 0, 1], n),
        "surprise_magnitude": np.abs(np.random.normal(0, 0.1, n)),
        "impact": ["high"] * n,
        "impact_score": [3] * n,
        "category": ["employment"] * n,
        "is_high_impact": [True] * n,
        "eurusd_relevant": [True] * n,
    })


def make_trade_df(n: int = 20) -> pd.DataFrame:
    np.random.seed(1)
    dates = pd.date_range("2015-01-01", periods=n, freq="6h", tz="UTC")
    pnl_usd = np.random.normal(0, 100, n)
    return pd.DataFrame({
        "strategy_id": np.random.choice(["momentum", "bollinger"], n),
        "direction": np.random.choice([-1, 1], n),
        "entry_time": dates,
        "exit_time": dates + pd.Timedelta(hours=4),
        "entry_price": 1.2 + np.random.normal(0, 0.001, n),
        "exit_price": 1.2 + np.random.normal(0, 0.001, n),
        "quantity": 100_000.0,
        "position_size_usd": 100_000.0,
        "stop_loss": 1.19,
        "take_profit": 1.21,
        "exit_reason": "tp",
        "holding_bars": 4,
        "pnl": pnl_usd / 100_000,
        "pnl_usd": pnl_usd,
        "pnl_pct": pnl_usd / 100_000,
        "win": pnl_usd > 0,
        "rolling_win_rate": 0.5,
        "rolling_avg_pnl": pnl_usd.mean(),
        "prev_trade_pnl": np.roll(pnl_usd, 1),
    })


# ── Schema Tests ──────────────────────────────────────────────────────────────

class TestDatabaseSchema:
    def test_all_tables_created(self, db):
        """Verify all 8 ORM tables exist in the schema."""
        from sqlalchemy import inspect
        inspector = inspect(db.engine)
        tables = set(inspector.get_table_names())
        expected = {
            "forex_candles", "macro_events", "trades",
            "model_runs", "walk_forward_folds", "predictions",
            "strategy_performance", "feature_importance",
        }
        missing = expected - tables
        assert not missing, f"Missing tables: {missing}"

    def test_forex_candles_columns(self, db):
        from sqlalchemy import inspect
        inspector = inspect(db.engine)
        cols = {c["name"] for c in inspector.get_columns("forex_candles")}
        required = {"id", "timestamp", "bid_close", "ask_close", "mid_close", "spread_pips"}
        assert required.issubset(cols)

    def test_trades_columns(self, db):
        from sqlalchemy import inspect
        inspector = inspect(db.engine)
        cols = {c["name"] for c in inspector.get_columns("trades")}
        required = {"id", "strategy_id", "entry_time", "exit_time", "pnl_usd", "win"}
        assert required.issubset(cols)


# ── ForexRepository Tests ─────────────────────────────────────────────────────

class TestForexRepository:
    def test_bulk_insert_returns_count(self, forex_repo):
        df = make_forex_df(50)
        count = forex_repo.bulk_insert(df)
        assert count == 50

    def test_load_all_returns_dataframe(self, forex_repo):
        df = make_forex_df(50)
        forex_repo.bulk_insert(df)
        result = forex_repo.load_all()
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 50

    def test_load_all_has_correct_columns(self, forex_repo):
        df = make_forex_df(20)
        forex_repo.bulk_insert(df)
        result = forex_repo.load_all()
        assert "mid_close" in result.columns
        assert "bid_close" in result.columns

    def test_count_matches_inserts(self, forex_repo):
        df = make_forex_df(30)
        forex_repo.bulk_insert(df)
        assert forex_repo.count() == 30

    def test_bulk_insert_replaces_existing(self, forex_repo):
        """Second insert should overwrite, not append."""
        forex_repo.bulk_insert(make_forex_df(10))
        forex_repo.bulk_insert(make_forex_df(25))
        assert forex_repo.count() == 25

    def test_load_all_has_utc_index(self, forex_repo):
        forex_repo.bulk_insert(make_forex_df(20))
        result = forex_repo.load_all()
        assert result.index.tz is not None, "Index should be UTC-aware"


# ── MacroRepository Tests ─────────────────────────────────────────────────────

class TestMacroRepository:
    def test_bulk_insert_returns_count(self, macro_repo):
        df = make_macro_df(10)
        count = macro_repo.bulk_insert(df)
        assert count == 10

    def test_load_all_returns_dataframe(self, macro_repo):
        macro_repo.bulk_insert(make_macro_df(10))
        result = macro_repo.load_all()
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 10

    def test_event_names_preserved(self, macro_repo):
        df = make_macro_df(5)
        macro_repo.bulk_insert(df)
        result = macro_repo.load_all()
        assert set(result["event_name"]) == {f"Event {i}" for i in range(5)}

    def test_empty_load_returns_empty_df(self, macro_repo):
        result = macro_repo.load_all()
        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_skips_nan_timestamps(self, macro_repo):
        df = make_macro_df(5)
        df.loc[df.index[2], "timestamp_utc"] = None
        count = macro_repo.bulk_insert(df)
        assert count == 4  # Row with NaN timestamp is skipped


# ── TradeRepository Tests ─────────────────────────────────────────────────────

class TestTradeRepository:
    def test_bulk_insert_returns_count(self, trade_repo):
        df = make_trade_df(20)
        count = trade_repo.bulk_insert(df)
        assert count == 20

    def test_load_all_returns_sorted(self, trade_repo):
        trade_repo.bulk_insert(make_trade_df(20))
        result = trade_repo.load_all()
        assert result["entry_time"].is_monotonic_increasing

    def test_load_all_has_correct_columns(self, trade_repo):
        trade_repo.bulk_insert(make_trade_df(20))
        result = trade_repo.load_all()
        for col in ["strategy_id", "direction", "entry_time", "pnl_usd", "win"]:
            assert col in result.columns

    def test_win_column_is_bool(self, trade_repo):
        trade_repo.bulk_insert(make_trade_df(20))
        result = trade_repo.load_all()
        assert result["win"].dtype in [bool, object]
        # All values should be True or False
        assert result["win"].isin([True, False]).all()

    def test_trade_pnl_stored_correctly(self, trade_repo):
        df = make_trade_df(5)
        df["pnl_usd"] = [100.0, -50.0, 200.0, -30.0, 0.0]
        df["win"] = df["pnl_usd"] > 0
        trade_repo.bulk_insert(df)
        result = trade_repo.load_all()
        assert abs(result["pnl_usd"].sum() - 220.0) < 0.01


# ── ModelRepository Tests ─────────────────────────────────────────────────────

class TestModelRepository:
    def _run_data(self, run_id="test_run_001"):
        return {
            "run_id": run_id,
            "model_type": "xgboost",
            "target_variable": "pnl_usd",
            "n_features": 200,
            "n_train_samples": 5000,
            "n_test_samples": 1000,
            "n_folds": 5,
            "mae": 150.0,
            "rmse": 200.0,
            "r2": 0.15,
            "sharpe": 1.2,
            "win_rate": 0.52,
            "model_path": "models/test_run_001.joblib",
            "status": "complete",
            "best_params": {"n_estimators": 500, "max_depth": 6},
        }

    def test_save_run_succeeds(self, model_repo):
        run_id = model_repo.save_run(self._run_data())
        assert run_id == "test_run_001"

    def test_load_latest_run_returns_dict(self, model_repo):
        model_repo.save_run(self._run_data())
        result = model_repo.load_latest_run()
        assert result is not None
        assert result["run_id"] == "test_run_001"
        assert "overall_metrics" in result
        assert result["overall_metrics"]["mae"] == 150.0
        assert result["overall_metrics"]["sharpe"] == 1.2

    def test_load_latest_run_loads_artifact_from_model_path(self, model_repo, tmp_path):
        artifact = {
            "overall_metrics": {"mae": 12.3, "rmse": 45.6, "r2": 0.88, "sharpe": 2.1, "win_rate": 0.65},
            "fold_results": [{"fold_index": 0, "mae": 12.3, "rmse": 45.6, "r2": 0.88, "sharpe": 2.1, "win_rate": 0.65}],
            "all_predictions": [1.0, 2.0],
            "all_actuals": [1.1, 1.9],
        }
        artifact_path = tmp_path / "test_run_001.joblib"
        joblib.dump(artifact, artifact_path)

        data = self._run_data()
        data["model_path"] = str(artifact_path)
        model_repo.save_run(data)

        result = model_repo.load_latest_run()
        assert result is not None
        assert result["overall_metrics"]["mae"] == 12.3
        assert result["fold_results"][0]["fold_index"] == 0
        assert result["all_predictions"] == [1.0, 2.0]
        assert result["all_actuals"] == [1.1, 1.9]

    def test_load_latest_run_empty_db(self, model_repo):
        result = model_repo.load_latest_run()
        assert result is None

    def test_save_and_load_folds(self, model_repo):
        model_repo.save_run(self._run_data("fold_test_run"))
        folds_data = [
            {
                "fold_index": i,
                "train_start": datetime(2015, 1, 1),
                "train_end": datetime(2015, 4, 1),
                "test_start": datetime(2015, 4, 6),
                "test_end": datetime(2015, 5, 6),
                "n_train": 2160, "n_test": 720,
                "mae": 100.0 + i, "rmse": 150.0 + i,
                "r2": 0.1, "sharpe": 0.5, "win_rate": 0.5,
            }
            for i in range(3)
        ]
        model_repo.save_folds("fold_test_run", folds_data)
        result = model_repo.load_folds("fold_test_run")
        assert len(result) == 3
        assert list(result["fold_index"]) == [0, 1, 2]

    def test_save_run_upserts(self, model_repo):
        """Saving the same run_id twice should update, not create a duplicate."""
        data = self._run_data("upsert_run")
        model_repo.save_run(data)
        data["mae"] = 999.0  # Update MAE
        model_repo.save_run(data)
        result = model_repo.load_latest_run()
        assert result["mae"] == pytest.approx(999.0)

    def test_model_path_preserved(self, model_repo):
        model_repo.save_run(self._run_data())
        result = model_repo.load_latest_run()
        assert result["model_path"] == "models/test_run_001.joblib"


# ── Session / Transaction Tests ───────────────────────────────────────────────

class TestDatabaseSessionManagement:
    def test_session_commits_on_success(self, db):
        with db.session() as sess:
            sess.add(MacroEvent(
                event_name="Test Event",
                country="US",
                currency="USD",
                timestamp_utc=datetime(2015, 1, 1),
                impact_score=2,
            ))
        # Should be accessible outside the session
        with db.session() as sess:
            count = sess.query(MacroEvent).count()
        assert count == 1

    def test_session_rolls_back_on_failure(self, db):
        initial_count = 0
        with db.session() as sess:
            initial_count = sess.query(MacroEvent).count()

        try:
            with db.session() as sess:
                sess.add(MacroEvent(
                    event_name="Good Event",
                    country="US",
                    currency="USD",
                    timestamp_utc=datetime(2015, 1, 1),
                    impact_score=2,
                ))
                # Force an error mid-transaction
                raise RuntimeError("Simulated failure")
        except RuntimeError:
            pass

        with db.session() as sess:
            count = sess.query(MacroEvent).count()
        assert count == initial_count  # Rolled back, no new rows
