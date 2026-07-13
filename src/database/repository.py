"""
QuantTrade ML Pipeline — Database Repository Layer
Repository pattern encapsulating all database operations.
"""
from __future__ import annotations

import sys
import uuid
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Generator

import joblib
import pandas as pd
from loguru import logger
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import settings
from src.database.models import (
    Base, FeatureImportanceRecord, ForexCandle, MacroEvent, ModelRun,
    Prediction, StrategyPerformance, Trade, WalkForwardFold,
)


class DatabaseManager:
    """
    Central database manager.
    Provides connection management and schema initialization.
    """

    def __init__(self, db_path: Path | str | None = None) -> None:
        self.db_path = Path(db_path or settings.db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.engine = create_engine(
            f"sqlite:///{self.db_path.as_posix()}",
            echo=False,
            connect_args={"check_same_thread": False, "timeout": 30},
        )
        self.SessionLocal = sessionmaker(bind=self.engine, expire_on_commit=False)
        Base.metadata.create_all(self.engine)
        logger.info("Database initialized at {}", self.db_path)

    @contextmanager
    def session(self) -> Generator[Session, None, None]:
        """Context manager for database sessions."""
        sess = self.SessionLocal()
        try:
            yield sess
            sess.commit()
        except Exception as exc:
            sess.rollback()
            logger.exception("Database session error: {}", exc)
            raise
        finally:
            sess.close()

    def execute(self, sql: str, params: dict | None = None) -> pd.DataFrame:
        """Execute a raw SQL query and return a DataFrame."""
        with self.engine.connect() as conn:
            result = conn.execute(text(sql), params or {})
            return pd.DataFrame(result.fetchall(), columns=list(result.keys()))


class ForexRepository:
    """Repository for EUR/USD candle data."""

    def __init__(self, db: DatabaseManager) -> None:
        self.db = db

    def bulk_insert(self, df: pd.DataFrame) -> int:
        """Insert forex candles from DataFrame. Returns count inserted."""
        records = []
        required_cols = ["bid_open", "bid_high", "bid_low", "bid_close",
                         "ask_open", "ask_high", "ask_low", "ask_close"]
        for ts, row in df.iterrows():
            records.append(ForexCandle(
                timestamp=pd.Timestamp(ts).to_pydatetime().replace(tzinfo=None),
                instrument=str(row.get("instrument", "EURUSD")),
                bid_open=float(row.get("bid_open", 0)),
                bid_high=float(row.get("bid_high", 0)),
                bid_low=float(row.get("bid_low", 0)),
                bid_close=float(row.get("bid_close", 0)),
                ask_open=float(row.get("ask_open", 0)),
                ask_high=float(row.get("ask_high", 0)),
                ask_low=float(row.get("ask_low", 0)),
                ask_close=float(row.get("ask_close", 0)),
                mid_open=float(row.get("mid_open", 0)) if "mid_open" in row else None,
                mid_high=float(row.get("mid_high", 0)) if "mid_high" in row else None,
                mid_low=float(row.get("mid_low", 0)) if "mid_low" in row else None,
                mid_close=float(row.get("mid_close", 0)) if "mid_close" in row else None,
                spread=float(row.get("spread", 0)) if "spread" in row else None,
                spread_pips=float(row.get("spread_pips", 0)) if "spread_pips" in row else None,
                is_outlier=bool(row.get("is_outlier", False)),
                is_weekend_gap=bool(row.get("is_weekend_gap", False)),
                data_quality_score=float(row.get("data_quality_score", 1.0)),
            ))

        with self.db.session() as sess:
            # Upsert via delete + insert for simplicity
            sess.query(ForexCandle).delete()
            for chunk_start in range(0, len(records), 5000):
                chunk = records[chunk_start:chunk_start + 5000]
                sess.bulk_save_objects(chunk)
            count = len(records)

        logger.info("Inserted {} forex candles", count)
        return count

    def load_all(self) -> pd.DataFrame:
        """Load all forex candles as DataFrame with DatetimeIndex."""
        with self.db.session() as sess:
            rows = sess.query(ForexCandle).order_by(ForexCandle.timestamp).all()
        if not rows:
            return pd.DataFrame()
        data = [{
            "timestamp": r.timestamp,
            "bid_open": r.bid_open, "bid_high": r.bid_high,
            "bid_low": r.bid_low, "bid_close": r.bid_close,
            "ask_open": r.ask_open, "ask_high": r.ask_high,
            "ask_low": r.ask_low, "ask_close": r.ask_close,
            "mid_close": r.mid_close, "spread_pips": r.spread_pips,
            "is_outlier": r.is_outlier,
        } for r in rows]
        df = pd.DataFrame(data)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        return df.set_index("timestamp")

    def count(self) -> int:
        with self.db.session() as sess:
            return sess.query(ForexCandle).count()


class MacroRepository:
    """Repository for macroeconomic events."""

    def __init__(self, db: DatabaseManager) -> None:
        self.db = db

    def bulk_insert(self, df: pd.DataFrame) -> int:
        """Insert macro events from DataFrame."""
        records = []
        for _, row in df.iterrows():
            ts = row.get("timestamp_utc")
            if pd.isna(ts) or ts is None:
                continue
            ts_naive = pd.Timestamp(ts).to_pydatetime().replace(tzinfo=None)
            records.append(MacroEvent(
                event_name=str(row.get("event_name", ""))[:255],
                country=str(row.get("country", ""))[:100],
                currency=str(row.get("currency", ""))[:10],
                timestamp_utc=ts_naive,
                forecast=self._safe_float(row.get("forecast")),
                actual=self._safe_float(row.get("actual")),
                previous=self._safe_float(row.get("previous")),
                surprise=self._safe_float(row.get("surprise")),
                surprise_direction=int(row.get("surprise_direction", 0)),
                surprise_magnitude=self._safe_float(row.get("surprise_magnitude")),
                impact=str(row.get("impact", "medium"))[:20],
                impact_score=int(row.get("impact_score", 1)),
                category=str(row.get("category", "other"))[:50],
                is_high_impact=bool(row.get("is_high_impact", False)),
                eurusd_relevant=bool(row.get("eurusd_relevant", True)),
            ))

        with self.db.session() as sess:
            sess.query(MacroEvent).delete()
            sess.bulk_save_objects(records)

        logger.info("Inserted {} macro events", len(records))
        return len(records)

    def load_all(self) -> pd.DataFrame:
        with self.db.session() as sess:
            rows = sess.query(MacroEvent).order_by(MacroEvent.timestamp_utc).all()
        if not rows:
            return pd.DataFrame()
        data = [{
            "event_name": r.event_name, "country": r.country, "currency": r.currency,
            "timestamp_utc": pd.Timestamp(r.timestamp_utc, tz="UTC"),
            "forecast": r.forecast, "actual": r.actual, "previous": r.previous,
            "surprise": r.surprise, "surprise_direction": r.surprise_direction,
            "surprise_magnitude": r.surprise_magnitude, "impact": r.impact,
            "impact_score": r.impact_score, "category": r.category,
            "is_high_impact": r.is_high_impact, "eurusd_relevant": r.eurusd_relevant,
        } for r in rows]
        return pd.DataFrame(data)

    @staticmethod
    def _safe_float(val: Any) -> float | None:
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return None
        try:
            return float(val)
        except (ValueError, TypeError):
            return None

    def count(self) -> int:
        with self.db.session() as sess:
            return sess.query(MacroEvent).count()


class TradeRepository:
    """Repository for simulated trade records."""

    def __init__(self, db: DatabaseManager) -> None:
        self.db = db

    def bulk_insert(self, df: pd.DataFrame, sim_run_id: str | None = None) -> int:
        """Insert trades from DataFrame."""
        sim_run_id = sim_run_id or str(uuid.uuid4())[:8]
        records = []
        for _, row in df.iterrows():
            records.append(Trade(
                strategy_id=str(row.get("strategy_id", "unknown")),
                direction=int(row.get("direction", 0)),
                entry_time=pd.Timestamp(row["entry_time"]).to_pydatetime().replace(tzinfo=None),
                exit_time=pd.Timestamp(row["exit_time"]).to_pydatetime().replace(tzinfo=None),
                entry_price=float(row.get("entry_price", 0)),
                exit_price=float(row.get("exit_price", 0)),
                quantity=float(row.get("quantity", 0)),
                position_size_usd=float(row.get("position_size_usd", 0)),
                stop_loss=float(row.get("stop_loss", 0)),
                take_profit=float(row.get("take_profit", 0)),
                exit_reason=str(row.get("exit_reason", ""))[:20],
                holding_bars=int(row.get("holding_bars", 0)),
                pnl=float(row.get("pnl", 0)),
                pnl_usd=float(row.get("pnl_usd", 0)),
                pnl_pct=float(row.get("pnl_pct", 0)),
                win=bool(row.get("win", False)),
                rolling_win_rate=float(row.get("rolling_win_rate", 0)) if "rolling_win_rate" in row else None,
                rolling_avg_pnl=float(row.get("rolling_avg_pnl", 0)) if "rolling_avg_pnl" in row else None,
                prev_trade_pnl=float(row.get("prev_trade_pnl", 0)) if "prev_trade_pnl" in row else None,
                sim_run_id=sim_run_id,
            ))

        with self.db.session() as sess:
            sess.query(Trade).delete()
            for chunk_start in range(0, len(records), 1000):
                sess.bulk_save_objects(records[chunk_start:chunk_start + 1000])

        logger.info("Inserted {} trades", len(records))
        return len(records)

    def load_all(self) -> pd.DataFrame:
        with self.db.session() as sess:
            rows = sess.query(Trade).order_by(Trade.entry_time).all()
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame([{
            "id": r.id, "strategy_id": r.strategy_id, "direction": r.direction,
            "entry_time": pd.Timestamp(r.entry_time, tz="UTC"),
            "exit_time": pd.Timestamp(r.exit_time, tz="UTC"),
            "entry_price": r.entry_price, "exit_price": r.exit_price,
            "quantity": r.quantity, "pnl": r.pnl, "pnl_usd": r.pnl_usd,
            "pnl_pct": r.pnl_pct, "win": r.win, "holding_bars": r.holding_bars,
            "exit_reason": r.exit_reason, "rolling_win_rate": r.rolling_win_rate,
            "rolling_avg_pnl": r.rolling_avg_pnl, "prev_trade_pnl": r.prev_trade_pnl,
            "sim_run_id": r.sim_run_id,
        } for r in rows])

    def count(self) -> int:
        with self.db.session() as sess:
            return sess.query(Trade).count()


class ModelRepository:
    """Repository for model run metadata and fold results."""

    def __init__(self, db: DatabaseManager) -> None:
        self.db = db

    def save_run(self, run_data: dict) -> str:
        run_id = run_data.get("run_id") or str(uuid.uuid4())[:12]
        with self.db.session() as sess:
            existing = sess.query(ModelRun).filter_by(run_id=run_id).first()
            if existing:
                for k, v in run_data.items():
                    setattr(existing, k, v)
            else:
                run = ModelRun(run_id=run_id, **{k: v for k, v in run_data.items() if k != "run_id"})
                run.run_id = run_id
                sess.add(run)
        logger.info("Saved model run: {}", run_id)
        return run_id

    def save_folds(self, run_id: str, folds_data: list[dict]) -> None:
        with self.db.session() as sess:
            sess.query(WalkForwardFold).filter_by(run_id=run_id).delete()
            for fold in folds_data:
                sess.add(WalkForwardFold(run_id=run_id, **fold))

    def _load_model_artifact(self, model_path: str) -> dict | None:
        try:
            if not Path(model_path).exists():
                return None
            artifact = joblib.load(model_path)
            if not isinstance(artifact, dict):
                return None

            loaded: dict[str, object] = {}
            if "overall_metrics" in artifact and isinstance(artifact["overall_metrics"], dict):
                loaded["overall_metrics"] = artifact["overall_metrics"]
            if "fold_results" in artifact:
                loaded["fold_results"] = artifact.get("fold_results", []) or []
            if "xgb_importance" in artifact:
                loaded["xgb_importance"] = pd.DataFrame(artifact.get("xgb_importance", []))
            if "shap_importance" in artifact:
                loaded["shap_importance"] = pd.DataFrame(artifact.get("shap_importance", []))
            for extra_key in ["all_predictions", "all_actuals", "best_params", "feature_names"]:
                if extra_key in artifact:
                    loaded[extra_key] = artifact[extra_key]
            return loaded
        except Exception as exc:
            logger.warning("Unable to load model artifact from %s: %s", model_path, exc)
            return None

    def load_latest_run(self) -> dict | None:
        with self.db.session() as sess:
            run = sess.query(ModelRun).order_by(ModelRun.created_at.desc()).first()
        if run is None:
            return None

        result: dict[str, object] = {
            "run_id": run.run_id,
            "created_at": run.created_at,
            "model_type": run.model_type,
            "target_variable": run.target_variable,
            "n_features": run.n_features,
            "mae": run.mae,
            "rmse": run.rmse,
            "r2": run.r2,
            "sharpe": run.sharpe,
            "win_rate": run.win_rate,
            "model_path": run.model_path,
            "status": run.status,
            "best_params": run.best_params,
        }

        if run.model_path:
            artifact_data = self._load_model_artifact(run.model_path)
            if artifact_data:
                result.update(artifact_data)

        # Normalize legacy database-only runs into expected frontend output.
        if "overall_metrics" not in result:
            result["overall_metrics"] = {
                key: getattr(run, key)
                for key in ["mae", "rmse", "r2", "sharpe", "sortino", "max_drawdown", "win_rate", "profit_factor"]
                if getattr(run, key, None) is not None
            }
            result["overall_metrics"] = result["overall_metrics"] or {}

        if "fold_results" not in result:
            fold_df = self.load_folds(run.run_id)
            if not fold_df.empty:
                result["fold_results"] = fold_df.to_dict("records")
                result["n_folds"] = len(result["fold_results"])

        if "n_folds" not in result and "fold_results" in result:
            result["n_folds"] = len(result["fold_results"])

        return result

    def load_folds(self, run_id: str) -> pd.DataFrame:
        with self.db.session() as sess:
            rows = sess.query(WalkForwardFold).filter_by(run_id=run_id).order_by(WalkForwardFold.fold_index).all()
        return pd.DataFrame([{
            "fold_index": r.fold_index, "train_start": r.train_start, "train_end": r.train_end,
            "test_start": r.test_start, "test_end": r.test_end, "n_train": r.n_train,
            "n_test": r.n_test, "mae": r.mae, "rmse": r.rmse, "r2": r.r2,
            "sharpe": r.sharpe, "win_rate": r.win_rate,
        } for r in rows])

    def count(self) -> int:
        with self.db.session() as sess:
            return sess.query(ModelRun).count()


# ------------------------------------------------------------------ #
# Module-level singleton
# ------------------------------------------------------------------ #
_db_manager: DatabaseManager | None = None


def get_db() -> DatabaseManager:
    """Get or create the database manager singleton."""
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager()
    return _db_manager


def get_repos() -> tuple[ForexRepository, MacroRepository, TradeRepository, ModelRepository]:
    """Return all repository instances."""
    db = get_db()
    return (
        ForexRepository(db),
        MacroRepository(db),
        TradeRepository(db),
        ModelRepository(db),
    )
