"""
Shared dashboard state and UI utilities for the Streamlit analytics application.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st
from loguru import logger

from config.settings import settings
from src.database.repository import get_repos


def render_status_card(
    title: str,
    summary: str,
    details: str | None = None,
    icon: str = "ℹ️",
) -> None:
    st.markdown(
        f"""
        <div style="
            background: linear-gradient(180deg, rgba(15,23,42,0.95), rgba(20,31,47,0.98));
            border: 1px solid rgba(96, 165, 250, 0.16);
            border-radius: 18px;
            padding: 1.4rem;
            margin-bottom: 1.5rem;
            box-shadow: 0 16px 45px rgba(0, 0, 0, 0.08);
        ">
            <div style="display: flex; align-items: flex-start; gap: 1rem; margin-bottom: 0.75rem;">
                <div style="font-size: 1.8rem; line-height: 1;">{icon}</div>
                <div>
                    <div style="font-size: 1.15rem; font-weight: 700; color: #f8fafc; margin-bottom: 0.2rem;">{title}</div>
                    <div style="font-size: 0.95rem; color: #b8c4d0;">{summary}</div>
                </div>
            </div>
            {f'<div style="font-size: 0.88rem; color: #9ca3af;">{details}</div>' if details else ''}
        </div>
        """,
        unsafe_allow_html=True,
    )


def ensure_dashboard_state() -> None:
    forex_repo, macro_repo, trade_repo, model_repo = get_repos()
    outputs_dir = Path("data/outputs")

    steps = {
        "data_loaded": forex_repo.count() > 0,
        "macro_scraped": macro_repo.count() > 0 or (outputs_dir / "datasets/macro_events.csv").exists(),
        "features_built": (outputs_dir / "datasets/engineered_features.csv").exists(),
        "simulation_run": trade_repo.count() > 0,
        "model_trained": (outputs_dir / "models/model.joblib").exists() or model_repo.count() > 0,
    }

    st.session_state.pipeline_steps = steps
    st.session_state.pipeline_complete = all(steps.values())
    st.session_state.model_loaded = steps["model_trained"]

    st.session_state.forex_df = pd.DataFrame()
    st.session_state.macro_df = pd.DataFrame()
    st.session_state.feature_df = pd.DataFrame()
    st.session_state.trade_log = pd.DataFrame()
    st.session_state.strategy_summary = pd.DataFrame()
    st.session_state.training_results = {}

    try:
        if steps["data_loaded"]:
            st.session_state.forex_df = forex_repo.load_all()

        if steps["macro_scraped"]:
            st.session_state.macro_df = macro_repo.load_all()

        if steps["features_built"]:
            feature_cache_path = Path(settings.data_processed_path) / "features.parquet"
            if feature_cache_path.exists():
                st.session_state.feature_df = pd.read_parquet(feature_cache_path)
            elif (outputs_dir / "datasets/engineered_features.csv").exists():
                st.session_state.feature_df = pd.read_csv(outputs_dir / "datasets/engineered_features.csv")

        if steps["simulation_run"]:
            st.session_state.trade_log = trade_repo.load_all()

        if not st.session_state.trade_log.empty:
            from src.simulation.engine import SimulationEngine
            engine = SimulationEngine()
            st.session_state.strategy_summary = engine.get_strategy_summary(st.session_state.trade_log)

        st.session_state.training_results = model_repo.load_latest_run() or {}
        if st.session_state.training_results and "overall_metrics" not in st.session_state.training_results:
            metrics = {
                k: st.session_state.training_results.get(k)
                for k in ["mae", "rmse", "r2", "sharpe", "sortino", "max_drawdown", "win_rate", "profit_factor"]
                if st.session_state.training_results.get(k) is not None
            }
            st.session_state.training_results["overall_metrics"] = metrics

        # If we loaded a model run but the metrics are still all zeros, refresh from the repository again.
        if st.session_state.training_results and st.session_state.training_results.get("best_params"):
            current_metrics = st.session_state.training_results.get("overall_metrics", {})
            if all(current_metrics.get(key, 0) == 0 for key in ["mae", "rmse", "r2", "sharpe", "win_rate"]):
                latest = model_repo.load_latest_run() or {}
                if latest and latest.get("overall_metrics", {}).get("mae", 0) != 0:
                    st.session_state.training_results = latest
    except Exception as exc:
        logger.exception("Unable to initialize dashboard state: %s", exc)
    finally:
        st.session_state._dashboard_loaded = True
