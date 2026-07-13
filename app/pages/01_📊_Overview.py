"""
QuantTrade ML Pipeline — Overview Dashboard Page
Aggregated KPIs, equity curves, strategy comparison.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.dashboard import ensure_dashboard_state, render_status_card
from src.visualization.themes import get_css, COLORS
from src.visualization.charts import equity_curve_chart, strategy_radar_chart, rolling_metrics_chart

st.set_page_config(page_title="Overview | QuantTrade", page_icon="📊", layout="wide")
st.markdown(get_css(), unsafe_allow_html=True)

ensure_dashboard_state()

st.markdown("## 📊 Executive Dashboard")
st.markdown("Enterprise-level performance view for dataset health, model status, and strategy analytics.")
st.divider()

from src.database.models import Prediction, ModelRun
from src.database.repository import get_db
from config.settings import settings

db = get_db()
with db.session() as sess:
    pred_count = sess.query(Prediction).count() if sess.query(Prediction).count() is not None else 0
    latest_run_obj = sess.query(ModelRun).order_by(ModelRun.created_at.desc()).first()
    last_run_time = latest_run_obj.created_at.strftime("%Y-%m-%d %H:%M:%S") if latest_run_obj else "N/A"

trade_log = st.session_state.get("trade_log", pd.DataFrame())
strategy_summary = st.session_state.get("strategy_summary", pd.DataFrame())
training_results = st.session_state.get("training_results", {})

loaded = st.session_state.pipeline_steps

st.markdown("### 🖥️ System Status Dashboard")
sc1, sc2, sc3, sc4 = st.columns(4)
with sc1:
    st.markdown(f"**Forex Dataset:** `{ 'Loaded' if loaded['data_loaded'] else 'Missing' }` ({len(st.session_state.forex_df):,} rows)")
    st.markdown(f"**Macro Events:** `{ 'Loaded' if loaded['macro_scraped'] else 'Missing' }` ({len(st.session_state.macro_df):,} records)")
with sc2:
    st.markdown(f"**Trade Simulation:** `{ 'Completed' if loaded['simulation_run'] else 'Missing' }` ({len(trade_log):,} trades)")
    st.markdown(f"**Features Matrix:** `{ 'Built' if loaded['features_built'] else 'Missing' }` ({len(st.session_state.feature_df.columns) if hasattr(st.session_state.feature_df, 'columns') else 0} cols)")
with sc3:
    st.markdown(f"**Model Status:** `{ 'Trained' if loaded['model_trained'] else 'Missing' }` ({training_results.get('n_folds', 0)} folds)")
    st.markdown(f"**Predictions:** `{ 'Ready' if pred_count > 0 else 'Missing' }` ({pred_count} records)")
with sc4:
    st.markdown(f"**Database:** `Connected` (`sqlite`)")
    st.markdown(f"**Last Run:** `{last_run_time}`")

if not st.session_state.pipeline_complete:
    render_status_card(
        "Pipeline Outputs Incomplete",
        "Some backend outputs are missing. The dashboard still displays available analytics and diagnostics.",
        details="Ensure the backend ML pipeline has generated cleaned data, engineered features, trade simulation logs, and model artifacts. Current dashboard panels show whichever outputs are available.",
        icon="⚠️",
    )

st.divider()

# ------------------------------------------------------------------ #
# KPI Metrics Row
# ------------------------------------------------------------------ #
total_trades = 0
win_rate = 0.0
total_pnl = 0.0
avg_pnl = 0.0
if trade_log is not None and not trade_log.empty:
    total_trades = len(trade_log)
    win_rate = trade_log["win"].mean() if "win" in trade_log.columns else 0
    total_pnl = trade_log["pnl_usd"].sum() if "pnl_usd" in trade_log.columns else 0
    avg_pnl = trade_log["pnl_usd"].mean() if "pnl_usd" in trade_log.columns else 0

# Compute Sharpe from trade PnL
pnl_series = trade_log["pnl_usd"] if "pnl_usd" in trade_log.columns else pd.Series([0])
sharpe = float(pnl_series.mean() / (pnl_series.std() + 1e-10) * np.sqrt(5040)) if len(pnl_series) > 1 else 0

cum_pnl = pnl_series.cumsum()
peak = cum_pnl.cummax()
max_drawdown = float((cum_pnl - peak).min())

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Total Trades", f"{total_trades:,}", help="All simulated trades")
col2.metric("Win Rate", f"{win_rate:.1%}", delta=f"{win_rate - 0.5:.1%} vs 50%")
col3.metric("Total PnL", f"${total_pnl:,.0f}", delta=f"${avg_pnl:.2f} avg")
col4.metric("Sharpe Ratio", f"{sharpe:.2f}", delta="annualized")
col5.metric("Max Drawdown", f"${max_drawdown:,.0f}")

st.divider()

# ------------------------------------------------------------------ #
# Charts Row 1: Equity + Radar
# ------------------------------------------------------------------ #
c1, c2 = st.columns([3, 2])

with c1:
    st.markdown("### Equity Curves by Strategy")
    import plotly.io as pio
    chart_path_equity = Path("data/outputs/charts/strategy_equity_curves.json")
    if chart_path_equity.exists():
        fig_equity = pio.read_json(str(chart_path_equity))
    elif trade_log is not None and not trade_log.empty:
        fig_equity = equity_curve_chart(trade_log, "All Strategy Equity Curves")
    else:
        fig_equity = None
        st.info("No trade data available for equity curves.")
        
    if fig_equity:
        st.plotly_chart(fig_equity, use_container_width=True)

with c2:
    st.markdown("### Strategy Radar")
    chart_path_radar = Path("data/outputs/charts/strategy_radar.json")
    if chart_path_radar.exists():
        fig_radar = pio.read_json(str(chart_path_radar))
    elif strategy_summary is not None and not strategy_summary.empty:
        fig_radar = strategy_radar_chart(strategy_summary, "Strategy Comparison")
    else:
        fig_radar = None
        st.info("Strategy comparison radar not available")
    if fig_radar:
        st.plotly_chart(fig_radar, use_container_width=True)

# ------------------------------------------------------------------ #
# Charts Row 2: Rolling Metrics + Strategy Table
# ------------------------------------------------------------------ #
c3, c4 = st.columns([2, 3])

with c3:
    st.markdown("### Rolling Performance")
    chart_path_rolling = Path("data/outputs/charts/rolling_performance.json")
    if chart_path_rolling.exists():
        fig_rolling = pio.read_json(str(chart_path_rolling))
    elif trade_log is not None and not trade_log.empty:
        fig_rolling = rolling_metrics_chart(trade_log, window=50)
    else:
        fig_rolling = None
        st.info("No trade data available for rolling performance.")
        
    if fig_rolling:
        st.plotly_chart(fig_rolling, use_container_width=True)

with c4:
    st.markdown("### Strategy Performance Leaderboard")
    if strategy_summary is not None and not strategy_summary.empty:
        display_cols = ["strategy_id", "total_trades", "win_rate", "total_pnl_usd",
                        "avg_pnl_usd", "sharpe", "max_drawdown"]
        display_cols = [c for c in display_cols if c in strategy_summary.columns]

        styled = strategy_summary[display_cols].copy()
        if "win_rate" in styled.columns:
            styled["win_rate"] = styled["win_rate"].map("{:.1%}".format)
        if "total_pnl_usd" in styled.columns:
            styled["total_pnl_usd"] = styled["total_pnl_usd"].map("${:,.0f}".format)
        if "avg_pnl_usd" in styled.columns:
            styled["avg_pnl_usd"] = styled["avg_pnl_usd"].map("${:.2f}".format)
        if "sharpe" in styled.columns:
            styled["sharpe"] = styled["sharpe"].map("{:.3f}".format)

        st.dataframe(styled, use_container_width=True, hide_index=True)
    else:
        st.info("Strategy summary not available")

# ------------------------------------------------------------------ #
# ML Model Summary
# ------------------------------------------------------------------ #
if training_results:
    st.divider()
    st.markdown("### 🤖 ML Model Summary")
    metrics = training_results.get("overall_metrics", {})
    mc1, mc2, mc3, mc4, mc5 = st.columns(5)
    mc1.metric("MAE", f"{metrics.get('mae', 0):.4f}")
    mc2.metric("RMSE", f"{metrics.get('rmse', 0):.4f}")
    mc3.metric("R²", f"{metrics.get('r2', 0):.4f}")
    mc4.metric("Model Sharpe", f"{metrics.get('sharpe', 0):.2f}")
    mc5.metric("Model Win Rate", f"{metrics.get('win_rate', 0):.1%}")
