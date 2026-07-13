"""
QuantTrade ML Pipeline — Trade Simulation Page
Strategy execution, equity curves, trade statistics.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.dashboard import ensure_dashboard_state, render_status_card
from src.simulation.engine import run_simulation, STRATEGY_REGISTRY
from src.visualization.themes import get_css, COLORS, STRATEGY_COLORS
from src.visualization.charts import equity_curve_chart, return_distribution_chart, drawdown_chart

st.set_page_config(page_title="Trade Simulation | QuantTrade", page_icon="⚡", layout="wide")
st.markdown(get_css(), unsafe_allow_html=True)

ensure_dashboard_state()

st.markdown("## ⚡ Trade Simulation Engine")
st.markdown("Realistic trade simulation analytics across strategy equity, returns, and trade behavior.")
st.divider()

if st.session_state.get("feature_df", pd.DataFrame()).empty:
    render_status_card(
        "Features Missing",
        "Engineered features are required for advanced trade simulation analytics.",
        details="Once the backend pipeline produces the feature matrix, this page will enable strategy equity curves, exit reason analysis, and trade distributions.",
        icon="⚠️",
    )

if st.session_state.get("trade_log", pd.DataFrame()).empty:
    render_status_card(
        "Trade Simulation Data Missing",
        "Simulated trade records are not available yet.",
        details="This page will visualize trades, strategy leaderboards, and holding time distributions when the backend simulation stage produces trade logs.",
        icon="⚠️",
    )
    st.stop()

trade_log = st.session_state.trade_log
strategy_summary = st.session_state.strategy_summary

# ------------------------------------------------------------------ #
# KPI Row
# ------------------------------------------------------------------ #
total_trades = len(trade_log)
win_rate = trade_log["win"].mean()
total_pnl = trade_log["pnl_usd"].sum()
best_strategy = strategy_summary.iloc[0]["strategy_id"] if not strategy_summary.empty else "N/A"

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Total Trades", f"{total_trades:,}")
m2.metric("Win Rate", f"{win_rate:.1%}", delta=f"{win_rate - 0.5:.1%}")
m3.metric("Total PnL", f"${total_pnl:,.0f}")
m4.metric("Avg PnL / Trade", f"${trade_log['pnl_usd'].mean():.2f}")
m5.metric("Best Strategy", best_strategy)

st.divider()

# ------------------------------------------------------------------ #
# Strategy Filter + Equity Curve
# ------------------------------------------------------------------ #
strategy_filter = st.multiselect(
    "Filter Strategies",
    options=trade_log["strategy_id"].unique().tolist(),
    default=trade_log["strategy_id"].unique().tolist(),
)
filtered_trades = trade_log[trade_log["strategy_id"].isin(strategy_filter)]

c1, c2 = st.columns([3, 2])
with c1:
    st.markdown("### Strategy Equity Curves")
    import plotly.io as pio
    chart_path_equity = Path("data/outputs/charts/strategy_equity_curves.json")
    if chart_path_equity.exists() and len(strategy_filter) == len(trade_log["strategy_id"].unique()):
        fig_equity = pio.read_json(str(chart_path_equity))
    else:
        fig_equity = equity_curve_chart(filtered_trades)
    st.plotly_chart(fig_equity, use_container_width=True)

with c2:
    st.markdown("### Trade Exit Reasons")
    if "exit_reason" in trade_log.columns:
        exit_counts = filtered_trades["exit_reason"].value_counts()
        fig_exit = px.pie(
            values=exit_counts.values,
            names=exit_counts.index,
            color_discrete_sequence=[COLORS["neon_green"], COLORS["coral"], COLORS["gold"], COLORS["purple"]],
        )
        fig_exit.update_layout(template="quanttrade", height=300)
        st.plotly_chart(fig_exit, use_container_width=True)

# ------------------------------------------------------------------ #
# Strategy Performance Table
# ------------------------------------------------------------------ #
st.divider()
st.markdown("### Strategy Leaderboard")
if not strategy_summary.empty:
    display_cols = ["strategy_id", "total_trades", "win_rate", "total_pnl_usd",
                    "avg_pnl_usd", "sharpe", "max_drawdown", "avg_holding_bars"]
    display_cols = [c for c in display_cols if c in strategy_summary.columns]
    st.dataframe(strategy_summary[display_cols], use_container_width=True, hide_index=True)

# ------------------------------------------------------------------ #
# Return Distribution + Drawdown
# ------------------------------------------------------------------ #
c3, c4 = st.columns(2)
with c3:
    st.markdown("### PnL Distribution")
    fig_dist = return_distribution_chart(filtered_trades["pnl_usd"], "Trade PnL Distribution")
    st.plotly_chart(fig_dist, use_container_width=True)

with c4:
    st.markdown("### Holding Time Distribution")
    if "holding_bars" in filtered_trades.columns:
        fig_hold = px.histogram(
            filtered_trades, x="holding_bars", nbins=50,
            color="strategy_id",
            color_discrete_map=STRATEGY_COLORS,
        )
        fig_hold.update_layout(
            template="quanttrade", height=350,
            xaxis_title="Holding Bars (hours)", yaxis_title="Count",
        )
        st.plotly_chart(fig_hold, use_container_width=True)

# ------------------------------------------------------------------ #
# Trade Log Table
# ------------------------------------------------------------------ #
st.divider()
with st.expander("📋 Trade Log (Recent 500 trades)"):
    display_cols = ["strategy_id", "direction", "entry_time", "exit_time",
                    "entry_price", "exit_price", "pnl_usd", "win", "holding_bars", "exit_reason"]
    display_cols = [c for c in display_cols if c in filtered_trades.columns]
    st.dataframe(
        filtered_trades[display_cols].tail(500).sort_values("exit_time", ascending=False),
        use_container_width=True, hide_index=True,
    )
