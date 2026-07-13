"""
QuantTrade ML Pipeline — Advanced Analytics Page
Monthly heatmap, session analysis, rolling performance, drawdown.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.visualization.themes import get_css, COLORS
from src.visualization.charts import (
    monthly_returns_heatmap, drawdown_chart, rolling_metrics_chart,
    session_heatmap, return_distribution_chart
)

st.set_page_config(page_title="Analytics | QuantTrade", page_icon="🔬", layout="wide")
st.markdown(get_css(), unsafe_allow_html=True)

from src.dashboard import ensure_dashboard_state, render_status_card

ensure_dashboard_state()

st.markdown("## 🔬 Advanced Analytics")
st.markdown("Deep statistical analysis across market returns, trading sessions, and drawdown behavior.")
st.divider()

if st.session_state.get("trade_log", pd.DataFrame()).empty:
    render_status_card(
        "Analytics Data Missing",
        "Trade simulation outputs are required for advanced analytics.",
        details="This page will show monthly return heatmaps, drawnown analysis, and session performance once trade logs are available.",
        icon="⚠️",
    )
    st.stop()

trade_log = st.session_state.trade_log
feature_df = st.session_state.get("feature_df")

# ------------------------------------------------------------------ #
# Monthly Returns Heatmap
# ------------------------------------------------------------------ #
st.markdown("### 📅 Monthly Returns Heatmap")
strategy_select = st.selectbox(
    "Strategy",
    ["All"] + sorted(trade_log["strategy_id"].unique().tolist()),
)
if strategy_select == "All":
    log_filtered = trade_log
else:
    log_filtered = trade_log[trade_log["strategy_id"] == strategy_select]

fig_monthly = monthly_returns_heatmap(log_filtered, f"Monthly Returns — {strategy_select}")
st.plotly_chart(fig_monthly, use_container_width=True)

st.divider()

# ------------------------------------------------------------------ #
# Rolling Metrics + Drawdown
# ------------------------------------------------------------------ #
c1, c2 = st.columns(2)

with c1:
    st.markdown("### Rolling Sharpe & Win Rate")
    window = st.slider("Rolling Window", 20, 200, 50, 10)
    import plotly.io as pio
    chart_path_rolling = Path("data/outputs/charts/rolling_performance.json")
    if chart_path_rolling.exists() and strategy_select == "All" and window == 50:
        fig_rolling = pio.read_json(str(chart_path_rolling))
    else:
        fig_rolling = rolling_metrics_chart(log_filtered, window=window)
    st.plotly_chart(fig_rolling, use_container_width=True)

with c2:
    st.markdown("### Drawdown Analysis")
    chart_path_dd = Path("data/outputs/charts/portfolio_drawdown.json")
    if chart_path_dd.exists() and strategy_select == "All":
        fig_dd = pio.read_json(str(chart_path_dd))
    else:
        pnl_series = log_filtered.sort_values("exit_time")["pnl_usd"]
        pnl_indexed = pnl_series.reset_index(drop=True)
        fig_dd = drawdown_chart(pnl_indexed, "Portfolio Drawdown")
    st.plotly_chart(fig_dd, use_container_width=True)

st.divider()

# ------------------------------------------------------------------ #
# Session Performance Analysis
# ------------------------------------------------------------------ #
st.markdown("### ⏰ Trading Session Performance")
if "entry_time" in log_filtered.columns:
    df_sess = log_filtered.copy()
    df_sess["entry_time"] = pd.to_datetime(df_sess["entry_time"])
    df_sess["hour"] = df_sess["entry_time"].dt.hour

    def assign_session(hour):
        if 22 <= hour or hour < 8:
            return "Asian"
        elif 8 <= hour < 13:
            return "London"
        elif 13 <= hour < 17:
            return "London+NY"
        else:
            return "New York"

    df_sess["session"] = df_sess["hour"].apply(assign_session)
    sess_perf = df_sess.groupby("session").agg(
        trades=("pnl_usd", "count"),
        win_rate=("win", "mean"),
        total_pnl=("pnl_usd", "sum"),
        avg_pnl=("pnl_usd", "mean"),
        sharpe=("pnl_usd", lambda x: x.mean() / (x.std() + 1e-10) * np.sqrt(5040)),
    ).reset_index()

    c3, c4 = st.columns(2)
    with c3:
        st.dataframe(sess_perf.round(4), use_container_width=True, hide_index=True)
    with c4:
        import plotly.express as px
        fig_sess = px.bar(
            sess_perf, x="session", y="avg_pnl",
            color="win_rate",
            color_continuous_scale=["#ff4444", "#ffd700", "#00ff88"],
            title="Avg PnL by Session",
        )
        fig_sess.update_layout(template="quanttrade", height=320)
        st.plotly_chart(fig_sess, use_container_width=True)

st.divider()

# ------------------------------------------------------------------ #
# Macro Event Impact Analysis
# ------------------------------------------------------------------ #
if "nearby_event_impact_score" in log_filtered.columns:
    st.markdown("### 🌍 Macro Event Impact on Trade Performance")

    macro_impact = log_filtered.groupby(
        log_filtered["nearby_event_impact_score"].fillna(0).astype(int)
    ).agg(
        trades=("pnl_usd", "count"),
        win_rate=("win", "mean"),
        avg_pnl=("pnl_usd", "mean"),
    ).reset_index()
    macro_impact.columns = ["Impact Score", "Trades", "Win Rate", "Avg PnL"]

    st.dataframe(macro_impact, use_container_width=True, hide_index=True)

# ------------------------------------------------------------------ #
# Holding Time Analysis
# ------------------------------------------------------------------ #
st.divider()
st.markdown("### ⏳ Holding Time vs PnL")
if "holding_bars" in log_filtered.columns:
    import plotly.graph_objects as go
    fig_hold = go.Figure(go.Scatter(
        x=log_filtered["holding_bars"],
        y=log_filtered["pnl_usd"],
        mode="markers",
        marker=dict(
            color=log_filtered["pnl_usd"],
            colorscale=["#ff4444", "#1a2035", "#00ff88"],
            size=4,
            opacity=0.5,
            cmid=0,
        ),
        text=log_filtered["strategy_id"],
    ))
    fig_hold.add_hline(y=0, line_dash="dash", line_color=COLORS["coral"])
    fig_hold.update_layout(
        template="quanttrade", height=400,
        title="Holding Bars vs Trade PnL",
        xaxis_title="Holding Bars", yaxis_title="PnL (USD)",
    )
    st.plotly_chart(fig_hold, use_container_width=True)
