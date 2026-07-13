"""
QuantTrade ML Pipeline — Strategy Recommendation Page
AI-powered strategy ranking and regime-conditional recommendations.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.visualization.themes import get_css, COLORS, STRATEGY_COLORS
from src.visualization.charts import strategy_radar_chart, equity_curve_chart

st.set_page_config(page_title="Strategy Recommendations | QuantTrade", page_icon="🏆", layout="wide")
st.markdown(get_css(), unsafe_allow_html=True)

from src.dashboard import ensure_dashboard_state, render_status_card

ensure_dashboard_state()

st.markdown("## 🏆 Strategy Recommendation Engine")
st.markdown("AI-powered strategy ranking based on historical performance and market regime analytics.")
st.divider()

if st.session_state.get("strategy_summary", pd.DataFrame()).empty:
    render_status_card(
        "Strategy Analytics Unavailable",
        "No strategy summary data is available for ranking and recommendation analysis.",
        details="When trade simulation outputs exist, this page will rank strategies and provide regime-aware recommendations.",
        icon="⚠️",
    )
    st.stop()

strategy_summary = st.session_state.strategy_summary
trade_log = st.session_state.get("trade_log")
feature_df = st.session_state.get("feature_df")

# ------------------------------------------------------------------ #
# Market Regime Detection
# ------------------------------------------------------------------ #
st.markdown("### 📊 Current Market Regime")

regime = "Unknown"
regime_color = COLORS["text_muted"]
regime_description = ""

if feature_df is not None:
    recent = feature_df.tail(24)
    adx_val = recent["adx"].mean() if "adx" in recent.columns else 20
    vol_val = recent["realized_vol_24h"].mean() if "realized_vol_24h" in recent.columns else 0.001
    rsi_val = recent["rsi_14"].mean() if "rsi_14" in recent.columns else 50

    if adx_val > 30:
        regime = "Trending"
        regime_color = COLORS["neon_green"]
        regime_description = "Strong directional movement. Favor trend-following strategies."
        recommended_strategies = ["trend_following", "momentum", "ma_crossover"]
    elif adx_val > 20:
        regime = "Weak Trend"
        regime_color = COLORS["gold"]
        regime_description = "Moderate trend. Mixed strategy approach recommended."
        recommended_strategies = ["momentum", "bollinger", "breakout"]
    else:
        regime = "Ranging"
        regime_color = COLORS["electric_blue"]
        regime_description = "Low ADX, mean-reverting market. Favor oscillator strategies."
        recommended_strategies = ["rsi_reversion", "bollinger", "support_resistance"]

    st.markdown(f"""
    <div style="
        background: rgba(0,0,0,0.2);
        border: 1px solid {regime_color}40;
        border-left: 4px solid {regime_color};
        border-radius: 10px;
        padding: 1rem 1.5rem;
        margin-bottom: 1rem;
    ">
        <span style="color: {regime_color}; font-size: 1.2rem; font-weight: 700;">
            📈 {regime} Market
        </span>
        <p style="color: #90a4ae; margin: 0.3rem 0 0; font-size: 0.9rem;">{regime_description}</p>
        <div style="margin-top: 0.5rem;">
            <span style="color: #546e7a; font-size: 0.8rem;">
                ADX: <strong style="color: {regime_color};">{adx_val:.1f}</strong> |
                Volatility: <strong style="color: {regime_color};">{vol_val*10000:.1f} pips/h</strong> |
                RSI: <strong style="color: {regime_color};">{rsi_val:.1f}</strong>
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)
else:
    st.info("Load features to detect market regime")
    recommended_strategies = list(strategy_summary["strategy_id"]) if not strategy_summary.empty else []

# ------------------------------------------------------------------ #
# Strategy Rankings
# ------------------------------------------------------------------ #
st.divider()
st.markdown("### 🏅 Strategy Leaderboard (Risk-Adjusted)")

# Compute composite ranking score
ranked = strategy_summary.copy()
if not ranked.empty:
    for col in ["sharpe", "win_rate", "total_pnl_usd", "profit_factor"]:
        if col in ranked.columns:
            col_min = ranked[col].min()
            col_max = ranked[col].max()
            ranked[f"{col}_norm"] = (ranked[col] - col_min) / (col_max - col_min + 1e-10)

    # Weighted composite score
    score_cols = [c for c in ["sharpe_norm", "win_rate_norm", "total_pnl_usd_norm", "profit_factor_norm"]
                  if c in ranked.columns]
    if score_cols:
        ranked["composite_score"] = ranked[score_cols].mean(axis=1)
        ranked = ranked.sort_values("composite_score", ascending=False)
        ranked["rank"] = range(1, len(ranked) + 1)

    # Regime recommendation
    ranked["regime_match"] = ranked["strategy_id"].isin(recommended_strategies).map({True: "✅ Recommended", False: "—"})

    # Display table
    display_cols = ["rank", "strategy_id", "composite_score", "win_rate", "sharpe",
                    "total_pnl_usd", "profit_factor", "regime_match"]
    display_cols = [c for c in display_cols if c in ranked.columns]

    styled = ranked[display_cols].copy()
    if "composite_score" in styled.columns:
        styled["composite_score"] = styled["composite_score"].map("{:.3f}".format)
    if "win_rate" in styled.columns:
        styled["win_rate"] = styled["win_rate"].map("{:.1%}".format)
    if "total_pnl_usd" in styled.columns:
        styled["total_pnl_usd"] = styled["total_pnl_usd"].map("${:,.0f}".format)
    if "sharpe" in styled.columns:
        styled["sharpe"] = styled["sharpe"].map("{:.3f}".format)

    st.dataframe(styled, use_container_width=True, hide_index=True)

# ------------------------------------------------------------------ #
# Radar + Bar Chart
# ------------------------------------------------------------------ #
c1, c2 = st.columns(2)

with c1:
    st.markdown("### Multi-Metric Strategy Radar")
    import plotly.io as pio
    chart_path = Path("data/outputs/charts/strategy_radar.json")
    if chart_path.exists():
        fig_radar = pio.read_json(str(chart_path))
    else:
        fig_radar = strategy_radar_chart(strategy_summary)
    st.plotly_chart(fig_radar, use_container_width=True)

with c2:
    st.markdown("### Sharpe Ratio Comparison")
    if "sharpe" in strategy_summary.columns:
        strat_sharpe = strategy_summary[["strategy_id", "sharpe"]].sort_values("sharpe", ascending=True)
        colors = [COLORS["neon_green"] if s > 0 else COLORS["coral"] for s in strat_sharpe["sharpe"]]
        fig_sharpe = go.Figure(go.Bar(
            x=strat_sharpe["sharpe"],
            y=strat_sharpe["strategy_id"],
            orientation="h",
            marker_color=colors,
        ))
        fig_sharpe.update_layout(
            template="quanttrade", height=350,
            xaxis_title="Sharpe Ratio",
        )
        st.plotly_chart(fig_sharpe, use_container_width=True)

# ------------------------------------------------------------------ #
# Top Strategy Deep Dive
# ------------------------------------------------------------------ #
if not strategy_summary.empty and trade_log is not None:
    st.divider()
    top_strategy = strategy_summary.iloc[0]["strategy_id"]
    st.markdown(f"### 🥇 Deep Dive: {top_strategy.upper()}")

    top_trades = trade_log[trade_log["strategy_id"] == top_strategy]
    if not top_trades.empty:
        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("Trades", f"{len(top_trades):,}")
        mc2.metric("Win Rate", f"{top_trades['win'].mean():.1%}")
        mc3.metric("Total PnL", f"${top_trades['pnl_usd'].sum():,.0f}")
        mc4.metric("Avg Hold", f"{top_trades['holding_bars'].mean():.0f}h")

        fig_top = equity_curve_chart(top_trades, f"{top_strategy} — Equity Curve")
        st.plotly_chart(fig_top, use_container_width=True)
