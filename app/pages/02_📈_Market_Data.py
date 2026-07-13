"""
QuantTrade ML Pipeline — Market Data Page
EUR/USD price exploration, candlestick charts, spread analysis.
"""
import sys
from pathlib import Path
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import settings
from src.dashboard import ensure_dashboard_state, render_status_card
from src.ingestion.forex_loader import load_forex_data
from src.preprocessing.cleaner import clean_forex_data
from src.preprocessing.validator import validate_forex_data
from src.visualization.themes import get_css, COLORS
from src.visualization.charts import candlestick_chart, session_heatmap, return_distribution_chart

st.set_page_config(page_title="Market Data | QuantTrade", page_icon="📈", layout="wide")
st.markdown(get_css(), unsafe_allow_html=True)

ensure_dashboard_state()

st.markdown("## 📈 Market Data — EUR/USD")
st.markdown("Explore and validate the historical EUR/USD hourly dataset.")
st.divider()

df = st.session_state.get("forex_df", pd.DataFrame())

if df.empty:
    render_status_card(
        "Market Data Unavailable",
        "EUR/USD market data is not currently available for visualization.",
        details="This page will render hourly price charts, session heatmaps, and validation diagnostics once cleaned forex data exists in the backend outputs.",
        icon="⚠️",
    )
    st.stop()

if "validation_report" not in st.session_state:
    with st.spinner("Validating market data..."):
        validation = validate_forex_data(df)
        st.session_state.validation_report = validation.to_dict()

df = st.session_state.forex_df

# ------------------------------------------------------------------ #
# Data Overview Metrics
# ------------------------------------------------------------------ #
st.markdown("### 📋 Dataset Overview")
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Total Bars", f"{len(df):,}")
m2.metric("Date Range", f"{df.index[0].year}–{df.index[-1].year}")
m3.metric("Spread (avg)", f"{df['spread_pips'].mean():.2f} pips" if "spread_pips" in df.columns else "N/A")
m4.metric("Outlier Bars", f"{df['is_outlier'].sum():,}" if "is_outlier" in df.columns else "0")
m5.metric("Quality Score", f"{df['data_quality_score'].mean():.3f}" if "data_quality_score" in df.columns else "N/A")

st.divider()

# ------------------------------------------------------------------ #
# Chart Controls
# ------------------------------------------------------------------ #
with st.expander("🎛️ Chart Controls", expanded=True):
    c1, c2, c3 = st.columns(3)
    with c1:
        n_bars = st.slider("Bars to Display", 100, 5000, 1000, 100)
    with c2:
        indicators = st.multiselect(
            "Indicator Overlays",
            ["ema_21", "ema_50", "ema_200", "sma_20", "bb_upper", "bb_lower", "bb_middle"],
            default=["ema_21", "ema_200"],
        )
    with c3:
        year_filter = st.selectbox(
            "Year",
            ["All"] + sorted(df.index.year.unique().tolist()),
        )

import plotly.io as pio
chart_path_candlestick = Path("data/outputs/charts/market_candlestick.json")

if year_filter != "All":
    df_chart = df[df.index.year == int(year_filter)]
else:
    df_chart = df

if chart_path_candlestick.exists() and set(indicators) == {"ema_21", "ema_200"} and n_bars == 1000 and year_filter == "All":
    fig_candlestick = pio.read_json(str(chart_path_candlestick))
else:
    fig_candlestick = candlestick_chart(df_chart, indicators=indicators, n_bars=n_bars)

st.plotly_chart(fig_candlestick, use_container_width=True)

# ------------------------------------------------------------------ #
# Session Heatmap + Return Distribution
# ------------------------------------------------------------------ #
c3, c4 = st.columns(2)
with c3:
    st.markdown("### Session Return Heatmap")
    chart_path_session = Path("data/outputs/charts/market_session_heatmap.json")
    if chart_path_session.exists():
        fig_session = pio.read_json(str(chart_path_session))
    else:
        if "return_1h" not in df.columns and "mid_close" in df.columns:
            df["return_1h"] = df["mid_close"].pct_change()
        fig_session = session_heatmap(df)
    st.plotly_chart(fig_session, use_container_width=True)

with c4:
    st.markdown("### Return Distribution")
    chart_path_returns = Path("data/outputs/charts/market_returns_dist.json")
    if chart_path_returns.exists():
        fig_returns = pio.read_json(str(chart_path_returns))
    else:
        if "return_1h" not in df.columns and "mid_close" in df.columns:
            df["return_1h"] = df["mid_close"].pct_change()
        returns_series = df["return_1h"].dropna() * 10000  # Convert to pips
        fig_returns = return_distribution_chart(returns_series, "Hourly Return Distribution (pips)")
    st.plotly_chart(fig_returns, use_container_width=True)

# ------------------------------------------------------------------ #
# Validation Report
# ------------------------------------------------------------------ #
st.divider()
st.markdown("### ✅ Data Validation Report")
if "validation_report" in st.session_state:
    report = st.session_state.validation_report
    if report["passed"]:
        st.success(f"All critical validation checks passed! ({report['warning_count']} warnings)")
    else:
        st.error(f"Validation failed! {report['error_count']} errors, {report['warning_count']} warnings")

    checks_df = pd.DataFrame(report["checks"])
    if not checks_df.empty:
        checks_df["status"] = checks_df["passed"].map({True: "✅", False: "❌"})
        st.dataframe(
            checks_df[["status", "name", "message", "severity"]].rename(columns={
                "status": "Status", "name": "Check", "message": "Result", "severity": "Severity"
            }),
            use_container_width=True, hide_index=True,
        )

# ------------------------------------------------------------------ #
# Raw Data Sample
# ------------------------------------------------------------------ #
with st.expander("📄 Raw Data Sample (first 100 rows)"):
    display_cols = [c for c in ["mid_open", "mid_high", "mid_low", "mid_close",
                                "spread_pips", "is_outlier", "data_quality_score"] if c in df.columns]
    st.dataframe(df[display_cols].head(100), use_container_width=True)
