"""
QuantTrade ML Pipeline — Streamlit Application
Production-grade quantitative analytics dashboard.
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Page config — MUST be first Streamlit call
st.set_page_config(
    page_title="QuantTrade ML | Quantitative Intelligence Platform",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": None,
        "Report a bug": None,
        "About": "QuantTrade ML Pipeline — CrowdWisdomTrading Assessment",
    },
)

from config.logging_config import setup_logging
from config.settings import settings
from src.visualization.themes import get_css
from loguru import logger
from src.dashboard import ensure_dashboard_state

# Apply global CSS
st.markdown(get_css(), unsafe_allow_html=True)

# Initialize logging
setup_logging(settings.log_level, settings.log_format, settings.log_path)

# ------------------------------------------------------------------ #
from src.database.repository import get_repos, get_db
from src.database.models import MacroEvent, Trade
import pandas as pd

ensure_dashboard_state()
pipeline_ready = st.session_state.pipeline_complete


# ------------------------------------------------------------------ #
# Sidebar Navigation
# ------------------------------------------------------------------ #
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 1rem 0 1.5rem;">
        <div style="font-size: 2.5rem;">📈</div>
        <h1 style="
            background: linear-gradient(135deg, #00d4ff, #00ff88);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 1.3rem;
            font-weight: 800;
            margin: 0;
            letter-spacing: -0.02em;
        ">QuantTrade ML</h1>
        <p style="color: #546e7a; font-size: 0.7rem; margin: 4px 0 0; letter-spacing: 0.08em; text-transform: uppercase;">
            Quantitative Intelligence Platform
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Pipeline Status
    steps = st.session_state.pipeline_steps
    statuses = {
        "Data Loaded": steps["data_loaded"],
        "Macro Scraped": steps["macro_scraped"],
        "Features Built": steps["features_built"],
        "Simulation Run": steps["simulation_run"],
        "Model Trained": steps["model_trained"],
    }
    complete = sum(statuses.values())
    total = len(statuses)

    st.markdown("### 🔄 Pipeline Status")
    progress_pct = complete / total
    st.progress(progress_pct)
    st.markdown(f"""
    <p style="color: #546e7a; font-size: 0.75rem; text-align: center; margin-top: 4px;">
        {complete}/{total} steps complete
    </p>
    """, unsafe_allow_html=True)

    for step_name, done in statuses.items():
        icon = "✅" if done else "⬜"
        color = "#00ff88" if done else "#546e7a"
        st.markdown(
            f'<span style="color: {color}; font-size: 0.8rem;">{icon} {step_name}</span>',
            unsafe_allow_html=True,
        )

    st.divider()
    st.markdown("### 📊 Quick Stats")

    if st.session_state.forex_df is not None:
        n_rows = len(st.session_state.forex_df)
        st.markdown(f'<span style="color: #90a4ae; font-size: 0.8rem;">📉 Market bars: **{n_rows:,}**</span>', unsafe_allow_html=True)

    if st.session_state.macro_df is not None:
        n_events = len(st.session_state.macro_df)
        st.markdown(f'<span style="color: #90a4ae; font-size: 0.8rem;">🌍 Macro events: **{n_events:,}**</span>', unsafe_allow_html=True)

    if st.session_state.trade_log is not None and not st.session_state.trade_log.empty:
        n_trades = len(st.session_state.trade_log)
        win_rate = st.session_state.trade_log["win"].mean() if "win" in st.session_state.trade_log.columns else 0
        st.markdown(f'<span style="color: #90a4ae; font-size: 0.8rem;">⚡ Trades: **{n_trades:,}** | WR: **{win_rate:.1%}**</span>', unsafe_allow_html=True)

    st.divider()
    st.markdown(
        '<p style="color: #2d3748; font-size: 0.65rem; text-align: center;">CrowdWisdomTrading Assessment</p>',
        unsafe_allow_html=True,
    )


# ------------------------------------------------------------------ #
# Main Landing Page
# ------------------------------------------------------------------ #

st.markdown(
    """
    <div style="
        background: linear-gradient(135deg, rgba(0,212,255,0.05), rgba(155,89,182,0.05));
        border: 1px solid rgba(0,212,255,0.1);
        border-radius: 16px;
        padding: 2.5rem;
        margin-bottom: 2rem;
        position: relative;
        overflow: hidden;
    ">
        <div style="position: relative; z-index: 1;">
            <h1 style="
                background: linear-gradient(135deg, #00d4ff, #00ff88, #ffd700);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                font-size: 2.5rem;
                font-weight: 900;
                margin: 0;
                letter-spacing: -0.03em;
            ">QuantTrade ML Pipeline</h1>
            <p style="color: #90a4ae; font-size: 1.1rem; margin: 0.5rem 0 0; font-weight: 300;">
                Production-Grade Quantitative Trading Intelligence Platform
            </p>
            <div style="display: flex; gap: 10px; margin-top: 1rem; flex-wrap: wrap;">
                <span class="status-badge badge-blue">XGBoost ML</span>
                <span class="status-badge badge-blue">Walk-Forward Validation</span>
                <span class="status-badge badge-blue">7 Strategies</span>
                <span class="status-badge badge-blue">SHAP Explainability</span>
                <span class="status-badge badge-blue">EUR/USD</span>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Quick-start guide
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    ### 🚀 Getting Started
    The ML pipeline outputs are pre-rendered and loaded into the dashboard.
    
    **Workflow overview:**
    1. **Overview**: Monitor overall system state.
    2. **Analysis**: Inspect strategy metrics and model evaluation.
    3. **Inference**: Review predicted PnL for active positions.
    4. **Download**: Retrieve files on the Downloads page.
    """)

with col2:
    st.markdown("""
    ### 📋 Platform Features
    - **93,000+** hourly candles (2005–2020)
    - **60+** engineered features
    - **7** trading strategies
    - **Walk-forward** validation with embargo
    - **Optuna** hyperparameter tuning
    - **SHAP** explainability
    - **12+** interactive Plotly charts
    """)

with col3:
    st.markdown("""
    ### 🎯 ML Pipeline
    - **Target**: Expected Trade PnL
    - **Model**: XGBoost Regressor
    - **Validation**: Walk-Forward (90/30 days)
    - **Tuning**: 50-trial Optuna TPE
    - **Metrics**: MAE, RMSE, R², Sharpe, Sortino
    - **Output**: Strategy recommendations
    """)

st.divider()

# Navigation cards
st.markdown("### Navigate to Pages")
pages = [
    ("📊 Overview", "Aggregated KPIs and equity curves", "01"),
    ("📈 Market Data", "EUR/USD price exploration", "02"),
    ("🌍 Macro Events", "Economic calendar & impact", "03"),
    ("🔧 Feature Engineering", "60+ feature inspection", "04"),
    ("⚡ Trade Simulation", "7 strategy live backtest", "05"),
    ("🤖 Model Training", "XGBoost walk-forward training", "06"),
    ("📋 Evaluation", "Regression & trading metrics", "07"),
    ("🔮 Predictions", "Real-time PnL predictions", "08"),
    ("🏆 Recommendations", "AI strategy ranking", "09"),
    ("🔬 Analytics", "Advanced statistical analysis", "10"),
    ("⚙️ Settings", "Configuration & controls", "11"),
]

for i in range(0, len(pages), 4):
    cols = st.columns(4)
    for j, col in enumerate(cols):
        if i + j < len(pages):
            icon_name, desc, num = pages[i + j]
            with col:
                st.markdown(f"""
                <div class="metric-card" style="text-align: center; cursor: pointer; min-height: 100px;">
                    <div style="font-size: 1.5rem; margin-bottom: 8px;">{icon_name.split()[0]}</div>
                    <div style="color: #e8eaf6; font-weight: 600; font-size: 0.9rem;">{' '.join(icon_name.split()[1:])}</div>
                    <div style="color: #546e7a; font-size: 0.75rem; margin-top: 4px;">{desc}</div>
                </div>
                """, unsafe_allow_html=True)
