"""
QuantTrade ML Pipeline — Settings Page
Configuration, API keys, pipeline controls, log viewer.
"""
import sys
from pathlib import Path
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import settings
from src.visualization.themes import get_css

st.set_page_config(page_title="Settings | QuantTrade", page_icon="⚙️", layout="wide")
st.markdown(get_css(), unsafe_allow_html=True)

st.markdown("## ⚙️ Settings & Configuration")
st.markdown("Configure API keys, pipeline parameters, and system controls.")
st.divider()

# ------------------------------------------------------------------ #
# API Keys
# ------------------------------------------------------------------ #
st.markdown("### 🔑 API Keys")
with st.form("api_keys_form"):
    apify_key = st.text_input(
        "Apify API Key",
        value=settings.apify_api_key or "",
        type="password",
        help="Used for scraping macroeconomic events from economic calendars",
    )
    submitted = st.form_submit_button("💾 Save API Key")
    if submitted:
        if apify_key:
            env_path = Path(settings.project_root) / ".env"
            content = ""
            if env_path.exists():
                with open(env_path) as f:
                    content = f.read()
            if "APIFY_API_KEY=" in content:
                lines = [l if not l.startswith("APIFY_API_KEY=") else f"APIFY_API_KEY={apify_key}" for l in content.splitlines()]
                content = "\n".join(lines)
            else:
                content += f"\nAPAIFY_API_KEY={apify_key}"
            with open(env_path, "w") as f:
                f.write(content)
            st.success("✅ API key saved to .env")

st.divider()

# ------------------------------------------------------------------ #
# Walk-Forward Parameters
# ------------------------------------------------------------------ #
st.markdown("### 🔄 Walk-Forward Parameters")
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Train Days", settings.wf_train_days)
with c2:
    st.metric("Test Days", settings.wf_test_days)
with c3:
    st.metric("Step Days", settings.wf_step_days)
with c4:
    st.metric("Embargo Days", settings.wf_embargo_days)

st.caption("Modify these in your `.env` file: `WF_TRAIN_DAYS`, `WF_TEST_DAYS`, `WF_STEP_DAYS`, `WF_EMBARGO_DAYS`")

st.divider()

# ------------------------------------------------------------------ #
# Feature Engineering Parameters
# ------------------------------------------------------------------ #
st.markdown("### 🔧 Feature Engineering Parameters")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("RSI Period", settings.rsi_period)
c2.metric("MACD Fast", settings.macd_fast)
c3.metric("MACD Slow", settings.macd_slow)
c4.metric("BB Period", settings.bb_period)
c5.metric("ATR Period", settings.atr_period)

st.divider()

# ------------------------------------------------------------------ #
# Simulation Parameters
# ------------------------------------------------------------------ #
st.markdown("### ⚡ Simulation Parameters")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Initial Capital", f"${settings.initial_capital:,.0f}")
c2.metric("Risk per Trade", f"{settings.risk_per_trade:.1%}")
c3.metric("Max Spread", f"{settings.max_spread_pips:.1f} pips")
c4.metric("Slippage", f"{settings.slippage_pips:.1f} pips")

st.divider()

# ------------------------------------------------------------------ #
# Pipeline Reset
# ------------------------------------------------------------------ #
st.markdown("### 🔄 Session Controls")
c1, c2, c3 = st.columns(3)

with c1:
    if st.button("🗑️ Clear Session State", type="secondary", use_container_width=True):
        for key in ["forex_df", "macro_df", "feature_df", "trade_log",
                    "strategy_summary", "training_results", "model_loaded",
                    "predictions", "validation_report"]:
            if key in st.session_state:
                del st.session_state[key]
        for step in st.session_state.get("pipeline_steps", {}).keys():
            st.session_state.pipeline_steps[step] = False
        st.success("Session state cleared!")
        st.rerun()

with c2:
    if st.button("🗄️ Clear Feature Cache", type="secondary", use_container_width=True):
        cache_dir = Path(settings.data_processed_path)
        cleared = 0
        for f in cache_dir.glob("features_*.parquet"):
            f.unlink()
            cleared += 1
        st.success(f"Cleared {cleared} cached feature files")

with c3:
    if st.button("📊 Check Database", type="secondary", use_container_width=True):
        try:
            from src.database.repository import get_repos
            forex_repo, macro_repo, trade_repo, model_repo = get_repos()
            n_candles = forex_repo.count()
            st.info(f"Database | Candles: {n_candles:,}")
        except Exception as e:
            st.error(f"DB check failed: {e}")

st.divider()

# ------------------------------------------------------------------ #
# System Info
# ------------------------------------------------------------------ #
st.markdown("### 📋 System Information")
import platform, sys as _sys
c1, c2 = st.columns(2)
with c1:
    st.markdown(f"""
    | Property | Value |
    |---|---|
    | Python | {_sys.version.split()[0]} |
    | Platform | {platform.system()} {platform.release()} |
    | Project Root | `{settings.project_root}` |
    | DB Path | `{settings.db_path}` |
    | Model Path | `{settings.model_path}` |
    """)
with c2:
    try:
        import xgboost, streamlit, pandas, numpy, shap
        st.markdown(f"""
        | Package | Version |
        |---|---|
        | XGBoost | {xgboost.__version__} |
        | Streamlit | {streamlit.__version__} |
        | Pandas | {pandas.__version__} |
        | NumPy | {numpy.__version__} |
        | SHAP | {shap.__version__} |
        """)
    except Exception:
        st.info("Install all packages via `pip install -r requirements.txt`")

# ------------------------------------------------------------------ #
# Log Viewer
# ------------------------------------------------------------------ #
st.divider()
with st.expander("📜 Recent Logs"):
    log_dir = Path(settings.log_path)
    log_files = sorted(log_dir.glob("*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
    if log_files:
        log_file = log_files[0]
        try:
            with open(log_file) as f:
                lines = f.readlines()
            st.code("".join(lines[-50:]), language="text")
        except Exception:
            st.info("No logs available yet")
    else:
        st.info("No log files found")
