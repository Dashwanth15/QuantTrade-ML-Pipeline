"""
QuantTrade ML Pipeline — Downloads Page
Provide buttons to download models, datasets, reports, and interactive/static charts.
"""
import sys
import os
from pathlib import Path
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.visualization.themes import get_css

st.set_page_config(page_title="Downloads | QuantTrade", page_icon="💾", layout="wide")
st.markdown(get_css(), unsafe_allow_html=True)

from src.dashboard import ensure_dashboard_state, render_status_card

ensure_dashboard_state()

st.markdown("## 💾 Download Platform Artifacts")
st.markdown("Download raw/processed datasets, trained models, performance reports, and generated charts.")

if not st.session_state.pipeline_complete:
    render_status_card(
        "Partial Output Files Available",
        "Some backend outputs are available for download while others are still missing.",
        details="Download whichever artifacts are present. This page displays only available datasets, model artifacts, and reports.",
        icon="⚠️",
    )

OUTPUTS_DIR = Path("data/outputs")
MODELS_DIR = OUTPUTS_DIR / "models"
DATASETS_DIR = OUTPUTS_DIR / "datasets"
REPORTS_DIR = OUTPUTS_DIR / "reports"
CHARTS_DIR = OUTPUTS_DIR / "charts"

def download_button_for_file(filepath: Path, label: str, mime: str = "application/octet-stream") -> None:
    if filepath.exists():
        with open(filepath, "rb") as f:
            data = f.read()
        st.download_button(
            label=label,
            data=data,
            file_name=filepath.name,
            mime=mime,
            use_container_width=True
        )
    else:
        st.button(f"⚠️ {filepath.name} missing", disabled=True, use_container_width=True)

c1, c2 = st.columns(2)

with c1:
    st.markdown("### 🗂️ Datasets & Models")
    
    st.write("**Trained XGBoost Model**")
    download_button_for_file(MODELS_DIR / "model.joblib", "📥 Download Model (.joblib)")
    
    st.write("**Out-of-Sample Predictions**")
    download_button_for_file(DATASETS_DIR / "predictions.csv", "📥 Download Predictions (CSV)", "text/csv")
    
    st.write("**Engineered Feature Matrix**")
    download_button_for_file(DATASETS_DIR / "engineered_features.csv", "📥 Download Engineered Features (CSV)", "text/csv")
    
    st.write("**Cleaned Forex Dataset**")
    download_button_for_file(DATASETS_DIR / "cleaned_forex.csv", "📥 Download Cleaned Forex (CSV)", "text/csv")
    
    st.write("**Trade Simulation Logs**")
    download_button_for_file(DATASETS_DIR / "trade_log.csv", "📥 Download Trade Log (CSV)", "text/csv")

with c2:
    st.markdown("### 📋 Reports & Valuations")
    
    st.write("**Strategy Summary Recommendations**")
    download_button_for_file(REPORTS_DIR / "strategy_recommendations.csv", "📥 Download Recommendations (CSV)", "text/csv")
    
    st.write("**Evaluation Metrics**")
    download_button_for_file(REPORTS_DIR / "evaluation_report.json", "📥 Download Evaluation Report (JSON)", "application/json")
    
    st.write("**Walk-Forward Fold Results**")
    download_button_for_file(REPORTS_DIR / "walk_forward_results.csv", "📥 Download WF Fold Metrics (CSV)", "text/csv")
    
    st.write("**Feature Importance (XGBoost Gain)**")
    download_button_for_file(REPORTS_DIR / "feature_importance.csv", "📥 Download Feature Importance (CSV)", "text/csv")
    
    st.write("**SHAP Explanation Report**")
    download_button_for_file(REPORTS_DIR / "shap_report.csv", "📥 Download SHAP Values (CSV)", "text/csv")

st.divider()
st.markdown("### 📊 Generated Visualization Charts")
st.write("Download pre-rendered interactive HTML visualizations or static high-resolution PNG images.")

charts_list = [
    ("market_candlestick", "EUR/USD Candlestick Overview"),
    ("market_returns_dist", "Forex Return Distribution"),
    ("market_session_heatmap", "Forex Session Heatmap"),
    ("macro_timeline", "Price + Macro Event Timeline"),
    ("feature_correlation", "Feature Correlation Matrix"),
    ("strategy_equity_curves", "Strategy Equity Curves"),
    ("strategy_radar", "Multi-Metric Strategy Radar"),
    ("rolling_performance", "Rolling Sharpe & Win Rate"),
    ("portfolio_drawdown", "Drawdown Anatomy Chart"),
    ("model_feature_importance", "XGBoost Feature Importance"),
    ("model_shap_summary", "SHAP Feature Explanations"),
    ("wf_sharpe", "Walk-Forward Sharpe by Fold"),
    ("wf_mae", "Walk-Forward MAE by Fold"),
    ("wf_win_rate", "Walk-Forward Win Rate by Fold"),
    ("prediction_vs_actual", "Predicted vs Actual PnL"),
    ("residuals_vs_actual", "Residuals vs Actual PnL"),
    ("predicted_returns_dist", "Predicted PnL Distribution"),
]

for i in range(0, len(charts_list), 3):
    cols = st.columns(3)
    for j in range(3):
        if i + j < len(charts_list):
            file_base, label = charts_list[i + j]
            with cols[j]:
                st.markdown(f"**{label}**")
                cc1, cc2 = st.columns(2)
                with cc1:
                    download_button_for_file(CHARTS_DIR / f"{file_base}.html", "🌐 HTML", "text/html")
                with cc2:
                    download_button_for_file(CHARTS_DIR / f"{file_base}.png", "🖼️ PNG", "image/png")
