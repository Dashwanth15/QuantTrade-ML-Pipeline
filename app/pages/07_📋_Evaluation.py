"""
QuantTrade ML Pipeline — Evaluation Page
Complete metrics: regression + trading + walk-forward breakdown.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.visualization.themes import get_css, COLORS
from src.visualization.charts import (
    prediction_vs_actual_chart, walk_forward_chart, drawdown_chart, return_distribution_chart
)

st.set_page_config(page_title="Evaluation | QuantTrade", page_icon="📋", layout="wide")
st.markdown(get_css(), unsafe_allow_html=True)

from src.dashboard import ensure_dashboard_state, render_status_card

ensure_dashboard_state()

st.markdown("## 📋 Model Evaluation")
st.markdown("Comprehensive evaluation across regression metrics and trading performance metrics.")
st.divider()

results = st.session_state.get("training_results", {})
metrics = results.get("overall_metrics", {})
fold_results = results.get("fold_results", [])
all_preds = results.get("all_predictions", [])
all_actuals = results.get("all_actuals", [])

if not results:
    render_status_card(
        "Evaluation Data Missing",
        "No evaluation artifacts are available from the current backend outputs.",
        details="Once the model training pipeline produces evaluation details and predictions, this page will display regression and trading diagnostics.",
        icon="⚠️",
    )
    st.stop()

# ------------------------------------------------------------------ #
# Metrics Tables
# ------------------------------------------------------------------ #
col1, col2 = st.columns(2)

with col1:
    st.markdown("### Regression Metrics")
    reg_metrics = pd.DataFrame([
        ("MAE", f"{metrics.get('mae', 0):.6f}", "Mean Absolute Error"),
        ("RMSE", f"{metrics.get('rmse', 0):.6f}", "Root Mean Squared Error"),
        ("R²", f"{metrics.get('r2', 0):.4f}", "Coefficient of Determination"),
        ("MAPE", f"{metrics.get('mape', 0):.2%}", "Mean Absolute % Error"),
        ("Direction Acc", f"{metrics.get('direction_accuracy', 0):.2%}", "Correct trade direction"),
    ], columns=["Metric", "Value", "Description"])
    st.dataframe(reg_metrics, use_container_width=True, hide_index=True)

with col2:
    st.markdown("### Trading Metrics")
    trd_metrics = pd.DataFrame([
        ("Sharpe Ratio", f"{metrics.get('sharpe', 0):.3f}", "Annualized risk-adjusted return"),
        ("Sortino Ratio", f"{metrics.get('sortino', 0):.3f}", "Downside risk-adjusted return"),
        ("Max Drawdown", f"${metrics.get('max_drawdown', 0):.2f}", "Worst peak-to-trough"),
        ("Win Rate", f"{metrics.get('win_rate', 0):.2%}", "% of profitable trades"),
        ("Profit Factor", f"{metrics.get('profit_factor', 0):.3f}", "Gross profit / gross loss"),
        ("Calmar Ratio", f"{metrics.get('calmar', 0):.3f}", "Return / Max Drawdown"),
        ("Avg Trade PnL", f"${metrics.get('avg_trade_pnl', 0):.4f}", "Mean trade profit"),
        ("Predicted Trades", f"{metrics.get('n_predicted_trades', 0):,}", "Trades where model predicts profit"),
    ], columns=["Metric", "Value", "Description"])
    st.dataframe(trd_metrics, use_container_width=True, hide_index=True)

st.divider()

# ------------------------------------------------------------------ #
# Prediction vs Actual + Walk-Forward Breakdown
# ------------------------------------------------------------------ #
c1, c2 = st.columns(2)

with c1:
    st.markdown("### Predicted vs Actual PnL")
    import plotly.io as pio
    chart_path_pred = Path("data/outputs/charts/prediction_vs_actual.json")
    if chart_path_pred.exists():
        fig_pred = pio.read_json(str(chart_path_pred))
    elif all_preds and all_actuals:
        fig_pred = prediction_vs_actual_chart(all_actuals, all_preds)
    else:
        fig_pred = None
        st.info("No out-of-sample predictions available")
    if fig_pred:
        st.plotly_chart(fig_pred, use_container_width=True)

with c2:
    st.markdown("### Walk-Forward Fold Metrics")
    if fold_results:
        tab1, tab2, tab3 = st.tabs(["Sharpe", "MAE", "Win Rate"])
        with tab1:
            chart_path_wf1 = Path("data/outputs/charts/wf_sharpe.json")
            if chart_path_wf1.exists():
                fig_wf1 = pio.read_json(str(chart_path_wf1))
            else:
                fig_wf1 = walk_forward_chart(fold_results, "sharpe", "Sharpe by Fold")
            st.plotly_chart(fig_wf1, use_container_width=True)
        with tab2:
            chart_path_wf2 = Path("data/outputs/charts/wf_mae.json")
            if chart_path_wf2.exists():
                fig_wf2 = pio.read_json(str(chart_path_wf2))
            else:
                fig_wf2 = walk_forward_chart(fold_results, "mae", "MAE by Fold")
            st.plotly_chart(fig_wf2, use_container_width=True)
        with tab3:
            chart_path_wf3 = Path("data/outputs/charts/wf_win_rate.json")
            if chart_path_wf3.exists():
                fig_wf3 = pio.read_json(str(chart_path_wf3))
            else:
                fig_wf3 = walk_forward_chart(fold_results, "win_rate", "Win Rate by Fold")
            st.plotly_chart(fig_wf3, use_container_width=True)

# ------------------------------------------------------------------ #
# Strategy Return Distribution
# ------------------------------------------------------------------ #
st.divider()
c3, c4 = st.columns(2)

with c3:
    st.markdown("### Predicted PnL Distribution (Out-of-Sample)")
    chart_path_dist = Path("data/outputs/charts/predicted_returns_dist.json")
    if chart_path_dist.exists():
        fig_dist = pio.read_json(str(chart_path_dist))
    elif all_preds:
        fig_dist = return_distribution_chart(np.array(all_preds), "Predicted PnL Distribution")
    else:
        fig_dist = None
    if fig_dist:
        st.plotly_chart(fig_dist, use_container_width=True)

with c4:
    st.markdown("### Actual vs Residual")
    chart_path_resid = Path("data/outputs/charts/residuals_vs_actual.json")
    if chart_path_resid.exists():
        fig_resid = pio.read_json(str(chart_path_resid))
    elif all_preds and all_actuals:
        import plotly.graph_objects as go
        residuals = np.array(all_actuals) - np.array(all_preds)
        fig_resid = go.Figure(go.Scatter(
            x=np.array(all_actuals),
            y=residuals,
            mode="markers",
            marker=dict(color=COLORS["electric_blue"], size=3, opacity=0.4),
            name="Residuals",
        ))
        fig_resid.add_hline(y=0, line_dash="dash", line_color=COLORS["coral"])
        fig_resid.update_layout(
            template="quanttrade", height=380,
            title="Residuals vs Actual",
            xaxis_title="Actual PnL", yaxis_title="Residual",
        )
    else:
        fig_resid = None
    if fig_resid:
        st.plotly_chart(fig_resid, use_container_width=True)

# ------------------------------------------------------------------ #
# Fold Detail Table
# ------------------------------------------------------------------ #
st.divider()
with st.expander("📊 Walk-Forward Fold Detail"):
    if fold_results:
        fold_df = pd.DataFrame(fold_results)
        display_cols = [c for c in ["fold_index", "train_start", "train_end", "test_start",
                                    "test_end", "n_train", "n_test", "mae", "rmse", "r2",
                                    "sharpe", "win_rate"] if c in fold_df.columns]
        st.dataframe(fold_df[display_cols], use_container_width=True, hide_index=True)
