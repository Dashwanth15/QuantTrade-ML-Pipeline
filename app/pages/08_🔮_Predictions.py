"""
QuantTrade ML Pipeline — Predictions Page
Real-time PnL predictions with SHAP explanations.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.ml.predictor import PredictionEngine
from src.visualization.themes import get_css, COLORS
from src.visualization.charts import return_distribution_chart

st.set_page_config(page_title="Predictions | QuantTrade", page_icon="🔮", layout="wide")
st.markdown(get_css(), unsafe_allow_html=True)

from src.dashboard import ensure_dashboard_state, render_status_card

ensure_dashboard_state()

st.markdown("## 🔮 Model Predictions")
st.markdown("Real-time PnL prediction analytics and prediction distribution visualization.")

if not st.session_state.pipeline_steps.get("model_trained", False):
    render_status_card(
        "Predictions Not Available",
        "Prediction outputs are not present in current backend artifacts.",
        details="This page will show predicted PnL series and distribution plots once model inference outputs are available.",
        icon="⚠️",
    )
    st.stop()

from src.database.repository import get_db
from src.database.models import Prediction

db = get_db()
with db.session() as sess:
    recs = sess.query(Prediction).order_by(Prediction.timestamp.asc()).all()

if recs:
    pred_df = pd.DataFrame([{
        "predicted_pnl": r.predicted_pnl
    } for r in recs])
    timestamps = pd.to_datetime([r.timestamp for r in recs], utc=True)
    st.session_state.predictions = pred_df
    st.session_state.pred_timestamps = timestamps
else:
    render_status_card(
        "No Predictions Found",
        "The prediction database table is empty.",
        details="Predictions will appear here after the backend ML pipeline writes inference outputs into the database.",
        icon="⚠️",
    )
    st.stop()

# ------------------------------------------------------------------ #
# Display Predictions
# ------------------------------------------------------------------ #
if st.session_state.get("predictions") is not None:
    pred_df = st.session_state.predictions
    timestamps = st.session_state.get("pred_timestamps")

    # KPI
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Predictions", f"{len(pred_df):,}")
    m2.metric("Avg Predicted PnL", f"${pred_df['predicted_pnl'].mean():.4f}")
    m3.metric("% Positive", f"{(pred_df['predicted_pnl'] > 0).mean():.1%}")
    m4.metric("Max Predicted", f"${pred_df['predicted_pnl'].max():.4f}")

    st.divider()

    # Time series of predictions
    import plotly.graph_objects as go
    fig = go.Figure()

    if timestamps is not None and len(timestamps) == len(pred_df):
        x = timestamps
    else:
        x = list(range(len(pred_df)))

    # Prediction line
    fig.add_trace(go.Scatter(
        x=x, y=pred_df["predicted_pnl"],
        mode="lines", name="Predicted PnL",
        line=dict(color=COLORS["electric_blue"], width=1.5),
    ))

    # CI band
    if "ci_lower" in pred_df.columns and "ci_upper" in pred_df.columns:
        fig.add_trace(go.Scatter(
            x=list(x) + list(x[::-1]),
            y=list(pred_df["ci_upper"]) + list(pred_df["ci_lower"][::-1]),
            fill="toself",
            fillcolor="rgba(0,212,255,0.1)",
            line=dict(color="rgba(0,0,0,0)"),
            name="90% CI",
            showlegend=True,
        ))

    fig.add_hline(y=0, line_dash="dash", line_color=COLORS["coral"], opacity=0.7)

    fig.update_layout(
        template="quanttrade", height=400,
        title="Predicted PnL Over Time",
        xaxis_title="Time", yaxis_title="Predicted PnL (USD)",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Distribution of predictions
    c1, c2 = st.columns(2)
    with c1:
        import plotly.io as pio
        chart_path = Path("data/outputs/charts/predicted_returns_dist.json")
        if chart_path.exists():
            fig_dist = pio.read_json(str(chart_path))
        else:
            fig_dist = return_distribution_chart(pred_df["predicted_pnl"], "Prediction Distribution")
        st.plotly_chart(fig_dist, use_container_width=True)

    with c2:
        st.markdown("### Top 10 Positive Predictions")
        top_preds = pred_df.nlargest(10, "predicted_pnl")
        if timestamps is not None:
            top_preds.index = timestamps[top_preds.index] if hasattr(timestamps, '__getitem__') else top_preds.index
        st.dataframe(top_preds.round(4), use_container_width=True)
