"""
QuantTrade ML Pipeline — Feature Engineering Page
Feature schema, correlation matrix, distribution explorer.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.dashboard import ensure_dashboard_state, render_status_card
from src.features.feature_store import FeatureStore
from src.visualization.themes import get_css, COLORS
from src.visualization.charts import correlation_heatmap, feature_importance_chart

st.set_page_config(page_title="Features | QuantTrade", page_icon="🔧", layout="wide")
st.markdown(get_css(), unsafe_allow_html=True)

ensure_dashboard_state()

st.markdown("## 🔧 Feature Engineering")
st.markdown("Explore the engineered feature matrix, distributions, correlations, and feature groups.")
st.divider()

feat_df = st.session_state.get("feature_df", pd.DataFrame())
store = st.session_state.get("feature_store") or FeatureStore()

if feat_df.empty:
    render_status_card(
        "Engineered Features Missing",
        "The feature matrix has not been loaded from backend outputs.",
        details="Feature analytics, group exploration, and correlation charts will be available once engineered features are present.",
        icon="⚠️",
    )
    st.stop()
store = st.session_state.get("feature_store") or FeatureStore()

# ------------------------------------------------------------------ #
# Feature Count KPIs
# ------------------------------------------------------------------ #
st.markdown("### Feature Matrix Overview")
store._infer_feature_groups(feat_df)
groups = store.get_feature_groups()
numeric_cols = feat_df.select_dtypes(include=[np.number]).columns

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Total Features", len(feat_df.columns))
m2.metric("Numeric Features", len(numeric_cols))
m3.metric("Time Features", len(groups.get("time", [])))
m4.metric("Price Features", len(groups.get("price", [])))
m5.metric("Technical Features", len(groups.get("technical", [])))

st.divider()

# ------------------------------------------------------------------ #
# Feature Schema Table
# ------------------------------------------------------------------ #
st.markdown("### Feature Schema")
schema_df = store.get_feature_schema(feat_df)
if not schema_df.empty:
    # Color groups
    group_colors = {
        "time": COLORS["electric_blue"],
        "price": COLORS["neon_green"],
        "technical": COLORS["gold"],
        "macro": COLORS["coral"],
        "other": COLORS["text_muted"],
    }

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["All", "⏰ Time", "💰 Price", "📊 Technical", "🌍 Macro"])
    with tab1:
        st.dataframe(schema_df, use_container_width=True, hide_index=True)
    for tab, grp in zip([tab2, tab3, tab4, tab5], ["time", "price", "technical", "macro"]):
        with tab:
            grp_df = schema_df[schema_df["group"] == grp]
            if not grp_df.empty:
                st.dataframe(grp_df, use_container_width=True, hide_index=True)
            else:
                st.info(f"No {grp} features")

st.divider()

# ------------------------------------------------------------------ #
# Correlation Heatmap
# ------------------------------------------------------------------ #
st.markdown("### Feature Correlation Matrix")
n_corr = st.slider("Features in correlation matrix", 10, 50, 30, 5)
import plotly.io as pio
chart_path = Path("data/outputs/charts/feature_correlation.json")
if chart_path.exists() and n_corr == 30:
    fig_corr = pio.read_json(str(chart_path))
else:
    fig_corr = correlation_heatmap(feat_df, n_features=n_corr)
st.plotly_chart(fig_corr, use_container_width=True)

# ------------------------------------------------------------------ #
# Feature Distribution Explorer
# ------------------------------------------------------------------ #
st.divider()
st.markdown("### Feature Distribution Explorer")
if not schema_df.empty:
    selected_feature = st.selectbox(
        "Select Feature",
        options=schema_df["feature"].tolist(),
        index=0,
    )

    if selected_feature in feat_df.columns:
        series = feat_df[selected_feature].dropna()
        c1, c2 = st.columns([3, 1])
        with c1:
            fig_dist = px.histogram(
                x=series.tail(5000),
                nbins=80,
                color_discrete_sequence=[COLORS["electric_blue"]],
                title=f"Distribution: {selected_feature}",
            )
            fig_dist.update_layout(template="quanttrade", height=350)
            st.plotly_chart(fig_dist, use_container_width=True)
        with c2:
            st.markdown("**Statistics**")
            st.write(series.describe().round(6))

# ------------------------------------------------------------------ #
# Null Rate Report
# ------------------------------------------------------------------ #
st.divider()
with st.expander("📊 Null Rate Report"):
    null_rates = feat_df.isnull().mean().sort_values(ascending=False)
    null_df = pd.DataFrame({"feature": null_rates.index, "null_rate": null_rates.values})
    null_df = null_df[null_df["null_rate"] > 0]
    if not null_df.empty:
        fig_null = px.bar(null_df.head(30), x="feature", y="null_rate",
                          color_discrete_sequence=[COLORS["coral"]])
        fig_null.update_layout(template="quanttrade", height=300,
                                xaxis_tickangle=-45, yaxis_tickformat=".1%")
        st.plotly_chart(fig_null, use_container_width=True)
    else:
        st.success("✅ No null values in numeric features!")
