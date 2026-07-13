"""
QuantTrade ML Pipeline — Macroeconomic Events Page
Apify scraping, event table, impact analysis, timeline.
"""
import sys
from pathlib import Path
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import settings
from src.dashboard import ensure_dashboard_state, render_status_card
from src.ingestion.macro_scraper import scrape_macro_events
from src.database.repository import get_repos
from src.visualization.themes import get_css, COLORS
from src.visualization.charts import macro_event_timeline

st.set_page_config(page_title="Macro Events | QuantTrade", page_icon="🌍", layout="wide")
st.markdown(get_css(), unsafe_allow_html=True)

ensure_dashboard_state()

st.markdown("## 🌍 Macroeconomic Events")
st.markdown("Economic calendar events scraped via Apify — last 180 days of major macro data.")
st.divider()

macro_df = st.session_state.get("macro_df", pd.DataFrame())

if macro_df.empty:
    render_status_card(
        "Macro Events Data Missing",
        "Macroeconomic event data is not available for this view.",
        details="Once the backend pipeline has produced macro event outputs, this page will show event tables, impact charts, and timeline overlays.",
        icon="⚠️",
    )
    st.stop()

# ------------------------------------------------------------------ #
# KPI Row
# ------------------------------------------------------------------ #
n_events = len(macro_df)
n_high = int((macro_df["impact_score"] >= 3).sum()) if "impact_score" in macro_df.columns else 0
n_categories = macro_df["category"].nunique() if "category" in macro_df.columns else 0
n_eurusd = int(macro_df["eurusd_relevant"].sum()) if "eurusd_relevant" in macro_df.columns else n_events

m1, m2, m3, m4 = st.columns(4)
m1.metric("Total Events", f"{n_events:,}")
m2.metric("High Impact", f"{n_high:,}")
m3.metric("EUR/USD Relevant", f"{n_eurusd:,}")
m4.metric("Categories", f"{n_categories}")

st.divider()

# ------------------------------------------------------------------ #
# Filters + Table
# ------------------------------------------------------------------ #
with st.expander("🎛️ Filters", expanded=True):
    c1, c2, c3 = st.columns(3)
    with c1:
        cat_filter = st.multiselect(
            "Category",
            options=macro_df["category"].unique().tolist() if "category" in macro_df.columns else [],
            default=[],
        )
    with c2:
        impact_filter = st.multiselect(
            "Impact Level",
            options=["high", "medium", "low"],
            default=["high", "medium"],
        )
    with c3:
        eurusd_only = st.checkbox("EUR/USD Relevant Only", value=True)

filtered = macro_df.copy()
if cat_filter and "category" in filtered.columns:
    filtered = filtered[filtered["category"].isin(cat_filter)]
if impact_filter and "impact" in filtered.columns:
    filtered = filtered[filtered["impact"].isin(impact_filter)]
if eurusd_only and "eurusd_relevant" in filtered.columns:
    filtered = filtered[filtered["eurusd_relevant"]]

st.markdown(f"### Economic Calendar ({len(filtered):,} events)")

display_cols = ["timestamp_utc", "event_name", "country", "currency", "impact",
                "forecast", "actual", "previous", "surprise", "category"]
display_cols = [c for c in display_cols if c in filtered.columns]

if not filtered.empty:
    st.dataframe(
        filtered[display_cols].sort_values("timestamp_utc", ascending=False),
        use_container_width=True,
        hide_index=True,
    )

# ------------------------------------------------------------------ #
# Category Breakdown Charts
# ------------------------------------------------------------------ #
import plotly.express as px

st.divider()
c1, c2 = st.columns(2)

with c1:
    st.markdown("### Events by Category")
    if "category" in macro_df.columns:
        cat_counts = macro_df["category"].value_counts().reset_index()
        cat_counts.columns = ["category", "count"]
        fig = px.pie(
            cat_counts, values="count", names="category",
            color_discrete_sequence=[COLORS["electric_blue"], COLORS["neon_green"],
                                     COLORS["gold"], COLORS["coral"], COLORS["purple"],
                                     COLORS["cyan"], "#e67e22"],
        )
        fig.update_layout(template="quanttrade", height=350, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)

with c2:
    st.markdown("### Surprise Magnitude by Category")
    if "category" in macro_df.columns and "surprise_magnitude" in macro_df.columns:
        surprise_by_cat = macro_df.groupby("category")["surprise_magnitude"].mean().reset_index()
        surprise_by_cat = surprise_by_cat.sort_values("surprise_magnitude", ascending=False)

        fig = px.bar(
            surprise_by_cat, x="category", y="surprise_magnitude",
            color="surprise_magnitude",
            color_continuous_scale=["#1a2035", COLORS["electric_blue"], COLORS["neon_green"]],
        )
        fig.update_layout(template="quanttrade", height=350, showlegend=False,
                          xaxis_title="Category", yaxis_title="Avg Surprise Magnitude")
        st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------------------ #
# Timeline
# ------------------------------------------------------------------ #
if st.session_state.get("forex_df") is not None:
    st.divider()
    st.markdown("### Price + Macro Event Timeline")
    import plotly.io as pio
    chart_path = Path("data/outputs/charts/macro_timeline.json")
    if chart_path.exists():
        fig_timeline = pio.read_json(str(chart_path))
    else:
        forex_df = st.session_state.forex_df
        fig_timeline = macro_event_timeline(forex_df, macro_df, n_bars=1000)
    st.plotly_chart(fig_timeline, use_container_width=True)
