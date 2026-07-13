"""
QuantTrade ML Pipeline — Model Training Page
Walk-forward training, hyperparameter tuning, feature importance.
"""
import sys
from pathlib import Path
import pandas as pd
import streamlit as st
import plotly.io as pio

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config.settings import settings
from src.dashboard import ensure_dashboard_state, render_status_card
from src.ml.trainer import ModelTrainer
from src.visualization.themes import get_css, COLORS
from src.visualization.charts import feature_importance_chart, walk_forward_chart, shap_summary_chart

st.set_page_config(page_title="Model Training | QuantTrade", page_icon="🤖", layout="wide")
st.markdown(get_css(), unsafe_allow_html=True)

ensure_dashboard_state()

st.markdown("## 🤖 Model Training Analytics")
st.markdown("Walk-forward performance, hyperparameter summaries, and feature explainability for trained models.")
st.divider()

results = st.session_state.get("training_results", {})
feature_df = st.session_state.get("feature_df", pd.DataFrame())
trade_log = st.session_state.get("trade_log", pd.DataFrame())
loaded = st.session_state.pipeline_steps

# Always reload the latest run if the session state is stale or zeroed.
if results and results.get("best_params") and all(
    results.get("overall_metrics", {}).get(key, 0) == 0
    for key in ["mae", "rmse", "r2", "sharpe", "win_rate"]
):
    from src.database.repository import get_repos
    repo = get_repos()[3]
    latest_results = repo.load_latest_run() or {}
    if latest_results and latest_results.get("overall_metrics", {}).get("mae", 0) != 0:
        results = latest_results
        st.session_state.training_results = latest_results

metrics = results.get("overall_metrics", {})

if feature_df.empty:
    render_status_card(
        "Feature Matrix Missing",
        "Engineered feature data is required to analyze model training outputs.",
        details="This page will show model fold metrics, importance charts, and SHAP explainability once features are available.",
        icon="⚠️",
    )

if results == {}:
    render_status_card(
        "Training Artifacts Missing",
        "No training results were loaded from backend outputs.",
        details="Ensure the ML pipeline has written walk-forward fold results and model metrics for model-centric analytics.",
        icon="⚠️",
    )

if results == {}:
    st.stop()

results = st.session_state.training_results
metrics = results.get("overall_metrics", {})

if results and results.get("best_params") and all(
    metrics.get(key, 0) == 0 for key in ["mae", "rmse", "r2", "sharpe", "win_rate"]
):
    st.warning("Detected a trained model artifact, but KPI values are all zero. Reloading the latest model run.")
    st.markdown("**Current loaded raw training state:**")
    st.json({
        "result_keys": sorted(results.keys()),
        "overall_metrics": metrics,
        "best_params_present": bool(results.get("best_params")),
        "model_path": results.get("model_path"),
    })

    from src.database.repository import get_repos
    repo = get_repos()[3]
    fresh_results = repo.load_latest_run() or {}
    fresh_metrics = fresh_results.get("overall_metrics", {})
    if fresh_results and fresh_metrics.get("mae", 0) != 0:
        results = fresh_results
        st.session_state.training_results = fresh_results
        metrics = fresh_metrics
    else:
        st.error("Unable to load nonzero KPI metrics from the latest model run. There may be a mismatch between the stored artifact and dashboard state.")
        with st.expander("Raw latest run from repository", expanded=True):
            st.json({
                "latest_run_keys": sorted(fresh_results.keys()),
                "latest_overall_metrics": fresh_metrics,
                "latest_model_path_exists": bool(fresh_results.get("model_path") and Path(fresh_results.get("model_path")).exists()),
            })

# ------------------------------------------------------------------ #
# Training Results KPIs
# ------------------------------------------------------------------ #
st.divider()
st.markdown("### Training Results")

m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.metric("Walk-Forward Folds", results.get("n_folds", 0))
m2.metric("MAE", f"{metrics.get('mae', 0):.4f}")
m3.metric("RMSE", f"{metrics.get('rmse', 0):.4f}")
m4.metric("R²", f"{metrics.get('r2', 0):.4f}")
m5.metric("Sharpe", f"{metrics.get('sharpe', 0):.2f}")
m6.metric("Win Rate", f"{metrics.get('win_rate', 0):.1%}")

st.divider()

# ------------------------------------------------------------------ #
# ------------------------------------------------------------------ #
# Walk-Forward Fold Chart
# ------------------------------------------------------------------ #
c1, c2 = st.columns(2)
with c1:
    st.markdown("### Walk-Forward Performance by Fold")
    fold_results = results.get("fold_results", [])
    if fold_results:
        metric_select = st.selectbox("Metric", ["sharpe", "mae", "rmse", "r2", "win_rate"])
        import plotly.io as pio
        chart_map = {"sharpe": "wf_sharpe", "mae": "wf_mae", "win_rate": "wf_win_rate"}
        chart_name = chart_map.get(metric_select)
        chart_path = Path(f"data/outputs/charts/{chart_name}.json") if chart_name else None
        
        if chart_path and chart_path.exists():
            fig_wf = pio.read_json(str(chart_path))
        else:
            fig_wf = walk_forward_chart(fold_results, metric=metric_select)
        st.plotly_chart(fig_wf, use_container_width=True)

with c2:
    st.markdown("### Best Hyperparameters")
    best_params = results.get("best_params", {})
    if best_params:
        params_df = pd.DataFrame(
            [(k, str(round(v, 6)) if isinstance(v, float) else str(v))
             for k, v in best_params.items() if k not in ("tree_method", "verbosity", "n_jobs", "objective", "random_state")],
            columns=["Parameter", "Value"]
        )
        st.dataframe(params_df, use_container_width=True, hide_index=True)

# ------------------------------------------------------------------ #
# Feature Importance Charts
# ------------------------------------------------------------------ #
st.divider()
st.markdown("### Feature Importance")

tab1, tab2 = st.tabs(["📊 XGBoost Gain", "🔍 SHAP Values"])

with tab1:
    xgb_imp = results.get("xgb_importance")
    chart_path_imp = Path("data/outputs/charts/model_feature_importance.json")
    if chart_path_imp.exists():
        fig_imp = pio.read_json(str(chart_path_imp))
    elif xgb_imp is not None and not xgb_imp.empty:
        fig_imp = feature_importance_chart(xgb_imp, top_n=25)
    else:
        fig_imp = None
        st.info("XGBoost importance not available")
    if fig_imp:
        st.plotly_chart(fig_imp, use_container_width=True)

with tab2:
    shap_imp = results.get("shap_importance")
    chart_path_shap = Path("data/outputs/charts/model_shap_summary.json")
    if chart_path_shap.exists():
        fig_shap = pio.read_json(str(chart_path_shap))
    elif shap_imp is not None and not shap_imp.empty:
        fig_shap = shap_summary_chart(shap_imp, top_n=20)
    else:
        fig_shap = None
        st.info("SHAP importance not available (install shap package)")
    if fig_shap:
        st.plotly_chart(fig_shap, use_container_width=True)

# ------------------------------------------------------------------ #
# Model Artifact Info
# ------------------------------------------------------------------ #
st.divider()
with st.expander("📁 Model Artifact"):
    st.markdown(f"**Model Path:** `{results.get('model_path', 'N/A')}`")
    st.markdown(f"**Run ID:** `{results.get('run_id', 'N/A')}`")
    st.markdown(f"**Features:** {results.get('n_features', 0)}")
    st.markdown(f"**Training Samples:** {results.get('n_samples', 0):,}")
