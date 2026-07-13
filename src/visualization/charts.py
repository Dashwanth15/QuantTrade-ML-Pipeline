"""
QuantTrade ML Pipeline — Plotly Chart Library
Production-grade, interactive charts for the Streamlit dashboard.
All charts use the dark QuantTrade Plotly template.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.visualization.themes import COLORS, STRATEGY_COLORS


# ------------------------------------------------------------------ #
# Helper
# ------------------------------------------------------------------ #
def _apply_layout(fig: go.Figure, title: str = "", height: int = 450) -> go.Figure:
    """Apply standard layout overrides."""
    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color=COLORS["text_primary"])),
        height=height,
        template="quanttrade",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


# ------------------------------------------------------------------ #
# Market Data Charts
# ------------------------------------------------------------------ #

def candlestick_chart(
    df: pd.DataFrame,
    indicators: list[str] | None = None,
    title: str = "EUR/USD — Candlestick",
    n_bars: int = 500,
) -> go.Figure:
    """
    Interactive candlestick chart with optional indicator overlays.
    
    Args:
        df: Feature DataFrame with mid_open/high/low/close
        indicators: List of column names to overlay (e.g., ["ema_21", "bb_upper"])
        n_bars: Number of most recent bars to display
    """
    df = df.tail(n_bars).copy()

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        row_heights=[0.6, 0.2, 0.2],
        vertical_spacing=0.03,
    )

    # Candlesticks
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df.get("mid_open", df.get("bid_open")),
        high=df.get("mid_high", df.get("bid_high")),
        low=df.get("mid_low", df.get("bid_low")),
        close=df.get("mid_close", df.get("bid_close")),
        name="EURUSD",
        increasing_line_color=COLORS["neon_green"],
        decreasing_line_color=COLORS["coral"],
        increasing_fillcolor=COLORS["neon_green"],
        decreasing_fillcolor=COLORS["coral"],
        showlegend=False,
    ), row=1, col=1)

    # Indicator overlays
    indicator_colors = [COLORS["electric_blue"], COLORS["gold"],
                        COLORS["purple"], COLORS["cyan"]]
    if indicators:
        for i, ind in enumerate(indicators):
            if ind in df.columns:
                color = indicator_colors[i % len(indicator_colors)]
                fig.add_trace(go.Scatter(
                    x=df.index, y=df[ind],
                    name=ind, mode="lines",
                    line=dict(color=color, width=1.5, dash="solid"),
                ), row=1, col=1)

    # RSI panel
    if "rsi_14" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["rsi_14"],
            name="RSI(14)", mode="lines",
            line=dict(color=COLORS["electric_blue"], width=1.5),
        ), row=2, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color=COLORS["coral"], opacity=0.5, row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color=COLORS["neon_green"], opacity=0.5, row=2, col=1)

    # Spread panel
    if "spread_pips" in df.columns:
        fig.add_trace(go.Bar(
            x=df.index, y=df["spread_pips"],
            name="Spread (pips)",
            marker_color=COLORS["text_muted"],
            opacity=0.7,
        ), row=3, col=1)

    fig.update_layout(
        title=title,
        template="quanttrade",
        height=700,
        xaxis_rangeslider_visible=False,
        yaxis_title="Price",
        yaxis2_title="RSI",
        yaxis3_title="Spread",
    )
    return fig


def equity_curve_chart(
    trade_log: pd.DataFrame,
    title: str = "Strategy Equity Curves",
) -> go.Figure:
    """
    Equity curve with per-strategy breakdown + drawdown overlay.
    """
    if trade_log.empty:
        return go.Figure()

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.7, 0.3],
        vertical_spacing=0.05,
    )

    for strategy_id in trade_log["strategy_id"].unique():
        strat = trade_log[trade_log["strategy_id"] == strategy_id].sort_values("exit_time")
        cum_pnl = strat["pnl_usd"].cumsum()
        color = STRATEGY_COLORS.get(strategy_id, COLORS["electric_blue"])

        fig.add_trace(go.Scatter(
            x=strat["exit_time"], y=cum_pnl,
            name=strategy_id,
            mode="lines",
            line=dict(color=color, width=2),
        ), row=1, col=1)

        # Drawdown
        peak = cum_pnl.cummax()
        drawdown = (cum_pnl - peak)
        fig.add_trace(go.Scatter(
            x=strat["exit_time"], y=drawdown,
            name=f"{strategy_id} DD",
            mode="lines",
            line=dict(color=color, width=1, dash="dot"),
            showlegend=False,
            fill="tozeroy",
            fillcolor=f"rgba{tuple(list(_hex_to_rgb(color)) + [0.1])}",
        ), row=2, col=1)

    fig.update_layout(
        title=title,
        template="quanttrade",
        height=550,
        yaxis_title="Cumulative PnL (USD)",
        yaxis2_title="Drawdown (USD)",
    )
    return fig


def feature_importance_chart(
    importance_df: pd.DataFrame,
    top_n: int = 25,
    title: str = "Feature Importance (XGBoost Gain)",
) -> go.Figure:
    """Horizontal bar chart of feature importance."""
    if importance_df.empty:
        return go.Figure()

    # Select top N
    col = "importance_gain" if "importance_gain" in importance_df.columns else "mean_abs_shap"
    df = importance_df.nlargest(top_n, col).sort_values(col)

    fig = go.Figure(go.Bar(
        x=df[col],
        y=df["feature"],
        orientation="h",
        marker=dict(
            color=df[col],
            colorscale=[[0, COLORS["purple"]], [0.5, COLORS["electric_blue"]], [1, COLORS["neon_green"]]],
            showscale=False,
        ),
        text=df[col].round(4),
        textposition="outside",
        textfont=dict(size=10, color=COLORS["text_secondary"]),
    ))

    fig = _apply_layout(fig, title, height=max(400, top_n * 20 + 100))
    fig.update_layout(yaxis=dict(tickfont=dict(size=11)))
    return fig


def shap_summary_chart(
    shap_importance: pd.DataFrame,
    top_n: int = 20,
    title: str = "SHAP Feature Importance",
) -> go.Figure:
    """SHAP mean |value| bar chart."""
    if shap_importance.empty:
        return go.Figure()

    df = shap_importance.head(top_n).sort_values("mean_abs_shap")
    fig = go.Figure(go.Bar(
        x=df["mean_abs_shap"],
        y=df["feature"],
        orientation="h",
        marker=dict(
            color=df["mean_abs_shap"],
            colorscale=[[0, "#1a1f35"], [1, COLORS["electric_blue"]]],
            showscale=True,
            colorbar=dict(title="Mean |SHAP|", thickness=10),
        ),
    ))
    fig = _apply_layout(fig, title, height=max(400, top_n * 22 + 100))
    return fig


def walk_forward_chart(
    fold_results: list[dict] | pd.DataFrame,
    metric: str = "sharpe",
    title: str = "Walk-Forward Validation Performance",
) -> go.Figure:
    """Bar chart of per-fold metrics."""
    if isinstance(fold_results, list):
        df = pd.DataFrame(fold_results)
    else:
        df = fold_results.copy()

    if df.empty or metric not in df.columns:
        return go.Figure()

    colors = [
        COLORS["neon_green"] if v > 0 else COLORS["coral"]
        for v in df[metric]
    ]

    fig = go.Figure(go.Bar(
        x=[f"Fold {i}" for i in df["fold_index"]],
        y=df[metric],
        marker_color=colors,
        text=df[metric].round(3),
        textposition="outside",
    ))

    # Add average line
    avg = df[metric].mean()
    fig.add_hline(
        y=avg, line_dash="dash",
        line_color=COLORS["gold"],
        annotation_text=f"Avg: {avg:.3f}",
        annotation_position="right",
    )

    fig = _apply_layout(fig, title, height=400)
    fig.update_layout(yaxis_title=metric.upper(), xaxis_title="Walk-Forward Fold")
    return fig


def prediction_vs_actual_chart(
    y_true: np.ndarray | list,
    y_pred: np.ndarray | list,
    title: str = "Predicted vs Actual PnL",
) -> go.Figure:
    """Scatter plot of predicted vs actual with regression line."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    fig = go.Figure()

    # Scatter points
    fig.add_trace(go.Scatter(
        x=y_true, y=y_pred,
        mode="markers",
        marker=dict(
            color=COLORS["electric_blue"],
            size=4,
            opacity=0.5,
        ),
        name="Predictions",
    ))

    # Perfect prediction line
    lim = max(abs(y_true).max(), abs(y_pred).max()) * 1.1
    fig.add_trace(go.Scatter(
        x=[-lim, lim], y=[-lim, lim],
        mode="lines",
        line=dict(color=COLORS["neon_green"], dash="dash", width=1.5),
        name="Perfect Prediction",
    ))

    fig = _apply_layout(fig, title, height=450)
    fig.update_layout(
        xaxis_title="Actual PnL (USD)",
        yaxis_title="Predicted PnL (USD)",
    )
    return fig


def drawdown_chart(
    pnl_series: pd.Series,
    title: str = "Drawdown Analysis",
) -> go.Figure:
    """Underwater equity (drawdown) chart."""
    cum_pnl = pnl_series.cumsum()
    peak = cum_pnl.cummax()
    drawdown = cum_pnl - peak

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.6, 0.4], vertical_spacing=0.05)

    fig.add_trace(go.Scatter(
        x=cum_pnl.index, y=cum_pnl,
        mode="lines", name="Cumulative PnL",
        line=dict(color=COLORS["electric_blue"], width=2),
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=drawdown.index, y=drawdown,
        mode="lines", name="Drawdown",
        line=dict(color=COLORS["coral"], width=1.5),
        fill="tozeroy",
        fillcolor="rgba(255,107,107,0.15)",
    ), row=2, col=1)

    fig = _apply_layout(fig, title, height=500)
    return fig


def correlation_heatmap(
    df: pd.DataFrame,
    n_features: int = 30,
    title: str = "Feature Correlation Matrix",
) -> go.Figure:
    """Interactive correlation heatmap."""
    numeric_df = df.select_dtypes(include=[np.number])

    # Select top N by variance
    top_cols = numeric_df.var().nlargest(n_features).index
    corr = numeric_df[top_cols].corr()

    fig = go.Figure(go.Heatmap(
        z=corr.values,
        x=list(corr.columns),
        y=list(corr.index),
        colorscale=[
            [0, COLORS["coral"]],
            [0.5, COLORS["bg_secondary"]],
            [1, COLORS["electric_blue"]],
        ],
        zmid=0,
        text=np.round(corr.values, 2),
        texttemplate="%{text}",
        textfont=dict(size=8),
        hoverongaps=False,
    ))

    fig = _apply_layout(fig, title, height=600)
    fig.update_layout(
        xaxis=dict(tickangle=-45, tickfont=dict(size=9)),
        yaxis=dict(tickfont=dict(size=9)),
    )
    return fig


def monthly_returns_heatmap(
    trade_log: pd.DataFrame,
    title: str = "Monthly Returns Heatmap",
) -> go.Figure:
    """Calendar heatmap of monthly PnL."""
    if trade_log.empty:
        return go.Figure()

    df = trade_log.copy()
    df["exit_time"] = pd.to_datetime(df["exit_time"])
    df["year"] = df["exit_time"].dt.year
    df["month"] = df["exit_time"].dt.month

    monthly = df.groupby(["year", "month"])["pnl_usd"].sum().reset_index()
    pivot = monthly.pivot(index="year", columns="month", values="pnl_usd").fillna(0)

    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                   "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    pivot.columns = [month_names[m - 1] for m in pivot.columns]

    max_abs = max(abs(pivot.values.max()), abs(pivot.values.min()), 1)

    fig = go.Figure(go.Heatmap(
        z=pivot.values,
        x=list(pivot.columns),
        y=[str(y) for y in pivot.index],
        colorscale=[
            [0, COLORS["coral"]],
            [0.5, COLORS["bg_card"]],
            [1, COLORS["neon_green"]],
        ],
        zmid=0,
        zmin=-max_abs,
        zmax=max_abs,
        text=[[f"${v:,.0f}" for v in row] for row in pivot.values],
        texttemplate="%{text}",
        textfont=dict(size=10, color=COLORS["text_primary"]),
        hoverongaps=False,
        colorbar=dict(title="PnL (USD)", thickness=12),
    ))

    fig = _apply_layout(fig, title, height=max(350, len(pivot) * 40 + 150))
    fig.update_layout(xaxis_title="Month", yaxis_title="Year")
    return fig


def return_distribution_chart(
    returns: pd.Series | np.ndarray,
    title: str = "Return Distribution",
) -> go.Figure:
    """Histogram + KDE of returns with normal overlay."""
    returns = pd.Series(np.asarray(returns).flatten()).dropna()

    fig = go.Figure()

    # Histogram
    fig.add_trace(go.Histogram(
        x=returns,
        nbinsx=80,
        name="Returns",
        marker=dict(
            color=COLORS["electric_blue"],
            opacity=0.7,
            line=dict(color=COLORS["bg_primary"], width=0.3),
        ),
        histnorm="probability density",
    ))

    # Normal distribution overlay
    x_range = np.linspace(returns.min(), returns.max(), 200)
    mu, sigma = returns.mean(), returns.std()
    normal_y = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_range - mu) / sigma) ** 2)

    fig.add_trace(go.Scatter(
        x=x_range, y=normal_y,
        mode="lines", name="Normal Distribution",
        line=dict(color=COLORS["gold"], width=2, dash="dash"),
    ))

    # Zero line
    fig.add_vline(x=0, line_color=COLORS["coral"], line_dash="dash", opacity=0.7)

    fig = _apply_layout(fig, title, height=380)
    fig.update_layout(
        xaxis_title="PnL (USD)",
        yaxis_title="Density",
        bargap=0.05,
    )
    return fig


def rolling_metrics_chart(
    trade_log: pd.DataFrame,
    window: int = 50,
    title: str = "Rolling Performance Metrics",
) -> go.Figure:
    """Rolling Sharpe and win rate over time."""
    if trade_log.empty:
        return go.Figure()

    df = trade_log.sort_values("exit_time").copy()
    pnl = df["pnl_usd"]

    rolling_mean = pnl.rolling(window).mean()
    rolling_std = pnl.rolling(window).std()
    rolling_sharpe = rolling_mean / (rolling_std + 1e-10) * np.sqrt(5040)
    rolling_win = df["win"].rolling(window).mean()

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05)

    fig.add_trace(go.Scatter(
        x=df["exit_time"], y=rolling_sharpe,
        mode="lines", name=f"Rolling Sharpe ({window})",
        line=dict(color=COLORS["electric_blue"], width=2),
    ), row=1, col=1)
    fig.add_hline(y=1.0, line_dash="dash", line_color=COLORS["neon_green"], opacity=0.5, row=1, col=1)

    fig.add_trace(go.Scatter(
        x=df["exit_time"], y=rolling_win,
        mode="lines", name=f"Rolling Win Rate ({window})",
        line=dict(color=COLORS["gold"], width=2),
    ), row=2, col=1)
    fig.add_hline(y=0.5, line_dash="dash", line_color=COLORS["coral"], opacity=0.5, row=2, col=1)

    fig = _apply_layout(fig, title, height=500)
    fig.update_layout(
        yaxis_title="Sharpe Ratio",
        yaxis2_title="Win Rate",
    )
    return fig


def strategy_radar_chart(
    strategy_summary: pd.DataFrame,
    title: str = "Strategy Comparison — Radar",
) -> go.Figure:
    """Radar chart comparing strategies across multiple metrics."""
    if strategy_summary.empty:
        return go.Figure()

    metrics = ["win_rate", "sharpe", "profit_factor", "avg_pnl_usd"]
    available_metrics = [m for m in metrics if m in strategy_summary.columns]

    if not available_metrics:
        return go.Figure()

    fig = go.Figure()

    for _, row in strategy_summary.iterrows():
        strat_id = str(row.get("strategy_id", "unknown"))
        values = [abs(float(row.get(m, 0) or 0)) for m in available_metrics]
        values.append(values[0])  # Close the radar

        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=available_metrics + [available_metrics[0]],
            name=strat_id,
            line=dict(color=STRATEGY_COLORS.get(strat_id, COLORS["electric_blue"]), width=2),
            fill="toself",
            fillcolor=f"rgba{tuple(list(_hex_to_rgb(STRATEGY_COLORS.get(strat_id, COLORS['electric_blue']))) + [0.1])}",
        ))

    fig.update_layout(
        polar=dict(
            bgcolor=COLORS["bg_card"],
            radialaxis=dict(
                visible=True,
                gridcolor="rgba(255,255,255,0.1)",
                tickfont=dict(color=COLORS["text_secondary"], size=9),
            ),
            angularaxis=dict(
                gridcolor="rgba(255,255,255,0.1)",
                tickfont=dict(color=COLORS["text_secondary"]),
            ),
        ),
        title=title,
        template="quanttrade",
        height=450,
    )
    return fig


def macro_event_timeline(
    df: pd.DataFrame,
    macro_df: pd.DataFrame,
    title: str = "Price with Macro Event Markers",
    n_bars: int = 500,
) -> go.Figure:
    """Price chart with macro event markers."""
    df_plot = df.tail(n_bars).copy()
    close = df_plot.get("mid_close", df_plot.get("bid_close"))

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df_plot.index, y=close,
        mode="lines", name="EUR/USD Close",
        line=dict(color=COLORS["electric_blue"], width=1.5),
    ))

    if not macro_df.empty:
        start = df_plot.index[0]
        end = df_plot.index[-1]
        macro_df = macro_df.copy()
        macro_df["timestamp_utc"] = pd.to_datetime(macro_df["timestamp_utc"], utc=True)
        in_range = macro_df[
            (macro_df["timestamp_utc"] >= start) &
            (macro_df["timestamp_utc"] <= end)
        ]

        # Color by impact
        impact_color_map = {3: COLORS["coral"], 2: COLORS["gold"], 1: COLORS["text_muted"]}

        for impact, group in in_range.groupby("impact_score"):
            color = impact_color_map.get(int(impact), COLORS["text_muted"])
            level_map = {3: "High", 2: "Medium", 1: "Low"}
            fig.add_trace(go.Scatter(
                x=group["timestamp_utc"],
                y=[close.reindex([group["timestamp_utc"].iloc[i]], method="nearest").iloc[0]
                   if len(group) > 0 else close.mean() for i in range(len(group))],
                mode="markers",
                name=f"{level_map.get(int(impact), 'Unknown')} Impact",
                marker=dict(color=color, size=10 - impact, symbol="triangle-up"),
                hovertext=group["event_name"],
            ))

    fig = _apply_layout(fig, title, height=450)
    return fig


def session_heatmap(
    df: pd.DataFrame,
    title: str = "Return by Hour of Day × Day of Week",
) -> go.Figure:
    """Heatmap of average return by hour and day of week."""
    if "return_1h" not in df.columns:
        return go.Figure()

    data = df.copy()
    data["hour"] = data.index.hour
    data["day_name"] = data.index.day_name()
    days_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]

    pivot = data.pivot_table(
        values="return_1h", index="day_name", columns="hour", aggfunc="mean"
    )
    pivot = pivot.reindex([d for d in days_order if d in pivot.index])

    fig = go.Figure(go.Heatmap(
        z=pivot.values * 10000,  # Show in pips
        x=[f"{h:02d}:00" for h in pivot.columns],
        y=list(pivot.index),
        colorscale=[
            [0, COLORS["coral"]],
            [0.5, COLORS["bg_card"]],
            [1, COLORS["neon_green"]],
        ],
        zmid=0,
        colorbar=dict(title="Avg Return (pips)", thickness=12),
        hoverongaps=False,
    ))

    fig = _apply_layout(fig, title, height=320)
    fig.update_layout(xaxis_title="Hour (UTC)", yaxis_title="Day of Week")
    return fig


# ------------------------------------------------------------------ #
# Utility
# ------------------------------------------------------------------ #

def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    """Convert hex color string to RGB tuple."""
    h = hex_color.lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))
