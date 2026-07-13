"""
QuantTrade ML Pipeline — Dark Fintech Theme
Custom Plotly template and CSS variables for a premium dark UI.
"""
from __future__ import annotations

import plotly.graph_objects as go
import plotly.io as pio

# ------------------------------------------------------------------ #
# Color Palette
# ------------------------------------------------------------------ #
COLORS = {
    # Backgrounds
    "bg_primary": "#0a0e1a",
    "bg_secondary": "#111827",
    "bg_card": "#1a2035",
    "bg_hover": "#1e2d4a",

    # Accents
    "electric_blue": "#00d4ff",
    "neon_green": "#00ff88",
    "gold": "#ffd700",
    "coral": "#ff6b6b",
    "purple": "#9b59b6",
    "cyan": "#00bcd4",

    # Text
    "text_primary": "#e8eaf6",
    "text_secondary": "#90a4ae",
    "text_muted": "#546e7a",

    # Strategy colors
    "momentum": "#00d4ff",
    "ma_crossover": "#00ff88",
    "rsi_reversion": "#ffd700",
    "bollinger": "#ff6b6b",
    "breakout": "#9b59b6",
    "trend_following": "#e67e22",
    "support_resistance": "#1abc9c",

    # Signal colors
    "bull": "#00ff88",
    "bear": "#ff4444",
    "neutral": "#546e7a",
    "win": "#00c853",
    "loss": "#ff1744",

    # Gradient stops
    "gradient_start": "#00d4ff",
    "gradient_end": "#9b59b6",
}

STRATEGY_COLORS = {
    "momentum": COLORS["momentum"],
    "ma_crossover": COLORS["ma_crossover"],
    "rsi_reversion": COLORS["rsi_reversion"],
    "bollinger": COLORS["bollinger"],
    "breakout": COLORS["breakout"],
    "trend_following": COLORS["trend_following"],
    "support_resistance": COLORS["support_resistance"],
}

FONT_FAMILY = "Inter, 'Segoe UI', system-ui, -apple-system, sans-serif"


# ------------------------------------------------------------------ #
# Plotly Template
# ------------------------------------------------------------------ #
def register_quanttrade_template() -> None:
    """Register the QuantTrade dark theme as a Plotly template."""
    template = go.layout.Template()
    template.layout = go.Layout(
        paper_bgcolor=COLORS["bg_primary"],
        plot_bgcolor=COLORS["bg_secondary"],
        font=dict(
            family=FONT_FAMILY,
            color=COLORS["text_primary"],
            size=12,
        ),
        title=dict(
            font=dict(size=18, color=COLORS["text_primary"], family=FONT_FAMILY),
            x=0.02,
        ),
        xaxis=dict(
            gridcolor="rgba(255,255,255,0.05)",
            linecolor="rgba(255,255,255,0.1)",
            tickcolor=COLORS["text_secondary"],
            tickfont=dict(color=COLORS["text_secondary"], size=11),
            showgrid=True,
            zeroline=False,
        ),
        yaxis=dict(
            gridcolor="rgba(255,255,255,0.05)",
            linecolor="rgba(255,255,255,0.1)",
            tickcolor=COLORS["text_secondary"],
            tickfont=dict(color=COLORS["text_secondary"], size=11),
            showgrid=True,
            zeroline=True,
            zerolinecolor="rgba(255,255,255,0.15)",
            zerolinewidth=1,
        ),
        legend=dict(
            bgcolor="rgba(17,24,39,0.8)",
            bordercolor="rgba(255,255,255,0.1)",
            borderwidth=1,
            font=dict(color=COLORS["text_secondary"], size=11),
        ),
        hoverlabel=dict(
            bgcolor=COLORS["bg_card"],
            bordercolor=COLORS["electric_blue"],
            font=dict(color=COLORS["text_primary"], family=FONT_FAMILY, size=12),
        ),
        margin=dict(l=60, r=20, t=60, b=60),
        colorway=[
            COLORS["electric_blue"],
            COLORS["neon_green"],
            COLORS["gold"],
            COLORS["coral"],
            COLORS["purple"],
            COLORS["cyan"],
            "#e67e22",
            "#1abc9c",
        ],
    )
    pio.templates["quanttrade"] = template
    pio.templates.default = "quanttrade"


# Register on import
register_quanttrade_template()


# ------------------------------------------------------------------ #
# Streamlit Custom CSS
# ------------------------------------------------------------------ #
STREAMLIT_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* Root variables */
:root {{
    --bg-primary: {COLORS["bg_primary"]};
    --bg-secondary: {COLORS["bg_secondary"]};
    --bg-card: {COLORS["bg_card"]};
    --electric-blue: {COLORS["electric_blue"]};
    --neon-green: {COLORS["neon_green"]};
    --gold: {COLORS["gold"]};
    --coral: {COLORS["coral"]};
    --text-primary: {COLORS["text_primary"]};
    --text-secondary: {COLORS["text_secondary"]};
    --font-family: {FONT_FAMILY};
}}

/* Main app */
.stApp {{
    background: linear-gradient(135deg, {COLORS["bg_primary"]} 0%, #0d1929 100%);
    font-family: var(--font-family);
}}

/* Hide Streamlit default branding */
#MainMenu, footer, header {{visibility: hidden;}}

/* Sidebar */
[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, #0a0e1a 0%, #0f1729 100%);
    border-right: 1px solid rgba(0, 212, 255, 0.1);
}}

[data-testid="stSidebar"] .stMarkdown h1,
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3 {{
    color: {COLORS["electric_blue"]};
    font-size: 0.85rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    font-weight: 600;
}}

/* Metric cards */
[data-testid="metric-container"] {{
    background: {COLORS["bg_card"]};
    border: 1px solid rgba(0, 212, 255, 0.15);
    border-radius: 12px;
    padding: 1rem !important;
    box-shadow: 0 4px 24px rgba(0, 0, 0, 0.4),
                inset 0 1px 0 rgba(255, 255, 255, 0.05);
    backdrop-filter: blur(10px);
    transition: border-color 0.2s ease;
}}

[data-testid="metric-container"]:hover {{
    border-color: rgba(0, 212, 255, 0.4);
}}

[data-testid="metric-container"] [data-testid="stMetricLabel"] {{
    color: {COLORS["text_secondary"]} !important;
    font-size: 0.75rem !important;
    font-weight: 500 !important;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}}

[data-testid="metric-container"] [data-testid="stMetricValue"] {{
    color: {COLORS["text_primary"]} !important;
    font-size: 1.6rem !important;
    font-weight: 700 !important;
    line-height: 1.2;
}}

[data-testid="metric-container"] [data-testid="stMetricDelta"] {{
    font-size: 0.75rem !important;
    font-weight: 500 !important;
}}

/* Buttons */
.stButton > button {{
    background: linear-gradient(135deg, {COLORS["electric_blue"]}, {COLORS["purple"]});
    color: white;
    border: none;
    border-radius: 8px;
    font-family: var(--font-family);
    font-weight: 600;
    font-size: 0.875rem;
    padding: 0.5rem 1.5rem;
    transition: all 0.2s ease;
    box-shadow: 0 4px 15px rgba(0, 212, 255, 0.3);
}}

.stButton > button:hover {{
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(0, 212, 255, 0.5);
}}

/* Selectbox, inputs */
.stSelectbox > div > div,
.stMultiSelect > div > div {{
    background: {COLORS["bg_card"]};
    border: 1px solid rgba(0, 212, 255, 0.2);
    border-radius: 8px;
    color: {COLORS["text_primary"]};
}}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {{
    background: {COLORS["bg_secondary"]};
    border-radius: 10px;
    padding: 4px;
    border: 1px solid rgba(255, 255, 255, 0.05);
}}

.stTabs [data-baseweb="tab"] {{
    color: {COLORS["text_secondary"]};
    font-weight: 500;
    font-size: 0.875rem;
    border-radius: 8px;
    transition: all 0.2s;
}}

.stTabs [aria-selected="true"] {{
    background: linear-gradient(135deg, rgba(0, 212, 255, 0.2), rgba(155, 89, 182, 0.2));
    color: {COLORS["electric_blue"]} !important;
    border: 1px solid rgba(0, 212, 255, 0.3);
}}

/* DataFrames */
.stDataFrame {{
    border: 1px solid rgba(0, 212, 255, 0.1);
    border-radius: 10px;
    overflow: hidden;
}}

/* Progress bar */
.stProgress > div > div > div {{
    background: linear-gradient(90deg, {COLORS["electric_blue"]}, {COLORS["neon_green"]});
}}

/* Headers */
h1 {{
    background: linear-gradient(135deg, {COLORS["electric_blue"]}, {COLORS["neon_green"]});
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 800;
    letter-spacing: -0.02em;
}}

h2 {{
    color: {COLORS["text_primary"]};
    font-weight: 700;
    border-bottom: 1px solid rgba(0, 212, 255, 0.2);
    padding-bottom: 0.5rem;
}}

h3 {{
    color: {COLORS["electric_blue"]};
    font-weight: 600;
    font-size: 1rem;
}}

/* Cards */
.metric-card {{
    background: {COLORS["bg_card"]};
    border: 1px solid rgba(0, 212, 255, 0.15);
    border-radius: 12px;
    padding: 1.25rem;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
}}

.status-badge {{
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.04em;
}}

.badge-green {{
    background: rgba(0, 255, 136, 0.15);
    color: {COLORS["neon_green"]};
    border: 1px solid rgba(0, 255, 136, 0.3);
}}

.badge-red {{
    background: rgba(255, 107, 107, 0.15);
    color: {COLORS["coral"]};
    border: 1px solid rgba(255, 107, 107, 0.3);
}}

.badge-blue {{
    background: rgba(0, 212, 255, 0.15);
    color: {COLORS["electric_blue"]};
    border: 1px solid rgba(0, 212, 255, 0.3);
}}

/* Divider */
hr {{
    border: none;
    border-top: 1px solid rgba(255, 255, 255, 0.06);
    margin: 1.5rem 0;
}}

/* Alert boxes */
.stAlert {{
    border-radius: 10px;
    border-left: 4px solid {COLORS["electric_blue"]};
}}

/* Spinner */
.stSpinner > div {{
    border-top-color: {COLORS["electric_blue"]} !important;
}}

/* Scrollbar */
::-webkit-scrollbar {{
    width: 6px;
    height: 6px;
}}
::-webkit-scrollbar-track {{ background: {COLORS["bg_primary"]}; }}
::-webkit-scrollbar-thumb {{
    background: rgba(0, 212, 255, 0.3);
    border-radius: 3px;
}}
::-webkit-scrollbar-thumb:hover {{
    background: rgba(0, 212, 255, 0.6);
}}
</style>
"""


def get_css() -> str:
    """Return the full Streamlit CSS injection string."""
    return STREAMLIT_CSS
