"""
dashboard/components.py
-----------------------
Ultra-premium UI components, design system, and topology visualizer
for the MatRisk AI executive dashboard.
"""

from __future__ import annotations

from typing import Optional, List, Dict, Any
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st


# ---------------------------------------------------------------------------
# Theme & Palette System
# ---------------------------------------------------------------------------

PALETTE = {
    "primary": "#00F2FE",        # Cyber Neon Cyan
    "secondary": "#4FACFE",      # Electric Sapphire Blue
    "accent_purple": "#7F00FF",  # Deep Tech Violet
    "accent_magenta": "#E100FF", # Radiant Neon Magenta
    "accent": "#FF007F",         # Neon Pink Accent
    "warning": "#FFB800",        # Glowing Amber Gold
    "danger": "#FF3366",         # Crimson Warning
    "success": "#00F5A0",        # Emerald Cyber Green
    "bg_dark": "#070B14",        # Deep Obsidian Void
    "bg_card": "#0F172A",        # Rich Frosted Slate
    "bg_card_trans": "rgba(15, 23, 42, 0.75)",
    "border": "rgba(51, 65, 85, 0.5)",
    "border_glow": "rgba(0, 242, 254, 0.35)",
    "text": "#F8FAFC",           # 100% Crisp Slate White
    "text_muted": "#94A3B8",     # High-contrast readable secondary
    "muted": "#94A3B8",
}

PLOTLY_TEMPLATE = "plotly_dark"


def apply_global_css() -> None:
    """Inject ultra-premium custom CSS styling, animations, and typography."""
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

        /* Root variables */
        :root {{
            --primary: {PALETTE['primary']};
            --secondary: {PALETTE['secondary']};
            --accent-purple: {PALETTE['accent_purple']};
            --text-main: {PALETTE['text']};
            --text-muted: {PALETTE['muted']};
            --card-bg: {PALETTE['bg_card_trans']};
            --border-color: {PALETTE['border']};
        }}

        /* Base app styling */
        html, body, [class*="css"] {{
            font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
            background-color: {PALETTE["bg_dark"]} !important;
            color: {PALETTE["text"]} !important;
        }}

        /* Overall background mesh glow */
        .stApp {{
            background: radial-gradient(circle at 15% 15%, rgba(0, 242, 254, 0.05) 0%, transparent 45%),
                        radial-gradient(circle at 85% 85%, rgba(127, 0, 255, 0.05) 0%, transparent 45%),
                        radial-gradient(circle at 50% 50%, rgba(15, 23, 42, 0.6) 0%, transparent 70%),
                        {PALETTE["bg_dark"]};
            background-attachment: fixed;
        }}

        /* Container width & layout */
        .main .block-container {{
            padding-top: 1.2rem !important;
            padding-bottom: 3rem !important;
            padding-left: 2rem !important;
            padding-right: 2rem !important;
            max-width: 1550px !important;
        }}

        /* Typography - Zero truncation, total legibility */
        h1, h2, h3, h4, h5, h6 {{
            color: #FFFFFF !important;
            font-weight: 700 !important;
            letter-spacing: -0.02em !important;
        }}

        h1 {{
            font-size: 2.2rem !important;
        }}

        h2 {{
            font-size: 1.7rem !important;
            border-bottom: 1px solid rgba(255, 255, 255, 0.08);
            padding-bottom: 0.4rem;
            margin-top: 1.5rem !important;
        }}

        h3 {{
            font-size: 1.3rem !important;
            color: #F1F5F9 !important;
        }}

        p, span, div, li {{
            color: {PALETTE['text']} !important;
            font-size: 0.95rem;
            line-height: 1.6;
        }}

        .stMarkdown p {{
            color: #E2E8F0 !important;
        }}

        /* Sidebar Styling */
        section[data-testid="stSidebar"] {{
            background: linear-gradient(180deg, #070B14 0%, #0F172A 100%) !important;
            border-right: 1px solid rgba(51, 65, 85, 0.4) !important;
            box-shadow: 4px 0 24px rgba(0, 0, 0, 0.3) !important;
        }}

        section[data-testid="stSidebar"] hr {{
            border-color: rgba(51, 65, 85, 0.5) !important;
        }}

        /* Native Streamlit Metric overrides - CRITICAL: No more '...' truncation! */
        [data-testid="stMetric"] {{
            background: rgba(15, 23, 42, 0.75) !important;
            backdrop-filter: blur(16px) !important;
            -webkit-backdrop-filter: blur(16px) !important;
            border: 1px solid rgba(51, 65, 85, 0.6) !important;
            border-radius: 14px !important;
            padding: 1.1rem 1.2rem !important;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.35) !important;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
            position: relative;
            overflow: visible !important;
        }}

        [data-testid="stMetric"]:hover {{
            transform: translateY(-3px) !important;
            border-color: rgba(0, 242, 254, 0.5) !important;
            box-shadow: 0 8px 30px rgba(0, 242, 254, 0.15) !important;
        }}

        [data-testid="stMetricLabel"] {{
            color: #94A3B8 !important;
            font-size: 0.82rem !important;
            font-weight: 600 !important;
            text-transform: uppercase !important;
            letter-spacing: 0.05em !important;
            white-space: normal !important;
            overflow: visible !important;
            text-overflow: clip !important;
            line-height: 1.3 !important;
        }}

        [data-testid="stMetricValue"] {{
            font-family: 'JetBrains Mono', 'Plus Jakarta Sans', monospace !important;
            color: #00F2FE !important;
            font-size: 1.65rem !important;
            font-weight: 700 !important;
            letter-spacing: -0.03em !important;
            white-space: normal !important;
            overflow: visible !important;
            text-overflow: clip !important;
            word-break: break-word !important;
            text-shadow: 0 0 20px rgba(0, 242, 254, 0.35) !important;
        }}

        [data-testid="stMetricDelta"] {{
            font-weight: 600 !important;
            font-size: 0.85rem !important;
        }}

        /* Custom MatRisk Glassmorphic Card */
        .matrisk-card {{
            background: rgba(15, 23, 42, 0.85) !important;
            backdrop-filter: blur(16px) !important;
            -webkit-backdrop-filter: blur(16px) !important;
            border: 1px solid rgba(51, 65, 85, 0.6) !important;
            border-radius: 12px !important;
            padding: 0.85rem 1rem !important;
            margin-bottom: 0.8rem !important;
            box-shadow: 0 4px 18px rgba(0, 0, 0, 0.35) !important;
            transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
            box-sizing: border-box !important;
            width: 100% !important;
            max-width: 100% !important;
            overflow: hidden !important;
        }}

        .matrisk-card:hover {{
            border-color: rgba(0, 242, 254, 0.5) !important;
            box-shadow: 0 8px 25px rgba(0, 242, 254, 0.15) !important;
            transform: translateY(-2px) !important;
        }}

        .matrisk-card-header {{
            display: flex !important;
            align-items: center !important;
            justify-content: space-between !important;
            margin-bottom: 0.4rem !important;
            flex-wrap: wrap !important;
            gap: 4px !important;
        }}

        .matrisk-card-title {{
            font-size: 0.72rem !important;
            font-weight: 700 !important;
            color: #94A3B8 !important;
            text-transform: uppercase !important;
            letter-spacing: 0.05em !important;
            line-height: 1.2 !important;
        }}

        .matrisk-card-value {{
            font-family: 'JetBrains Mono', monospace !important;
            font-size: 1.45rem !important;
            font-weight: 700 !important;
            color: #00F2FE !important;
            letter-spacing: -0.03em !important;
            text-shadow: 0 0 18px rgba(0, 242, 254, 0.35) !important;
            margin: 0.15rem 0 !important;
            line-height: 1.2 !important;
            white-space: nowrap !important;
            overflow: hidden !important;
            text-overflow: ellipsis !important;
        }}

        .matrisk-card-desc {{
            font-size: 0.76rem !important;
            color: #94A3B8 !important;
            margin-top: 0.25rem !important;
            line-height: 1.3 !important;
            white-space: nowrap !important;
            overflow: hidden !important;
            text-overflow: ellipsis !important;
        }}

        /* Tab styling - Wider, no overflow truncation */
        .stTabs [data-baseweb="tab-list"] {{
            gap: 10px !important;
            background: rgba(15, 23, 42, 0.5) !important;
            padding: 6px 8px !important;
            border-radius: 12px !important;
            border: 1px solid rgba(51, 65, 85, 0.5) !important;
            overflow-x: auto !important;
        }}

        .stTabs [data-baseweb="tab"] {{
            background: transparent !important;
            border-radius: 8px !important;
            border: 1px solid transparent !important;
            color: #94A3B8 !important;
            font-weight: 600 !important;
            font-size: 0.9rem !important;
            padding: 8px 16px !important;
            transition: all 0.25s ease !important;
            white-space: nowrap !important;
        }}

        .stTabs [data-baseweb="tab"]:hover {{
            background: rgba(255, 255, 255, 0.05) !important;
            color: #F8FAFC !important;
        }}

        .stTabs [aria-selected="true"] {{
            background: linear-gradient(135deg, rgba(0, 242, 254, 0.2) 0%, rgba(79, 172, 254, 0.15) 100%) !important;
            color: #00F2FE !important;
            border: 1px solid rgba(0, 242, 254, 0.4) !important;
            box-shadow: 0 0 15px rgba(0, 242, 254, 0.2) !important;
        }}

        /* Hero Banner with animated mesh & glow */
        .hero-banner {{
            position: relative;
            background: linear-gradient(135deg, rgba(15, 23, 42, 0.9) 0%, rgba(7, 11, 20, 0.95) 100%);
            border: 1px solid rgba(0, 242, 254, 0.3);
            border-radius: 18px;
            padding: 2.2rem 2.5rem;
            margin-bottom: 2rem;
            box-shadow: 0 10px 40px rgba(0, 0, 0, 0.5), inset 0 1px 0 rgba(255, 255, 255, 0.1);
            overflow: hidden;
        }}

        .hero-banner::before {{
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 200%;
            height: 2px;
            background: linear-gradient(90deg, transparent, #00F2FE, #7F00FF, transparent);
            animation: shimmer-border 6s infinite linear;
        }}

        @keyframes shimmer-border {{
            0% {{ transform: translateX(-50%); }}
            100% {{ transform: translateX(50%); }}
        }}

        .hero-banner-inner {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            flex-wrap: wrap;
            gap: 1.5rem;
        }}

        .hero-title-box {{
            flex: 1;
            min-width: 300px;
        }}

        .hero-banner h1 {{
            background: linear-gradient(135deg, #00F2FE 0%, #4FACFE 50%, #E100FF 100%);
            -webkit-background-clip: text !important;
            -webkit-text-fill-color: transparent !important;
            font-size: 2.6rem !important;
            font-weight: 800 !important;
            letter-spacing: -0.03em !important;
            margin: 0 0 0.5rem 0 !important;
        }}

        .hero-banner p {{
            color: #CBD5E1 !important;
            font-size: 1.05rem !important;
            margin: 0 !important;
            max-width: 800px;
        }}

        .hero-status-pill {{
            display: inline-flex;
            align-items: center;
            gap: 8px;
            background: rgba(0, 245, 160, 0.12);
            border: 1px solid rgba(0, 245, 160, 0.35);
            padding: 6px 14px;
            border-radius: 30px;
            font-size: 0.8rem;
            font-weight: 600;
            color: #00F5A0 !important;
            box-shadow: 0 0 16px rgba(0, 245, 160, 0.15);
        }}

        /* Animated pulsing dot */
        .pulse-indicator {{
            display: inline-block;
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background-color: #00F5A0;
            position: relative;
        }}

        .pulse-indicator::after {{
            content: '';
            position: absolute;
            top: -4px;
            left: -4px;
            width: 16px;
            height: 16px;
            border-radius: 50%;
            background-color: #00F5A0;
            opacity: 0.6;
            animation: pulse-ring 1.8s cubic-bezier(0.215, 0.61, 0.355, 1) infinite;
        }}

        @keyframes pulse-ring {{
            0% {{ transform: scale(0.6); opacity: 0.8; }}
            100% {{ transform: scale(2.2); opacity: 0; }}
        }}

        /* Form widgets & Controls */
        [data-testid="stWidgetLabel"] label, [data-testid="stWidgetLabel"] p {{
            color: #F8FAFC !important;
            font-weight: 600 !important;
            font-size: 0.88rem !important;
        }}

        div[data-baseweb="select"] > div {{
            background-color: #0F172A !important;
            border: 1px solid rgba(51, 65, 85, 0.8) !important;
            border-radius: 8px !important;
            color: #F8FAFC !important;
        }}

        div[data-baseweb="select"] span {{
            color: #F8FAFC !important;
        }}

        div[data-baseweb="popover"] ul {{
            background-color: #0F172A !important;
            border: 1px solid rgba(51, 65, 85, 0.8) !important;
        }}

        div[data-baseweb="popover"] li {{
            color: #F8FAFC !important;
        }}

        input[type="text"], input[type="number"] {{
            background-color: #0F172A !important;
            border: 1px solid rgba(51, 65, 85, 0.8) !important;
            color: #F8FAFC !important;
            border-radius: 8px !important;
        }}

        /* Buttons */
        .stButton button {{
            background: linear-gradient(135deg, #00F2FE 0%, #4FACFE 100%) !important;
            color: #070B14 !important;
            font-weight: 700 !important;
            border: none !important;
            border-radius: 10px !important;
            padding: 0.55rem 1.4rem !important;
            box-shadow: 0 4px 18px rgba(0, 242, 254, 0.35) !important;
            transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
        }}

        .stButton button:hover {{
            box-shadow: 0 6px 25px rgba(0, 242, 254, 0.55) !important;
            transform: translateY(-2px) !important;
            color: #000000 !important;
        }}

        /* DataTables */
        .stDataFrame {{
            border: 1px solid rgba(51, 65, 85, 0.6) !important;
            border-radius: 12px !important;
            overflow: hidden !important;
        }}

        /* Alert and Information callout cards */
        .matrisk-info {{
            background: linear-gradient(135deg, rgba(0, 242, 254, 0.1) 0%, rgba(79, 172, 254, 0.05) 100%);
            border-left: 4px solid #00F2FE;
            border-top: 1px solid rgba(0, 242, 254, 0.25);
            border-right: 1px solid rgba(0, 242, 254, 0.25);
            border-bottom: 1px solid rgba(0, 242, 254, 0.25);
            border-radius: 10px;
            padding: 0.9rem 1.25rem;
            margin: 0.8rem 0;
            color: #F1F5F9 !important;
        }}

        .matrisk-alert {{
            background: linear-gradient(135deg, rgba(255, 184, 0, 0.12) 0%, rgba(255, 51, 102, 0.06) 100%);
            border-left: 4px solid #FFB800;
            border-top: 1px solid rgba(255, 184, 0, 0.25);
            border-right: 1px solid rgba(255, 184, 0, 0.25);
            border-bottom: 1px solid rgba(255, 184, 0, 0.25);
            border-radius: 10px;
            padding: 0.9rem 1.25rem;
            margin: 0.8rem 0;
            color: #F8FAFC !important;
        }}

        /* Badges */
        .badge {{
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 3px 10px;
            border-radius: 20px;
            font-size: 0.76rem;
            font-weight: 700;
            letter-spacing: 0.03em;
            text-transform: uppercase;
        }}
        .badge-green {{
            background: rgba(0, 245, 160, 0.15);
            color: #00F5A0;
            border: 1px solid rgba(0, 245, 160, 0.4);
        }}
        .badge-yellow {{
            background: rgba(255, 184, 0, 0.15);
            color: #FFB800;
            border: 1px solid rgba(255, 184, 0, 0.4);
        }}
        .badge-red {{
            background: rgba(255, 51, 102, 0.15);
            color: #FF3366;
            border: 1px solid rgba(255, 51, 102, 0.4);
        }}
        .badge-blue {{
            background: rgba(0, 242, 254, 0.15);
            color: #00F2FE;
            border: 1px solid rgba(0, 242, 254, 0.4);
        }}

        /* Footer */
        .dashboard-footer {{
            text-align: center;
            color: #94A3B8;
            font-size: 0.82rem;
            margin-top: 3.5rem;
            padding: 1.5rem 0;
            border-top: 1px solid rgba(51, 65, 85, 0.4);
            letter-spacing: 0.02em;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Reusable UI Elements
# ---------------------------------------------------------------------------

def _render_html(html_str: str) -> None:
    """Helper to render raw HTML reliably via st.html or fallback to st.markdown."""
    if hasattr(st, "html"):
        st.html(html_str)
    else:
        st.markdown(html_str, unsafe_allow_html=True)


def hero_banner(title: str, subtitle: str, badge_text: str = "PRODUCTION READY · LOCAL ENGINE") -> None:
    """Render executive hero banner with glowing mesh and live operational badge."""
    html = (
        f'<div class="hero-banner">'
        f'<div class="hero-banner-inner">'
        f'<div class="hero-title-box">'
        f'<h1>{title}</h1>'
        f'<p>{subtitle}</p>'
        f'</div>'
        f'<div><span class="hero-status-pill">'
        f'<span class="pulse-indicator"></span>'
        f'{badge_text}'
        f'</span></div>'
        f'</div>'
        f'</div>'
    )
    _render_html(html)


def info_box(text: str) -> None:
    """Render sleek high-contrast info box."""
    _render_html(f'<div class="matrisk-info">💡 <b>INTELLIGENCE NOTE:</b> {text}</div>')


def warning_box(text: str) -> None:
    """Render sleek high-contrast warning box."""
    _render_html(f'<div class="matrisk-alert">⚠️ <b>MODEL LIMITATION & GOVERNANCE:</b> {text}</div>')


def not_implemented(feature: str, reason: str = "") -> None:
    """Standard 'not yet implemented' placeholder with clear visual status."""
    sub = reason or 'Under active empirical formulation. Data contract not yet finalized in production.'
    html = (
        f'<div style="background: rgba(255, 184, 0, 0.08); border: 1px dashed rgba(255, 184, 0, 0.4); border-radius: 12px; padding: 1.2rem; margin: 1rem 0;">'
        f'<div style="display:flex; align-items:center; gap:8px;">'
        f'<span class="badge badge-yellow">RESEARCH PIPELINE</span>'
        f'<span style="font-weight:700; color:#FFB800; font-size:1rem;">{feature}</span>'
        f'</div>'
        f'<div style="color:#CBD5E1; font-size:0.88rem; margin-top:0.4rem;">{sub}</div>'
        f'</div>'
    )
    _render_html(html)


def status_badge(status: str) -> str:
    """Return an animated glowing badge for component statuses."""
    mapping = {
        "complete": ("badge-green", "Complete"),
        "usable with caveats": ("badge-yellow", "Usable (Caveats)"),
        "demonstration-grade": ("badge-yellow", "Demo Grade"),
        "research-grade": ("badge-blue", "Research Grade"),
        "not implemented": ("badge-red", "Not Implemented"),
    }
    cls, label = mapping.get(status.lower(), ("badge-blue", status))
    return f'<span class="badge {cls}">{label}</span>'


def kpi_card(title: str, value: str, subtitle: str = "", delta: Optional[str] = None) -> str:
    """Generate HTML for a non-truncating, glassmorphic executive KPI card."""
    delta_html = ""
    if delta:
        color = "#00F5A0" if not delta.startswith("-") else "#FF3366"
        delta_html = f'<span style="color:{color}; font-size:0.75rem; font-weight:700; margin-left:6px;">{delta}</span>'

    desc_html = f'<div class="matrisk-card-desc">{subtitle}</div>' if subtitle else ""

    return (
        f'<div class="matrisk-card">'
        f'<div class="matrisk-card-header">'
        f'<span class="matrisk-card-title">{title}</span>'
        f'{delta_html}'
        f'</div>'
        f'<div class="matrisk-card-value">{value}</div>'
        f'{desc_html}'
        f'</div>'
    )


def metric_row(metrics: list[dict]) -> None:
    """Render a responsive row of KPI cards that never truncate."""
    cols = st.columns(len(metrics))
    for col, m in zip(cols, metrics):
        with col:
            card_html = kpi_card(
                title=m.get("label", ""),
                value=str(m.get("value", "")),
                subtitle=m.get("help", ""),
                delta=m.get("delta"),
            )
            _render_html(card_html)


# ---------------------------------------------------------------------------
# Interactive System Topology & Architecture Flow Visualizer
# ---------------------------------------------------------------------------

def render_system_topology(selected_layer: Optional[str] = None, highlight_path: bool = False) -> go.Figure:
    """Build an interactive, animated-style system topology graph of MatRisk AI.

    Connects:
      Tier 0: Data Asset Ingestion (DS1-DS6)
      Tier 1: Feature Transformation & Math Pipeline
      Tier 2: AI & ML Core Models
      Tier 3: Financial & Stress Simulation Engines
      Tier 4: Enterprise Capital & Engineering Decisions
    """
    # Define topology nodes with spacious horizontal tiers (spacing = 2.2)
    nodes = [
        # Tier 0: Ingestion
        {"id": "DS1", "name": "DS1: Materials", "tier": 0.0, "y": 5.0, "type": "data", "status": "Online (1,200 alloys)", "detail": "DFT & experimental bulk/shear moduli"},
        {"id": "DS2", "name": "DS2: Commodities", "tier": 0.0, "y": 4.0, "type": "data", "status": "Online (8 metals)", "detail": "10-year spot & futures time series"},
        {"id": "DS3", "name": "DS3: Bridge Assets", "tier": 0.0, "y": 3.0, "type": "data", "status": "Online (5,000 bridges)", "detail": "National inventory specs & condition"},
        {"id": "DS4", "name": "DS4: Macro & Supply", "tier": 0.0, "y": 2.0, "type": "data", "status": "Online (Fred macro)", "detail": "Tariffs, inflation & supply indices"},
        {"id": "DS5", "name": "DS5: Climate/Weather", "tier": 0.0, "y": 1.0, "type": "data", "status": "Online (NOAA)", "detail": "Freeze-thaw cycles & coastal salinity"},
        {"id": "DS6", "name": "DS6: NBI Inspections", "tier": 0.0, "y": 0.0, "type": "data", "status": "Online (30yr hist)", "detail": "Longitudinal degradation ratings"},

        # Tier 1: Featurizers
        {"id": "F_COMP", "name": "Magpie Featurizer", "tier": 2.2, "y": 4.8, "type": "feat", "status": "Active (Magpie)", "detail": "Stoichiometry & electronegativity"},
        {"id": "F_GARCH", "name": "GARCH Volatility", "tier": 2.2, "y": 3.8, "type": "feat", "status": "Active (Fat-tail)", "detail": "Conditional variance & tail-risk"},
        {"id": "F_EXPOSURE", "name": "Basel EAD Parser", "tier": 2.2, "y": 2.8, "type": "feat", "status": "Active (Basel II)", "detail": "Structural EAD & bridge replacement cost"},
        {"id": "F_CLIMATE", "name": "Stress Vectorizer", "tier": 2.2, "y": 1.2, "type": "feat", "status": "Active (SC-01..05)", "detail": "Cross-correlation shock factors"},

        # Tier 2: AI & ML Core
        {"id": "M_BULK", "name": "Bulk Modulus (EBM)", "tier": 4.4, "y": 4.8, "type": "ai", "status": "Validated (R²=0.88)", "detail": "Explainable Boosting Machine (EBM)"},
        {"id": "M_SHEAR", "name": "Shear Modulus (GBM)", "tier": 4.4, "y": 3.8, "type": "ai", "status": "Validated (R²=0.84)", "detail": "Gradient Boosting Regressor"},
        {"id": "M_DEGRAD", "name": "CatBoost Degradation", "tier": 4.4, "y": 2.8, "type": "ai", "status": "Validated (ROC=0.91)", "detail": "CatBoost monotonic degradation"},
        {"id": "M_CREDIT", "name": "Calibrated Credit PD", "tier": 4.4, "y": 1.5, "type": "ai", "status": "Validated (1.06%/yr)", "detail": "Isotonic Calibrated Logistic Model"},

        # Tier 3: Risk & Stress Simulator
        {"id": "E_EL", "name": "Closed-Form EL", "tier": 6.6, "y": 4.2, "type": "risk", "status": "Active (EL=$5.32M)", "detail": "EL = PD × LGD × EAD Identity Engine"},
        {"id": "E_MC", "name": "Monte Carlo VaR", "tier": 6.6, "y": 3.0, "type": "risk", "status": "Active (VaR95=$8.1M)", "detail": "50,000 correlated joint simulations"},
        {"id": "E_STRESS", "name": "Macro Stress Shocks", "tier": 6.6, "y": 1.8, "type": "risk", "status": "Active (SC-001..005)", "detail": "Severe stagflation & climate stress tests"},
        {"id": "E_ESG", "name": "ESG Green Premium", "tier": 6.6, "y": 0.6, "type": "risk", "status": "Active (Pareto)", "detail": "Embodied carbon vs modulus tradeoff"},

        # Tier 4: Enterprise Decisions
        {"id": "D_CAP", "name": "Capital Reserves", "tier": 8.8, "y": 4.0, "type": "decision", "status": "Executive Target", "detail": "$4.7B portfolio economic buffer"},
        {"id": "D_RETRO", "name": "Retrofit Priority", "tier": 8.8, "y": 2.8, "type": "decision", "status": "Targeted Assets", "detail": "Ranked by Expected Loss contribution"},
        {"id": "D_HEDGE", "name": "Dynamic Hedging", "tier": 8.8, "y": 1.8, "type": "decision", "status": "VaR Optimization", "detail": "Quantile hedging for Copper & Steel"},
        {"id": "D_ALLOY", "name": "Green Alloy Matching", "tier": 8.8, "y": 0.8, "type": "decision", "status": "Materials Search", "detail": "Inverse lookup for low-carbon replacement"},
    ]

    # Connections / Pipelines
    edges = [
        ("DS1", "F_COMP"), ("DS2", "F_GARCH"), ("DS3", "F_EXPOSURE"), ("DS4", "F_CLIMATE"), ("DS5", "F_CLIMATE"), ("DS6", "F_EXPOSURE"),
        ("F_COMP", "M_BULK"), ("F_COMP", "M_SHEAR"), ("F_GARCH", "M_CREDIT"), ("F_EXPOSURE", "M_DEGRAD"), ("F_CLIMATE", "M_DEGRAD"),
        ("M_BULK", "E_ESG"), ("M_SHEAR", "E_ESG"), ("M_DEGRAD", "M_CREDIT"),
        ("M_CREDIT", "E_EL"), ("M_CREDIT", "E_MC"), ("F_CLIMATE", "E_STRESS"), ("F_EXPOSURE", "E_EL"),
        ("E_EL", "D_CAP"), ("E_MC", "D_CAP"), ("E_EL", "D_RETRO"), ("E_STRESS", "D_CAP"),
        ("F_GARCH", "D_HEDGE"), ("E_ESG", "D_ALLOY"),
    ]

    node_map = {n["id"]: n for n in nodes}

    fig = go.Figure()

    # Draw transmission lines
    for src, dst in edges:
        s = node_map[src]
        d = node_map[dst]
        fig.add_trace(go.Scatter(
            x=[s["tier"], (s["tier"] + d["tier"]) / 2, d["tier"]],
            y=[s["y"], (s["y"] + d["y"]) / 2, d["y"]],
            mode="lines",
            line=dict(
                width=1.6,
                color="rgba(0, 242, 254, 0.22)" if not highlight_path else "rgba(79, 172, 254, 0.4)",
            ),
            hoverinfo="none",
            showlegend=False,
        ))

    # Node styling by category
    type_colors = {
        "data": "#00F2FE",      # Neon Cyan
        "feat": "#4FACFE",      # Blue
        "ai": "#E100FF",        # Violet / Magenta
        "risk": "#FFB800",      # Amber Gold
        "decision": "#00F5A0",  # Emerald Green
    }

    type_symbols = {
        "data": "square",
        "feat": "diamond",
        "ai": "hexagon",
        "risk": "circle",
        "decision": "star",
    }

    for ntype, color in type_colors.items():
        type_nodes = [n for n in nodes if n["type"] == ntype]
        fig.add_trace(go.Scatter(
            x=[n["tier"] for n in type_nodes],
            y=[n["y"] for n in type_nodes],
            mode="markers+text",
            name=ntype.upper(),
            text=[f"  {n['name']}" for n in type_nodes],
            textposition="middle right",
            textfont=dict(family="Plus Jakarta Sans", size=11, color="#F8FAFC"),
            marker=dict(
                symbol=type_symbols[ntype],
                size=16,
                color=color,
                line=dict(width=2, color="#FFFFFF"),
            ),
            customdata=[[n["id"], n["status"], n["detail"]] for n in type_nodes],
            hovertemplate=(
                "<b>%{text}</b><br>"
                "<b>Status:</b> %{customdata[1]}<br>"
                "<b>Intelligence:</b> %{customdata[2]}<br>"
                "<extra></extra>"
            ),
        ))

    # Layout tuning with comfortable spacing
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        height=540,
        plot_bgcolor="rgba(15, 23, 42, 0.4)",
        paper_bgcolor="rgba(0, 0, 0, 0)",
        xaxis=dict(
            showgrid=True,
            gridcolor="rgba(51, 65, 85, 0.25)",
            tickmode="array",
            tickvals=[0.0, 2.2, 4.4, 6.6, 8.8],
            ticktext=[
                "<b>1. Data Ingestion</b>",
                "<b>2. Featurization</b>",
                "<b>3. AI / ML Models</b>",
                "<b>4. Risk Engine</b>",
                "<b>5. Decision Layer</b>"
            ],
            tickfont=dict(family="Plus Jakarta Sans", size=12, color="#00F2FE"),
            range=[-0.6, 10.5],
            zeroline=False,
        ),
        yaxis=dict(
            showgrid=False,
            showticklabels=False,
            zeroline=False,
            range=[-0.7, 5.8],
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            font=dict(color="#CBD5E1", size=11),
            bgcolor="rgba(15, 23, 42, 0.8)",
            bordercolor="rgba(51, 65, 85, 0.6)",
            borderwidth=1,
        ),
        margin=dict(l=30, r=40, t=50, b=30),
    )

    return fig


# ---------------------------------------------------------------------------
# Plotly Helpers with Neon Glow & Maximum Readability
# ---------------------------------------------------------------------------

def plotly_bar(
    df: pd.DataFrame,
    x: str,
    y: str,
    title: str,
    color_col: Optional[str] = None,
    xaxis_title: Optional[str] = None,
    yaxis_title: Optional[str] = None,
    orientation: str = "v",
    height: int = 420,
) -> go.Figure:
    """Render glowing cyber-bar chart with high contrast."""
    fig = px.bar(
        df,
        x=x,
        y=y,
        color=color_col,
        title=f"<b>{title}</b>",
        orientation=orientation,
        template=PLOTLY_TEMPLATE,
        height=height,
        color_discrete_sequence=[PALETTE["primary"], PALETTE["secondary"], PALETTE["accent_magenta"]],
    )
    fig.update_layout(
        plot_bgcolor="rgba(15, 23, 42, 0.3)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Plus Jakarta Sans", color=PALETTE["text"], size=12),
        title_font=dict(size=15, color="#FFFFFF"),
        xaxis=dict(
            title=dict(text=xaxis_title or x, font=dict(color="#CBD5E1")),
            gridcolor="rgba(51, 65, 85, 0.25)",
            tickfont=dict(color="#CBD5E1"),
        ),
        yaxis=dict(
            title=dict(text=yaxis_title or y, font=dict(color="#CBD5E1")),
            gridcolor="rgba(51, 65, 85, 0.25)",
            tickfont=dict(color="#CBD5E1"),
        ),
        margin=dict(l=30, r=30, t=55, b=30),
    )
    return fig


def plotly_line(
    df: pd.DataFrame,
    x: str,
    y: str | list[str],
    title: str,
    color_col: Optional[str] = None,
    height: int = 440,
) -> go.Figure:
    """Render luminous multi-line chart with glow and clear legend."""
    vibrant_colors = [
        "#00F2FE",  # Cyan
        "#E100FF",  # Magenta
        "#00F5A0",  # Emerald Green
        "#FFB800",  # Amber Gold
        "#4FACFE",  # Blue
        "#FF3366",  # Crimson
        "#A855F7",  # Purple
        "#38BDF8",  # Sky
    ]

    if isinstance(y, list):
        fig = go.Figure()
        for i, col in enumerate(y):
            c = vibrant_colors[i % len(vibrant_colors)]
            fig.add_trace(go.Scatter(
                x=df[x],
                y=df[col],
                name=col,
                mode="lines",
                line=dict(color=c, width=2.2),
                hovertemplate=f"<b>{col}</b>: %{{y:.2f}}<extra></extra>",
            ))
        fig.update_layout(
            title=dict(text=f"<b>{title}</b>", font=dict(size=16, color="#FFFFFF")),
            template=PLOTLY_TEMPLATE,
            height=height,
            plot_bgcolor="rgba(15, 23, 42, 0.3)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Plus Jakarta Sans", color=PALETTE["text"]),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                bgcolor="rgba(15, 23, 42, 0.6)",
                bordercolor="rgba(51, 65, 85, 0.4)",
                font=dict(color="#F8FAFC"),
            ),
            xaxis=dict(gridcolor="rgba(51, 65, 85, 0.25)", tickfont=dict(color="#CBD5E1")),
            yaxis=dict(gridcolor="rgba(51, 65, 85, 0.25)", tickfont=dict(color="#CBD5E1")),
            margin=dict(l=30, r=30, t=60, b=30),
        )
    else:
        fig = px.line(
            df,
            x=x,
            y=y,
            color=color_col,
            title=f"<b>{title}</b>",
            template=PLOTLY_TEMPLATE,
            height=height,
            color_discrete_sequence=vibrant_colors,
        )
        fig.update_layout(
            plot_bgcolor="rgba(15, 23, 42, 0.3)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Plus Jakarta Sans", color=PALETTE["text"]),
            title_font=dict(size=16, color="#FFFFFF"),
            xaxis=dict(gridcolor="rgba(51, 65, 85, 0.25)", tickfont=dict(color="#CBD5E1")),
            yaxis=dict(gridcolor="rgba(51, 65, 85, 0.25)", tickfont=dict(color="#CBD5E1")),
            margin=dict(l=30, r=30, t=60, b=30),
        )
    return fig


def plotly_scatter(
    df: pd.DataFrame,
    x: str,
    y: str,
    title: str,
    color_col: Optional[str] = None,
    size_col: Optional[str] = None,
    hover_data: Optional[list] = None,
    height: int = 460,
    add_diagonal: bool = False,
) -> go.Figure:
    """Render high-clarity scatter plot with glowing points."""
    fig = px.scatter(
        df,
        x=x,
        y=y,
        color=color_col,
        size=size_col,
        hover_data=hover_data,
        title=f"<b>{title}</b>",
        template=PLOTLY_TEMPLATE,
        height=height,
        color_discrete_sequence=[PALETTE["primary"], PALETTE["accent_magenta"], PALETTE["success"]],
    )
    if add_diagonal:
        mn = min(df[x].min(), df[y].min())
        mx = max(df[x].max(), df[y].max())
        fig.add_trace(go.Scatter(
            x=[mn, mx],
            y=[mn, mx],
            mode="lines",
            name="Ideal Parity",
            line=dict(dash="dash", color="rgba(255, 255, 255, 0.5)", width=1.5),
        ))
    fig.update_layout(
        plot_bgcolor="rgba(15, 23, 42, 0.3)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Plus Jakarta Sans", color=PALETTE["text"]),
        title_font=dict(size=16, color="#FFFFFF"),
        xaxis=dict(gridcolor="rgba(51, 65, 85, 0.25)", tickfont=dict(color="#CBD5E1")),
        yaxis=dict(gridcolor="rgba(51, 65, 85, 0.25)", tickfont=dict(color="#CBD5E1")),
        margin=dict(l=30, r=30, t=60, b=30),
    )
    return fig


def plotly_histogram(
    series: pd.Series, title: str, xaxis_title: str = "", nbins: int = 50, height: int = 400
) -> go.Figure:
    """Render glowing histogram distribution."""
    fig = px.histogram(
        series,
        nbins=nbins,
        title=f"<b>{title}</b>",
        template=PLOTLY_TEMPLATE,
        height=height,
    )
    fig.update_traces(
        marker_color=PALETTE["primary"],
        marker_line_color="#FFFFFF",
        marker_line_width=0.8,
        opacity=0.85,
    )
    fig.update_layout(
        plot_bgcolor="rgba(15, 23, 42, 0.3)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Plus Jakarta Sans", color=PALETTE["text"]),
        title_font=dict(size=16, color="#FFFFFF"),
        xaxis=dict(
            title=dict(text=xaxis_title, font=dict(color="#CBD5E1")),
            gridcolor="rgba(51, 65, 85, 0.25)",
            tickfont=dict(color="#CBD5E1"),
        ),
        yaxis=dict(
            title=dict(text="Frequency Count", font=dict(color="#CBD5E1")),
            gridcolor="rgba(51, 65, 85, 0.25)",
            tickfont=dict(color="#CBD5E1"),
        ),
        showlegend=False,
        margin=dict(l=30, r=30, t=60, b=30),
    )
    return fig


def show_figure_or_missing(filename: str, caption: str) -> None:
    """Display a saved PNG figure or a placeholder message if missing."""
    from dashboard.data_loader import figure_path
    p = figure_path(filename)
    if p:
        st.image(str(p), caption=caption, width='stretch')
    else:
        st.info(f"Figure not found: {filename}")


def footer() -> None:
    """Render polished executive footer."""
    st.markdown(
        """
        <div class="dashboard-footer">
            <b>MatRisk AI Enterprise Engine</b> · Research Platform V1 · Local Execution Only<br>
            <span style="font-size:0.75rem; opacity:0.8;">
                Built on trained models (.joblib), processed feature stores (DS1–DS6), and Monte Carlo stress engines.
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )
